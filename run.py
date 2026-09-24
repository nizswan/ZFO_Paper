#!/usr/bin/env python3
"""Run one experiment, inspect a paper configuration, or launch a saved grid."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.dont_write_bytecode = True

ZFO_METHODS = ("taylor2", "taylor3", "pade2", "pade3")
METHODS = ("adamw", *ZFO_METHODS, "adamw_cosine", "prodigy", "muon", "zfo_seq", "mezo")
CONFIG_TYPES = {
    "a": (10, 4, 4, 512),
    "b": (8, 2, 2, 384),
    "c": (8, 8, 2, 384),
    "d": (5, 4, 2, 256),
}
# model, dataset, first-order LR, MeZO LR, bounds T2/T3/P2/P3, type.
TABLE1 = [
    ("qwen", "gsm8k", 5e-6, 1e-7, (10, 5, 10, 10), "a"),
    ("qwen", "math", 5e-6, 1e-6, (5, 5, 5, 5), "c"),
    ("qwen", "svamp", 1e-5, 5e-7, (3, 5, 3, 3), "a"),
    ("qwen", "asdiv", 5e-6, 5e-7, (5, 5, 5, 5), "c"),
    ("qwen", "openbookqa", 1e-5, 1e-7, (10, 5, 5, 5), "a"),
    ("phi2", "svamp", 5e-6, 5e-7, (3, 5, 5, 3), "a"),
    ("phi2", "asdiv", 5e-6, 1e-7, (3, 3, 5, 3), "d"),
    ("phi2", "openbookqa", 5e-6, 1e-7, (3, 5, 3, 3), "d"),
    ("gemma2_2b", "svamp", 1e-6, 5e-9, (3, 3, 3, 3), "b"),
    ("llama32_1b", "gsm8k", 5e-6, 1e-7, (3, 3, 3, 3), "a"),
    ("llama32_1b", "asdiv", 5e-6, 5e-8, (5, 5, 3, 3), "c"),
    ("llama32_1b", "openbookqa", 1e-6, 1e-7, (5, 5, 5, 3), "b"),
]
TABLE2 = [
    ("qwen", "arc_challenge", 1e-5, None, (3, 5, 3, 3), "c"),
    ("qwen", "strategyqa", 1e-5, None, (5, 5, 3, 5), "c"),
    ("qwen", "folio", 1e-5, None, (3, 3, 3, 3), "c"),
    ("qwen", "codah", 1e-5, None, (5, 5, 5, 5), "c"),
]
DEFAULTS = dict(
    model="qwen",
    dataset="gsm8k",
    method="taylor2",
    seed=100,
    epochs=10,
    batches=4,
    group_size=4,
    maxlength=512,
    max_new_tokens=192,
    lr=5e-6,
    bound=10.0,
    perturbation_scale=1e-3,
    search_type="optimizer_displacement",
    search_inside="auto",
    temperature=0.8,
    top_p=0.95,
    kl_beta=0.02,
    early_stop_patience=3,
    subset=1_000_000,
    eval_subset=1_000_000,
    asdiv_eval_fraction=0.2,
    asdiv_split_seed=0,
    beta1=0.9,
    beta2=0.999,
    adam_eps=1e-8,
    weight_decay=0.01,
    lr_type="constant",
    warmup_ratio=0.1,
    prodigy_lr=1.0,
    d_coef=1.0,
    d0=1e-6,
    muon_lr=0.02,
    muon_momentum=0.95,
    muon_ns_steps=5,
    zo_weight_decay=0.0,
    zo_max_update_norm=None,
    k=5,
    initial_model="p3",
    tol=None,
    device="auto",
    dtype="auto",
    attn_implementation="auto",
    deterministic=False,
    model_revision=None,
    dataset_revision=None,
    data_dir=None,
    save_model=False,
)


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--preset",
        choices=[f"{r[0]}_{r[1]}" for r in TABLE1 + TABLE2],
        help="Table 6/7 setting; method chooses its bound",
    )
    p.add_argument(
        "--suite",
        choices=["table1", "table2", "table10", "table11", "muon"],
        help="Grid to inspect or run",
    )
    p.add_argument("--run_index", type=int, help="Run only this zero-based suite entry")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Execute the selected suite (otherwise list it)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print resolved configuration without loading dependencies",
    )
    p.add_argument(
        "--config", help="Replay a config.json, with explicit command-line overrides"
    )
    p.add_argument(
        "--output",
        default="results",
        help="Output parent directory; each run gets a unique hash",
    )
    p.add_argument(
        "--prepare_only",
        action="store_true",
        help="Load and snapshot data without downloading model weights",
    )
    p.add_argument(
        "--summarize",
        help="Aggregate completed runs under this directory, by configuration",
    )
    p.add_argument(
        "--self_test",
        action="store_true",
        help="Run local numerical checks without model downloads",
    )
    texts = {
        "model": "Model preset or public model repository ID",
        "dataset": "Dataset key",
        "method": "Optimizer/search method",
        "seed": "Python, NumPy and tensor seed",
        "epochs": "Maximum epochs",
        "batches": "Prompts per minibatch",
        "group_size": "Sampled completions per prompt",
        "maxlength": "Maximum prompt tokens",
        "max_new_tokens": "Maximum completion tokens",
        "lr": "AdamW learning rate; MeZO rate; Muon auxiliary AdamW rate",
        "bound": "Dimensionless search radius multiplier beta",
        "perturbation_scale": "Parameter-space probe length epsilon; MeZO Gaussian noise scale",
        "search_type": "Coordinate representation for ZFO and ZFO-Seq",
        "search_inside": "auto preserves historical behavior; true adds stationary points; false forces model endpoints",
        "temperature": "Training sampling temperature",
        "top_p": "Training nucleus threshold",
        "kl_beta": "KL coefficient",
        "early_stop_patience": "Equal consecutive evaluation scores to stop; 0 disables",
        "subset": "Maximum ordered training rows",
        "eval_subset": "Maximum ordered evaluation rows",
        "asdiv_eval_fraction": "ASDiv held-out problem fraction when only one source split exists",
        "asdiv_split_seed": "ASDiv partition seed, independent of training seed",
        "beta1": "AdamW/Prodigy first moment",
        "beta2": "AdamW/Prodigy second moment",
        "adam_eps": "AdamW/Prodigy denominator epsilon",
        "weight_decay": "AdamW/Prodigy/Muon decay",
        "lr_type": "External schedule; adamw_cosine always uses warmup_cosine",
        "warmup_ratio": "Warmup fraction of planned optimizer steps",
        "prodigy_lr": "Prodigy adjustment scale",
        "d_coef": "Prodigy adaptation coefficient",
        "d0": "Prodigy initial distance estimate",
        "muon_lr": "Muon matrix learning rate",
        "muon_momentum": "Muon momentum",
        "muon_ns_steps": "Muon Newton-Schulz iterations",
        "zo_weight_decay": "MeZO decay, excluding bias and normalization parameters",
        "zo_max_update_norm": "Optional MeZO random-update norm limit",
        "k": "ZFO-Seq extra evaluation budget, including two symmetric probes",
        "initial_model": "ZFO-Seq initializer",
        "tol": "ZFO-Seq duplicate tolerance in multiplier units; default 0.001*bound",
        "device": "auto chooses CUDA, otherwise CPU",
        "dtype": "auto chooses CUDA bf16/fp16, CPU fp32",
        "attn_implementation": "Attention backend; auto follows the model default",
        "deterministic": "Request deterministic tensor algorithms",
        "model_revision": "Model commit/ref; omitted uses the packaged pin for known models",
        "dataset_revision": "Dataset commit/ref; omitted uses the packaged pin",
        "data_dir": "Prepared train.jsonl/eval.jsonl/data.json directory",
        "save_model": "Write final model tensors and tokenizer under the run directory",
    }
    choices = {
        "method": METHODS,
        "dataset": [
            "gsm8k",
            "svamp",
            "asdiv",
            "openbookqa",
            "math",
            "arc_challenge",
            "strategyqa",
            "folio",
            "codah",
        ],
        "search_type": ["optimizer_displacement", "normalized_direction"],
        "search_inside": ["auto", "true", "false"],
        "lr_type": ["constant", "warmup_cosine"],
        "initial_model": ["t2", "t3", "p2", "p3"],
        "device": ["auto", "cpu", "cuda"],
        "dtype": ["auto", "float32", "float16", "bfloat16"],
        "attn_implementation": ["auto", "eager", "sdpa", "flash_attention_2"],
    }
    for key, value in DEFAULTS.items():
        kwargs = {
            "default": argparse.SUPPRESS,
            "help": f"{texts[key]} (base default: {value})",
        }
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = (
                float
                if key in {"tol", "zo_max_update_norm"}
                else type(value)
                if value is not None
                else str
            )
            if key in choices:
                kwargs["choices"] = choices[key]
        p.add_argument("--" + key, **kwargs)
    return p


def preset_config(name, method):
    row = next(r for r in TABLE1 + TABLE2 if f"{r[0]}_{r[1]}" == name)
    model, dataset, lr, zo_lr, bounds, kind = row
    if method == "mezo" and zo_lr is None:
        raise ValueError("No MeZO setting is reported for this sub-study preset")
    epochs, batches, group, length = CONFIG_TYPES[kind]
    return dict(
        model=model,
        dataset=dataset,
        lr=zo_lr if method == "mezo" else lr,
        method=method,
        epochs=epochs,
        batches=batches,
        group_size=group,
        maxlength=length,
        subset=512 if dataset == "math" else 1_000_000,
        eval_subset=256 if dataset == "math" else 1_000_000,
        bound=float(bounds[ZFO_METHODS.index(method)])
        if method in ZFO_METHODS
        else 1.0,
    )


def resolve(args):
    values = vars(args)
    config = dict(DEFAULTS)
    if args.config:
        saved = json.loads(Path(args.config).read_text())
        unknown = set(saved) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        config.update(saved)
    if args.preset:
        config.update(
            preset_config(args.preset, values.get("method", config["method"]))
        )
    config.update({k: v for k, v in values.items() if k in DEFAULTS})
    if config["method"] == "adamw_cosine":
        config["lr_type"] = "warmup_cosine"
    config = pin_defaults(config)
    validate(config)
    return config


def pin_defaults(config):
    """Freeze known public assets to the revisions checked with this package."""
    from prep import MODEL_IDS, MODEL_REVISIONS, DATA_REVISIONS

    config = dict(config)
    alias = next(
        (k for k, v in MODEL_IDS.items() if v == config["model"]), config["model"]
    )
    if config["model_revision"] is None:
        config["model_revision"] = MODEL_REVISIONS.get(alias)
    if config["dataset_revision"] is None:
        config["dataset_revision"] = DATA_REVISIONS.get(config["dataset"])
    return config


def validate(c):
    from prep import DATASETS, MODEL_IDS
    import re

    for key in (
        "epochs",
        "batches",
        "group_size",
        "maxlength",
        "max_new_tokens",
        "seed",
        "subset",
        "eval_subset",
        "early_stop_patience",
        "muon_ns_steps",
        "k",
        "asdiv_split_seed",
    ):
        if type(c[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    for key in ("deterministic", "save_model"):
        if type(c[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if c["dataset"] not in DATASETS:
        raise ValueError("Unknown dataset")
    if c["model"] not in MODEL_IDS and not re.fullmatch(r"[\w.-]+/[\w.-]+", c["model"]):
        raise ValueError("model must be a preset or public repository ID")
    for key, choices in {
        "device": {"auto", "cpu", "cuda"},
        "dtype": {"auto", "float32", "float16", "bfloat16"},
        "lr_type": {"constant", "warmup_cosine"},
        "attn_implementation": {"auto", "eager", "sdpa", "flash_attention_2"},
        "initial_model": {"t2", "t3", "p2", "p3"},
    }.items():
        if c[key] not in choices:
            raise ValueError(f"Invalid {key}")
    positive = [
        "epochs",
        "batches",
        "group_size",
        "maxlength",
        "max_new_tokens",
        "lr",
        "bound",
        "perturbation_scale",
        "temperature",
        "subset",
        "eval_subset",
        "adam_eps",
        "prodigy_lr",
        "d_coef",
        "d0",
        "muon_lr",
        "muon_ns_steps",
    ]
    for key in positive:
        if (
            not isinstance(c[key], (float, int))
            or not math.isfinite(c[key])
            or c[key] <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
    for key in (
        "seed",
        "early_stop_patience",
        "kl_beta",
        "weight_decay",
        "zo_weight_decay",
    ):
        if not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("beta1", "beta2", "muon_momentum"):
        if not 0 <= c[key] < 1:
            raise ValueError(f"{key} must be in [0,1)")
    if not 0 < c["top_p"] <= 1 or not 0 <= c["warmup_ratio"] <= 1:
        raise ValueError("top_p must be in (0,1], warmup_ratio in [0,1]")
    if not 0 < c["asdiv_eval_fraction"] < 1:
        raise ValueError("asdiv_eval_fraction must be in (0,1)")
    if c["asdiv_split_seed"] < 0:
        raise ValueError("asdiv_split_seed must be nonnegative")
    if c["method"] not in METHODS:
        raise ValueError("Unknown method")
    if c["group_size"] < 2:
        raise ValueError("GRPO requires group_size >= 2")
    if c["k"] < 2:
        raise ValueError("k must be >= 2")
    for key in ("tol", "zo_max_update_norm"):
        if c[key] is not None and (not math.isfinite(c[key]) or c[key] <= 0):
            raise ValueError(f"{key} must be finite and positive when specified")
    if c["method"] in {"prodigy", "muon", "mezo"} and c["lr_type"] != "constant":
        raise ValueError("This baseline requires lr_type=constant")
    if c["search_inside"] not in {"auto", "true", "false"}:
        raise ValueError("search_inside must be auto/true/false")
    if c["search_type"] not in {"optimizer_displacement", "normalized_direction"}:
        raise ValueError("Invalid search_type")


def suite_configs(suite):
    if suite == "table1":
        rows, methods, seeds = TABLE1, ("adamw", "mezo", *ZFO_METHODS), (100, 101, 102)
    elif suite == "table2":
        rows, methods, seeds = TABLE2, ("adamw", *ZFO_METHODS), (100, 101, 102)
    elif suite in {"table10", "muon"}:
        rows = [r for r in TABLE1 if r[0] == "qwen" and r[1] != "math"] + TABLE2
        methods = ("adamw_cosine", "prodigy") if suite == "table10" else ("muon",)
        seeds = (100, 101, 102)
    else:
        for dataset in ("codah", "arc_challenge", "svamp"):
            yield pin_defaults(
                {
                    **DEFAULTS,
                    "dataset": dataset,
                    "method": "zfo_seq",
                    "lr": 1e-5,
                    "bound": 3.0,
                    "epochs": 8,
                    "batches": 8,
                    "group_size": 2,
                    "maxlength": 384,
                    "subset": 512,
                    "eval_subset": 256,
                }
            )
        return
    for row in rows:
        for method in methods:
            for seed in seeds:
                c = {
                    **DEFAULTS,
                    **preset_config(f"{row[0]}_{row[1]}", method),
                    "seed": seed,
                }
                if method == "adamw_cosine":
                    c["lr_type"] = "warmup_cosine"
                yield pin_defaults(c)


def public_config(config):
    """Never store filesystem locations or credentials in a run manifest."""
    result = dict(config)
    result["data_dir"] = None
    return result


def run_id(config):
    serial = json.dumps(public_config(config), sort_keys=True).encode()
    return f"{config['dataset']}_{config['method']}_s{config['seed']}_{hashlib.sha256(serial).hexdigest()[:12]}"


def environment_info():
    packages = {}
    for name in (
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "huggingface_hub",
        "tokenizers",
        "numpy",
        "prodigyopt",
        "safetensors",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_type": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("run.py", "zfo.py", "baselines.py", "prep.py")
        },
    }


def _generate_group_impl(
    model: nn.Module,
    tokenizer: Any,
    prompts: List[str],
    group_size: int,
    max_prompt_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    *,
    do_sample: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_prompt_len),
    ).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_lens = attention_mask.sum(dim=-1)
    B = len(prompts)
    G = int(group_size)
    input_ids_rep = input_ids.repeat_interleave(G, dim=0)
    attn_rep = attention_mask.repeat_interleave(G, dim=0)
    prompt_lens_rep = prompt_lens.repeat_interleave(G, dim=0)
    gen_kwargs: Dict[str, Any] = dict(
        input_ids=input_ids_rep,
        attention_mask=attn_rep,
        max_new_tokens=int(max_new_tokens),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if bool(do_sample):
        gen_kwargs.update(
            do_sample=True, temperature=float(temperature), top_p=float(top_p)
        )
    else:
        gen_kwargs.update(do_sample=False)
    gen = model.generate(**gen_kwargs)
    gen_attn = (gen != tokenizer.pad_token_id).long()
    completions: List[List[str]] = []
    for i in range(B):
        row: List[str] = []
        for j in range(G):
            idx = i * G + j
            pl = int(prompt_lens_rep[idx].item())
            comp_ids = gen[idx, pl:]
            row.append(tokenizer.decode(comp_ids, skip_special_tokens=True))
        completions.append(row)
    return (gen, gen_attn, completions, prompt_lens_rep)


def _logprobs_for_tokens(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def _masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum(dim=-1)


def _generate_group(*args, **kwargs):
    with torch.no_grad():
        return _generate_group_impl(*args, **kwargs)


def _generate_mezo(
    model,
    tokenizer,
    prompts,
    group_size,
    max_prompt_len,
    max_new_tokens,
    temperature,
    top_p,
    device,
    *,
    do_sample=True,
):
    """MeZO source uses padded prompt width when decoding completions."""
    with torch.no_grad():
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        ).to(device)
        width = enc["input_ids"].size(1)
        kwargs = dict(
            input_ids=enc["input_ids"].repeat_interleave(group_size, 0),
            attention_mask=enc["attention_mask"].repeat_interleave(group_size, 0),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            do_sample=do_sample,
        )
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        gen = model.generate(**kwargs)
        rows = [
            [
                tokenizer.decode(
                    gen[i * group_size + j, width:], skip_special_tokens=True
                )
                for j in range(group_size)
            ]
            for i in range(len(prompts))
        ]
        widths = torch.full(
            (len(prompts) * group_size,), width, dtype=torch.long, device=device
        )
        return gen, (gen != tokenizer.pad_token_id).long(), rows, widths


def initialize_runtime():
    global torch, nn
    import torch
    from torch import nn

    try:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
    except (ImportError, AttributeError):
        pass


def train(config, directory, *, components=None):
    """Shared GRPO loop. Optional components support offline integration checks."""
    import numpy as np
    import baselines
    import prep
    import zfo
    from torch.utils.data import DataLoader

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    if config["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype_name = config["dtype"]
    if dtype_name == "auto":
        dtype_name = (
            ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
            if device.type == "cuda"
            else "float32"
        )
    dtype = getattr(torch, dtype_name)
    if components is None:
        tokenizer, model, reference, model_info = prep.load_models(
            config, device, dtype
        )
    else:
        tokenizer, model, reference, model_info = components[:4]
    method = config["method"]
    optimizer = None if method == "mezo" else baselines.build_optimizer(model, config)
    if components is None:
        train_items, eval_items, data_info = prep.load_data(config)
    else:
        train_items, eval_items, data_info = components[4:]
    prep.save_data(directory, train_items, eval_items, data_info)
    # Plain-list collation preserves nested and heterogeneous dataset fields.
    train_dl = DataLoader(
        train_items, batch_size=config["batches"], shuffle=True, collate_fn=list
    )
    eval_dl = DataLoader(
        eval_items, batch_size=config["batches"], shuffle=False, collate_fn=list
    )
    scheduler = (
        baselines.build_scheduler(optimizer, config, config["epochs"] * len(train_dl))
        if optimizer is not None
        else None
    )
    env = environment_info()
    env.update(
        resolved_device=device.type,
        resolved_dtype=dtype_name,
        model=model_info,
        attention_backend=getattr(model.config, "_attn_implementation", None),
    )
    prep.write_json(directory / "environment.json", env)
    prompt_fn, reward_fn = (
        prep.PROMPTS[config["dataset"]],
        prep.REWARDS[config["dataset"]],
    )
    generate = _generate_mezo if method == "mezo" else _generate_group

    def generate_batch(items, evaluation=False):
        return generate(
            model,
            tokenizer,
            [prompt_fn(item) for item in items],
            group_size=1 if evaluation else config["group_size"],
            max_prompt_len=config["maxlength"],
            max_new_tokens=config["max_new_tokens"],
            temperature=1.0 if evaluation else config["temperature"],
            top_p=1.0 if evaluation else config["top_p"],
            device=device,
            do_sample=not evaluation,
        )

    def evaluate():
        model.eval()
        total, count = 0.0, 0
        try:
            with torch.no_grad():
                for items in eval_dl:
                    _, _, rows, _ = generate_batch(items, True)
                    for item, row in zip(items, rows):
                        total += reward_fn(row[0], item)
                        count += 1
            return total / count
        finally:
            model.train()

    def probe(gen_ids, gen_attn, tok_mask, reference_logp, rewards_flat):
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                out = model(gen_ids, attention_mask=gen_attn)
                logp = _logprobs_for_tokens(out.logits[:, :-1, :], gen_ids[:, 1:])
                seq_kl = _masked_sum(logp - reference_logp, tok_mask) / tok_mask.sum(
                    -1
                ).clamp(min=1.0)
                # This score deliberately preserves the reported implementation.
                return float(rewards_flat.mean().item()) - config["kl_beta"] * float(
                    seq_kl.mean().item()
                )
        finally:
            model.train(was_training)

    step, last_eval, equal_epochs = 0, None, 0
    start = time.monotonic()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with (directory / "metrics.jsonl").open("w") as log:

        def emit(record):
            log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            log.flush()

        for epoch in range(config["epochs"]):
            for items in train_dl:
                step += 1
                ids, attention, rows, widths = generate_batch(items)
                rewards = torch.tensor(
                    [
                        [reward_fn(text, item) for text in row]
                        for item, row in zip(items, rows)
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                rewards_flat = rewards.reshape(-1).detach()
                advantages = (
                    (rewards - rewards.mean(dim=-1, keepdim=True)).reshape(-1).detach()
                )
                tokens = ids[:, 1:]
                mask = attention[:, 1:].float()
                if method == "mezo":
                    with torch.no_grad():
                        ref_out = reference(ids, attention_mask=attention)
                        reference_logp = _logprobs_for_tokens(
                            ref_out.logits[:, :-1, :], tokens
                        ).detach()
                        positions = (
                            torch.arange(tokens.size(1), device=device).unsqueeze(0) + 1
                        )
                        mask = mask * (positions >= widths.unsqueeze(1)).float()
                    common = dict(
                        model=model,
                        ref_logp_tok=reference_logp,
                        adv_flat=advantages,
                        gen_ids=ids,
                        gen_attn=attention,
                        tok_mask=mask,
                        kl_beta=config["kl_beta"],
                        policy_kind="grpo",
                        ppo_clip=0.2,
                    )
                    f0, _, kl_theta, old_logp = (
                        baselines._rlvr_zo_objective_on_fixed_batch(**common)
                    )
                    noise_seed = int(np.random.randint(1_000_000_000))
                    eps = config["perturbation_scale"]
                    baselines._zo_perturb_trainable(model, eps, noise_seed, 1.0)
                    fp, _, _, _ = baselines._rlvr_zo_objective_on_fixed_batch(
                        **common, seq_logp_old=old_logp
                    )
                    baselines._zo_perturb_trainable(model, eps, noise_seed, -2.0)
                    fm, _, _, _ = baselines._rlvr_zo_objective_on_fixed_batch(
                        **common, seq_logp_old=old_logp
                    )
                    baselines._zo_perturb_trainable(model, eps, noise_seed, 1.0)
                    estimate = (fp - fm) / (2.0 * eps)
                    info = baselines._zo_apply_update(
                        model,
                        estimate,
                        config["lr"],
                        noise_seed,
                        weight_decay=config["zo_weight_decay"],
                        max_update_norm=config["zo_max_update_norm"],
                    )
                    info.update(f_theta=f0, f_p=fp, f_m=fm, gradient_estimate=estimate)
                    loss_value, alpha, lr_used = -f0, 1.0, config["lr"]
                else:
                    out = model(ids, attention_mask=attention)
                    ref_out = reference(ids, attention_mask=attention)
                    logp = _logprobs_for_tokens(out.logits[:, :-1, :], tokens)
                    reference_logp = _logprobs_for_tokens(
                        ref_out.logits[:, :-1, :], tokens
                    )
                    seq_logp = _masked_sum(logp, mask)
                    seq_kl = _masked_sum(logp - reference_logp, mask) / mask.sum(
                        -1
                    ).clamp(min=1.0)
                    pg_loss = -(advantages * seq_logp).mean()
                    kl_loss = config["kl_beta"] * seq_kl.mean()
                    loss = pg_loss + kl_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Training loss is non-finite")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    theta = zfo._snapshot_trainable(model)
                    grads = zfo._snapshot_trainable_grads(model)
                    lr_used = float(optimizer.param_groups[0]["lr"])
                    optimizer.step()
                    theta_hat = zfo._snapshot_trainable(model)
                    if scheduler is not None:
                        scheduler.step()
                    kl_theta = float(seq_kl.detach().mean().item())
                    f0 = (
                        float(rewards_flat.mean().item()) - config["kl_beta"] * kl_theta
                    )
                    alpha, info = 1.0, {}
                    if method in (*ZFO_METHODS, "zfo_seq"):
                        displacement = zfo._delta(theta, theta_hat)
                        # ZFO-Seq used a host sqrt; fixed-budget runs used tensor sqrt.
                        dnorm = (
                            math.sqrt(
                                max(
                                    0.0,
                                    float(zfo._dot(displacement, displacement).item()),
                                )
                            )
                            if method == "zfo_seq"
                            else zfo._norm(displacement)
                        )
                        d1 = -float(zfo._dot(grads, displacement).item())
                        tentative = zfo._snapshot_trainable(model)

                        def phi(a):
                            zfo._apply_trainable_blend(model, a, theta, theta_hat)
                            try:
                                return probe(
                                    ids,
                                    attention,
                                    mask,
                                    reference_logp.detach(),
                                    rewards_flat,
                                )
                            finally:
                                zfo._apply_trainable(model, tentative)

                        alpha, info = zfo.select_step(
                            phi,
                            f0=f0,
                            d1=d1,
                            dnorm=dnorm,
                            lr=lr_used,
                            method=method,
                            bound=config["bound"],
                            epsilon=config["perturbation_scale"],
                            search_type=config["search_type"],
                            search_inside=config["search_inside"],
                            k=config["k"],
                            initial_model=config["initial_model"],
                            tol=config["tol"],
                        )
                    zfo._apply_trainable_blend(model, alpha, theta, theta_hat)
                    loss_value = float(loss.detach().item())
                    if method == "prodigy":
                        info["prodigy_d"] = float(optimizer.param_groups[0]["d"])
                emit(
                    {
                        "kind": "train",
                        "step": step,
                        "epoch": epoch,
                        "loss": loss_value,
                        "reward": float(rewards_flat.mean().item()),
                        "kl": kl_theta,
                        "alpha": alpha,
                        "lr": lr_used,
                        "search": info,
                    }
                )
            accuracy = evaluate()
            equal_epochs = (
                equal_epochs + 1
                if last_eval is not None and accuracy == last_eval
                else 1
            )
            last_eval = accuracy
            stopped = (
                config["early_stop_patience"] > 0
                and equal_epochs >= config["early_stop_patience"]
            )
            emit(
                {
                    "kind": "eval",
                    "step": step,
                    "epoch": epoch,
                    "accuracy": accuracy,
                    "same_score_epochs": equal_epochs,
                    "early_stopped": stopped,
                }
            )
            print(f"epoch={epoch + 1} accuracy={accuracy:.6f} steps={step}", flush=True)
            if stopped:
                break
    if config["save_model"]:
        # Save tensors only; avoid serializing source paths from Python objects.
        from safetensors.torch import save_file

        model_dir = directory / "model"
        model_dir.mkdir()
        weights = {
            k: v.detach().cpu().contiguous().clone()
            for k, v in model.state_dict().items()
        }
        save_file(weights, model_dir / "model.safetensors")
        model_config = model.config.to_dict()
        model_config.pop("_name_or_path", None)
        prep.write_json(model_dir / "config.json", model_config)
        tokenizer.save_pretrained(model_dir)
    result = {
        "status": "complete",
        "final_accuracy": last_eval,
        "epochs_completed": epoch + 1,
        "steps": step,
        "early_stopped": stopped,
        "elapsed_seconds": time.monotonic() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else 0,
    }
    prep.write_json(directory / "result.json", result)
    return result


def summarize(directory):
    """Report both SD conventions and seed coverage; reject duplicate seeds."""
    groups = {}
    for path in sorted(Path(directory).rglob("result.json")):
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            continue
        config = json.loads(path.with_name("config.json").read_text())
        identity = {
            k: v
            for k, v in config.items()
            if k not in {"seed", "data_dir", "save_model"}
        }
        env = json.loads(path.with_name("environment.json").read_text())
        data = json.loads(path.with_name("data.json").read_text())
        identity.update(
            resolved_model_revision=env["model"]["revision"],
            resolved_dataset_revision=data["revision"],
            resolved_dtype=env["resolved_dtype"],
            package_versions=env["packages"],
            source_sha256=env["source_sha256"],
        )
        key = json.dumps(identity, sort_keys=True)
        group = groups.setdefault(key, {})
        if config["seed"] in group:
            raise ValueError(
                f"Duplicate seed {config['seed']} for one configuration; select one output collection"
            )
        group[config["seed"]] = result["final_accuracy"] * 100
    for key, values in groups.items():
        identity = json.loads(key)
        numbers = list(values.values())
        print(
            json.dumps(
                {
                    "configuration": identity,
                    "n": len(numbers),
                    "seeds": sorted(values),
                    "paper_seeds_complete": set(values) == {100, 101, 102},
                    "mean_percent": statistics.mean(numbers),
                    "std_sample_percent": statistics.stdev(numbers)
                    if len(numbers) > 1
                    else None,
                    "std_population_percent": statistics.pstdev(numbers),
                },
                sort_keys=True,
            )
        )
    if not groups:
        raise ValueError("No completed training results found")


def self_test():
    import baselines
    import prep
    import zfo

    examples = [{"body": f"{i} apples", "question": "How many?"} for i in range(10)]
    training, evaluation, _ = prep._partition_asdiv(
        {"validation": examples}, eval_fraction=0.2, split_seed=0
    )
    assert len(training) == 8 and len(evaluation) == 2
    assert not set(map(prep._asdiv_problem_key, training)) & set(
        map(prep._asdiv_problem_key, evaluation)
    )
    try:
        prep._assert_asdiv_disjoint(examples, [dict(examples[0], answer="different")])
    except ValueError:
        pass
    else:
        raise AssertionError("ASDiv overlap was not rejected")

    # P(x)=(1+x)/(1+x*x) has an interior maximum at sqrt(2)-1.
    end, _ = zfo.select_model("pade3", 1.0, 1.0, -2.0, -6.0, 2.0, 1.001, "false")
    inside, _ = zfo.select_model("pade3", 1.0, 1.0, -2.0, -6.0, 2.0, 1.001, "true")
    assert end == 0 and abs(inside - (math.sqrt(2.0) - 1.0)) < 1e-12
    assert zfo._maximize_quadratic(0.0, 2.0, -2.0, 3.0) == 1.0
    assert zfo._maximize_quadratic(0.0, 2.0, -2.0, 3.0, False) in {0.0, 3.0}
    assert zfo._maximize_cubic(0.0, 1.0, 0.0, -6.0, 2.0, False) in {0.0, 2.0}
    phi = lambda a: -((a - 0.4) ** 2)
    seq = baselines.select_zfo_seq(
        phi, phi0=phi(0), d1=0.8, dalpha=0.01, beta=1.0, k=5, initial_model="p3"
    )
    assert seq.probes_used == 5 and phi(seq.selected_a) >= phi(0)
    for kind in ("optimizer_displacement", "normalized_direction"):
        a, info = zfo.select_step(
            phi,
            f0=phi(0),
            d1=0.8,
            dnorm=2.0,
            lr=0.1,
            method="taylor2",
            bound=1.0,
            epsilon=0.01,
            search_type=kind,
        )
        assert abs(a - 0.4) < 1e-10 and info["n_extra_forwards"] == 2
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.AdamW([p], lr=0.1)
    schedule = baselines.build_scheduler(
        opt, {**DEFAULTS, "lr_type": "warmup_cosine"}, 10
    )
    for _ in range(10):
        p.grad = torch.ones_like(p)
        opt.step()
        schedule.step()
    assert opt.param_groups[0]["lr"] == 0.0
    assert [
        len(list(suite_configs(s)))
        for s in ("table1", "table2", "table10", "table11", "muon")
    ] == [216, 60, 48, 3, 24]
    print(
        "Checks passed: ASDiv separation, endpoints/interiors, radius coordinates, sequential budget, cosine schedule, and paper grids."
    )


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.summarize:
        summarize(args.summarize)
        return 0
    if args.self_test:
        initialize_runtime()
        self_test()
        return 0
    if args.suite:
        if args.config or args.preset or args.prepare_only:
            p.error("suite cannot be combined with config, preset, or prepare_only")
        explicit = set(vars(args)) & set(DEFAULTS)
        if explicit:
            p.error("suite grids are fixed; use --preset or --config for overrides")
        configs = list(suite_configs(args.suite))
        indices = range(len(configs)) if args.run_index is None else [args.run_index]
        if args.run_index is not None and not 0 <= args.run_index < len(configs):
            p.error(f"run_index must be 0..{len(configs) - 1}")
        for index in indices:
            c = configs[index]
            validate(c)
            print(
                json.dumps(
                    {"index": index, "suite": args.suite, "config": c}, sort_keys=True
                ),
                flush=True,
            )
            if args.execute and not args.dry_run:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    args.output,
                ]
                for key, value in c.items():
                    if isinstance(value, bool):
                        command.append("--" + ("" if value else "no-") + key)
                    elif value is not None:
                        command += ["--" + key, str(value)]
                subprocess.run(command, check=True)
        return 0
    if args.run_index is not None or args.execute:
        p.error("run_index/execute require --suite")
    try:
        c = resolve(args)
    except (ValueError, TypeError) as error:
        p.error(str(error))
    if args.dry_run:
        print(json.dumps(c, indent=2, sort_keys=True))
        return 0
    initialize_runtime()
    import prep

    directory = Path(args.output) / run_id(c)
    directory.mkdir(parents=True, exist_ok=False)
    prep.write_json(directory / "config.json", public_config(c))
    prep.write_json(
        directory / "result.json",
        {"status": "preparing" if args.prepare_only else "running"},
    )
    try:
        if args.prepare_only:
            items, evaluation, metadata = prep.load_data(c)
            prep.save_data(directory, items, evaluation, metadata)
            prep.write_json(directory / "result.json", {"status": "prepared"})
            print(
                f"Prepared {len(items)} train and {len(evaluation)} evaluation examples."
            )
        else:
            train(c, directory)
    except BaseException as error:
        # Exception messages can contain private cache paths; record only type.
        prep.write_json(
            directory / "result.json",
            {"status": "failed", "error_type": type(error).__name__},
        )
        raise
    print(f"Output directory: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
