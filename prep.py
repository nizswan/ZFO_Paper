"""Model and dataset preparation; prompts and verifiers used by the experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_NUM_RE = re.compile("[-+]?\\d+(?:\\.\\d+)?")


_GSM_HASH_RE = re.compile("####\\s*([-+]?\\d+(?:\\.\\d+)?)")


_BOX_RE = re.compile("\\\\boxed\\s*\\{([^}]*)\\}")


def _extract_last_number(text: str) -> Optional[str]:
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1].strip()
    return None


def _normalize_answer_text(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "$" and (s[-1] == "$"):
        s = s[1:-1].strip()
    s = re.sub("\\s+", " ", s).strip()
    return s


def _extract_gsm_like_number(text: str) -> Optional[str]:
    m = _GSM_HASH_RE.search(text)
    if m:
        return m.group(1).strip()
    return _extract_last_number(text)


def _extract_last_boxed_or_last(text: str) -> Optional[str]:
    matches = _BOX_RE.findall(text)
    if matches:
        return _normalize_answer_text(matches[-1])
    return _extract_last_number(text)


def _extract_choice_letter(text: str, max_letter: str = "D") -> Optional[str]:
    upper_max = max_letter.upper()
    lower_max = upper_max.lower()
    pattern_hash = f"####\\s*([A-{upper_max}a-{lower_max}])\\b"
    m = re.search(pattern_hash, text)
    if m:
        return m.group(1).upper()
    pattern_plain = f"\\b([A-{upper_max}a-{lower_max}])\\b"
    m = re.search(pattern_plain, text)
    if m:
        return m.group(1).upper()
    return None


def _extract_label_token(text: str) -> Optional[str]:
    m = re.search("####\\s*(true|false|uncertain|yes|no)\\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    m = re.search("\\b(true|false|uncertain|yes|no)\\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return None


def _normalize_boolish(x: Any) -> Optional[str]:
    s = str(x).strip().lower()
    if s in {"true", "yes", "1"}:
        return "yes"
    if s in {"false", "no", "0"}:
        return "no"
    return None


_GSM_PROMPT_PREFIX = "You are a helpful math assistant. Solve the problem carefully.\nReturn your final answer as: #### <number>\n\n"


def _gsm_prompt(item: Dict[str, Any]) -> str:
    q = str(item["question"])
    return _GSM_PROMPT_PREFIX + f"Problem:\n{q}\n\nSolution:\n"


def _gsm_reward(completion: str, item: Dict[str, Any]) -> float:
    gt = _extract_gsm_like_number(str(item["answer"]))
    pred = _extract_gsm_like_number(completion)
    if gt is None or pred is None:
        return 0.0
    return 1.0 if pred == gt else 0.0


_SVAMP_PROMPT_PREFIX = "You are a helpful math assistant. Solve the word problem carefully.\nReturn your final answer as: #### <number>\n\n"


def _svamp_prompt(item: Dict[str, Any]) -> str:
    body = str(item.get("Body", item.get("body", ""))).strip()
    q = str(item.get("Question", item.get("question", ""))).strip()
    if body and q:
        text = f"{body}\n\n{q}"
    else:
        text = body or q
    return _SVAMP_PROMPT_PREFIX + f"Problem:\n{text}\n\nSolution:\n"


def _svamp_reward(completion: str, item: Dict[str, Any]) -> float:
    gt = _extract_gsm_like_number(str(item.get("Answer", item.get("answer", ""))))
    pred = _extract_gsm_like_number(completion)
    if gt is None or pred is None:
        return 0.0
    return 1.0 if pred == gt else 0.0


_OBQA_PROMPT_PREFIX = "You are a helpful science question answering assistant.\nChoose the correct option.\nReturn your final answer as: #### <letter>\n\n"


def _obqa_prompt(item: Dict[str, Any]) -> str:
    stem = str(item.get("question_stem", item.get("question", ""))).strip()
    choices = item.get("choices", None)
    lines: List[str] = []
    if isinstance(choices, dict):
        labels = list(choices.get("label", []))
        texts = list(choices.get("text", []))
        for L, T in zip(labels, texts):
            lines.append(f"{str(L).strip()}. {str(T).strip()}")
    elif isinstance(choices, list):
        for c in choices:
            if isinstance(c, dict):
                L = str(c.get("label", "")).strip()
                T = str(c.get("text", "")).strip()
                if L:
                    lines.append(f"{L}. {T}")
                else:
                    lines.append(T)
    choices_block = "\n".join(lines).strip()
    return (
        _OBQA_PROMPT_PREFIX
        + f"Question:\n{stem}\n\nOptions:\n{choices_block}\n\nAnswer:\n"
    )


def _obqa_reward(completion: str, item: Dict[str, Any]) -> float:
    gt = str(item.get("answerKey", item.get("answer_key", ""))).strip().upper()
    pred = _extract_choice_letter(completion, max_letter="D")
    if not gt or pred is None:
        return 0.0
    return 1.0 if pred == gt else 0.0


_ASDIV_PROMPT_PREFIX = "You are a helpful math assistant. Solve the word problem carefully.\nReturn your final answer as: #### <number>\n\n"


def _asdiv_prompt(item: Dict[str, Any]) -> str:
    body = str(item.get("body", item.get("Body", ""))).strip()
    q = str(item.get("question", item.get("Question", ""))).strip()
    if body and q:
        text = f"{body}\n\n{q}"
    else:
        text = body or q
    return _ASDIV_PROMPT_PREFIX + f"Problem:\n{text}\n\nSolution:\n"


def _asdiv_reward(completion: str, item: Dict[str, Any]) -> float:
    ans = str(item.get("answer", item.get("Answer", ""))).strip()
    gold_first = ans.split(";")[0].strip() if ";" in ans else ans
    gt = _extract_gsm_like_number(gold_first)
    pred = _extract_gsm_like_number(completion)
    if gt is None or pred is None:
        return 0.0
    try:
        return 1.0 if abs(float(gt) - float(pred)) < 1e-06 else 0.0
    except ValueError:
        return 1.0 if gt == pred else 0.0


_MATH_PROMPT_PREFIX = "You are a helpful math assistant. Solve the problem carefully.\nPut your final answer in the form \\boxed{...}.\n\n"


def _math_prompt(item: Dict[str, Any]) -> str:
    prob = str(item.get("problem", item.get("question", ""))).strip()
    return _MATH_PROMPT_PREFIX + f"Problem:\n{prob}\n\nSolution:\n"


def _math_reward(completion: str, item: Dict[str, Any]) -> float:
    sol = str(item.get("solution", item.get("answer", "")))
    gt = _extract_last_boxed_or_last(sol)
    pred = _extract_last_boxed_or_last(completion)
    if gt is None or pred is None:
        return 0.0
    return 1.0 if _normalize_answer_text(pred) == _normalize_answer_text(gt) else 0.0


def _stringify_premises(premises: Any) -> str:
    if isinstance(premises, list):
        return "\n".join((f"- {str(p).strip()}" for p in premises))
    return str(premises).strip()


def _choice_lines_from_item(item: Dict[str, Any], choices_key: str = "choices") -> str:
    choices = item.get(choices_key, None)
    lines: List[str] = []
    if isinstance(choices, dict):
        labels = list(choices.get("label", []))
        texts = list(choices.get("text", []))
        for lab, txt in zip(labels, texts):
            lines.append(f"{str(lab).strip()}. {str(txt).strip()}")
    elif isinstance(choices, list):
        if choices and isinstance(choices[0], dict):
            for c in choices:
                lab = str(c.get("label", "")).strip()
                txt = str(c.get("text", c.get("content", c.get("answer", "")))).strip()
                if lab:
                    lines.append(f"{lab}. {txt}")
                else:
                    lines.append(txt)
        else:
            labels = [chr(ord("A") + i) for i in range(len(choices))]
            for lab, txt in zip(labels, choices):
                lines.append(f"{lab}. {str(txt).strip()}")
    return "\n".join(lines).strip()


def _split_train_only(
    items: List[Dict[str, Any]], seed: int, eval_frac: float = 0.1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(items) < 2:
        raise RuntimeError("Need at least 2 examples to create a train/eval split.")
    rng = random.Random(seed)
    items_copy = list(items)
    rng.shuffle(items_copy)
    n_eval = max(1, int(round(len(items_copy) * float(eval_frac))))
    n_eval = min(n_eval, len(items_copy) - 1)
    eval_items = items_copy[:n_eval]
    train_items = items_copy[n_eval:]
    return (train_items, eval_items)


_ARC_PROMPT_PREFIX = "You are a helpful science question answering assistant.\nChoose the correct option.\nReturn your final answer as: #### <letter>\n\n"


def _arc_prompt(item: Dict[str, Any]) -> str:
    q = str(item.get("question", "")).strip()
    choices_block = _choice_lines_from_item(item)
    return (
        _ARC_PROMPT_PREFIX + f"Question:\n{q}\n\nOptions:\n{choices_block}\n\nAnswer:\n"
    )


def _arc_reward(completion: str, item: Dict[str, Any]) -> float:
    gt = str(item.get("answerKey", "")).strip().upper()
    if gt.isdigit():
        pred_num = _extract_last_number(completion)
        return 1.0 if pred_num is not None and pred_num == gt else 0.0
    pred = _extract_choice_letter(completion, max_letter="E")
    return 1.0 if gt and pred == gt else 0.0


_STRATEGYQA_PROMPT_PREFIX = "You are a helpful reasoning assistant.\nAnswer the question with yes or no.\nReturn your final answer as: #### yes or #### no\n\n"


def _strategyqa_prompt(item: Dict[str, Any]) -> str:
    q = str(item.get("question", "")).strip()
    facts = item.get("facts", None)
    decomposition = item.get("decomposition", None)
    extra_parts: List[str] = []
    if facts:
        if isinstance(facts, list):
            extra_parts.append(
                "Evidence:\n" + "\n".join((f"- {str(x).strip()}" for x in facts))
            )
        else:
            extra_parts.append("Evidence:\n" + str(facts).strip())
    if decomposition:
        if isinstance(decomposition, list):
            extra_parts.append(
                "Reasoning hints:\n"
                + "\n".join((f"- {str(x).strip()}" for x in decomposition))
            )
        else:
            extra_parts.append("Reasoning hints:\n" + str(decomposition).strip())
    extra = "\n\n".join(extra_parts)
    if extra:
        extra += "\n\n"
    return _STRATEGYQA_PROMPT_PREFIX + extra + f"Question:\n{q}\n\nAnswer:\n"


def _strategyqa_reward(completion: str, item: Dict[str, Any]) -> float:
    gt = _normalize_boolish(item.get("answer", item.get("label", "")))
    pred_token = _extract_label_token(completion)
    pred = _normalize_boolish(pred_token) if pred_token is not None else None
    return 1.0 if gt is not None and pred == gt else 0.0


_FOLIO_PROMPT_PREFIX = "You are a logic reasoning assistant.\nDecide whether the conclusion follows from the premises.\nReturn your final answer as exactly one of: #### True, #### False, or #### Uncertain\n\n"


def _folio_prompt(item: Dict[str, Any]) -> str:
    premises = _stringify_premises(item.get("premises", ""))
    conclusion = str(item.get("conclusion", "")).strip()
    return (
        _FOLIO_PROMPT_PREFIX
        + f"Premises:\n{premises}\n\nConclusion:\n{conclusion}\n\nAnswer:\n"
    )


def _normalize_folio_label(x: Any) -> str:
    s = str(x).strip().lower()
    if s == "unknown":
        return "uncertain"
    return s


def _folio_reward(completion: str, item: Dict[str, Any]) -> float:
    gt = _normalize_folio_label(item.get("label", ""))
    pred_raw = _extract_label_token(completion)
    pred = _normalize_folio_label(pred_raw) if pred_raw is not None else None
    return 1.0 if gt and pred == gt else 0.0


_CODAH_PROMPT_PREFIX = "You are a commonsense reasoning assistant.\nChoose the most plausible completion.\nReturn your final answer as: #### <letter>\n\n"


def _codah_prompt(item: Dict[str, Any]) -> str:
    q = str(
        item.get(
            "question_propmt", item.get("question_prompt", item.get("question", ""))
        )
    ).strip()
    answers = item.get("candidate_answers", [])
    choices = {"label": ["A", "B", "C", "D"], "text": list(answers)}
    choices_block = _choice_lines_from_item({"choices": choices})
    return (
        _CODAH_PROMPT_PREFIX + f"Prompt:\n{q}\n\nOptions:\n{choices_block}\n\nAnswer:\n"
    )


def _codah_reward(completion: str, item: Dict[str, Any]) -> float:
    answers = item.get("candidate_answers", [])
    n_choices = len(answers) if isinstance(answers, (list, tuple)) else 4
    try:
        idx = int(item.get("correct_answer_idx", -1))
    except Exception:
        return 0.0
    if idx < 0 or idx >= n_choices:
        return 0.0
    gt = chr(ord("A") + idx)
    pred = _extract_choice_letter(completion, max_letter="D")
    return 1.0 if pred == gt else 0.0


MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-Math-1.5B",
    "phi2": "microsoft/phi-2",
    "gemma2_2b": "google/gemma-2-2b",
    "llama32_1b": "meta-llama/Llama-3.2-1B",
}
MODEL_REVISIONS = {
    "qwen": "4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
    "phi2": "810d367871c1d460086d9f82db8696f2e0a0fcd0",
    "gemma2_2b": "c5ebcd40d208330abc697524c919956e692655cf",
    "llama32_1b": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
}
DATA_REVISIONS = {
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "svamp": "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef",
    "openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "asdiv": "8f95807222d87b4c688c3c22a6ba2801e1fa03e2",
    "math": "e839825f9ec5c6cfa585c654a59610969ec13993",
    "arc_challenge": "210d026faf9955653af8916fad021475a3f00453",
    "strategyqa": "705562638fe1d8ca6bb98c66fc8f94d45fda8c83",
    "codah": "4b0e0e7f33b3ae8cd8e69bcdfa09af9611d8f3bd",
    "folio": "5d7bb84c7edab3fb358e057d2807f19cf5cf5e2d",
}
DATASETS = {
    "gsm8k": ("gsm8k", "main", "train", "test"),
    "svamp": ("ChilleD/SVAMP", None, "train", "test"),
    "openbookqa": ("allenai/openbookqa", "main", "train", "test"),
    "asdiv": ("EleutherAI/asdiv", None, None, None),
    "math": ("qwedsacf/competition_math", None, "train", None),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "train", "test"),
    "strategyqa": ("ChilleD/StrategyQA", None, "train", "test"),
    "codah": ("jaredfern/codah", "codah", "train", None),
    "folio": (None, None, "train", "validation"),
}
PROMPTS = {
    "gsm8k": _gsm_prompt,
    "svamp": _svamp_prompt,
    "openbookqa": _obqa_prompt,
    "asdiv": _asdiv_prompt,
    "math": _math_prompt,
    "arc_challenge": _arc_prompt,
    "strategyqa": _strategyqa_prompt,
    "folio": _folio_prompt,
    "codah": _codah_prompt,
}
REWARDS = {
    "gsm8k": _gsm_reward,
    "svamp": _svamp_reward,
    "openbookqa": _obqa_reward,
    "asdiv": _asdiv_reward,
    "math": _math_reward,
    "arc_challenge": _arc_reward,
    "strategyqa": _strategyqa_reward,
    "folio": _folio_reward,
    "codah": _codah_reward,
}


def json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def rows_hash(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json_bytes(row) + b"\n")
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )


def _asdiv_problem_key(item):
    """Identify a problem independently of its answer, ID, or other metadata."""
    body = str(item.get("body", item.get("Body", "")))
    question = str(item.get("question", item.get("Question", "")))
    key = tuple(" ".join(text.split()).casefold() for text in (body, question))
    if not any(key):
        raise ValueError("ASDiv example has no problem text")
    return key


def _assert_asdiv_disjoint(train, evaluation):
    if not train or not evaluation:
        raise ValueError("ASDiv training and evaluation partitions must be nonempty")
    shared = {_asdiv_problem_key(x) for x in train} & {
        _asdiv_problem_key(x) for x in evaluation
    }
    if shared:
        raise ValueError(
            f"ASDiv training/evaluation overlap: {len(shared)} shared problems. "
            "Use separate partitions or regenerate the prepared data."
        )


def _partition_asdiv(splits, *, eval_fraction, split_seed):
    """Use disjoint native splits or group duplicate problems before holdout."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("asdiv_eval_fraction must be in (0,1)")
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("asdiv_split_seed must be a nonnegative integer")
    pair = next(
        (
            (a, b)
            for a, b in (
                ("train", "test"),
                ("train", "validation"),
                ("validation", "test"),
            )
            if a in splits and b in splits
        ),
        None,
    )
    if pair is not None:
        train_name, eval_name = pair
        train = [dict(row) for row in splits[train_name]]
        evaluation = [dict(row) for row in splits[eval_name]]
        rule = "distinct native splits"
    elif len(splits) == 1:
        source_name, rows = next(iter(splits.items()))
        groups = {}
        for row in rows:
            groups.setdefault(_asdiv_problem_key(row), []).append(dict(row))
        if len(groups) < 2:
            raise ValueError("ASDiv needs at least two distinct problems for a holdout")
        keys = sorted(groups)
        random.Random(split_seed).shuffle(keys)
        n_eval = max(1, min(round(len(keys) * eval_fraction), len(keys) - 1))
        evaluation = [row for key in keys[:n_eval] for row in groups[key]]
        train = [row for key in keys[n_eval:] for row in groups[key]]
        train_name, eval_name = f"{source_name}:train", f"{source_name}:heldout"
        rule = "seeded problem-group holdout"
    else:
        raise ValueError("ASDiv has no unambiguous train/evaluation split pair")
    _assert_asdiv_disjoint(train, evaluation)
    return (
        train,
        evaluation,
        {
            "train_split": train_name,
            "eval_split": eval_name,
            "source_splits": sorted(splits),
            "split_rule": rule,
            "partition_policy": "asdiv-disjoint-v1",
            "shared_problem_keys": 0,
            "asdiv_split_seed": split_seed,
            "asdiv_eval_fraction": eval_fraction,
        },
    )


def load_data(config):
    """Load exact split choices; never substitute another dataset on failure."""
    key, seed = config["dataset"], config["seed"]
    fields = ("dataset", "seed", "subset", "eval_subset")
    signature = {k: config[k] for k in fields}
    if key == "asdiv":
        signature.update(
            {k: config[k] for k in ("asdiv_split_seed", "asdiv_eval_fraction")}
        )
    if config["data_dir"]:
        directory = Path(config["data_dir"])
        metadata = json.loads((directory / "data.json").read_text())
        if metadata["selection"] != signature:
            raise ValueError(
                "Prepared data selection differs from dataset/seed/subset/eval_subset"
            )
        sets = []
        for split in ("train", "eval"):
            rows = [
                json.loads(line)
                for line in (directory / f"{split}.jsonl").read_text().splitlines()
                if line
            ]
            if rows_hash(rows) != metadata[f"{split}_sha256"]:
                raise ValueError(f"Prepared {split} data hash mismatch")
            sets.append(rows)
        if (
            config["dataset_revision"]
            and config["dataset_revision"] != metadata["revision"]
        ):
            raise ValueError("Prepared data revision differs from dataset_revision")
        if key == "asdiv":
            _assert_asdiv_disjoint(*sets)
            metadata = {**metadata, "shared_problem_keys": 0}
        return *sets, metadata

    repo, name, train_split, eval_split = DATASETS[key]
    partition_info = {}
    if key == "folio":
        revision = config["dataset_revision"]
        if not revision:
            url = "https://api.github.com/repos/Yale-LILY/FOLIO/commits/main"
            request = urllib.request.Request(
                url, headers={"User-Agent": "dataset-loader"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                revision = json.load(response)["sha"]
        root = f"https://raw.githubusercontent.com/Yale-LILY/FOLIO/{revision}/data/v0.0"
        sets = []
        for split in ("train", "validation"):
            with urllib.request.urlopen(
                f"{root}/folio-{split}.jsonl", timeout=120
            ) as response:
                sets.append(
                    [
                        json.loads(line)
                        for line in response.read().decode().splitlines()
                        if line.strip()
                    ]
                )
        train, evaluation = sets
        repo = "Yale-LILY/FOLIO"
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        revision = (
            HfApi()
            .dataset_info(repo, revision=config["dataset_revision"] or "main")
            .sha
        )
        kwargs = dict(revision=revision, trust_remote_code=True)
        if key == "asdiv":
            splits = load_dataset(repo, name, **kwargs)
            train, evaluation, partition_info = _partition_asdiv(
                splits,
                eval_fraction=config["asdiv_eval_fraction"],
                split_seed=config["asdiv_split_seed"],
            )
        else:
            data = load_dataset(repo, name, split=train_split, **kwargs)
            if key == "math":
                if len(data) < 2:
                    raise ValueError("MATH needs at least two examples")
                # Historical shuffle and held-out prefix, before applying train cap.
                data = data.shuffle(seed=seed)
                items = [dict(item) for item in data]
                count = max(1, min(config["eval_subset"], len(items) - 1))
                evaluation, train = items[:count], items[count:]
            elif key == "codah":
                train, evaluation = _split_train_only(
                    [dict(item) for item in data], seed
                )
            else:
                train = [dict(item) for item in data]
                evaluation = [
                    dict(item)
                    for item in load_dataset(repo, name, split=eval_split, **kwargs)
                ]
    train = train[: config["subset"]]
    evaluation = evaluation[: config["eval_subset"]]
    if not train or not evaluation:
        raise ValueError("Both training and evaluation data must be nonempty")
    if key == "asdiv":
        _assert_asdiv_disjoint(train, evaluation)
    overlap = len(
        {hashlib.sha256(json_bytes(x)).digest() for x in train}
        & {hashlib.sha256(json_bytes(x)).digest() for x in evaluation}
    )
    metadata = {
        "selection": signature,
        "repository": repo,
        "revision": revision,
        "train_split": train_split,
        "eval_split": eval_split,
        "train_count": len(train),
        "eval_count": len(evaluation),
        "train_sha256": rows_hash(train),
        "eval_sha256": rows_hash(evaluation),
        "shared_unique_rows": overlap,
        "split_rule": "seeded held-out prefix"
        if key == "math"
        else "seeded 90/10 split"
        if key == "codah"
        else "named splits",
        **partition_info,
    }
    return train, evaluation, metadata


def save_data(directory, train, evaluation, metadata):
    """Store the exact ordered rows so later runs do not depend on moving data."""
    directory = Path(directory)
    for split, rows in (("train", train), ("eval", evaluation)):
        with (directory / f"{split}.jsonl").open("wb") as stream:
            for row in rows:
                stream.write(json_bytes(row) + b"\n")
    write_json(directory / "data.json", metadata)


def load_models(config, device, dtype):
    """Load policy and frozen reference from the same immutable revision."""
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_IDS.get(config["model"], config["model"])
    revision = (
        HfApi().model_info(model_id, revision=config["model_revision"] or "main").sha
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    common = {"revision": revision}
    if token:
        common["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs = {**common, "torch_dtype": dtype, "low_cpu_mem_usage": True}
    if config["attn_implementation"] != "auto":
        kwargs["attn_implementation"] = config["attn_implementation"]
    policy = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)
    policy.train()
    policy.config.use_cache = False
    # Keep the second tokenizer initialization in the historical loading order.
    AutoTokenizer.from_pretrained(model_id, use_fast=True, **common)
    reference = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)
    reference.eval()
    reference.config.use_cache = False
    for parameter in reference.parameters():
        parameter.requires_grad = False
    return tokenizer, policy, reference, {"model_id": model_id, "revision": revision}
