# ZFO experiment package

A self-contained command-line package for the large-model experiments: four ZFO
local models, AdamW, AdamW with cosine decay, Prodigy, MeZO, ZFO-Seq, and Muon.
The distribution contains six files:

| File | Purpose |
| --- | --- |
| `run.py` | Configuration, paper grids, training, evaluation, local logging, aggregation |
| `zfo.py` | Taylor/Padé models, search coordinates, safeguards, parameter updates |
| `baselines.py` | Optimizers, cosine schedule, MeZO, sequential ZFO refinement |
| `prep.py` | Models, datasets, prompts, reward functions, split snapshots |
| `requirements.txt` | Pinned Python package dependencies |
| `README.md` | Complete option reference, reproduction commands, implementation notes |

No account-specific tracker, remote experiment dashboard, source-tree dependency,
or repository checkout is required. All results are written locally.

## Usage and hyperparameters

The command-line spelling uses underscores. Both `--search_type=value` and
`--search_type value` work. Inspect the fully resolved settings before a run:

```bash
python run.py --preset qwen_gsm8k --method pade3 --dry_run
python run.py --help
```

Configuration precedence is: base defaults, `--config` file, `--preset`, then
explicit command-line values. A preset chooses the model, dataset, learning
rate, training configuration, subset limits, and method-specific bound.
`adamw_cosine` always selects the cosine schedule. Run manifests store the
resolved settings, not just overrides.

### Methods

| `--method` | Behavior |
| --- | --- |
| `adamw` | Full-parameter AdamW with a constant learning rate by default |
| `taylor2` | Quadratic Taylor model, two symmetric probes |
| `taylor3` | Cubic Taylor model, two symmetric probes and backward derivative |
| `pade2` | Padé [1/1] model |
| `pade3` | Padé [1/2] model |
| `adamw_cosine` | AdamW with linear warmup followed by cosine decay |
| `prodigy` | Adaptive optimizer with its own distance/step estimate |
| `mezo` | Two Gaussian directional probes of a fixed-rollout RLVR objective |
| `zfo_seq` | Initial Taylor/Padé proposal followed by evaluated refinements |
| `muon` | Muon on hidden 2D weights; AdamW on embeddings, output head, and remaining parameters |

`muon` is an additional comparison, with no claimed target score in the paper.
Its matrix learning rate must be tuned independently; it is not the selected
AdamW rate. All methods here use full parameter training and the GRPO policy.
There is no quantization, adapter training, gradient clipping, or gradient
accumulation in the FO/ZFO reproduction path.

### Search coordinates: `search_type`

Let `theta_hat` be the tentative optimizer parameters, `d = theta_hat - theta`,
`eta` the optimizer learning rate, and `beta = bound`.

| Value | Search variable and interval | Physical update |
| --- | --- | --- |
| `optimizer_displacement` **(default)** | Multiplier `a` in `[0, beta]` | `theta + a*d` |
| `normalized_direction` | Distance `s` in `[0, R]`, `R = beta*eta*||g||`, `g = d/eta` | `theta + s*(d/||d||)` |

The default is the implemented large-model experiment convention from Appendix
C.1.1. The second is the normalized representation in the methodology. Since
`g` is the **optimizer-processed direction**, `eta*||g|| = ||d||`: these describe
the same physical interval in exact arithmetic. The normalized option fits and
maximizes in distance coordinates. Absolute numerical thresholds and floating
point arithmetic can therefore change decisions. It is an alternate experiment,
not the setting to select for reproducing the reported numbers.

For either representation, `perturbation_scale` is the physical probe distance
`epsilon`: probes use multipliers `+epsilon/||d||` and `-epsilon/||d||`.
Probes can lie outside the nonnegative search interval. The final selected
multiplier remains in `[0, bound]`. Optimizer moments advance on each tentative
update even when the selected multiplier is zero.

### Interior candidates: `search_inside`

| Value | Taylor2 / Taylor3 | Padé3 | Quadratic safeguard |
| --- | --- | --- | --- |
| `auto` **(default)** | Endpoints and real stationary points | Endpoints | Can select an interior point |
| `false` | Endpoints only | Endpoints only | Endpoints only |
| `true` | Endpoints and real stationary points | Endpoints and real stationary points, after pole checks | Can select an interior point |

```bash
python run.py --preset qwen_svamp --method pade3 --search_inside=true
python run.py --preset qwen_svamp --method taylor3 --search_inside=false
python run.py --preset qwen_svamp --method taylor2 --search_type=normalized_direction
```

The main LLM Padé3 experiments used endpoint selection (`search_inside=auto`), with interior steps possible only through the quadratic fallback triggered by poles or other numerical safeguards.

Padé3 uses the [1/2] rational form. Seven interior denominator checks detect
near-poles; they are not step-selection candidates. `false` also restricts the
quadratic safeguard to endpoints; `true` adds the analytic roots of the rational
derivative without extra model forwards.

Padé2 has no isolated interior stationary point when its denominator is regular,
so it remains an endpoint selector. Its historical degeneracy and denominator
handling are preserved. Taylor and Padé tie-breaking also follow the original
implementations; flat Taylor models prefer zero, while endpoint ties in Padé
prefer the upper bound.

For **ZFO-Seq**, `auto` preserves the separate sequential implementation,
including its 513-point Padé3 initializer grid. Explicit `true`/`false` changes
which candidates the **initial local model** considers; sequential refinements
can still evaluate and select interior points. The final sequential choice is
the best evaluated point in the interval, including the initial zero step.

### Model, data, training, and sampling options

Defaults below are the base configuration; presets can override the marked
training and data settings.

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `qwen` | Model alias below, or a public model repository ID |
| `--dataset` | `gsm8k` | `gsm8k`, `math`, `svamp`, `asdiv`, `openbookqa`, `arc_challenge`, `strategyqa`, `folio`, `codah` |
| `--method` | `taylor2` | Method from the table above |
| `--seed` | `100` | Python, NumPy, CPU, and CUDA seed |
| `--epochs` | `10` | Maximum complete training epochs; preset dependent |
| `--batches` | `4` | Number of prompts per minibatch; preset dependent |
| `--group_size` | `4` | Sampled responses per prompt, at least 2; preset dependent |
| `--maxlength` | `512` | Maximum prompt tokens; preset dependent |
| `--max_new_tokens` | `192` | Maximum generated response tokens |
| `--subset` | `1000000` | Training prefix cap, clamped to available data; preset dependent |
| `--eval_subset` | `1000000` | Evaluation prefix cap; also the MATH held-out size; preset dependent |
| `--asdiv_eval_fraction` | `0.2` | Fraction of distinct ASDiv problems held out if only one source split is available |
| `--asdiv_split_seed` | `0` | ASDiv partition seed, independent of training seed; shared across methods/models |
| `--temperature` | `0.8` | Training sampling temperature |
| `--top_p` | `0.95` | Training nucleus-sampling probability |
| `--kl_beta` | `0.02` | KL penalty coefficient, distinct from search `bound` |
| `--early_stop_patience` | `3` | Stop after this many exactly equal consecutive epoch scores; `0` disables |
| `--model_revision` | unset | Model commit or ref; unset uses the checked pin for the four known models |
| `--dataset_revision` | unset | Dataset commit or ref; unset uses the checked pin in `prep.py`; FOLIO uses its public source repository |
| `--data_dir` | unset | Replay a prepared directory containing `train.jsonl`, `eval.jsonl`, `data.json` |

Evaluation occurs after each epoch, using one greedy response per prompt. The
reported result is the **last** evaluation, not the best checkpoint. There is
no evaluation before epoch 1. Early stopping counts the first occurrence as 1:
three consecutive identical scores stop after the third occurrence.

| Model alias | Repository ID |
| --- | --- |
| `qwen` | `Qwen/Qwen2.5-Math-1.5B` |
| `phi2` | `microsoft/phi-2` |
| `gemma2_2b` | `google/gemma-2-2b` |
| `llama32_1b` | `meta-llama/Llama-3.2-1B` |

Gated models require access to their public distribution and an `HF_TOKEN` in
the environment. Tokens are not written to manifests.

### Optimizer and search options

| Option | Default | Meaning |
| --- | --- | --- |
| `--lr` | `5e-6` | AdamW learning rate; MeZO update rate; auxiliary AdamW rate for Muon. Preset dependent. Prodigy uses `prodigy_lr` instead. |
| `--bound` | `10` | Dimensionless upper multiplier `beta`; preset/method dependent; unused by plain optimizers and MeZO |
| `--perturbation_scale` | `1e-3` | ZFO physical probe distance; MeZO unnormalized Gaussian perturbation scale |
| `--search_type` | `optimizer_displacement` | Coordinate system described above |
| `--search_inside` | `auto` | Candidate policy described above |
| `--beta1`, `--beta2` | `0.9`, `0.999` | AdamW and Prodigy moment coefficients |
| `--adam_eps` | `1e-8` | AdamW and Prodigy denominator epsilon |
| `--weight_decay` | `0.01` | AdamW, Prodigy and Muon decay |
| `--lr_type` | `constant` | `constant` or `warmup_cosine` for AdamW-backed methods |
| `--warmup_ratio` | `0.1` | Fraction of the planned epoch-times-batch update count used for warmup |
| `--prodigy_lr` | `1.0` | Prodigy adjustment scale |
| `--d_coef` | `1.0` | Prodigy adaptation coefficient |
| `--d0` | `1e-6` | Initial Prodigy distance estimate |
| `--muon_lr` | `0.02` | Hidden matrix learning rate, separate from `lr` |
| `--muon_momentum` | `0.95` | Muon momentum, with Nesterov enabled |
| `--muon_ns_steps` | `5` | Muon Newton-Schulz iterations |
| `--zo_weight_decay` | `0.0` | MeZO decay; excludes bias and normalization parameters |
| `--zo_max_update_norm` | unset | Optional MeZO update norm limit; unset leaves updates unclipped |
| `--k` | `5` | ZFO-Seq extra objective calls including its first two symmetric probes; at least 2 |
| `--initial_model` | `p3` | ZFO-Seq initializer: `t2`, `t3`, `p2`, `p3` |
| `--tol` | unset | Sequential duplicate-point tolerance in multiplier units; unset means `0.001*bound` |

Prodigy uses decoupled decay and no external schedule. Muon uses the PyTorch
`match_rms_adamw` scaling rule, Nesterov momentum, default Newton-Schulz
coefficients and epsilon; tied embeddings/output weights are assigned to AdamW
exactly once. Muon and MeZO also require a constant external schedule here.
See the [Muon API](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html)
and [Prodigy implementation](https://github.com/konstmish/prodigy).

Cosine follows the historical `LambdaLR` indexing: warmup uses
`min(total_steps, max(1, round(warmup_ratio*total_steps)))` steps, and the schedule
advances after the optimizer update. Even `warmup_ratio=0` retains one warmup
step. Early stopping can terminate before the schedule reaches its end.

The old `alpha_fd_min`/`alpha_fd_max` options did not affect the supplied surrogate
trainer. They are omitted rather than exposed as ineffective controls.

### Execution and output options

| Option | Default | Meaning |
| --- | --- | --- |
| `--preset` | unset | One `model_dataset` row from the configuration table below |
| `--config` | unset | Replay a saved resolved `config.json` |
| `--suite` | unset | `table1`, `table2`, `table10`, `table11`, or `muon`; lists configurations by default |
| `--run_index` | unset | Zero-based suite entry to list or execute |
| `--execute` | off | Execute a suite, one fresh process per run; stops on failure |
| `--dry_run` | off | Print settings without loading models or training libraries |
| `--output` | `results` | Output parent directory; an existing run directory is never overwritten |
| `--prepare_only` | off | Download/select data and write its snapshot, without loading model weights |
| `--summarize` | unset | Aggregate completed runs below a given directory |
| `--self_test` | off | Run local numerical tests without downloading models or datasets |
| `--device` | `auto` | `auto`, `cuda`, or `cpu`; auto prefers CUDA |
| `--dtype` | `auto` | `float32`, `float16`, `bfloat16`, or auto: CUDA bf16 if supported, otherwise fp16; CPU fp32 |
| `--attn_implementation` | `auto` | Model default, `eager`, `sdpa`, or `flash_attention_2`; the latter requires its own compatible installation |
| `--deterministic` | off | Enable deterministic tensor algorithms; unsupported kernels raise an error |
| `--save_model` | off | Save final tensor weights, model configuration and tokenizer |

Boolean switches also accept `--no-deterministic` and `--no-save_model`.
`--search_inside` takes a value; it is not a valueless switch. Suite grids are
fixed: use a single preset or a saved configuration for modified experiments.
`--help` and `--dry_run` work with the Python standard library alone.

## Reproducing the experiments

### Install

`requirements.txt` pins the package versions used in the packaging checks.
Keep installed environments and generated outputs separate from the submission
folder. From this folder, use Python 3.12 in an isolated environment:

```bash
python3.12 -m venv ../zfo-env
source ../zfo-env/bin/activate
python -m pip install -r requirements.txt
python run.py --self_test
```

Install the CUDA build of the same PyTorch release suitable for the training
machine, using the [official installation instructions](https://pytorch.org/get-started/previous-versions/).
The original GPU software lockfile and exact model/data revisions were not
available. The four known models and nine datasets have immutable revision
pins in `prep.py`, checked during packaging. These are tested package dependencies
and asset pins, **not a recovered historical environment**. Large-model runs need a CUDA GPU with room for the policy,
frozen reference, optimizer state, and ZFO parameter snapshots. Full-model
memory requirements are much larger than inference requirements.

### Paper configuration table

The presets transcribe Tables 4-7. Seeds are `100`, `101`, `102` for Tables 1,
2 and 10. Each tuple lists bounds in the order **Taylor2, Taylor3, Padé2, Padé3**.

| Preset | FO/ZFO LR | MeZO LR | Bounds | Type |
| --- | --- | --- | --- | --- |
| `qwen_gsm8k` | 5e-6 | 1e-7* | 10, 5, 10, 10 | a |
| `qwen_math` | 5e-6 | 1e-6 | 5, 5, 5, 5 | c |
| `qwen_svamp` | 1e-5 | 5e-7 | 3, 5, 3, 3 | a |
| `qwen_asdiv` | 5e-6 | 5e-7 | 5, 5, 5, 5 | c |
| `qwen_openbookqa` | 1e-5 | 1e-7 | 10, 5, 5, 5 | a |
| `phi2_svamp` | 5e-6 | 5e-7 | 3, 5, 5, 3 | a |
| `phi2_asdiv` | 5e-6 | 1e-7 | 3, 3, 5, 3 | d |
| `phi2_openbookqa` | 5e-6 | 1e-7 | 3, 5, 3, 3 | d |
| `gemma2_2b_svamp` | 1e-6 | 5e-9 | 3, 3, 3, 3 | b |
| `llama32_1b_gsm8k` | 5e-6 | 1e-7 | 3, 3, 3, 3 | a |
| `llama32_1b_asdiv` | 5e-6 | 5e-8 | 5, 5, 3, 3 | c |
| `llama32_1b_openbookqa` | 1e-6 | 1e-7* | 5, 5, 5, 3 | b |
| `qwen_arc_challenge` | 1e-5 | - | 3, 5, 3, 3 | c |
| `qwen_strategyqa` | 1e-5 | - | 5, 5, 3, 5 | c |
| `qwen_folio` | 1e-5 | - | 3, 3, 3, 3 | c |
| `qwen_codah` | 1e-5 | - | 5, 5, 5, 5 | c |

\* The paper reports `ALL` for these MeZO cells. `1e-7` is an explicit
representative setting, not a uniquely recovered winning rate.

| Type | Epochs | Batch | Group | Prompt limit | Output limit |
| --- | --- | --- | --- | --- | --- |
| a | 10 | 4 | 4 | 512 | 192 |
| b | 8 | 2 | 2 | 384 | 192 |
| c | 8 | 8 | 2 | 384 | 192 |
| d | 5 | 4 | 2 | 256 | 192 |

The main and sub-study presets use the full named splits where available.
MATH uses 512 training and 256 evaluation rows. ASDiv uses an 80/20 split
with partition seed 0: 1,844 training and 461 evaluation examples. Duplicate
problem texts stay in the same partition. Training seeds 100/101/102 share
the same evaluation examples, and subset caps apply after partitioning.
The sequential comparison uses prefix caps of 512 and 256, including the
CODAH split's smaller available held-out set. Subset size and exact row
order matter.

### Run one cell or a full grid

```bash
# One cell, three seeds.
for seed in 100 101 102; do
  python run.py --preset qwen_gsm8k --method taylor2 --seed "$seed" --output ../zfo-results/table1
done

# Inspect a complete grid, then launch it.
python run.py --suite table1 --dry_run
python run.py --suite table1 --execute --output ../zfo-results/table1

# Or execute one indexed job, suitable for a scheduler array.
python run.py --suite table1 --run_index 0 --execute --output ../zfo-results/table1
```

Choose either individual jobs or the complete suite for a given output parent;
already-existing runs cause an error instead of being silently reused or
replaced. To rerun a cell, use a new output parent. No run resume is implemented.

| Suite | Contents | Jobs |
| --- | --- | --- |
| `table1` | 12 model/dataset settings x 6 methods x 3 seeds | 216 |
| `table2` | 4 additional Qwen datasets x 5 FO/ZFO methods x 3 seeds | 60 |
| `table10` | 8 Qwen datasets x AdamW-cosine/Prodigy x 3 seeds; excludes MATH | 48 |
| `table11` | ZFO-Seq on CODAH, ARC-Challenge, SVAMP; seed 100 | 3 |
| `muon` | 8 Qwen datasets x 3 seeds; additional untuned comparison | 24 |

```bash
python run.py --suite table2 --execute --output ../zfo-results/table2
python run.py --suite table10 --execute --output ../zfo-results/table10
python run.py --suite table11 --execute --output ../zfo-results/table11
python run.py --suite muon --execute --output ../zfo-results/muon
```

The Table 11 suite fixes Qwen, type c, LR `1e-5`, bound `3`, epsilon `1e-3`,
`k=5`, initializer `p3`, seed `100`, and subset caps `512/256`. It reproduces
the ZFO-Seq column only; Armijo, BB, Polyak and PLS are outside this package.
The classical/toy figures and additional appendix ablations are also outside
these large-model grids.

### Preserve data and repeat a run

```bash
python run.py --preset qwen_gsm8k --method taylor2 --prepare_only --output ../zfo-prepared
```

Every data preparation or training run saves its exact ordered selected rows.
To use a prepared run, replace `PREPARED_RUN` below with its directory. It must
match `dataset`, `seed`, `subset`, and `eval_subset`; content hashes are checked.

```bash
python run.py --config PREPARED_RUN/config.json --data_dir PREPARED_RUN --output ../zfo-results/frozen
```

For a trained run, `environment.json` records the resolved model commit and
`data.json` records the dataset commit. To repeat it, replace `MODEL_COMMIT`
with that recorded model SHA and `RUN` with that run directory:

```bash
python run.py --config RUN/config.json --data_dir RUN \
  --model_revision MODEL_COMMIT --output ../zfo-results/repeat
```

Presets use the checked immutable pins in `prep.py`; they do not claim to
identify the historical upstream revisions used in every paper run. Custom
model repositories without a packaged pin resolve their requested ref (or
`main`) once at load time. Save snapshots and use the recorded commit plus the
recorded software versions for later reruns.
The data loader fails if a designated dataset/split is unavailable; it never
silently switches to a different mirror.

### Outputs and aggregation

A training run creates:

- `config.json`: resolved hyperparameters, without the local prepared-data path.
- `environment.json`: package versions, device/dtype, model commit, and source-file hashes.
- `data.json`, `train.jsonl`, `eval.jsonl`: exact selected examples, provenance and hashes.
- `metrics.jsonl`: every update and epoch evaluation, including selected steps and fallback flags.
- `result.json`: completion status, final accuracy, stopping epoch, elapsed time and memory.
- `model/`: final tensors and tokenizer, only with `--save_model`.

```bash
python run.py --summarize ../zfo-results/table1 > table1_summary.jsonl
```

Aggregation groups matching configurations, source hashes, resolved revisions,
dtypes and package versions. It rejects duplicate seeds and reports seed
coverage, the mean percentage, and both sample and population standard
deviations. A three-seed claim requires `paper_seeds_complete: true`.
Table 11 intentionally uses one seed. Accuracy in per-run files is a fraction;
summary means and standard deviations are in percentage points.

### Implementation fidelity and limits

The training and evaluation implementation has the following properties:

1. The GRPO backward pass supplies the update direction and first-order
   coefficient. In maximization notation, its objective is
   `F(theta) = mean(A * sequence_logp(theta)) - kl_beta * K(theta)`, where
   `K` is the mean signed token log-ratio estimate. The symmetric probes use
   `Q(theta) = mean(frozen_rewards) - kl_beta * K(theta)` on the same sampled
   sequences. The local step-selection model combines the slope from `F`
   with values from `Q`. Freezing rewards and advantages does not freeze
   `sequence_logp(theta)`, so `F` and `Q` remain distinct objectives. Guarantees
   requiring a slope and probe values from one scalar objective do not directly
   cover this mixed construction.
2. FO/ZFO masks include nonpadding prompt and completion tokens. With left
   padding, the historical decoder slices each completion at its unpadded
   prompt length, which can include part of the prompt. Training rollouts run
   in training mode; probes and greedy evaluation use evaluation mode. When
   dropout is active, this mode change can affect forward values before any
   parameter update; fixed rollouts do not imply a shared dropout realization.
   The pinned Phi-2 configuration has residual dropout probability 0.1.
   These details are retained because changing them can change rewards and updates.
3. ASDiv uses the 80/20 data protocol described above. `data.json` records
   the partition seed, counts, and content hashes. MATH uses a seeded shuffled
   training-split holdout; CODAH uses a seeded 90/10 split. StrategyQA prompts
   include the provided facts and decomposition hints.
4. MeZO follows the available MeZO source, which has a completion-only mask,
   padded-width decoding and a nonnegative k3 KL estimate. Its behavior is not
   identical to the FO/ZFO probe objective. Its in-place symmetric perturbation
   restoration also inherits finite-precision rounding.
5. Padé3 has pole and near-pole checks with quadratic fallback. Padé2 retains
   the original endpoint denominator handling rather than introducing a new
   interval-wide pole safeguard. Thresholds remain coordinate dependent.
6. The selected configurations are transcribed from the paper and checked
   against available launchers, not reconstructed from a complete immutable
   manifest of the exact three runs behind every published cell. The Qwen/MATH
   preset uses type c (8 epochs, batch size 8, group size 2, and prompt limit
   384), matching the revised Table 6 and the available MATH launcher.

Accordingly, this is an executable reproduction package with preserved default
numerical logic and explicit settings, **not a certification that fresh runs
will match the published means**. Historical model/data commits, the exact
GPU environment, and the complete selected-run manifest are still needed for
that certification. 

### Validation

The package includes `--self_test` for data partitioning, endpoints, interior roots, equivalent
physical probe coordinates, the sequential budget, cosine scheduling, and grid
sizes. Additional packaging checks compare scalar selectors, prompts, verifiers,
schedules and generation against the supplied source, and exercise tiny local
training runs for every method. Forty-eight checks passed, including exact final
tensor agreement with the original training loop on seven small FO/ZFO cases.
All nine dataset loaders were checked against their pinned sources. A command-line
run with a small downloaded model also completed training, evaluation and result
aggregation. These checks do not substitute for full GPU reruns of the paper grids.
