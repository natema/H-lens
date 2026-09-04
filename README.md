# H-lens

**Does a diagonal Hessian correction make the Jacobian lens more faithful?**
A controlled negative result on Qwen3.5-4B, and a 3,344-item dataset of
Jacobian-lens successes and failures that any alternative lens can be scored on.

The [Jacobian lens](https://github.com/anthropics/jacobian-lens) (J-lens) reads
intermediate concepts out of a transformer's residual stream by pushing the
residual at a token through the *average Jacobian* of the downstream computation
and unembedding the result. It fails on some early-layer cases where the
R-lens does not. The H-lens adds the cheapest second-order term, a
development-averaged **diagonal Hessian** correction in the spirit of Optimal
Brain Damage:

```text
H(x) = J·x + ½·D·((x − μ)² − Var[x]),      D[:, j] = E[∂²F/∂x_j²]
```

with `D` the same size as `J`, everything estimated on pretraining text, and no
free parameter. The results are laid out in [`data/RESULTS.md`](data/RESULTS.md)
and [`data/README.md`](data/README.md); [`docs/STAGES.md`](docs/STAGES.md)
covers the earlier stages.

## Findings

- **The correction does nothing.** On 3,344 held-out items the H-lens is
  statistically indistinguishable from the plain J-lens, and indistinguishable
  from a version of itself with the coordinate structure destroyed. At layer 12
  it moves exactly zero items out of the J-lens-failure cell.
- **A selection trap that would have produced a false positive.** Evaluated only
  on the items where the J-lens fails, the correction beats the J-lens 601 to
  510 (sign test p = 0.007). On the same items the coordinate-shuffled control
  wins 630 to 463 (p = 5×10⁻⁷). That cell is *defined* by J-lens failure, so
  regression to the mean flatters any perturbation. Every cell is scored, not
  only the failures.
- **The dataset validates itself against a known effect.** On it the R-lens
  reverses across layers exactly as its authors claim: it removes 161 J-lens
  failures at layer 6 and adds 91 at layer 12. The same instrument that moves
  161 items for the R-lens moves 2 for the Hessian correction.

## The dataset

Each item is a sentence fragment that **ends on a probe word**, with a target
concept drawn from Qwen3.5-4B's own vocabulary (3,344 concrete nouns). Two
independent readings are taken at the probe: does the J-lens readout *name* the
concept, and does the model itself name it when shown the same prefix and asked.
The J-lens side is its top-50 collapsed to the first 10 distinct concepts, so
both lists carry ten concepts. A model judge decides both readings with the same
tolerance; nothing is filtered on agreement.

| cell | n (strong) | meaning |
|---|---:|---|
| `self_report_only` | 1040 (319) | the model has the concept, the J-lens misses it |
| `both` | 974 (247) | positive control |
| `lens_only` | 347 (74) | the lens finds it, the self-report does not |
| `neither` | 983 (186) | the model does not hold the concept at the probe |

A lens-blind grader marks 826 items `strong`. Start at
[`data/README.md`](data/README.md) for the file formats, the known measurement
noise, and the reasoning behind each design choice; the per-cell tables under
[`data/browse/`](data/browse/README.md) are the fastest way to look at items.

## Layout

| path | contents |
|---|---|
| `src/j2_lens/` | the package: baselines, forward-mode Hessian estimation and evaluation (`evaluation.py`), dataset construction (`jspace.py`, `dataset.py`, `concepts.py`, `scoring.py`, `collapse.py`), API cost ledger (`spend.py`) |
| `data/` | the dataset, its raw-top-10 predecessor, quality grades, every lens's readouts and cells |
| `configs/` | the concept list (artifact of record), the 33-case battery, the frozen development/evaluation splits |
| `results/` | JSON results with full provenance for every stage; the fitted operators (`*.pt`, 26 MB each) are regenerable and not tracked |
| `pilot/` | the scripts that build, grade, collapse and evaluate the dataset, the judge fixtures, and the API spend ledger |
| `docs/STAGES.md` | stage-by-stage working notes, from the first three-case reproduction to the pretraining-fitted operator |
| `PROJECT_IDEA.md`, `PROJECT_LOG.md` | the plan converged on before starting, and the time log |

The code predates the name: the package is `j2_lens`, the commands are `j2-*`,
and the corrected lens appears as `J²` in result files and notes.

## Reproducing

```bash
uv sync                                   # Python 3.11–3.13, PyTorch, the jlens package (pinned)
uv run j2-baselines                       # logit / J / R lens on the published failure cases
uv run j2-screen --offline                # lens-blind screen of the 33-case battery
```

Fitting the operator is one forward-over-forward pass per (residual coordinate,
development pair) and was run on H100s, four coordinate shards per layer merged
with `j2-merge`; about 28 GPU-hours for layers 6 and 12:

```bash
uv run j2-evaluate --offline --estimator forward --layer 12 \
  --cases configs/battery_cases.json --split configs/evaluation_split_pile.json \
  --hessian-pairs 128 --coordinate-batch-size 1 \
  --artifact results/hessian_pile_l12_qwen3.5-4b.pt \
  --output results/evaluation_pile_l12_qwen3.5-4b.json
```

Building the dataset needs a GPU for the readouts and a Mistral API key
(`MISTRAL_API_KEY` in the environment or in `.env`) for GLM-5.2, which writes
the fragments, judges the readouts, grades the items and collapses the lists.
Every paid call is appended to `pilot/spend.json`; the whole project cost about
$65.

```bash
uv run j2-concepts                        # vocabulary → configs/concepts.json (not bit-reproducible; see its docstring)
uv run j2-dataset --phase generate        # fragments ending on the probe
uv run j2-dataset --phase read --dtype float32   # J-lens readouts and self-reports
uv run j2-dataset --phase judge           # cells on the raw top-10
uv run python pilot/score_dataset.py      # lens-blind quality grades
uv run python pilot/collapse_lists.py --lists data/lists50_l12.jsonl --out data/collapsed_jlens_l12.jsonl --methods j_lens
uv run python pilot/judge_method_cells.py --lists data/collapsed_jlens_l12.jsonl --out data/method_cells_collapsed_jlens_l12.jsonl
uv run python pilot/apply_collapsed_cells.py     # promote the collapsed cells to data/dataset.jsonl
uv run python pilot/split_cells.py        # per-cell files and browse tables
uv run python pilot/eval_hlens.py         # H-lens, shuffled control and R-lens on every item
```

`uv run pytest` runs the unit tests (padding exactness, estimator agreement,
baseline bookkeeping) on toy models without downloading anything.

## Provenance

Model `Qwen/Qwen3.5-4B` and the J-lens/R-lens matrices from
`camilablank/workspace-lenses` are pinned by revision and SHA-256. The
development corpus is `NeelNanda/pile-10k`, the corpus the published J-lens was
fitted on, so first- and second-order operators come from the same distribution
and no evaluation item informs the fit. Every model-facing prompt is a module
constant. The code was written with Codex on the first day and with Claude Code
afterwards, with GLM-5.2 as the instrument inside the pipeline; `PROJECT_LOG.md`
records the timeline and `docs/STAGES.md` and `data/README.md` record how each
part was checked, including the checks that overturned an earlier conclusion.

## License

MIT. See [`LICENSE`](LICENSE).
