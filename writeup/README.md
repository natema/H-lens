# Write-up notes

Working notes for the write-up, not prose. Each file collects facts established
in this project with the numbers needed to state them, so the argument can be
assembled without re-deriving anything.

| file | contents |
|---|---|
| `01-result.md` | the central claim and the evidence for it |
| `02-dataset.md` | how the evaluation set was built and why |
| `03-pitfalls.md` | measurement traps found, several of which changed a conclusion |
| `04-limitations.md` | what this does not show |
| `05-open-questions.md` | what a follow-up should do |
| `06-provenance.md` | compute, cost, artifacts, reproducibility |

The rule applied throughout: every number here is measured and traceable to a
file in `results/`, `data/`, or `data_fp32/`. Where a claim was made during the
work and later found wrong, the correction is recorded rather than the claim
quietly replaced — those reversals are among the more useful content.
