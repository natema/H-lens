# J²-lens: testing second-order corrections to the Jacobian lens

## Research question

Can a diagonal second-order correction make the Jacobian lens more faithful on early-layer examples where the ordinary J-lens fails?

More specifically:

1. How much curvature of the residual-to-final-residual map lies in the coordinate-diagonal Hessian, rather than cross-coordinate terms?
2. Does adding that diagonal curvature improve prediction of concepts that the J-lens recovers only late or not at all?
3. Does it explain failures beyond what is already recovered by the R-lens?

The project should produce a clear positive or negative result. It need not establish a generally superior lens.

## Motivation

The J-lens approximates the downstream computation by its average first-order Jacobian. Known failures, especially at early layers, may arise because this linearization ignores important curvature. A diagonal Hessian correction is the simplest scalable second-order extension and is analogous in spirit—but not in mathematical setting—to the diagonal Hessian approximation in *Optimal Brain Damage*.

The central uncertainty is whether diagonal curvature contains useful signal.  Residual-stream coordinates are basis-dependent, and individually small cross terms can matter collectively. This is therefore an empirical hypothesis, not an assumption to build into the conclusion.

## Definition

Let

$$
F_\ell:\mathbb R^d\longrightarrow\mathbb R^d
$$

denote the downstream map from the residual at layer $\ell$ to the final
residual representation used by the lens. The existing J-lens estimates the
average Jacobian

$$
J_\ell=\mathbb E[\nabla F_\ell(x)].
$$

The proposed derivative-defined correction estimates

$$
D_{\ell,:,j}=\mathbb E\left[
  \frac{\partial^2F_\ell(x)}{\partial x_j^2}
\right],
$$

so that $D_\ell\in\mathbb R^{d\times d}$, the same size as $J_\ell$.
The initial candidate lens is

$$
\widehat F^{J^2}_\ell(x)
=b_\ell+J_\ell\delta
+\frac12D_\ell\left(
  \delta\odot\delta-\mathbb E[\delta\odot\delta]
\right),
\qquad \delta=x-\mu_\ell.
$$

The centering, intercept, normalization, and any scalar calibration must be fixed on development data and reported explicitly. They must not be tuned on the held-out evaluation cases.

This is not a full quadratic regression. A full Hessian would additionally use

$$
\sum_{j<k}H_{\ell,:,j,k}\,\delta_j\delta_k.
$$

Those cross terms could be probed statistically or inside a small predetermined subspace, not fitted freely in all $d(d+1)/2$ quadratic features, but we will probably not have time to look at that.

## Hypotheses and alternatives

Primary hypothesis:

> Some early-layer J-lens failures are caused by locally important curvature,
> and a diagonal Hessian correction recovers a measurable fraction of them.

Plausible alternatives:

- The downstream map is curved, but the curvature is predominantly
  off-diagonal in the residual coordinate basis.
- Curvature is prompt-local and cancels when averaged into a universal lens.
- The main problem is attribution through RMSNorm and gated MLPs, so R-lens
  helps while a Taylor correction does not.
- Apparent gains come only from an intercept, rescaling, extra capacity, or
  selecting examples after seeing J²-lens results.
- A second-order Taylor approximation remains poor because the relevant
  displacement is too large or higher-order effects dominate.

## Evaluation cases

First reproduce several published J-lens failures, including examples of the following kinds:

- typo correction, such as `aganst` → `against`;
- associations, such as sushi → Japan and Michael Jordan → basketball;
- factual completion, such as Verona → Italy;
- category or profession recovery, such as Einstein → physicists.

Then construct a modest evaluation set across the same broad categories. Split it before examining J²-lens outputs:

- **development set:** implementation checks and fixed calibration choices;
- **held-out set:** the actual comparison.

An evaluation item may be selected because the ordinary J-lens fails, but never because J²-lens succeeds. Preserve the selection rule and all selected items.

## Models

- Primary target: the modern Qwen model for which the existing J-lens artifacts and known failures can be reproduced, provisionally Qwen3.5-4B, then 9B, then if time allows Qwen3.6-27B.
- Use only one main model in the reported experiment unless the primary model is technically blocked.

Jean Zay can be used for GPU-heavy estimation, if Maserati's GPU cannot handle it.
On Jean Zay, model weights, dependencies, prompts, and configs must be staged before compute jobs because compute nodes have no outbound internet.

## Experimental sequence

### 1. Reproduce the baselines

Reproduce ordinary logit-lens, J-lens, and R-lens behavior on at least three known failure cases. Do not proceed to a large curvature job until the layer, position, normalization, and unembedding conventions agree with the reference implementation.

### 2. Validate second derivatives (not urgent but sanity check)

This we don't manually check, we vibe-code it and only inspect it manually if it raises anomalies.
Actually, it might be worth doing this later, after quickly testing the main idea.

On the small model and one layer:

- check Hessian-vector or mixed directional derivatives against centered finite differences;
- check symmetry, $H[u,v]\approx H[v,u]$;
- compare at least two precisions or step sizes;
- verify that gradients refer to an intervention at the intended layer rather than unintended cross-layer graph dependencies (i.e. that derivatives are taken carefully w.r.t. what they should be taken from).

### 3. Test the diagonal-curvature premise cheaply

Before estimating all of $D_\ell$, use randomized second-directional probes.
For independent standardized random directions $u,v$,

$$
\mathbb E\|H_\ell[u,v]\|^2=\|H_\ell\|_F^2.
$$

Coordinate sampling estimates the diagonal energy:

$$
d\,\mathbb E_j\|H_\ell[e_j,e_j]\|^2=\|D_\ell\|_F^2.
$$

Estimate their ratio at a few early, middle, and late layers, and also measure the approximation error along natural activation displacements. This decides whether a complete diagonal estimate is worth computing.

### 4. Estimate the diagonal correction

If the premise survives, estimate $D_\ell$ for the selected layers.
Claude suggests initially using Hutchinson-style diagonal estimation or another checked second-order estimator.
Not clear to me that we should do that instead of just using autodiff on [e_j,e_j].
If we go for Hutchinson, we should increase the number of probes until the lens-level metrics stabilize; do not require entrywise convergence when it is irrelevant to the conclusion.

Start with a small set of early layers where J-lens fails. Expand across every layer only if the implementation already works and the early result warrants it.

### 5. Evaluate and try to falsify the result

Compare on held-out cases:

1. logit lens;
2. J-lens;
3. J-lens plus matched intercept/rescaling controls;
4. J²-lens;
5. R-lens.

Primary outcomes are target-token rank and top-$k$ recovery by layer.
Secondary outcomes are divergence from the model's final token distribution and
cosine or normalized error against the final residual representation.

Also inspect raw examples and report failures. A useful negative result could be
that the Hessian has substantial curvature but diagonal terms fail because
cross interactions dominate.

## Minimal successful deliverable

A reproducible result on one model containing:

- several verified J-lens failure examples;
- a checked implementation of second-directional derivatives;
- a layerwise estimate of diagonal versus total curvature;
- a held-out comparison of J-lens, J²-lens, and R-lens;
- enough negative controls to distinguish curvature from trivial calibration;
- a short main write-up explaining the result and its limitations.

Computing a dense $d\times d\times d$ Hessian at every layer is not required.

## Stop, pivot, and scope-cut conditions

- **Baseline gate:** if reference J-lens failures are not reproduced...
- **Autodiff gate:** if reliable second derivatives are not available by the end of the implementation block...
- **Curvature gate:** if diagonal curvature is negligible relative to total
  curvature...
- **Performance gate:** if J²-lens shows no development-set improvement...
- Prefer fewer layers and stronger validation over a nominal all-layer sweep.

## Counted-time budget

Plan proposed by Claude:

| Active work | Hours |
|---|---:|
| Freeze definitions, cases, metrics, and environment | 1.0 |
| Reproduce J-lens and R-lens baselines | 2.0 |
| Implement and validate second derivatives | 3.0 |
| Run curvature-energy diagnostic and decide the gate | 2.0 |
| Estimate J² correction on selected layers | 2.0 |
| Held-out evaluation, controls, and manual inspection | 3.0 |
| Figures, reproducibility checks, and main write-up | 3.0 |
| **Total counted project work** | **16.0** |

Work should normally be performed in 1–2 hour sessions.

## First counted session

Budget: 1–2 hours.

Concrete objective: freeze the mathematical definition and evaluation protocol, then reproduce one known J-lens failure with the existing artifacts. The session should end with a tiny executable baseline and a decision about the exact model, layers, positions, and development/held-out split—not with a cluster-scale job.

## Core references

- [J-lens / global workspace](https://transformer-circuits.pub/2026/workspace/index.html)
- [Anthropic J-lens code](https://github.com/anthropics/jacobian-lens)
- [R-lens](https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers)
- [Short Horizons and Sparse Concepts](https://arxiv.org/abs/2608.25347)
- [Optimal Brain Damage](https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html)
