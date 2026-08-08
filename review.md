# Code Review — LearnSim (A Variational Bayesian Method for Similarity Learning in Non-Rigid Image Registration, CVPR 2022)

Scope: full repository at commit `7158597` (all of `base/`, `data_loader/`, `logger/`, `model/`, `optimizers/`, `trainer/`, `utils/`, `parse_config.py`, `train.py`, `test.py`, `tests/`, `configs/`).

## Summary

The codebase implements a well-structured amortized/per-pair variational-inference pipeline for non-rigid registration: a stationary-velocity-field (SVF or SVFFD) transformation model, a learnable similarity metric (small 3D U-Net + 1×1×1 aggregation, initialised to approximate SSD/LCC), Sobolev-gradient preconditioning via a custom `autograd.Function`, antithetic low-rank-Gaussian sampling for variance reduction, and full `DistributedDataParallel` training with per-image-pair persisted variational parameters. Module boundaries (config → data loader → model → losses → transformation/registration → trainer) are clean and the `ConfigParser`-driven object construction is a nice pattern for reproducible experiments. Tests cover the numerically trickiest primitives (gradient/Jacobian-determinant, warping, closed-form entropy, DSC parity between CPU/GPU).

The issues below are grouped by severity. Nothing here suggests the published results are wrong — the flagged correctness bugs are either in code paths unused by the released configs, or in secondary metric bookkeeping — but several are worth fixing for anyone extending or reproducing the work.

---

## Correctness issues

### 1. `calc_DSC_GPU`: the zero-overlap fallback never triggers — Dice becomes `NaN` instead of `0.0`
`utils/util.py:96-121`

```python
try:
    score = numerator / denominator
except:
    score = 0.0
```
`numerator` and `denominator` are 0-dim `torch.Tensor`s. Tensor division by zero does **not** raise `ZeroDivisionError` — it silently produces `nan` (0/0) or `inf`, following IEEE‑754 float rules. So whenever a structure is absent from both the fixed and moving segmentation (a real, non-rare occurrence for small subcortical structures), `score` becomes `nan`, and the `except` branch is dead code that never executes.

This matters because `calc_DSC_GPU` is on the hot path (`calc_metrics` → called every batch, in both `_step_q_v` and `_test`, with `GPU=True` by default). The resulting `NaN`s flow into `MetricTracker.update_avg_metrics`, where `pandas.Series.mean()` silently drops `NaN`s — so the reported per-structure/overall average DSC quietly excludes affected image pairs rather than counting them as 0, which is presumably not the intended semantics given the explicit `except: score = 0.0` fallback that was written.

**Suggested fix:** guard explicitly, e.g. `score = numerator / denominator if denominator > 0 else 0.0`.

### 2. `SVF_2D.forward` crashes — `self.identity_grid.permute` is never called
`utils/transformation.py:47`

```python
transformation = self.identity_grid.permute + displacement
```
This adds a bound method object (`Tensor.permute`) to a tensor, which raises `TypeError`. Compare with the (correct) 3D counterpart two classes down:
```python
transformation = self.identity_grid + displacement   # SVF_3D, line 75
```
`SVF_2D` is not referenced by any config (all released configs use `SVF_3D`/`SVFFD_3D`), and the only tests that exercise it are commented out in `tests/test_utils.py` (`test_scaling_and_squaring_2D_translation`, `test_scaling_and_squaring_2D_rotation`) — which is presumably *why* this has gone unnoticed. If 2D support is meant to still work (the class and commented-out tests suggest it was), this needs a one-line fix and its tests re-enabled.

### 3. Resuming with a new `-c/--config` silently drops nested config keys
`parse_config.py:93-97`

```python
config = read_json(cfg_fname)
if args.config and resume:
    config.update(read_json(args.config))
```
`dict.update` is a shallow merge. If someone resumes a run and passes a new config intending to override just one nested field (e.g. `{"trainer": {"no_epochs": 100}}`), the entire `"trainer"` sub-dict from the checkpoint's config is replaced wholesale — `save_dir`, `no_iters_q_v`, `uniform_noise`, etc. are silently lost rather than merged, likely producing a `KeyError` deeper in `BaseTrainer.__init__` or `Trainer.__init__` (or worse, quietly falling back to whatever default the consuming code assumes). Given the intended use is "resume, but override a couple of keys," a deep merge (or an explicit warning that overrides are shallow) would avoid this footgun.

### 4. Abstract-method contracts aren't actually enforced on two base classes
`base/base_model.py`, `base/base_dataset.py`

`BaseModel(nn.Module)` declares `encode()` as `@abstractmethod`, and `BaseImageRegistrationDataset(Dataset)` declares `__preprocess`, `_preprocess_im`, `_preprocess_mask_or_seg` the same way — but neither class mixes in `ABC`/`ABCMeta`, so Python never enforces these as abstract; a subclass that forgets to override one just gets an `AttributeError`/`NotImplementedError` at first *call* time instead of a clear `TypeError` at instantiation time. This is inconsistent with the rest of the codebase, which does it correctly elsewhere (`DataLoss(nn.Module, ABC)`, `RegLoss(nn.Module, ABC)`, `TransformationModule(nn.Module, ABC)`, `DifferentialOperator(nn.Module, ABC)` in `model/loss.py`, `utils/transformation.py`, `utils/diff_op.py` all correctly combine `nn.Module` with `ABC`). Worth aligning `BaseModel`/`BaseImageRegistrationDataset` the same way, i.e. `class BaseModel(nn.Module, ABC)`.

Relatedly, `BaseImageRegistrationDataset.__preprocess` (name-mangled, double leading underscore) is never actually reachable as a base-class contract anyway, since name mangling ties it to `_BaseImageRegistrationDataset__preprocess` while `BiobankDataset.__preprocess` mangles to a different, unrelated name (`_BiobankDataset__preprocess`) that just happens to be called consistently from within `BiobankDataset` itself. It works today, but the intended "override this" abstract-method contract is effectively fictional here — a single leading underscore (as used for `_preprocess_im`/`_preprocess_mask_or_seg`) would be both correct and less confusing.

### 5. `EntropyMultivariateNormal.forward` dispatches on **number** of kwargs, not their identity
`model/loss.py:154-175`

```python
def forward(self, **kwargs):
    if len(kwargs) == 2:
        ...  # closed-form entropy of q(v), expects log_var, u
    elif len(kwargs) == 4:
        ...  # MC estimate of -log q(v_sample), expects sample, mu, log_var, u
```
Correct for the two call sites that exist today (`trainer/trainer.py:56` and `:79`), but this is a fragile contract: passing any other 2-key or 4-key combination (e.g. a typo'd kwarg name) fails with a `KeyError` deep inside instead of a clear signature error, and there's no defence against someone adding a third calling convention with 2 or 4 kwargs that collides with an existing branch. Prefer two explicitly named methods (or a `mode=` argument) over dispatching on `len(kwargs)`.

---

## Robustness / portability

### 6. `optimizers/adam_rate_decay.py` uses long-removed PyTorch APIs
`optimizers/adam_rate_decay.py:85-97` — `exp_avg.mul_(beta1).add_(1 - beta1, grad)`, `exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)`, `p.data.addcdiv_(-step_size, exp_avg, denom)`.

These are the pre-1.5 positional-`alpha` overloads of `add_`/`addcmul_`/`addcdiv_`, deprecated for several years and removed entirely in current PyTorch (`add_()` now requires `alpha=` as a keyword; the 3-positional-arg form of `addcdiv_`/`addcmul_` raises `TypeError`). `requirements.txt` pins `torch==1.7.1`, so this works as shipped, but it means the custom `optimizer_q_v`/`optimizer_q_phi`/`optimizer_q_f` (all configured as `"type": "Adam"` pointing at `torch.optim`... actually verify: `config.init_obj('optimizer_q_v', torch.optim, ...)` uses **`torch.optim.Adam`**, not this custom one) — double-check whether `optimizers/adam_rate_decay.py` is even wired up (see note 6a below); if it's meant to be used, it will hard-fail on any modern PyTorch install, which is a real barrier to reproducing this work today.

**6a.** Grepping the repo, `optimizers.adam_rate_decay.Adam` is never imported by `parse_config.py` or anywhere else — all configs use `"optimizer_q_*": {"type": "Adam", ...}` resolved against `torch.optim` (`base/base_trainer.py:157,161`, `trainer/trainer.py:317`), i.e. the *stock* `torch.optim.Adam`. So `optimizers/adam_rate_decay.py` appears to be dead code today; worth confirming and either wiring it in (if the rate-decay/reinit behaviour is actually desired) or removing it, since as-is it's both unused and unusable on current PyTorch.

### 7. Hardcoded, machine-specific absolute paths
- `data_loader/biobank_dataset.py:11`: `data_path = '/vol/biodata/data/biobank/12579/brain/t0/affine_to_mni/images/'`
- All `configs/**/*.json`: `"im_pairs": "/vol/bitbucket/dig15/datasets/mine/biobank/..."`
- `tests/test_setup.py:64`, `tests/test_metrics.py`: same `/vol/bitbucket/dig15/...` path, plus tests hardcode `device = 'cuda:0'` (so the test suite can't run on CPU-only machines at all).

Understandable for a research cluster, but anyone outside the original Imperial/Biobank environment cannot run training, testing, or the test suite without manually patching source (not just config) — in particular `data_path` in `biobank_dataset.py` isn't config-driven at all, unlike everything else in the pipeline. Making it a constructor argument (already partially there via `save_dirs`/`im_pairs`) would make the released code actually runnable end-to-end by third parties given their own copy of the data.

### 8. Run-directory naming can collide under concurrent launches
`parse_config.py:29-58` — `run_id` comes from `timestamp = datetime.now().strftime(r'%m%d_%H%M')` (minute resolution only), and directories are created with `exist_ok = (run_id == '')`, i.e. `exist_ok=False` in the normal case. Two training/testing jobs (e.g. SSD and LCC configs, or SVF and SVFFD) launched within the same minute will race on `mkdir` and one will crash with `FileExistsError`. Adding seconds (or a short random suffix) to the timestamp format would remove this sharp edge cheaply.

---

## Test coverage gaps

### 9. The exact bug class already fixed once (non-square grids) is not regression-tested
`tests/test_utils.py: test_separable_conv_3D`, `test_cubic_B_spline_FFD_3D`, `test_SVFFD_3D` all use cubic volumes (`D = H = W = 16`, or `dims = (64, 64, 64)`). Commit `25885ab` ("Fix separable convolution when using images not on a square grid") shows `separable_conv_3D`'s reshape logic previously mixed up `D`/`H`/`W` for non-cubic input — exactly the kind of silent shape bug that's easy to reintroduce and hard to catch by inspection. None of the current tests would catch a regression here, since every test volume used is a cube. Given production configs always use `dims: [128,128,128]` this is low-urgency, but a single test with `D≠H≠W` for `separable_conv_3D`/`Cubic_B_spline_FFD_3D` would close a real hole cheaply (the pattern already exists elsewhere — `dims_v_non_square` is defined and used in `test_grad.py`/`test_warping.py`, just not plumbed through to these two tests).

### 10. `EntropyMultivariateNormal`'s low-rank branch is untested
`tests/test_loss.py: test_entropy` only exercises `u_v = 0` (both for the deterministic and the random-log-variance case), which reduces the low-rank-plus-diagonal formula to the plain diagonal-Gaussian closed form. The `u ≠ 0` path (the actual low-rank correction, i.e. the interesting part of `EntropyMultivariateNormal.forward`'s `len(kwargs)==2` branch — the `log1p(...)` matrix-determinant-lemma term) has no test verifying it against a reference (e.g. `torch.logdet` on the explicit `sigma^2 I + u u^T` covariance for a small case). Since this term feeds directly into the training ELBO, a small closed-form cross-check would be valuable.

---

## Minor / dead code

- `model/distributions.py`: `DirichletPrior`, `LogPrecisionExpGammaPrior`, `LogEnergyExpGammaPrior`, `LogScaleNormalPrior`, `ExpInverseGammaDistribution` are never referenced outside this file or any config — apparently hyperprior building blocks for extensions not used by the released models. Harmless, but worth a `# unused, kept for future extensions` note or moving to a separate module so readers don't go looking for where they're wired in.
- `utils/functions.py`: `SGLD`, `GaussianGrad`, `Gaussian_kernel_1D` are likewise unreferenced. `GaussianGrad.forward` uses `torch.rfft`/`torch.irfft` (`utils/functions.py:99-100`), removed in PyTorch ≥1.8 in favour of `torch.fft.rfftn`/`torch.fft.irfftn` — currently harmless only because the class is never instantiated, but it will raise `AttributeError` immediately on any modern-PyTorch environment if someone tries to use it.
- `base/base_trainer.py:59-60`: `self.log_period` and `self.log_period_var_params` are initialized from config but the code says `# NOTE (DG): unused` — dead config knobs that configs still don't set (`log_period_var_params` is absent from every shipped config, so that whole branch in `BaseTrainer.train()` — periodic `copy_var_params_to_backup_dirs` — never fires in practice). Either wire it up or remove it to reduce config surface area.
- `model/__init__.py` is empty (relies on submodule imports `model.model`, `model.loss` elsewhere), while `base/__init__.py`, `utils/__init__.py`, `data_loader/__init__.py`, `logger/__init__.py` all re-export via `from .x import *`. Not a bug (nothing does `from model import Foo`), just an inconsistency worth a comment if intentional.

---

## Worth a second look against the paper (not claimed as bugs)

`trainer/trainer.py:79-82`, the ELBO's entropy term for `q(v)` mixes a closed-form analytic term with two Monte-Carlo (antithetic) sample-based terms:
```python
entropy_term3 = self.entropy_loss(log_var=var_params_q_v['log_var'], u=var_params_q_v['u'])       # closed form
entropy_term = entropy_term1.sum() / 2.0 + entropy_term2.sum() / 2.0 + entropy_term3.sum()          # + 2 MC terms
```
The closed-form entropy of a Gaussian doesn't normally need an additional Monte-Carlo estimate on top — it's plausible this is a specific gradient-variance-reduction/control-variate construction described in the paper's appendix, but it's non-obvious from the code alone why both the exact and the sampled `-log q(v)` terms are summed rather than one or the other. Worth a quick sanity check against the derivation to make sure this isn't double-counting the entropy contribution.

---

## Strengths worth calling out

- Custom `SobolevGrad`/`SGLD` `autograd.Function`s implementing forward-smoothing/identity-backward tricks are a clean, idiomatic way to get Sobolev-preconditioned gradients without hand-rolling backprop.
- Antithetic sampling (`sample_q_v(..., no_samples=2)`, `LowRankMultivariateNormalDistribution.forward(no_samples=2)`) is applied consistently everywhere a variance-reduced gradient estimate is needed.
- `DistributedDataParallel` integration, including per-rank metric reduction (`dist.reduce`/`dist.all_gather`) and rank-0-only checkpointing/logging, is done carefully and consistently across `BaseTrainer`/`Trainer`.
- Per-image-pair variational parameters (`mu`, `log_var`, `u`) are persisted to disk and resumed correctly, including an epoch-indexed backup/restore mechanism (`copy_var_params_to_backup_dirs`/`copy_var_params_from_backup_dirs`) so a crash mid-run doesn't corrupt in-flight state.
- Core differential-geometry primitives (`GradientOperator`, `calc_det_J`, scaling-and-squaring integration) are unit-tested against hand-derived closed forms (`test_grad.py`), which is exactly the right place to invest test effort in a project like this.
