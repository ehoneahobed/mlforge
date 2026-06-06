# Module 0.2 — The ML Engineer's Toolkit

**Mode:** curated breadth
**Gate:** ship gate
**Est. effort:** 5–8 focused hours

## Why this matters

The difference between a research notebook and an engineered ML project lives almost
entirely in this module. Code that "works on my machine" is close to worthless on a team;
reproducibility, experiment tracking, and disciplined tooling are what let others trust a
result, rerun it months later, and debug it under pressure.

Machine learning adds two problems ordinary software does not have. First, experiments are
stochastic and numerous: you will run hundreds of training runs and need to know exactly
which configuration produced which number. Second, environments are heavier and more
fragile, with GPU drivers, pinned framework versions, and large datasets all able to break
reproducibility. This module builds the habits that tame both, and leaves you with a
project template you will reuse for every project that follows.

## What you will be able to do

By the end of this module you will be able to:

- Create a reproducible, locked Python environment for an ML project.
- Track an experiment's configuration, metrics, and artifacts so any run can be found and
  rerun.
- Use the scientific Python stack idiomatically rather than fighting it.
- Profile numerical code to find the real bottleneck before optimizing.
- Assemble all of this into a reusable project skeleton.

## Prerequisites

- [Module 0.1](./01-autograd-from-numpy.md) complete, so you have a small project to
  package.
- Working knowledge of Git and the command line.

## Curated path

Work these in order. Keep it pragmatic: set each tool up, use it on your Module 0.1 engine,
and move on.

1. **Environments and dependencies with `uv`.** `uv` is the current default for Python
   project and environment management: a single fast tool for virtual environments,
   dependency resolution, lockfiles, and Python versions. Read the official guide and
   create a locked environment for your Module 0.1 project.
   https://docs.astral.sh/uv/
   *When to reach for conda instead:* when you need GPU or CUDA binaries, or non-Python
   system libraries that conda resolves as a unit. For pure-Python ML work, `uv` is faster
   and simpler. Note that `conda` and `mamba` exist for that case and move on; do not
   rabbit-hole here.

2. **Reproducibility discipline.** The non-negotiables: pin every dependency with a
   lockfile, set and record random seeds, keep configuration separate from code, and never
   commit data or secrets. Read the *Twelve-Factor App* principle on configuration (config
   belongs in the environment) and apply it.
   https://12factor.net/config

3. **Experiment tracking.** Choose one of Weights & Biases (hosted, batteries included) or
   MLflow (open-source, self-hostable) and instrument a training run to log
   hyperparameters, metrics over time, and the final artifact. This single habit is the
   difference between "I think the bigger model was better" and "here is the run, the
   config, and the curve."
   - Weights & Biases quickstart: https://docs.wandb.ai/quickstart
   - MLflow tracking: https://mlflow.org/docs/latest/tracking.html
   Choose Weights & Biases for zero-setup dashboards, or MLflow if you prefer self-hosting
   and an integrated model registry (which returns in Phase 4).

4. **The scientific stack, used well.** The goal is not to memorize the pandas API, which
   you can look up, but to internalize the idioms: vectorize instead of looping, understand
   the difference between a view and a copy, and make plots that answer a question. Skim
   these as references rather than reading cover to cover.
   - NumPy basics: https://numpy.org/doc/stable/user/absolute_beginners.html
   - pandas getting started: https://pandas.pydata.org/docs/getting_started/index.html
   - Matplotlib, the lifecycle of a plot: https://matplotlib.org/stable/tutorials/lifecycle.html

5. **Profiling and debugging numerical code.** Learn to find the slow line with `cProfile`,
   `%timeit`, or a line profiler *before* optimizing, and to sanity-check tensors for
   shape, dtype, and `NaN` values. This habit saves entire days later.
   https://docs.python.org/3/library/profile.html

**Deliberately skip for now:** Docker, Kubernetes, continuous integration pipelines, and
cloud training. They are real and important, but they belong to Phase 4, where they have
context. Adopting them here would be assembling infrastructure before there is anything to
run on it.

## Knowledge check

1. What does a lockfile guarantee that an unpinned `requirements.txt` does not, and why does
   that matter for reproducing a result months later?
2. Besides the source code, name three things you must record to make a training run
   reproducible.
3. Why is logging metrics to an experiment tracker better than printing them, once you have
   more than a handful of runs?
4. What is the difference between a NumPy view and a copy, and how can that distinction
   cause a silent bug?
5. You profile a training script and one function dominates the runtime. What should you do
   before trying to optimize it?

## Project

**Build a reusable project template, and apply it to your Module 0.1 engine.**

Turn the autograd engine from Module 0.1 into a properly engineered project:

- A `uv`-managed environment with a committed lockfile and a README a stranger can follow
  from clone, to environment creation, to passing tests, to a reproduced training run.
- A clean repository layout: a source package, tests, a configuration block or directory,
  and random seeds that are set and recorded.
- One experiment tracked end to end: instrument the training loop to log hyperparameters
  and the loss curve to Weights & Biases or MLflow, and include a screenshot or link in the
  README.
- A short `TEMPLATE.md` describing the structure, so you can copy this skeleton for future
  projects.

**Definition of done:** a stranger could clone the repository, build the environment from
the lockfile, run the tests green, reproduce the training run, and view the tracked
experiment. That bar, met once here, becomes your default for every project that follows.

## Going deeper (optional)

- Add a `Makefile` or `justfile` with `setup`, `test`, and `train` targets so common tasks
  are a single command.
- Read about `pre-commit` hooks for automatic formatting and linting, and add one.
- Skim Data Version Control (DVC) to see how data and model versioning works; it returns
  properly in Phase 4.

## Canonical references

- Astral. *uv documentation.*
- *The Twelve-Factor App*, factor III (Config).
- Weights & Biases and MLflow documentation.
