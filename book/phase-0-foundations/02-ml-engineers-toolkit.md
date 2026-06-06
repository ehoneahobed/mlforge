# Module 0.2 — The ML Engineer's Toolkit & Your First Model

**Mode:** curated breadth
**Gate:** ship gate
**Est. effort:** 5–8 focused hours

> New to all this? This module is also where you train and track your very first machine
> learning model, with a few lines of a library, before you ever build the internals by
> hand. The goal is a quick, motivating win and a clean workspace, so the harder modules
> later have somewhere solid to stand.

## Why this matters

The difference between a research notebook and an engineered ML project lives almost
entirely in this module. Code that "works on my machine" is close to worthless on a team;
reproducibility, experiment tracking, and disciplined tooling are what let others trust a
result, rerun it months later, and debug it under pressure.

Machine learning adds two problems ordinary software does not have. First, experiments are
stochastic and numerous: you will run hundreds of training runs and need to know exactly
which configuration produced which number. Second, environments are heavier and more
fragile, with GPU drivers, pinned framework versions, and large datasets all able to break
reproducibility. This module builds the habits that tame both, and leaves you with a project
template you will reuse for every project that follows.

A note on how this module teaches: these are tools, not concepts, so it gives you the
essentials inline, the small subset you will actually use and how to use it, rather than
sending you to read entire documentation sites. The official docs are linked at the end of
each part for when you need the rest.

## What you will be able to do

By the end of this module you will be able to:

- Create a reproducible, locked Python environment for an ML project with `uv`.
- Track an experiment's configuration, metrics, and artifacts so any run can be found and
  rerun.
- Use the scientific Python stack idiomatically rather than fighting it.
- Profile numerical code to find the real bottleneck before optimizing.
- Train and track your first machine learning model with a library, end to end.
- Assemble all of this into a reusable project skeleton.

## Prerequisites

- [Module 0.1](./01-landscape-of-ml.md) complete, so you know what a model and the training
  workflow are.
- Working knowledge of Git and the command line.

## Your toolkit: what these tools are and how they fit together

Before the details, here is the map, so each tool below has a place to land and you know why
you are learning it. A machine learning project is a loop, and each tool owns one part of it:

- **An environment manager: `uv`.** Every project needs a specific version of Python and
  specific versions of its libraries, kept separate so one project does not break another.
  `uv` creates and locks that environment. You will run almost everything through it. Without
  something like it, "it worked yesterday" becomes a daily mystery.
- **The scientific stack: NumPy, pandas, Matplotlib.** The everyday tools for holding and
  manipulating data (pandas), doing fast math on arrays of numbers (NumPy), and drawing plots
  (Matplotlib). You will touch these in essentially every module from here on.
- **A modeling library: scikit-learn.** Ready-made machine learning models you can train in a
  few lines. This is how you experience machine learning working before you build the
  internals yourself in later phases.
- **An experiment tracker: Weights & Biases (or MLflow).** Records which settings produced
  which results across the many training runs you will do, so you compare runs in a dashboard
  instead of trusting your memory.
- **Version control: git and GitHub.** Saves the history of your work and publishes it. Every
  project you build ships here, and by the end of the curriculum this is your portfolio.
- **A profiler, when you need it.** Finds the genuinely slow part of your code so you do not
  waste time optimizing the wrong thing.

Here is how they work together, in one sentence: you create an isolated environment with
`uv`, load and explore data with the scientific stack, train a model with scikit-learn,
record the run with your experiment tracker, and version and publish the whole thing to
GitHub. That loop, in some form, repeats in every project for the rest of the curriculum. The
rest of this module teaches the pieces you have not already met, starting with the one you
always set up first: `uv`.

## uv: the essentials

`uv` is the current default for Python project and environment management: one fast tool
that handles your Python version, your virtual environment, your dependencies, and a
lockfile, replacing the old juggle of `pip` plus `venv` plus `requirements.txt`.

**The mental model.** `pyproject.toml` declares what you want (your dependencies, loosely).
`uv.lock` records the exact versions uv resolved, so the environment can be reproduced
precisely on any machine. uv manages a hidden `.venv` for you, and you rarely activate it by
hand because `uv run` does that automatically.

**The commands you will actually use:**

```bash
# Install uv (macOS / Linux). Windows and other options are in the docs.
curl -LsSf https://astral.sh/uv/install.sh | sh

# Start a project (creates pyproject.toml, .python-version, a README, and main.py)
uv init my-ml-project
cd my-ml-project

# Optionally pin a Python version for the project
uv python install 3.12

# Add dependencies: updates pyproject.toml AND uv.lock, and installs into .venv
uv add numpy pandas scikit-learn matplotlib

# Run code in the project environment (uv syncs it first, automatically)
uv run python train.py
uv run jupyter lab

# Reproduce the exact environment on another machine
uv sync            # installs precisely what is pinned in uv.lock

# Remove or upgrade a package
uv remove matplotlib
uv lock --upgrade-package scikit-learn
```

**The one rule that matters:** commit `pyproject.toml` and `uv.lock`, and never commit
`.venv/`. That lockfile is what turns "works on my machine" into "works on every machine."

That is essentially all the uv you need for this curriculum. For the rest (workspaces,
publishing, Docker integration), the official guide is the reference:
https://docs.astral.sh/uv/

## Reproducibility discipline

Beyond the lockfile, three habits make a result trustworthy:

- **Set and record random seeds.** ML is stochastic; without a fixed seed you cannot
  reproduce your own numbers. Set the seed for Python, NumPy, and your framework, and record
  it with the run.
- **Keep configuration out of code.** Put hyperparameters and paths in a config block or
  file, not scattered as literals, so a run is defined by its config. The
  [Twelve-Factor App](https://12factor.net/config) principle (config lives in the
  environment) is the idea in one page.
- **Never commit data or secrets.** Keep them out of Git; reference them by path or pull
  them in a documented step.

## Experiment tracking: the essentials

Once you run more than a handful of experiments, you cannot remember which settings produced
which result. An experiment tracker records each run's configuration and its metrics over
time, so you can compare runs in a dashboard instead of in your memory.

**Weights & Biases**, the three calls that do almost everything:

```python
import wandb

# 1. Start a run and record its hyperparameters
wandb.init(project="my-ml-project", config={"lr": 0.01, "epochs": 20})

for epoch in range(20):
    # ... train for one epoch, compute loss and accuracy ...
    # 2. Log metrics; each becomes a chart you can compare across runs
    wandb.log({"loss": loss, "val_accuracy": acc})

# 3. Close the run
wandb.finish()
```

Run `wandb login` once in your terminal first (a free account is enough). Everything you
pass to `config` is saved as that run's settings; everything you `log` becomes a time series
you can chart and compare. That is the whole core loop.

**MLflow** is the self-hostable alternative if you would rather not use a hosted service.
The equivalent calls are `mlflow.start_run()`, `mlflow.log_params({...})`, and
`mlflow.log_metric("loss", value, step=epoch)`. Pick one tool and use it consistently.

Deeper references: https://docs.wandb.ai/quickstart and
https://mlflow.org/docs/latest/tracking.html

## The scientific stack, used well

You will look up the pandas and NumPy APIs constantly, and that is fine. What pays off is
internalizing a few idioms rather than memorizing functions:

- **Vectorize instead of looping.** Operate on whole arrays and columns at once; an explicit
  Python `for` loop over rows is usually a sign something can be done faster and cleaner.
- **Know views versus copies.** Slicing a NumPy array or pandas frame can give you a view
  that shares memory with the original, so writing to it changes both. When in doubt, call
  `.copy()`.
- **Plot to answer a question.** A chart should make one thing clear. Reach for histograms
  (distributions), scatter plots (relationships), and bar charts (comparisons).

References for when you need them: the
[NumPy basics](https://numpy.org/doc/stable/user/absolute_beginners.html),
[pandas getting started](https://pandas.pydata.org/docs/getting_started/index.html), and
[Matplotlib's lifecycle of a plot](https://matplotlib.org/stable/tutorials/lifecycle.html).

## Your first model, with scikit-learn

Before any from-scratch work later, see machine learning actually run. The scikit-learn API
is the same for almost every model: create it, `fit` it on training data, `predict` or
`score` on test data.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)              # learn from training data
print(model.score(X_test, y_test))       # accuracy on unseen data
```

That is a complete, if tiny, machine learning project: data in, a trained model, a score on
data it never saw. Work the official tutorial to understand each step, then make it yours in
the project below.
https://scikit-learn.org/stable/getting_started.html

## Profiling and debugging numerical code

When something is slow, measure before you optimize. Find the slow line with `cProfile` or,
in a notebook, `%timeit` on a suspect cell, or a line profiler. And when results look wrong,
sanity-check your arrays first: print their `.shape` and `.dtype`, and check for `NaN` values
with `np.isnan(x).any()`. Most "the model is broken" bugs are actually a wrong shape or a
silent `NaN`. Reference: https://docs.python.org/3/library/profile.html

**Deliberately skip for now:** Docker, Kubernetes, continuous integration pipelines, and
cloud training. They are real and important, but they belong to Phase 4, where they have
context. Adopting them here would be assembling infrastructure before there is anything to
run on it.

## Knowledge check

1. What does `uv.lock` guarantee that an unpinned dependency list does not, and which files
   should you commit to Git?
2. What does `uv run` do for you that saves you from activating a virtual environment by
   hand?
3. Besides the source code, name three things you must record to make a training run
   reproducible.
4. In Weights & Biases, what is the difference between what you pass to `config` and what you
   pass to `log`?
5. What is the difference between a NumPy view and a copy, and how can it cause a silent bug?
6. You profile a training script and one function dominates the runtime. What should you do
   before trying to optimize it?

## Project (ship gate)

**Train a simple model, and wrap it in a reusable, professional project template.**

This is your first end-to-end machine learning project. The model itself is deliberately
simple, because the point of this module is the engineering around it, the template you will
reuse for every project in the curriculum, including the autograd engine later.

- **Train a model.** Using scikit-learn, train a model on a small built-in dataset (iris,
  or the diabetes regression set). Split the data, fit the model, and score it on the
  held-out test set.
- **Wrap it properly.** Put it in a `uv`-managed project with a committed `uv.lock` and a
  README a stranger can follow from clone, to environment creation, to running the code and
  reproducing your result. Keep a clean layout: a source module, a test or two, a config
  block, and random seeds that are set and recorded.
- **Track the run.** Instrument it to log the hyperparameters and the score to Weights &
  Biases or MLflow, and include a screenshot or link in the README.
- **Capture the template.** Write a short `TEMPLATE.md` describing the structure, so you can
  copy this skeleton for every future project.

**Definition of done:** a stranger could clone the repository, build the environment from
the lockfile, run it, reproduce your model's score, and view the tracked experiment. That
bar, met once here, becomes your default for every project that follows.

## The workshop: build it and ship it

Follow these steps in order, in your terminal, building as you go. Do not read ahead and
skip; the point is to do it. By the end you will have a real, reproducible project on GitHub
that becomes the template for everything you build later. This project lives in its own
repository, `mlforge-first-model`.

**1. Create the project.** `uv init` sets up a project folder, a `pyproject.toml`, a
`.gitignore`, and a git repository all at once:

```bash
mkdir mlforge-first-model && cd mlforge-first-model
uv init
uv add scikit-learn pandas matplotlib wandb
mkdir src
```

**2. Write the training script.** Create `src/train.py` from this scaffold and fill in the
`TODO`s yourself. The structure is the lesson; the substance is your work:

```python
import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import wandb

SEED = 0
CONFIG = {"n_estimators": 100, "max_depth": None, "test_size": 0.2}

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

def main():
    set_seeds(SEED)
    wandb.init(project="mlforge-first-model", config=CONFIG)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=SEED)

    # TODO: create the model using the values in CONFIG
    model = RandomForestClassifier(...)
    model.fit(X_train, y_train)

    # TODO: score the model on the test set and log it
    accuracy = ...
    wandb.log({"test_accuracy": accuracy})
    print(f"Test accuracy: {accuracy:.3f}")
    wandb.finish()

if __name__ == "__main__":
    main()
```

**3. Checkpoint one, get it running and commit.** Log in to Weights & Biases once
(`wandb login`), then run and commit:

```bash
wandb login          # one-time, free account
uv run python src/train.py    # should print a test accuracy and create a W&B run
git add -A && git commit -m "Train and score a first model, tracked with W&B"
```

**4. Make it reproducible and documented.** Write a `README.md` that a stranger could
follow: what the project does, and the exact commands to recreate the environment
(`uv sync`) and run it (`uv run python src/train.py`). Add a screenshot or link to your
W&B run. Then commit:

```bash
git add -A && git commit -m "Add README with setup and results"
```

**5. Capture the template.** Write a short `TEMPLATE.md` describing the structure (where
code goes, where config lives, how seeds and tracking are handled), so you can copy this
skeleton for future projects. Commit it.

```bash
git add -A && git commit -m "Add reusable project template notes"
```

**6. Ship it to GitHub:**

```bash
gh repo create mlforge-first-model --public --source=. --push
```

No `gh`? Create an empty public repo named `mlforge-first-model` on github.com, then:
`git remote add origin <its-url>` and `git push -u origin main`.

**Done when:** `mlforge-first-model` is on your GitHub, and a stranger could clone it, run
`uv sync`, run the script, reproduce your accuracy, and find your tracked run. You now own a
template you will reuse for the rest of the curriculum.

## Going deeper (optional)

- Add a `Makefile` or `justfile` with `setup`, `test`, and `train` targets so common tasks
  are a single command.
- Read about `pre-commit` hooks for automatic formatting and linting, and add one.
- Skim Data Version Control (DVC) to see how data and model versioning works; it returns
  properly in Phase 4.

## Canonical references

- Astral, *uv documentation*.
- *The Twelve-Factor App*, factor III (Config).
- Weights & Biases and MLflow documentation.
- scikit-learn, *An introduction to machine learning with scikit-learn*.
