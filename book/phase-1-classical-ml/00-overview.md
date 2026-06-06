# Phase 1 — Classical Machine Learning

Deep learning gets the attention, but classical machine learning still runs an enormous
share of the models in production. When the data is tabular, which describes most business
problems, a well-tuned gradient-boosted tree often beats a neural network, trains in
seconds, and is far easier to reason about. More importantly, this is where the statistical
core of the field lives: the ideas about generalization, bias and variance, evaluation, and
data leakage that you will carry into every later phase, neural networks included.

Each module follows the curriculum's core rhythm: you use a model first, with scikit-learn,
to build intuition for what it does, and then, for the foundational ones, you implement it
from scratch to understand how it works. You arrive having already trained models in
Phase 0, so the from-scratch builds here land on real, hands-on experience rather than on a
blank page. Throughout, the emphasis is on the difference between getting a number and
getting a number you can trust. A model that scores well on a leaky evaluation is worse than
no model, because it will fail silently in production. Phase 1 teaches you to be suspicious
in the right ways.

Mathematics begins to carry weight here. Each module links back to the reference layer in
[Module 0.4](../phase-0-foundations/04-math-just-in-time.md) for the specific results it
uses. If a linked result is unfamiliar, refresh it there before continuing.

## Modules

- **1.1 — The Learning Problem** *(hybrid · concept check)*
  The conceptual frame: supervised versus unsupervised, generalization, the bias-variance
  trade-off, and the train/validation/test discipline that everything else depends on.
- **1.2 — Linear & Logistic Regression from First Principles** *(from-scratch · build gate)*
  Derive and implement the two models that underpin the entire field, with your own
  gradient descent and a study of what regularization actually does.
- **1.3 — Trees, Forests, and Gradient Boosting** *(hybrid · build gate)*
  Decision trees from scratch, then the ensembles that dominate tabular machine learning.
- **1.4 — SVMs & Kernels** *(curated breadth · concept check)*
  Margins, the kernel trick, and where this elegant idea still earns its place.
- **1.5 — Unsupervised: Clustering & Dimensionality Reduction** *(hybrid · build gate)*
  k-means and PCA from scratch, then modern visualization, and what each method distorts.
- **1.6 — Evaluation, Calibration & Leakage** *(curated breadth · ship gate)*
  The skill that separates practitioners from tutorial-followers: building an evaluation
  you can actually trust.
- **1.7 — Feature Engineering & The Data Reality** *(curated breadth · concept check)*
  Real data is messy, and feature work often beats model choice. The unglamorous craft
  that decides most projects.

## Phase capstone

**Capstone 1: an end-to-end classical machine learning project on real, messy data.**

This is the ship gate for the whole phase. Take a real tabular dataset (not a pre-cleaned
teaching set) and carry it from raw data to a defensible result. Ship it to its own
repository, `modelwright-phase-1-capstone`, the way you have shipped every project so far. The
deliverable is that repository plus a written report that an interviewer or a hiring manager
would respect.

Definition of done:

- **An honest baseline.** Start with a trivial model (majority class, or a simple linear
  model) so every later gain is measured against something real.
- **A leak-free pipeline.** All preprocessing fit on training data only, inside
  cross-validation, with a held-out test set you touch exactly once. Be able to argue why
  no information leaked from test to train.
- **A well-tuned primary model.** Gradient boosting, properly tuned, compared honestly
  against the baseline and at least one simpler model.
- **Calibrated probabilities** where the problem calls for them, with a calibration check.
- **A metric that fits the problem,** chosen and justified rather than defaulting to
  accuracy.
- **A written report:** the problem, the data and its issues, what you tried, what worked,
  what did not, and what you would do with more time. Honesty about limitations counts for
  more than a high score.

Suggested data sources: the [UCI Machine Learning Repository](https://archive.ics.uci.edu/),
a [Kaggle](https://www.kaggle.com/datasets) tabular competition (past or present), or a
public dataset from your own domain of interest. Pick something you find genuinely
interesting; you will be staring at it for a while.

Begin with [Module 1.1](./01-the-learning-problem.md).
