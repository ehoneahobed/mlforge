# Module 1.3 — Trees, Forests, and Gradient Boosting

**Mode:** hybrid
**Gate:** build gate
**Est. effort:** 12–18 focused hours

## Why this matters

On tabular data, the kind of structured rows-and-columns data that runs most of the world's
business decisions, gradient-boosted decision trees are usually the model to beat. They win
a large share of Kaggle competitions on tabular problems, they train fast, they handle
mixed feature types and missing values gracefully, and they need far less tuning than neural
networks to get a strong result. If you can build only one family of models well, this is
the one that pays the most rent.

The path to understanding them runs through a single idea: the decision tree. A tree is
simple to the point of crudeness, splitting the data one feature at a time. Its power comes
from how you combine many of them. Random forests average many decorrelated trees to crush
variance. Gradient boosting builds trees in sequence, each one correcting the errors of the
last, to crush bias. Understanding that contrast, averaging independent models versus
sequentially correcting one, is understanding the two great strategies of ensemble learning.

## What you will be able to do

By the end of this module you will be able to:

- Implement a decision tree from scratch, including the splitting criterion.
- Explain how random forests reduce variance through bagging and feature randomness.
- Explain how gradient boosting reduces bias by fitting successive trees to residuals.
- Use and tune XGBoost or LightGBM competently, and read what the model learned.

## Prerequisites

- [Module 1.1](./01-the-learning-problem.md) and
  [Module 1.2](./02-linear-logistic-regression.md) complete. Boosting in particular builds
  on the idea of a loss function and its gradient.

## Curated path

1. **StatQuest, the tree and ensemble series.** Josh Starmer's sequence here is exceptional:
   "Decision Trees," "Decision Trees, Part 2," "Random Forests Part 1 and 2," and the
   "Gradient Boost" series (parts 1 to 4, regression and classification), plus "XGBoost." Watch
   them in that order. This is the clearest available explanation of how boosting actually
   works, step by step.
   https://statquest.org/video-index/

2. **ISLP, Chapter 8 (*Tree-Based Methods*).** The written treatment: trees, bagging, random
   forests, and boosting, with the reasoning for why each step helps. Free.
   https://www.statlearning.com/

3. **The XGBoost documentation, "Introduction to Boosted Trees."** The canonical explanation
   of gradient boosting as it is actually implemented, including the second-order
   (Newton-style) refinement that XGBoost adds. Read it after the StatQuest videos so the
   math has intuition behind it.
   https://xgboost.readthedocs.io/en/stable/tutorials/model.html

4. **scikit-learn user guide, ensemble methods,** for the practical API of random forests
   and gradient boosting, and the `HistGradientBoosting` estimators.
   https://scikit-learn.org/stable/modules/ensemble.html

**Deliberately skip for now:** exotic boosting variants and heavy hyperparameter-search
machinery. Get one tree right, understand the two ensembling strategies, and learn to tune
the handful of parameters that matter.

## Background: the ideas in words

A **decision tree** repeatedly splits the data on the feature and threshold that best
separate the target, measured by a criterion such as Gini impurity or entropy for
classification, or variance reduction for regression. Splitting continues until a stopping
rule applies. A single deep tree has low bias but high variance: it overfits badly, because
small changes in the data produce very different trees.

**Bagging and random forests** attack that variance. Train many trees, each on a bootstrap
resample of the data and each allowed to consider only a random subset of features at every
split. The trees become decorrelated, and averaging decorrelated high-variance models
sharply reduces variance while keeping bias low. This is why a random forest is a strong,
low-effort baseline.

**Gradient boosting** takes the opposite strategy. Start with a weak model, then repeatedly
fit a new small tree to the part of the error the current ensemble still gets wrong (the
gradient of the loss, which for squared error is just the residual), and add it in with a
small learning rate. Each tree nudges the ensemble toward lower bias. Because the trees are
added in sequence, boosting is more prone to overfitting than bagging and needs the learning
rate, tree depth, and number of trees to be balanced. XGBoost and LightGBM are highly
optimized implementations that add regularization and a second-order view of the loss.

## Knowledge check

1. How does a decision tree decide where to split, and what criterion does it use for
   classification versus regression?
2. Why does a single deep decision tree tend to overfit?
3. Random forests reduce variance. By what two mechanisms do they decorrelate the trees, and
   why does decorrelation matter for averaging?
4. In gradient boosting, what exactly is each new tree fit to? Why does this reduce bias?
5. Contrast bagging and boosting in one or two sentences: what does each primarily reduce,
   and why is boosting more prone to overfitting?
6. Name the handful of XGBoost or LightGBM hyperparameters that matter most, and say what
   each one trades off.

## Build gate

Implement a decision tree from scratch, then a minimal gradient-boosted ensemble on top of
it.

**Specification:**

- A `DecisionTree` for classification (or regression) built by recursive splitting:
  choose the best split by impurity reduction, respect a maximum depth and a minimum samples
  per leaf, and predict by majority class (or mean) at the leaf. No tree library.
- A minimal **gradient-boosting** regressor that fits a sequence of shallow trees to the
  residuals, combined with a learning rate. Start with squared-error loss, where the
  "gradient" each tree fits is simply the residual, so the mechanism is transparent.

**Tests it must pass:**

- **Your tree matches scikit-learn's** `DecisionTreeClassifier` closely on a clean dataset
  with the same depth and criterion (small differences from tie-breaking are acceptable;
  understand where they come from).
- **Boosting improves with rounds.** Show validation error decreasing as boosting rounds
  increase, then eventually overfitting, demonstrating you understand the learning-rate and
  number-of-trees trade-off.
- **Your boosted model is competitive** with scikit-learn's `GradientBoostingRegressor` on
  the same data.

## Project

A tabular modeling project that also serves as a rehearsal for the phase capstone:

- Take a real tabular dataset and build three models: a tuned random forest, a tuned
  gradient-boosting model (XGBoost or LightGBM), and your from-scratch boosting model as a
  sanity check.
- Tune the gradient-boosting model properly with cross-validation, and report feature
  importance plus one partial-dependence or SHAP plot to interpret what it learned.
- Write a short report comparing the three models on a sensible metric, with an honest
  baseline from Module 1.1, and a paragraph on what the feature importances tell you about
  the problem.

**Definition of done:** the from-scratch tree and booster (packaged and tested), the
three-model comparison on real data, and the interpretation write-up.

## The workshop: ship it

Build this in its own repository, `modelwright-trees-boosting`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir modelwright-trees-boosting && cd modelwright-trees-boosting
uv init && uv add numpy scikit-learn xgboost matplotlib && mkdir src tests
```

2. Implement the build gate in `src/` (your `DecisionTree` and minimal gradient booster) with
   tests against scikit-learn, then add the three-model comparison from the project.
3. Commit at checkpoints: "Decision tree from scratch", then "Minimal gradient boosting",
   then "Three-model comparison on real data".
4. Add a README and your short comparison report with the feature-importance plot.
5. Ship it:

```bash
gh repo create modelwright-trees-boosting --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `modelwright-trees-boosting` is on GitHub, the tests pass, and the README presents
your model comparison and what the feature importances revealed.

## Going deeper (optional)

- Read the original XGBoost paper (Chen & Guestrin, 2016) and the LightGBM paper to see the
  engineering that makes boosting fast at scale.
- Explore SHAP values for principled feature attribution.
  https://shap.readthedocs.io/
- Implement boosting for classification (log-loss), where the residual interpretation
  becomes the negative gradient of the cross-entropy.

## Canonical references

- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, Chapter 8.
- Chen, T. & Guestrin, C. (2016), *XGBoost: A Scalable Tree Boosting System*.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Chapter 10
  (Boosting) and Chapter 15 (Random Forests).
