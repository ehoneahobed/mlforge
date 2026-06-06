# Module 1.6 — Evaluation, Calibration & Leakage

**Mode:** curated breadth
**Gate:** ship gate
**Est. effort:** 8–12 focused hours

## Why this matters

This is the module that separates people who can run a model from people who can be trusted
with one. Anyone can call `fit` and read off an accuracy number. Knowing whether that number
means anything is the actual skill, and it is the most common point of failure in real
machine learning. Models that look excellent in a notebook routinely collapse in production
for one of two reasons: the metric did not reflect what mattered, or information leaked from
the future or from the test set into training, inflating the score.

Three competencies live here. **Choosing the right metric**, because accuracy is misleading
the moment classes are imbalanced or the costs of different errors differ. **Calibration**,
because a model that says "80% probability" should be right about 80% of the time when you
need to act on the probability itself, not just the ranking. And **leakage detection**, the
detective work of finding the subtle ways your evaluation is lying to you. Of the three,
leakage is the most dangerous precisely because it makes results look better, so nobody is
motivated to go looking for it.

## What you will be able to do

By the end of this module you will be able to:

- Choose and justify an evaluation metric appropriate to the problem and its error costs.
- Read and explain ROC and precision-recall curves, and know when each is the right lens.
- Check whether a model's probabilities are calibrated and fix them when they are not.
- Identify and prevent the common forms of data leakage.
- Build an evaluation protocol you can defend to a skeptical reviewer.

## Prerequisites

- [Module 1.1](./01-the-learning-problem.md) (the train/validation/test discipline) and at
  least one modeling module (1.2 or 1.3) so you have models to evaluate.

## Curated path

1. **Google Machine Learning Crash Course, the classification and metrics units.** Clear
   coverage of accuracy, precision, recall, the precision-recall trade-off, ROC and AUC,
   and prediction bias. Free.
   https://developers.google.com/machine-learning/crash-course/classification

2. **scikit-learn user guide, model evaluation and calibration.** The metrics section is a
   reference for the full menu of scores and when to use each; the probability calibration
   section explains reliability diagrams, Platt scaling, and isotonic regression. Read both.
   https://scikit-learn.org/stable/modules/model_evaluation.html and
   https://scikit-learn.org/stable/modules/calibration.html

3. **StatQuest, the ROC and confusion-matrix videos,** for visual intuition on sensitivity,
   specificity, ROC, and AUC.
   https://statquest.org/video-index/

4. **A serious treatment of data leakage.** Read the Kaggle Learn "Data Leakage" lesson for
   concrete examples (target leakage and train-test contamination), then keep the failure
   patterns in mind as a permanent checklist.
   https://www.kaggle.com/code/alexisbcook/data-leakage

**Deliberately skip for now:** decision-theoretic threshold optimization with full cost
matrices, which is valuable but specialized. Focus on metric choice, calibration, and
leakage.

## Background: the ideas in words

**Metric choice.** Accuracy answers "what fraction did I get right," which is meaningless
when 99% of examples are one class, because predicting that class blindly scores 99%.
Precision asks, of the items you flagged, how many were correct; recall asks, of the items
that should have been flagged, how many you caught. The two trade off as you move the
decision threshold. The **ROC curve** plots this trade-off and its area (AUC) summarizes
ranking quality across all thresholds, but on heavily imbalanced data the
**precision-recall curve** is the more honest picture. The right metric is the one aligned
with the real-world cost of each kind of error.

**Calibration.** Ranking and probability are different goals. A model can rank cases
perfectly (high AUC) yet output probabilities that are systematically too confident. If you
will act on the probability, by setting a price, sizing a risk, or triaging, it must be
calibrated: among cases assigned 70%, about 70% should be positive. A **reliability diagram**
checks this, and methods like Platt scaling (a logistic fit) or isotonic regression correct
it. Tree ensembles and SVMs are often poorly calibrated out of the box.

**Leakage.** Leakage is any way information reaches the model that it would not have at
prediction time, making evaluation optimistic. Classic forms: a feature that is a proxy for
the target (a "days until churn" column when predicting churn), preprocessing such as scaling
or imputation fit on the whole dataset before splitting (so test statistics leak into
training), and time series where future data informs predictions about the past. The
discipline is to fit every transformation inside cross-validation on training data only, to
respect time order when time matters, and to ask of every strong feature, "would I really
have this value at prediction time?"

## Knowledge check

1. Why can accuracy be deeply misleading, and give a concrete example where a high-accuracy
   model is useless?
2. Define precision and recall, and describe a situation where you would prioritize one over
   the other.
3. When is a precision-recall curve more informative than an ROC curve, and why?
4. What does it mean for a model to be well calibrated, and how do you check it?
5. Give three distinct examples of data leakage and how each inflates the apparent score.
6. Why must preprocessing steps be fit inside cross-validation rather than on the full
   dataset beforehand?

## Project (ship gate)

Build a reusable, defensible evaluation harness, and use it to expose a problem.

- Implement an evaluation module that, given a model and data, produces the full picture:
  the appropriate metrics for the problem, an ROC and a precision-recall curve, a confusion
  matrix at a justified threshold, and a reliability diagram with a calibration step applied
  if needed. Build it on a leak-free cross-validation protocol with a single held-out test
  evaluation at the end.
- **Demonstrate leakage on purpose.** Construct or find a case where a naive pipeline (for
  example, scaling or target-encoding before the split, or including a leaky feature) yields
  an inflated score, then show the corrected pipeline and the honest, lower score. Explain
  exactly what leaked and how you caught it.
- Write a report that an experienced reviewer would accept: the metric choice and its
  justification, the calibration result, and the leakage demonstration with before-and-after
  numbers.

**Definition of done:** the reusable evaluation harness (packaged and tested), the leakage
demonstration with corrected results, and the defensible report. This harness becomes part
of your standard toolkit for the phase capstone and beyond.

## The workshop: ship it

Build this in its own repository, `modelwright-evaluation-harness`. This is a tool you will reuse
in later modules, so make it clean.

1. Set up the project:

```bash
mkdir modelwright-evaluation-harness && cd modelwright-evaluation-harness
uv init && uv add scikit-learn numpy pandas matplotlib && mkdir src tests notebooks
```

2. Implement the evaluation harness in `src/` (metrics, ROC and PR curves, confusion matrix,
   reliability diagram and calibration), then build the leakage demonstration in a notebook.
3. Commit at checkpoints: "Evaluation harness with tests", then "Leakage demonstration,
   before and after".
4. Add a README and your defensible report (metric choice, calibration result, leakage with
   before-and-after numbers).
5. Ship it:

```bash
gh repo create modelwright-evaluation-harness --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `modelwright-evaluation-harness` is on GitHub, the tests pass, and the README
documents your leakage demonstration. Reuse this harness in the phase capstone.

## Going deeper (optional)

- Nested cross-validation, for unbiased performance estimation when you also tune
  hyperparameters.
- Proper scoring rules (the Brier score, log loss) as single numbers that reward
  calibration.
- Time-series cross-validation and the special leakage risks of temporal data.

## Canonical references

- scikit-learn documentation, *Metrics and scoring* and *Probability calibration*.
- Kaggle Learn, *Data Leakage*.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Chapter 7.
