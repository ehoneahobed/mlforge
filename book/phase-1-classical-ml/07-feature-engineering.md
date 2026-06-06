# Module 1.7 — Feature Engineering & The Data Reality

**Mode:** curated breadth
**Gate:** concept check
**Est. effort:** 6–10 focused hours

## Why this matters

Tutorials hand you clean data. Reality does not. The data you actually face has missing
values, inconsistent categories, skewed distributions, mixed types, outliers, and a class
you care about that makes up two percent of the rows. On tabular problems, the work of
turning that mess into informative features routinely matters more than the choice of model.
Two practitioners with the same gradient-boosting library and different feature work will get
very different results, and the gap is usually feature work, not hyperparameters.

This is the least glamorous module in the phase and one of the most valuable. It is curated
breadth with a concept-check gate, because the skill is built through repeated practice on
real datasets rather than a single from-scratch build. The goal is to internalize the menu
of transformations and, more importantly, the judgment for when each applies and how to apply
it without leaking information, which connects directly back to the previous module.

## What you will be able to do

By the end of this module you will be able to:

- Handle missing data thoughtfully rather than reflexively dropping or zero-filling it.
- Encode categorical variables appropriately, including high-cardinality ones, without
  leakage.
- Scale and transform numerical features when the model requires it.
- Handle class imbalance with the right tool and evaluate it honestly.
- Build all of this as a pipeline that fits on training data only.

## Prerequisites

- [Module 1.6](./06-evaluation-calibration-leakage.md), because feature engineering and
  leakage are inseparable: most leakage happens during preprocessing.
- Working comfort with pandas from [Module 0.2](../phase-0-foundations/02-ml-engineers-toolkit.md).

## Curated path

1. **Kaggle Learn, the Feature Engineering and Data Cleaning courses.** Practical,
   hands-on, and focused on real tabular problems: missing values, categorical encodings,
   scaling, and creating features that help tree and linear models. Free, with exercises.
   https://www.kaggle.com/learn/feature-engineering and
   https://www.kaggle.com/learn/data-cleaning

2. **scikit-learn user guide, preprocessing and pipelines.** The reference for encoders,
   scalers, imputers, and column transformers, and crucially how to compose them into a
   `Pipeline` so every step is fit on training data only. This is the mechanism that
   prevents the leakage from the previous module.
   https://scikit-learn.org/stable/modules/preprocessing.html and
   https://scikit-learn.org/stable/modules/compose.html

3. **A grounded treatment of class imbalance.** Read the `imbalanced-learn` documentation's
   introduction and the guidance on resampling (SMOTE and friends) versus class weights,
   along with the warning that resampling must happen inside cross-validation, never before
   the split.
   https://imbalanced-learn.org/stable/

4. **Feature engineering for categorical data: target and other encodings.** Read the
   scikit-learn `TargetEncoder` documentation, which explains the technique for
   high-cardinality categories and the cross-fitting that makes it safe from leakage.
   https://scikit-learn.org/stable/modules/preprocessing.html#target-encoder

**Deliberately skip for now:** automated feature engineering tools and heavy
domain-specific feature libraries. Build the manual judgment first; automation is only
useful once you know what good features look like.

## Background: the ideas in words

**Missing data** is information, not just absence. Sometimes a value is missing at random;
sometimes the fact that it is missing is itself predictive (an unanswered income field may
correlate with the target). Reflexively dropping rows discards data and can bias the sample;
filling with the mean distorts the distribution. Better approaches include model-based
imputation and adding an explicit "was missing" indicator so the model can use the
missingness signal. Tree-based models can often handle missing values natively.

**Categorical encoding** turns categories into numbers a model can use. One-hot encoding is
the safe default for low-cardinality categories. For high-cardinality categories (zip codes,
product IDs), one-hot explodes the feature space, and target encoding, replacing a category
with a smoothed average of the target for that category, is powerful but dangerous: done
naively it leaks the target, so it must be cross-fitted.

**Numerical transformations** matter for some models and not others. Linear models and SVMs
need features on comparable scales and often benefit from transforming skewed features (a log
transform, for instance); tree-based models are invariant to monotonic rescaling and need
neither. Knowing which model cares saves wasted effort.

**Class imbalance** distorts both training and evaluation. The fixes include class weights
(telling the model to care more about the rare class), resampling the data (oversampling the
minority with techniques like SMOTE, or undersampling the majority), and, always, evaluating
with metrics suited to imbalance from Module 1.6. The cardinal rule: any resampling happens
inside the cross-validation loop on training folds only, or you leak and fool yourself.

**The pipeline** ties it together. Every transformation, imputation, encoding, scaling,
resampling, must be fit on training data and merely applied to validation and test data. A
`Pipeline` enforces this automatically, which is why it is the single most important habit in
applied tabular machine learning.

## Knowledge check

1. Give two reasons that dropping rows with missing values can be a mistake, and a better
   alternative for each.
2. When is one-hot encoding a poor choice, and what would you use instead? What makes that
   alternative risky?
3. Which kinds of models require feature scaling and which do not, and why?
4. Why must target encoding be cross-fitted, and what leaks if it is not?
5. Name two ways to handle class imbalance and the one rule you must never break when
   resampling.
6. Why is a scikit-learn `Pipeline` the right tool for preventing preprocessing leakage?

## Project

A feature-engineering study on a genuinely messy dataset:

- Take a real dataset with missing values, categorical features (including at least one
  high-cardinality column), and ideally some class imbalance. Document its problems first.
- Build two pipelines: a naive one (drop missing rows, one-hot everything, no scaling
  thought, no imbalance handling) and a considered one (thoughtful imputation with
  missingness indicators, appropriate encodings including a safe target encoding, scaling
  where the model needs it, and imbalance handled correctly inside cross-validation).
- Compare them with the evaluation harness from Module 1.6, and write up how much feature
  work alone moved the honest metric, and which single change mattered most.

**Definition of done:** the two pipelines, a fair comparison using leak-free evaluation, and
a write-up quantifying the value of the feature work and naming the highest-impact decision.

## Going deeper (optional)

- Time-based features and the special care temporal data demands.
- Feature selection methods (recursive elimination, permutation importance) and when they
  help.
- Read about feature stores, the production systems that manage features at scale, which you
  will meet properly in Phase 4.

## Canonical references

- Kaggle Learn, *Feature Engineering* and *Data Cleaning*.
- scikit-learn documentation, *Preprocessing data* and *Pipelines and composite estimators*.
- `imbalanced-learn` documentation.
