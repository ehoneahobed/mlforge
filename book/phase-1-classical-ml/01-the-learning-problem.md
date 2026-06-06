# Module 1.1 — The Learning Problem

**Mode:** hybrid
**Gate:** concept check
**Est. effort:** 6–10 focused hours

## Why this matters

Before any algorithm, there is a frame. Machine learning is the problem of using data to
make a function that generalizes to examples it has never seen. Everything that follows,
every model, every evaluation, every production failure, is a consequence of how well you
understand that one sentence and its implications.

The central tension of the whole field appears here. A model that is too simple cannot
capture the pattern (it is biased). A model that is too flexible captures noise as if it
were signal and fails on new data (it has high variance, it overfits). Managing this
trade-off, and measuring it honestly with data the model has not seen, is the core
discipline of machine learning. Get this module right and the rest of Phase 1 is detail.
Get it wrong and you will build models that look brilliant in a notebook and collapse in
the world.

## What you will be able to do

By the end of this module you will be able to:

- Frame a problem as supervised or unsupervised learning, and as regression or
  classification.
- Explain overfitting and underfitting in terms of the bias-variance decomposition.
- Set up a correct train/validation/test split and explain the role of each part.
- Run and interpret cross-validation, and say why it exists.
- Recognize when a reported result is too good to be true.

## Prerequisites

- Phase 0 complete, or equivalent comfort with Python and the scientific stack.
- The probability and statistics entries in
  [Module 0.4](../phase-0-foundations/04-math-just-in-time.md): expectation, variance, and
  the idea of a data-generating distribution.

## Curated path

1. **Google Machine Learning Crash Course, the framing and generalization sections.** A
   clear, modern, and free introduction to the vocabulary: features and labels, training
   and inference, generalization, overfitting, and the train/test split. Work the
   "Framing," "Generalization," and "Training and Test Sets" units.
   https://developers.google.com/machine-learning/crash-course

2. **An Introduction to Statistical Learning (ISLP), Chapter 2.** The conceptual heart of
   classical ML, written to be readable. Chapter 2, *Statistical Learning*, covers what it
   means to estimate a function, the bias-variance trade-off, and assessing model accuracy.
   This is the single best written treatment of this module's ideas. The book is free.
   https://www.statlearning.com/ (the free PDF is linked from that page)

3. **StatQuest, the bias-variance and cross-validation videos.** Josh Starmer's explanations
   make these concepts stick visually. Watch "Machine Learning Fundamentals: Bias and
   Variance" and "Cross Validation."
   https://statquest.org/video-index/

**Deliberately skip for now:** the ISLP labs in this chapter (you will do hands-on work in
later modules) and any deep dive into specific models, which are coming. Stay on the
concepts.

## Background: the ideas in words

**Supervised learning** uses labeled examples to learn a mapping from inputs to outputs:
regression when the output is a number, classification when it is a category. **Unsupervised
learning** finds structure in unlabeled data, such as clusters or a lower-dimensional
representation.

The reason machine learning is hard is **generalization**. You can always fit the training
data perfectly by memorizing it, but that teaches you nothing about new data. The quantity
that matters is performance on examples drawn from the same distribution that the model has
never seen.

The **bias-variance trade-off** explains why. Expected error on new data decomposes into
three parts: bias (error from the model being too simple to capture the truth), variance
(error from the model being so flexible that it changes wildly with the particular training
sample), and irreducible noise. Simple models have high bias and low variance; flexible
models have low bias and high variance. The art is finding the middle.

Because you cannot measure generalization on data you trained on, you **hold data out**. A
training set fits the model, a validation set tunes choices like model complexity, and a
test set, touched exactly once at the very end, estimates real-world performance.
**Cross-validation** makes this efficient: rotate which slice of data is held out so every
example serves as both training and validation across folds, giving a more stable estimate
than a single split.

## Knowledge check

Answer without notes.

1. What does it mean for a model to generalize, and why can you not measure generalization
   on the training set?
2. Decompose expected prediction error into its three parts and describe what each one is.
3. A model gets 99% accuracy on training data and 70% on validation data. What is happening,
   and what are two ways to address it?
4. What is each of the training, validation, and test sets for, and why must the test set be
   used only once?
5. What problem does k-fold cross-validation solve that a single train/validation split does
   not?
6. You read that a model achieves near-perfect accuracy on a hard real-world problem. Name
   three things you would check before believing it.

## Project

A short analytical project rather than a build:

Take one real dataset and write a two-to-three page "learning problem brief" before training
anything serious. Identify whether it is supervised or unsupervised, regression or
classification; propose a sensible target and features; describe how you would split the
data and why; name the metric you would optimize and justify it; and list the specific ways
this dataset could leak or mislead. Then fit one deliberately simple model and one
deliberately complex model, and use the gap between their training and validation scores to
illustrate the bias-variance trade-off concretely on your data.

**Definition of done:** the brief, plus a single plot or table showing the
training-versus-validation gap for your two models, with a paragraph interpreting it.

## The workshop: log it

This module is about understanding, not a portfolio build, so it goes in your
`modelwright-learning-log` repository (from Module 0.1), not a new one.

1. Add a folder `phase-1/01-learning-problem/` containing your brief (`brief.md`) and a small
   notebook (`bias-variance.ipynb`) that fits the simple and the complex model and plots the
   train-versus-validation gap.
2. Commit and push:

```bash
git add -A && git commit -m "Module 1.1: the learning problem" && git push
```

**Done when:** the brief and the bias-variance plot are visible in `modelwright-learning-log`.

## Going deeper (optional)

- ISLP Chapter 5, *Resampling Methods*, for cross-validation and the bootstrap in depth.
- The Elements of Statistical Learning (ESL), Chapter 7, *Model Assessment and Selection*,
  for the rigorous treatment of everything in this module.
  https://hastie.su.domains/ElemStatLearn/

## Canonical references

- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (ISLP),
  Chapters 2 and 5.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Chapter 7.
