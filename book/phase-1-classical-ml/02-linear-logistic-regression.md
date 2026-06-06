# Module 1.2 — Linear & Logistic Regression from First Principles

**Mode:** from-scratch deep-dive
**Gate:** build gate
**Est. effort:** 10–16 focused hours

## Why this matters

Linear and logistic regression are the two most important models in machine learning, not
because they are the most powerful, but because they are the foundation everything else
builds on. A neural network is, in a real sense, a stack of linear models with
nonlinearities between them, ending in exactly the logistic (or softmax) layer you will
build here. Understanding these two models deeply, including deriving their loss functions
and the gradients that train them, means understanding the machinery of the entire field in
its simplest form.

This is a from-scratch module because the derivations are load-bearing. When you derive the
logistic loss from the principle of maximum likelihood, you are learning the pattern by
which almost every loss function in machine learning is justified. When you implement
gradient descent on it by hand, you are building the exact optimization loop that, scaled
up, trains every deep network. Do this once, carefully, and later phases become
recognition rather than mystery.

## What you will be able to do

By the end of this module you will be able to:

- Derive the least-squares loss for linear regression and the cross-entropy loss for
  logistic regression, the latter from maximum likelihood.
- Implement both models from scratch with your own gradient descent, and verify them against
  scikit-learn.
- Explain what L1 and L2 regularization do, both as a penalty and geometrically.
- Interpret model coefficients and know the assumptions that make that interpretation valid.

## Prerequisites

- [Module 1.1](./01-the-learning-problem.md) complete.
- From [Module 0.3](../phase-0-foundations/03-math-just-in-time.md): the gradient and the
  multivariable chain rule (calculus), vectors and matrices (linear algebra), and maximum
  likelihood estimation (probability). Refresh these first if needed; this module uses all
  of them directly.
- Your autograd work from Phase 0 is helpful intuition, though here you will write the
  gradients explicitly.

## Curated path

1. **ISLP, Chapter 3 (*Linear Regression*) and Chapter 4 (*Classification*, the logistic
   regression sections).** The clearest written derivation and discussion. Read for the
   model definitions, the least-squares and maximum-likelihood reasoning, interpretation of
   coefficients, and the assumptions. Free.
   https://www.statlearning.com/

2. **StatQuest, the linear and logistic regression series.** Watch the linear regression
   "main ideas," then the full logistic regression sequence (logistic regression, the
   logit, maximum likelihood, and why the loss has the form it does). These build the
   intuition behind the algebra.
   https://statquest.org/video-index/

3. **Andrew Ng's derivation of logistic regression (CS229 notes or the Machine Learning
   Specialization).** For the explicit gradient derivation, Ng's treatment is the canonical
   one: the sigmoid, the log-likelihood, and the gradient of the cross-entropy loss. The
   CS229 main notes are freely available.
   https://cs229.stanford.edu/

4. **scikit-learn user guide, linear models.** Read the `LinearRegression`,
   `LogisticRegression`, `Ridge`, and `Lasso` sections to see how the production API frames
   regularization, which you will then verify your own implementation against.
   https://scikit-learn.org/stable/modules/linear_model.html

**Deliberately skip for now:** generalized linear models beyond logistic, and the
statistical-inference machinery (p-values, confidence intervals on coefficients). Useful
later, not needed for the build.

## Background: the two derivations in words

**Linear regression** predicts a number as a weighted sum of features. Fitting it means
choosing weights that minimize the sum of squared errors. That squared-error loss is not
arbitrary: it is what maximum likelihood gives you if you assume the noise around the true
line is Gaussian. The minimum has a closed form (the normal equations), but you will also
reach it by gradient descent, because gradient descent is what scales to everything later.

**Logistic regression** predicts a probability for a binary outcome. It passes the same
linear combination through the sigmoid function to squash it into the range zero to one.
You cannot use squared error here; instead you choose weights by maximum likelihood, which
for this model produces the **cross-entropy** (log) loss. Its gradient has a strikingly
clean form, prediction minus target times the feature, the same shape as linear
regression's gradient, which is not a coincidence and is worth understanding.

**Regularization** adds a penalty on the size of the weights to fight overfitting. L2
(ridge) penalizes squared weights and shrinks them smoothly toward zero. L1 (lasso)
penalizes absolute weights and drives some exactly to zero, performing feature selection.
The geometric picture, the loss contours meeting a circular (L2) or diamond-shaped (L1)
constraint region, explains why L1 produces sparsity and L2 does not.

## Knowledge check

1. Why is the squared-error loss the natural choice for linear regression? What assumption
   about the noise does it correspond to?
2. Why can you not train logistic regression by minimizing squared error? What loss do you
   use instead, and where does it come from?
3. Write the gradient of the logistic regression loss with respect to the weights. Why does
   it have the same form as the linear regression gradient?
4. What is the difference between L1 and L2 regularization in what they do to the weights,
   and why does L1 produce exact zeros?
5. You add a strongly regularizing penalty and training error rises while validation error
   falls. Explain what happened in bias-variance terms.
6. What does a logistic regression coefficient mean, and what has to be true about your
   features for that interpretation to hold?

## Build gate

Implement both models from scratch and verify against scikit-learn.

**Specification:**

- A `LinearRegression` class with a `fit` method using batch gradient descent (and,
  optionally, the closed-form normal-equations solution as a cross-check) and a `predict`
  method. Standardize features and handle the intercept.
- A `LogisticRegression` class with `fit` (gradient descent on the cross-entropy loss),
  `predict_proba`, and `predict`. Implement the sigmoid in a numerically stable way.
- Optional L2 regularization on both, controlled by a strength parameter.

**Tests it must pass:**

- **Match scikit-learn.** On a shared dataset, your learned coefficients and predictions
  should match scikit-learn's to a small tolerance (allowing for differences in
  regularization and optimization settings, which you should reconcile deliberately).
- **Gradient check.** Verify your analytic gradient against a numerical finite-difference
  gradient.
- **Regularization behaves.** Show that increasing L2 strength shrinks the weights, and (if
  you implement L1) that it drives some weights to exactly zero.

Use a clean dataset such as scikit-learn's diabetes (regression) and breast cancer
(classification) to keep the focus on the mechanism.

One rule: write the gradients and the optimization yourself. scikit-learn appears only in
the verification step.

## Project

Turn the build into an artifact, reusing the project template from Module 0.2:

- A packaged module with both models, tests, and a README.
- A short report deriving, in your own words, the least-squares loss and the logistic
  cross-entropy loss and its gradient, then demonstrating regularization: a plot of weight
  magnitudes versus regularization strength, and the effect on validation performance.
- A brief reflection on coefficient interpretation: pick one fitted model and explain what
  its coefficients say, and what would invalidate that reading.

**Definition of done:** packaged code, passing tests, the derivation-and-regularization
report, and the interpretation reflection.

## Going deeper (optional)

- Implement multinomial (softmax) regression, the direct generalization to more than two
  classes and exactly the final layer of a classification neural network.
- Read ESL Chapters 3 and 4 for the rigorous statistical treatment.
- Add stochastic and mini-batch gradient descent and compare convergence with batch.

## Canonical references

- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, Chapters 3
  and 4.
- Ng, A. *CS229 Lecture Notes*, supervised learning and logistic regression.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Chapters 3 and 4.
