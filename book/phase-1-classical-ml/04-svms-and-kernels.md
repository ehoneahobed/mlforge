# Module 1.4 — SVMs & Kernels

**Mode:** curated breadth
**Gate:** concept check
**Est. effort:** 5–8 focused hours

## Why this matters

Support vector machines were the dominant approach to classification before deep learning,
and the ideas behind them remain some of the most elegant in the field. Two are worth
carrying with you regardless of whether you ever deploy an SVM. The first is the **margin**:
the insight that, among all the boundaries that separate two classes, the best one is the
boundary that sits as far as possible from the nearest points, because that is the boundary
most likely to generalize. The second is the **kernel trick**: a way to give a linear model
the power to draw curved boundaries by implicitly mapping data into a higher-dimensional
space, without ever computing the coordinates in that space.

This is a curated-breadth module with a concept-check gate, not a from-scratch build. The
optimization that fits an SVM is involved, and reimplementing it teaches less per hour than
the modules that surround it. The goal here is to understand the ideas well enough to use
SVMs correctly and to recognize the margin and kernel concepts when they reappear, which
they do, in places as varied as Gaussian processes and the theory of why some models
generalize.

## What you will be able to do

By the end of this module you will be able to:

- Explain what a maximum-margin classifier is and why margins relate to generalization.
- Explain the kernel trick and what a kernel function actually computes.
- Describe the role of the soft-margin parameter and common kernels, and choose between
  them.
- Say honestly when an SVM is and is not the right tool today.

## Prerequisites

- [Module 1.2](./02-linear-logistic-regression.md) complete; an SVM is another linear
  classifier, viewed through the lens of margins.
- From [Module 0.4](../phase-0-foundations/04-math-just-in-time.md): dot products and the
  geometry of vectors (linear algebra), and the idea of constrained optimization
  (optimization).

## Curated path

1. **StatQuest, the Support Vector Machines series.** "Support Vector Machines, Clearly
   Explained," then the parts on the polynomial and radial (RBF) kernels. The clearest
   intuition for margins and kernels available.
   https://statquest.org/video-index/

2. **ISLP, Chapter 9 (*Support Vector Machines*).** The written treatment: the maximal margin
   classifier, the soft margin and the cost parameter, and the extension to kernels. Free.
   https://www.statlearning.com/

3. **scikit-learn user guide, SVMs,** for the practical API, the meaning of the `C` and
   `gamma` parameters, and the important note that SVMs require feature scaling and do not
   scale well to very large datasets.
   https://scikit-learn.org/stable/modules/svm.html

**Deliberately skip for now:** the full derivation of the dual problem and the
Karush-Kuhn-Tucker conditions. Worth seeing once if you enjoy optimization (it is in ESL),
but not required to use SVMs well or to pass this gate.

## Background: the ideas in words

A linear classifier draws a hyperplane between two classes. Usually infinitely many
hyperplanes separate the training data, so which is best? The **maximum-margin** answer: the
one with the largest gap to the nearest points of either class. Those nearest points are the
**support vectors**, and they alone determine the boundary. A wide margin is a form of
built-in caution that tends to generalize better.

Real data is rarely perfectly separable, so the **soft margin** allows some points to sit
inside the margin or on the wrong side, with a parameter (called `C` in scikit-learn)
controlling how much such violations are penalized. Small `C` means a wider, more tolerant
margin (more regularization); large `C` means a narrower margin that tries hard to classify
every training point correctly (less regularization).

The **kernel trick** is the elegant part. The SVM's optimization depends on the data only
through dot products between points. A **kernel function** computes what the dot product
*would be* if the points were first mapped into some higher-dimensional space, without ever
performing that mapping. Swap the ordinary dot product for a kernel and a linear margin in
the high-dimensional space becomes a curved boundary in the original space. The radial basis
function kernel, in effect, gives an infinite-dimensional space and very flexible
boundaries, governed by a `gamma` parameter that sets how local each point's influence is.

## Knowledge check

1. Among many separating hyperplanes, which does a maximum-margin classifier choose, and why
   is that choice linked to better generalization?
2. What is a support vector, and why do only the support vectors determine the boundary?
3. What does the soft-margin parameter `C` control, and which direction means more
   regularization?
4. Explain the kernel trick. What does a kernel function compute, and why is it a
   computational shortcut?
5. What do the RBF kernel and its `gamma` parameter do to the decision boundary, and what
   happens as `gamma` grows large?
6. Given modern alternatives, when would you reach for an SVM today, and when would you not?

## Project

A focused empirical study rather than a build:

On a dataset with a clearly non-linear boundary (the two-moons or concentric-circles toy
sets are ideal for visualization, plus one real dataset), train SVMs with a linear kernel,
a polynomial kernel, and an RBF kernel. Visualize the decision boundaries. Then
systematically vary `C` and `gamma` and document, with plots, how each affects the boundary
and the validation score. Write a short note explaining, in margin-and-kernel terms, what
you observed, including a case where large `gamma` or large `C` clearly overfits.

**Definition of done:** the decision-boundary visualizations, the `C`-and-`gamma` study with
plots, and the written interpretation connecting what you saw to the concepts.

## Going deeper (optional)

- ESL Chapter 12 for the dual formulation and the optimization behind SVMs.
  https://hastie.su.domains/ElemStatLearn/
- Read about the connection between the kernel trick and Gaussian processes, a Bayesian
  cousin you may meet later.
- Support vector regression, the same margin idea applied to predicting numbers.

## Canonical references

- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, Chapter 9.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Chapter 12.
