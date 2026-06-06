# Module 1.5 — Unsupervised: Clustering & Dimensionality Reduction

**Mode:** hybrid
**Gate:** build gate
**Est. effort:** 10–14 focused hours

## Why this matters

Everything so far has been supervised: you had labels and learned to predict them. Most of
the world's data has no labels. Unsupervised learning finds structure anyway, and two of its
techniques are essential tools. **Clustering** groups similar examples, useful for customer
segmentation, anomaly detection, and exploratory analysis. **Dimensionality reduction**
compresses many features into a few while preserving as much structure as possible, useful
for visualization, for speeding up downstream models, and for understanding what really
varies in your data.

Principal component analysis, the workhorse of dimensionality reduction, is also one of the
most beautiful applications of the linear algebra you have been carrying since Phase 0. It is
literally an eigenvalue problem: the principal components are the directions of greatest
variance, and they are the eigenvectors of the data's covariance matrix. Building PCA from
scratch turns abstract linear algebra into something you can see. This module also teaches a
crucial habit of skepticism, because the popular visualization methods that follow PCA can
produce beautiful, convincing pictures that are partly artifacts of their own settings.

## What you will be able to do

By the end of this module you will be able to:

- Implement k-means clustering from scratch and explain what it optimizes and where it
  fails.
- Implement PCA from scratch via the covariance matrix and its eigendecomposition, and
  explain it as variance maximization.
- Use t-SNE or UMAP for visualization while understanding what they distort.
- Choose a sensible number of clusters and components, and justify the choice.

## Prerequisites

- [Module 1.1](./01-the-learning-problem.md) complete.
- From [Module 0.3](../phase-0-foundations/03-math-just-in-time.md): eigenvalues and
  eigenvectors, covariance, and projections (linear algebra). PCA uses these directly, so
  refresh them first if they are not solid.

## Curated path

1. **StatQuest, the clustering and PCA series.** "K-means clustering," "Hierarchical
   clustering," then "PCA, Step-by-Step" and "PCA, Practical Tips." The PCA step-by-step
   video in particular makes the eigenvector picture concrete.
   https://statquest.org/video-index/

2. **ISLP, Chapter 12 (*Unsupervised Learning*).** The written treatment of PCA and
   clustering (k-means and hierarchical), including how to choose the number of components
   and the pitfalls of clustering. Free.
   https://www.statlearning.com/

3. **The t-SNE and UMAP cautions.** Read *How to Use t-SNE Effectively* (Distill), an
   interactive article showing how t-SNE's pictures change with its parameters and how to
   avoid over-reading them. Then skim the UMAP documentation's overview for the modern
   alternative.
   https://distill.pub/2016/misread-tsne/ and https://umap-learn.readthedocs.io/

4. **scikit-learn user guide, clustering and decomposition,** for the practical APIs and a
   comparison of clustering algorithms beyond k-means (DBSCAN, Gaussian mixtures).
   https://scikit-learn.org/stable/modules/clustering.html

## Background: the ideas in words

**k-means** partitions data into k groups by an iterative two-step dance: assign each point
to the nearest cluster center, then move each center to the mean of its assigned points.
Repeat until stable. It minimizes the total squared distance from points to their centers.
It is fast and useful, but it assumes roughly spherical, similarly sized clusters, it
requires you to choose k in advance, and its result depends on the random initialization, so
it is run several times and the best is kept.

**Principal component analysis** finds a new set of axes for your data, ordered by how much
variance each captures. The first principal component is the direction along which the data
varies most; the second is the next such direction at right angles to the first; and so on.
Mathematically these directions are the eigenvectors of the covariance matrix, and the
variance along each is its eigenvalue. Keeping only the top few components gives a faithful
low-dimensional summary. Crucially, PCA is sensitive to feature scale, so features are
standardized first.

**t-SNE and UMAP** are nonlinear methods built for visualization in two or three dimensions.
They try to keep points that are close in high-dimensional space close in the picture. They
can reveal structure PCA misses, but they come with a serious warning: cluster sizes and the
distances between clusters in their output are often meaningless, and the appearance changes
with parameters like perplexity. They are tools for generating hypotheses, not for drawing
firm conclusions.

## Knowledge check

1. Describe the two repeating steps of k-means and the quantity it minimizes.
2. Name three assumptions or limitations of k-means, and how each can mislead you.
3. What does the first principal component represent, and what is its relationship to the
   covariance matrix?
4. Why must features be standardized before PCA?
5. How would you decide how many principal components to keep?
6. What specifically should you not conclude from a t-SNE or UMAP plot, and why?

## Build gate

Implement k-means and PCA from scratch and verify against scikit-learn.

**Specification:**

- A `KMeans` class implementing the assign-and-update loop, with multiple random restarts
  keeping the best result by inertia, and a method to report the final inertia.
- A `PCA` class that standardizes the data, computes the covariance matrix, finds its
  eigenvectors and eigenvalues (or uses the SVD), and projects the data onto the top
  components. Report the explained-variance ratio per component.

**Tests it must pass:**

- **k-means matches scikit-learn** in cluster assignments (up to label permutation) and
  final inertia on a clean dataset.
- **PCA matches scikit-learn** in explained-variance ratios, and your reconstructed data
  from the top components matches scikit-learn's to a small tolerance. Beware that
  eigenvectors are defined up to a sign flip; account for it in your comparison.
- **The elbow appears.** Plot inertia versus k and explained variance versus number of
  components, and use them to justify a choice.

## Project

An unsupervised exploration of a real, unlabeled (or label-ignored) dataset:

- Use your PCA to reduce a high-dimensional dataset and plot the first two components, then
  apply your k-means to the data and overlay the discovered clusters.
- Produce a t-SNE or UMAP visualization of the same data and compare it to the PCA view,
  writing explicitly about what each reveals and what each might be hiding.
- Interpret the clusters: what, if anything, distinguishes them in terms of the original
  features? Be honest if the structure is weak.

**Definition of done:** the from-scratch k-means and PCA (packaged and tested), the
PCA-plus-clustering and t-SNE/UMAP visualizations, and a written interpretation that treats
the visualizations with appropriate skepticism.

## Going deeper (optional)

- Implement Gaussian mixture models and the expectation-maximization algorithm, a
  probabilistic generalization of k-means.
- DBSCAN, for clusters of arbitrary shape and built-in outlier detection.
- Read the UMAP paper for the topological ideas behind it.

## Canonical references

- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, Chapter 12.
- Wattenberg, Viégas & Johnson (2016), *How to Use t-SNE Effectively*, Distill.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Chapter 14.
