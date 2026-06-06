# Module 0.4 — Math, Just-in-Time: The Reference Layer

**Mode:** hybrid
**Gate:** concept check
**Est. effort:** 2–3 hours to build the index; revisited on demand thereafter. The
foundational track below is open-ended and depends on your starting point.

> Worried about the math? Do not be. This module does not ask you to learn it all now. You
> build a quick reference, take a short self-check to see where you stand, and, if some
> areas are rusty, follow the foundational track at your own pace. The curriculum then feeds
> you each result exactly when a model needs it, so the math always arrives with a purpose.

## Why this matters

Machine learning rests on a surprisingly small set of mathematical ideas, used over and
over. MLForge does not teach them as a separate semester for two reasons. First,
front-loading months of math you will not apply for a long time is one of the most reliable
ways to lose momentum. Second, mathematics is retained far better when it is learned in
service of a concrete model than when it is studied in the abstract. So the curriculum
introduces each result at the moment a model needs it, motivates it, and uses it
immediately.

For that to work, you need two things: a fast, trusted **reference layer** mapping the
specific results ML leans on to the single best explainer of each, and, if your foundations
are thin, a **foundational track** to build them up first. This module provides both. You
are not expected to learn all of this now. You build the index, run a short self-check to
see where you stand, and then return here, result by result, for the rest of the
curriculum.

## What you will be able to do

By the end of this module you will have:

- A bookmarked, indexed reference covering the linear algebra, calculus, probability, and
  optimization the curriculum uses.
- An honest sense of which areas are solid and which need work, from the concept check.
- A clear path to build any weak area up, through the foundational track.

## How to use this module

If your math is solid, skim the reference layer, bookmark the anchors, take the concept
check to confirm, and move on. You will return here as later modules link back to specific
results.

If your math is rusty or incomplete, treat the foundational track as real work. Build up
the area you are weakest in before starting Phase 1, where linear algebra and calculus
begin to carry weight. There is no shame and no rush here; a solid foundation pays off
across the entire curriculum.

## Anchor resources

Two resources are used throughout, one for intuition and one for rigor:

- **3Blue1Brown**, the *Essence of Linear Algebra* and *Essence of Calculus* series, for
  visual intuition:
  https://www.3blue1brown.com/topics/linear-algebra and
  https://www.3blue1brown.com/topics/calculus
- **Deisenroth, Faisal & Ong, *Mathematics for Machine Learning*** (MML), a rigorous and
  completely free textbook written specifically for this purpose:
  https://mml-book.com/ (direct PDF: https://mml-book.github.io/book/mml-book.pdf)

## The reference layer

For each area: what machine learning uses it for, the objects to keep familiar, and where
to go for intuition and for rigor.

### Linear algebra (MML chapters 2 to 4)
*Used for:* representing data as vectors and matrices, every linear layer, principal
component analysis, embeddings, and attention as a product of matrices.
*Objects to know:* vectors and matrices as linear maps, matrix multiplication as
composition, rank and null space, dot products and projections, eigenvalues and
eigenvectors, and the singular value decomposition.
*Intuition:* 3Blue1Brown, *Essence of Linear Algebra*. *Rigor:* MML chapters 2 to 4.

### Calculus and differentiation (MML chapter 5)
*Used for:* training. Gradient descent, backpropagation, and every optimizer depend on it.
*Objects to know:* derivatives and partial derivatives, the gradient, the multivariable
chain rule, Jacobians and the Hessian, and the Taylor expansion that underlies
optimization.
*Intuition:* 3Blue1Brown, *Essence of Calculus*, plus the *Backpropagation calculus*
chapter from the neural networks series. *Rigor:* MML chapter 5 (Vector Calculus).

### Probability and statistics (MML chapter 6)
*Used for:* modeling uncertainty, loss functions such as cross-entropy, evaluation,
generative models, and Bayesian reasoning.
*Objects to know:* random variables, common distributions (Bernoulli, Gaussian,
categorical), expectation and variance, conditional probability and Bayes' rule,
likelihood and maximum likelihood estimation, and KL divergence and cross-entropy.
*Intuition:* StatQuest, which explains statistics clearly and with an ML orientation:
https://statquest.org/video-index/ *Rigor:* MML chapter 6.

### Optimization (MML chapter 7)
*Used for:* the act of learning itself. Convexity, gradient descent and its variants, and
constrained optimization.
*Objects to know:* convex versus non-convex landscapes, gradient descent and step size,
stochastic gradient descent, momentum, and why neural network loss surfaces are non-convex
yet we optimize them with gradient methods anyway.
*Rigor:* MML chapter 7 (Continuous Optimization).

### Information theory (light, as needed)
*Used for:* loss functions and evaluation. Entropy, cross-entropy, and KL divergence recur
across classification, variational methods, distillation, and alignment.
*Resources:* the 3Blue1Brown and StatQuest treatments, plus MML's probability chapter, are
enough each time these appear.

## Foundational track (if you need to build the math up)

If the concept check below reveals gaps, work the matching resource until the ideas are
comfortable. These are full courses, not quick references.

- **Linear algebra:** 3Blue1Brown, *Essence of Linear Algebra* (for intuition), then Gilbert
  Strang's MIT 18.06 lectures for depth:
  https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- **Calculus:** 3Blue1Brown, *Essence of Calculus*, then Khan Academy's multivariable
  calculus for practice: https://www.khanacademy.org/math/multivariable-calculus
- **Probability and statistics:** StatQuest's statistics fundamentals playlist, then MML
  chapter 6: https://statquest.org/video-index/
- **All four, integrated for ML:** work straight through *Mathematics for Machine Learning*,
  which assumes only high-school mathematics and builds everything the field needs:
  https://mml-book.com/

## Knowledge check

Answer without notes. This is a thermometer, not an exam: each shaky answer simply points
to an area to warm up before Phase 1.

1. Geometrically, what does multiplying a vector by a matrix do, and what do that matrix's
   eigenvectors represent?
2. What is the gradient of a scalar function of several variables, and in what direction
   does it point relative to the function's level sets?
3. State Bayes' rule and name one place machine learning uses it.
4. What does maximum likelihood estimation do, and why does minimizing cross-entropy loss
   correspond to it for a classifier?
5. Why is the loss surface of a neural network generally non-convex, and why do we still
   use gradient-based optimization on it?
6. What does KL divergence measure, and why is it not symmetric?

## How this module is used later

When a later module invokes a result, it links back here. At that moment your job is the
just-in-time refresh: read the relevant section of *Mathematics for Machine Learning* or
watch the one explainer, apply it, and continue. Across the whole curriculum you will cover
most of the mathematics in this reference, but always in service of a model you are
building, never as mathematics for its own sake.

## Canonical references

- Deisenroth, Faisal & Ong (2020), *Mathematics for Machine Learning*, Cambridge University
  Press.
- 3Blue1Brown, *Essence of Linear Algebra* and *Essence of Calculus*.
- Strang, G. *Linear Algebra* (MIT 18.06).
