# Module 2.3 — Neural Nets & Backprop from Scratch

**Mode:** from-scratch deep-dive
**Gate:** build gate
**Est. effort:** 10–16 focused hours

> In Module 2.2 you built a scalar autograd engine and trained a tiny network with it. That
> was the idea at its smallest. Here you scale the same idea up to the form real networks use:
> whole arrays of numbers at once, trained on a real dataset. The leap from one scalar at a
> time to a matrix at a time is the leap from a toy to something practical.

## Why this matters

Your scalar engine proved you understand backpropagation. But no real network computes one
scalar at a time; it would be impossibly slow. Real networks push entire vectors and matrices
through each layer in a single operation, and their gradients flow back the same way. Making
that jump, from scalar autodiff to **vectorized** forward and backward passes, is what turns
your understanding into something that can actually train on real data in reasonable time.

Doing this once by hand cements two things for the rest of your career. First, the exact shape
of a layer's forward and backward pass (a matrix multiply forward, two matrix multiplies
backward), which you will recognize inside every framework. Second, the practical machinery of
training: mini-batches, an epoch loop, a learning rate, and watching a loss curve fall on a
real problem. After this, PyTorch in the next module will feel like a faster, safer version of
something you already built.

## What you will be able to do

By the end of this module you will be able to:

- Derive and implement the vectorized forward and backward pass of a fully-connected layer.
- Build a multilayer perceptron in NumPy and train it with mini-batch stochastic gradient
  descent.
- Implement the softmax-with-cross-entropy output and its gradient correctly and stably.
- Train your network on a real dataset (MNIST) to a respectable accuracy, and read its loss
  curve.

## Prerequisites

- [Module 2.2](./02-autograd-from-numpy.md) (your scalar autograd engine) and
  [Module 2.1](./01-neural-networks-hands-on.md) (you have trained a network with a framework).
- From [Module 0.4](../phase-0-foundations/04-math-just-in-time.md): matrix multiplication and
  the multivariable chain rule. The backward pass here is the chain rule applied to matrices.

## Curated path

1. **Karpathy, *Neural Networks: Zero to Hero*, the `makemore` videos (parts 1 to 4).** These
   move from the scalar engine to real, array-based networks: building an MLP language model,
   then carefully working through the forward and backward passes by hand. Part 4,
   "becoming a backprop ninja," has you implement the backward pass manually, which is exactly
   this module's build.
   https://github.com/karpathy/nn-zero-to-hero

2. **CS231n notes: "Optimization" and "Neural Networks."** Stanford's notes derive the
   vectorized gradients cleanly and cover the practical training loop. Read the backpropagation
   and neural-network sections.
   https://cs231n.github.io/optimization-2/ and https://cs231n.github.io/neural-networks-1/

3. **3Blue1Brown, Neural Networks Chapter 3 (backpropagation) and Chapter 4 (the calculus).**
   Rewatch with your scalar engine in mind; the matrix form will now click.
   https://www.3blue1brown.com/topics/neural-networks

**Deliberately skip for now:** convolutions, fancy optimizers, and normalization layers. They
come in Modules 2.5 and 2.6. Here, a plain MLP trained with plain SGD is the whole point.

## Background: the vectorized layer, in words

A fully-connected layer computes `Z = X W + b`, where `X` is a batch of inputs (one row per
example), `W` is the weight matrix, and `b` is the bias. A nonlinearity like ReLU is applied
elementwise. Stack a couple of these and you have an MLP.

The backward pass is the chain rule in matrix form. Given the gradient of the loss with
respect to a layer's output (call it `dZ`), the gradients are: `dW = X^T dZ`, `db = sum(dZ)`
over the batch, and the gradient passed to the previous layer is `dX = dZ W^T`. Those three
formulas, plus the elementwise derivative of the activation, are the entire backward pass of a
network. Everything frameworks do is this, repeated and optimized.

The output layer uses **softmax** to turn scores into probabilities and **cross-entropy** as
the loss. Their combined gradient simplifies beautifully to `prediction - one_hot_target`,
the same clean form you saw for logistic regression in Phase 1, which is no accident: the
output layer of a classifier *is* (multinomial) logistic regression.

## Knowledge check

1. Write the forward pass of a fully-connected layer in matrix form, and say what each
   dimension represents.
2. Given `dZ` (the gradient at a layer's output), write `dW`, `db`, and `dX`. Why does `dW`
   involve `X^T`?
3. Why do we process data in mini-batches rather than one example at a time or the whole
   dataset at once?
4. What is the combined gradient of softmax-with-cross-entropy, and why is it the same shape
   as the logistic regression gradient?
5. What is an epoch, and what typically happens to training and validation loss over many
   epochs?
6. Your loss is `NaN` after the first step. Name two likely causes and how you would check
   each.

## Build gate

Build a multilayer perceptron from scratch in NumPy and train it on MNIST.

**Specification:**

- A small framework of layers: a `Linear` layer (with vectorized forward and backward), a
  `ReLU`, and a softmax-cross-entropy loss. You may either extend your Module 2.2 engine to
  tensors, or write explicit forward/backward methods per layer (either is valid; explicit
  layers are often clearer here).
- A training loop: mini-batch SGD over several epochs, tracking training and validation loss
  and accuracy.
- Sensible weight initialization (small random values, or He initialization for ReLU).

**Tests it must pass:**

- **Gradient check.** Verify your analytic gradients against numerical finite-difference
  gradients on a small network.
- **It learns MNIST.** Reach at least ~95% test accuracy on MNIST with a simple MLP. (This is
  very achievable; if you are far below, debug rather than move on.)
- **Curves behave.** Plot training and validation loss over epochs and show them decreasing.

## Project

Package the build and reflect on it (reuse your project template from Phase 0):

- A clean, tested implementation with a README a stranger can run.
- A short report: the forward and backward formulas for your layers (derived, not just
  stated), your final accuracy and curves, and one bug you hit and how you found it. Debugging
  stories are where the real learning shows.

**Definition of done:** the from-scratch MLP, passing the gradient check and hitting the
accuracy bar on MNIST, plus the report.

## The workshop: ship it

Build this in its own repository, `modelwright-mlp-from-scratch`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir modelwright-mlp-from-scratch && cd modelwright-mlp-from-scratch
uv init && uv add numpy matplotlib scikit-learn torch && mkdir src tests
```

2. Implement the layers and training loop in `src/`, with the gradient check in `tests/`.
   (`torch` is only for the gradient check and to load MNIST conveniently; the network is
   yours.)
3. Commit at checkpoints: "Linear + ReLU forward/backward", then "Gradient check passes",
   then "MLP trains on MNIST to 95%+".
4. Add a README and your derivation-and-results report.
5. Ship it:

```bash
gh repo create modelwright-mlp-from-scratch --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `modelwright-mlp-from-scratch` is on GitHub, the gradient check passes, and the
README shows your MNIST curves and accuracy.

## Going deeper (optional)

- Add momentum to your SGD and compare convergence; you will meet optimizers properly in
  Module 2.5.
- Implement dropout as an extra layer and observe its regularizing effect.
- Read the d2l.ai chapter on implementing an MLP from scratch for a second perspective.
  https://d2l.ai/

## Canonical references

- Karpathy, A. *Neural Networks: Zero to Hero* (`makemore`).
- Stanford CS231n notes, *Optimization* and *Neural Networks*.
- 3Blue1Brown, *Neural Networks*, Chapters 3 and 4.
