# Module 2.2 — From NumPy to a Working Autograd Engine

**Mode:** from-scratch deep-dive
**Gate:** build gate
**Est. effort:** 8–14 focused hours (longer if backpropagation is new, which is fine)

> In the previous module you trained a neural network with PyTorch and watched it learn.
> Now you build the engine that made it possible. This is the most demanding module so far,
> and that is by design: you have used a network, so you know exactly what you are building
> and why. If it feels hard, that is the work doing its job, not a sign you are behind. Go
> slowly and lean on the curated path.

## Why this matters

Every deep learning framework in existence, PyTorch, JAX, TensorFlow, is at its core one
idea: **reverse-mode automatic differentiation**. You define a computation, and the
framework computes the gradient of the output with respect to every input, automatically.
Training a hundred-billion-parameter language model and training a two-neuron toy use the
exact same machinery. They differ only in scale.

If the line that computes gradients feels like magic, that magic will eventually limit
you: the first time a gradient is wrong, a loss becomes `NaN`, or a custom layer
misbehaves, you will be stuck staring at a black box. So before using a framework, you
build one. It is small, scalar-valued, and around 150 lines. You implement the backward
pass for every operation by hand, then watch your own engine train a neural network. After
this module, nothing in the deep learning phases is mysterious. It is your engine, scaled
up.

## What you will be able to do

By the end of this module you will be able to:

- Explain what a computation graph is and how reverse-mode autodiff traverses it.
- Derive and implement the backward pass for the core arithmetic operations and a
  nonlinearity.
- Build a small neural network library on top of your own engine.
- Train that network with a hand-written optimization loop and verify its gradients
  against a production framework.

## Prerequisites

- [Module 2.1](./01-neural-networks-hands-on.md), so you have trained a neural network and
  seen the pieces (layers, a loss, an optimizer, a training loop) working together. This
  module builds the machinery underneath what you used there.
- From Phase 0: [the toolkit](../phase-0-foundations/02-ml-engineers-toolkit.md) for a
  working environment and the reusable project template, and
  [the math reference](../phase-0-foundations/04-math-just-in-time.md) for the calculus
  below.
- Comfortable Python, including classes and special methods such as `__add__`, and basic
  use of closures.
- The **chain rule** from single and multivariable calculus. This is the entire
  mathematical content of backpropagation. If it is rusty, the 3Blue1Brown chapter in the
  path below rebuilds the intuition in about twenty minutes, and
  [Module 0.4](../phase-0-foundations/04-math-just-in-time.md) has the deeper references.
- A Python environment with `numpy`, `matplotlib`, and `torch` installed. PyTorch is used
  here only to *check* your gradients, never to build them.

## Curated path

Work these in order. The Karpathy lecture is the backbone; the rest support it.

1. **3Blue1Brown — Neural Networks, Chapters 1 to 4.** About ninety minutes total. All four
   are worth watching, but Chapter 4, *Backpropagation calculus*, is the load-bearing one.
   It gives you the geometric and calculus intuition before you write any code.
   Series: https://www.3blue1brown.com/topics/neural-networks
   Chapter 4 directly: https://www.youtube.com/watch?v=tIeHLnjs5U8

2. **Andrej Karpathy — "The spelled-out intro to neural networks and backpropagation:
   building micrograd."** This roughly two-and-a-half hour video is the spine of the
   module. Karpathy builds a complete automatic differentiation engine from nothing. Watch
   it with your editor open and **build alongside him, pausing constantly.** Do not watch
   passively; type every line and predict each step before he takes it.
   Part of the *Neural Networks: Zero to Hero* series, where the micrograd video is
   Lecture 1: https://github.com/karpathy/nn-zero-to-hero

3. **The `micrograd` reference repository.** Karpathy's finished implementation, roughly
   150 lines. Use it *only* to unstick yourself, and read it after your own attempt, not
   before.
   https://github.com/karpathy/micrograd

**Deliberately skip for now:** the support-vector-machine loss in the micrograd demo (a
simpler loss is used in the build gate below) and any tangent into convolutions or
transformers. Those belong to later phases. Stay scalar and stay small.

## Background: the one idea, in words

A computation is a graph. Each node holds a value and remembers the operation and inputs
that produced it. A **forward pass** computes the output by walking the graph from inputs
to output. The **backward pass** computes gradients by walking it the other way.

The engine relies on a single fact, the chain rule: the gradient of the final output with
respect to any node equals the gradient flowing into that node from above, multiplied by
the node's *local* gradient (how its output changes with respect to its inputs). For
example, in `c = a * b`, the local gradients are `dc/da = b` and `dc/db = a`. Each operation
therefore needs to know only its own local gradient; the engine chains them together.

Two details cause most bugs. First, gradients must **accumulate** (`+=`), because a value
used in more than one place receives gradient from each path, and those contributions add.
Second, the backward pass must visit nodes in **reverse topological order**, so that a
node's full upstream gradient is known before it passes gradient to its inputs.

## Knowledge check

Answer these without notes before attempting the build gate. If any is shaky, return to
the relevant part of the path.

1. What is a computation graph, and why does reverse-mode autodiff traverse it backward
   rather than forward?
2. In `c = a * b`, what are `dc/da` and `dc/db`, and how does the chain rule combine a
   node's local gradient with the gradient flowing into it from above?
3. Why must gradients accumulate rather than overwrite, and what bug appears if a variable
   is used twice and you forget to accumulate?
4. What is a topological sort, and why does backpropagation visit nodes in reverse
   topological order?
5. Derive the local gradient of `tanh(x)`. Why must a nonlinearity sit between linear
   layers for a network to be more expressive than a single linear map?
6. What does zeroing the gradients between optimization steps accomplish, and what goes
   wrong if you skip it?
7. In one sentence, what is the difference between what your engine does and what a
   framework like PyTorch does?

## Build gate

You pass this module when your engine trains a network and your gradients match PyTorch.

**Specification — build a scalar reverse-mode automatic differentiation engine:**

- A `Value` class wrapping a single scalar, storing its data, its gradient (initialized to
  zero), the set of child nodes that produced it, and a `_backward` function that pushes
  gradient to those children.
- Operations: addition, multiplication, raising to a constant power, negation,
  subtraction, division, and at least one nonlinearity (`tanh` or `relu`). Each defines its
  local gradient correctly.
- A `backward()` method that topologically sorts the graph, seeds the output gradient to
  one, and walks the nodes in reverse, invoking each node's `_backward`.
- A small neural network library on top: `Neuron`, `Layer`, and `MLP`, built only from your
  `Value` class.

**The tests it must pass:**

- **Gradient check against PyTorch.** Build the same expression in your engine and in
  PyTorch, call backward in both, and assert that every gradient matches to within `1e-6`.
  This is the non-negotiable correctness test. A scaffold is provided in the starter
  notebook.
- **It learns.** Train your `MLP` on a small two-dimensional dataset, such as scikit-learn's
  `make_moons`, using your own stochastic gradient descent loop, and show the loss
  decreasing to a good fit.

A starter notebook with the test scaffolding and `TODO` markers is provided alongside this
lesson: `02-autograd-from-numpy-starter.ipynb`. Fill in the gaps so you can focus on the
engine itself.

One rule: **no framework where the point is the mechanism.** PyTorch appears only inside
the gradient-check assertion. The engine itself is yours, written from scratch.

## Project

Turn the build into a portfolio artifact:

- Package your engine as a small importable module with a README a stranger could follow
  from clone to running tests.
- Include the test suite: the PyTorch gradient check plus a couple of unit tests per
  operation.
- Write a short report, about one page. For **each** operation you implemented, state the
  forward formula and derive its backward pass. Then describe training the network: the
  dataset, the loss curve, and one thing that surprised you or that you got wrong at first.

**Definition of done:** a clean repository, passing tests, a network that visibly learns,
and the derivation report. Reuse the project template you built in Phase 0 so this is a
proper, reproducible project, not a loose notebook.

## The workshop: ship it

Build this in its own repository, `modelwright-autograd`, using the project habits from
Module 0.2. (Start from the provided starter notebook, then package the result.)

1. Set up the project:

```bash
mkdir modelwright-autograd && cd modelwright-autograd
uv init && uv add numpy matplotlib torch scikit-learn && mkdir src tests
```

2. Implement the engine in `src/` (your `Value` class, the operations, `backward`, and the
   small neural-net library), with the PyTorch gradient check in `tests/`.
3. Commit at checkpoints: "Value class and operations", then "backward + gradient check
   passes", then "MLP trains on make_moons".
4. Add a README and your one-page derivation report (forward and backward for each operation).
5. Ship it:

```bash
gh repo create modelwright-autograd --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `modelwright-autograd` is on GitHub, the gradient check passes, the MLP visibly
learns, and the README contains your derivations. This is a portfolio centerpiece.

## Going deeper (optional)

- Extend the engine from scalars to **tensors** using NumPy arrays. This is precisely the
  leap Module 2.1 makes, so doing it now is a head start.
- Continue the *Neural Networks: Zero to Hero* series (the `makemore` videos, then building
  a GPT). It leads naturally into the deep learning phase.
- Read the autograd mechanics section of the PyTorch documentation and notice how your
  `_backward` functions correspond to its internal gradient functions.

## Canonical references

- Karpathy, A. *micrograd* and *Neural Networks: Zero to Hero*.
- 3Blue1Brown. *Neural Networks*, Chapters 1 to 4.
- Rumelhart, Hinton & Williams (1986), *Learning representations by back-propagating
  errors*. The original backpropagation paper; read it once you have built the thing it
  describes.
