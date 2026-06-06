# Phase 0 — Foundations & The Autograd Engine

Phase 0 is the on-ramp. It takes you from "I am curious about machine learning" to standing
on solid ground: you will know what the field is, have a working professional setup, be
ready for the math, and have built, from scratch, the one mechanism that powers every model
in the phases to come.

It is built as a gentle ramp on purpose. The first module is pure orientation, no code, just
the map of the field. The second gets your tools working and walks you through training your
very first model with a library, a quick win that shows machine learning actually running.
The third makes sure the math you will need is warm, with a path to build it up if it is
rusty. Only then, with all of that in place, does the final module ask you to build
something genuinely demanding: a working automatic differentiation engine.

That last module is the heart of the phase. Most people treat the line of code that computes
gradients as magic. By the end of Phase 0 you will have written that magic yourself, which
means nothing in the deep learning phases will be a black box. It will just be your engine,
scaled up.

## Modules (work them in order)

- **0.1 — The Landscape of Machine Learning** *(orientation · concept check)*
  No code. What machine learning is, how AI, ML, deep learning, and LLMs fit together, the
  types of learning, the end-to-end workflow, and what an ML engineer does. The map before
  the journey.
- **0.2 — The ML Engineer's Toolkit** *(curated · ship gate)*
  Set up a reproducible environment and experiment tracking, learn the scientific Python
  stack, and train and track your first model with a library. Produces a reusable project
  template.
- **0.3 — Math, Just-in-Time** *(hybrid · concept check)*
  An indexed reference to the specific math the curriculum uses, plus a foundational track
  if you need to build it up. Get math-ready before the build.
- **0.4 — From NumPy to a Working Autograd Engine** *(from-scratch · build gate)*
  The capstone build. Construct a reverse-mode automatic differentiation engine from scratch
  and use it to train a small neural network.

## Phase capstone

**Capstone 0:** your autograd engine, packaged and tested against a reference framework's
gradients, accompanied by a written derivation of the backward pass of every operation it
supports.

Begin with [Module 0.1](./01-landscape-of-ml.md).
