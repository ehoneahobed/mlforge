# Phase 0 — Foundations & The Autograd Engine

Phase 0 builds the floor the rest of the curriculum stands on. It has two jobs. The first
is to turn general programming skill into ML-grade engineering practice: reproducible
environments, experiment tracking, and the habits that let a result be trusted and rerun
months later. The second, and more important, is to remove the single biggest source of
mystery in everything that follows: **automatic differentiation**.

Most people treat the line that computes gradients as magic. By the end of this phase you
will have written that magic yourself. Every model in later phases, every neural network,
every transformer, every diffusion model, is built on the abstraction you construct here by
hand. Understanding it once, at the bottom, pays off at every level above.

This phase also teaches mathematics the way the whole curriculum does: just in time. Rather
than front-loading a math course, Phase 0 gives you an indexed reference layer and a
foundational track, so you can pull in each result at the moment a model demands it.

## Modules

- **0.1 — From NumPy to a Working Autograd Engine** *(from-scratch · build gate)*
  Build a reverse-mode automatic differentiation engine from scratch and use it to train a
  small neural network.
- **0.2 — The ML Engineer's Toolkit** *(curated · ship gate)*
  Reproducible environments, experiment tracking, profiling, and disciplined use of the
  scientific Python stack. Produces a reusable project template.
- **0.3 — Math, Just-in-Time: The Reference Layer** *(hybrid · concept check)*
  An indexed map of the math the curriculum invokes, plus a foundational track for anyone
  building the math up from the start.

## Phase capstone

**Capstone 0:** your autograd engine, packaged and tested against a reference framework's
gradients, accompanied by a written derivation of the backward pass of every operation it
supports.

Begin with [Module 0.1](./01-autograd-from-numpy.md).
