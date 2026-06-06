# Phase 2 — Deep Learning

This is where neural networks arrive, and where MLForge's experience-first principle does
its most important work. You will train a real neural network with PyTorch and watch it
learn before you build any of the machinery underneath it. Only then, knowing exactly what a
network is and what it is for, do you open it up and construct the engine that powers it,
from scratch.

That ordering matters. Building an automatic differentiation engine is one of the most
clarifying things you can do in all of machine learning, but it is a miserable place to
start. By the time you reach it here, you will have used a network, seen gradients do their
work, and have real questions that the from-scratch build answers. The build stops being
abstract and becomes the satisfying reveal of how something you already use actually works.

From there the phase climbs the architecture ladder: tensors and a full network from
scratch, then real command of PyTorch, the craft of making training actually work, and the
major architectures (convolutional networks, sequence models, and finally the transformer),
each used first and then built.

## Modules (work them in order)

- **2.1 — Neural Networks, Hands-On** *(hands-on · ship gate)*
  Train your first neural network with PyTorch and watch it learn. Experience the pieces
  working together before building any of them.
- **2.2 — From NumPy to a Working Autograd Engine** *(from-scratch · build gate)*
  Build the reverse-mode automatic differentiation engine behind every framework.
- **2.3 — Neural Nets & Backprop from Scratch** *(from-scratch · build gate)*
  Extend the engine to tensors; implement a multilayer perceptron, backprop, and SGD in
  NumPy. *(To be built.)*
- **2.4 — PyTorch in Depth** *(curated · build gate)*
  From using PyTorch to commanding it; reimplement your from-scratch network and confirm it
  matches. *(To be built.)*
- **2.5 — Making Training Work** *(hybrid · ship gate)*
  Initialization, normalization, optimizers, schedules, and debugging training. *(To be
  built.)*
- **2.6 — Convolutional Networks & Vision** *(curated · build gate)* *(To be built.)*
- **2.7 — Sequence Models** *(curated · concept check)* *(To be built.)*
- **2.8 — The Transformer from Scratch** *(from-scratch · build gate)* *(To be built.)*

## Phase capstone

**Capstone 2:** reproduce a published result. Take a small model or a compact training run,
reproduce it from scratch, and write up where your numbers matched, where they did not, and
why.

Begin with [Module 2.1](./01-neural-networks-hands-on.md).
