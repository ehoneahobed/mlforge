# Module 2.1 — Neural Networks, Hands-On

**Mode:** hands-on
**Gate:** ship gate
**Est. effort:** 6–10 focused hours

## Why this matters

You are about to spend several modules building neural networks from the ground up, all the
way down to the automatic differentiation engine that powers them. Before any of that, you
should train one and watch it work. Experiencing a network in action, seeing the loss fall
and the accuracy climb, gives you the intuition and the questions that make the from-scratch
builds meaningful instead of abstract.

This module is the experience half of "experience it, then build it." You will use a
framework (PyTorch) to assemble and train a real neural network on real data, in not many
lines of code. You will not yet understand everything happening underneath, and that is the
point. You will meet the pieces (layers, activations, a loss function, an optimizer, a
training loop) as things you can use, and form a mental picture of how they fit. The next
modules then open each piece up.

## What you will be able to do

By the end of this module you will be able to:

- Assemble a simple neural network in PyTorch from layers and activations.
- Write a training loop: forward pass, loss, backward pass, optimizer step.
- Train the network on a real dataset and track its loss and accuracy.
- Recognize the moving parts (layers, loss, optimizer, epochs, batches) and say what each
  one does, even before you can build them.

## Prerequisites

- Phase 0 complete: you can train and evaluate a model with a library, and you can load and
  split data.
- Phase 1 is recommended but not strictly required. The classical-ML intuition (especially
  logistic regression, which is a one-layer network) makes neural networks feel like a
  natural next step rather than a leap.
- A Python environment with `torch` installed.

## Curated path

1. **The picture first: 3Blue1Brown, Neural Networks, Chapters 1 and 2.** If you have not
   watched these since Module 0.1, watch them again now with fresh eyes. Chapter 1 shows what
   a network is; Chapter 2 shows how it learns by gradient descent. About forty minutes, and
   it will click far more now.
   https://www.3blue1brown.com/topics/neural-networks

2. **The official PyTorch "Learn the Basics" tutorial.** This is the spine of the module.
   It walks you, with runnable code, through tensors, datasets and data loaders, building a
   model with `nn.Module`, the training loop, and saving a model. Follow it end to end and
   run every cell.
   https://pytorch.org/tutorials/beginner/basics/intro.html

3. **The 60-minute blitz, for a second pass.** PyTorch's "Deep Learning with PyTorch: A
   60 Minute Blitz" trains an image classifier and reinforces the same loop on a different
   problem. Use it to consolidate, not to learn everything anew.
   https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

**Deliberately skip for now:** how autograd works internally (that is the next module, where
you build it), GPUs and performance, and advanced architectures. Right now you are learning
to drive, not to build the engine.

## Background: the pieces, in words

A neural network is a stack of simple steps. A **layer** takes a vector of numbers, multiplies
it by a matrix of **weights**, adds a bias, and passes the result through a nonlinear
**activation function** (such as ReLU). Stack several of these and the network can represent
remarkably complex relationships. The last layer produces the prediction, often through the
same logistic or softmax step you met in classical ML.

Training works by the loop you will see in every example. A **forward pass** runs the data
through the network to produce predictions. A **loss function** measures how wrong those
predictions are. The **backward pass** computes how each weight should change to reduce the
loss (this is the automatic differentiation you will build next). The **optimizer** then
nudges every weight a small step in the improving direction. Repeat this over the data many
times (**epochs**), usually in small **batches**, and the network learns.

For now, treat the backward pass as the framework's job. Notice *that* it works: call
`loss.backward()`, then `optimizer.step()`, and the network improves. The next module
removes the magic by having you build that step yourself.

## Knowledge check

1. What are the four steps of one iteration of a training loop?
2. What does an activation function do, and why must a network have them to be more powerful
   than a single linear model?
3. What is the difference between an epoch and a batch?
4. What does the optimizer do after the loss is computed?
5. A logistic regression model is a neural network with how many layers? What does that tell
   you about how classical ML and deep learning relate?
6. You call `loss.backward()`. In one sentence, what has it just computed (even if you cannot
   yet build it)?

## Project (ship gate)

**Train a neural network, end to end, and report on it.**

- Using PyTorch, build and train a small neural network on a real dataset (the tutorials use
  FashionMNIST or CIFAR-10; either is ideal). Write the model, the training loop, and the
  evaluation yourself, even if you started from the tutorial code.
- Track and plot the training and validation loss and accuracy across epochs.
- Run one small experiment: change something (the number of layers or hidden units, the
  learning rate, or the number of epochs) and report how it affected the curves.
- Write a short report: what you built, what the curves show, what your experiment changed,
  and at least three honest questions you now have about how it works underneath. Those
  questions are exactly what the next modules answer.

**Definition of done:** a trained network with plotted learning curves, one documented
experiment, and the written report including your open questions. Use the project template
from Phase 0 so it is reproducible.

## Going deeper (optional)

- Continue Andrej Karpathy's *Neural Networks: Zero to Hero*, which begins exactly where the
  next module goes (building the engine) and is the perfect bridge.
  https://github.com/karpathy/nn-zero-to-hero
- Try the same network on a different dataset to see how much of the workflow carries over.

## Canonical references

- PyTorch documentation, *Learn the Basics* and *60 Minute Blitz*.
- 3Blue1Brown, *Neural Networks*, Chapters 1 and 2.
