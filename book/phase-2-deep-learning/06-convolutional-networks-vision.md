# Module 2.6 — Convolutional Networks & Vision

**Mode:** curated breadth
**Gate:** build gate
**Est. effort:** 12–16 focused hours

> Until now your networks have treated inputs as flat vectors. Images are not flat; nearby
> pixels are related, and the same pattern (an edge, a texture) can appear anywhere. The
> convolutional network is the architecture built around those two facts, and it is what made
> deep learning famous. You will use a powerful pretrained one first, then build the
> convolution operation itself from scratch.

## Why this matters

Convolutional networks were the breakthrough that started the modern deep learning era, and
they remain the workhorse of computer vision. Two ideas make them special, and both are worth
internalizing because they recur throughout deep learning. **Local connectivity:** a
convolution looks at a small patch at a time, matching the fact that meaningful image structure
is local. **Weight sharing:** the same small filter slides across the whole image, so a pattern
learned in one place is detected everywhere, which is both efficient and a form of built-in
generalization.

This module also introduces **transfer learning**, one of the most practically important
techniques in all of ML: instead of training from scratch, you take a network already trained
on millions of images and adapt it to your problem with very little data. It is how most real
vision projects are actually done.

## What you will be able to do

By the end of this module you will be able to:

- Explain convolution, pooling, padding, and stride, and what each does to a feature map.
- Implement a 2D convolution forward pass from scratch and verify it against PyTorch.
- Build and train a small convolutional network on a real image dataset.
- Use transfer learning to adapt a pretrained model to a new task with limited data.

## Prerequisites

- [Module 2.5](./05-making-training-work.md): you can train networks well in PyTorch.
- From [Module 0.4](../phase-0-foundations/04-math-just-in-time.md): comfort with arrays and
  matrix operations; a convolution is a structured pattern of multiplies and sums.

## Curated path

1. **Experience it first: the PyTorch transfer-learning tutorial.** Before building anything,
   fine-tune a pretrained convolutional network on a new dataset and see how powerful and
   data-efficient it is. This is the "use it" half of the module.
   https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

2. **CS231n: the Convolutional Networks notes (and lectures).** The definitive explanation of
   convolution, pooling, padding, stride, and how real architectures are assembled. This is the
   conceptual spine.
   https://cs231n.github.io/convolutional-networks/

3. **3Blue1Brown, "But what is a convolution?"** Builds the core intuition for the operation
   itself, visually, before you implement it.
   https://www.youtube.com/watch?v=KuXjwB4LzSA

4. **Dive into Deep Learning (d2l.ai), the convolutional-networks chapters,** for a hands-on,
   code-first treatment including modern architectures (LeNet, AlexNet, ResNet).
   https://d2l.ai/

**Deliberately skip for now:** object detection, segmentation architectures, and the full zoo
of CNN variants. Understand the core operation and one solid architecture, plus transfer
learning; the rest is specialization (Phase 5).

## Background: the ideas in words

A **convolution** slides a small grid of weights (a **filter** or **kernel**) across the image,
computing a dot product at each position to produce a **feature map**. Early filters learn to
detect edges and simple textures; deeper layers combine these into parts and then objects. The
same filter is used at every position (**weight sharing**), so the network needs far fewer
parameters than a fully-connected layer and detects a pattern wherever it appears.

**Padding** adds a border so the output keeps a convenient size and edge pixels are not
under-used. **Stride** is how far the filter jumps each step; a larger stride shrinks the
output. **Pooling** (usually max pooling) downsamples a feature map, summarizing each small
region, which adds a little translation tolerance and reduces computation. A typical CNN stacks
convolution-activation-pooling blocks to build up from pixels to abstract features, then ends
with fully-connected layers (or global pooling) for the prediction.

**Transfer learning** exploits the fact that the early-to-middle features a network learns
(edges, textures, shapes) are general. You take a model pretrained on a huge dataset
(ImageNet), keep its learned features, and retrain only the final layers (or fine-tune the
whole thing gently) on your smaller dataset. You get strong results with a fraction of the data
and compute.

## Knowledge check

1. What two ideas (about how images are structured) make convolution the right operation for
   them?
2. What do padding and stride each control in the output of a convolution?
3. What does pooling do, and why is it useful?
4. Why does a convolutional layer have far fewer parameters than a fully-connected layer over
   the same input?
5. Explain transfer learning. Why do the early layers of a vision model transfer well to a new
   task?
6. You have 500 labeled images for a new task. Would you train a CNN from scratch or use
   transfer learning, and why?

## Build gate

Implement convolution from scratch, then train real CNNs.

**Specification:**

- A from-scratch **2D convolution forward pass** in NumPy (a single channel is fine to start,
  then extend to multiple input/output channels), supporting stride and padding.
- A small CNN built and trained in **PyTorch** on a real dataset (CIFAR-10 is the standard
  choice), reaching a reasonable accuracy.
- A **transfer-learning** run: take a pretrained backbone (e.g., ResNet from `torchvision`),
  adapt it to a dataset, and compare its accuracy and training time to your from-scratch CNN.

**Tests it must pass:**

- **Convolution matches PyTorch.** Your from-scratch `conv2d` output matches
  `torch.nn.functional.conv2d` to a small tolerance on the same input and kernel.
- **The CNN learns CIFAR-10** to a sensible accuracy (well above chance; aim for a result you
  would not be embarrassed by).
- **Transfer learning wins.** Show that the pretrained model reaches higher accuracy faster
  than your from-scratch CNN on the same data.

## Project

Package the build and write it up (reuse your template):

- The from-scratch convolution (tested against PyTorch), the trained CNN, and the
  transfer-learning comparison.
- A short report: what convolution computes (in your own words), your CIFAR-10 result, and the
  transfer-learning comparison with curves and a sentence on why pretraining helped.

**Definition of done:** the verified from-scratch convolution, a trained CNN, and the
transfer-learning comparison, with the report.

## The workshop: ship it

Build this in its own repository, `modelwright-cnn-vision`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir modelwright-cnn-vision && cd modelwright-cnn-vision
uv init && uv add torch torchvision numpy matplotlib && mkdir src tests
```

2. Implement the from-scratch convolution in `src/` with the PyTorch comparison in `tests/`,
   then the CNN and transfer-learning scripts. (Training is much faster on a GPU; a free Colab
   or Kaggle notebook is fine, with the code committed here.)
3. Commit at checkpoints: "conv2d from scratch matches PyTorch", then "CNN trains on CIFAR-10",
   then "Transfer learning comparison".
4. Add a README with results, curves, and your write-up.
5. Ship it:

```bash
gh repo create modelwright-cnn-vision --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `modelwright-cnn-vision` is on GitHub, the convolution test passes, and the README
shows your CNN and transfer-learning results.

## Going deeper (optional)

- Implement the **backward** pass of your convolution; it is excellent gradient practice.
- Read the ResNet paper to understand residual connections, which make very deep networks
  trainable and reappear in transformers.
- Explore data augmentation (`torchvision.transforms`) and measure its effect on overfitting.

## Canonical references

- Stanford CS231n notes, *Convolutional Networks*.
- PyTorch documentation, *Transfer Learning* tutorial.
- Dive into Deep Learning (d2l.ai), convolutional-networks chapters.
