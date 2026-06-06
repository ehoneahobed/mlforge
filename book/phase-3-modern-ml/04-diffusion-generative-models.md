# Module 3.4 — Diffusion & Generative Models

**Mode:** hybrid
**Gate:** build gate
**Est. effort:** 12–16 focused hours

> Language models generate text by predicting the next token. Image generators work on a
> completely different and beautiful principle: start with pure noise and gradually remove it
> until an image appears. This module is your introduction to that idea. You will use a powerful
> image generator first, then build the diffusion process yourself and train a small one from
> scratch.

## Why this matters

Diffusion models are the technology behind modern image, audio, and video generation. They are
worth studying both for their practical importance and because the core idea is one of the most
elegant in machine learning: learn to reverse a gradual noising process, and you get a model
that can turn noise into data. Understanding diffusion rounds out your knowledge of generative
modeling beyond the autoregressive (next-token) approach of LLMs, and the from-scratch build is
the final rung of your from-scratch ladder.

It also reinforces ideas you already own. Diffusion models are trained with the same machinery
you have used throughout Phase 2 (a network, a loss, gradient descent); what changes is the
clever training objective. Building a small one makes the headline image generators concrete
rather than magical.

## What you will be able to do

By the end of this module you will be able to:

- Use a pretrained diffusion model to generate images and guide the output with prompts.
- Explain the forward (noising) and reverse (denoising) diffusion processes.
- Implement and train a small diffusion model (DDPM) from scratch on a simple dataset.
- Explain, conceptually, classifier-free guidance and latent diffusion (Stable Diffusion).

## Prerequisites

- [Module 2.6](../phase-2-deep-learning/06-convolutional-networks-vision.md): convolutional
  networks, since the denoising network is typically a convolutional U-Net.
- [Module 2.5](../phase-2-deep-learning/05-making-training-work.md): the training craft.
- A GPU for training (a free Colab or Kaggle notebook; keep the dataset and model small).

## Curated path

1. **Experience it first: use a pretrained model with the Hugging Face Diffusion Course,
   Unit 1 introduction.** Generate images with the `diffusers` library before building anything,
   to see what these models do and how guidance steers them.
   https://huggingface.co/learn/diffusion-course/en/unit1/1

2. **Build it from scratch: the Diffusion Course "Diffusion Models from Scratch" notebook.**
   This constructs the forward and reverse process and trains a small model, which is exactly
   your build gate.
   https://huggingface.co/learn/diffusion-course/en/unit1/3

3. **The mathematics, clearly: Lilian Weng, "What are Diffusion Models?"** The definitive
   written explanation of the forward and reverse processes, the training objective, and
   guidance. Read it alongside the build to understand what each line of code corresponds to.
   https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

4. **Stable Diffusion, conceptually: the Diffusion Course Unit 3.** How latent diffusion makes
   high-resolution text-to-image generation practical by working in a compressed latent space.
   https://huggingface.co/learn/diffusion-course/en/unit3/1

**Deliberately skip for now:** training a large text-to-image model (far beyond a single GPU)
and the newer sampler/architecture variants. Build a small unconditional DDPM you understand
fully.

## Background: the ideas in words

**The forward process** is simple and fixed: take a real image and add a small amount of
Gaussian noise, repeatedly, over many steps, until it becomes pure noise. There is nothing to
learn here; it is a defined corruption schedule.

**The reverse process** is what the model learns: given a noisy image at some step, predict the
noise that was added (equivalently, predict a slightly less noisy version). If the model can do
this at every noise level, then you can start from pure noise and run the reverse process step
by step, removing a little noise each time, until a clean, novel image emerges. The training
objective is beautifully simple: add known noise to a training image, ask the network to predict
that noise, and minimize the error. That is the whole DDPM training loop.

**The denoising network** is usually a **U-Net**, a convolutional architecture (from Module 2.6)
with skip connections, conditioned on which noise step it is at. This is why convolutions were a
prerequisite: the engine of an image diffusion model is a CNN.

**Guidance and latent diffusion.** To control *what* is generated (text-to-image),
**classifier-free guidance** steers the reverse process toward a prompt. To make
high-resolution generation affordable, **latent diffusion** (Stable Diffusion) runs the whole
process in a compressed latent space produced by an autoencoder, rather than on raw pixels. You
will understand these conceptually rather than build them at scale.

## Knowledge check

1. Describe the forward diffusion process. Why is there nothing to learn in it?
2. What exactly does the model learn to predict in the reverse process, and how is it trained?
3. How do you generate a new image once the denoising model is trained?
4. Why is the denoising network typically a U-Net, and how does this connect to Module 2.6?
5. What problem does latent diffusion (Stable Diffusion) solve, and how?
6. What does classifier-free guidance let you control?

## Build gate

Train a small diffusion model from scratch.

**Specification:**

- Implement the **forward noising process** (a noise schedule) and a **denoising network**
  (a small U-Net), following the Diffusion Course "from scratch" unit.
- Implement the **training loop**: add noise to training images at random steps, predict the
  noise, minimize the error.
- Implement **sampling**: start from noise and run the reverse process to generate images.
- Train on a simple dataset (MNIST or Fashion-MNIST) and generate samples.

**What it must show:**

- A training loss that decreases.
- Generated samples that clearly resemble the training distribution (recognizable digits or
  garments), not noise.
- That you can explain the training objective and the sampling loop in your own words.

## Project

Package it and write it up (reuse your template):

- The from-scratch diffusion model, training code, and a grid of generated samples.
- A report: the forward and reverse processes in your own words, the training objective, your
  samples, and a sentence on how this scales up to something like Stable Diffusion.

**Definition of done:** a trained from-scratch diffusion model generating recognizable images,
with the write-up.

## The workshop: ship it

Build this in its own repository, `modelwright-diffusion`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir modelwright-diffusion && cd modelwright-diffusion
uv init && uv add torch torchvision diffusers matplotlib && mkdir src
```

2. Implement the noise schedule, U-Net, training loop, and sampler in `src/`. Train on a free
   GPU notebook with the code committed here. (`diffusers` is for the "use it first" step and for
   reference; the from-scratch model is yours.)
3. Commit at checkpoints: "Forward process and U-Net", then "Training loop", then "Sampling
   generates recognizable images".
4. Add a README with your sample grid and explanation.
5. Ship it:

```bash
gh repo create modelwright-diffusion --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `modelwright-diffusion` is on GitHub, generates recognizable images from your own
diffusion code, and the README explains how it works.

## Going deeper (optional)

- Read the DDPM paper (Ho et al., 2020) and "The Annotated Diffusion Model" (Hugging Face blog).
- Add class-conditioning to your model so you can generate a specific digit.
- Explore the latent-diffusion and Stable Diffusion units of the course to see how the small
  model you built scales to text-to-image.

## Canonical references

- Hugging Face, *Diffusion Models Course*.
- Weng, L. *What are Diffusion Models?*
- Ho et al. (2020), *Denoising Diffusion Probabilistic Models*.
