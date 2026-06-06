# Module 2.7 — Sequence Models

**Mode:** curated breadth
**Gate:** concept check
**Est. effort:** 6–10 focused hours

> Convolutional networks handle grids like images. But much of the world is sequential: text,
> speech, time series, where order matters and the meaning of one element depends on others
> around it. This module covers the recurrent networks built for sequences, and, just as
> importantly, the limits that drove the field to invent attention. It is the bridge to the
> transformer.

## Why this matters

For years, recurrent neural networks were how machine learning handled sequences. Understanding
them matters for two reasons. First, they are still used and still instructive: the idea of a
network carrying a **hidden state** through time, updating it as it reads each element, is
fundamental. Second, and more importantly for what comes next, their weaknesses are the entire
motivation for the transformer. You cannot fully appreciate why attention was such a leap until
you feel what recurrent networks struggle with: long-range dependencies, vanishing gradients,
and the fact that they must process a sequence one step at a time, which is slow and hard to
parallelize.

This is a concept-check module with a small hands-on taste. The goal is genuine understanding
of recurrent models and a visceral sense of their limits, so that Module 2.8 lands with full
force.

## What you will be able to do

By the end of this module you will be able to:

- Explain how a recurrent network processes a sequence using a hidden state, and what
  backpropagation through time is.
- Explain the vanishing-gradient problem and why it makes long-range dependencies hard.
- Describe how LSTMs and GRUs use gates to carry information over longer spans.
- Articulate the specific limitations of recurrent models that motivated attention.

## Prerequisites

- [Module 2.5](./05-making-training-work.md): you can train networks and understand gradient
  flow, which is central to why recurrent models struggle.

## Curated path

1. **Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks."** The classic,
   motivating read: simple character-level RNNs producing surprisingly coherent text, with
   clear explanation and code. Start here for intuition and excitement.
   https://karpathy.github.io/2015/05/21/rnn-effectiveness/

2. **colah, "Understanding LSTM Networks."** The single clearest explanation of LSTMs and their
   gates, beautifully illustrated. This is the reference everyone learns LSTMs from.
   https://colah.github.io/posts/2015-08-Understanding-LSTMs/

3. **Dive into Deep Learning (d2l.ai), the recurrent-networks chapters,** for the mechanics of
   RNNs, backpropagation through time, and LSTMs/GRUs in code.
   https://d2l.ai/

4. **The motivation for attention.** Read the introduction and "why attention" framing of
   "The Illustrated Transformer" (just the opening sections here; you will read it fully in
   Module 2.8). Notice the problems it says attention solves; those are exactly the recurrent
   limits from this module.
   https://jalammar.github.io/illustrated-transformer/

**Deliberately skip for now:** sequence-to-sequence with attention as a bolt-on, and the full
transformer. Module 2.8 builds the transformer properly. Here, understand recurrence and why it
was not enough.

## Background: the ideas in words

A **recurrent neural network** reads a sequence one element at a time, maintaining a **hidden
state** that summarizes everything seen so far. At each step it combines the current input with
the previous hidden state to produce a new hidden state (and optionally an output). The same
weights are used at every step, so the network can handle sequences of any length. Training
uses **backpropagation through time**: unroll the network across the sequence and apply the
usual chain rule.

The trouble is gradients. When you backpropagate through many time steps, the gradient is
multiplied by the same factors over and over, so it tends to either **vanish** (shrink toward
zero, so the network cannot learn long-range dependencies) or explode. This is the core
weakness of plain RNNs.

**LSTMs and GRUs** address vanishing gradients with **gates**: small learned controllers that
decide what to keep, what to forget, and what to output, plus a **cell state** that information
can flow along with little interference. This lets them carry signal over longer spans, and they
worked well enough to power machine translation and speech for years.

But two problems remained. Even LSTMs struggle with truly long-range dependencies, and,
critically, recurrence is **inherently sequential**: you must compute step 1 before step 2,
which makes training hard to parallelize and slow on modern hardware. **Attention** removes
both limits by letting every position look directly at every other position, in parallel. That
is the door Module 2.8 walks through.

## Knowledge check

1. How does a recurrent network use its hidden state to process a sequence?
2. What is backpropagation through time?
3. Explain the vanishing-gradient problem and why it specifically hurts long-range
   dependencies.
4. How do the gates in an LSTM help information persist over many time steps?
5. Name the two key limitations of recurrent networks that motivated attention.
6. Why is the sequential nature of RNNs a problem for training on modern hardware, and how
   does attention sidestep it?

## Project

A small hands-on taste, to feel both the capability and the limits:

- Train a **character-level RNN or LSTM** on a text corpus of your choice (Karpathy's blog and
  code make this easy; PyTorch's `nn.LSTM` makes it short). Generate some text from it.
- Observe and document its limits: where does the generated text lose coherence over longer
  spans? Try a longer-range dependency (matching brackets, or remembering a name introduced
  earlier) and note how it does.
- Write a short reflection connecting what you observed to the vanishing-gradient and
  sequential-processing limitations, and to why attention is expected to help.

**Definition of done:** a trained char-level recurrent model that generates text, plus a
written reflection on its limits and the motivation for attention.

## The workshop: ship it

Build this in its own repository, `modelwright-char-rnn`.

1. Set up the project:

```bash
mkdir modelwright-char-rnn && cd modelwright-char-rnn
uv init && uv add torch matplotlib && mkdir src
```

2. Write the char-level RNN/LSTM training and sampling code in `src/`, train on a text file of
   your choice, and generate samples.
3. Commit at checkpoints: "Train char-level RNN", then "Generate text and probe limits".
4. Add a README with sample outputs and your reflection on the limits and the motivation for
   attention.
5. Ship it:

```bash
gh repo create modelwright-char-rnn --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `modelwright-char-rnn` is on GitHub with generated samples and a reflection that
sets up why attention is the next step.

## Going deeper (optional)

- Implement a simple RNN cell from scratch and watch gradients vanish across time steps.
- Read about sequence-to-sequence models and the original (pre-transformer) attention
  mechanism of Bahdanau et al.
- Skim the GRU paper to compare its simpler gating to the LSTM.

## Canonical references

- Karpathy, A. *The Unreasonable Effectiveness of Recurrent Neural Networks*.
- Olah, C. *Understanding LSTM Networks*.
- Dive into Deep Learning (d2l.ai), recurrent-networks chapters.
