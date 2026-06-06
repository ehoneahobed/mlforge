# Module 2.8 — The Transformer from Scratch

**Mode:** from-scratch deep-dive
**Gate:** build gate
**Est. effort:** 14–20 focused hours

> This is the summit of Phase 2 and the gateway to everything modern. The transformer is the
> architecture behind every large language model, and most of today's frontier systems in
> vision and beyond. You will use a small one first, then build the whole thing yourself:
> self-attention, multi-head attention, and a complete GPT that generates text. When you
> finish, the headlines about large language models will describe a machine you have built.

## Why this matters

The transformer is the most important architecture in modern machine learning. It solved the
two problems that limited recurrent networks (Module 2.7): it lets every position attend
directly to every other position, capturing long-range dependencies in a single step, and it
does so in parallel, which is what made training on internet-scale data feasible. Everything in
Phase 3 (large language models, fine-tuning, retrieval, generation) sits on top of it.

Building one from scratch is the capstone of your from-scratch ladder: autograd, then an MLP,
then convolution, and now attention. After this, no part of a modern model is a black box to
you. You will have implemented the exact mechanism, `softmax(QK^T / sqrt(d)) V`, that the whole
field runs on, and trained it to produce text.

## What you will be able to do

By the end of this module you will be able to:

- Explain self-attention precisely: queries, keys, values, and the scaled dot-product.
- Implement self-attention and multi-head attention from scratch.
- Assemble a full transformer block (attention, feed-forward, residual connections, layer
  normalization) and a complete decoder-only model.
- Train a small character-level GPT and generate coherent text from it.

## Prerequisites

- [Module 2.7](./07-sequence-models.md): you understand why attention was needed.
- [Module 2.5](./05-making-training-work.md): residual connections and layer normalization,
  which the transformer relies on.
- [Module 2.4](./04-pytorch-in-depth.md): you are fluent in PyTorch, which you will build this
  in.

## Curated path

1. **Build it with Karpathy: "Let's build GPT: from scratch, in code, spelled out."** This is
   the spine of the module, a roughly two-hour build-along where Karpathy constructs a working
   GPT and trains it on Shakespeare, ending at the core of nanoGPT. Build alongside him,
   pausing constantly, exactly as you did for the autograd engine.
   Video in the series: https://github.com/karpathy/nn-zero-to-hero
   The companion code: https://github.com/karpathy/nanoGPT

2. **The Illustrated Transformer (Jay Alammar).** The clearest visual explanation of attention,
   queries/keys/values, and multi-head attention. Read it before and alongside the build to
   keep the picture in your head.
   https://jalammar.github.io/illustrated-transformer/

3. **The Annotated Transformer (Harvard NLP).** A line-by-line PyTorch implementation of the
   original paper, annotated. Use it as a reference to check your own implementation against.
   https://nlp.seas.harvard.edu/annotated-transformer/

4. **The original paper: "Attention Is All You Need" (Vaswani et al., 2017).** Read it once you
   have built the thing it describes; it will be remarkably readable, and reading the source is
   a skill Phase 5 develops fully.
   https://arxiv.org/abs/1706.03762

**Deliberately skip for now:** the encoder-decoder machine-translation setup, and efficiency
tricks (flash attention, KV caching). Build the decoder-only GPT, which is the modern default;
the rest is refinement covered later.

## Background: attention, in words

The heart of the transformer is **self-attention**. For each position in the sequence, the
model produces three vectors: a **query** (what this position is looking for), a **key** (what
this position offers), and a **value** (the information it carries). To update a position, the
model compares its query against every position's key (a dot product), scales the result, and
turns the scores into weights with a softmax. Those weights then mix the values: each position
becomes a weighted blend of the information at every position, attending most to the ones it
matches. In one line: `attention = softmax(QK^T / sqrt(d_k)) V`. The `sqrt(d_k)` scaling keeps
the dot products from growing too large and saturating the softmax.

**Multi-head attention** runs several of these in parallel, each with its own learned
query/key/value projections, so the model can attend to different kinds of relationships at
once (one head might track syntax, another long-range references). Their outputs are
concatenated and projected back.

A **transformer block** wraps attention with the machinery from earlier modules: a
position-wise feed-forward network (an MLP applied to each position), **residual connections**
around each sub-layer (so gradients flow, as in ResNet), and **layer normalization**. Stack
several blocks, add an embedding for tokens and a **positional encoding** (since attention
itself is order-blind, the model needs to be told position), and end with a linear layer that
predicts the next token. That decoder-only stack, trained to predict the next token, is a GPT.

## Knowledge check

1. Define query, key, and value, and describe how they combine to produce one position's
   attention output.
2. Why is the dot product scaled by `sqrt(d_k)` before the softmax?
3. What does multi-head attention add over single-head attention?
4. What are the components of a transformer block, and what does each contribute?
5. Why does a transformer need positional encodings when a recurrent network does not?
6. In a decoder-only GPT, what is the training objective, and why is a causal mask needed?

## Build gate

Build a working transformer and train it to generate text.

**Specification:**

- **Self-attention** from scratch (queries, keys, values, scaled dot-product, causal mask),
  then **multi-head attention**.
- A full **transformer block**: multi-head attention with a residual connection and layer
  norm, then a position-wise feed-forward network with its own residual and layer norm.
- A complete **decoder-only model**: token embeddings, positional encodings, a stack of blocks,
  and a final projection to vocabulary logits.
- A training loop on a character-level dataset (Shakespeare is the standard), and a sampling
  function to generate text.

**Tests it must pass:**

- **Shapes and masking are correct.** Verify attention output shapes, and confirm the causal
  mask prevents a position from attending to future positions (a position's output must not
  change when later tokens change).
- **It learns.** The training loss falls substantially, and the model generates text that is
  locally coherent (real words, plausible structure) rather than random characters.
- **You can explain every line.** This is the gate behind the gate: you should be able to point
  at any line of your attention code and say what it computes and why.

## Project

Package your GPT and write it up (reuse your template):

- Your from-scratch transformer, trained, with generated samples and a loss curve.
- A short report explaining self-attention in your own words (with the `softmax(QK^T/sqrt(d))V`
  formula), the role of each part of a block, and one thing that was harder or more surprising
  than you expected.

**Definition of done:** a trained character-level GPT that generates coherent text, built from
your own attention implementation, with the write-up.

## The workshop: ship it

Build this in its own repository, `modelwright-nano-transformer`. This is a portfolio centerpiece;
make it clean.

1. Set up the project:

```bash
mkdir modelwright-nano-transformer && cd modelwright-nano-transformer
uv init && uv add torch numpy matplotlib && mkdir src
```

2. Implement attention, the block, and the model in `src/`, with the training and sampling
   scripts. (Training is much faster on a GPU; a free Colab or Kaggle notebook works, with the
   code committed here.)
3. Commit at checkpoints: "Self-attention with causal mask", then "Multi-head attention and
   block", then "Full GPT trains and generates text".
4. Add a README with generated samples, your loss curve, and your explanation of attention.
5. Ship it:

```bash
gh repo create modelwright-nano-transformer --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `modelwright-nano-transformer` is on GitHub, the model generates coherent text from
your own attention code, and the README explains how it works. You have built a GPT.

## Going deeper (optional)

- Follow Karpathy's "Let's reproduce GPT-2" (the `build-nanogpt` repo) to scale your toy up
  toward a real model.
  https://github.com/karpathy/build-nanogpt
- Add KV caching to make generation faster, and read about flash attention.
- Read "Attention Is All You Need" again and map every equation to a line of your code.

## Canonical references

- Vaswani et al. (2017), *Attention Is All You Need*.
- Karpathy, A. *Let's build GPT* and *nanoGPT*.
- Alammar, J. *The Illustrated Transformer*; Harvard NLP, *The Annotated Transformer*.
