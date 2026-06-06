# Phase 3 — Modern ML: LLMs & Generative Models

This is the frontier, and you arrive at it having earned the right to be here. In Phase 2 you
built a transformer and a working GPT from scratch, so the systems behind today's headlines
are no longer mysterious to you; they are your nano-GPT, scaled up and refined. Phase 3 is
about how that scaling actually happens, and how you build real things on top of it: adapting
large models to your needs, grounding them in your own data, generating images, and, hardest
of all, knowing whether any of it is actually good.

The phase keeps the curriculum's rhythm. You use each capability with mature tools first, then
go under the hood where it deepens mastery. The difference from earlier phases is the compute
reality: these models are large. You will work with small open models, parameter-efficient
techniques, and free GPU notebooks (Google Colab or Kaggle), so everything here is doable
without a cluster. The skills transfer directly to the large-scale versions.

A note on pace of change: this is the fastest-moving area in the field. The principles here
(how pretraining works, what fine-tuning and retrieval do, how to evaluate) are stable, but
specific tools and best models change month to month. The curated resources lean on the most
durable sources, and you should expect to supplement them with current documentation.

## Modules (work them in order)

- **3.1 — Inside Large Language Models** *(hybrid · concept check)*
  How the transformer you built becomes ChatGPT: tokenization, pretraining, scaling laws,
  emergence, and the decoder-only recipe at scale.
- **3.2 — Fine-tuning & Adaptation** *(hybrid · build gate)*
  Adapt a pretrained model to a task. Full fine-tuning versus LoRA and other
  parameter-efficient methods, instruction tuning, and the alignment family. Fine-tune an open
  model and measure the lift.
- **3.3 — Embeddings, Retrieval & RAG** *(curated · ship gate)*
  Ground a model in your own data. Embeddings, vector search, and a retrieval-augmented system
  that genuinely grounds its answers, including where the approach quietly fails.
- **3.4 — Diffusion & Generative Models** *(hybrid · build gate)*
  Use an image generator, then build the diffusion process from the mathematics up.
- **3.5 — Evaluating Generative Systems** *(curated · ship gate)*
  The hardest problem in modern ML: knowing whether a system is actually good. Benchmarks and
  their flaws, model-graded evaluation, and building task-specific evaluations.

## Phase capstone

**Capstone 3:** build, fine-tune, and rigorously evaluate an end-to-end LLM application, with
an evaluation suite you designed and can defend. Ship it to its own repository,
`mlforge-phase-3-capstone`. This is the project that demonstrates you can not only use modern
models but adapt them, ground them, and judge them honestly, which is exactly what the work
requires.

Begin with [Module 3.1](./01-inside-large-language-models.md).
