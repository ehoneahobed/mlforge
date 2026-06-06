# The Curriculum Map

This is the full territory: six phases that take you from foundations to original,
research-level work. Each phase ends in a portfolio-grade **capstone**. Each module
declares its mode (curated breadth, from-scratch deep-dive, or hybrid) and its gate
(concept check, build gate, or ship gate).

The ordering is deliberate. Later modules assume you arrive having passed the earlier
gates. You are welcome to look ahead, but the gates are written as though the foundations
beneath them are solid.

Mathematics is taught **just-in-time**. There is no separate math phase. Each result is
introduced at the moment a model needs it, motivated, and then used. Phase 0 includes a
math reference module and a foundational track for anyone whose math is rusty.

A note on honesty about time: done properly, this is a multi-year journey. That is the
correct timescale for reaching the level of someone with both a research training and real
industry experience. The curriculum is walked one module at a time.

---

## Phase 0 — Foundations & The Autograd Engine

*Goal: establish ML-grade tooling and demystify the single abstraction everything else
stands on, automatic differentiation.*

- **0.1 — From NumPy to a Working Autograd Engine** *(from-scratch deep-dive · build gate)*
  Vectorized NumPy, then build a scalar reverse-mode autodiff engine from scratch and use
  it to train a small neural network. The mechanism behind every deep learning framework.
- **0.2 — The ML Engineer's Toolkit** *(curated breadth · ship gate)*
  Reproducible environments, experiment tracking, profiling, and disciplined use of the
  scientific Python stack. Produces a reusable project template.
- **0.3 — Math, Just-in-Time: The Reference Layer** *(hybrid · concept check)*
  A curated, indexed map of the specific linear algebra, calculus, probability, and
  optimization results the curriculum invokes, plus a foundational track for anyone who
  needs to build the math up first.

**Capstone 0:** a packaged, tested autograd engine whose gradients match a reference
framework, accompanied by a written derivation of the backward pass of every operation it
supports.

---

## Phase 1 — Classical Machine Learning

*Goal: the statistical core of the field. A large share of machine learning in production
is still classical. You will derive, implement, and correctly evaluate the foundational
models, and learn the failure modes that separate practitioners from tutorial-followers.*

- **1.1 — The Learning Problem** *(hybrid · concept check)*
  Supervised versus unsupervised learning, the bias-variance decomposition,
  generalization, overfitting, the train/validation/test discipline, and why
  cross-validation exists.
- **1.2 — Linear & Logistic Regression from First Principles** *(from-scratch · build gate)*
  Derive least squares and the logistic loss, implement both with your own gradient
  descent, and verify against a reference library. Regularization and what it does
  geometrically.
- **1.3 — Trees, Forests, and Gradient Boosting** *(hybrid · build gate)*
  Decision trees from scratch, then ensembles. Why gradient boosting dominates tabular
  data. Implement a minimal gradient-boosted tree.
- **1.4 — SVMs & Kernels** *(curated breadth · concept check)*
  Margins, the kernel trick, and where this still matters.
- **1.5 — Unsupervised: Clustering & Dimensionality Reduction** *(hybrid · build gate)*
  k-means and PCA from scratch, then modern visualization methods, and what each distorts.
- **1.6 — Evaluation, Calibration & Leakage** *(curated breadth · ship gate)*
  Metrics beyond accuracy, probability calibration, the subtle ways data leakage inflates
  results, and how to build an evaluation you can trust.
- **1.7 — Feature Engineering & The Data Reality** *(curated breadth · concept check)*
  Real data is messy. Encoding, scaling, missing values, class imbalance, and why feature
  work often beats model choice.

**Capstone 1:** an end-to-end classical ML project on a real, messy tabular dataset: an
honest baseline, a leak-free pipeline, properly tuned gradient boosting, calibrated
probabilities, and a written report.

---

## Phase 2 — Deep Learning

*Goal: neural networks earned from the ground up. Build the mechanisms, master the
framework, learn the architectures, then learn the craft of actually making training
work.*

- **2.1 — Neural Nets & Backprop from Scratch** *(from-scratch · build gate)*
  Extend Phase 0's engine to tensors. Implement a multilayer perceptron, backpropagation,
  and stochastic gradient descent in NumPy, and train it before any framework is involved.
- **2.2 — PyTorch Mastery** *(curated breadth · build gate)*
  Tensors, autograd, modules, data loaders, and a correct training loop. Reimplement the
  from-scratch network in PyTorch and confirm identical behavior.
- **2.3 — Making Training Work** *(hybrid · ship gate)*
  Initialization, normalization, optimizers, learning-rate schedules, regularization, and
  a systematic methodology for diagnosing a model that will not learn.
- **2.4 — Convolutional Networks & Vision** *(curated breadth · build gate)*
  Convolution from scratch, then modern architectures, transfer learning, and a real
  computer vision task.
- **2.5 — Sequence Models** *(curated breadth · concept check)*
  Recurrent networks and their limits, and the motivation that produced attention.
- **2.6 — The Transformer from Scratch** *(from-scratch · build gate)*
  Build self-attention, multi-head attention, and a complete transformer, then train a
  small language model. The gateway to everything modern.

**Capstone 2:** reproduce a published result. Take a small model or a compact training run,
reproduce it from scratch, and write up where your numbers matched, where they did not, and
why.

---

## Phase 3 — Modern ML: LLMs & Generative Models

*Goal: the frontier. How today's systems are actually built, adapted, evaluated, and
applied.*

- **3.1 — Inside Large Language Models** *(hybrid · concept check)*
  Tokenization, pretraining objectives, scaling laws, emergent behavior, and the
  decoder-only recipe at scale.
- **3.2 — Fine-tuning & Adaptation** *(hybrid · build gate)*
  Full fine-tuning versus parameter-efficient methods, instruction tuning, and the
  alignment family. Fine-tune an open model and measure the improvement.
- **3.3 — Embeddings, Retrieval & RAG** *(curated breadth · ship gate)*
  Embeddings, vector search, and building a retrieval-augmented system that genuinely
  grounds its answers, including where the approach quietly fails.
- **3.4 — Diffusion & Generative Models** *(hybrid · build gate)*
  The diffusion process from the mathematics up, then a small working image generator.
- **3.5 — Evaluating Generative Systems** *(curated breadth · ship gate)*
  The hardest problem in modern ML: knowing whether a system is actually good. Benchmarks
  and their flaws, model-graded evaluation, and building task-specific evaluations.

**Capstone 3:** build, fine-tune, and rigorously evaluate an end-to-end LLM application,
with an evaluation suite you designed and can defend.

---

## Phase 4 — ML Systems & MLOps

*Goal: the gap between a notebook that works and a system that serves real users reliably.
This is what industry experience actually means.*

- **4.1 — Data Engineering for ML** *(curated breadth · concept check)*
  Pipelines, feature stores, data versioning, and why data problems cause most ML
  failures.
- **4.2 — Training at Scale** *(hybrid · concept check)*
  GPUs, mixed precision, distributed training, and the economics of large training runs.
- **4.3 — Serving & Inference Optimization** *(hybrid · build gate)*
  Latency, throughput, batching, quantization, distillation, and getting a model behind a
  fast, reliable endpoint.
- **4.4 — The Production Lifecycle** *(curated breadth · ship gate)*
  Continuous integration and deployment for models, model registries, monitoring, drift
  detection, and A/B testing.
- **4.5 — ML System Design** *(hybrid · ship gate)*
  Both the interview and the job: design a recommender, a fraud detector, a search ranker.
  Requirements, trade-offs, and defensible architecture.

**Capstone 4:** deploy a production-grade ML service end to end, with a trained model,
optimized inference, monitoring, drift detection, and a system-design document.

---

## Phase 5 — Specialization & Research Maturity

*Goal: cross from knowing the field to extending it. Read the literature fluently,
reproduce papers, and make an original contribution.*

- **5.1 — Reading & Reproducing Papers** *(hybrid · ship gate)*
  How to read a paper for real, a canonical reading list, and reproducing a non-trivial
  result from the paper alone.
- **5.2 — Reinforcement Learning** *(hybrid · build gate)*
  Markov decision processes, value and policy methods, policy gradients, and a working
  agent. RL's growing role in alignment and agents.
- **5.3 — A Chosen Specialization** *(self-directed · ship gate)*
  Go deep in one track: natural language processing, computer vision, generative models,
  reinforcement learning, or ML systems.
- **5.4 — Original Work** *(self-directed · ship gate)*
  A research-quality project or a meaningful open-source contribution: a novel experiment,
  a careful reproduction and extension, or a tool the community uses.

**Capstone 5:** a portfolio-defining piece of original work, written up to a standard you
would be proud to present to a hiring committee or a paper reviewer.

---

## Threads that run through every phase

- **The from-scratch ladder:** autograd, then a multilayer perceptron, then a convolutional
  network, then a transformer, then a diffusion model. Each rung proves you own the
  mechanism, not just the interface.
- **The reading log:** every canonical paper a module touches, accumulating into real
  command of the literature.
- **The portfolio:** every ship gate produces an artifact. By the end, the portfolio is the
  credential.
- **Math, just-in-time:** results appear exactly when a model needs them.

---

Modules are built out in order, each following the same structure. Begin with
[Phase 0](./phase-0-foundations/00-overview.md).
