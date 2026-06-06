# The Curriculum Map

This is the full territory: six phases that take you from your first trained model to
original, research-level work. The order is deliberate and follows one principle above all:
**you experience a technique before you build it.** You use something with a library, see it
work, and get comfortable, and only then do you open it up and implement it from scratch.
The accessible entry and the expert ceiling are the same path, walked in the right sequence.

The structure mirrors the logical flow of the field (and the widely used
[roadmap.sh ML roadmap](https://roadmap.sh/machine-learning)): get oriented, set up your
tools, learn to handle data, use and then build the classical models, learn to evaluate
honestly, then go deep into deep learning, modern systems, and research.

Each phase ends in a portfolio-grade **capstone**. Each module declares its mode (hands-on,
curated breadth, from-scratch deep-dive, or hybrid) and its gate (concept check, build gate,
or ship gate). Mathematics is taught just-in-time, with a foundational track in Phase 0 for
anyone who needs it. There is no cold start: nothing asks you to build a mechanism you have
not already used.

A note on time: done properly, this is a multi-year journey, the correct timescale for
reaching the level of someone with both research training and real industry experience. You
walk it one module at a time.

---

## Phase 0 — Getting Started

*Goal: go from "curious about machine learning" to having trained, evaluated, and understood
real models, with a professional setup and the data skills to support it. Fully accessible.
No neural networks and no from-scratch internals here; this phase is about experiencing
machine learning and getting comfortable.*

- **0.1 — The Landscape of Machine Learning** *(orientation · concept check)*
  No code. What ML is, how AI, ML, deep learning, and LLMs nest, the types of learning, the
  end-to-end workflow, and what an ML engineer does. The map before the journey.
- **0.2 — The ML Engineer's Toolkit & Your First Model** *(curated breadth · ship gate)*
  Set up a reproducible environment and experiment tracking, learn the scientific Python
  stack, and train and track your first model with scikit-learn. Produces a reusable project
  template.
- **0.3 — Working with Data** *(hands-on · ship gate)*
  Where models actually begin. Loading data from files and formats (CSV, JSON, Parquet),
  exploring and cleaning it with pandas, handling missing values, and the train/validation/
  test split done correctly. The unglamorous skill behind every real project.
- **0.4 — Math, Just-in-Time: The Reference Layer** *(hybrid · concept check)*
  An indexed map of the math the curriculum uses, plus a foundational track for anyone
  building it up from the start. A reference you return to, not a wall to climb.

**Capstone 0:** a complete, beginner-friendly end-to-end project with scikit-learn. Take a
real dataset, explore and clean it, train and compare a couple of models, evaluate them
honestly, and write up what you found. Your first real result, start to finish, with no
internals required.

---

## Phase 1 — Classical Machine Learning

*Goal: the statistical core of the field, which still runs most of machine learning in
production. For each model you use it first to build intuition, then implement it from
scratch to truly understand it. You arrive having already trained models in Phase 0, so this
phase deepens comfort into mastery.*

- **1.1 — The Learning Problem** *(hybrid · concept check)*
  Generalization, the bias-variance trade-off, overfitting, and the train/validation/test
  discipline that everything depends on.
- **1.2 — Linear & Logistic Regression** *(from-scratch · build gate)*
  Use them, then derive and implement them with your own gradient descent. The two models
  that underpin the entire field, including the final layer of every neural network.
- **1.3 — Trees, Forests, and Gradient Boosting** *(hybrid · build gate)*
  Use the ensembles that dominate tabular data, then build a decision tree and a minimal
  gradient-boosted model from scratch.
- **1.4 — SVMs & Kernels** *(curated breadth · concept check)*
  Margins, the kernel trick, and where this elegant idea still earns its place.
- **1.5 — Unsupervised: Clustering & Dimensionality Reduction** *(hybrid · build gate)*
  Use clustering and PCA, then implement k-means and PCA from scratch. What each method
  reveals and what it distorts.
- **1.6 — Evaluation, Calibration & Leakage** *(curated breadth · ship gate)*
  The skill that separates practitioners from tutorial-followers: building an evaluation you
  can actually trust, and catching the leakage that quietly inflates results.
- **1.7 — Feature Engineering & The Data Reality** *(curated breadth · concept check)*
  Real data is messy, and feature work often beats model choice.

**Capstone 1:** an end-to-end classical ML project on real, messy data, with an honest
baseline, a leak-free pipeline, a well-tuned model, calibrated probabilities, and a written
report an interviewer would respect.

---

## Phase 2 — Deep Learning

*Goal: neural networks, experienced then earned. You train a network with a framework and
watch it work before you build the machinery underneath it. This is where the from-scratch
neural-network work lives, now that you know what a network is for.*

- **2.1 — Neural Networks, Hands-On** *(hands-on · ship gate)*
  Train your first neural network with PyTorch on a real dataset and watch it learn. See the
  pieces (layers, activations, a loss, an optimizer, a training loop) working together before
  you build any of them. The experience that motivates everything next.
- **2.2 — From NumPy to a Working Autograd Engine** *(from-scratch · build gate)*
  Now build the engine behind what you just used. A scalar reverse-mode automatic
  differentiation engine from scratch, the mechanism inside every deep learning framework.
- **2.3 — Neural Nets & Backprop from Scratch** *(from-scratch · build gate)*
  Extend the engine to tensors and implement a multilayer perceptron, backpropagation, and
  stochastic gradient descent in NumPy.
- **2.4 — PyTorch in Depth** *(curated breadth · build gate)*
  Go from using PyTorch to commanding it: tensors, autograd, modules, data loaders, and a
  correct training loop. Reimplement your from-scratch network and confirm it matches.
- **2.5 — Making Training Work** *(hybrid · ship gate)*
  Initialization, normalization, optimizers, schedules, regularization, and a systematic way
  to debug a model that will not learn.
- **2.6 — Convolutional Networks & Vision** *(curated breadth · build gate)*
  Use a modern vision model, then build convolution from scratch. Transfer learning and a
  real computer vision task.
- **2.7 — Sequence Models** *(curated breadth · concept check)*
  Recurrent networks and their limits, and the motivation that produced attention.
- **2.8 — The Transformer from Scratch** *(from-scratch · build gate)*
  Use a small transformer, then build self-attention, multi-head attention, and a complete
  transformer, and train a small language model. The gateway to everything modern.

**Capstone 2:** reproduce a published result, from scratch, and write up where your numbers
matched, where they did not, and why.

---

## Phase 3 — Modern ML: LLMs & Generative Models

*Goal: the frontier. How today's systems are built, adapted, evaluated, and applied. Use
each capability first, then go under the hood.*

- **3.1 — Inside Large Language Models** *(hybrid · concept check)*
  Tokenization, pretraining objectives, scaling laws, emergent behavior, and the
  decoder-only recipe at scale.
- **3.2 — Fine-tuning & Adaptation** *(hybrid · build gate)*
  Use a fine-tuning workflow, then go deep: full fine-tuning versus parameter-efficient
  methods, instruction tuning, and the alignment family. Fine-tune an open model and measure
  the lift.
- **3.3 — Embeddings, Retrieval & RAG** *(curated breadth · ship gate)*
  Build a retrieval-augmented system that genuinely grounds its answers, and learn where the
  approach quietly fails.
- **3.4 — Diffusion & Generative Models** *(hybrid · build gate)*
  Use an image generator, then build the diffusion process from the mathematics up.
- **3.5 — Evaluating Generative Systems** *(curated breadth · ship gate)*
  The hardest problem in modern ML: knowing whether a system is actually good. Benchmarks
  and their flaws, model-graded evaluation, and building task-specific evaluations.

**Capstone 3:** build, fine-tune, and rigorously evaluate an end-to-end LLM application,
with an evaluation suite you designed and can defend.

---

## Phase 4 — ML Systems & MLOps

*Goal: the gap between a notebook that works and a system that serves real users reliably.
This is what industry experience actually means, and it absorbs roadmap.sh's data
engineering, data sources, and deployment topics.*

- **4.1 — Data Engineering for ML** *(curated breadth · concept check)*
  Data sources and databases (SQL and NoSQL), pipelines, feature stores, data versioning,
  and why data problems cause most ML failures.
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

*Goal: cross from knowing the field to extending it. Read the literature fluently, reproduce
papers, and make an original contribution.*

- **5.1 — Reading & Reproducing Papers** *(hybrid · ship gate)*
  How to read a paper for real, a canonical reading list, and reproducing a non-trivial
  result from the paper alone.
- **5.2 — Reinforcement Learning** *(hybrid · build gate)*
  Use a reinforcement learning library, then build the ideas: Markov decision processes,
  value and policy methods, policy gradients, and a working agent. RL's growing role in
  alignment and agents.
- **5.3 — A Chosen Specialization** *(self-directed · ship gate)*
  Go deep in one track: natural language processing, computer vision, generative models,
  reinforcement learning, or ML systems.
- **5.4 — Original Work** *(self-directed · ship gate)*
  A research-quality project or a meaningful open-source contribution.

**Capstone 5:** a portfolio-defining piece of original work, written up to a standard you
would be proud to present to a hiring committee or a paper reviewer.

---

## Threads that run through every phase

- **Experience then build:** every major technique is used with a library first, then
  implemented from scratch where that deepens mastery.
- **The from-scratch ladder:** the autograd engine, then a multilayer perceptron, then a
  convolutional network, then a transformer, then a diffusion model. Each rung proves you
  own the mechanism.
- **The reading log:** every canonical paper a module touches, accumulating into real
  command of the literature.
- **The portfolio:** every ship gate produces an artifact. By the end, the portfolio is the
  credential.
- **Math, just-in-time:** results appear exactly when a model needs them.

---

## How MLForge maps to the roadmap.sh ML roadmap

MLForge is cross-checked against the
[roadmap.sh Machine Learning roadmap](https://roadmap.sh/machine-learning) so its coverage
is at least as complete, while going deeper through the from-scratch gates and the research
phase. The mapping, roughly:

- **Introduction, what ML is, types of ML, the workflow** map to Phase 0, Module 0.1.
- **Programming and the scientific stack** (Python, NumPy, pandas, Matplotlib) are assumed
  as a prerequisite and reinforced in Module 0.2. Absolute beginners to Python should learn
  it first.
- **Data collection, data formats, and data cleaning** are Module 0.3, with the heavier data
  engineering (SQL and NoSQL databases, pipelines, feature stores) in Phase 4.
- **Mathematical foundations** (calculus, linear algebra, probability, statistics) are the
  reference layer in Module 0.4, taught just-in-time across the phases.
- **Scikit-learn, supervised and unsupervised learning, model evaluation** are Phases 0 and 1.
- **Deep learning** (neural network basics, CNNs, RNNs, attention, transformers) is Phase 2;
  **autoencoders, GANs, and diffusion** appear in Phases 2 and 3.
- **NLP, LLMs, and generative models** are Phase 3.
- **MLOps and deployment** are Phase 4.
- **Reinforcement learning** is Phase 5, Module 5.2.

A few roadmap.sh items (semi-supervised learning, explainable AI, specific NLP preprocessing
steps) are introduced where they naturally arise rather than as standalone modules. These
are noted here so the coverage is transparent and nothing is lost by omission.

---

Modules are built out in order, each following the same structure. Begin with
[Phase 0](./phase-0-foundations/00-overview.md).
