# Phase 4 — ML Systems & MLOps

Everything so far has produced models that work in a notebook. This phase is about the gap
between that and a system that serves real users reliably, day after day, as data shifts and
things break. This gap is enormous, and crossing it is precisely what the word "engineer" in
"machine learning engineer" refers to. It is also where most of the actual job is: a model is a
small part of a real ML system, and the parts around it (data, serving, monitoring,
deployment) are where projects succeed or fail.

This is the most "industry" of all the phases. Where earlier phases made you understand models
deeply, this one makes you able to ship them. You will learn how data flows into ML systems at
scale, how training works beyond a single GPU, how to serve a model fast and reliably, how to
operate it in production with monitoring and continuous delivery, and how to design whole
systems under real constraints. Much of this is invisible in courses and papers and learned
only on the job; this phase brings it forward.

A note on the material: this area is tool-heavy and the tools change, so the curriculum leans on
two durable spines, Chip Huyen's *Designing Machine Learning Systems* for the concepts and
Goku Mohandas's *Made With ML* for hands-on MLOps, and teaches the specific tools inline. Some
topics here (training across many GPUs, serving at massive scale) you will understand
conceptually and practice in miniature, because the full version needs infrastructure you may
not have. The judgment transfers; the scale is a detail.

## Modules (work them in order)

- **4.1 — Data Engineering for ML** *(curated breadth · concept check)*
  Where ML systems really begin: data sources and databases, pipelines, feature stores, data
  versioning, and why data problems cause most ML failures.
- **4.2 — Training at Scale** *(hybrid · concept check)*
  GPUs, mixed precision, distributed training, and the economics of large training runs.
- **4.3 — Serving & Inference Optimization** *(hybrid · build gate)*
  Get a model behind a fast, reliable endpoint: latency, throughput, batching, quantization,
  and distillation.
- **4.4 — The Production Lifecycle** *(curated breadth · ship gate)*
  Continuous integration and deployment for models, model registries, monitoring, drift
  detection, and A/B testing.
- **4.5 — ML System Design** *(hybrid · ship gate)*
  Both the interview and the job: design a recommender, a fraud detector, a search ranker, with
  defensible architecture and honest trade-offs.

## Phase capstone

**Capstone 4:** deploy a production-grade ML service end to end, with a trained model, optimized
inference behind an API, monitoring and drift detection, and a system-design document. Ship it
to its own repository, `mlforge-phase-4-capstone`. This is the project that proves you can take
a model all the way to a running, observable service, which is the core of the job.

Begin with [Module 4.1](./01-data-engineering-for-ml.md).
