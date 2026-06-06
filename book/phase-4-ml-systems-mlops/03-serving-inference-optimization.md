# Module 4.3 — Serving & Inference Optimization

**Mode:** hybrid
**Gate:** build gate
**Est. effort:** 10–14 focused hours

> A trained model sitting in a notebook helps no one. To be useful it must answer requests:
> behind an API, fast enough, reliably, and cheaply. This module is about that last mile, turning
> a model into a service. You will put a model behind an endpoint, then make it faster with the
> standard inference-optimization techniques, measuring every step.

## Why this matters

Serving is where ML meets software engineering and where a model's real-world cost and usability
are decided. Training happens once; inference happens millions of times, so its latency and cost
dominate the economics of a deployed model. A model that is wonderful but takes two seconds to
respond, or costs too much per call, will not ship. The techniques here, batching, quantization,
distillation, and good serving architecture, routinely make inference an order of magnitude
faster or cheaper without meaningfully hurting quality.

This is a build-gate module because serving is a doing skill. You will write a real service, load
test it, and optimize it with numbers in hand. The result is one of the most directly
employable artifacts in the curriculum: a deployed, optimized, measured model service.

## What you will be able to do

By the end of this module you will be able to:

- Serve a model behind a REST API and reason about latency versus throughput.
- Apply dynamic batching to raise throughput under load.
- Quantize a model and measure the speed/size/accuracy trade-off.
- Explain distillation and when to reach for it, and know the specialized serving stack for LLMs.

## Prerequisites

- [Module 2.4](../phase-2-deep-learning/04-pytorch-in-depth.md): you can build and run models in
  PyTorch.
- Comfort with HTTP/REST basics (request/response, JSON).

## The tools: what you will use and how they fit together

Serving has a small, layered toolkit:

- **A web framework: FastAPI.** Wraps your model's `predict` in an HTTP endpoint. Simple, fast,
  and the standard for Python ML services.
- **A serving framework (optional but real): BentoML or TorchServe.** Adds production features
  on top, packaging, versioning, and built-in dynamic batching, so you do not hand-roll them.
- **An optimization/runtime layer: ONNX Runtime or PyTorch's own tools.** Convert and run the
  model in an optimized runtime; apply quantization here.
- **For LLMs specifically: vLLM.** A high-throughput LLM serving engine with optimizations
  (paged attention, continuous batching) that general frameworks lack. Use it when serving a
  language model.
- **A load-testing tool: `locust` or `wrk`.** Generates concurrent requests so you can measure
  latency and throughput honestly, rather than guessing.

The flow: wrap the model with FastAPI (or BentoML), optimize it (quantize, batch), and load test
to measure the gains. For an LLM, reach for vLLM instead of a hand-built server.

## Curated path

1. **Serve a model: FastAPI for ML.** Read the FastAPI tutorial enough to build an endpoint, and
   a short guide to serving a model with it. Made With ML's serving lessons show this in a real
   project structure.
   https://fastapi.tiangolo.com/tutorial/ and https://madewithml.com/courses/mlops/

2. **Inference optimization concepts: Chip Huyen, *Designing Machine Learning Systems*, the
   model-deployment and inference chapters.** Latency vs throughput, batching, quantization,
   distillation, and edge vs cloud, framed as system decisions.
   https://github.com/chiphuyen/dmls-book

3. **Quantization, hands-on: the PyTorch quantization tutorials and ONNX Runtime.** Apply
   post-training quantization and measure the trade-off.
   https://pytorch.org/docs/stable/quantization.html and https://onnxruntime.ai/docs/

4. **LLM serving: vLLM.** Read the vLLM overview to understand why LLMs need a specialized serving
   engine (continuous batching, paged attention) and how throughput is achieved.
   https://docs.vllm.ai/

**Deliberately skip for now:** Kubernetes autoscaling and multi-region deployment. Build one solid
service and optimize it; orchestration at scale is a specialization.

## Background: the ideas in words

**Latency versus throughput.** **Latency** is how long one request takes; **throughput** is how
many requests you serve per second. They trade off: waiting to **batch** several requests
together raises throughput (the GPU is used efficiently) but adds a little latency to each. The
right balance depends on the application, a chatbot cares about latency, a nightly scoring job
cares about throughput.

**Batching.** GPUs are far more efficient processing many inputs at once than one at a time.
**Dynamic batching** collects incoming requests for a few milliseconds and runs them together,
dramatically increasing throughput under load. Serving frameworks provide this; it is one of the
highest-leverage optimizations.

**Quantization.** Running a model in lower precision (8-bit integers instead of 32-bit floats, or
4-bit for LLMs) shrinks it and speeds up inference, often with negligible accuracy loss. This is
one of the most effective ways to make models cheaper and faster to serve, and it is why large
models can run on modest hardware.

**Distillation.** Train a small "student" model to mimic a large "teacher." The student is much
cheaper to serve and can retain most of the teacher's quality on the target task. Reach for it
when you need the quality of a big model at the cost of a small one.

**LLM serving is special.** Generating text token by token, with variable-length outputs and
huge models, breaks naive serving. Engines like **vLLM** add continuous batching (start new
requests as others finish) and paged attention (efficient memory for the attention cache),
giving order-of-magnitude throughput gains. Know to reach for these rather than hand-rolling.

## Knowledge check

1. Define latency and throughput, and explain how batching trades one for the other.
2. Why are GPUs more efficient with batched inputs, and what does dynamic batching do?
3. What is quantization, and what trade-off does it make? Why is it so impactful for serving?
4. What is distillation, and when would you use it?
5. Why do LLMs need a specialized serving engine like vLLM rather than a plain FastAPI loop?
6. For a real-time recommendation API versus a nightly batch-scoring job, which matters more,
   latency or throughput, and how would you serve each differently?

## Build gate

Serve a model, optimize it, and prove the gains with measurements.

**Specification:**

- Take a model from an earlier module (your Module 2.6 CNN or a small classifier) and serve it
  behind a **FastAPI** endpoint that accepts input and returns predictions.
- **Load test** it with `locust` (or similar) and record baseline latency (p50, p95) and
  throughput under concurrency.
- Apply at least two optimizations and measure each: **dynamic batching** (via a serving
  framework or your own micro-batching) and **quantization** (8-bit). Record the new latency,
  throughput, model size, and any accuracy change.
- Summarize the trade-offs in a table.

**What it must show:**

- A working endpoint a stranger could call.
- Before-and-after measurements demonstrating a real improvement from your optimizations.
- An honest note on any accuracy cost.

## Project

Package the service and write it up (reuse your template):

- The FastAPI service, the load-test setup, and the optimization code.
- A report: your latency/throughput/size/accuracy table before and after each optimization, and
  a paragraph on which optimization you would ship and why.

**Definition of done:** a deployed, load-tested, optimized model service with measured gains and
the write-up.

## The workshop: ship it

Build this in its own repository, `mlforge-model-serving`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir mlforge-model-serving && cd mlforge-model-serving
uv init && uv add fastapi uvicorn torch onnxruntime locust && mkdir src tests
```

2. Write the FastAPI service in `src/`, a `locustfile.py` for load testing, and the optimization
   (quantization, batching) code.
3. Commit at checkpoints: "FastAPI endpoint serving the model", then "Load-test baseline", then
   "Quantization and batching with measured gains".
4. Add a README with run instructions, your trade-off table, and which optimization you would
   ship.
5. Ship it:

```bash
gh repo create mlforge-model-serving --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `mlforge-model-serving` is on GitHub, the endpoint runs, and the README shows
measured latency/throughput improvements from your optimizations.

## Going deeper (optional)

- Containerize the service with Docker and document how it would deploy.
- Serve a small LLM with vLLM and compare its throughput to a naive loop.
- Implement knowledge distillation: train a small student from one of your larger models.

## Canonical references

- FastAPI documentation; Made With ML (serving lessons).
- Huyen, C. *Designing Machine Learning Systems*, deployment and inference chapters.
- PyTorch quantization docs; ONNX Runtime; vLLM documentation.
