# Module 4.5 — ML System Design

**Mode:** hybrid
**Gate:** ship gate
**Est. effort:** 10–14 focused hours

> This module ties the whole phase together. You can now build each piece of an ML system; here
> you learn to design the whole thing, starting from a vague business problem and ending with a
> defensible architecture and a clear-eyed account of its trade-offs. This is both the most
> common senior-level interview and the daily work of a real ML engineer.

## Why this matters

Knowing the components is not the same as knowing how to assemble them for a specific problem
under real constraints. ML system design is the skill of going from "we want to reduce fraud" or
"we want better recommendations" to a concrete system: how to frame it as an ML problem, what
data and features, which model, how to serve it within the latency budget, how to evaluate it
online, and what to monitor. It forces every decision to be justified by the system's actual
goals.

This is the capstone skill of the systems phase, and it is what ML system design interviews
test, because it reveals whether you understand how the pieces fit and where the hard trade-offs
are. Practicing it makes you noticeably more senior, because you stop thinking "what model?" and
start thinking "what system, and why?"

## What you will be able to do

By the end of this module you will be able to:

- Take a vague business goal and frame it as a well-posed ML problem with clear objectives and
  metrics.
- Reason through the full stack of an ML system: data, features, model, serving, evaluation,
  monitoring.
- Make and defend architectural trade-offs under constraints (latency, cost, data, scale).
- Write a clear ML system design document and talk through it.

## Prerequisites

- The rest of Phase 4 (data, scale, serving, lifecycle) and the modeling phases. This module
  composes everything.

## Curated path

1. **Chip Huyen, *Designing Machine Learning Systems*, read as a whole with the framework in
   mind.** The book is organized around exactly this skill: each decision (objectives, data,
   features, model, deployment, monitoring) considered in the context of the system's goals. The
   introduction and the case studies are especially valuable here.
   https://github.com/chiphuyen/dmls-book

2. **A structured framework for ML system design interviews.** Read a solid, current walkthrough
   of the standard structure (clarify requirements, frame as ML, data and features, model,
   serving, evaluation, monitoring) so you have a repeatable template. Alex Xu's
   *Machine Learning System Design Interview* (Aminian & Xu) is the common reference; free
   alternatives include the ML system design write-ups collected by the community.
   https://github.com/chiphuyen/machine-learning-systems-design

3. **Worked examples: Eugene Yan's essays and applied-ML case studies.** Real, detailed accounts
   of designing recommendation, search, and other ML systems in industry. Excellent models for
   how to reason and how to write a design.
   https://eugeneyan.com/writing/

4. **The roadmap.sh ML system design references and real engineering blogs.** Read a couple of
   actual system write-ups (recommendations, fraud, search) from company engineering blogs to see
   the patterns and constraints in the wild.

**Deliberately skip for now:** memorizing specific company architectures. The goal is a
repeatable design *method*, not trivia.

## Background: the method, in words

A good ML system design follows a repeatable arc, and the skill is applying it under the specific
problem's constraints:

**Clarify and frame.** Start by pinning down the real goal and constraints: who uses this, what
does success mean as a business metric, what is the latency budget, the scale, the data
available? Then frame it as an ML problem: what exactly are you predicting, is it classification,
regression, ranking, retrieval, and what is the target?

**Data and features.** Where does training data come from, and how do you get labels (this is
often the hardest part)? What features, and can they be computed at serving time within budget
(recall training/serving skew from Module 4.1)?

**Model.** Choose a model class appropriate to the problem, data size, and latency budget, often
starting with a simple baseline (a strong default from Phase 1) before anything fancy. Justify
the choice by the constraints, not by novelty.

**Serving and scale.** How is it served (real-time API, batch, on-device), and does it meet the
latency and throughput requirements (Modules 4.3 and 4.2)? What are the cost implications?

**Evaluation and monitoring.** How do you measure success offline and, crucially, online (A/B
tests, business metrics)? What do you monitor, and how do you handle drift and retraining
(Module 4.4)?

The recurring theme: every choice is justified by the system's goals and constraints. A
"better" model that blows the latency budget or cannot get labels is the wrong choice. Designing
well means holding the whole system in view and making honest trade-offs.

## Knowledge check

1. What questions do you ask first to turn a vague business goal into a well-posed ML problem?
2. Why is obtaining labels often the hardest part of a real ML system, and how might you get them?
3. How does the serving latency budget constrain your choice of model and features?
4. Why start with a simple baseline model in a system design?
5. What is the difference between offline and online evaluation, and why do you need both?
6. Walk through designing a system for one of: recommendations, fraud detection, or search
   ranking, hitting each stage of the method.

## Project (ship gate)

**Write two full ML system design documents.**

Choose two different problems (for example: a recommendation system, a fraud detector, a search
ranker, an ETA predictor, or a content-moderation system). For each, write a complete design
document that walks the full method:

- Requirements and constraints (users, business metric, latency, scale).
- ML framing (what you predict, problem type, target, how you get labels).
- Data and features (sources, the label strategy, serving-time feasibility).
- Model choice with a justified baseline-first approach.
- Serving architecture (real-time vs batch, how it meets the latency/throughput budget).
- Evaluation (offline metrics and an online A/B plan) and monitoring (what you watch, drift,
  retraining).
- Explicit trade-offs and risks: what you chose not to do, and why.

A simple architecture diagram for each system strengthens it considerably.

**Definition of done:** two coherent, defensible design documents that another engineer could
read and critique, each covering the full stack with honest trade-offs.

## The workshop: ship it

Build this in its own repository, `mlforge-ml-system-design`. This is a writing-and-architecture
deliverable, and a strong portfolio and interview-prep artifact.

1. Set up the repo:

```bash
mkdir mlforge-ml-system-design && cd mlforge-ml-system-design && git init
mkdir designs diagrams
```

2. Write each design as a Markdown document in `designs/` (for example
   `designs/recommender.md` and `designs/fraud-detection.md`), with diagrams (hand-drawn and
   photographed is fine, or a tool like Excalidraw) in `diagrams/`.
3. Commit at checkpoints: one commit per completed design.
4. Add a README that lists the designs and the method you followed.
5. Ship it:

```bash
gh repo create mlforge-ml-system-design --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `mlforge-ml-system-design` is on GitHub with two complete, defensible design
documents covering the full ML system stack with explicit trade-offs.

## Going deeper (optional)

- Do a mock ML system design interview with a peer, talking through a design aloud under time.
- Read several company engineering blogs on the same problem (e.g., recommendations) and compare
  their choices.
- Pick one of your designs and actually build a thin end-to-end slice of it (this naturally
  becomes the phase capstone).

## Canonical references

- Huyen, C. *Designing Machine Learning Systems* (and the *machine-learning-systems-design*
  primer).
- Aminian & Xu, *Machine Learning System Design Interview*.
- Yan, E. *Applied ML writing* (eugeneyan.com).
