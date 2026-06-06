# Module 4.4 — The Production Lifecycle

**Mode:** curated breadth
**Gate:** ship gate
**Est. effort:** 12–16 focused hours

> Shipping a model once is not the job. The job is operating it: deploying new versions safely,
> tracking which version is live, watching it in production, catching it when the world shifts
> under it, and deciding whether a new model is actually better. This module is the operational
> heart of MLOps, the discipline that keeps ML systems alive and trustworthy over time.

## Why this matters

Models decay. Not because the code changes, but because the world does: user behavior shifts,
upstream data changes, a once-rare case becomes common. A model that was excellent at launch can
quietly degrade with no error message at all. The production lifecycle is the set of practices
that make this visible and manageable: continuous integration and deployment so changes ship
safely, a model registry so you know what is running, monitoring and drift detection so you see
problems before users do, and A/B testing so "the new model is better" is a measured claim, not
a hope.

This is what "MLOps" really means, and it is a large part of what distinguishes an ML engineer
from someone who can train a model. It is also where ML and classic software operations meet, so
much of it will feel like good engineering applied to the peculiarities of models and data.

## What you will be able to do

By the end of this module you will be able to:

- Set up continuous integration and deployment for an ML project.
- Use a model registry to version and track models through stages.
- Monitor a deployed model and detect data and concept drift.
- Design an A/B test to decide whether a new model is genuinely better.

## Prerequisites

- [Module 4.3](./03-serving-inference-optimization.md): you can serve a model behind an API.
- [Module 1.6](../phase-1-classical-ml/06-evaluation-calibration-leakage.md): trustworthy
  evaluation, which underpins monitoring and A/B testing.

## The tools: what you will use and how they fit together

The lifecycle stitches several tools into a loop:

- **CI/CD: GitHub Actions.** Runs your tests and, on success, deploys, automatically, on every
  change. You already used Actions to deploy this book's site; here it deploys and validates a
  model service.
- **A model registry: MLflow.** Versions trained models and tracks their stage (staging,
  production), so you always know what is deployed and can roll back. This extends the experiment
  tracking from Module 0.2.
- **Monitoring and drift detection: Evidently.** Compares live data and predictions against a
  reference (your training data) to detect data drift, concept drift, and quality degradation,
  and produces dashboards and alerts.
- **A/B testing.** Not a single tool but a method: split traffic between the current and
  candidate model and measure the difference on a real metric.

The loop: a change triggers CI (tests), CD deploys it, the registry records the version,
monitoring watches it in production, drift detection flags when it degrades, and an A/B test
decides whether a retrained candidate should replace it. Then it repeats.

## Curated path

1. **Made With ML, the MLOps lessons (testing, CI/CD, monitoring).** The most hands-on,
   project-based treatment of the production lifecycle, building the pieces into a real
   repository. This is the spine and the basis of the workshop.
   https://madewithml.com/courses/mlops/

2. **Chip Huyen, *Designing Machine Learning Systems*, the deployment, monitoring, and
   continual-learning chapters.** The conceptual framework: types of drift, what to monitor, and
   how models fail in production.
   https://github.com/chiphuyen/dmls-book

3. **Monitoring and drift, hands-on: the Evidently documentation and tutorials.** How to compute
   drift, build a monitoring report, and set up alerts.
   https://docs.evidentlyai.com/

4. **Continuous delivery for ML, conceptually: Martin Fowler's "CD4ML."** The canonical article on
   what continuous delivery means when data and models, not just code, are part of what you ship.
   https://martinfowler.com/articles/cd4ml.html

**Deliberately skip for now:** full platform builds (Kubeflow, SageMaker pipelines end to end).
Understand the lifecycle and build a working slice of it; platform mastery is a specialization.

## Background: the ideas in words

**CI/CD for ML.** Continuous integration runs your tests automatically on every change; continuous
deployment ships passing changes to production. For ML, "tests" include not just code tests but
data validation and model-quality checks, and "what ships" includes the model and its data
dependencies, not just code. This is the safety net that lets you change a live system without
fear.

**The model registry.** With many trained models and versions, you need a source of truth for
what exists and what is deployed. A **registry** stores models with versions and stages
(staging, production, archived), so you can promote a model, see exactly what is live, and roll
back instantly if it misbehaves.

**Monitoring and drift.** Once a model is serving, you watch it. **Data drift** is when the
inputs change distribution (the model sees data unlike its training data). **Concept drift** is
when the relationship between inputs and the right answer changes (the same input should now
yield a different prediction). Both degrade a model silently, with no error, so you monitor input
distributions, prediction distributions, and, when labels arrive, live accuracy, and alert when
they move.

**A/B testing.** When you have a candidate model, you do not just swap it in and hope. You route
a fraction of traffic to it, measure both models on a real business metric, and switch only if
the candidate is significantly better. This is the same rigor as Phase 1 evaluation, applied
live, and it guards against the common trap of a model that looks better offline but is worse in
production.

## Knowledge check

1. What does CI/CD add for an ML project beyond what it adds for ordinary software?
2. What is a model registry for, and how does it enable safe rollbacks?
3. Distinguish data drift from concept drift, and give an example of each.
4. Why can a model degrade in production with no error being raised, and what do you monitor to
   catch it?
5. Why A/B test a new model instead of deploying it directly, and what do you measure?
6. Sketch the full production loop from a code change to a monitored, possibly-replaced model.

## Project (ship gate)

**Build the production lifecycle around a model service.**

Take your Module 4.3 model service (or build a small one) and wrap it in the operational loop:

- **CI/CD:** a GitHub Actions workflow that runs tests (including a basic data/model check) on
  every push and deploys on success.
- **Registry:** register the trained model with MLflow, with a version and stage.
- **Monitoring:** an Evidently-based report that detects data drift by comparing fresh data
  against a reference, with a clear signal when drift occurs (demonstrate it by feeding in
  deliberately shifted data).
- **A/B design:** a written A/B test plan for replacing the model, the metric, the traffic split,
  how long to run, and the decision rule.

**Definition of done:** a repository with working CI/CD, a registered model, a drift-detection
report that fires on shifted data, and a defensible A/B test plan.

## The workshop: ship it

Build this in its own repository, `mlforge-ml-lifecycle`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir mlforge-ml-lifecycle && cd mlforge-ml-lifecycle
uv init && uv add mlflow evidently scikit-learn fastapi pytest && mkdir src tests .github/workflows
```

2. Build a small model service in `src/`, register the model with MLflow, write an Evidently
   drift report, and add a GitHub Actions workflow in `.github/workflows/` that runs `pytest`.
3. Commit at checkpoints: "Model + registry", then "Drift monitoring fires on shifted data",
   then "CI workflow green", then "A/B test plan".
4. Add a README documenting the lifecycle and including your A/B plan.
5. Ship it:

```bash
gh repo create mlforge-ml-lifecycle --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `mlforge-ml-lifecycle` is on GitHub with passing CI, a registered model, a drift
report that detects shifted data, and a written A/B test plan.

## Going deeper (optional)

- Add automated retraining triggered by a drift alert (the continual-learning loop).
- Explore a full platform (MLflow end to end, or a managed service) on your project.
- Read about shadow deployments and canary releases as safer rollout strategies.

## Canonical references

- Made With ML (Goku Mohandas), MLOps lessons.
- Huyen, C. *Designing Machine Learning Systems*, deployment and monitoring chapters.
- Evidently documentation; Fowler, M. *Continuous Delivery for Machine Learning (CD4ML)*.
