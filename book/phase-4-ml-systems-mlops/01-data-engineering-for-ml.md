# Module 4.1 — Data Engineering for ML

**Mode:** curated breadth
**Gate:** concept check
**Est. effort:** 8–12 focused hours

> In Phase 0 you learned to load and clean a CSV. Real ML systems do not run on CSVs; they run
> on data flowing from databases, APIs, and event streams, transformed by pipelines, served by
> feature stores, and versioned so results are reproducible. This module is where data stops
> being a file and becomes infrastructure.

## Why this matters

Ask experienced ML engineers what causes most production failures and few will say "the model."
They will say the data: a pipeline that silently broke, a feature computed differently in
training than in serving, a schema that changed upstream, data that drifted. The model is a
small, well-understood piece; the data systems around it are large and fragile, and they are
where reliability is won or lost.

This module gives you the vocabulary and the mental model of data engineering for ML: where data
comes from, how it is moved and transformed, how features are managed so training and serving
agree, and how data is versioned so an experiment can be reproduced. You do not need to become a
data engineer, but you must be able to reason about and build the data side of an ML system,
because that is where you will spend a surprising amount of your time.

## What you will be able to do

By the end of this module you will be able to:

- Describe common data sources for ML and the basics of querying a database with SQL.
- Explain what a data pipeline does and the difference between batch and streaming.
- Explain what a feature store solves, especially training/serving skew.
- Version data and pipelines so an ML result is reproducible.

## Prerequisites

- [Module 0.3](../phase-0-foundations/03-working-with-data.md): you can load, explore, and clean
  tabular data. This module scales that up to systems.

## Curated path

1. **Chip Huyen, *Designing Machine Learning Systems*, the data chapters (3 and 4).** The
   definitive treatment of data sources, data formats, data models, the training-data and
   feature-engineering pipeline, and the failure modes. This is the conceptual spine.
   https://github.com/chiphuyen/dmls-book (book resources and summaries)

2. **SQL for data work.** If your SQL is thin, work a practical tutorial until you can comfortably
   `SELECT`, `JOIN`, `GROUP BY`, and aggregate. Mode's SQL tutorial is a clear, free option.
   https://mode.com/sql-tutorial/

3. **Made With ML, the data and preprocessing lessons.** Hands-on treatment of preparing and
   versioning data in a real project structure, which feeds directly into the workshop.
   https://madewithml.com/courses/mlops/

4. **Feature stores and data versioning, conceptually.** Read the Feast introduction (an
   open-source feature store) for what a feature store is and the training/serving-skew problem
   it solves, and the DVC introduction for data and pipeline versioning.
   https://docs.feast.dev/ and https://dvc.org/doc

**Deliberately skip for now:** standing up Spark or a full data warehouse, and streaming systems
like Kafka in depth. Understand the concepts and build a small batch pipeline; the heavy
infrastructure is a specialization.

## Background: the ideas in words

**Data sources.** ML data comes from many places: relational databases (queried with SQL),
NoSQL stores, third-party APIs, files, and event streams from applications. A real first step in
any project is getting reliable access to this data, which is why SQL is a baseline skill.

**Pipelines.** A data pipeline moves and transforms data from source to a form a model can use:
extract, transform, load (ETL), or the modern ELT variant. Pipelines run in **batch** (process a
chunk on a schedule) or **streaming** (process events as they arrive). Most ML training is batch;
some serving needs fresh streaming features.

**Feature stores and training/serving skew.** A subtle, costly bug: a feature is computed one way
in the training pipeline and a slightly different way in the serving code, so the model sees
different inputs in production than it trained on, and quietly underperforms. A **feature store**
solves this by computing features once and serving the same definitions to both training and
inference. Even without a dedicated tool, the principle, compute features once, share the
definition, is essential.

**Data and pipeline versioning.** Code versioning (Git) is not enough for ML, because results
depend on data too. **Data versioning** (DVC and similar) tracks which version of the data
produced which result, so an experiment is reproducible. This extends the reproducibility
discipline from Module 0.2 to the data itself.

## Knowledge check

1. Name three common data sources for an ML system and how you would access each.
2. What does a data pipeline do, and what is the difference between batch and streaming
   processing?
3. What is training/serving skew, and how does a feature store prevent it?
4. Why is versioning code not sufficient for reproducible ML, and what does data versioning add?
5. Write (in words or SQL) how you would get the average value of a column grouped by a category
   from a table.
6. Why do experienced engineers attribute most ML production failures to data rather than models?

## Project

A small but real data pipeline (this is concept-focused, but build something concrete):

- Build a reproducible pipeline that ingests data from a non-trivial source (a SQL database you
  populate, or a public API), transforms it into model-ready features, and versions both the raw
  and processed data with DVC.
- Document the feature definitions clearly enough that the same features could be computed at
  serving time (addressing skew), and write the SQL or transformation logic explicitly.
- Write a short note on where this pipeline could break in production and how you would detect it.

**Definition of done:** a runnable, versioned pipeline from a real source to model-ready
features, with documented feature definitions and a failure-mode note.

## The workshop: ship it

Build this in its own repository, `modelwright-data-pipeline`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir modelwright-data-pipeline && cd modelwright-data-pipeline
uv init && uv add pandas sqlalchemy dvc requests && mkdir src data
```

2. Write the ingestion and transformation steps in `src/` (from a SQL database or an API), and
   initialize DVC to version the data (`uv run dvc init`).
3. Commit at checkpoints: "Ingest from source", then "Transform to features", then
   "Version data with DVC".
4. Add a README documenting the feature definitions, how to run the pipeline, and the
   failure-mode note.
5. Ship it:

```bash
gh repo create modelwright-data-pipeline --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`. Keep large data out of Git; DVC handles it.)

**Done when:** `modelwright-data-pipeline` is on GitHub with a reproducible, versioned pipeline and
documented feature definitions.

## Going deeper (optional)

- Read about the modern data stack (dbt for transformations, orchestration with Airflow or
  Dagster).
- Read Kleppmann's *Designing Data-Intensive Applications* for the deep foundations of data
  systems.
- Stand up Feast on your pipeline to see a real feature store in action.

## Canonical references

- Huyen, C. *Designing Machine Learning Systems*, data chapters.
- Made With ML (Goku Mohandas), data and preprocessing lessons.
- Feast and DVC documentation.
