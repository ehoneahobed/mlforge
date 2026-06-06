# Module 0.3 — Working with Data

**Mode:** hands-on
**Gate:** ship gate
**Est. effort:** 6–10 focused hours

## Why this matters

Models get the glory, but data is where machine learning actually lives. In real projects,
a large majority of the effort goes into finding, loading, understanding, and cleaning data,
not into the model. A mediocre model on excellent data beats an excellent model on messy
data almost every time. This module builds the everyday data fluency that the rest of the
curriculum quietly assumes, so that when later modules say "load the dataset and split it,"
you already know exactly what to do and why.

It also fills in a step the field's roadmaps put right at the start: where data comes from,
the formats it arrives in, how to explore it, how to clean it, and how to split it without
fooling yourself. None of this is glamorous. All of it is the difference between a project
that works and one that silently does not.

## What you will be able to do

By the end of this module you will be able to:

- Load data from the common formats (CSV, JSON, Parquet) into pandas.
- Explore a new dataset quickly to understand its shape, types, and problems.
- Clean real data: handle missing values, fix types, and spot obvious errors.
- Split data into training, validation, and test sets correctly, and explain why the order
  of operations matters.
- Make a few clear plots that answer a question about the data.

## Prerequisites

- [Module 0.1](./01-landscape-of-ml.md) (the workflow and where data fits) and
  [Module 0.2](./02-ml-engineers-toolkit.md) (a working environment and basic pandas).

## Curated path

1. **Where data comes from, and the formats it arrives in.** Read a short overview of common
   data formats and when each is used: CSV and Excel for tabular exports, JSON for nested or
   API data, and Parquet for efficient columnar storage. The pandas input/output guide is
   the practical reference.
   https://pandas.pydata.org/docs/user_guide/io.html

2. **Exploring a dataset: Kaggle Learn, "Pandas."** A hands-on, exercise-driven course in
   the operations you will use constantly: selecting, filtering, grouping, and summarizing.
   Work the exercises, do not just read.
   https://www.kaggle.com/learn/pandas

3. **Cleaning real data: Kaggle Learn, "Data Cleaning."** Missing values, scaling and
   normalization basics, inconsistent entries, and dates. The practical companion to the
   pandas course.
   https://www.kaggle.com/learn/data-cleaning

4. **Splitting data correctly: scikit-learn on cross-validation and `train_test_split`.**
   Read the "Cross-validation" introduction for what training, validation, and test sets are
   for, and why any cleaning or scaling must be fit on training data only. This is the habit
   that prevents the most common and most damaging mistake in applied ML.
   https://scikit-learn.org/stable/modules/cross_validation.html

5. **Making plots that answer a question.** Skim the Matplotlib and seaborn quick-starts,
   focusing on the handful of plots you will actually use to understand data: histograms,
   scatter plots, and bar charts.
   https://seaborn.pydata.org/tutorial/introduction.html

**Deliberately skip for now:** databases and SQL, data pipelines, and feature stores. They
matter at scale and are covered properly in Phase 4. Here the goal is fluency with files and
pandas, which is what real projects start from.

## Background: the ideas in words

**Formats.** Most data you meet early is tabular and arrives as CSV or Excel. JSON shows up
from APIs and has nested structure. Parquet is a compressed columnar format used when data
gets large because it is fast to read and small on disk. pandas reads all of them with a
single function each, and a DataFrame is the table you work in.

**Exploration.** Before modeling anything, you look. How many rows and columns? What are the
types? How many values are missing, and where? What do the distributions look like, and are
there obvious outliers or impossible values (a negative age, a date in the future)? This
first look saves you from training a model on broken data and trusting the result.

**Cleaning.** Real data has gaps and errors. Missing values can be dropped (carefully),
filled with a sensible estimate, or flagged with an indicator so the model can use the fact
that they were missing. Types may need fixing (numbers stored as text, dates as strings).
The goal is data the model can actually learn from, without distorting what it represents.

**Splitting, and the cardinal rule.** You hold data out so you can measure how a model does
on examples it has never seen: a training set to learn from, a validation set to tune
choices, and a test set you touch once at the very end. The cardinal rule, which you will
meet again and again, is that every cleaning and transformation step must be decided and
fit on the training data only. If you scale or fill using statistics computed from the whole
dataset, information from the test set leaks into training, and your results become a
comfortable lie.

## Knowledge check

1. Name three common data formats and when you would expect to meet each.
2. What are the first few things you check when you open an unfamiliar dataset?
3. Give two ways to handle missing values and a situation where each is appropriate.
4. What are the training, validation, and test sets each for?
5. Why must cleaning and scaling be fit on the training data only? What goes wrong if you
   fit them on the full dataset first?

## Project (ship gate)

**Take a real, messy dataset from raw file to clean, split, and understood.**

- Find a real dataset (a Kaggle dataset or one from the
  [UCI repository](https://archive.ics.uci.edu/)) that is not pre-cleaned. Load it into
  pandas.
- Explore it: report its shape, types, missing values, and at least three plots that reveal
  something about it. Write a short paragraph on what you found and what looks problematic.
- Clean it: handle the missing values and any type or consistency issues, explaining each
  decision.
- Split it into training, validation, and test sets, and write down the rule you will follow
  to keep cleaning leak-free when you start modeling.

**Definition of done:** a notebook or script that goes from the raw file to a clean, split
dataset, with the exploration plots and a short written summary of the data's problems and
how you handled them. This is the foundation your Phase 0 capstone model will sit on.

## The workshop: explore, clean, and ship

Build this in its own repository, `modelwright-data-cleaning`, following along step by step. You
will reuse the project habits from Module 0.2.

**1. Create the project:**

```bash
mkdir modelwright-data-cleaning && cd modelwright-data-cleaning
uv init
uv add pandas matplotlib seaborn scikit-learn jupyter
mkdir data notebooks
```

**2. Get a real, messy dataset.** Download one (from Kaggle or the
[UCI repository](https://archive.ics.uci.edu/)) into `data/`. Pick something with missing
values and mixed types, not a pre-cleaned teaching set.

**3. Explore, in a notebook.** Start `uv run jupyter lab`, create
`notebooks/01-explore.ipynb`, and work through the dataset: its shape, column types, missing
values per column, and at least three plots that reveal something. Write a short markdown
cell summarizing what you found and what looks problematic.

**4. Checkpoint one, commit your exploration:**

```bash
git add -A && git commit -m "Explore the raw dataset"
```

**5. Clean and split.** In a second notebook or a `src/prepare.py` script, handle the
missing values and any type or consistency issues (explaining each decision in a comment or
markdown cell), then split into train, validation, and test sets. Write down the rule you
will follow to keep cleaning leak-free once you start modeling.

**6. Checkpoint two, commit the pipeline:**

```bash
git add -A && git commit -m "Clean and split the dataset, leak-free"
```

**7. Write a README** explaining the dataset, its problems, and how to run your code, then
ship it:

```bash
git add -A && git commit -m "Add README"
gh repo create modelwright-data-cleaning --public --source=. --push
```

(No `gh`? Create an empty public repo, `git remote add origin <url>`, `git push -u origin
main`. Note: keep large raw data out of Git; if the file is big, document where to download
it in the README rather than committing it.)

**Done when:** `modelwright-data-cleaning` is on GitHub, showing the raw-to-clean journey, the
exploration plots, and a clear written summary of the data's problems and your fixes.

## Going deeper (optional)

- Read about tidy data (Hadley Wickham's principles) for a clear mental model of how tabular
  data should be shaped.
- Practice on a second, messier dataset; data skill comes from reps, not reading.
- Skim how APIs return data and how to pull it with `requests`, a common real-world source.

## Canonical references

- pandas documentation, *IO tools* and *Getting started*.
- Kaggle Learn, *Pandas* and *Data Cleaning*.
- scikit-learn documentation, *Cross-validation*.
