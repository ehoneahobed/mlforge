# Module 0.1 — The Landscape of Machine Learning

**Mode:** orientation
**Gate:** concept check
**Est. effort:** 4–7 focused hours

## Why this matters

Before you build anything, you need a map. It is hard to learn a field when you do not yet
know its shape: what the big words mean, how they relate, and where the thing you are
learning today fits into the whole. This module is that map. By the end of it you will be
able to hold a real conversation about machine learning, place every later topic in
context, and understand what you are actually working toward.

This matters more in machine learning than in most fields, because the vocabulary is
crowded and often used loosely. Artificial intelligence, machine learning, deep learning,
neural networks, large language models, generative AI: these are not synonyms, and they are
not interchangeable. They nest inside one another in a specific way. Once you see that
structure, the rest of the curriculum stops being a list of disconnected topics and becomes
a guided tour through a landscape you can already picture.

You do not need to build anything here. You need to understand the terrain. The hands-on
work starts in the next module.

## What you will be able to do

By the end of this module you will be able to:

- Explain, in plain language, what machine learning is and how it differs from ordinary
  programming.
- Place artificial intelligence, machine learning, deep learning, neural networks, and
  large language models correctly inside one another.
- Name the main types of machine learning (supervised, unsupervised, self-supervised,
  reinforcement) and give an everyday example of each.
- Describe the end-to-end workflow that turns a question and some data into a working,
  deployed model.
- Describe what a machine learning engineer actually does, and how the role differs from a
  data scientist or an AI engineer.

## Prerequisites

- Curiosity, and comfort reading a little code. You do not need any prior machine learning
  knowledge; that is the point of this module.
- If you have never programmed before, learn the basics of Python first (variables, loops,
  functions, and a little NumPy and pandas), then return. The rest of the curriculum
  assumes you can write and run Python.

## Curated path

Work these in order. They are chosen to build the mental map quickly, from the big picture
down to the vocabulary.

1. **See the destination first: 3Blue1Brown, "But what is a neural network?" (Chapter 1).**
   About twenty minutes. You will not understand every detail yet, and that is fine. Watch
   it to see, visually, the kind of thing this whole field builds toward. It makes the
   abstract concrete before you meet the words.
   https://www.youtube.com/watch?v=aircAruvnKk

2. **What machine learning is: Google Machine Learning Crash Course, "Introduction to ML"
   and "ML problem framing."** A short, clear, official explanation of the core idea:
   learning patterns from data instead of writing explicit rules. Read the introduction and
   the problem-framing sections.
   https://developers.google.com/machine-learning/intro-to-ml and
   https://developers.google.com/machine-learning/crash-course

3. **The types of learning: StatQuest, "A Gentle Introduction to Machine Learning."** Plus
   the short StatQuest videos on what supervised learning and clustering are. These give you
   the categories (supervised, unsupervised, and the rest) with clear, friendly examples.
   https://statquest.org/video-index/

4. **The field as a visual map: the roadmap.sh Machine Learning roadmap.** Scroll through it
   once, top to bottom, as a bird's-eye view of the territory. Do not try to learn the
   topics from it; just notice the major regions (foundations, data, the model families,
   evaluation, deep learning) and how they connect. Modelwright is a deeper, project-based walk
   through this same landscape.
   https://roadmap.sh/machine-learning

5. **Optional, for the broader picture: Andrew Ng, "AI for Everyone" (Coursera, free to
   audit), or Week 1 of his Machine Learning Specialization.** A gentle, non-technical
   framing of where ML sits in the world and what it can and cannot do.
   https://www.coursera.org/learn/ai-for-everyone

**Deliberately skip for now:** the mathematics, any specific algorithm, and the deep
learning details. They all come later, each in its place. Right now you are drawing the
map, not walking it.

## Background: the landscape in words

**What machine learning is.** In ordinary programming, you write the rules: if this, then
that. Machine learning flips it around. You show a program many examples, and it learns the
rules itself by finding patterns in the data. You do not tell it how to recognize a cat; you
show it thousands of labeled pictures and it works out what "cat" looks like. That shift,
from writing rules to learning them from data, is the whole idea.

**The nested map.** These terms fit inside one another:

- **Artificial intelligence (AI)** is the broadest idea: machines doing things that seem to
  require intelligence.
- **Machine learning (ML)** is the part of AI that learns from data. It is the subject of
  this curriculum.
- **Deep learning** is the part of ML that uses **neural networks** with many layers. It
  powers most of the recent breakthroughs.
- **Large language models (LLMs)** and **generative AI** are recent, powerful applications
  of deep learning. An LLM like the ones behind modern chat assistants is a very large
  neural network trained on enormous amounts of text.

So: large language models are deep learning, deep learning is machine learning, and machine
learning is artificial intelligence. Each sits inside the next.

**The types of machine learning.** How a model learns depends on what data it gets:

- **Supervised learning:** the data is labeled. You have examples with the right answers,
  and the model learns to predict the answer for new examples. Two flavors: **classification**
  (predict a category, such as spam or not spam) and **regression** (predict a number, such
  as a house price). Most machine learning in industry is supervised.
- **Unsupervised learning:** the data has no labels. The model finds structure on its own,
  such as grouping similar customers (**clustering**) or compressing many features into a few
  (**dimensionality reduction**).
- **Self-supervised learning:** a clever trick where the data labels itself. Large language
  models learn this way: hide the next word in a sentence and have the model predict it,
  over and over, across the whole internet. No human labeling required.
- **Reinforcement learning:** the model learns by trial and error, taking actions and
  receiving rewards or penalties, the way you might train a dog or a game-playing agent.
- (You will also hear **semi-supervised learning**, which mixes a little labeled data with a
  lot of unlabeled data.)

**The anatomy of a model.** A few words you will see constantly. **Features** are the inputs
(the columns describing each example). **Labels** are the answers in supervised learning.
**Parameters** (or weights) are the numbers inside the model that get adjusted during
learning. **Training** is the process of adjusting those parameters to fit the data.
**Inference** is using the trained model to make predictions on new data.

**The end-to-end workflow.** Almost every machine learning project, from a weekend
experiment to a system serving millions, follows the same arc. Every later phase of this
curriculum is a deep zoom into one part of it:

1. **Frame the problem.** What are you predicting, and why? What would success look like?
2. **Collect data.** Gather the raw material, from databases, files, APIs, or sensors.
3. **Clean and prepare it.** Real data is messy. Fix missing values, fix formats, and build
   useful features.
4. **Choose and train a model.** Pick an approach, and let it learn from the data.
5. **Evaluate it honestly.** Measure how well it works on data it has never seen, and decide
   whether to trust it.
6. **Deploy it.** Put it into a real system where it makes predictions for real users.
7. **Monitor and maintain it.** Watch for failures and drift, and improve it over time.

**What a machine learning engineer does.** A machine learning engineer builds and ships the
systems that put models into the real world. The role sits between a **data scientist** (who
leans toward analysis, experimentation, and statistics) and a **software or AI engineer**
(who leans toward building applications, increasingly with pre-built models and APIs). An ML
engineer needs enough of both: the statistical understanding to build a model that works,
and the engineering skill to make it reliable, fast, and maintainable in production. That
dual demand is exactly what this curriculum is built to satisfy.

## Knowledge check

Answer these without notes. If you can, you have the map.

1. In one or two sentences, how is machine learning different from ordinary programming?
2. Put these in order from broadest to narrowest: deep learning, artificial intelligence,
   large language models, machine learning. Explain the nesting.
3. Name the main types of machine learning and give a real, everyday example of each.
4. What is the difference between classification and regression? Give an example of each.
5. How does a large language model learn from text without anyone labeling it? Which type of
   learning is that?
6. What is the difference between training a model and using it for inference?
7. Sketch the end-to-end machine learning workflow from a question to a deployed, monitored
   model.
8. What does a machine learning engineer do, and how does the role differ from a data
   scientist?

## Project

A short writing-and-thinking project. No code required.

- **Draw your own map.** In one page, in your own words, explain the landscape of machine
  learning to a smart friend who knows nothing about it: what ML is, how AI, ML, deep
  learning, and LLMs nest, and the main types of learning. Writing it in your own words is
  the test of whether the map is really in your head.
- **Classify five real products.** Pick five machine learning products or features you use
  (a recommendation feed, a spam filter, a chat assistant, a photo search, a navigation app,
  and so on). For each, identify which type of learning it most likely uses and which step
  of the workflow you think is hardest for that product, and why.
- **Place yourself.** Write a short paragraph on which parts of the workflow you are most
  drawn to (the modeling, the data work, the production engineering) and why. This is not
  binding; it is a first read on where your interest sits.

**Definition of done:** the one-page map, the five-product classification, and the
self-placement paragraph. Keep it; it is a useful thing to look back on as the field stops
feeling foreign.

## The workshop: start your learning log

Do not just write the project in a local file and move on. Put it on GitHub today. Shipping
something on day one builds the habit that makes everything in this curriculum real, and it
starts the public portfolio that will eventually be your credential. Follow these steps, do
not just read them.

You will create one repository, `modelwright-learning-log`, that holds all your *written* work
across the whole curriculum (writeups, concept-check reflections, your reading log). Your
build projects later will each get their own repository; this one is for words.

**1. Create the repository.** With the GitHub CLI:

```bash
gh repo create modelwright-learning-log --public --clone
cd modelwright-learning-log
```

No `gh`? Create an empty public repository named `modelwright-learning-log` on github.com (do
not add a README), then `git clone <its-url>` and `cd` into it.

**2. Add your writeup.** Create a file `phase-0/01-landscape.md` and write the three pieces
of the project above into it: your one-page map, your five-product classification, and your
self-placement paragraph.

**3. Add a README** so the repo explains itself:

```bash
echo "# Modelwright Learning Log" > README.md
echo "My written work as I go through the Modelwright curriculum." >> README.md
```

**4. Commit and push:**

```bash
git add -A
git commit -m "Module 0.1: the landscape of machine learning"
git push
```

**Done when:** `modelwright-learning-log` is live on your GitHub with your writeup visible in the
`phase-0/` folder. From now on, every written deliverable and concept-check reflection goes
here, building a record of your thinking across the journey.

## Going deeper (optional)

- Andrew Ng's **Machine Learning Specialization** (Coursera, free to audit) is the classic
  gentle on-ramp to the actual techniques, and a good companion to the next several modules.
- Read a short **history of machine learning** to see how the ideas arrived, from the
  perceptron to deep learning to transformers. It makes the field feel less like a pile of
  tricks and more like a story.
- Skim the **AI Engineer** and **MLOps** roadmaps on roadmap.sh to see the neighboring roles
  and how they overlap with machine learning engineering.

## Canonical references

- 3Blue1Brown, *Neural Networks*, Chapter 1.
- Google, *Machine Learning Crash Course* and *Introduction to Machine Learning*.
- roadmap.sh, *Machine Learning Roadmap* (used as a visual map of the field).
