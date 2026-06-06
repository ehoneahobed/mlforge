# Module 5.1 — Reading & Reproducing Papers

**Mode:** hybrid
**Gate:** ship gate
**Est. effort:** 12–20 focused hours (and a skill you keep practicing forever)

> The field moves through papers, and the practitioners who stay at the frontier are the ones
> who can read them fluently and, when it matters, reproduce them. You have already read parts of
> the foundational papers throughout this curriculum. Now you make paper-reading a deliberate
> skill, and prove it the hard way: by taking a paper and getting its result running yourself,
> from the paper alone.

## Why this matters

Courses and tutorials always lag the frontier by months or years. The actual state of the art
lives in papers, and the ability to read them, quickly extract what matters, judge whether to
trust them, and reproduce their results, is what separates someone who consumes ML from someone
who can advance it. It is the core skill of research and a quietly decisive one in industry,
where being able to read a new method and implement it is enormous leverage.

Reproduction is the deepest form of reading. Anyone can nod along to a paper; reproducing its
result forces you to understand every detail the authors assumed you knew, confront the gaps
between what they wrote and what they did, and develop the judgment to fill those gaps. It is
also where you learn that not all papers reproduce, an essential, humbling lesson about how to
read the literature critically.

## What you will be able to do

By the end of this module you will be able to:

- Read a paper efficiently with a structured method, extracting its contribution quickly.
- Find the papers that matter and place a new one in its context.
- Judge a paper critically: its claims, evidence, and limitations.
- Reproduce a non-trivial result from a paper, and honestly report where you matched it and where
  you did not.

## Prerequisites

- The earlier phases, especially the from-scratch builds, which gave you the implementation
  skill reproduction requires. A paper to reproduce in deep learning will lean on Phase 2 and 3.

## Curated path

1. **The method: Keshav, "How to Read a Paper."** A two-page classic describing the three-pass
   approach (a quick skim, a closer read, and a deep reconstruction). Internalize it; it is the
   single most useful thing here.
   https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf

2. **Finding and tracking papers.** With Papers with Code now sunset, the living hubs are arXiv
   (the source) and Hugging Face Papers (curated, trending, with discussion and links to code).
   Learn to use them to find a paper, its citations, and its implementations.
   https://arxiv.org/ and https://huggingface.co/papers

3. **Reading with guidance: the annotated-paper genre.** For a hard paper, a good annotated
   walkthrough (like the Annotated Transformer you used in Phase 2) bridges the gap between the
   text and the code. Use these to learn *how* an expert reads, then wean off them.

4. **Reproducibility practice.** Read about the ML Reproducibility Challenge and reproducibility
   norms (release code, seeds, configs, exact data). This frames what you are doing in the ship
   gate and why it matters to the field.
   Search for the current ML Reproducibility Challenge (run annually by the community).

**Deliberately skip for now:** trying to read everything. Frontier reading is a lifelong habit;
here you build the method and prove it once with a real reproduction.

## Background: the ideas in words

**The three-pass method.** Do not read a paper start to finish like a novel. **First pass**
(five to ten minutes): title, abstract, introduction, section headings, conclusion, and figures.
Decide what the paper claims and whether to keep going. **Second pass** (an hour): read the body
for the main idea and method, noting what you do not understand, but not yet every proof or
detail. **Third pass** (several hours, only for papers that matter to you): reconstruct the work
as if you were re-deriving or reimplementing it, which is exactly what reproduction forces.

**Reading critically.** A paper is an argument, not a fact. Ask: what exactly is the claim, what
is the evidence, what are the baselines (and are they fair), what are the limitations (stated and
unstated), and would this hold outside the paper's setting? The replication crisis is real in ML
too; healthy skepticism is a feature, not cynicism.

**Reproduction in practice.** Reproducing a result means getting the paper's reported outcome
from your own implementation (or from running their code on your setup). The gaps you hit, an
unstated hyperparameter, a preprocessing step buried in an appendix, a detail that "everyone
knows", are where the real learning is. The honest outcome is sometimes "I could not fully
reproduce it, and here is what I think is missing," which is a genuine and valuable result.

## Knowledge check

1. Describe the three-pass method and what each pass is for.
2. After a first pass, what should you be able to say about a paper?
3. What questions do you ask to read a paper critically rather than credulously?
4. Why is reproducing a result the deepest form of reading a paper?
5. Where do you now look to find papers and their code, given Papers with Code is gone?
6. Why is "I could not fully reproduce this, and here is what is likely missing" a legitimate and
   useful outcome?

## Project (ship gate)

**Reproduce a non-trivial result from a paper.**

- Choose a paper with a concrete, checkable result that is within reach of your skills and
  compute (a classic architecture or technique on a standard dataset is ideal; you have the
  tools from Phases 2 and 3). Read it with the three-pass method.
- Reproduce its central result, implementing the method yourself where feasible (lean on your
  from-scratch experience) or running and adapting the authors' code where reimplementation is
  impractical. Either way, you must understand every part.
- **Report honestly:** which numbers you matched, which you did not, and your analysis of any gap
  (an unstated detail, a compute difference, a possible flaw). Include your three-pass notes on
  the paper.

**Definition of done:** a working reproduction with results compared to the paper's, plus a
written analysis of the match and the gaps. The honesty of the gap analysis matters as much as
the match.

## The workshop: ship it

Build this in its own repository, `mlforge-paper-reproduction`, using the project habits from
Module 0.2.

1. Set up the project:

```bash
mkdir mlforge-paper-reproduction && cd mlforge-paper-reproduction
uv init && uv add torch numpy matplotlib && mkdir src notes
```

2. Put your three-pass reading notes in `notes/`, your implementation in `src/`, and run the
   reproduction (a free GPU notebook is fine for training, with code committed here).
3. Commit at checkpoints: "Reading notes and plan", then "Method implemented", then
   "Reproduced result and gap analysis".
4. Add a README stating the paper, your reproduced numbers versus the paper's, and your honest
   analysis of any differences.
5. Ship it:

```bash
gh repo create mlforge-paper-reproduction --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `mlforge-paper-reproduction` is on GitHub with a reproduced result, a comparison
to the paper, and an honest gap analysis. This is a standout portfolio piece; few candidates can
show one.

## Going deeper (optional)

- Participate in the annual ML Reproducibility Challenge with your reproduction.
- Start a reading habit: one paper a week with three-pass notes, building your own annotated
  literature log (in your `mlforge-learning-log`).
- Read a paper you suspect is flawed and write a critical review.

## Canonical references

- Keshav, S. *How to Read a Paper*.
- arXiv and Hugging Face Papers (paper discovery and code).
- The ML Reproducibility Challenge (community).
