# The Modelwright Project

**An open, project-based curriculum for becoming a Machine Learning Engineer.**

**Read the book: https://ehoneahobed.github.io/modelwright**

The Modelwright Project (Modelwright for short) is a free, self-directed path through machine learning, from foundations to
research-grade and production-grade work. It is inspired by The Odin Project: rather than
rewriting the world's best courses, books, and papers, it sequences them with precision and
surrounds them with projects you cannot fake your way through. Where it differs is its
insistence on understanding the internals. In machine learning you can finish a course and
still not understand how a model learns, so every load-bearing concept here is earned by
reimplementing it from scratch.

## What makes it different

- **Curate, don't rewrite.** The curriculum links exact lectures, chapters, and sections of
  the best free material, and tells you what to skip. The original writing is the
  connective tissue and the project specifications.
- **Understanding is proven by building.** A from-scratch ladder runs through the whole
  curriculum: an autograd engine, then a neural network, then a convolutional network, then
  a transformer, then a diffusion model.
- **Everything ships.** Every project is a real artifact. By the end, the portfolio is the
  credential.

## How it is structured

Six phases, each ending in a capstone:

0. **Foundations & The Autograd Engine**
1. **Classical Machine Learning**
2. **Deep Learning**
3. **Modern ML: LLMs & Generative Models**
4. **ML Systems & MLOps**
5. **Specialization & Research Maturity**

Progression is gated, not timed: you advance when you can prove the learning landed,
through concept checks, from-scratch build gates, and shipped projects.

See [`book/curriculum-map.md`](./book/curriculum-map.md) for the full roadmap.

## Reading the curriculum

The curriculum is a [Jupyter Book](https://jupyterbook.org). To read it locally:

```bash
pip install jupyter-book          # installs Jupyter Book v2 (MyST)
cd book
jupyter book start                # live preview at http://localhost:3000
# or build static HTML:
jupyter book build --html
```

You can also read the source Markdown directly, starting with
[`book/intro.md`](./book/intro.md).

## Who this is for

Anyone who wants to become a machine learning engineer and is willing to do the work:
software engineers moving into ML, self-taught learners who want real depth, students who
want a practical complement to formal study, and practitioners filling the gap between
using the tools and understanding them. You need comfort with programming, ideally Python,
and a willingness to engage with mathematics when a model calls for it. The math is taught
just-in-time, with a foundational track for anyone who needs it (Phase 0, Module 0.3).

## Tracking your progress

Copy [`progress.md`](./progress.md) and keep your own version. It mirrors the curriculum map
and records which gates you have passed, the papers you have read, and the artifacts you
have shipped.

## Contributing and extending

Contributions are welcome, from link fixes to entire new modules. Start with
[`CONTRIBUTING.md`](./CONTRIBUTING.md), which explains the module template and quality bar.
The full authoring standard that governs how modules are written and maintained lives in
[`AGENTS.md`](./AGENTS.md).

## License

Modelwright is open and free to use, in the spirit of The Odin Project.

- **Curriculum content** (the written lessons and curriculum map) is licensed under
  **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**. See
  [`LICENSE-CONTENT.md`](./LICENSE-CONTENT.md).
- **Source code** (scripts, notebooks, build configuration, examples) is licensed under the
  **MIT License**. See [`LICENSE`](./LICENSE).

Linked external resources remain the property of their owners under their own licenses.
