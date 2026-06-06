# AGENTS.md — The Constitution of *MLForge*

> **MLForge** is a self-directed, project-driven curriculum for becoming a Machine
> Learning Engineer with the depth of someone who completed a research degree (MSc/PhD)
> **and** the practical judgment of someone who has shipped ML systems in industry.
>
> This file governs every future session. Any agent (Claude or otherwise) working on
> this curriculum reads this file first and obeys it. `CLAUDE.md` points here.

---

## 0. Who this is for (the audience)

MLForge is an open, public curriculum. Write for a general audience of aspiring machine
learning engineers, not for any single individual. Assume the reader:

- **Can program**, ideally in Python, and is comfortable with Git and the command line.
- **Has variable math backgrounds.** Some arrive with solid linear algebra, calculus, and
  probability; others are rusty or incomplete. Content must serve both: teach math
  just-in-time, and point readers who need foundations to the foundational track in
  Module 0.3.
- **Wants depth, not just usage.** The target outcome is research-level *understanding*
  plus industry-level *capability*: able to derive a method, reimplement it from scratch,
  scale it, ship it, debug it in production, and read the paper it came from.

**Math is taught just-in-time**, never front-loaded for its own sake. Surface the specific
result the moment a model needs it, motivate it, then use it. There is no separate math
semester; Module 0.3 provides the reference layer and a foundational track for those who
need it.

Never write learner-facing content that assumes one specific reader's background, names an
individual, or refers to private context. Lessons must stand on their own for any reader.

---

## 1. Philosophy (why MLForge is built the way it is)

The Odin Project's real genius is not its writing. It is its **spine**: ruthless
sequencing of the best existing free material, plus original projects you cannot bluff
your way through. We copy that, and we add the one thing ML needs that web dev does not.

**The three commitments:**

1. **Curate, don't rewrite.** The world already has fast.ai, CS231n, CS229, *Dive into
   Deep Learning*, 3Blue1Brown, the *Deep Learning* book, and the original papers. We
   link the best of it with surgical precision (specific lectures, chapters, sections),
   and we own the connective tissue: *why this, why now, what to ignore.*

2. **Understanding is proven by building, not by watching.** ML's unique trap: you can
   finish a course and still not understand backprop, because the framework hid it. So
   every load-bearing concept has a **from-scratch reimplementation** gate. You don't
   advance past attention until you've written attention in NumPy. You don't advance past
   gradient descent until your own autograd engine trains a net.

3. **Everything ships.** Industry capability is not theoretical. Projects are not
   exercises; they are artifacts: tested code, a written report, a deployed endpoint, a
   reproduced result. The portfolio *is* the curriculum's output.

**The central tension we deliberately hold:** *use-it* vs *build-it-from-scratch.* Pure
curation produces people who can use tools but panic when they break. Pure from-scratch
is slow and reinvents wheels. MLForge alternates: curated breadth to move fast, then a
from-scratch deep-dive on the concepts where not understanding the internals will someday
cost you. Every module declares which mode it is in.

---

## 2. The learning model (how a learner moves through it)

The curriculum is a sequence of **Phases**, each containing **Modules**, each containing
**Units**. A Unit is one sitting's worth of focused work. A Module is a coherent skill.
A Phase is a level of the field.

Progression is **gated, not timed.** You advance when you pass the gate, however long it
takes. The gates are the point.

Three gate types, used in increasing severity:

- **Concept check** — short questions you must answer without notes. Catches passive
  watching. (Every unit.)
- **Build gate** — reimplement the mechanism from scratch and make it pass tests.
  (Every load-bearing concept.)
- **Ship gate** — a project delivered as a real artifact with a writeup. (Every module
  has at least one; every phase ends in a capstone.)

Nothing is "done" until its gate is green. A half-built project keeps its task open.

---

## 3. The Module Template (NON-NEGOTIABLE structure)

Every module file follows this exact skeleton. Future agents: do not improvise a new
shape. Fill this one.

```
# Module X.Y — <Title>

**Mode:** curated breadth | from-scratch deep-dive | hybrid
**Gate:** concept check | build gate | ship gate
**Est. effort:** <range, honest>

## Why this matters
2–4 sentences. The motivation and where it sits in the field. If a learner could skip
this, say why they can't. Connect to what's next.

## Prerequisites
Exact prior modules + the specific math results invoked (linked, just-in-time).

## Curated path (the spine)
An ORDERED list. Each item: source, the EXACT scope (lecture N / chapter N / §N), why
it's here, and what to deliberately skip. Quality over volume. 3–6 items, not 20.

## Concept checks
5–10 questions answerable without notes. The honest test of "did it land."

## Build gate (if applicable)
The from-scratch reimplementation. Spec, starter notebook reference, and the tests it
must pass. No framework allowed where the point is the mechanism.

## Project / Ship gate
The artifact to deliver: what to build, the definition of done, and the writeup prompt.

## Going deeper (optional)
Papers, harder resources, rabbit holes for the curious. Clearly marked optional.

## Canonical references
The papers/books this module draws on, for the portfolio reading log.
```

---

## 4. Project & code standards

A project counts only if it meets all of these:

- **It runs.** Reproducible from a clean environment. Pinned dependencies. A README that
  a stranger could follow.
- **It's tested.** From-scratch implementations carry unit tests that check correctness
  against a reference (e.g. gradients vs `torch.autograd`, predictions vs sklearn).
- **It's measured.** Real metrics, an honest baseline, and the comparison. "It seems to
  work" is not a result.
- **It's written up.** A short report: what you built, what you found, what surprised
  you, what you'd do next. Writing forces understanding and builds the portfolio.
- **It's version-controlled.** Each project is a commit history, not a single dump.

---

## 5. Assessment philosophy

We do not grade for completion. We test for **transfer**: can you apply it to a problem
you haven't seen? Hence the from-scratch gates (you cannot reimplement what you only
recognize) and the writeups (you cannot explain what you don't understand). The learner
is trusted to be honest at the gates; the curriculum's value collapses the moment a gate
is faked, so the incentive is self-aligned.

End-of-phase **capstones** are the real exams: open-ended, portfolio-grade, and designed
to look like something an employer or a paper reviewer would respect.

---

## 6. Operating instructions for future sessions (read every time)

When a session starts on this curriculum, the agent must:

1. **Read this file, then `book/myst.yml` and `book/curriculum-map.md`** to know the
   current shape and where the learner is.
2. **Check `progress.md`** for what's done and what's next. Respect the gates: do not
   wave the learner forward past an unmet gate.
3. **When building a new module, follow the Module Template in §3 exactly.** Curate with
   real specificity (lecture numbers, chapter numbers), never vague "watch some videos."
4. **Verify every external resource still exists and is still the best option** before
   recommending it. Resources rot. Search; do not trust memory for links, prices, or
   "newest" claims.
5. **Update three things together** whenever content changes: the module file,
   the `toc` in `book/myst.yml`, and `progress.md`. They must never drift apart.
6. **Prefer curation; reserve original writing for from-scratch deep-dives** and for the
   connective tissue (the "why," the transitions, the project specs). Do not rewrite
   textbooks.
7. **Build the book to validate** structure after edits: `cd book && jupyter book start`
   for live preview, or `jupyter book build --html`. (We use Jupyter Book v2 / MyST, which
   is configured by `myst.yml`, not the v1 `_config.yml` + `_toc.yml`.)
8. **Be a strategic partner, not an order-taker.** Challenge the learner's plan when a
   better sequence exists. Surface trade-offs. Push back with reasons.

### Style rules for all generated content
- No em dashes. Use better connecting words or restructure.
- Be concise; elaborate only when depth is the point.
- Curation lists are ordered and specific. No filler.
- Every claim about the outside world (a link, a tool, a benchmark) is verified by search
  before it ships.
- Learner-facing pages are self-contained. Do not cite internal scaffolding files
  (`AGENTS.md`, `progress.md`) or refer to the curriculum's authoring process inside a
  lesson. Cross-reference other lessons and the curriculum map freely; those are part of
  the product.

---

## 7. Tech stack & conventions

- **Format:** Jupyter Book v2 / MyST (markdown lessons + runnable `.ipynb` projects).
  Source lives in `book/`, configured by `book/myst.yml`.
- **Build:** `cd book && jupyter book start` (live preview) or `jupyter book build --html`.
  Lessons render code as display blocks; project notebooks are run by the learner locally.
- **Environment:** Python 3.10+, the scientific stack (numpy, scipy, pandas, matplotlib,
  scikit-learn), PyTorch as the primary deep-learning framework.
- **Repo layout:**
  ```
  Machine Learning/
    AGENTS.md            <- this file (canonical)
    CLAUDE.md            <- pointer to this file
    progress.md          <- the learner's progress tracker
    book/
      myst.yml           <- Jupyter Book v2 config + table of contents
      intro.md
      curriculum-map.md  <- the full phase/module roadmap
      phase-0-foundations/
      phase-1-classical-ml/
      ... etc
  ```

---

## 8. How the field maps in (grounding)

The curriculum is cross-checked against the roadmap.sh Machine Learning roadmap
(https://roadmap.sh/machine-learning) for coverage, but it is deeper and more rigorous:
roadmap.sh tells you the *topics*; MLForge makes you *earn* them. Where roadmap.sh is
breadth-first and job-oriented, we add the research layer (paper reproduction, theory,
original work) that takes a learner past "employable" to "expert."

---

*Version 1.0. This is a living document. When the curriculum's philosophy or structure
evolves, this file changes first, and everything else follows it.*
