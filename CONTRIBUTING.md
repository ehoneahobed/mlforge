# Contributing to Modelwright

Thank you for wanting to improve Modelwright. This curriculum is built on a specific philosophy
and a strict structure, and contributions are most useful when they fit that shape. This
guide explains how to contribute well. The full authoring standard lives in
[`AGENTS.md`](./AGENTS.md); read it before proposing content.

## The philosophy in one minute

Modelwright follows three commitments:

1. **Curate, don't rewrite.** We link the best existing free material with precision (exact
   lectures, chapters, sections) and write the connective tissue and projects. We do not
   reproduce textbooks.
2. **Understanding is proven by building.** Load-bearing concepts get from-scratch
   reimplementation gates.
3. **Everything ships.** Projects are real, tested, documented artifacts.

If a contribution works against any of these, it will not fit, however good it is on its own
terms.

## Ways to contribute

- **Fix errors:** broken links, outdated resources, typos, incorrect technical claims. These
  are always welcome and easy to accept.
- **Improve an existing module:** a clearer explanation, a better-sequenced resource, a
  sharper project spec, an additional knowledge-check question.
- **Propose a new module or build out a drafted one:** the curriculum map lists many modules
  that are planned but not yet written. These are the highest-value contributions and the
  ones that most need to follow the standard exactly.

For anything larger than a fix, open an issue to discuss the approach before writing, so
effort is not wasted on something that will not fit.

## The module template (required)

Every module follows the same structure. Do not invent a new shape; fill this one.

- **Title line** with the mode (`curated breadth` | `from-scratch deep-dive` | `hybrid`),
  the gate (`concept check` | `build gate` | `ship gate`), and an honest effort estimate.
- **Why this matters** — motivation and placement in the field.
- **What you will be able to do** — concrete learning outcomes.
- **Prerequisites** — prior modules and the specific math results invoked (linked to the
  Module 0.3 reference layer where relevant).
- **Curated path** — an ordered list. Each item names the source, the *exact* scope, why it
  is here, and what to skip. Quality over volume: three to six items, not twenty.
- **Knowledge check** — questions answerable without notes.
- **Build gate** (where applicable) — the from-scratch spec and the tests it must pass.
- **Project / ship gate** — the artifact to deliver and its definition of done.
- **Going deeper** (optional) and **Canonical references**.

Look at any existing Phase 0 or Phase 1 module as a worked example.

## Quality bar

- **Verify every external resource** before adding it. Links rot and "best" resources change.
  Confirm the resource exists, is current, and is genuinely among the best options.
- **Be specific in curation.** "Watch the lectures" is not acceptable; "CS229 lecture 3,
  sections on the logistic loss" is.
- **From-scratch gates must be verifiable.** Specify the tests (for example, gradients
  checked against a framework, predictions matched to scikit-learn).
- **Self-contained, general-audience voice.** Lessons must stand on their own for any
  reader. Do not assume one reader's background, name individuals, or reference the
  authoring process inside a lesson.

## Style rules

- No em dashes; use clearer connecting words or restructure.
- Be concise. Elaborate only where depth is the point.
- Write in prose. Use lists only where they genuinely aid clarity (curated paths, knowledge
  checks, specs).
- Cross-link other lessons and the curriculum map freely; do not cite internal files
  (`AGENTS.md`, `progress.md`) inside a lesson.

## Building the book locally

The curriculum is a [Jupyter Book](https://jupyterbook.org) (v2 / MyST).

```bash
pip install jupyter-book
cd book
jupyter book start          # live preview at http://localhost:3000
# or build static HTML:
jupyter book build --html
```

When you add a page, also add it to the table of contents in `book/myst.yml`. Build the book
before opening a pull request and confirm it compiles with no errors or warnings.

## Submitting

1. Open an issue first for anything beyond a small fix.
2. Make your change in a branch.
3. Confirm the book builds cleanly and all links resolve.
4. Open a pull request describing what you changed and why, and noting which part of the
   curriculum map it addresses.

By contributing, you agree that your contributions are licensed under the same terms as the
project: MIT for code and CC BY-SA 4.0 for content (see [`LICENSE`](./LICENSE) and
[`LICENSE-CONTENT.md`](./LICENSE-CONTENT.md)).
