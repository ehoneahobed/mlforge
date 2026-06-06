# Module 5.4 — Original Work

**Mode:** self-directed
**Gate:** ship gate
**Est. effort:** open-ended (this is the work that defines you)

> This is the end of the curriculum and the beginning of your own. Every phase until now had a
> path laid out. This module has none, because original work, by definition, is not on anyone's
> path. You will conceive, execute, and ship something that did not exist before: a piece of
> research, a careful extension of existing work, or a tool the community uses. This is where you
> stop being a learner and become a contributor.

## Why this matters

There is a difference between someone who can do machine learning and someone who has *done*
something with it that others recognize. The first is competent; the second is established. The
ability to take a vague idea, scope it into a tractable project, execute it to a real result, and
communicate it so others can build on it, is the highest skill the curriculum develops, and the
one that most distinguishes a research-trained, industry-seasoned engineer.

Original work is also how you become known. A reproduced-and-extended paper, a genuinely novel
experiment, or a widely-used open-source tool is the artifact that opens doors, into research
roles, senior engineering positions, or simply the respect of the field. It is the natural
culmination of everything you built, and the seed of whatever you do next.

## What you will be able to do

By the end of this module you will be able to:

- Scope a vague idea into a tractable, valuable project.
- Execute an open-ended project to a real, defensible result.
- Communicate your work so that others can understand, trust, and build on it.
- Point to a piece of original work that is genuinely yours.

## Prerequisites

- The whole curriculum. Module 5.1 (reading and reproducing) and 5.3 (your specialization) most
  directly feed this, your original work usually grows out of your chosen area.

## Choosing what to do

Original work in ML usually takes one of these forms; any is valid:

- **A novel experiment or finding.** Ask a question no one has clearly answered and answer it
  carefully. It need not be groundbreaking; a small, honest, well-executed result is real
  contribution. (For example: a careful study of how some technique behaves under conditions the
  original authors did not test.)
- **A reproduction-and-extension.** Take a paper (perhaps your Module 5.1 reproduction), and
  extend it: apply it to a new domain, test a variation, or probe a limitation. Extension is one
  of the most accessible routes to genuinely new results.
- **A tool the community uses.** Build and release something useful, a library, a dataset, a
  benchmark, an evaluation, a clear implementation of a hard method, and get it into others'
  hands. Open-source contributions to existing major projects also count.

The best choice sits at the intersection of what interests you (you will live with it), what your
skills can reach, and what would be genuinely useful or interesting to others.

## How to do it (the method)

1. **Scope ruthlessly.** The most common failure is a project too big to finish. State the
   smallest version that would still be a real contribution, and start there. You can extend a
   finished small thing; you cannot finish an unfinished big thing.
2. **Establish what exists.** Use your Module 5.1 skills to find related work, so you know what is
   already done and where your contribution sits. This is what makes work *original* rather than
   accidentally redundant.
3. **Execute with rigor.** Apply everything the curriculum taught about trustworthy results:
   honest baselines, leak-free evaluation, reproducibility, and skepticism of your own findings.
   Original work that is not rigorous is not contribution.
4. **Communicate it well.** A result no one can understand or reproduce does not exist. Write it
   up clearly, a short paper, a detailed README, or a blog post, with enough that someone could
   build on it. Communication is half of contribution.

## Self-assessment

Original work is judged by the field, not a rubric, but ask yourself:

1. Is the result genuinely new, even in a small way, rather than a re-run of known work?
2. Did I establish how it relates to what already exists?
3. Is it rigorous enough that a skeptic would believe it?
4. Could another person understand and build on it from what I have written and shipped?

## Project (ship gate)

**Conceive, execute, and ship a piece of original work.**

- Choose one of the forms above and scope it to something you can actually finish to a real
  result.
- Situate it in the existing work, execute it with full rigor, and produce a clear writeup.
- Ship it where it belongs: a public repository for a tool or experiment, a pull request to an
  open-source project for a contribution, or a write-up (blog or short paper) for a finding,
  always with the code and evidence public.

**Definition of done:** a genuinely original artifact, public, rigorous, and clearly
communicated, that you would be proud to put your name on.

## The workshop: ship it (and make it your capstone)

This module *is* the phase capstone. Ship your original work to its own repository,
`modelwright-capstone` (or a name fitting the work), and treat it as the single best demonstration of
everything you have built.

1. Set it up with `uv` and your full standard project structure: clean code, reproducible
   environment, tests where they fit, experiment tracking, and a thorough README.
2. Build it over as long as it takes, with a commit history that tells the story of the work.
3. Write the definitive writeup: the question or goal, the related work, what you did, the
   results with honest limitations, and how someone could build on it.
4. Ship it publicly, and, if it is a contribution to an existing project, open the pull request.

```bash
gh repo create modelwright-capstone --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** your original work is public, rigorous, and clearly written, the capstone of the
whole curriculum and the start of your own body of work. Add it, and the row of project repos you
built across all five phases, to your portfolio and your resume. You have earned them.

## Going deeper (this is the beginning, not the end)

- If your work is research-shaped, consider submitting it (to a workshop, the ML Reproducibility
  Challenge, or an appropriate venue).
- If it is a tool, maintain it: respond to issues, improve it, build a small community around it.
- Keep going. The habits this curriculum built, reading papers, building from scratch, shipping
  rigorously, are a practice, not a finish line. The field will keep moving; now you can move
  with it, and contribute to where it goes.

## Canonical references

- Your own related-work reading (Module 5.1).
- The standards of rigor and communication built throughout the curriculum.
