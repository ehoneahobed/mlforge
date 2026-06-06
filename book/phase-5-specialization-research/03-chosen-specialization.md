# Module 5.3 — A Chosen Specialization

**Mode:** self-directed
**Gate:** ship gate
**Est. effort:** open-ended (several weeks of focused work)

> Until now you have built breadth, a working command of the whole field. Real expertise,
> though, is deep, not just wide. This module is where you choose one area and go far enough into
> it to have genuine, current expertise: to know its state of the art, its open problems, and its
> craft. Breadth got you here; depth is what makes you valuable.

## Why this matters

No one is an expert in all of machine learning; the field is too large. What employers and the
research community value is a T-shaped profile: broad competence across the field (which you now
have) plus deep expertise in at least one area. Depth is where you stop following others' work
and start having informed opinions, recognizing what is hard, what is overrated, and where the
real opportunities are. It is also what makes your portfolio distinctive: a deep project in a
chosen area says far more than another tutorial-level breadth project.

This module is deliberately self-directed because the choice is yours and the path depends on it.
The curriculum gives you the strongest entry point for each major track and the standard of what
"going deep" means; you choose the direction and drive it.

## What you will be able to do

By the end of this module you will be able to:

- Speak fluently about the state of the art in your chosen area.
- Use its specialized tools and techniques at a practitioner's level.
- Read its current literature without hand-holding (using Module 5.1's skills).
- Ship a substantial project that demonstrates real depth, not breadth.

## Prerequisites

- The earlier phases, and ideally Module 5.1 (you will read current papers in your area).

## Choosing a track

Pick one area to go deep in. The common tracks, each with its canonical deep resource:

- **Natural Language Processing.** Beyond the LLM work of Phase 3, into the depth of language
  modeling, understanding, and generation. Deep resource: Stanford CS224n (Natural Language
  Processing with Deep Learning). https://web.stanford.edu/class/cs224n/
- **Computer Vision.** Beyond Phase 2's CNNs, into detection, segmentation, vision transformers,
  and generative vision. Deep resource: Stanford CS231n. https://cs231n.github.io/
- **Generative Models.** Diffusion, advanced generative architectures, multimodal generation.
  Deep resource: the Hugging Face Diffusion Course (full) plus the current literature.
  https://huggingface.co/learn/diffusion-course/
- **Reinforcement Learning.** Beyond Module 5.2, into advanced algorithms, multi-agent, and RL
  for alignment. Deep resources: Sutton & Barto (full) and Spinning Up.
  http://incompleteideas.net/book/the-book-2nd.html
- **ML Systems.** Beyond Phase 4, into the deep engineering of large-scale training, serving, and
  infrastructure. Deep resource: Chip Huyen's *Designing Machine Learning Systems* (full) plus
  the systems literature. https://github.com/chiphuyen/dmls-book

Other valid tracks exist (speech, recommender systems, graph ML, ML for science, AI safety). The
test of a good choice is simple: it genuinely interests you (you will spend weeks on it), and it
has depth to reward the time.

## How to go deep (the method)

Depth is not a single resource but a process:

1. **Work the canonical course or book for your track, fully.** Not skimmed, worked, with its
   assignments. This is the structured backbone of your expertise.
2. **Read the current literature.** Using Module 5.1's three-pass method, read the seminal papers
   and the recent important ones in your area. Keep notes in your learning log.
3. **Master the specialized tooling.** Each area has its own stack; become genuinely fluent in
   it.
4. **Build something substantial.** A project that requires real depth in the area, not a
   weekend tutorial. This is the ship gate.

## Knowledge check

These are self-assessed, because the content is yours:

1. Can you explain the state of the art in your chosen area to another engineer, including its
   open problems?
2. Can you read a new paper in your area and understand it without external help?
3. Are you fluent in the area's specialized tools and techniques?
4. Does your project demonstrate depth that a breadth-level practitioner could not produce?

## Project (ship gate)

**Ship a substantial project that demonstrates real depth in your chosen area.**

This should be clearly more advanced than the phase projects so far, something that requires the
specialized knowledge you have built. Examples by track:

- **NLP:** build and evaluate a non-trivial language system (e.g., a fine-tuned, retrieval-grounded
  domain assistant with rigorous evaluation), or reproduce and extend a recent NLP paper.
- **Computer Vision:** an object detection or segmentation system on a real dataset, or a
  vision-transformer reproduction with your own analysis.
- **Generative:** a working conditional diffusion model you trained and analyzed, or a careful
  reproduction of a generative paper.
- **RL:** a hard environment solved with an advanced algorithm you implemented, with analysis.
- **ML Systems:** a genuinely scaled or production-grade system component, benchmarked and
  documented.

The project must include a written analysis at the level of someone who knows the area, not a
beginner: design choices justified against the state of the art, real evaluation, and honest
limitations.

**Definition of done:** a substantial, deep project shipped to its own repository, with a writeup
that demonstrates expertise in your chosen area.

## The workshop: ship it

Build this in its own repository, named for your specialization (for example
`modelwright-nlp-<project>`, `modelwright-cv-<project>`, or `modelwright-rl-<project>`). Use every habit from
the curriculum: clean structure, reproducible environment, tests where appropriate, experiment
tracking, and a thorough README.

1. Set up the project with `uv`, the specialized libraries your track needs, and your standard
   project structure.
2. Build it over several focused sessions, committing meaningfully as you go (this is a larger
   project; the commit history should tell the story).
3. Write a README and a deeper writeup (in the repo or your learning log) covering design,
   evaluation, related work, and limitations.
4. Ship it to GitHub with `gh repo create ... --public --source=. --push`.

**Done when:** a substantial, depth-demonstrating project is on your GitHub, documented to a
standard that signals genuine expertise in your chosen area.

## Going deeper (optional)

- Contribute to an open-source project in your area; reading and improving real code is deep
  learning of a different kind (and leads naturally into Module 5.4).
- Write a blog post or explainer teaching a hard concept from your area; teaching is the final
  test of understanding.
- Engage with the community: follow the key researchers, read the venues (the top conferences for
  your area), and join the conversation.

## Canonical references

- Your track's canonical course or book (CS224n, CS231n, Sutton & Barto, the Diffusion Course,
  or *Designing Machine Learning Systems*).
- The seminal and current papers of your chosen area.
