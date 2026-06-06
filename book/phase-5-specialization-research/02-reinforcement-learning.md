# Module 5.2 — Reinforcement Learning

**Mode:** hybrid
**Gate:** build gate
**Est. effort:** 14–20 focused hours

> Every model so far learned from a fixed dataset. Reinforcement learning is different: an agent
> learns by acting in an environment and receiving rewards, discovering good behavior through
> trial and error. It is the framework behind game-playing agents, robotics, and, increasingly,
> the alignment of large language models (the RL in RLHF). You will use an agent first, then
> build the core algorithms yourself and solve a real control problem.

## Why this matters

Reinforcement learning is a distinct and powerful paradigm, and its importance is growing. It is
how models are taught to optimize for long-term outcomes rather than imitate a dataset, and it
sits at the heart of modern alignment (RLHF and its relatives, which you met conceptually in
Phase 3) and of the agentic systems that act over many steps. Understanding RL rounds out your
grasp of machine learning's three great paradigms, supervised, unsupervised, and reinforcement,
and opens a major branch of the field and the job market.

It is also conceptually rich in a way that sharpens you: the ideas of states, actions, rewards,
value, and policy, and the exploration-exploitation trade-off, are a different and beautiful way
of thinking about learning. Building an agent from scratch and watching it learn to balance a
pole or land a craft is one of the more magical experiences in the field.

## What you will be able to do

By the end of this module you will be able to:

- Frame a problem as a Markov decision process: states, actions, rewards, and the goal.
- Explain value-based and policy-based methods and the exploration-exploitation trade-off.
- Implement a core RL algorithm from scratch and use it to solve a standard environment.
- Use a mature RL library to train agents, and know when to reach for it.

## Prerequisites

- [Module 2.5](../phase-2-deep-learning/05-making-training-work.md): you can train neural
  networks, which deep RL uses as function approximators.
- From [Module 0.4](../phase-0-foundations/04-math-just-in-time.md): probability and expectation,
  which underpin returns and value functions.

## Curated path

1. **Experience it first: the Hugging Face Deep RL Course, Unit 1 hands-on.** Train an agent
   (using Stable-Baselines3 and Gymnasium) before building anything, to see RL working and meet
   the vocabulary in action.
   https://huggingface.co/learn/deep-rl-course/

2. **The foundations: Sutton & Barto, *Reinforcement Learning: An Introduction* (free).** The
   definitive textbook. Read the early chapters on Markov decision processes, value functions,
   temporal-difference learning, and Q-learning. The conceptual bedrock.
   http://incompleteideas.net/book/the-book-2nd.html

3. **Build deep RL: OpenAI Spinning Up.** The best bridge from theory to working deep RL code,
   with clean explanations of the key algorithms (policy gradients, PPO) and an emphasis on
   implementation. This is your guide for the build gate.
   https://spinningup.openai.com/

4. **Reference implementations: CleanRL.** Single-file, readable implementations of the major RL
   algorithms. Use them to check your own implementation and to see how a clean version is
   structured.
   https://github.com/vwxyzjn/cleanrl

5. **Optional lectures: David Silver's RL Course.** The classic lecture series for a deeper
   theoretical treatment, if you want it.

**Deliberately skip for now:** the more advanced and unstable algorithms and large-scale
distributed RL. Build one core algorithm well and solve a classic environment; depth comes after
the fundamentals are solid.

## Background: the ideas in words

**The setup.** An **agent** interacts with an **environment** in steps. At each step it observes
a **state**, takes an **action**, and receives a **reward** and the next state. The goal is to
learn a **policy** (a way of choosing actions) that maximizes cumulative reward over time. This
is formalized as a **Markov decision process**. Unlike supervised learning, there is no labeled
"right action"; the agent must discover good behavior from the reward signal alone, which is what
makes RL both powerful and hard.

**Value and policy.** Two broad families. **Value-based** methods (like Q-learning) learn the
expected long-term reward of actions and act greedily with respect to it; **Deep Q-Networks**
use a neural network to approximate this for large state spaces. **Policy-based** methods (like
policy gradients and PPO) directly learn the policy, adjusting it to make rewarding actions more
likely. **Actor-critic** methods combine both. Each family has strengths; policy-gradient
methods dominate much of modern deep RL and underpin RLHF.

**Exploration versus exploitation.** A central tension: the agent must **exploit** what it knows
works to get reward, but also **explore** new actions to discover something better. Balancing
the two is fundamental to RL and has no free lunch; strategies like epsilon-greedy and entropy
bonuses manage it.

**Why it is hard.** Rewards can be sparse and delayed (you only learn you played well at the end
of the game), the data the agent learns from depends on its own current behavior (a moving
target), and training can be unstable. These difficulties are why RL implementations are
notoriously finicky, and why building one teaches you so much.

## Knowledge check

1. Define state, action, reward, and policy, and state the agent's objective.
2. What makes reinforcement learning fundamentally different from supervised learning?
3. Contrast value-based and policy-based methods, and say what actor-critic combines.
4. Explain the exploration-exploitation trade-off and one way to manage it.
5. Why are sparse, delayed rewards a hard problem, and how does the idea of a value function
   help?
6. Where does reinforcement learning show up in modern large-language-model training?

## Build gate

Implement a core RL algorithm from scratch and solve a standard environment.

**Specification:**

- Choose a core algorithm: **DQN** (value-based) or a **policy-gradient** method (REINFORCE,
  then ideally a simple PPO) following Spinning Up.
- Implement it yourself in PyTorch (use Gymnasium for the environment, but write the agent and
  the learning algorithm yourself).
- Solve a standard control environment: **CartPole** at minimum, ideally **LunarLand**er.
- Verify against a reference (CleanRL) and against the environment's "solved" threshold.

**What it must show:**

- An agent that learns: a reward curve climbing to the environment's solved threshold.
- A correct implementation you can explain line by line (the gate behind the gate).
- A comparison to a reference implementation or the known solved score.

## Project

Package it and write it up (reuse your template):

- Your from-scratch RL agent, the training code, and reward curves.
- A report: the algorithm in your own words (the update rule and why it works), your results,
  and the hardest part of getting RL to train stably (there will be one).

**Definition of done:** a from-scratch agent that solves CartPole (or harder), with the
write-up.

## The workshop: ship it

Build this in its own repository, `mlforge-rl`, using the project habits from Module 0.2.

1. Set up the project:

```bash
mkdir mlforge-rl && cd mlforge-rl
uv init && uv add torch gymnasium matplotlib && mkdir src
```

2. Implement the agent and training loop in `src/`, train on Gymnasium, and save reward curves.
3. Commit at checkpoints: "Environment loop and random agent", then "Algorithm implemented",
   then "Agent solves CartPole (reward curve)".
4. Add a README with your reward curves, the algorithm explanation, and your stability notes.
5. Ship it:

```bash
gh repo create mlforge-rl --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `mlforge-rl` is on GitHub with a from-scratch agent that demonstrably solves a
standard environment, and a README that explains the algorithm.

## Going deeper (optional)

- Implement PPO fully and solve a harder environment (a Box2D or Atari task).
- Read about RLHF and how policy-gradient methods align language models, connecting to Phase 3.
- Explore model-based RL and offline RL as major branches.

## Canonical references

- Sutton & Barto, *Reinforcement Learning: An Introduction*.
- OpenAI, *Spinning Up in Deep RL*; CleanRL; Hugging Face *Deep RL Course*.
