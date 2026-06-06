# Module 3.3 — Embeddings, Retrieval & RAG

**Mode:** curated breadth
**Gate:** ship gate
**Est. effort:** 10–14 focused hours

> Fine-tuning teaches a model new behavior. But often you do not need it to *learn* your data,
> you need it to *look things up* in your data at the moment it answers. That is retrieval, and
> retrieval-augmented generation (RAG) is the most common way real applications give an LLM
> access to private, current, or domain-specific knowledge. You will build one, and learn the
> ways it quietly fails.

## Why this matters

RAG is the workhorse pattern of applied LLM engineering. Most production LLM applications,
support bots, document assistants, search over internal knowledge, are some form of RAG. It
solves problems fine-tuning cannot: keeping answers current without retraining, grounding
responses in sources you can cite, and working with data far larger than any context window.

The technique rests on **embeddings**, a concept that ties back to the whole curriculum: a way
to turn text into vectors so that similar meanings are near each other in space. You met the
idea of representing data as vectors in Phase 1 and embeddings inside the transformer in Phase
2; here it becomes the engine of search. Building a RAG system, and then honestly evaluating
it, teaches you both the pattern and its many failure modes, which is where most real RAG
projects struggle.

## What you will be able to do

By the end of this module you will be able to:

- Explain what embeddings are and how semantic (vector) search works.
- Build a RAG pipeline: chunk documents, embed them, store them, retrieve, and generate.
- Evaluate a RAG system on both retrieval quality and answer quality.
- Diagnose the common ways RAG fails and what to do about them.

## Prerequisites

- [Module 3.1](./01-inside-large-language-models.md): how LLMs work and the role of prompting
  and context.

## The tools: what you will use and how they fit together

A RAG system is a small pipeline, and each tool owns one stage:

- **An embedding model: `sentence-transformers`.** Turns each chunk of text (and each query)
  into a vector that captures its meaning. Similar meanings produce nearby vectors.
- **A vector store: FAISS or Chroma.** Holds the chunk vectors and finds the nearest ones to a
  query vector quickly. FAISS is a fast local library; Chroma is a simple local vector database.
- **An orchestration framework: LlamaIndex (or LangChain).** Ties the stages together, the
  loading, chunking, embedding, retrieval, and the call to the LLM, so you do not wire every
  piece by hand. You can also build the pipeline directly to understand it, then adopt a
  framework.
- **An evaluation tool: RAGAS.** Scores a RAG system on retrieval and answer-quality metrics so
  you can measure it rather than eyeball it.

The flow: documents are chunked and embedded into the vector store once (indexing). At query
time, the question is embedded, the nearest chunks are retrieved, and those chunks are placed in
the LLM's prompt as context so it answers from them rather than from memory.

## Curated path

1. **Embeddings: the `sentence-transformers` documentation and quickstart.** Understand what an
   embedding is and compute semantic similarity between texts.
   https://www.sbert.net/

2. **Build a RAG system: the LlamaIndex starter tutorial.** A clean, end-to-end walk through
   loading documents, indexing them, and querying with retrieval. This is the practical spine.
   https://docs.llamaindex.ai/en/stable/getting_started/starter_example/

3. **A conceptual explainer of RAG.** Read a solid overview of the pattern, chunking strategies,
   hybrid search, and reranking, so you understand the choices the framework is making for you.
   The Hugging Face cookbook RAG entries are a good, current source.
   https://huggingface.co/learn/cookbook/index

4. **Evaluation: the RAGAS documentation.** How to measure faithfulness, answer relevance, and
   retrieval quality. Evaluation is the ship gate, so study this properly.
   https://docs.ragas.io/en/stable/

**Deliberately skip for now:** agentic and multi-hop RAG, and heavy production infrastructure.
Build one solid single-hop RAG system and evaluate it well.

## Background: the ideas in words

**Embeddings.** An embedding model maps text to a vector such that semantically similar texts
land near each other (measured by cosine similarity). This is what lets you search by *meaning*
rather than keywords: the query "how do I reset my password" can match a document about
"recovering account access" even with no shared words.

**Indexing.** Once, up front, you split your documents into **chunks** (passages small enough to
be precise but large enough to be meaningful), embed each chunk, and store the vectors in a
**vector store**. Chunking strategy matters more than beginners expect: too large and retrieval
is imprecise, too small and context is lost.

**Retrieval and generation.** At query time you embed the question, retrieve the top-k nearest
chunks, and insert them into the LLM's prompt with an instruction to answer from the provided
context. The model now answers grounded in your data, and can cite which chunks it used.

**Where RAG fails.** This is the part tutorials skip and the ship gate emphasizes. Retrieval can
miss the relevant chunk (so the model has nothing to work with), retrieve irrelevant chunks
(distracting the model), or the model can ignore the context and answer from memory anyway
(hallucinating despite retrieval). **Reranking** (a second, more precise scoring of retrieved
chunks) and **hybrid search** (combining keyword and vector search) address some of this.
Honest evaluation is the only way to know which failure you have.

## Knowledge check

1. What is an embedding, and how does semantic search use it to find relevant text?
2. Why does chunking strategy matter, and what goes wrong if chunks are too large or too small?
3. Walk through the two phases of a RAG system (indexing and query time).
4. When would you reach for RAG instead of fine-tuning, and vice versa?
5. Name three distinct ways a RAG system can fail, and a mitigation for each.
6. Why is evaluating retrieval quality separately from answer quality important?

## Project (ship gate)

**Build a RAG system over a real corpus, and evaluate it honestly.**

- Pick a document corpus you care about (documentation, a set of papers, a book, your own
  notes). Chunk, embed, and index it.
- Build the retrieval-and-generation pipeline so questions are answered from the corpus, with
  the retrieved sources shown.
- **Evaluate it with RAGAS** (or an equivalent): measure retrieval quality and answer
  faithfulness/relevance on a set of test questions.
- **Find and document failures.** Deliberately probe for the failure modes above, show
  examples, and try at least one improvement (better chunking, reranking, or a hybrid search),
  measuring whether it helped.

**Definition of done:** a working RAG system, a real evaluation with numbers, and a documented
analysis of its failures and your attempt to fix them.

## The workshop: ship it

Build this in its own repository, `modelwright-rag`, using the project habits from Module 0.2.

1. Set up the project:

```bash
mkdir modelwright-rag && cd modelwright-rag
uv init && uv add sentence-transformers faiss-cpu llama-index ragas && mkdir src data
```

2. Build the indexing and query pipeline in `src/`, point it at your corpus in `data/` (or
   document how to fetch it), and add an evaluation script using RAGAS.
3. Commit at checkpoints: "Index and retrieve over the corpus", then "End-to-end RAG answers",
   then "RAGAS evaluation and a failure-mode fix".
4. Add a README with example questions, your evaluation numbers, and the failure analysis.
5. Ship it:

```bash
gh repo create modelwright-rag --public --source=. --push
```

   (No `gh`? Create an empty public repo, then `git remote add origin <url>` and
   `git push -u origin main`.)

**Done when:** `modelwright-rag` is on GitHub, answers questions from your corpus with sources, and
the README reports a real evaluation plus what you learned about where it fails.

## Going deeper (optional)

- Add a reranker (a cross-encoder) and measure the precision gain.
- Read the original RAG paper (Lewis et al., 2020) for the idea's origin.
- Explore hybrid search (combining BM25 keyword search with vector search).

## Canonical references

- `sentence-transformers` documentation (SBERT).
- LlamaIndex documentation; Hugging Face cookbook (RAG).
- RAGAS documentation; Lewis et al. (2020), *Retrieval-Augmented Generation*.
