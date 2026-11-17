# Siamese Semantic Search

A small, from-scratch **Siamese bi-encoder** for semantic search, built in PyTorch to make the mechanism transparent. You train one shared-weight network to map sentences into a vector space where *meaning* becomes *geometry* — similar sentences land close together — and then retrieve by nearest-neighbour search. This is the same idea that powers production sentence embeddings and vector search in engines like Elasticsearch and OpenSearch.

> Built as a learning project while working through search → vectors → LangGraph. The goal is understanding, so every file is short and commented — including an honest look at where a tiny from-scratch model falls down.

## The idea in one picture

```
            ┌──────────────────────────────┐
query ─────▶│   Siamese encoder (shared)   │────▶  query vector ┐
            └──────────────────────────────┘                    │  cosine
            ┌──────────────────────────────┐                    ├─ similarity ─▶ ranked results
document ──▶│   Siamese encoder (shared)   │────▶  doc vector  ┘
            └──────────────────────────────┘
```

"Siamese" just means there is **one** network, applied to **both** items in a pair. Because both go through identical weights, the distance between their output vectors is a meaningful measure of semantic similarity.

## How it works

1. **Encoder** (`src/model.py`) — token ids → embedding lookup → masked mean-pooling → dropout → small MLP → L2-normalised vector. Normalising to unit length means the dot product between two vectors *is* their cosine similarity.
2. **Contrastive loss** (`src/model.py`) — pulls same-meaning pairs together and pushes different-meaning pairs apart beyond a margin.
3. **Training data** (`data/dataset.py`) — a topic-structured corpus where same-topic sentences are "similar" and different-topic sentences are "dissimilar." No downloads required. Swap in real labelled pairs (duplicate questions, Q/A, NLI) and nothing else changes.
4. **Search** (`src/search.py`) — embed the corpus once, embed the query, rank by cosine similarity.
5. **Evaluation** (`src/evaluate.py`) — the part most tutorials skip. Measures both the structure learned on the corpus and accuracy on a labelled set of *novel* queries.

## Run it

```bash
pip install -r requirements.txt

# Train, then run example searches end-to-end (a few seconds on CPU):
python examples/quickstart.py

# Measure it honestly (corpus structure + novel-query accuracy):
python src/evaluate.py

# Train once, then query interactively:
python src/train.py
python src/search.py "how do I invest my savings"
```

## Example

Novel queries (not in the training corpus) retrieve the right topic by meaning:

```
query: "how do I invest my savings"
  0.991  [finance ] the central bank raised rates to cool down inflation
  0.989  [finance ] I want to open a tax-free savings account

query: "will it rain this weekend"
  0.993  [weather ] it has been unusually warm and sunny this week
  0.982  [weather ] thunderstorms are expected later tonight
```

Note the query words don't need to appear verbatim in the matched sentence — that's semantic matching, not keyword overlap.

## Honest evaluation & limitations

Run `src/evaluate.py` and you get two very different numbers:

- **Corpus structure: leave-one-out topic accuracy 100%.** The model cleanly separates the sentences it trained on (avg intra-topic similarity ≈ 0.99 vs inter-topic ≈ 0.69).
- **Novel-query accuracy: ~70%.** On brand-new queries it mostly works — but it reliably fails when a query uses a **word form the tiny corpus never contained** (e.g. "rainy" when training only had "rain", or "fit" which never appeared). An out-of-vocabulary content word becomes `<unk>` and carries no meaning.

That gap is the whole lesson. A model trained from scratch on 64 sentences has no knowledge of language beyond those sentences, so it can't generalise to unseen vocabulary. **This is exactly why production systems use *pretrained* bi-encoders** (e.g. `sentence-transformers`): they are the same Siamese architecture, but trained on millions of pairs, so they already understand "rainy ≈ rain" and far beyond. This repo is for seeing the machine; reach for a pretrained encoder when you want it to actually work at scale.

## How this maps to real vector search

Keep the same shape, upgrade the pieces:

| This repo | Production equivalent |
|---|---|
| From-scratch encoder, toy corpus | A pretrained bi-encoder (`sentence-transformers`) — the same Siamese idea at scale |
| In-memory tensor of vectors | A `dense_vector` field (Elasticsearch) or `knn_vector` field (OpenSearch) |
| Brute-force dot product | Approximate nearest-neighbour (HNSW) index inside the engine |
| Semantic only | **Hybrid search**: combine with BM25 keyword scores (e.g. Reciprocal Rank Fusion) — usually the biggest real-world relevance win |

## Project layout

```
siamese-semantic-search/
├── data/dataset.py      # tokenizer, vocab, topic-structured pair dataset
├── src/model.py         # Siamese encoder + contrastive loss
├── src/train.py         # training loop (augmentation + regularisation), saves model
├── src/search.py        # in-memory vector index + query
├── src/evaluate.py      # corpus diagnostics + novel-query accuracy
├── examples/quickstart.py
└── requirements.txt
```

## Next steps to extend it

- Replace the from-scratch encoder with `sentence-transformers` and watch the novel-query accuracy jump — the OOV failures above should disappear.
- Push embeddings into an Elasticsearch `dense_vector` / OpenSearch `knn_vector` field and run k-NN against the engine instead of in memory.
- Add a BM25 baseline and build a hybrid ranker; compare both with precision@k / nDCG on a labelled query set.
- Swap contrastive loss for triplet loss (anchor / positive / negative) and measure the difference.
