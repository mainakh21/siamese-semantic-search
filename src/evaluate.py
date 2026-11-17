"""
Evaluate the trained encoder — honestly.

Two things worth measuring, and they tell different stories:

1. Structure the model learned on the corpus it was trained on:
     - leave-one-out nearest-neighbour topic accuracy
     - average intra-topic vs inter-topic similarity (the bigger the gap, the
       cleaner the clusters)

2. Generalisation to NOVEL queries it has never seen, using a small labelled
   query set (query -> expected topic). This is the number that actually
   matters for search, and it is deliberately not perfect: a from-scratch model
   trained on a tiny corpus struggles with word forms it never saw ("rainy" vs
   "rain"). That gap is the whole argument for pretrained bi-encoders in
   production — see the README.

Run:  python src/evaluate.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search import load, SemanticIndex, embed

# A small labelled evaluation set: novel queries the model never trained on.
# A realistic mix — most use in-vocabulary words; a few deliberately use word
# forms the corpus never contained, to expose the out-of-vocabulary weakness.
EVAL_QUERIES = [
    ("how do I invest my savings", "finance"),
    ("how long should I bake the vegetables", "cooking"),
    ("will it rain this weekend", "weather"),
    ("my laptop is very slow", "technology"),
    ("book a cheap flight", "travel"),
    ("our team won the game", "sports"),
    ("the stock market dropped today", "finance"),
    ("a concert tour across europe", "music"),
    ("will it be rainy later today", "weather"),       # OOV: "rainy" (corpus has "rain")
    ("i want to get fit and healthy", "health"),        # OOV: "fit"
]


def corpus_diagnostics(index):
    sims = index.matrix @ index.matrix.T
    topics = index.topics
    n = len(topics)
    correct = 0
    for i in range(n):
        row = sims[i].clone()
        row[i] = -1e9
        if topics[int(torch.argmax(row))] == topics[i]:
            correct += 1
    intra, inter = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            (intra if topics[i] == topics[j] else inter).append(float(sims[i, j]))
    return correct, n, sum(intra) / len(intra), sum(inter) / len(inter)


def query_accuracy(index):
    hits = 0
    rows = []
    for q, expected in EVAL_QUERIES:
        top = index.search(q, k=1)[0]
        ok = top[1] == expected
        hits += ok
        rows.append((ok, q, top[1], expected))
    return hits, len(EVAL_QUERIES), rows


def main():
    model, vocab = load()
    index = SemanticIndex(model, vocab)

    c, n, intra, inter = corpus_diagnostics(index)
    print("Corpus structure (sentences the model trained on)")
    print(f"  leave-one-out topic accuracy : {c}/{n} = {c/n:.2f}")
    print(f"  avg intra-topic similarity    : {intra:.3f}")
    print(f"  avg inter-topic similarity    : {inter:.3f}")

    hits, total, rows = query_accuracy(index)
    print(f"\nNovel-query topic accuracy: {hits}/{total} = {hits/total:.2f}")
    for ok, q, got, exp in rows:
        mark = "OK " if ok else "XX "
        note = "" if ok else f"  (got {got})"
        print(f"  {mark}{q!r} -> {exp}{note}")


if __name__ == "__main__":
    main()
