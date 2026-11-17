"""
Semantic search with the trained Siamese encoder.

Embed every sentence in the corpus once, then embed a query and return the
nearest sentences by cosine similarity. This is exactly the shape of a real
vector-search system — only the storage (an in-memory tensor here) and the
nearest-neighbour step (a brute-force dot product here) differ from what an
engine like Elasticsearch / OpenSearch does at scale with an ANN index.

Run:  python src/search.py "how do I grow my money"
"""

import os
import sys
import pickle

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import all_sentences
from src.model import SiameseEncoder

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
MAX_LEN = 16


def load():
    with open(os.path.join(ARTIFACT_DIR, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    model = SiameseEncoder(vocab_size=len(vocab), pad_idx=vocab.pad_idx)
    model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "model.pt")))
    model.eval()
    return model, vocab


@torch.no_grad()
def embed(model, vocab, texts):
    ids = torch.tensor([vocab.encode(t, MAX_LEN) for t in texts])
    return model(ids)  # already L2-normalised


class SemanticIndex:
    """A tiny in-memory vector index. Build once, query many times."""

    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.sentences = [s for s, _ in all_sentences()]
        self.topics = [t for _, t in all_sentences()]
        self.matrix = embed(model, vocab, self.sentences)  # (N, dim)

    def search(self, query, k=5):
        q = embed(self.model, self.vocab, [query])          # (1, dim)
        scores = (self.matrix @ q.T).squeeze(1)             # cosine sims (unit vectors)
        topk = torch.topk(scores, k=min(k, len(self.sentences)))
        return [
            (self.sentences[i], self.topics[i], scores[i].item())
            for i in topk.indices.tolist()
        ]


def main():
    query = " ".join(sys.argv[1:]) or "how do I grow my money"
    model, vocab = load()
    index = SemanticIndex(model, vocab)
    print(f'\nquery: "{query}"\n')
    for sent, topic, score in index.search(query, k=5):
        print(f"  {score:5.3f}  [{topic:<10}] {sent}")


if __name__ == "__main__":
    main()
