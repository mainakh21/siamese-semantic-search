"""
Train the Siamese encoder on same-topic / different-topic pairs, then save the
model weights and vocabulary so `search.py` can load them.

Two regularisers matter a lot on a corpus this small, or the model just
memorises the training sentences:
  - token-dropout augmentation: randomly drop ~30% of words from each sentence
    at training time, so the model can't lean on one exact phrasing;
  - dropout + weight decay in the model/optimiser.

Run:  python src/train.py
"""

import os
import sys
import pickle
import random

import torch
from torch.utils.data import Dataset, DataLoader

# Make sibling packages importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import Vocab, make_pairs, all_sentences, tokenize
from src.model import SiameseEncoder, ContrastiveLoss

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
MAX_LEN = 16


def encode_augmented(vocab, text, token_dropout=0.0, rng=None, max_len=MAX_LEN):
    """Encode text to padded ids, optionally dropping a fraction of tokens."""
    ids = [vocab.stoi.get(t, vocab.unk_idx) for t in tokenize(text)]
    if token_dropout > 0 and rng is not None and len(ids) > 3:
        kept = [i for i in ids if rng.random() > token_dropout]
        ids = kept or ids
    ids = ids[:max_len]
    ids += [vocab.pad_idx] * (max_len - len(ids))
    return ids


class PairDataset(Dataset):
    def __init__(self, pairs, vocab, token_dropout=0.3):
        self.pairs = pairs
        self.vocab = vocab
        self.token_dropout = token_dropout
        self.rng = random.Random(1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b, label = self.pairs[i]
        return (
            torch.tensor(encode_augmented(self.vocab, a, self.token_dropout, self.rng)),
            torch.tensor(encode_augmented(self.vocab, b, self.token_dropout, self.rng)),
            torch.tensor(label, dtype=torch.float32),
        )


def train(epochs=20, batch_size=64, lr=1e-3, weight_decay=1e-4,
          token_dropout=0.3, margin=0.3, seed=0):
    torch.manual_seed(seed)

    sentences = [s for s, _ in all_sentences()]
    vocab = Vocab(sentences)

    pairs = make_pairs(n_pairs=3000, seed=seed)
    loader = DataLoader(PairDataset(pairs, vocab, token_dropout), batch_size=batch_size, shuffle=True)

    model = SiameseEncoder(vocab_size=len(vocab), pad_idx=vocab.pad_idx)
    loss_fn = ContrastiveLoss(margin=margin)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for a, b, label in loader:
            optim.zero_grad()
            loss = loss_fn(model(a), model(b), label)
            loss.backward()
            optim.step()
            total += loss.item() * len(label)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {total / len(pairs):.4f}")

    model.eval()  # turn off dropout so downstream embedding/search is deterministic
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "model.pt"))
    with open(os.path.join(ARTIFACT_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    print(f"\nSaved model + vocab to {ARTIFACT_DIR}/")
    return model, vocab


if __name__ == "__main__":
    train()
