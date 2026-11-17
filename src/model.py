"""
The Siamese bi-encoder.

A Siamese network is just ONE network ("one tower") with shared weights that we
apply to BOTH items in a pair. It maps any sentence to a fixed-size vector (an
"embedding"). Because both items go through the *same* weights, similar inputs
land near each other in vector space — which is exactly what you need for
semantic search.

Architecture here (deliberately simple so you can read every line):

    token ids --> embedding lookup --> masked mean pooling --> dropout --> MLP --> L2 normalize

The L2 normalisation at the end means every embedding is a unit vector, so the
dot product between two embeddings equals their cosine similarity.

Dropout is important here: with a tiny corpus the model will happily *memorise*
the exact training sentences and fail to generalise to new queries. Dropout
(plus the token-dropout augmentation in train.py and weight decay) pushes it to
rely on content words rather than memorising whole sentences. See the
"Limitations" section of the README for why this only goes so far.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=48, hidden_dim=64, out_dim=48,
                 pad_idx=0, dropout=0.2):
        super().__init__()
        self.pad_idx = pad_idx
        # Each token gets a learned vector. padding_idx keeps <pad> fixed at zero.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        # A small MLP that turns the pooled word vectors into a sentence embedding.
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, token_ids):
        """token_ids: (batch, seq_len) -> embeddings: (batch, out_dim), unit-norm."""
        # Mask out padding positions so they don't pollute the average.
        mask = (token_ids != self.pad_idx).unsqueeze(-1).float()   # (B, L, 1)
        emb = self.embedding(token_ids)                            # (B, L, E)
        summed = (emb * mask).sum(dim=1)                           # (B, E)
        counts = mask.sum(dim=1).clamp(min=1e-9)                   # (B, 1)
        mean = summed / counts                                     # masked mean pool
        out = self.proj(self.dropout(mean))                        # (B, out_dim)
        return F.normalize(out, p=2, dim=1)                        # unit vectors


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss over a pair of embeddings and a 0/1 label.

    - Similar pairs (label=1): minimise their cosine distance (pull together).
    - Dissimilar pairs (label=0): push them apart until they are at least
      `margin` apart; beyond that, no further penalty.

    distance = 1 - cosine_similarity  (in [0, 2] for unit vectors)
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, label):
        cos = (emb_a * emb_b).sum(dim=1)                 # cosine similarity
        dist = 1.0 - cos                                 # cosine distance
        label = label.float()
        pos = label * dist.pow(2)                        # pull similar together
        neg = (1 - label) * F.relu(self.margin - dist).pow(2)  # push dissimilar apart
        return (pos + neg).mean()
