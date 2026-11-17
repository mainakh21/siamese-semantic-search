"""
End-to-end quickstart: train a small model, then run a few semantic searches.
Trains from scratch each time (a few seconds on CPU).

These queries are novel (not in the training corpus) and use everyday wording,
to show the model generalising by meaning rather than exact-sentence match.
For the fuller, honest picture — including the queries it gets wrong — run
`python src/evaluate.py`.

Run:  python examples/quickstart.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train
from src.search import SemanticIndex

QUERIES = [
    "how do I invest my savings",          # -> finance
    "how long should I bake the vegetables",  # -> cooking
    "will it rain this weekend",           # -> weather
    "my laptop is very slow",              # -> technology
    "book a cheap flight",                 # -> travel
    "our team won the game",               # -> sports
]


def main():
    print("Training the Siamese encoder...\n")
    model, vocab = train()

    index = SemanticIndex(model, vocab)
    for q in QUERIES:
        print(f'\nquery: "{q}"')
        for sent, topic, score in index.search(q, k=3):
            print(f"  {score:5.3f}  [{topic:<10}] {sent}")


if __name__ == "__main__":
    main()
