"""
Dataset + tokenizer for the Siamese semantic-search demo.

We build a small, *topic-structured* corpus: sentences are grouped by topic,
so two sentences from the same topic are semantically "similar" (label 1) and
two from different topics are "dissimilar" (label 0). That gives the Siamese
network a real signal to learn from without needing to download anything.

In the real world you would swap this out for labelled pairs (e.g. question/
answer pairs, duplicate-question datasets like Quora, or NLI data). The rest of
the code does not change.
"""

import re
import random
from collections import Counter

# --------------------------------------------------------------------------- #
# A tiny topic-structured corpus. Same topic => similar, different => dissimilar.
# --------------------------------------------------------------------------- #
TOPICS = {
    "finance": [
        "how should I invest my savings for retirement",
        "the stock market fell sharply after the interest rate decision",
        "diversifying a portfolio reduces investment risk",
        "I want to open a tax-free savings account",
        "inflation erodes the value of money held in cash",
        "what is a good strategy for paying off debt quickly",
        "index funds usually beat actively managed funds on fees",
        "the central bank raised rates to cool down inflation",
    ],
    "cooking": [
        "how long should I bake chicken breast in the oven",
        "this pasta recipe needs more garlic and olive oil",
        "let the dough rest before you roll it out",
        "add salt to the boiling water before the pasta",
        "the sauce tastes better after simmering for an hour",
        "what temperature do I roast vegetables at",
        "fresh basil makes the tomato soup taste amazing",
        "knead the bread dough until it becomes smooth",
    ],
    "weather": [
        "will it rain tomorrow afternoon in the city",
        "a cold front is moving in bringing strong winds",
        "the forecast predicts heavy snow over the weekend",
        "it has been unusually warm and sunny this week",
        "thunderstorms are expected later tonight",
        "the humidity makes the heat feel much worse",
        "pack an umbrella because showers are likely",
        "temperatures will drop below freezing overnight",
    ],
    "sports": [
        "the team won the championship in the final minute",
        "he scored a goal in the second half of the match",
        "the marathon runner trained for six months straight",
        "our basketball team lost by a narrow margin",
        "she broke the world record in the hundred metres",
        "the coach changed the lineup before the big game",
        "fans celebrated wildly after the last-second shot",
        "the tennis final went to a deciding fifth set",
    ],
    "travel": [
        "what are the best places to visit in Japan",
        "I need to book a cheap flight to Lisbon",
        "the train from Paris to Amsterdam is very scenic",
        "remember to renew your passport before the trip",
        "backpacking across Southeast Asia on a budget",
        "the hotel was right next to the old town square",
        "which neighbourhood should I stay in while in Rome",
        "we rented a car to drive along the coast",
    ],
    "technology": [
        "how do I speed up my slow laptop",
        "the new phone has a much better camera",
        "I need to back up my files to the cloud",
        "the software update fixed several security bugs",
        "machine learning models need a lot of training data",
        "my wifi keeps dropping the connection",
        "they released a faster processor this year",
        "debugging this code is taking all afternoon",
    ],
    "health": [
        "regular exercise improves your cardiovascular health",
        "I should drink more water throughout the day",
        "getting enough sleep is important for recovery",
        "a balanced diet includes plenty of vegetables",
        "stretching before a workout prevents injury",
        "chronic stress can affect your immune system",
        "walking ten thousand steps a day is a good goal",
        "cutting back on sugar helped me feel better",
    ],
    "music": [
        "she learned to play the guitar last summer",
        "the concert tickets sold out within minutes",
        "this album has a really catchy chorus",
        "he composes film scores for a living",
        "the band is going on tour across Europe",
        "I practice the piano for an hour every evening",
        "the song has an unforgettable bass line",
        "streaming changed how artists earn money",
    ],
}


def tokenize(text):
    """Lowercase and split into word-ish tokens."""
    return re.findall(r"[a-z0-9']+", text.lower())


class Vocab:
    """Maps tokens <-> integer ids. Index 0 is <pad>, index 1 is <unk>."""

    def __init__(self, sentences, min_freq=1):
        counter = Counter(tok for s in sentences for tok in tokenize(s))
        self.itos = ["<pad>", "<unk>"] + sorted(
            w for w, f in counter.items() if f >= min_freq
        )
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_idx = 0
        self.unk_idx = 1

    def encode(self, text, max_len=16):
        """Turn text into a fixed-length list of token ids (padded/truncated)."""
        ids = [self.stoi.get(t, self.unk_idx) for t in tokenize(text)][:max_len]
        ids += [self.pad_idx] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.itos)


def all_sentences():
    """Flat list of (sentence, topic) for the whole corpus."""
    return [(s, topic) for topic, sents in TOPICS.items() for s in sents]


def make_pairs(n_pairs=2000, seed=0):
    """
    Build (sentence_a, sentence_b, label) training pairs.
    label = 1 if same topic (similar), 0 if different topic (dissimilar).
    We keep it roughly balanced between positives and negatives.
    """
    rng = random.Random(seed)
    corpus = all_sentences()
    topics = list(TOPICS.keys())
    pairs = []
    for _ in range(n_pairs):
        if rng.random() < 0.5:
            # positive pair: two distinct sentences from the same topic
            topic = rng.choice(topics)
            a, b = rng.sample(TOPICS[topic], 2)
            pairs.append((a, b, 1))
        else:
            # negative pair: sentences from two different topics
            t1, t2 = rng.sample(topics, 2)
            a = rng.choice(TOPICS[t1])
            b = rng.choice(TOPICS[t2])
            pairs.append((a, b, 0))
    return pairs
