# type: ignore

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from tqdm import tqdm

from nav_to_center.experiment_configs import log_range


def get_score(tgt, x):
    return np.cos((tgt - x) * 2 * np.pi) / tau


# get_score = np.vectorize(_get_score, excluded={0})


def get_probs(counts, tgt):
    fitnesses = get_score(
        tgt.reshape(-1, 1),
        np.arange(lex_size).reshape(1, -1) / lex_size,
    )
    fit_denom = np.exp(fitnesses).sum(-1, keepdims=True)
    fit_probs = np.exp(fitnesses) / fit_denom
    # scaled_alpha = alpha / max(1, (counts == 0).sum())
    # _counts = counts + (counts == 0) * scaled_alpha
    _counts = counts + alpha
    count_probs = (_counts / _counts.sum()).reshape(1, -1)
    final_probs = count_probs * fit_probs
    return final_probs / final_probs.sum(-1, keepdims=True)


rng = np.random.default_rng()


def get_vocab(counts):
    n = 1000
    x = np.arange(n) / n
    probs = get_probs(counts, x)
    uses = np.unique(probs.argmax(-1), return_counts=True)[1]
    return uses / n


def entropy(x):
    x = x[x > 0]
    return -(x * np.log2(x)).sum()


def do_run():
    counts = np.zeros((lex_size,), dtype=np.int32)
    xs = []
    xs.append(entropy(get_vocab(counts)))
    for i in range(1000):
        tgt = rng.uniform(0, 1)
        probs = get_probs(counts, np.array([tgt])).squeeze()
        idx = rng.choice(lex_size, p=probs)
        # print(f"{tgt:.2f}\t{idx / lex_size:.2f}")
        counts[idx] += 1
        # if not i % 10:
        #     xs.append(entropy(get_vocab(counts)))
    # return xs
    return entropy(get_vocab(counts))


xs = []
ys = []
n = 100

lex_size = 2 ** 5
alpha = 1.0
tau = 1e-1

# for val in log_range(0x4, 0x400, n):
for val in tqdm(log_range(2 ** 4, 2 ** 16, n), total=n):
    lex_size = int(val)
    xs.append(np.log2(int(val)))
    ys.append(do_run())
plt.scatter(xs, ys)
plt.savefig("plot.pdf", format="pdf")
