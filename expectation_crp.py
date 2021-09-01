import numpy as np
from matplotlib import pyplot as plt  # type: ignore
import matplotlib
from tqdm import tqdm  # type: ignore
from joblib import Parallel, delayed  # type: ignore

from nav_to_center.experiment_configs import log_range



def entropy(counts):
    x = counts / max(counts.sum(), 1)
    x = x[x > 0]
    return -(x * np.log2(x)).sum()

def do_run(
        alpha,
        beta,
        time_steps,
        ):
    counts = np.zeros((max_size,), dtype=np.float32)
    counts[0] = 1.0
    xs = []
    rng = np.random.default_rng()
    for i in range(time_steps):
        probs = counts.copy()
        probs[np.flatnonzero(probs == 0)[0]] = alpha
        probs /= probs.sum()
        # print(beta)
        # print(rng.choice(len(counts), beta, p=probs))
        counts += eye[rng.choice(len(counts), beta, p=probs)].sum(0) / beta
        # if not (i + 1) % 100:
        #     print(entropy(counts))
        #     # xs.append(entropy(counts))
    return entropy(counts)

max_size = 2 ** 12
eye = np.eye(max_size)


n = 200

xs = []
ys = []
params = {
    "alpha": 5.0,
    "beta": int(1e2),
    "time_steps": int(1e3),
}

# for val in log_range(0x4, 0x400, n):
# for val in tqdm(log_range(2 ** 4, 2 ** 16, n), total=n):
jobs = []
# for val in log_range(1, 1e3, n):
#     val = max(int(val), 1)
#     params['beta'] = val
lo = 1.1
hi = 1000
for val in log_range(lo, hi, n):
    val = int(val)
    params['beta'] = val
    xs.append(val)
    jobs.append(delayed(do_run)(**params))
ys = Parallel(n_jobs=20)(j for j in tqdm(jobs))
plt.scatter(np.log10(xs), ys)

# jobs = []
# params['beta'] = 1
# for val in log_range(lo, hi, n):
#     params['alpha'] = val
#     jobs.append(delayed(do_run)(**params))
# ys2 = Parallel(n_jobs=20)(j for j in tqdm(jobs))

# ys = Parallel(n_jobs=1)(j for j in (jobs))
# plt.scatter(np.log10(xs), ys2)
# for y in ys:
#     plt.plot(y)
# plt.hist(ys, bins=10)
plt.savefig("results/ecrp.pdf", format="pdf")
