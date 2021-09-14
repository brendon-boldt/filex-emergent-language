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
    dirichlet=False,
):
    counts = np.zeros((max_size,), dtype=np.float32)
    counts[0] = 1.0
    xs = []
    rng = np.random.default_rng()
    for i in range(time_steps):
        probs = counts.copy()
        probs[np.flatnonzero(probs == 0)[0]] = alpha
        if dirichlet:
            counts += rng.dirichlet((1e-7 + probs) * beta)
        else:
            probs /= probs.sum()
            counts += eye[rng.choice(len(counts), beta, p=probs)].sum(0) / beta
        # if not (i + 1) % 100:
        #     print(entropy(counts))
        #     # xs.append(entropy(counts))
    return entropy(counts)


max_size = 2 ** 12
eye = np.eye(max_size)


xs = []
ys = []
params = {
    "alpha": 5.0,
    "beta": int(1e2),
    "time_steps": int(1e3),
    "dirichlet": False,
}
n = 300

# for val in log_range(0x4, 0x400, n):
# for val in tqdm(log_range(2 ** 4, 2 ** 16, n), total=n):
jobs = []
# for val in log_range(1, 1e3, n):
#     val = max(int(val), 1)
#     params['beta'] = val
lo = 1
hi = 1000
for val in log_range(lo, hi, n):
    val = int(val)
    params["beta"] = val
    xs.append(val)
    jobs.append(delayed(do_run)(**params))
ys = Parallel(n_jobs=20)(j for j in tqdm(jobs))

fig = plt.figure(figsize=(2, 1.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.scatter(np.log10(xs), ys, s=2.0)
ticks = [1, 10, 100, 1000]
ax.set_xticks([np.log10(x) for x in ticks])
ax.set_xticklabels(ticks)
ax.set_xlabel(r"$\beta$")
ax.set_yticks([3, 4, 5])

# jobs = []
# for val in log_range(lo, hi, n):
#     val /= 100
#     params['beta'] = val
#     jobs.append(delayed(do_run)(**params))
# ys2 = Parallel(n_jobs=1)(j for j in tqdm(jobs))
# plt.scatter(np.log10(xs), ys2)

# ys = Parallel(n_jobs=1)(j for j in (jobs))
# plt.scatter(np.log10(xs), ys2)
# for y in ys:
#     plt.plot(y)
# plt.hist(ys, bins=10)
fig.savefig("results/ecrp.pdf", bbox_inches="tight", format="pdf")
