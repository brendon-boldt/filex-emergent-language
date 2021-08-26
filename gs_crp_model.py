from typing import Any

import torch  # type: ignore
from torch import nn
import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from tqdm import tqdm  # type: ignore
from joblib import Parallel, delayed  # type: ignore

from nav_to_center.experiment_configs import log_range


class Model(nn.Module):
    def __init__(self, gsm_size: int, tau: float) -> None:
        super(self.__class__, self).__init__()
        self.gsm_size = gsm_size
        self.tau = tau

        self.weights = nn.parameter.Parameter(torch.zeros(1, self.gsm_size, dtype=torch.float32))

    def forward(self, x) -> torch.Tensor:
        x *= self.weights
        if self.training:
            x = nn.functional.gumbel_softmax(x, dim=-1, tau=self.tau)
        else:
            x = nn.functional.one_hot(x.argmax(-1), num_classes=self.gsm_size)
        return x


rng = np.random.default_rng()


def get_entropy(model) -> float:
    model.eval()
    with torch.no_grad():
        n = 0x200
        bs = 0x40
        counts = np.zeros(model.gsm_size, dtype=np.int32)
        for _ in range(n):
            _input = torch.FloatTensor(
                rng.lognormal(size=(bs, model.gsm_size)),
                # np.zeros(size=(bs, model.gsm_size)),
            )
            output = model(_input).sum(0).detach().numpy()
            counts += output
        props = counts / counts.sum()
        entropy = -(np.log2(props.clip(1e-10)) * props).sum()
    model.train()
    return entropy


def do_run(
    n_iters,
    eval_freq,
    bs,
    lr,
    gsm_size,
    tau,
) -> Any:
    model = Model(gsm_size, tau)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xs = []
    for i in range(n_iters):
        _input = torch.tensor(
            np.ones((bs, model.gsm_size)),
            # rng.lognormal(
            #     size=(
            #         bs,
            #         model.gsm_size,
            #     )
            # ),
            dtype=torch.float32,
        )

        optimizer.zero_grad()

        output = model(_input)
        loss = criterion(output, output.detach().argmax(-1))
        loss.backward()
        optimizer.step()

        # if i == 0 or not (i + 1) % eval_freq:
        #     xs.append(get_entropy(model))
    # return xs
    breakpoint()
    return get_entropy(model)


params = {
    "n_iters": 30000,
    "eval_freq": 1000,
    "bs": 0x20,
    "lr": 1e-3,
    "gsm_size": 0x20,
    "tau": 1,
}

# for lr in [1e-4, 1e-3, 1e-2, 1e-1]:
jobs = []
xs = []
n = 100
for val in log_range(1e-3, 1e0, n):
    params['lr'] = val
    xs.append(val)
    jobs.append(delayed(do_run)(**params))

# ys = Parallel(n_jobs=20)(j for j in tqdm(jobs))
ys = Parallel(n_jobs=1)(j for j in tqdm(jobs))
# for y in ys:
#     plt.plot(y)
plt.scatter(np.log2(xs), ys)
plt.savefig("crp_plot.pdf", format="pdf")
