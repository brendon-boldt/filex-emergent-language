from typing import Any

import torch  # type: ignore
from torch import nn
import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from tqdm import tqdm  # type: ignore


class Model(nn.Module):
    def __init__(self, gsm_size: int, tau: float) -> None:
        super(self.__class__, self).__init__()
        self.gsm_size = gsm_size
        self.tau = tau

        self.linear1 = nn.Linear(gsm_size, gsm_size, bias=False)

    def forward(self, x, one_hot=False) -> torch.Tensor:
        x = self.linear1(x)
        if one_hot:
            x = nn.functional.one_hot(x.argmax(-1), num_classes=self.gsm_size)
        else:
            x = nn.functional.gumbel_softmax(x, dim=-1, tau=self.tau)
        return x


rng = np.random.default_rng()

gsm_size = 0x20
tau = 1


def get_entropy(model) -> float:
    n = 0x200
    bs = 0x40
    counts = np.zeros(gsm_size, dtype=np.int32)
    for _ in range(n):
        _input = torch.FloatTensor(
            rng.normal(size=(bs, model.gsm_size)),
        )
        output = model(_input, one_hot=True).sum(0).detach().numpy()
        counts += output
    props = counts / counts.sum()
    entropy = -(np.log2(props.clip(1e-10)) * props).sum()
    return entropy


def do_run() -> Any:
    model = Model(gsm_size, tau)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xs = []
    for i in tqdm(range(n_iters)):
        _input = torch.tensor(
            rng.normal(
                size=(
                    bs,
                    model.gsm_size,
                )
            ),
            dtype=torch.float32,
        )

        optimizer.zero_grad()

        output = model(_input)
        loss = criterion(output, output.detach().argmax(-1))
        loss.backward()
        optimizer.step()

        if i == 0 or not (i + 1) % 2000:
            xs.append(get_entropy(model))
    # plt.scatter(xs, ys)
    return xs


n_iters = 30000
bs = 0x32
lr = 1e-3

for lr in [1e-4, 1e-3, 1e-2, 1e-1]:
    xs = do_run()
    plt.plot(xs)
plt.savefig("crp_plot.pdf", format="pdf")


do_run()
