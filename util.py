import numpy as np  # type: ignore

from typing import Dict


def _xlx(x: float) -> float:
    if x == 0.0:
        return 0.0
    else:
        return -x * np.log2(x)


xlx = np.vectorize(_xlx)


def calc_entropies(o: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate entropies based on network outputs."""
    return {
        "argmax": xlx(np.eye(o.shape[-1])[o.argmax(-1)].mean(0)).sum(0),
        "fractional": xlx(o.mean(0)).sum(0),
        "individual": xlx(o).sum(-1).mean(0),
    }
