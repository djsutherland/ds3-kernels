import torch


def as_tensors(X, *rest):
    "Calls as_tensor on a bunch of args, all of the first's device and dtype."
    X = torch.as_tensor(X)
    return [X] + [
        None if r is None else torch.as_tensor(r, device=X.device, dtype=X.dtype)
        for r in rest
    ]
