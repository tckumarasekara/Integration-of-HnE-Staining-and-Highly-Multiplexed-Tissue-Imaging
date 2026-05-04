import torch
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, torch.nn.Sequential):
        for val in m:
            weights_init(val)
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


