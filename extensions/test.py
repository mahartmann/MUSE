import torch
import numpy as np
from torch import tensor


if __name__=="__main__":
    a = torch.nn.Linear(3,3)
    print(a.weight.detach().numpy())
    i = np.array([[1,2,3], [3,4,5], [4,5,6]])
    t = torch.from_numpy(i)
    a.weight = torch.nn.Parameter(t)
    print(a.weight.detach().numpy())
