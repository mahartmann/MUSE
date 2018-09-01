import torch
import numpy as np
from torch import tensor
import matplotlib.pyplot as plt


if __name__=="__main__":
    fig, ax = plt.subplots()
    x = [1,2]
    y = [1,2]
    labels = ['1', '2']
    ax.scatter([1,2], [1,2])
    for i , label in enumerate(labels):
        ax.annotate(labels[i], (x[i], y[i]))
    plt.show()