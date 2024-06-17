import os
import re

import numpy as np
from matplotlib import pyplot as plt


def has_ext(filename):
    _, ext = os.path.splitext(filename)
    return len(ext) > 0


def savefig(filename, fig=None):
    if fig is None:
        fig = plt.gcf()
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)

    if has_ext(filename):
        fig.savefig(filename)
    else:
        fig.savefig(filename + ".pdf", backend="pgf")
        fig.savefig(filename + ".pgf")
        fig.savefig(filename + ".svg")


def style(stylename):
    dirname = os.path.dirname(os.path.realpath(__file__))
    filename = dirname + "/" + stylename + ".mplstyle"

    plt.style.use(filename)

    pattern = re.compile(
        r"^\s*figure\.figsize\s*:\s*(?P<width>\d*\.?\d*)\s*,\s*(?P<height>\d*\.?\d*)$"
    )
    with open(filename, "r") as f:
        for line in f:
            figsize = pattern.match(line)
            if figsize:
                figsize = np.array(
                    [float(figsize.group("width")), float(figsize.group("height"))]
                )
                break

    return figsize
