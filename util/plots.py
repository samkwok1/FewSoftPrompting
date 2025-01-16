from typing import (
    List,
    Tuple,
)
import os
import pandas as pd 
import numpy as np
import json

import colorsys
import seaborn as sns
from matplotlib.axes import Axes
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker
from sentence_transformers import SentenceTransformer, util
from scipy.stats import sem
from os import makedirs




def change_saturation(
    rgb: Tuple[float, float, float],
    saturation: float = 0.6,
) -> Tuple[float, float, float]:
    """
    Changes the saturation of a color by a given amount. 
    Args:
        rgb (tuple): rgb color
        saturation (float, optional): saturation chante. 
    """
    hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    saturation = max(0, min(hsv[1] * saturation, 1))
    return colorsys.hsv_to_rgb(hsv[0], saturation, hsv[2])

def get_palette(
    n: int = 3,
    palette_name: str = 'colorblind',
    saturation: float = 0.6,
) -> List[Tuple[float, float, float]]:
    """
    Get color palette
    Args:
        n (int, optional): number of colors. 
        palette (str, optional): color palette. Defaults to 'colorblind'.
        saturation (float, optional): saturation of the colors. Defaults to 0.6.
    """
    palette = sns.color_palette(palette_name, n)
    return [change_saturation(color, saturation) for color in palette]

def lighten_color(
    color, 
    amount=0.5, 
    desaturation=0.2,
) -> Tuple[float, float, float]:
    """
    Copy-pasted from Eric's slack.
    Lightens and desaturates the given color by multiplying (1-luminosity) by the given amount
    and decreasing the saturation by the specified desaturation amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Examples:
    >> lighten_color('g', 0.3, 0.2)
    >> lighten_color('#F034A3', 0.6, 0.4)
    >> lighten_color((.3,.55,.1), 0.5, 0.1)
    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(h, 1 - amount * (1 - l), max(0, s - desaturation))

def get_fancy_bbox(
    bb, 
    boxstyle, 
    color, 
    background=False, 
    mutation_aspect=3,
) -> FancyBboxPatch:
    """
    Copy-pasted from Eric's slack.
    Creates a fancy bounding box for the bar plots.
    """
    if background:
        height = bb.height - 2
    else:
        height = bb.height
    if background:
        base = bb.ymin # - 0.2
    else:
        base = bb.ymin
    return FancyBboxPatch(
        (bb.xmin, base),
        abs(bb.width), height,
        boxstyle=boxstyle,
        ec="none", fc=color,
        mutation_aspect=mutation_aspect, # change depending on ylim
        zorder=2)

def plot_proposal_line_graph(ax: Axes,
                             x: list,
                             graph_title: str,
                             xlabel: str,
                             ylabel: str,
                             directory: str,
                             font_family: str = 'Avenir',
                             font_size: int = 35,
                             y_label_coords: Tuple[float, float] = (-0.07, 0.5),
                             y_ticks: List[int] = [0, 0.2, 0.4, 0.6, 0.8, 1],
                             y_ticklabels: List[int] = [0, 20, 40, 60, 80, 100],
                             y_lim: Tuple[float, float] = (-0.1, 1.1),
                             legend: bool = False,
                             legend_title: str = 'Model Type',
                             legend_loc: str = 'center left',
                             bbox_to_anchor: Tuple[float, float] = (1.0, 0.6),
                             ):
    plt.xlabel(xlabel, family=font_family, size=font_size + 10)
    sns.despine(left=True, bottom=False)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, fontsize=font_size)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True)) 

    ax.set_ylabel(ylabel, family=font_family, size=font_size + 10)

    pos = ax.get_position()  # Get the original position
    ax.set_position([pos.x0 + 50, pos.y0, pos.width, pos.height])
    # ax.yaxis.set_label_coords(*y_label_coords)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, size=font_size)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, zorder=-100)
    plt.ylim(y_lim)
    plt.subplots_adjust(left=0.1, right=0.8)

    #graph_title = f"Average_Proportion_of_Currency_Offered_to_the_Decider_Across_all_Iterations"
    plt.title(" ".join(graph_title.split('_')), family=font_family, size=font_size + 10)
    if legend:
        ax.legend(title=legend_title, 
                  frameon=False,
                  ncol=1, 
                  bbox_to_anchor=bbox_to_anchor,
                  loc=legend_loc,
                  fontsize=font_size,  # Change the font size of legend items
                  title_fontsize=font_size
                  )
    plt.tight_layout()
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}/{graph_title}.pdf', format='pdf')
    plt.clf()

def get_ax():
    _, ax = plt.subplots(figsize=(13, 11))
    palette = sns.color_palette("mako", 10)
    palette = [palette[i] for i in [0, 3, 6, 9]]
    x = [f"{shot}–shot" for shot in [0, 1, 3]]
    labels = ["Base Model", "5–token", "10–token", "20–token"]
    y = [
        [0.0657, 0, 0],
        [0.291, 0, 0],
        [0.217, 0, 0],
        [0.083, 0, 0]
    ]
    for i in range(4):
        ax.plot(x, y[i], color=palette[i], linewidth=5, zorder=1, label=labels[i])

    plot_proposal_line_graph(ax,
                             x,
                             "Model_Incorrect_Generation_on_Winogrande",
                             "Number of Shots",
                             "Percent Incorret",
                             "outputs/arc-c_gen",
                            )

if __name__ == "__main__":
    get_ax()

# wino:   [0.50, 0.506, 0.511],
#         [0.394, 0.496, 0.496],
#         [0.495, 0.496, 0.496],
#         [0.490, 0.496, 0.496]
    
# arc-c:  [0.193, 0.207, 0.207],
#         [0.163, 0.213, 0.197],
#         [0.003, 0.21, 0.21],
#         [0.186, 0.21, 0.214]
    
# arc-e:   [0.266, 0.280, 0.286],
#         [0.162, 0.164, 0.231],
#         [0.107, 0.279, 0.279],
#         [0.05, 0.2786, 0.2786]