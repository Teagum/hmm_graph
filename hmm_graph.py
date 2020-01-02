# Licensed under the terms of the BSD-3-Clause license.
# Copyright (C) 2019 Michael Blaß
# mblass@posteo.net

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches
import numpy as np
import networkx as nx
from scipy.spatial import distance


def scale(inp: np.ndarray, new_min: float = 0., new_max: float = 1.,
          axis: int = -1) -> np.ndarray:
    """Scale `x` between `new_min` and `new_max`.

    Args:
        inp:     Array to be scaled.
        new_min: Lower bound.
        new_max: Upper bound.

    Returns:
        One-dimensional array of transformed values.
    """
    xmax = inp.max(axis=axis, keepdims=True)
    xmin = inp.min(axis=axis, keepdims=True)
    a = (inp-xmin) / (xmax - xmin)
    y = a * (new_max - new_min) + new_min
    return y


def _prepare_fig(pos: dict) -> tuple:
    """Prepare a figure with the correct size.

    Args:
        pos: {node_name_i: np.array([pos_x, pos_y]), ...}
                      
    Returns:
        figure, axis
    """
    pos_data = np.array(list(pos.values()))
    diameter = distance.pdist(pos_data).max()
    dd = diameter / 2 + 1

    fig = plt.figure(figsize=(7, 7), frameon=False)
    ax = fig.add_subplot(111)
    r = 10
    ax.axis([-(dd+r), (dd+r), -(dd+r), (dd+r)])
    ax.set_axis_off()

    return fig, ax


def _draw_nodes(ax: mpl.axes.Subplot, graph: nx.classes.Graph,
                pos: dict, draw_labels: False) -> dict:
    """Draw the nodes of a (small) networkx graph.

    Params:
        ax:    Axis
        graph: Networkx graph object.
        pos:   Positions computed by nx.layout methods.
        

    Return:
        (dict) of Circle patches.
    """
    degree = np.array([deg for node, deg in graph.degree], dtype=float)
    degree /= degree.sum()

    flare_kwargs = {'alpha'    : 0.2,
                    'edgecolor': (0, 0, 0, 1),
                    'facecolor': None}

    node_kwargs = {'alpha'    : 0.8,
                   'edgecolor': (0, 0, 0, 1),
                   'facecolor': None}

    nodes = {}
    node_params = zip(pos.items())

    for i, (label, xy) in enumerate(pos.items()):
        size = graph.nodes[label]['size']
        fsize = graph.nodes[label]['fsize']
        flare_kwargs['facecolor'] = 'C{}'.format(i)
        flare = patches.Circle(xy, fsize, **flare_kwargs)

        node_kwargs['facecolor'] = 'C{}'.format(i)
        node = patches.Circle(xy, size, **node_kwargs)

        ax.add_patch(flare)
        ax.add_patch(node)
        if draw_labels:
            font_style = {'size':15, 'weight':'bold'}
            text_kwargs = {'color': (0, 0, 0, .8),
                       'verticalalignment': 'center',
                       'horizontalalignment': 'center',
                       'fontdict': font_style}
            ax.text(*xy, i+1, **text_kwargs)

        nodes[label] = node
    return nodes


def _draw_edges(ax: mpl.axes.Subplot, graph: nx.classes.Graph,
                pos, nodes: dict) -> dict:
    """Draw the edges of a (small) networkx graph.

    Args:
        ax:    Axis.
        graph: Networkx graph object.
        pos:   Positions computed by nx.layout methods.
        nodes: Circle patches.

    Returns:
        Dictionary of Circle patches.
    """
    pointer = patches.ArrowStyle.Fancy(head_width=10, head_length=15)
    curved_edge = patches.ConnectionStyle('arc3', rad=.2)

    arrow_kwargs = {'arrowstyle': pointer,
                    'antialiased': True,
                    'connectionstyle': curved_edge,
                    'edgecolor': None,
                    'facecolor': None,
                    'linewidth': 1.}

    edges = {}
    for i, (a, b, attr) in enumerate(graph.edges.data()):
        arrow_kwargs['edgecolor'] = attr['color']
        arrow_kwargs['facecolor'] = attr['color']

        edge = patches.FancyArrowPatch(pos[a], pos[b],
                               patchA=nodes[a], patchB=nodes[b],
                               shrinkA=5, shrinkB=5,
                               **arrow_kwargs)
        ax.add_patch(edge)
        edges[(a, b)] = edge
    return edges


def _legend(ax: mpl.axes.Subplot, graph: nx.classes.Graph,
            nodes: list) -> mpl.legend.Legend:
    """Draw the legend for a (small) nx graph.

    Args:
        ax:    Axis.
        grap:  A networkx graph object.
        nodes: Circle patches.

    Returns:
        AxesSubplot
    """
    legend_kwargs = {'fancybox': True,
                     'fontsize': 14,
                     'bbox_to_anchor': (1.02, 1.0)}

    labels = [r'$f_c = {:>9.3f}$ Hz'.format(key) for key in graph.nodes.keys()]
    legend = ax.legend(nodes.values(), labels, **legend_kwargs, borderaxespad=0)
    return legend


def draw_hmm(sdm: np.ndarray, tpm: np.ndarray, delta: np.ndarray,
                layout='circular', draw_labels=False):
    """Draw the graph of a HMM's transition probability matrix.

    Args:
        sdm:    State dependent means.
        gamma:  Transition probability matrix.
        delta:  Initial distribution.
        layout: Either 'circular', or 'spring'

    Returns:
        figure, axis, graph
    """
    graph = nx.MultiDiGraph()
    
    scaled_sdm = scale(sdm, 1, 5)
    for i, from_state in enumerate(sdm):
        graph.add_node(from_state, fsize=scaled_sdm[i])

        for j, to_state in enumerate(sdm):
            if not np.isclose(tpm[i, j], 0.):
                graph.add_edge(from_state, to_state,
                           weight=tpm[i, j],
                           color='k')

    sd = np.sum([np.exp(degree) for node, degree in graph.degree()])
    for node, degree in graph.degree():
        graph.node[node]['size'] = .5 + np.exp(degree) / sd

    if layout == 'circular':
        pos = nx.layout.circular_layout(graph, center=(0., 0.), scale=10)
    elif layout == 'spring':
        pos = nx.layout.spring_layout(graph, center=(0., 0.), scale=10)
    else:
        raise ValueError("Layout must be 'spring', or 'circular'")

    fig, ax = _prepare_fig(pos)
    nodes = _draw_nodes(ax, graph, pos, draw_labels)
    edges = _draw_edges(ax, graph, pos, nodes)
    legend = _legend(ax, graph, nodes)

    return fig, ax, graph
