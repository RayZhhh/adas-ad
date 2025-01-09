import json
import os
from typing import Dict, List

import numpy as np

from .expt_group import ExptGroup
from .utils import get_funcs_and_scores_with_extend

try:
    from matplotlib import pyplot as plt
except:
    pass

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def _get_mean_std_scores_for_group(expt_group: ExptGroup, num_samples, steps: List = None, valid_only=False):
    scores = [get_funcs_and_scores_with_extend(p, num_samples, valid_only=valid_only, optimal_value=o)[1]
              for p, o in zip(expt_group.paths, expt_group.optimal_values)]
    scores = np.array(scores, dtype=np.float64)
    # scores = np.maximum.accumulate(scores, axis=1)
    scores = np.minimum.accumulate(scores, axis=1)
    if steps:
        scores = scores[:, steps]
    return np.mean(scores, axis=0), np.std(scores, axis=0)


def plot_expts(expt_groups: List[ExptGroup],
               num_samples: int,
               steps: List[int],
               figsize=(6, 4),
               title='',
               valid_only=False,
               save_file_path=None,
               baseline_score=None,
               baseline_name='Baseline',
               logscale=True,
               **kwargs):
    """Plot multiple experiments. Plot mean+-std for each parameter setting.
    Args:
        num_samples: maximum samples number visualized in the figure
        expt_groups: {'param_setting1': ['log_path1', 'log_path2', 'log_path3', ...], ...}
        valid_only: only count valid functions (which the score is not None)
        save_file_path: the path to save the figure
        figsize: the size of plt figure
        title: title of the figure
        steps: the steps that the user want to visualize
        **kwargs: 'xlabel', 'ylabel', 'ylim'
    """
    assert len(expt_groups) <= 10, print('Too much methods, support less or equal than 10.')
    _colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:red', 'tab:grey', 'tab:brown', 'tab:purple', 'tab:pink', 'black']
    _markers = ['o', '^', 'v', 's', '*', '<', '>', '+', 'x', 'D']
    _step = np.arange(len(steps))

    method_names = [g.label for g in expt_groups]
    all_mean, all_std = [], []

    for group in expt_groups:
        m, s = _get_mean_std_scores_for_group(group, num_samples + 1, steps, valid_only=valid_only)
        all_mean.append(m)
        all_std.append(s)
    plt.figure(figsize=figsize, dpi=500)

    for g, m, s, c, l, mark in zip(
            expt_groups, all_mean, all_std, _colors, method_names, _markers
    ):
        plt.plot(_step, m, color=c, label=l, marker=mark, markerfacecolor='none', markersize=6, linestyle=g.kwargs.get('linestyle', None))
        plt.fill_between(_step, m - s, m + s, color=c, alpha=0.1)

    if baseline_score is not None:
        plt.plot(_step, [baseline_score] * len(steps), color='black', linestyle='--', alpha=0.5, label=baseline_name)

    plt.ylabel(kwargs.get('ylabel', 'Score'))
    plt.xlabel(kwargs.get('xlabel', 'Number of Evaluated Functions'))
    plt.title(title)
    plt.xticks(_step, [str(i) for i in steps])
    if logscale:
        plt.yscale('log')
    ylim = kwargs.get('ylim', None)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.legend(fontsize=7)
    plt.grid(alpha=0.2)

    if save_file_path is not None:
        plt.savefig(save_file_path, bbox_inches='tight')

    plt.show()
