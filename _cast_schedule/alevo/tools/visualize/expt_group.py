from __future__ import annotations

import copy
from typing import List, Dict


class ExptGroup:
    def __init__(self,
                 label: str,
                 paths: List[str],
                 optimal_values: List[float | int] = None, **kwargs):
        """
        Args:
            label: label of the experiment.
            paths: experiment paths.
            optimal_values: optimal vales of each path.
        """
        self.label = label
        self.paths = paths
        self.optimal_values = optimal_values
        self.kwargs = kwargs

        if optimal_values is None:
            self.optimal_values = [None] * len(paths)
        elif isinstance(optimal_values, (int, float)):
            self.optimal_values = [optimal_values] * len(paths)


def merge_group_by_label(expt_groups: List[ExptGroup]):
    """Merge the experiment group by grouping experiments with same labels.
    """
    label_dict = {}
    for group in expt_groups:
        if group.label not in label_dict:
            label_dict[group.label] = copy.deepcopy(group)
        else:
            label_dict[group.label].paths += group.paths
            label_dict[group.label].optimal_values += group.optimal_values
    return [v for k, v in label_dict.items()]