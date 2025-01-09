from typing import Any, Optional, Dict, List

import requests

from alevo.base import Evaluator, Sampler

from pilp_4in1.pilp import pilp
from pilp_4in1.data import get_50k, get_100k
from pilp_4in1.deb_op import init_deb_wo_mut


class EvalPxy(Evaluator):
    def __init__(self, timeout_seconds):
        super().__init__(timeout_seconds=timeout_seconds)

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        w, r, eta, cycle = get_50k()
        score = pilp(init=init_deb_wo_mut, gen_new_pop=callable_func, w=w, r=r, eta=eta, cycle=cycle, num_gen=30)
        return score


class EvalTar(Evaluator):
    def __init__(self):
        super().__init__(timeout_seconds=800)

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        w, r, eta, cycle = get_100k()
        score = pilp(init=init_deb_wo_mut, gen_new_pop=callable_func, w=w, r=r, eta=eta, cycle=cycle, num_gen=30)
        return score


class EvalPxy_API(Evaluator):  # noqa
    def __init__(self, url='http://47.119.189.240:9003/submit_task'):
        super().__init__(timeout_seconds=150)
        self.url = url

    def send_task_and_get_result(self, program):  # noqa
        response = requests.post(self.url, json={'programs': program, 'data': 'proxy'})
        if response.status_code == 200:
            result = response.json().get('result')
            # task_id = response.json().get("task_id")
            # print(f"Result for task {task_id}: {result}")
            return result

    def evaluate_program(self, program_str, callable_funcs, **kwargs) -> Any | None:
        score = self.send_task_and_get_result(program_str)
        return score


class EvalTar_API(Evaluator):  # noqa
    def __init__(self, url='http://47.119.189.240:9003/submit_task'):
        super().__init__(timeout_seconds=150)
        self.url = url

    def send_task_and_get_result(self, program):  # noqa
        response = requests.post(self.url, json={'programs': program, 'data': 'target'})
        if response.status_code == 200:
            result = response.json().get('result')
            # task_id = response.json().get("task_id")
            # print(f"Result for task {task_id}: {result}")
            return result

    def evaluate_program(self, program_str, callable_func, **kwargs) -> Any | None:
        score = self.send_task_and_get_result(program_str)
        return score


if __name__ == '__main__':

    a = '''\
import numpy as np
from typing import List
import random
import copy

def gen_new_population(pop: List[np.ndarray], scores: List[float], W: np.ndarray, w: np.ndarray, r: np.ndarray, evaluate: callable) -> List[np.ndarray]:
    """Perform selection, recombination, mutation, evaluation and survival and return a new population.
    The new population should have the same length with current population X.
    Args:
        pop: current population, pop[i] is with shape=(H, N) indicating an individual solution.
        scores: scores[i] is the score pop[i], which is higher the better.
        W: shape=(H,), the capacity of crucible in melt 0, 1, ..., H. An example of W can be np.array([650] * 10 +[500] * 13).
        w: shape=(N,), the weights of object 0, 1, ..., N. An example of w can be np.array([79, 66, 31, 26, 44, 35, 88, 9, 57, 22]).
        r: shape=(N,), the required copies of object 0, 1, ..., N
        evaluate: the function to evaluate each solution X_i. The input of the function is the solution, W, w, and r. The output is the score which is the higher the better.
    Return:
        new population
    """
    new_pop = copy.deepcopy(pop)
    scores = []
    for x in pop:
        score = evaluate(x, W, w, r)
        scores.append(score)
    return new_pop
    '''
    r = EvalPxy_API().evaluate_program(a, '')
    print(r)
    r = EvalTar_API().evaluate_program(a, '')
    print(r)
