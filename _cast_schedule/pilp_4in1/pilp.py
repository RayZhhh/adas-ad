import copy
import time
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
import traceback


def fit_eval(X: np.ndarray, W: np.ndarray, w: np.ndarray, r: np.ndarray, *, R=1000, lo_bo=0, up_bo=1_000):
    """Evaluate the Casting Scheduling Problem.
    Args:
        X: shape=(H, N), individual solution
        W: shape=(H,), the size of crucible in melt 0, 1, ..., H
        w: shape=(N,), the weights of object 0, 1, ..., N
        r: shape=(N,), the required copies of object 0, 1, ..., N
        R: the penalty constant
        up_bo: the upper bound value of each element in X
        lo_bo: the lower bound value of each element in X
    """
    X = X.clip(min=lo_bo, max=up_bo)
    # calculate f(x)
    fx = (np.dot(X, w) / W).mean()
    # calculate constraint 1
    constr1 = ((X.sum(0) - r) ** 2).sum()
    # calculate constraint 2
    constr2 = (np.maximum(0, np.dot(X, w) / W - 1) ** 2).sum()
    return fx - R * (constr1 + constr2) - 1


def estimate_heats(cycle: list | np.ndarray, w: np.ndarray, r: np.ndarray, eta: float):
    """
    Args:
        cycle: the cycle, e.g., [650] * 10 + [500] * 13
        w: shape=(N,), the weights of object 0, 1, ..., N
        r: shape=(N,), the required copies of object 0, 1, ..., N
        eta: objective target
    """
    # total metal needed
    M = (r * w).sum()
    metal = 0
    i = 1
    while True:
        metal += eta * cycle[(i - 1) % len(cycle)]
        if metal >= M:
            break
        i += 1
    return i


def generate_W(H: int, cycle: list):
    """Generate W based on the number of heats H.
    """
    num_cycles = int(H / len(cycle) + 1)
    cycle = cycle * num_cycles
    return np.array(cycle)[:H]


def pilp(init: callable, gen_new_pop: callable,
         w: ndarray, r: ndarray, eta: float, cycle: list, pop_size=12, num_gen=30,
         *, mut_iter=1, verbose=False):
    """
    Args:
        init    : initialization function
        w       : shape=(N,), the weights of object 0, 1, ..., N
        r       : shape=(N,), the required copies of object 0, 1, ..., N
        eta     : target
        cycle   : the cycle of heats
        pop_size: population size
        num_gen : number of generations
    """

    assert len(w) == len(r)
    w = w.astype(np.int64)
    r = r.astype(np.int64)
    H: int = estimate_heats(cycle, w, r, eta)
    N: int = len(r)
    W: ndarray = generate_W(H, cycle).astype(np.int64)

    population = []
    scores = []

    # ================ population init ================
    for _ in range(pop_size):
        W_ = copy.deepcopy(W)
        w_ = copy.deepcopy(w)
        r_ = copy.deepcopy(r)
        indi = init(H, N, W_, w_, r_)
        del W_, w_, r_
        indi.astype(np.int64)
        population.append(indi)
    # =================================================

    # ================ fitness evaluation ================
    for indi in population:
        indi_ = copy.deepcopy(indi)
        W_ = copy.deepcopy(W)
        w_ = copy.deepcopy(w)
        r_ = copy.deepcopy(r)
        score = fit_eval(indi_, W_, w_, r_)
        del W_, w_, r_
        scores.append(score)
    # ====================================================

    if verbose:
        print(f'Iter 0, Best Fitness: {max(scores)}')

    for iter in range(num_gen - 1):
        population = gen_new_pop(copy.deepcopy(population), scores, copy.deepcopy(W), copy.deepcopy(w), copy.deepcopy(r), fit_eval)
        scores = []
        for i, x in enumerate(population):
            population[i] = x.astype(np.int64)
            score = fit_eval(population[i], copy.deepcopy(W), copy.deepcopy(w), copy.deepcopy(r))
            scores.append(score)

        if verbose:
            print(f'Iter {iter + 1}, Best Fitness: {max(scores)}')

    scores = [s for s in scores if s is not None]
    return max(scores)


if __name__ == '__main__':
    import numpy as np
    from typing import List
    import random
    from deb_op import init_deb_wo_mut


    def gen_new_population(pop: List[np.ndarray], scores: List[float], W: np.ndarray, w: np.ndarray, r: np.ndarray, evaluate: callable) -> List[np.ndarray]:
        """Perform selection, recombination, mutation, evaluation and survival and return a new population.
        The new population should have the same length with current population X.
        Args:
            pop: Current population, pop[i] is with shape=(H, N) indicating an individual solution.
            scores: scores[i] is the score pop[i], which is higher the better.
            W: shape=(H,), the capacity of crucible in melt 0, 1, ..., H. An example of W can be np.array([650] * 10 +[500] * 13).
            w: shape=(N,), the weights of object 0, 1, ..., N. An example of w can be np.array([79, 66, 31, 26, 44, 35, 88, 9, 57, 22]).
            r: shape=(N,), the required copies of object 0, 1, ..., N
            evaluate: the function to evaluate each solution X_i. The input of the function is a solution, W, w, r. The output is the score which is the higher the better.
        Return:
            new s
        """
        new_pop = copy.deepcopy(pop)
        scores = []
        for x in pop:
            score = evaluate(x, W, w, r)
            scores.append(score)
        return new_pop


    import data

    w, r, eta, cycle = data.get_100k()

    start = time.time()
    score = pilp(init_deb_wo_mut, gen_new_population, w, r, eta, cycle, verbose=True)
    print(score)
    print(f'time: {time.time() - start}')
