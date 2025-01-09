from dataclasses import dataclass
from typing import List

import numpy as np
from numpy import ndarray


def recomb_deb(X1: np.ndarray, X2: np.ndarray, W: np.ndarray, w: np.ndarray):
    """Perform recombination between X1 and X2.
    Args:
        X1: shape=(H, N), solution 1
        X2: shape=(H, N), solution 2
        W : shape=(H,), the size of crucible in melt 0, 1, ..., H
        w : shape=(N,), the weights of object 0, 1, ..., N
    """
    # calculate remain capacity
    U1 = W - np.dot(X1, w)
    U2 = W - np.dot(X2, w)
    U1 = U1.reshape((-1, 1))  # shape=(len(U1),1)
    U2 = U2.reshape((-1, 1))  # shape=(len(U2),1)
    # select the row with better metal utilization
    Xc = np.where(np.abs(U1) <= np.abs(U2), X1, X2)
    return Xc


def mut1_deb(X: ndarray, W: ndarray, w: ndarray, r: ndarray, *, mut1_iter: int = 500):
    """Perform 'mutation 1' to the given solution X, to make it fulfill constraints 1.
    Args:
        X        : shape=(H, N), individual solution
        W        : shape=(H,), the size of crucible in melt 0, 1, ..., H
        w        : shape=(N,), the weights of object 0, 1, ..., N
        r        : shape=(N,), the required copies of object 0, 1, ..., N
        mut1_iter: maximum iteration times in mutation 1

    Return:
        the mutated solution X
    """
    H, N = X.shape

    for j in range(N):
        for iteration_count in range(mut1_iter):
            # the sum of item_j
            sum_Xj = X[:, j].sum()
            # if the equation has already fulfilled
            if sum_Xj == r[j]:
                break
            # remaining capacity of crucible in each heat
            U = W - np.dot(X, w)
            # if more copies are assigned
            if sum_Xj > r[j]:
                # find the indices where copies larger than 0
                indices = np.nonzero(X[:, j] > 0)[0]
                if indices.size > 0:
                    # identify heat that occupies to minimum (or negative) space in crucible
                    idx_to_reduce = indices[np.argmin(U[indices])]
                    X[idx_to_reduce, j] -= 1
            else:
                # if less copies are assigned
                idx_to_increase = np.argmax(U)
                X[idx_to_increase, j] += 1
    return X


def mut2_deb(X: ndarray, W: ndarray, w: ndarray, *, mut2_iter: int = 1000):
    """Perform 'mutation 2' to the given solution X, to make it fulfill constraints 2.
    Args:
        X        : shape=(H, N), individual solution
        W        : shape=(H,), the size of crucible in melt 0, 1, ..., H
        w        : shape=(N,), the weights of object 0, 1, ..., N
        mut2_iter: maximum iteration times in mutation 2
    Return:
        the mutated X
    """
    for _ in range(mut2_iter):
        # we need 'weights_per_heat[i]' (kg) metal in heat_i
        weights_per_heat = np.dot(X, w)
        remaining_capacities = W - weights_per_heat
        # if all row fulfill constraints 2, finish iteration
        if np.all(remaining_capacities >= 0):
            break
        min_id = np.argmin(remaining_capacities)
        max_id = np.argmax(remaining_capacities)
        # find indices in min_id row, of which the copies > 0
        assigned_objects = np.nonzero(X[min_id, :] > 0)[0]
        if assigned_objects.size > 0:
            obj_id = np.random.choice(assigned_objects)
            X[min_id, obj_id] -= 1
            X[max_id, obj_id] += 1
    return X


def mut_deb(X: ndarray, W: ndarray, w: ndarray, r: ndarray):
    """
    Args:
        X : shape=(H, N), individual solution
        W : shape=(H,), the size of crucible in melt 0, 1, ..., H
        w : shape=(N,), the weights of object 0, 1, ..., N
        r: shape=(N,), the required copies of object 0, 1, ..., N

    Return:
        the mutated solution X
    """
    X = mut1_deb(X, W, w, r)
    X = mut2_deb(X, W, w)
    return X


def init_deb(H: int, N: int, W: ndarray, w: ndarray, r: ndarray, a: int | None = None):
    """Initialize an individual.
    Args:
        a: a fixed integer
        H: number of heats
        N: number of items
        W: shape=(H,), the size of crucible in melt 0, 1, ..., H
        w: shape=(N,), the weights of object 0, 1, ..., N
        r: shape=(N,), the required copies of object 0, 1, ..., N

    Return:
        an individual with shape=(H, N)
    """
    if a is None:
        a = max(w)
    random_values = np.random.randint(0, a + 1, size=(H, N))
    sum_Xj = random_values.sum(axis=0)

    # prevent dividing by 0
    X = np.where(sum_Xj != 0, np.round((random_values * r) / sum_Xj), 0).astype(np.int64)

    # mutation
    X = mut1_deb(X, W, w, r)
    X = mut2_deb(X, W, w)
    return X


def init_deb_wo_mut(H: int, N: int, W: ndarray, w: ndarray, r: ndarray, a: int | None = None):
    """Initialize an individual.
    Args:
        a: a fixed integer
        H: number of heats
        N: number of items
        W: shape=(H,), the size of crucible in melt 0, 1, ..., H
        w: shape=(N,), the weights of object 0, 1, ..., N
        r: shape=(N,), the required copies of object 0, 1, ..., N

    Return:
        an individual with shape=(H, N)
    """
    if a is None:
        a = max(w)
    random_values = np.random.randint(0, a + 1, size=(H, N))
    sum_Xj = random_values.sum(axis=0)

    # prevent dividing by 0
    X = np.where(sum_Xj != 0, np.round((random_values * r) / sum_Xj), 0).astype(np.int64)
    return X


def selection_deb(Xs: List[np.ndarray], scores: List[float]):
    indices1 = np.random.choice(range(len(scores)), size=2, replace=False).tolist()
    indices2 = np.random.choice(range(len(scores)), size=2, replace=False).tolist()
    idx1 = max(indices1, key=lambda i: scores[i])
    idx2 = max(indices2, key=lambda i: scores[i])
    return idx1, idx2


def survival_deb(new_population_and_old_pop: List[np.ndarray], scores: List[float], pop_size: int) -> List[int]:
    """Returns 'pop_size' indices for the survived individuals.
    Args:
        new_population_and_old_pop: a list of individuals that combines individials in the new population and old population,
                                    the length of which is 'pop_size * 2'. Each individual is a 2-D numpy array.
        scores: the scores of each individual in new_population_and_old_pop, the length of which is 'pop_size * 2',
                the scores is higher the better.
        pop_size: the population size.
    Return:
        returns a list of survived individuals. the length of these indices equals to 'pop_size'.
    """
    indices = np.argsort(scores)[::-1].tolist()
    return indices[:pop_size]
