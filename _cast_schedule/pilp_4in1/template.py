template_rec = '''
import random
import numpy as np

def recombination(X1: np.ndarray, X2: np.ndarray, W: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Perform recombination between X1 and X2.
    Args:
        X1: shape=(H, N), solution 1
        X2: shape=(H, N), solution 2
        W : shape=(H,), the capacity of crucible in melt 0, 1, ..., H. An example of W can be np.array([650] * 10 +[500] * 13)
        w : shape=(N,), the weights of object 0, 1, ..., N. An example of w can be np.array([79, 66, 31, 26, 44, 35, 88, 9, 57, 22])
    """
    return X1
'''

template_mut = '''
import random
import numpy as np

def mutation(X: np.ndarray, W: np.ndarray, w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Perform mutation to the solution X.
    Args:
        X: shape=(H, N), individual solution
        W: shape=(H,), the capacity of crucible in melt 0, 1, ..., H. An example of W can be np.array([650] * 10 +[500] * 13)
        w: shape=(N,), the weights of object 0, 1, ..., N. An example of w can be np.array([79, 66, 31, 26, 44, 35, 88, 9, 57, 22])
        r: shape=(N,), the required copies of object 0, 1, ..., N
    Return:
        the mutated solution X
    """
    return X
'''

template_sel = '''
import random
import numpy as np
from typing import Tuple, List

def selection(Xs: List[np.ndarray], scores: List[float]) -> Tuple[int, int]:
    """Select two solutions based on there scores, and return their indices.
    Args:
        Xs: a list of solutions, each solution is a np.ndarray with shape=(H * N,).
        scores: a list which contains the scores of each solution.
    """
    return random.randint(0, len(scores) - 1), random.randint(0, len(scores) - 1)
'''

template_sur = '''
import random
import numpy as np
from typing import Tuple, List

def survival(new_population_and_old_pop: List[np.ndarray], scores: List[float], pop_size: int) -> List[int]:
    """Returns 'pop_size' indices for the survived individuals.
    Args:
        new_population_and_old_pop: a list of individuals that combines individuals in the new population and old population,
                                    the length of which is 'pop_size * 2'. Each individual is a 2-D numpy array.
        scores: the scores of each individual in new_population_and_old_pop, the length of which is 'pop_size * 2',
                the scores is higher the better.
        pop_size: the population size.
    Return:
        returns a list of survived individuals. the length of these indices equals to 'pop_size'.
    """
    indices = [i for i in range(len(new_population_and_old_pop))]
    indices = np.random.choice(indices, size=pop_size, replace=False)
    return indices
'''
