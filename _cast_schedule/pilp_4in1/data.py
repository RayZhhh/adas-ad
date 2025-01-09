import numpy as np


def get_40():
    """num_vars=40
    Wi_cycle = [650] * 10 + [650] * 13
    """
    eta = 0.95308
    w = [154, 136, 57, 55, 67, 83, 187, 20, 123, 50]
    r = [3, 2, 2, 2, 4, 3, 2, 3, 3, 4]
    cycle = [650] * 10 + [650] * 13
    return np.array(w), np.array(r), eta, cycle


def get_310():
    """num_vars=310
    Wi_cycle = [650] * 10 + [650] * 13
    """
    eta = 0.99256
    w = [175, 145, 65, 55, 95, 75, 195, 20, 125, 50]
    r = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    cycle = [650] * 10 + [650] * 13
    return np.array(w), np.array(r), eta, cycle


def get_320():
    """num_vars=320
    Wi_cycle = [650] * 10 + [650] * 13
    """
    eta = 0.96154
    w = [175, 145, 65, 55, 95, 75, 195, 20, 125, 50]
    r = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    cycle = [650] * 10 + [650] * 13
    return np.array(w), np.array(r), eta, cycle


def get_1k():
    """num_vars=1k
    Wi_cycle = [650] * 10 + [650] * 13
    """
    eta = 0.99462
    w = [175, 145, 65, 55, 95, 75, 195, 20, 125, 50]
    r = [63, 65, 65, 65, 65, 65, 65, 65, 65, 65]
    cycle = [650] * 10 + [650] * 13
    return np.array(w), np.array(r), eta, cycle


def get_2k():
    """num_vars=2k
    Wi_cycle = [650] * 10 + [650] * 13
    """
    eta = 0.996
    w = [175, 145, 65, 55, 95, 75, 195, 20, 125, 50]
    r = [127, 130, 130, 130, 130, 130, 130, 130, 130, 130]
    cycle = [650] * 10 + [650] * 13
    return np.array(w), np.array(r), eta, cycle


def get_50k():
    """num_var=50k
    Wi_cycle = [650] * 10 + [500] * 13
    """
    eta = 0.997
    w = [79, 66, 31, 26, 44, 35, 88, 9, 57, 22]
    r = [6240, 6262, 6217, 6267, 6262, 6172, 6076, 6052, 6017, 6012]
    cycle = [650] * 10 + [500] * 13
    return np.array(w), np.array(r), eta, cycle


def get_100k():
    """num_vars=10w, H=1w
    Wi_cycle = [650] * 10 + [500] * 13
    """
    eta = 0.997
    r = [12560, 12562, 12517, 12567, 12562, 12172, 12076, 12052, 12017, 12012]
    w = [79, 66, 31, 26, 44, 35, 88, 9, 57, 22]
    cycle = [650] * 10 + [500] * 13
    return np.array(w), np.array(r), eta, cycle


def get_1m():
    """num_vars=1m
    Wi_cycle = [650] * 10 + [500] * 13
    """
    eta = 0.997
    r = [125600, 125620, 125170, 125670, 125620, 121720, 120760, 120520, 120170, 120120]
    w = [79, 66, 31, 26, 44, 35, 88, 9, 57, 22]
    cycle = [650] * 10 + [500] * 13
    return np.array(w), np.array(r), eta, cycle
