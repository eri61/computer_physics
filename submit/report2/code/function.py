import matplotlib.pyplot as plt
import numpy as np


def f_kdv(t, u, df1, df3):
    u_x = df1.dot(u)
    u_xxx = df3.dot(u)
    return -6.0 * u * u_x - u_xxx

def soliton(x:np.ndarray, x0=5., lam=1.):
    """initial values

    Args:
        x (np.ndarray): coordinates
        x0 (float or ndarray, optional): if type is ndarray, the shape should be (1, x). Defaults to 5..
        lam (float or ndarray, optional): float or ndarray. if x0 is ndarray, type of lam mast be float. Defaults to 1..

    Returns:
        initial values: ndarray
    """
    if type(x0) == np.ndarray:
        assert x0.shape[0] == 1, "shape of x0 is wrong"
    if type(lam) == np.ndarray:
        assert lam.shape[0] == 1, "shape of lam is wrong"
    cosh = np.cosh(np.sqrt(lam) * (x - x0) / 2)
    return lam / (2 * cosh**2)
