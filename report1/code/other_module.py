from re import T

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def quadratic_function(x, a, b, c):
    return a*x**2 + b*x + c

def get_maxL_improve_accuracy(x:np.ndarray, y:np.ndarray, split_num=1000):
    """2次関数のフィッティングを用いて到達距離最大となる角度thetaを求める。

    Args:
        x (np.ndarray): 角度theta, 4点程度
        y (np.ndarray): 到達距離L, 4点程度

    return:
        theta:float(角度), L:float(到達距離)
    """
    popt, _ = curve_fit(quadratic_function, x, y)
    theta = np.linspace(x.min(), x.max(), split_num)
    L = quadratic_function(theta, *popt)
    max_L = L.max()
    return theta[L == max_L].squeeze(), L[L == max_L].squeeze()

def reaching_point(xt, yt, num_exclude_point=10, theta=0):
    # exclude the origin
    xt = xt[num_exclude_point:]
    yt = yt[num_exclude_point:]

    # select the 2point of y closest to y=0
    assert (yt >= 0).sum() > 0, f"non data y > 0, {theta}"
    assert (yt <= 0).sum() > 0, "non data y < 0"
    ymin_greater_0 = yt[yt >= 0.].min()
    ymax_lessthan_0 = yt[yt <= 0.].max()

    L_over0 = xt[yt == ymin_greater_0]
    L_under0 = xt[yt == ymax_lessthan_0]

    # take the mean
    L = (L_over0 + L_under0) / 2
    abs_err = (L_over0 - L_under0) / 2
    return L[0], abs_err[0]
    
      
def plot(
    results:np.ndarray, 
    xlabel:str, ylabel, 
    fig=None, ax=None, **kwargs
    ):
    x, y = results.values.T
    plt.plot(x, y, '.', markersize=5, **kwargs)
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    assert xlabel is not None, ("xlabel is not specified")
    assert ylabel is not None, ("ylabel is not specified")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize='x-small')
    # fig.show()
    # fig.savefig("../pic/newton_angles.png")

