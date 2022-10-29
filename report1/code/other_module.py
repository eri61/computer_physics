import matplotlib.pyplot as plt
import numpy as np


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
    
      
def plot(results, x:str, y:str, **kwargs):
    fig, ax = plt.subplots()
    for data in results.T.items():
        name, val = data
        r = val.to_dict()
        _kwargs = dict()
        if ('label' in kwargs.keys()):
            _kwargs['label'] = r[kwargs['label']]
        ax.plot(r[x], r[y], marker='.', markersize=3, **_kwargs)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    assert 'xlabel' in kwargs.keys(), ("xlabel is not specified")
    assert 'ylabel' in kwargs.keys(), ("ylabel is not specified")
    ax.set_xlabel(kwargs['xlabel'])
    ax.set_ylabel(kwargs['ylabel'])
    ax.set_aspect('equal')  # plot x and y in an equal scale
    ax.legend(fontsize='x-small')
    # fig.show()
    # fig.savefig("../pic/newton_angles.png")

