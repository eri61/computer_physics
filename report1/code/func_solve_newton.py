import importlib

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import other_module

importlib.reload(other_module)
from other_module import reaching_point


def improve_accuracy_L() -> pd.DataFrame:

    return 3

def get_theta_of_maxL(
    theta_range, t_range, n_theta, n_t, bm:float=1, g:float=9.81, v0=100./3.6
    ) -> pd.DataFrame:

    """到達距離最大となる角度thetaを返す関数

    Args:
        theta_range (tuple(theta_start, theta_end)): 到達距離を計算する角度の範囲, [rad]
        t_range (tuple(t_start, t_end)): 計算する時間の範囲
        n_theta (int): theta_range内で分割する個数
        n_t (int): t_rangeの範囲で分割する時間の個数
        bm (b/m): 抵抗係数/質量
        g (float, optional): 重力加速度. Defaults to 9.81.

    Returns:
        _type_: _description_
    """

     # Results will be stored into a list as pandas
    L = pd.DataFrame(columns=['theta', 'l', "err"])
    theta_start, theta_end = theta_range
    theta_rad = np.linspace(theta_start, theta_end, n_theta)

    # X0 = [x0, v0_x, y0, v0_y]
    Xinit = np.array([
        np.zeros_like(theta_rad), 
        v0 * np.cos(theta_rad), 
        np.zeros_like(theta_rad),
        v0 * np.sin(theta_rad)
        ])

    for i, (X0, theta) in enumerate(zip(Xinit.T, theta_rad)):
        # Solve Newton equation
        # x(t), y(t)
        xt, _, yt, _ = solve_newton(
            eq_params=(g, bm), X0=X0, t_range=t_range, n_t=n_t
            )

        # calculate the maximum distance of achieving
        l, err = reaching_point(xt, yt, num_exclude_point=10, theta=theta)
        L.loc[i, 'theta'], L.loc[i, 'l'], L.loc[i, 'err'] = theta, l, err
    
    max_L = L['l'].max()
    max_point = L[L['l'] == max_L]

    return max_point
    

# Newton equation
def f_newton(t, X, g, resist_coeff):
    # X = [x, v_x, y, v_y]
    # resist_coeff: b/m
    # g: gravity coefficient
    dXdt = np.array([
        X[1],              # dx/dt = v_x
        -(resist_coeff) * X[1],     # dv_x/dt = -(b/m) v_x
        X[3],              # dy/dt = v_y
        -(resist_coeff) * X[3] - g  # dv_y/dt = -(b/m) v_y - g
    ])
    return dXdt

def solve_newton(
    eq_params, 
    X0, 
    t_range, 
    n_t, 
    func=f_newton,
    ):
    t_start, t_end = t_range

    # Solve ODE
    sol = solve_ivp(func, (t_start, t_end), X0, args=eq_params, dense_output=True)
    file = "Log.info.for_solve_newton.conf"
    output_log(file, sol.message)

    # Get dense output for plot
    t = np.linspace(t_start, t_end, n_t)
    Xt = sol.sol(t)
    assert Xt.shape == (4, n_t)

    # shape: [[x(t)], [v_x(t)], [y(t)], [v_y(t) ]]
    return Xt

    

def output_log(file_name:str, _message:str):
    with open(file_name, "a") as wf:
        print(_message, file=wf)
