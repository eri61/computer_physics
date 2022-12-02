import matplotlib.pyplot as plt
import numpy as np
from differential import make_differential_ops
from scipy.integrate import solve_ivp


def solve_soliton(
    soliton, params=[5., 1], save_name:str="kdv_solve_ivp",
    acc:int=4
    ):
    # x mesh
    nx = 100
    x_max = 10.0
    x = np.linspace(0, x_max, nx, endpoint=False)
    dx = x[1] - x[0]
    print("dx=", dx)

    # initial condition
    u0 = soliton(x, *params)

    # differential operators
    op_df1 = make_differential_ops(1, acc, nx, dx)
    op_df3 = make_differential_ops(3, acc, nx, dx)

    print("Solving equation...")
    t_max = 10.0
    sol = solve_ivp(f_kdv, (0, t_max), u0, dense_output=True, args=(op_df1, op_df3), rtol=1e-8)
    print(sol.message)
    print(" Number of time steps :", sol.t.size)
    print(" Minimum time step    :", min(np.diff(sol.t)))
    print(" Maximum time step    :", max(np.diff(sol.t)))

    # t mesh
    nt = 101
    t_max = 5
    t = np.linspace(0, t_max, nt)
    dt = t[1] - t[0]
    print("dt =", dt)

    # get u(t, x)
    u_xt = sol.sol(t)  # u(x, t)
    u_tx = u_xt.T  # u(t, x)
    print("shape of u(t, x) :", u_tx.shape)

    # Save results
    np.savez(f"../out/{save_name}", x=x, t=t, u_tx=u_tx)

def view_solit_t0(path:str):
    npz = np.load(path)
    x = npz['x']
    t = npz['t']
    ut = npz['u_tx']
    variable = path.split("/")[-1].split(".")[-2].split("_")
    plt.plot(x, ut[t==0.].flatten(), label= variable[0] + "=" + variable[1])

def view_solit_t065(path:str):
    npz = np.load(path)
    x = npz['x']
    t = npz['t']
    ut = npz['u_tx']
    variable = path.split("/")[-1].split(".")[-2].split("_")
    plt.plot(x, ut[t==0.65].flatten(), label= variable[0] + "=" + variable[1])

def f_kdv(t, u, df1, df3):
    u_x = df1.dot(u)
    u_xxx = df3.dot(u)
    return -6.0 * u * u_x - u_xxx

def soliton(x:np.ndarray, x0=5., lam=1.):
    if type(x0) == np.ndarray:
        assert x0.shape[0] == 1, "shape of x0 is wrong"
    if type(lam) == np.ndarray:
        assert lam.shape[0] == 1, "shape of lam is wrong"
    cosh = np.cosh(np.sqrt(lam) * (x - x0) / 2)
    return lam / (2 * cosh**2)

def two_soliton(
    x, x0:float=3., lam:float=5., 
    x1:float=1., lam1:float=3.
    ):
    u0 = soliton(x, x0, lam)
    u1 = soliton(x, x1, lam1)
    return u0 + u1
