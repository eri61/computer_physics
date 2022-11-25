import numpy as np
from scipy.integrate import solve_ivp


def f_kdv(t, u, df1, df3):
    u_x = df1.dot(u)
    u_xxx = df3.dot(u)
    return -6.0 * u * u_x - u_xxx


def main(op_df1, op_df3):
    # x mesh
    nx = 1000
    x_max = 100.0
    x = np.linspace(0, x_max, nx, endpoint=False)
    dx = x[1] - x[0]
    print("dx =", dx)

    # initial condition
    u0 = np.sin(x * (2.0 * np.pi / x_max))

    print("Solving equation...")
    t_max = 10.0
    sol = solve_ivp(f_kdv, (0, t_max), u0, dense_output=True, args=(op_df1, op_df3), rtol=1e-8)
    print(sol.message)
    print(" Number of time steps :", sol.t.size)
    print(" Minimum time step    :", min(np.diff(sol.t)))
    print(" Maximum time step    :", max(np.diff(sol.t)))

    # t mesh
    nt = 101
    t = np.linspace(0, t_max, nt)
    dt = t[1] - t[0]
    print("dt =", dt)

    # get u(t, x)
    u_xt = sol.sol(t)  # u(x, t)
    u_tx = u_xt.T  # u(t, x)
    print("shape of u(t, x) :", u_tx.shape)

    # Save results
    np.savez("kdv_solve_ivp", x=x, t=t, u_tx=u_tx)


if __name__ == '__main__':
    main()