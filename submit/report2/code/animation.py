import pathlib as pth
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Example of ArtistAnimation
# https://matplotlib.org/stable/gallery/animation/dynamic_image.html

def save_animation(dir:Path=Path(".")) -> None:
    assert type(dir) == pth.PosixPath, \
        f"The {str(dir)} type expected was pathlib.PosixPath but {type(dir)} were given."
    # Load results
    npz = np.load(str(dir / "kdv_solve_ivp.npz"))
    print("npz.files =", npz.files)

    x = npz['x']
    t = npz['t']
    u_tx = npz['u_tx']
    print("x.shape =", x.shape)
    print("t.shape =", t.shape)
    print("u_tx.shape =", u_tx.shape)

    # make an animation
    print("Making animation...")
    make_animation(x, t, u_tx, ymin=-1.5, ymax=3.0, filename=str(dir / "kdv_solve_ivp.gif"))

def make_animation(x, t, u_tx, ymin, ymax, filename) -> None:
    fig, ax = plt.subplots()

    # common setting for plot
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x)$")
    ax.set_ylim((ymin, ymax))

    artists = []  # list of plot
    for i in range(t.size):
        # Make i-th plot
        # ax.set_title("t = %f" % t[i])
        artist = ax.plot(x, u_tx[i, :], '-b')
        artist += [ax.text(0.05, 1.05, "t = %.2f" % t[i], transform=ax.transAxes)]
        artists.append(artist)

    # Make animation
    anim = animation.ArtistAnimation(fig, artists, interval=100, repeat=False)
    # plt.show()

    # Save animation
    anim.save(filename, writer="pillow")  # writer="pillow" or "imagemagick" for GIF
    print("saved as '{}'".format(filename))

