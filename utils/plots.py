import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from utils import init_identity_grid_2D, init_identity_grid_3D


def plot_2D(v, transformation):
    # initialise the grid
    nx, ny = transformation.shape[3], transformation.shape[2]
    x, y = np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny)

    xv, yv = np.meshgrid(x, y, indexing='ij')
    xv, yv = np.expand_dims(xv, axis=0), np.expand_dims(yv, axis=0)

    # get the velocity field components
    v_x, v_y = v[0, 0], v[0, 1]

    # plot velocity
    fig_v, ax_v = plt.subplots()
    ax_v.quiver(xv, yv, v_x, v_y)

    ax_v.set_title('velocity')
    ax_v.set_xlabel('x')
    ax_v.set_ylabel('y')

    # get the displacement vectors
    identity_grid = init_identity_grid_2D(nx, ny)
    displacement_field = transformation - identity_grid.permute([0, 3, 1, 2])

    u, v = displacement_field[0, 0], displacement_field[0, 1]

    # plot displacement
    fig, ax = plt.subplots()
    ax.quiver(xv, yv, u, v)

    ax.set_title('displacement')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # show
    plt.show()


def plot_3D(v, transformation):
    # initialise the grid
    nx, ny, nz = transformation.shape[4], transformation.shape[3], transformation.shape[2]
    x, y, z = np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny), np.linspace(-1.0, 1.0, nz)

    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    xv, yv, zv = np.expand_dims(xv, axis=0), np.expand_dims(yv, axis=0), np.expand_dims(zv, axis=0)

    # get the velocity field components
    v_x, v_y, v_z = v[0, 0], v[0, 1], v[0, 2]

    # plot velocity
    fig_v = plt.figure()
    ax_v = fig_v.gca(projection='3d')
    ax_v.quiver(xv, yv, zv, v_x, v_y, v_z, length=0.25)

    ax_v.set_title('velocity')
    ax_v.set_xlabel('x')
    ax_v.set_ylabel('y')
    ax_v.set_zlabel('z')

    # get the displacement vectors
    identity_grid = init_identity_grid_3D(nx, ny, nz)
    displacement_field = transformation - identity_grid.permute([0, 4, 1, 2, 3])

    u, v, w = displacement_field[0, 0], displacement_field[0, 1], displacement_field[0, 2]

    # plot displacement
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(xv, yv, zv, u, v, w, length=0.25)

    ax.set_title('displacement')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # show
    plt.show()


def plot_grid(grid):
    grid = grid.cpu().numpy()
    grid_x, grid_y, grid_z = grid[0, 0], grid[0, 1], grid[0, 2]

    dim = grid_x.shape[0]
    idx = np.array([i for i in range(dim) if i % 16 == 0])
    idx = np.append(idx, [dim - 1])

    grid_x = grid_x[idx]
    grid_x = grid_x[:, idx]
    grid_x = grid_x[:, :, idx]

    grid_y = grid_y[idx]
    grid_y = grid_y[:, idx]
    grid_y = grid_y[:, :, idx]

    grid_z = grid_z[idx]
    grid_z = grid_z[:, idx]
    grid_z = grid_z[:, :, idx]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(grid_x, grid_y, grid_z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show(block=False)
    plt.pause(10)
    plt.close()
