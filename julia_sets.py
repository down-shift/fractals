import numpy as np 
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")


def iterate(z: complex, c: complex = (-0.5251993 + 0.5251993j)) -> complex:
    return z ** 2 + c


def make_grid(re_min: float = -1.5, re_max: float = 1.5, 
              im_min: float = -1.5, im_max: float = 1.5,
              n_points: int = 500) -> np.array:
    re_axis = np.linspace(re_min, re_max, num=n_points)
    im_axis = np.linspace(complex(0, im_min), complex(0, im_max), num=n_points)
    return re_axis + im_axis.reshape(-1, 1)


def julia_set(grid: np.array, max_iter: int = 200, max_radius: float = 7) -> np.array:
    nondivergent_iters = np.zeros(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            z = grid[i][j]
            for curr_iter in range(max_iter):
                z = iterate(z)
                if abs(z) > max_radius:
                    nondivergent_iters[i][j] = curr_iter
                    break
    return nondivergent_iters


def plot_julia_set(jul_set: np.array):
    plt.figure(figsize=(8, 8))
    plt.grid(False)
    plt.imshow(jul_set, cmap='magma')
    plt.show()


if __name__ == '__main__':
    grid = make_grid(n_points=600)
    jul_set = julia_set(grid)
    plot_julia_set(jul_set)
    