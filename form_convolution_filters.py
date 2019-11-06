# Generate figure for form convolution figures

import numpy as np
import matplotlib.pyplot as plt
import math

plt.rcParams.update({'font.size': 10})

def norm_pdf(x, mu, sigma):
    if np.isinf(sigma):
        return np.ones(len(x), dtype="float") * (1 / len(x))
    else:
        return (1 / np.sqrt(2 * math.pi * (sigma ** 2))) * \
               np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))

spx = 3
spy = 2

fig, ax = plt.subplots(spy, spx, figsize=(10, 6))
fig.patch.set_alpha(0.0)

n_games = np.array([1, 2, 3, 4, 5, 5])
sigma = np.array([np.inf, np.inf, 1.5, 1.5, 1.5, 3])

for i in range(0, spx*spy):

    j, k = np.unravel_index(i, (spy, spx))
    response = np.zeros((2 * n_games[i]) + 1, dtype="float")
    response[(n_games[i] + 1):] = norm_pdf(np.array(range(n_games[i])), 0,
                                        sigma[i])
    response = response / np.sum(response)
    ax[j, k].plot(response)

    ax[j, k].title.set_text("games: " + str(n_games[i]) + ", sigma: " + str(
        sigma[i]))
    ax[j, k].title.set_size(10)

    plt.sca(ax[j, k])

    if k == 0:
        plt.ylabel("Relative Weighting")

plt.savefig("figures/form_filters.svg", bbox_inches='tight')

plt.show()