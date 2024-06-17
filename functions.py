import random

import numpy as np
import numpy.random as rd
import scipy.linalg as la
from numpy import pi


def allDiffs(vector):
    diffs = np.array([])

    for i, x in enumerate(vector[:-1]):
        diffs = np.concatenate((diffs, vector[(1 + i) :] - x))

    return diffs


def randomTargetAngles(numberOfTargets, aperture=0, rng=None):
    if rng is None:
        rng = rd.default_rng()

    angles = rng.uniform(-pi / 2, pi / 2, numberOfTargets)

    min_diff = 2 * 0.886 / aperture if aperture > 0 else 0

    while np.min(allDiffs(angles)) <= min_diff:
        angles = rng.uniform(-pi / 2, pi / 2, numberOfTargets)

    return angles


def uniform_target_angles(
    number_of_targets, aperture=0, noise_std=np.pi / (65 * 3), rng=None
):
    if rng is None:
        rng = rd.default_rng()

    angles = np.linspace(-np.pi / 2, np.pi / 2, int(number_of_targets) + 2)

    if noise_std > 0:
        if rng is None:
            rng = rd.default_rng()

        min_diff = 2 * 0.886 / aperture if aperture > 0 else 0

        noisy_angles = angles[1:-1] + rng.normal(0, noise_std, int(number_of_targets))

        while np.min(allDiffs(noisy_angles)) <= min_diff:
            noisy_angles = angles[1:-1] + rng.normal(
                0, noise_std, int(number_of_targets)
            )

        return noisy_angles

    return angles[1:-1]


def generateSumArrayResponse(
    targetAngles,
    numberOfTransmitPositions,
    numberOfReceivePositions,
    elementSpacing=0.5,
):
    virtualPositions = elementSpacing * (
        np.arange(numberOfTransmitPositions + numberOfReceivePositions - 1)
        - (numberOfTransmitPositions + numberOfReceivePositions - 2) / 2
    )

    return np.exp(2 * pi * 1j * np.outer(virtualPositions, targetAngles))


def worst_crb(sum_array_response, multiplicity):
    return np.max(
        np.real(
            np.diag(
                la.inv(
                    sum_array_response.conj().T
                    @ np.diag(multiplicity)
                    @ sum_array_response
                )
            )
        )
    )


def sum_array_response_df(row):
    return generateSumArrayResponse(
        row["Target Angles"],
        row["Number of Transmit Positions"],
        row["Number of Receive Positions"],
        row["Element Spacing"],
    )


def worst_crb_df(row):
    return worst_crb(sum_array_response_df(row), np.convolve(row["Tx"], row["Rx"]))


def randomized_rounding(vector, M, trials=1, oneShot=False):
    result = np.zeros_like(vector)
    indices = list(range(len(vector)))

    while np.sum(result) < M:
        i_index = random.randrange(0, len(indices), 1)
        index = indices[i_index]
        result[index] = int(random.uniform(0, 1) < vector[index])
        if result[index] == 1:
            del indices[i_index]

    return result


def elementary_vector(length, index):
    vector = np.zeros(length)
    vector[index] = 1
    return vector


def mp_inv(matrix):
    if matrix.shape[0] > matrix.shape[1]:
        return la.inv(matrix.conj().T @ matrix) @ matrix.conj().T
    return matrix.conj().T @ la.inv(matrix @ matrix.conj().T)


def rank_one_approximation(A):
    U, _, Vh = np.linalg.svd(A, full_matrices=False)
    Al = U[:, 0].squeeze()
    Ar = Vh[0, :].squeeze()
    return Al, Ar


def min_eig_fisher(sum_array_response, multiplicity):
    fisher = np.real(
        sum_array_response.conj().T @ np.diag(multiplicity) @ sum_array_response
    )
    eigenvalues = np.linalg.eigvalsh(fisher)
    return eigenvalues[0]
