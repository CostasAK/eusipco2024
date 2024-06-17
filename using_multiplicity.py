import cvxpy as cp
import numpy as np
import scipy.linalg as la
from scipy.fft import fft

from functions import (
    randomized_rounding,
    mp_inv,
    rank_one_approximation,
    min_eig_fisher,
)


def hot_start_tx_rx(
    multiplicity,
    number_of_transmit_positions,
    number_of_receive_positions,
    number_of_transmitters=None,
    number_of_receivers=None,
):
    sum_array_size = number_of_transmit_positions + number_of_receive_positions - 1

    dft_sum = la.dft(sum_array_size, scale="sqrtn")
    dft_transmit = dft_sum[:, :number_of_transmit_positions]
    dft_receive = dft_sum[:, :number_of_receive_positions]

    diagonal_selection = np.zeros((sum_array_size, sum_array_size**2))

    for i in range(sum_array_size):
        diagonal_selection[i, i * sum_array_size + i] = 1

    outer_pt_pr = np.reshape(
        mp_inv(diagonal_selection @ np.kron(dft_receive.conj(), dft_transmit))
        @ dft_sum
        @ multiplicity,
        (number_of_transmit_positions, number_of_receive_positions),
        order="F",
    ) / np.sqrt(sum_array_size)

    pt, pr = rank_one_approximation(outer_pt_pr)

    if np.sum(pt) + np.sum(pr) < 0:
        pt *= -1
        pr *= -1

    pt = np.real(pt)
    if number_of_transmitters is not None:
        for _ in range(5):
            pt *= number_of_transmitters / np.sum(pt)
            pt = pt.clip(min=0, max=1)

    pr = np.real(pr)
    if number_of_receivers is not None:
        for _ in range(5):
            pr *= number_of_receivers / np.sum(pr)
            pr = pr.clip(min=0, max=1)

    return pt, pr


def solve_multiplicity(sum_array_response, number_of_transmitters, number_of_receivers):
    lamb = cp.Variable()
    multiplicity = cp.Variable(sum_array_response.shape[0])

    cost = cp.Maximize(lamb)

    constraints = [
        cp.real(
            sum_array_response.conj().T @ cp.diag(multiplicity) @ sum_array_response
        )
        - lamb * np.eye(sum_array_response.shape[1])
        >> 0,
        multiplicity >= 0,
        multiplicity <= min(number_of_transmitters, number_of_receivers),
        multiplicity <= np.arange(sum_array_response.shape[0]) + 1,
        multiplicity <= np.flip(np.arange(sum_array_response.shape[0]) + 1),
        cp.sum(multiplicity) == number_of_transmitters * number_of_receivers,
    ]

    problem = cp.Problem(cost, constraints)
    problem.solve()

    stats = {
        "solverStats": problem.solver_stats,
        "status": problem.status,
        "value": problem.value,
    }

    return multiplicity.value, stats


def solve_tx_or_rx_using_multiplicity(
    multiplicity,
    xx_fixed,
    number_of_elements,
    number_of_positions,
    weights=None,
):
    xx_var = cp.Variable(number_of_positions)

    sum_array_size = multiplicity.shape[0]
    dft_matrix = la.dft(sum_array_size, scale="sqrtn")

    cost = cp.Minimize(
        cp.norm(
            weights
            * dft_matrix.conj().T
            @ (
                np.sqrt(sum_array_size)
                * np.diag(fft(xx_fixed, sum_array_size, norm="ortho"))
                @ dft_matrix[:, :number_of_positions]
                @ xx_var
                - fft(multiplicity, sum_array_size, norm="ortho")
            )
        )
    )

    constraints = [xx_var >= 0, xx_var <= 1, cp.sum(xx_var) == number_of_elements]

    problem = cp.Problem(cost, constraints)
    problem.solve(solver=cp.ECOS)

    stats = {
        "solverStats": problem.solver_stats,
        "status": problem.status,
        "value": problem.value,
    }

    return xx_var.value, stats


def solve_tx_and_rx_using_multiplicity(
    sum_array_response,
    number_of_transmitters,
    number_of_receivers,
    number_of_transmit_positions,
    number_of_receive_positions,
    weighted=False,
    round_intermediates=False,
    initialization=None,
    hotstart=False,
):
    multiplicity, multiplicity_stats = solve_multiplicity(
        sum_array_response, number_of_transmitters, number_of_receivers
    )

    stats = {
        "solveMultiplicity": multiplicity_stats,
        "total": {"alternatingDescentIterations": 0, "solveTime": 0},
    }

    if hotstart:
        tx, rx = hot_start_tx_rx(
            multiplicity,
            number_of_transmit_positions,
            number_of_receive_positions,
            number_of_transmitters,
            number_of_receivers,
        )
    elif initialization is None:
        tx = (
            np.ones(number_of_transmit_positions)
            * number_of_transmitters
            / number_of_transmit_positions
        )
        rx = (
            np.ones(number_of_receive_positions)
            * number_of_receivers
            / number_of_receive_positions
        )
    else:
        tx = initialization["tx"]
        rx = initialization["rx"]

    weights = (
        1 - np.abs(multiplicity - np.round(multiplicity))
        if weighted
        else np.ones_like(multiplicity)
    )

    stats["total"]["weights"] = weights

    best_crb = np.inf
    current_crb = 10 * min_eig_fisher(sum_array_response, np.convolve(tx, rx))
    no_improvement_counter = 0

    stats["alternatingDescent"] = []
    while (
        no_improvement_counter < 3
        and stats["total"]["alternatingDescentIterations"] < 100
    ):
        best_crb = current_crb

        if round_intermediates:
            rx = randomized_rounding(rx, number_of_receivers)

        tx, tx_stats = solve_tx_or_rx_using_multiplicity(
            multiplicity,
            rx,
            number_of_transmitters,
            number_of_transmit_positions,
            weights,
        )

        stats["alternatingDescent"].append({"txStats": tx_stats})

        stats["total"]["solveTime"] += tx_stats["solverStats"].solve_time

        if round_intermediates:
            tx = randomized_rounding(tx, number_of_transmitters)

        rx, rx_stats = solve_tx_or_rx_using_multiplicity(
            multiplicity,
            tx,
            number_of_receivers,
            number_of_receive_positions,
            weights,
        )

        stats["alternatingDescent"][stats["total"]["alternatingDescentIterations"]][
            "rxStats"
        ] = rx_stats

        stats["total"]["solveTime"] += rx_stats["solverStats"].solve_time

        current_crb = min_eig_fisher(sum_array_response, np.convolve(tx, rx))

        if current_crb < best_crb:
            if best_crb - current_crb < 1e-3:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0
            best_crb = current_crb
        else:
            no_improvement_counter += 1

        stats["total"]["alternatingDescentIterations"] += 1

    return (
        randomized_rounding(tx, number_of_transmitters),
        randomized_rounding(rx, number_of_receivers),
        stats,
    )
