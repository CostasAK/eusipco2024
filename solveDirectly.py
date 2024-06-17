import cvxpy as cp
import numpy as np
import scipy.linalg as la
from scipy.fft import fft

from functions import randomized_rounding, min_eig_fisher


def solveTxDirectly(
    sumArrayResponse,
    rx,
    numberOfTransmitters,
    numberOfTransmitPositions,
):
    sumArraySize = sumArrayResponse.shape[0]
    F = la.dft(sumArraySize, scale="sqrtn")

    lamb = cp.Variable()
    tx = cp.Variable(numberOfTransmitPositions)

    cost = cp.Maximize(lamb)

    constraints = [
        tx >= 0,
        tx <= 1,
        cp.sum(tx) == numberOfTransmitters,
        cp.real(
            sumArrayResponse.conj().T
            @ cp.diag(
                F.conj().T
                @ cp.multiply(
                    fft(rx, sumArraySize, norm="ortho"),
                    (F[:, :numberOfTransmitPositions] @ tx),
                )
            )
            @ sumArrayResponse
        )
        - lamb * np.eye(sumArrayResponse.shape[1])
        >> 0,
    ]

    problem = cp.Problem(cost, constraints)
    problem.solve()

    stats = {
        "solverStats": problem.solver_stats,
        "status": problem.status,
        "value": problem.value,
    }

    return tx.value, stats


def solveRxDirectly(
    sumArrayResponse,
    tx,
    numberOfReceivers,
    numberOfReceivePositions,
):
    sumArraySize = sumArrayResponse.shape[0]
    F = la.dft(sumArraySize, scale="sqrtn")

    lamb = cp.Variable()
    rx = cp.Variable(numberOfReceivePositions)

    cost = cp.Maximize(lamb)

    constraints = [
        rx >= 0,
        rx <= 1,
        cp.sum(rx) == numberOfReceivers,
        cp.real(
            sumArrayResponse.conj().T
            @ cp.diag(
                F.conj().T
                @ cp.multiply(
                    fft(tx, sumArraySize, norm="ortho"),
                    (F[:, :numberOfReceivePositions] @ rx),
                )
            )
            @ sumArrayResponse
        )
        - lamb * np.eye(sumArrayResponse.shape[1])
        >> 0,
    ]

    problem = cp.Problem(cost, constraints)
    problem.solve()

    stats = {
        "solverStats": problem.solver_stats,
        "status": problem.status,
        "value": problem.value,
    }

    return rx.value, stats


def solveTxRxDirectly(
    sumArrayResponse,
    numberOfTransmitters,
    numberOfReceivers,
    numberOfTransmitPositions,
    numberOfReceivePositions,
    roundIntermediates=False,
    initialization=None,
):
    stats = {"total": {"alternatingDescentIterations": 0, "solveTime": 0}}

    if initialization is None:
        tx = (
            np.ones(numberOfTransmitPositions)
            * numberOfTransmitters
            / numberOfTransmitPositions
        )
        rx = (
            np.ones(numberOfReceivePositions)
            * numberOfReceivers
            / numberOfReceivePositions
        )
    else:
        tx = initialization["tx"]
        rx = initialization["rx"]

    bestCrb = np.inf
    currentCrb = 10 * min_eig_fisher(sumArrayResponse, np.convolve(tx, rx))
    no_improvement_counter = 0

    stats["alternatingDescent"] = []
    while (
        no_improvement_counter < 3
        and stats["total"]["alternatingDescentIterations"] < 100
    ):
        if roundIntermediates:
            rx = randomized_rounding(rx, numberOfReceivers)

        tx, tx_stats = solveTxDirectly(
            sumArrayResponse,
            rx,
            numberOfTransmitters,
            numberOfTransmitPositions,
        )

        stats["alternatingDescent"].append({"txStats": tx_stats})

        stats["total"]["solveTime"] += tx_stats["solverStats"].solve_time

        if roundIntermediates:
            tx = randomized_rounding(tx, numberOfTransmitters)

        rx, rx_stats = solveRxDirectly(
            sumArrayResponse, tx, numberOfReceivers, numberOfReceivePositions
        )

        iteration = stats["total"]["alternatingDescentIterations"]

        stats["alternatingDescent"][iteration]["rxStats"] = rx_stats

        stats["total"]["solveTime"] += rx_stats["solverStats"].solve_time

        currentCrb = min_eig_fisher(sumArrayResponse, np.convolve(tx, rx))

        if currentCrb < bestCrb:
            if bestCrb - currentCrb < 1e-3:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0
            bestCrb = currentCrb
        else:
            no_improvement_counter += 1

        stats["total"]["alternatingDescentIterations"] += 1

    return (
        randomized_rounding(tx, numberOfTransmitters),
        randomized_rounding(rx, numberOfReceivers),
        stats,
    )
