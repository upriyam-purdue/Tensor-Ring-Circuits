from typing import List, Tuple, Final, Optional, Union

import torch
from torch import Tensor

from tensor_ring import QGate, Hamiltonian
from maxcut import create_hamiltonian

from numpy import pi, sqrt
from time import perf_counter
from random import random

# globals
_NUM_QUBITS: Final[int] = 4
_CIRCUIT_BLOCK_DEPTH: Final[int] = 1

_HAMILTONIAN: Hamiltonian = Hamiltonian(create_hamiltonian(_NUM_QUBITS, [
    ((0, 1), 5.0),
    ((0, 2), 3.0),
    ((1, 3), 1.0),
    ((2, 4), 5.0),
    ((3, 4), 7.0)
]))

# tensor ring
_TENSOR_RING_RANK: Final[int] = 10

# f-vqe
_NUM_FVQE_PARAMETERS: Final[int] = _NUM_QUBITS + 2 * _CIRCUIT_BLOCK_DEPTH * (_NUM_QUBITS - 1)
_FVQE_PARAMETERS: Final[List[float]] = [0] * _NUM_FVQE_PARAMETERS
for i in range(1 - _NUM_QUBITS, 0):
    _FVQE_PARAMETERS[i] = pi * random()
_FVQE_PARAMETERS[2 - 2 * _NUM_QUBITS] = pi * random()


def _apply_filter(energy: float, tau: float) -> float:
    # inverse filter --> see paper (https://arxiv.org/pdf/2106.10055.pdf) Figure 1 (page 1), Figure 2 (page 2)
    return energy ** -tau


def _compute_energy_helper(*, square_hamiltonian: bool) -> float:
    with torch.no_grad():
        # create ring tensors
        ring_tensors = []
        for _ in range(_NUM_QUBITS):
            kth_tensor = torch.zeros((_TENSOR_RING_RANK, _TENSOR_RING_RANK, 2), dtype=torch.cfloat)
            kth_tensor[0, 0, 0] = 1.0
            ring_tensors.append(kth_tensor)

        # apply f-vqe
        param_index = 0

        def _get_next_param_as_tensor() -> Tensor:
            nonlocal param_index
            assert 0 <= param_index < _NUM_FVQE_PARAMETERS

            param = _FVQE_PARAMETERS[param_index]
            param_index += 1
            return torch.tensor(param)

        for n in range(_NUM_QUBITS):
            QGate.apply_RY(_get_next_param_as_tensor(), n, ring_tensors)

        for _ in range(_CIRCUIT_BLOCK_DEPTH):
            for n in range(1, _NUM_QUBITS, 2):
                QGate.apply_CNOT(n - 1, n, ring_tensors)
                QGate.apply_RY(_get_next_param_as_tensor(), n - 1, ring_tensors)
                QGate.apply_RY(_get_next_param_as_tensor(), n, ring_tensors)

            for n in range(2, _NUM_QUBITS, 2):
                QGate.apply_CNOT(n - 1, n, ring_tensors)
                QGate.apply_RY(_get_next_param_as_tensor(), n - 1, ring_tensors)
                QGate.apply_RY(_get_next_param_as_tensor(), n, ring_tensors)

        assert param_index == _NUM_FVQE_PARAMETERS

        # compute energy
        nrg = _HAMILTONIAN.compute_energy(_NUM_QUBITS, ring_tensors, square_hamiltonian).item()
        assert nrg.imag < 1e-6, "Hamiltonian calculation is broken -- returned complex value"
        return nrg.real


def _compute_energy(
        param_shift_index: Optional[int] = None,
        *,
        square_hamiltonian: bool = False
) -> Union[float, Tuple[float, float]]:
    if param_shift_index is None:
        return _compute_energy_helper(square_hamiltonian=square_hamiltonian)
    else:
        _FVQE_PARAMETERS[param_shift_index] += pi / 2
        plus_nrg = _compute_energy_helper(square_hamiltonian=square_hamiltonian)
        _FVQE_PARAMETERS[param_shift_index] -= pi
        minus_nrg = _compute_energy_helper(square_hamiltonian=square_hamiltonian)
        _FVQE_PARAMETERS[param_shift_index] += pi / 2

        return plus_nrg, minus_nrg


def _compute_gradient(*, use_filtering_scaling: bool) -> List[float]:
    # # compute value of tau --> see paper (https://arxiv.org/pdf/2106.10055.pdf) Section II.D (page 4)
    # tau = 0.4  # TODO dynamically compute value of tau
    # # TODO implement filter
    # # sqrt(_apply_filter(_compute_energy(), tau))

    # memoize denominator value --> 4 * sqrt(f_squared) OR 2 (for non-filtering)
    derivative_denominator = 4 * sqrt(_compute_energy(square_hamiltonian=True)) if use_filtering_scaling else 2

    # compute numerator values and adjusted parameters
    def _compute_param_gradient(ind: int) -> float:
        param_plus, param_minus = _compute_energy(ind)  # psi(j+), psi(j-)

        # psi(j+) - psi(j-) --> see paper (https://arxiv.org/pdf/2106.10055.pdf) eq. 6 (page 3)
        derivative_numerator = param_plus - param_minus

        # see paper (https://arxiv.org/pdf/2106.10055.pdf) eq. 7 (page 4)
        return -derivative_numerator / derivative_denominator
        # is negative to correct for sign issue with gradient??

    return [_compute_param_gradient(ind) for ind in range(_NUM_FVQE_PARAMETERS)]


# NTS possible improvements to algorithm???
#  -- [DONE] variable learning rate (lambda = c/n^alpha)
#  -- [DONE] filtering ideas???
#  -- [IN PROGRESS??] early termination (hyper-parameter)

_var_learning_rate_init: float = 3.0
_var_learning_rate_scale_exp: float = 0.5  # between ~ 0.5 and 1
_var_learning_rate_iteration: List[int] = [1]


def _compute_learning_rate(iteration: int) -> float:
    return _var_learning_rate_init / iteration ** _var_learning_rate_scale_exp


# # compute learning rate --> see paper (https://arxiv.org/pdf/2106.10055.pdf) Table I (page 6), Appendix B1 (page 10)
# # TODO see why learning rate = 1 / (common hessian diagonal) is not working well
# # hessian_numerator = _compute_energy()
# learning_rate = 1.0  # derivative_denominator / hessian_numerator

def _adjust_params() -> None:
    nrg_old = _compute_energy()
    orig_params = [p for p in _FVQE_PARAMETERS]

    def _restore_params() -> None:
        for ind in range(_NUM_FVQE_PARAMETERS):
            _FVQE_PARAMETERS[ind] = orig_params[ind]

    gradient = _compute_gradient(use_filtering_scaling=True)
    assert len(gradient) == _NUM_FVQE_PARAMETERS

    def _update_params(learning_rate: float) -> None:
        for ind in range(_NUM_FVQE_PARAMETERS):
            _FVQE_PARAMETERS[ind] += learning_rate * gradient[ind]

    it_increment = 1
    prev_it = curr_it = _var_learning_rate_iteration[0]

    # performing increasing power of 2 search for optimal gradient iteration window
    _update_params(_compute_learning_rate(curr_it))
    while nrg_old < _compute_energy():
        _restore_params()

        prev_it, curr_it = curr_it, curr_it + it_increment
        it_increment *= 2

        _update_params(_compute_learning_rate(curr_it))

    # perform binary search within iteration window -- range = (prev_it, curr_it]
    low_it, high_it = prev_it + 1, curr_it
    while low_it < high_it:
        mid_it = (low_it + high_it) // 2

        _restore_params()
        _update_params(_compute_learning_rate(mid_it))

        if nrg_old < _compute_energy():
            low_it = mid_it + 1
        else:
            high_it = mid_it

    _restore_params()
    _update_params(_compute_learning_rate(high_it))
    _var_learning_rate_iteration[0] = high_it + 1


# NTS compare fvqe original vs this
#  -- comparison of average iteration runtime vs number of qubits
#  -- comparison of accuracy (approximation ratio) vs training epoch

def _test_circuit_training():
    # before energy
    print(f"Before Training: score = {_compute_energy()}")
    print("---------------------------------------------------")

    time_tot = 0
    # repeated training iterations
    for iteration in range(100):
        start_time = perf_counter()
        _adjust_params()
        end_time = perf_counter()

        # print current problem cost of circuit
        print(f"After Iteration #{iteration + 1}: score = {_compute_energy()}\ttime = {end_time - start_time}")
        time_tot += end_time - start_time
        # print(f"-- Using gradient iteration #{_var_learning_rate_iteration[0] - 1}")

    # after training
    print("---------------------------------------------------")
    print(f"After Training: score = {_compute_energy()}")
    print(f"Time per iteration: {time_tot / 100} seconds")


if __name__ == '__main__':
    _test_circuit_training()
