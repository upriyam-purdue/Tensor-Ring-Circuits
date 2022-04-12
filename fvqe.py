from typing import List, Dict, Tuple, Final, Optional, Union

import torch
from torch import Tensor

from tensor_ring import QGate, Hamiltonian
from maxcut import create_hamiltonian

from numpy import pi, sqrt

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
    _FVQE_PARAMETERS[i] = pi / 2
_FVQE_PARAMETERS[2 - 2 * _NUM_QUBITS] = pi / 2


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
        return _HAMILTONIAN.compute_energy(_NUM_QUBITS, ring_tensors, square_hamiltonian).item()


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


def _adjust_params() -> None:
    # compute value of tau --> see paper (https://arxiv.org/pdf/2106.10055.pdf) Section II.D (page 4)
    tau = 0.4  # TODO dynamically compute value of tau

    # compute learning rate --> see paper (https://arxiv.org/pdf/2106.10055.pdf) Table I (page 6)
    learning_rate = 1  # TODO compute learning rate

    # memoize denominator value --> 4 * sqrt(f_squared)
    # TODO implement filter
    # sqrt(_apply_filter(_compute_energy(), tau))
    derivative_denominator = 4 * sqrt(_compute_energy(square_hamiltonian=True))

    # compute numerator values and adjusted parameters
    new_params = [p for p in _FVQE_PARAMETERS]
    for ind in range(_NUM_FVQE_PARAMETERS):
        param_plus, param_minus = _compute_energy(ind)  # psi(j+), psi(j-)

        # psi(j+) - psi(j-) --> see paper (https://arxiv.org/pdf/2106.10055.pdf) eq. 6 (page 3)
        derivative_numerator = param_plus - param_minus

        # see paper (https://arxiv.org/pdf/2106.10055.pdf) eq. 7 (page 4)
        new_params[ind] -= learning_rate * derivative_numerator / derivative_denominator

    # save adjusted parameters
    for ind in range(_NUM_FVQE_PARAMETERS):
        _FVQE_PARAMETERS[ind] = new_params[ind]


def _test_circuit_training():
    # before energy
    print(f"Before Training: score = {_compute_energy()}")
    print("---------------------------------------------------")

    # repeated training iterations
    for iteration in range(30):
        _adjust_params()

        # print current problem cost of circuit
        print(f"After Iteration #{iteration + 1}: score = {_compute_energy()}")

    # after training
    print("---------------------------------------------------")
    print(f"After Training: score = {_compute_energy()}")


if __name__ == '__main__':
    _test_circuit_training()
