from typing import List, Dict, Tuple, Final, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as functions

from maxcut import create_hamiltonian
from tensor_ring import QGate, Hamiltonian

from numpy import log2

# globals
_NUM_QUBITS: Final[int] = 4
_QAOA_DEPTH: Final[int] = 1

_IDENTITY: Final[Tensor] = torch.tensor([[1, 0], [0, 1]], dtype=torch.cfloat)
_PAULI_X: Final[Tensor] = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
_PAULI_Z: Final[Tensor] = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
_UNIFORM: Final[Tensor] = 0.5 * torch.tensor([[1, 1], [1, 1]], dtype=torch.cfloat)


# qaoa helpers
def _zip_hamiltonian_parts(ret: List[bool], hamiltonian: List[Tuple[float, Tuple[bool, ...]]], ind: int):
    for i in range(len(hamiltonian)):
        yield ret[ind + i], hamiltonian[i]


def _compress_ret_tuple(ret: List[bool], params: List[Tuple[Tensor, Tensor]],
                        hamiltonian: List[Tuple[float, Tuple[bool, ...]]], first_half: bool):
    ind: int = 0 if first_half else _QAOA_DEPTH * (len(hamiltonian) + 1)
    for i in range(_QAOA_DEPTH):
        yield ret[ind], *params[-i - 1 if first_half else i], _zip_hamiltonian_parts(ret, hamiltonian, ind + 1)
        ind += len(hamiltonian) + 1


def _qaoa_matrix(params: List[Tuple[Tensor, Tensor]], hamiltonian: List[Tuple[float, Tuple[bool, ...]]]):
    num_states: Final[int] = 2 * _QAOA_DEPTH * (len(hamiltonian) + 1)

    ret: List[bool] = [False] * num_states

    for i in range(num_states):
        if i != 0:
            ind = int(log2(i & ~(i - 1)))
            ret[ind] = not ret[ind]
        yield _compress_ret_tuple(ret, params, hamiltonian, True), _compress_ret_tuple(ret, params, hamiltonian, False)


# qaoa main funcs
def _compute_qaoa_hamiltonian(params: List[Tuple[Tensor, Tensor]],
                              hamiltonian: List[Tuple[float, Tuple[bool, ...]]]) -> Tensor:
    assert len(params) == _QAOA_DEPTH, "QAOA parameter count/depth mismatch"

    for _, pauli_z_flags in hamiltonian:
        assert len(pauli_z_flags) == _NUM_QUBITS, "Invalid hamiltonian structure"

    trace = torch.tensor(0, dtype=torch.cfloat)
    # H times -> O(DN H^2 2^(DH))
    for weight, pauli_z_flags in hamiltonian:
        # print(trace)
        # 2^(DH) times -> O(DHN 2^(DH))
        for first_half, second_half in _qaoa_matrix(params, hamiltonian):
            # set up initial hamiltonian matrices
            ring = [_PAULI_Z if pauli_z_flags[n] else _IDENTITY for n in range(_NUM_QUBITS)]

            # TODO fix -i<beta/gamma> multiplier on matrix exponential
            def _apply_qaoa_matrix(generator):
                """
                e^(kron(I|pauli_z, ...) = ARG) = cosh_1 * kron(I, ...) + sinh_1 * ARG

                Runtime: D (QAOA depth) * H (num Hamiltonian parts) * N (num qubits)

                :param generator:
                :return:
                """
                # D times -> O(DHN)
                for apply_pauli_x_kron, beta, gamma, hamiltonian_gen in generator:
                    mult = (-1j * torch.sin(beta) * _PAULI_X) if apply_pauli_x_kron else torch.cos(beta)
                    # N times -> O(N)
                    for n in range(_NUM_QUBITS):
                        ring[n] = ring[n] * mult

                    # H times -> O(HN)
                    for apply_hamiltonian_kron, hamiltonian_edge in hamiltonian_gen:
                        edge_weight, hamiltonian_pauli_z_flags = hamiltonian_edge
                        theta = gamma * edge_weight
                        mult = (-1j * torch.sin(theta)) if apply_hamiltonian_kron else torch.cos(theta)
                        # N times -> O(N)
                        for n in range(_NUM_QUBITS):
                            ring[n] = ring[n] * mult

                        if apply_hamiltonian_kron:
                            for n in range(_NUM_QUBITS):
                                if hamiltonian_pauli_z_flags[n]:
                                    ring[n] = ring[n] * _PAULI_Z

            # print("A")

            # apply U = Bk, Ck, B(k-1), C(k-1), ..., B1, C1
            _apply_qaoa_matrix(first_half)

            # print("B")

            # apply rho0
            for n in range(_NUM_QUBITS):
                ring[n] = ring[n] * _UNIFORM

            # print("C")

            # apply U† = C1, B1, C2, B2, ..., Ck, Bk
            _apply_qaoa_matrix(second_half)

            # print("D")

            # add to trace
            temp = 1
            for mat in ring:
                # print(torch.sum(torch.diag(mat)))
                # print(temp)
                temp = temp * torch.sum(torch.diag(mat))
                # print(temp)

            # print(trace + weight * temp)
            trace += weight * temp

            # print("E")

            # print(ring)
            # print(trace)
            # print(temp)

    return trace


# noinspection PyPep8Naming
class QAOA(nn.Module):
    def __init__(self):
        super(QAOA, self).__init__()

        # VQC parameters
        self.beta_params = nn.parameter.Parameter(torch.randn(_QAOA_DEPTH))
        self.gamma_params = nn.parameter.Parameter(torch.randn(_QAOA_DEPTH))

    def forward(self, hamiltonian: List[Tuple[float, Tuple[bool, ...]]]) -> Tensor:
        params: List[Tuple[Tensor, Tensor]] = []
        for i in range(_QAOA_DEPTH):
            params.append((self.beta_params[i], self.gamma_params[i]))
        return _compute_qaoa_hamiltonian(params, hamiltonian)

    def get_optimal_state(self):
        # TODO implement ret = U * |psi0>
        pass


# main
def _main():
    # initialize VQC/optimizer, etc.
    qaoa = QAOA()
    optimizer = torch.optim.Adam(qaoa.parameters(), lr=000.001)

    # prepare for training
    qaoa.train()

    hamiltonian = create_hamiltonian(_NUM_QUBITS, [
        ((0, 1), 5.0),
        ((0, 2), 3.0),
        ((1, 3), 1.0),
        ((2, 4), 5.0),
        ((3, 4), 7.0)
    ])

    print(hamiltonian)

    # train
    qaoa_result = qaoa(hamiltonian)
    print(qaoa_result)
    loss = functions.cross_entropy(qaoa_result, torch.tensor(2, dtype=torch.cfloat))
    loss.backward()


if __name__ == '__main__':
    _main()
