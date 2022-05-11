from typing import List, Tuple, Optional

import torch
from torch import Tensor

from math import e, pi


def _apply_single_qubit_gate(gate_matrix: Tensor, qu_state_tensor: Tensor) -> Tensor:
    """ Apply the specified 1-qubit gate matrix on the specified ring-tensor """
    # gate_matrix: 2 × 2
    # qu_state_tensor: χ1 × χ2 × 2
    qu_state_tensor = torch.tensordot(gate_matrix, qu_state_tensor, ([1], [2]))
    # qu_state_tensor: (2 × [2]) . (χ1 × χ2 × [2]) = 2 × χ1 × χ2
    qu_state_tensor = torch.moveaxis(qu_state_tensor, 0, 2)
    # qu_state_tensor: χ1 × χ2 × 2
    return qu_state_tensor


def _apply_double_qubit_gate(gate_matrix: Tensor, qu_state_tensors: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    """ Apply the specified 2-qubit gate matrix on the specified ring-tensors """
    # gate_matrix: 4 × 4
    qu0, qu1 = qu_state_tensors
    # qu0: χ1 × χ2 × 2
    # qu1: χ2 × χ3 × 2

    chi_1 = qu0.shape[0]
    chi_3 = qu1.shape[1]
    # chi_1 = χ1
    # chi_3 = χ3

    mps = torch.tensordot(qu0, qu1, ([1], [0]))
    # mps: (χ1 × [χ2] × 2) . ([χ2] × χ3 × 2) = χ1 × 2 × χ3 × 2
    mps = torch.moveaxis(mps, 2, 1)
    # mps: χ1 × χ3 × 2 × 2

    gate_tensor = torch.reshape(gate_matrix, (2, 2, 2, 2))
    # gate_tensor: 2 × 2 × 2 × 2

    mps = torch.tensordot(gate_tensor, mps, ([2, 3], [2, 3]))
    # mps: (2 × 2 × [2] × [2]) . (χ1 × χ3 × [2] × [2]) = 2 × 2 × χ1 × χ3
    mps = torch.moveaxis(mps, 1, 2).reshape((chi_1 * 2, chi_3 * 2))
    # mps: 2 × χ1 × 2 × χ3 --> (2 * χ1) × (2 * χ3)

    u, s, v = torch.linalg.svd(mps)
    # u: (2 * χ1) × (2 * χ1)
    # s: 2 * min(χ1,χ3)
    # y: (2 * χ3) × (2 * χ3)

    # TODO apply rescaling to sx (below) b/c dim = min(χ1,χ3) ?= χ1
    #  -- not technically necessary, unless chi values start off different
    #  -- not necessary right now, but a future-proofing good-to-have

    # noinspection PyTypeChecker
    x, sx, y = u[:, :chi_1], torch.diag(s[:chi_1]).type(torch.cfloat), v[:chi_3, :]
    # x: (2 * χ1) × χ1
    # sx: χ1 × χ1
    # y: χ3 × (2 * χ3)

    qu0 = torch.mm(x, sx).reshape((2, chi_1, chi_1))
    # qu0: ((2 * χ1) × [χ1]) . ([χ1] × χ1) = (2 * χ1) × χ1 --> 2 × χ1 × χ1
    qu1 = y.reshape((chi_3, 2, chi_3))
    # qu1: χ3 × 2 × χ3

    qu0 = torch.moveaxis(qu0, 0, 2)
    # qu0: χ1 × χ1 × 2
    qu1 = torch.moveaxis(qu1, 1, 2)
    # qu1: χ3 × χ3 × 2

    return qu0, qu1


# noinspection PyPep8Naming
class QGate:
    @staticmethod
    def apply_H(i: int, ring_tensors: List[Tensor]) -> None:
        """ Alter ring tensors to where H [Hadamard gate] has acted on the ith qubit """
        H_tensor = 1 / (2 ** 0.5) * torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat)
        ring_tensors[i] = _apply_single_qubit_gate(H_tensor, ring_tensors[i])

    @staticmethod
    def _apply_PAULI_X(i: int, ring_tensors: List[Tensor]) -> None:
        """ Alter ring tensors to where sX [Pauli-X gate] has acted on the ith qubit """
        sigX_tensor = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
        ring_tensors[i] = _apply_single_qubit_gate(sigX_tensor, ring_tensors[i])

    @staticmethod
    def apply_PAULI_Y(i: int, ring_tensors: List[Tensor]) -> None:
        """ Alter ring tensors to where sY [Pauli-Y gate] has acted on the ith qubit """
        sigY_tensor = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat)
        ring_tensors[i] = _apply_single_qubit_gate(sigY_tensor, ring_tensors[i])

    @staticmethod
    def apply_PAULI_Z(i: int, ring_tensors: List[Tensor]) -> None:
        """ Alter ring tensors to where sZ [Pauli-Z gate] has acted on the ith qubit """
        sigZ_tensor = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
        ring_tensors[i] = _apply_single_qubit_gate(sigZ_tensor, ring_tensors[i])

    @staticmethod
    def apply_RX(theta: Tensor, i: int, ring_tensors: List[Tensor]) -> None:
        """ Alter ring tensors to where RX(theta) [x-axis rotation gate] has acted on the ith qubit """
        cos = torch.atleast_1d(torch.cos(theta / 2))
        sin = torch.atleast_1d(torch.sin(theta / 2))
        RX_tensor = torch.reshape(torch.cat([cos, -1j * sin, -1j * sin, cos]), (2, 2))

        ring_tensors[i] = _apply_single_qubit_gate(RX_tensor, ring_tensors[i])

    @staticmethod
    def apply_RY(theta: Tensor, i: int, ring_tensors: List[Tensor]) -> None:
        """ Alter ring tensors to where RY(theta) [y-axis rotation gate] has acted on the ith qubit """
        cos = torch.atleast_1d(torch.cos(theta / 2))
        sin = torch.atleast_1d(torch.sin(theta / 2))
        # noinspection PyTypeChecker
        RY_tensor = torch.reshape(torch.cat([cos, -sin, sin, cos]).type(torch.cfloat), (2, 2))

        ring_tensors[i] = _apply_single_qubit_gate(RY_tensor, ring_tensors[i])

    @staticmethod
    def apply_RZ(theta: Tensor, i: int, ring_tensors: List[Tensor]) -> None:
        """ Alter ring tensors to where RZ(theta) [z-axis rotation gate] has acted on the ith qubit """
        exp = torch.atleast_1d(torch.exp(1j * theta / 2))
        zero = torch.zeros(1)
        RZ_tensor = torch.reshape(torch.cat([1 / exp, zero, zero, exp]), (2, 2))

        ring_tensors[i] = _apply_single_qubit_gate(RZ_tensor, ring_tensors[i])

    @staticmethod
    def apply_CNOT(i: int, j: int, ring_tensors: List[Tensor]) -> None:
        """
        Alter ring tensors to where CNOT [controlled-not gate] has acted on the ith & jth qubits
        > control qubit: i
        > target qubit: j
        """
        CNOT_matrix = torch.tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]],
            dtype=torch.cfloat
        )

        ring_tensors[i], ring_tensors[j] = _apply_double_qubit_gate(CNOT_matrix, (ring_tensors[i], ring_tensors[j]))


class Hamiltonian:
    ListType = List[Tuple[float, Tuple[bool, ...]]]
    _components: ListType

    def __init__(self, parts: ListType, /):
        self._components = parts

    def compute_energy(self, num_qubits: int, ring_tensors: List[Tensor], squared: bool = False) -> Tensor:
        nrg = torch.zeros(1, dtype=torch.cfloat)
        if squared:
            for weight_1, pauli_z_mask_1 in self._components:
                for weight_2, pauli_z_mask_2 in self._components:
                    nrg += weight_1 * weight_2 * Hamiltonian._compute_energy(
                        num_qubits, ring_tensors, pauli_z_mask_1, pauli_z_mask_2
                    )
        else:
            for weight, pauli_z_mask in self._components:
                nrg += weight * Hamiltonian._compute_energy(num_qubits, ring_tensors, pauli_z_mask)
        return nrg

    @staticmethod
    def _RZ_pow_tensor(power: float) -> Tensor:
        return torch.tensor([[e ** (2j * pi * power), 0], [0, e ** (1j * pi * power)]], dtype=torch.cfloat)

    @staticmethod
    def _compute_energy(num_qubits: int, ring_tensors: List[Tensor], pauli_z_mask_1: Tuple[bool, ...],
                        pauli_z_mask_2: Optional[Tuple[bool, ...]] = None, /) -> Tensor:
        work_ring = [torch.clone(tens) for tens in ring_tensors]
        for i in range(num_qubits):
            if pauli_z_mask_1[i] or (pauli_z_mask_2 is not None and pauli_z_mask_2[i]):
                QGate.apply_PAULI_Z(i, work_ring)
                if pauli_z_mask_1[i] and (pauli_z_mask_2 is not None and pauli_z_mask_2[i]):
                    QGate.apply_PAULI_Z(i, work_ring)

            # work_ring[i]: χ1 × χ2 × 2
            # ring_tensors[i]: χ1 × χ2 × 2
            work_ring[i] = torch.tensordot(work_ring[i], ring_tensors[i], ([2], [2]))
            # work_ring[i]: χ1 × χ2 × χ1 × χ2

        ret_tens = work_ring[0]
        # ret_tens: χ1 × χ2 × χ1 × χ2
        for i in range(1, num_qubits - 1):
            # ret_tens: χA × χB × χA × χB
            # work_ring[i]: χB × χC × χB × χC
            ret_tens = torch.tensordot(ret_tens, work_ring[i], ([1, 3], [0, 2]))
            # ret_tens: χA × χA × χC × χC
            ret_tens = torch.moveaxis(ret_tens, 2, 1)
            # ret_tens: χA × χC × χA × χC

        # ret_tens: χ1 × χn × χ1 × χn
        # work_ring[-1]: χn × χ1 × χn × χ1
        ret_tens = torch.tensordot(ret_tens, work_ring[-1], ([0, 1, 2, 3], [1, 0, 3, 2]))
        # ret_tens: 1
        return ret_tens


def print_tensor_ring(ring_tensors: List[Tensor]):
    print("-------------------------MPS STRUCTURE SHAPE------------------------------------------")
    for tensor in ring_tensors:
        print(tensor)
    print("-------------------------SHAPE------------------------------------------")


def _main():
    pass


if __name__ == '__main__':
    _main()
