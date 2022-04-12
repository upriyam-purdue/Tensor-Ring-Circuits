from typing import List, Tuple, Optional

from tensor_ring import Hamiltonian


def create_hamiltonian(num_qubits: int, edge_weights: List[Tuple[Tuple[int, int], float]]) -> Hamiltonian.ListType:
    def _make_edge_tuple(edge: Optional[Tuple[int, int]] = None, /) -> Tuple[bool, ...]:
        list_form = [False] * num_qubits

        if edge is not None:
            n1, n2 = edge
            if 0 <= n1 < num_qubits:
                list_form[n1] = True
            if 0 <= n2 < num_qubits:
                list_form[n2] = True

        return tuple(list_form)

    total_edge_cost = sum(cost for _, cost in edge_weights)
    return [
        # +1 so energy is always > 0
        (total_edge_cost / 2.0 + 1.0, _make_edge_tuple()),
        *[(cost / 2.0, _make_edge_tuple(edge)) for edge, cost in edge_weights]
    ]
