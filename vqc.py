from typing import Final, List, Optional

from tensor_ring import QGate, Hamiltonian

import torch
from torch import nn, Tensor
from torch.nn import functional as functions
from torch.autograd import Variable

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _get_iris_dataset():
    # get iris dataset
    iris = load_iris()

    # get data
    input_data = iris['data']
    target_results = iris['target']

    # get metadata
    target_names = iris['target_names']
    feature_names = iris['feature_names']

    # split data into train/test sets
    train_in, test_in, train_out, test_out = \
        train_test_split(StandardScaler().fit_transform(input_data), target_results, test_size=0.2, random_state=2)

    # convert data to pytorch values
    train_in = (torch.from_numpy(train_in)).float()
    train_out = (torch.from_numpy(train_out)).long().reshape(train_in.shape[0], 1)
    test_in = Variable(torch.from_numpy(test_in)).float()
    test_out = Variable(torch.from_numpy(test_out)).long().reshape(test_in.shape[0], 1)

    # return (input train, input test, target train, target test)
    return train_in, train_out, test_in, test_out


class VQC(nn.Module):
    def __init__(self, num_qubits: int, vqc_depth: int):
        super(VQC, self).__init__()

        # VQC bounds
        self.num_qubits = num_qubits
        self.vqc_depth = vqc_depth

        # VQC parameters
        self.paramsRX = nn.parameter.Parameter(torch.randn(self.vqc_depth, self.num_qubits))
        self.paramsRY = nn.parameter.Parameter(torch.randn(self.vqc_depth, self.num_qubits))
        self.paramsRZ = nn.parameter.Parameter(torch.randn(self.vqc_depth, self.num_qubits))

    def forward(self, ring_tensors: List[Tensor], psi: List[Tensor], hamiltonian: Optional[Hamiltonian]) -> Tensor:
        """ Compute the resultant VQC values for the input data psi, given the specified ring tensors """
        self._apply_forward_pass(ring_tensors, psi)

        if hamiltonian is None:
            # compute single ring-equivalent tensor
            k = VQC._compute_ring_tensor(ring_tensors)

            # output measured response
            k1 = k[0, 0, 0, 0].abs().reshape(1) ** 2
            k2 = k[0, 0, 1, 0].abs().reshape(1) ** 2
            k3 = k[1, 1, 1, 1].abs().reshape(1) ** 2
            return torch.cat([k1, k2, k3]).reshape(1, 3)
        else:
            return hamiltonian.compute_energy(self.num_qubits, ring_tensors)

    def _apply_forward_pass(self, ring_tensors: List[Tensor], psi: List[Tensor]) -> None:
        # initialize qubits to psi
        for n in range(self.num_qubits):
            QGate.apply_RX(psi[n], n, ring_tensors)

        # enable gradient tracking for parametrized VQC
        for n in range(self.num_qubits):
            ring_tensors[n].requires_grad_(True)

        # apply VQC
        for d in range(self.vqc_depth):
            # add cyclical CNOTs
            for n in range(self.num_qubits):
                QGate.apply_CNOT(n, (n + 1) % self.num_qubits, ring_tensors)
            # add parametrized rotations
            for n in range(self.num_qubits):
                QGate.apply_RX(self.paramsRX[d][n], n, ring_tensors)
                QGate.apply_RY(self.paramsRY[d][n], n, ring_tensors)
                QGate.apply_RZ(self.paramsRZ[d][n], n, ring_tensors)

    @staticmethod
    def _compute_ring_tensor(ring_tensors: List[Tensor]) -> Tensor:
        """ Compute a single equivalent tensor for the ring tensors """
        ring_tensor = ring_tensors[0]
        for i in range(len(ring_tensors) - 2):
            ring_tensor = torch.tensordot(ring_tensor, ring_tensors[i + 1], ([1], [0]))
            ring_tensor = torch.moveaxis(ring_tensor, -2, 1)
        return torch.tensordot(ring_tensor, ring_tensors[-1], ([0, 1], [1, 0]))


_num_epochs: Final[int] = 50
_tensor_ring_rank: Final[int] = 10


def _init_ring_tensors(num_qubits: int) -> List[Tensor]:
    ring_tensors = []
    for _ in range(num_qubits):
        kth_tensor = torch.zeros((_tensor_ring_rank, _tensor_ring_rank, 2), dtype=torch.cfloat)
        kth_tensor[0, 0, 0] = 1.0
        ring_tensors.append(kth_tensor)
    return ring_tensors


def _main():
    # initialize VQC/optimizer, etc.
    tr_vqc = VQC(4, 1)
    optimizer = torch.optim.Adam(tr_vqc.parameters(), lr=000.001)

    # prepare for training
    tr_vqc.train()

    vqc_training_loss_history = np.zeros((_num_epochs,))
    vqc_train_accuracy_history = np.zeros((_num_epochs,))
    vqc_test_accuracy_history = np.zeros((_num_epochs,))

    # train/test dataset
    train_in, train_out, test_in, test_out = _get_iris_dataset()

    # training loop
    for epoch in range(_num_epochs):
        # prepare for epoch
        epoch_cumulative_loss = 0
        epoch_cumulative_correctness = []
        epoch_training_iteration_count = 0

        # train VQC on data
        for input_data, target_output in zip(train_in, train_out):
            # compute loss
            vqc_result = functions.softmax(tr_vqc(_init_ring_tensors(tr_vqc.num_qubits), input_data))
            loss = functions.cross_entropy(vqc_result, target_output)
            epoch_cumulative_correctness.append(
                (torch.argmax(vqc_result, dim=1) == target_output).type(torch.FloatTensor).item()
            )
            epoch_cumulative_loss = epoch_cumulative_loss + loss / 4

            # update iteration count
            epoch_training_iteration_count += 1

            # every few iterations, compute gradients
            if epoch_training_iteration_count % 4 == 0:
                optimizer.zero_grad()
                epoch_cumulative_loss.backward()
                optimizer.step()
                vqc_training_loss_history[epoch] = epoch_cumulative_loss.item()

                # reset iterative_loss
                epoch_cumulative_loss = 0
        vqc_train_accuracy_history[epoch] = np.mean(epoch_cumulative_correctness)

        # test VQC on data
        epoch_cumulative_correctness = []
        with torch.no_grad():
            for input_data, target_output in zip(test_in, test_out):
                vqc_result = functions.softmax(tr_vqc(_init_ring_tensors(tr_vqc.num_qubits), input_data))
                epoch_cumulative_correctness.append(
                    (torch.argmax(vqc_result, dim=1) == target_output).type(torch.FloatTensor).item()
                )
        vqc_test_accuracy_history[epoch] = np.mean(epoch_cumulative_correctness)
        test_acc_disp = float(int(1000 * vqc_test_accuracy_history[epoch])) / 10
        train_acc_disp = float(int(1000 * vqc_train_accuracy_history[epoch])) / 10
        print(f"Epoch (#{epoch + 1}) Average Accuracy: Train={train_acc_disp}%, Test={test_acc_disp}%")

    print(vqc_training_loss_history)
    print(vqc_train_accuracy_history)
    print(vqc_test_accuracy_history)


if __name__ == '__main__':
    _main()
