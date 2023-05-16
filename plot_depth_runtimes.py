import matplotlib.pyplot as plt
import re
from math import sqrt


# noinspection PyShadowingNames
def _avg(data):
    return sum(data) / len(data)


# noinspection PyShadowingNames
def _std(data, avg):  # of a sample
    return sqrt(sum((v - avg) ** 2 for v in data) / (len(data) - 1))


def _load(path):
    with open(path, 'r') as f:
        times = []
        for line in f.readlines():
            match = re.match(r'After Iteration #\d+: score = \d+\.?\d*\ttime = (\d+\.?\d*)', line)
            if match is not None:
                times.append(float(match.group(1)))
        avg = _avg(times)
        std = _std(times, avg)
        return avg, std


def _load_all(alg):
    y = [_load(f"tmp/{alg}/d{n}.txt") for n in [1, 2, 3, 4, 5]]
    return [avg for avg, _ in y], [std for _, std in y]


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    trvqe = _load_all("trvqe")
    fvqe = _load_all("fvqe")
    vqe = _load_all("vqe")

    plt.errorbar(x, trvqe[0], yerr=trvqe[1])
    plt.errorbar(x, vqe[0], yerr=vqe[1], fmt='g')
    plt.errorbar(x, fvqe[0], yerr=fvqe[1], fmt='orange')

    plt.suptitle('Average Iteration Runtime by Circuit Depth', fontsize=16)
    plt.title('MaxCut Graph: 10 Nodes, 3 Edges per Node', fontsize=12)
    plt.xlabel('Circuit Depth (# layers)')
    plt.ylabel('Iteration Runtime (s)')

    plt.legend(['TR-VQE', 'MPS-VQE', 'F-VQE'])

    plt.savefig('data/depth-runtimes-plot.png')
    plt.show()
