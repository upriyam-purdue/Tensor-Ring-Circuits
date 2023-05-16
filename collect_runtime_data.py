import re
from math import sqrt


# noinspection PyShadowingNames
def _parse_file(path):
    # noinspection PyBroadException
    try:
        with open(path, 'r') as f:
            for line in f.readlines():
                match = re.match(r'Time per iteration: (\d*\.?\d+) seconds', line)
                if match is not None:
                    groups = match.groups()
                    if len(groups) > 0:
                        return float(groups[0])
    except Exception:
        pass
    return None


# noinspection PyShadowingNames
def _avg(data):
    return sum(data) / len(data)


# noinspection PyShadowingNames
def _std(data, avg):  # of a sample
    return sqrt(sum((v - avg) ** 2 for v in data) / (len(data) - 1))


if __name__ == '__main__':
    folders = [
        ('run-n6-grad2', '6'),
        ('run-n8', '8'),
        ('run-n8-grad2', '8'),
        ('run-n10', '10'),
        ('run-n10-grad2', '10'),
        ('run-n16', '16'),
    ]

    data = {
        '6': {'trvqc': [], 'fvqe': [], 'vqc': []},
        '8': {'trvqc': [], 'fvqe': [], 'vqc': []},
        '10': {'trvqc': [], 'fvqe': [], 'vqc': []},
        '16': {'trvqc': [], 'fvqe': [], 'vqc': []},
    }

    for folder, num_nodes in folders:
        for alg in ['trvqc', 'fvqe']:
            for i in range(150):
                secs = _parse_file(path=f"data/{folder}/{alg}{i + 1}.txt")
                if secs is not None:
                    data[num_nodes][alg].append(secs)

    for num_nodes in ['6', '8', '10', '16']:
        for i in range(5):
            secs = _parse_file(path=f"data/vqc-runs/vqc-n{num_nodes}-{i + 1}.txt")
            if secs is not None:
                data[num_nodes]['vqc'].append(secs)

    for alg in ['trvqc', 'fvqe', 'vqc']:
        print(f"Algorithm: {alg}")
        for num_nodes in ['6', '8', '10', '16']:
            avg = _avg(data[num_nodes][alg])
            std = _std(data[num_nodes][alg], avg)
            print(f"{num_nodes} Nodes: {avg} s ± {std} s")
        print()

    # for alg in ['trvqc', 'fvqe', 'vqc']:
    #     print(f"Algorithm: {alg}")
    #     avg, std = [], []
    #     for num_nodes in ['6', '8', '10', '16']:
    #         avg.append(_avg(data[num_nodes][alg]))
    #         std.append(_std(data[num_nodes][alg], _avg(data[num_nodes][alg])))
    #     print(f"'avg': {avg},")
    #     print(f"'std': {std},")
    #     print()
