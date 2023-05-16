import re
from math import sqrt
import json


# noinspection PyShadowingNames
def _parse_file(path):
    data = [-1.0] * 101
    # noinspection PyBroadException
    try:
        with open(path, 'r') as f:
            for line in f.readlines():
                match = re.match(r'Before Training: score = (\d*\.?\d+)', line)
                if match is not None:
                    groups = match.groups()
                    if len(groups) > 0:
                        data[0] = float(groups[0])
                match = re.match(r'After Iteration #(\d+): score = (\d*\.?\d+)', line)
                if match is not None:
                    groups = match.groups()
                    if len(groups) >= 2:
                        data[int(groups[0])] = float(groups[1])
    except Exception:
        pass
    return None if data[0] < 0 else data


# noinspection PyShadowingNames
def _get_bounds(path):
    with open(path, 'r') as f:
        for line in f.readlines():
            match = re.match(r'bounds: (\d+), (\d+)', line)
            if match is not None:
                groups = match.groups()
                if len(groups) >= 2:
                    return float(groups[0]), float(groups[1])
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


    def _make_data_arr():
        return [[] for _ in range(101)]


    data = {
        '6': {'trvqc': _make_data_arr(), 'fvqe': _make_data_arr(), 'vqc': _make_data_arr()},
        '8': {'trvqc': _make_data_arr(), 'fvqe': _make_data_arr(), 'vqc': _make_data_arr()},
        '10': {'trvqc': _make_data_arr(), 'fvqe': _make_data_arr(), 'vqc': _make_data_arr()},
        '16': {'trvqc': _make_data_arr(), 'fvqe': _make_data_arr(), 'vqc': _make_data_arr()},
    }

    for folder, num_nodes in folders:
        bounds = _get_bounds(f"data/{folder}/graph.txt")
        if bounds is None:
            continue
        low, high = bounds
        for alg in ['trvqc', 'fvqe']:
            for i in range(150):
                scores = _parse_file(path=f"data/{folder}/{alg}{i + 1}.txt")
                if scores is not None:
                    for ind, score in enumerate(scores):
                        data[num_nodes][alg][ind].append((high - score) / (high - low))

    for num_nodes in ['6', '8', '10', '16']:
        for i in range(5):
            path = f"data/vqc-runs/vqc-n{num_nodes}-{i + 1}.txt"
            bounds = _get_bounds(path)
            if bounds is None:
                continue
            low, high = bounds
            scores = _parse_file(path)
            if scores is not None:
                for ind, score in enumerate(scores):
                    data[num_nodes]['vqc'][ind].append((high - score) / (high - low))


    def _summarize_data(d_list):
        avgs = []
        stds = []
        # noinspection PyShadowingNames
        for data in d_list:
            # noinspection PyShadowingNames
            avg = _avg(data)
            # noinspection PyShadowingNames
            std = _std(data, avg)

            avgs.append(avg)
            stds.append(std)
        return {'avg': avgs, 'std': stds}


    for alg in ['trvqc', 'fvqe', 'vqc']:
        print(f"Algorithm: {alg}")
        for num_nodes in ['6', '8', '10', '16']:
            # noinspection PyTypeChecker
            data[num_nodes][alg] = _summarize_data(data[num_nodes][alg])
            # noinspection PyTypeChecker
            avg = data[num_nodes][alg]['avg'][100]
            # noinspection PyTypeChecker
            std = data[num_nodes][alg]['std'][100]
            print(f"{num_nodes} Nodes: {avg} ± {std}")
        print()

    with open("data/accuracy-data.json", 'w') as f:
        json.dump(data, f)
