import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    with open("data/accuracy-data.json", 'r') as f:
        data = json.load(f)
        data = data['10']  # extract data for 10 nodes only

    x = [i for i in range(101)]


    def _mult100(arr):
        return [100 * d for d in arr]


    for alg in ['trvqc', 'fvqe', 'vqc']:
        values = _mult100(data[alg]['avg'])
        errors = _mult100(data[alg]['std'])
        vpe = [v + e for v, e in zip(values, errors)]
        vme = [v - e for v, e in zip(values, errors)]
        plt.plot(x, values)
        plt.fill_between(x, vme, vpe, alpha=0.3)

    plt.title('Algorithm Accuracy vs Training Iteration')
    plt.xlabel('Training Iterations Completed')
    plt.ylabel('Per-Iteration Accuracy (%)')

    plt.legend(['TR-VQE', '_', 'F-VQE', '_', 'MPS-VQE', '_'])

    plt.savefig('data/accuracy-plot.png')
    plt.show()
