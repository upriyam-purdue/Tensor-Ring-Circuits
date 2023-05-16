import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = [6, 8, 10, 16, 24]
    trvqc = {
        'avg': [0.7269960753406666, 1.2263350154815558, 2.7239795610515, 14.91868929, 29.04120975, 68.36575077],
        'std': [0.1826621240288, 0.136995143606, 0.9390878951322869, 2.726963674835414, 3.239145212, 12.10042477],
    }
    fvqe = {
        'avg': [1.7476666682480002, 1.9242891526687786, 3.5853576445083006, 14.98501019, 107.0207788],
        'std': [0.40716162923475757, 0.42422438502889176, 1.305238012077937, 3.171032483972606, 8.585942267],
    }
    vqc = {
        'avg': [0.9986857363933331, 1.4995053915920002, 2.4883170919659996, 19.688772429618005, 106.5001899],
        'std': [0.07740127905114619, 0.14363693359998386, 0.07673080119093652, 0.0989918858135613, 12.18180284],
    }

    plt.errorbar([*x, 32], trvqc['avg'], yerr=trvqc['std'])
    plt.errorbar(x, vqc['avg'], fmt='g', yerr=vqc['std'])
    plt.errorbar(x, fvqe['avg'], fmt='orange', yerr=fvqe['std'])

    plt.title('Average Iteration Runtime by MaxCut Graph Size')
    plt.xlabel('MaxCut Graph Size (# nodes)')
    plt.ylabel('Iteration Runtime (s)')

    plt.legend(['TR-VQE', 'MPS-VQE', 'F-VQE'])

    plt.savefig('data/runtimes-plot.png')
    plt.show()
