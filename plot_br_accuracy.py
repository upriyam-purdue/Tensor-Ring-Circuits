import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = [4, 8, 10, 12]
    y_avg = [100 * v for v in [0.8746744924, 0.8520945576, 0.920835128, 0.9581555525]]
    y_std = [100 * v for v in [0.03921882485, 0.02213693463, 0.072584189, 0.06718161798]]

    plt.errorbar(x, y_avg, yerr=y_std)
    # plt.ylim(80, 100)

    plt.title('Algorithm Accuracy vs Tensor Ring Bond Rank')
    plt.xlabel('TR-VQE Bond Rank (χ)')
    plt.ylabel('Final Evaluation Accuracy (%)')

    plt.savefig('data/br-accuracy-plot.png')
    plt.show()
