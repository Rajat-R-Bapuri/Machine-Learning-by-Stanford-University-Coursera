from matplotlib import pyplot as plt


def plotData(X, y):
    plt.figure(figsize=(9, 7))
    plt.plot(X, y, 'ro', marker='x', ms=5)
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show(block=False)
