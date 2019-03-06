import csv

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    agent_index = input('Enter Agent index : ')

    data = []
    with open('bin/benchmarks/benchmark_agent_{}.csv'.format(agent_index), 'r') as csv_file:
        readCSV = csv.reader(csv_file, delimiter=',')
        for row in readCSV:
            data.append(row)

        data = data[1:]

        rewards = [float(d[1]) for d in data]
        losses = [float(d[2]) for d in data]

        time_step = np.arange(len(data))

        f, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(time_step, rewards)
        ax2.plot(time_step, losses)

        plt.show()
