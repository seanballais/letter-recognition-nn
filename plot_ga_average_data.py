import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import statistics


plt.rcdefaults()


def main(data_filename, output_dir, hexcolour):
    generation_values = list(range(1, 11))
    x = np.arange(500)
    generation_data = []
    with open(data_filename) as f:
        for line in f.readlines():
            if len(line.strip()) != 0:
                data = line.strip().split(',')
                generation_data.append(statistics.mean(map(float, data)))

    plt.figure(figsize=(20, 10))

    line_colour = '#{}'.format(hexcolour)
    plt.plot(x, generation_data, linewidth=0.5, color=line_colour)

    output_filename = data_filename[:len(data_filename) - 4]
    data_metadata = output_filename.split('|')
    lambda_value = float(data_metadata[4])
    num_random_parents = int(data_metadata[7])
    data_type = data_metadata[len(data_metadata) - 1] \
                    .split('-')[1]                    \
                    .capitalize()
    if data_type == 'Fitness':
        data_type = 'Cost'

    plot_title = ('Mean {} of Models, Optimized Via Genetic '.format(data_type)
                  + 'Algorithm and With λ = {} '.format(lambda_value)
                  + 'and {} Random Parents, '.format(num_random_parents)
                  + 'As Number of Generations Increase')

    if data_type == 'Accuracy':
        y_label = 'Accuracy (in Percentage)'
    else:
        y_label = data_type

    plt.title(plot_title)
    plt.xlabel('Number of Generations')
    plt.ylabel(y_label)

    plt.show()


if __name__ == '__main__':
    argument_list = sys.argv[1:]
    if len(argument_list) != 3:
        print('Usage: plot_ga_accuracy.py <filename> <output dir>'
              + '<colour in hex (no prefix)>')
        sys.exit(-1)

    main(argument_list[0], argument_list[1], argument_list[2])
