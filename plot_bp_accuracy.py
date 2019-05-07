import operator
import sys

from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np


plt.rcdefaults()


def main(data_file, output_file, hexcolour):
    with open(data_file) as f:
        accuracies = f.readline().strip().split(',')
    
    lambda_values = [ 0.0001, 0.001, 0.01, 0.1, 1.0, 10 ];
    accuracies = [ float(accuracy) for accuracy in accuracies ]
    
    class_indexes = np.arange(len(lambda_values))
    graph_data = list(zip(lambda_values, accuracies))

    plt.bar(class_indexes,
            accuracies,
            align='center',
            color='#{}'.format(hexcolour))
    plot_title = ('Accuracy of Models, Optimized Via '
                  + 'Backpropagation and With Varying λ Values, '
                  + 'With 100 Iterations', 100)
    plot_title = '\n'.join(wrap(plot_title, 60))
    plt.title(plot_title)
    plt.xlabel('Lambda Values')
    plt.ylabel('Accuracy (in Percentage)')
    plt.xticks(class_indexes, lambda_values)

    plt.savefig(output_file)


if __name__ == '__main__':
    argument_list = sys.argv[1:]
    if len(argument_list) != 3:
        print('Usage: plot_bp_accuracy.py <data> <output_file> '
              + '<colour in hex (no prefix)>')
        sys.exit(-1)

    main(argument_list[0], argument_list[1], argument_list[2])