import glob
import sys

from textwrap import wrap
import matplotlib.pyplot as plt
import numpy as np


def main(filename_pattern, output_file):
    cost_files = glob.glob(filename_pattern)
    if len(cost_files) == 0:
        print('No files with the pattern `{}`'.format(filename_pattern))
        sys.exit(0)

    cost_files.sort()

    lambda_values = [ 0.0001, 0.001, 0.01, 0.1, 1.0, 10 ];
    x = np.arange(100)  # Since we know that we used 100 iterations.
    for cost_file in cost_files:
        with open(cost_file) as f:
            cost_vals = f.readline().strip().split(',')

        plt.plot(x, [ float(cost_val) for cost_val in cost_vals ])

    plot_title = '\n'.join(wrap('Cost of Models, Optimized Via '
                                + 'Backpropagation and With Varying λ Values, '
                                + 'As Number of Iterations Increase',
                                60))
    plt.title(plot_title)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.legend([ 'λ = {}'.format(lamba_val) for lamba_val in lambda_values ],
               loc='upper right')
    
    plt.savefig(output_file)


if __name__ == '__main__':
    argument_list = sys.argv[1:]
    if len(argument_list) != 2:
        print('Usage: plot_bp_cost.py <filename pattern> <output_file>')
        print('\nIn some shells (e.g. ZSH), you would need to surround '
              + 'the filename pattern with quotation marks.')
        sys.exit(-1)

    main(argument_list[0], argument_list[1])