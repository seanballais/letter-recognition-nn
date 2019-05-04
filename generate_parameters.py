import sys


def generate_parameters(use_ga=0):
    input_layer_size = 16
    hidden_layer_size = 20
    num_labels = 26

    lambda_values = [ 0.0001, 0.001, 0.01, 0.1, 1.0, 10 ]
    population_sizes = [ 10 ] if use_ga else [ 50 ]
    num_random_parents = [ 0, 1, 4 ] if use_ga else [ 0 ]
    num_generations = [ 15000 ] if use_ga else [ 5 ]
    mutation_probabilities = [ 0.15 ] if use_ga else [ 0 ]
    training_items = 16000
    testing_items = 4000

    filename = 'ga' if use_ga else 'bp'

    print('Generating parameters...')
    print('File: `parameters/{}.csv`'.format(filename))

    with open('parameters/{}.csv'.format(filename), 'w') as f:
        for lambda_value in lambda_values:
          for population_size in population_sizes:
            for num_random_parent in num_random_parents:
              for num_generation in num_generations:
                for mutation_probability in mutation_probabilities:
                  line = '{},{},{},{},{},{},{},{},{},{},{}\n'.format(input_layer_size,
                                                                     hidden_layer_size,
                                                                     num_labels,
                                                                     lambda_value,
                                                                     use_ga,
                                                                     population_size,
                                                                     num_random_parent,
                                                                     num_generation,
                                                                     mutation_probability,
                                                                     training_items,
                                                                     testing_items)
                  f.write(line)


if __name__ == '__main__':
    use_ga = int(sys.argv[1])
    generate_parameters(use_ga)
