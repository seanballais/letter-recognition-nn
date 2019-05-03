def generate_ga_parameters():
    input_layer_size = 16
    hidden_layer_sizes = [ 20, 30, 50, 75, 100 ]
    num_labels = 26

    lambda_values = [ 0.0001, 0.001, 0.01, 0.1, 1.0, 10 ]
    use_ga = 1
    population_sizes = [ 5, 10, 15 ]
    num_random_parents = [ 1, 2, 3, 4 ]
    num_generations = [ 10, 10000, 15000 ]
    mutation_probabilities = [ 0.07, 0.1 ]
    training_items = [ 16000, 10000, 8000, 3000, 1000 ]
    testing_items = 4000

    print('Generating parameters...')

    with open('parameters/ga.csv', 'w') as f:
        for hidden_layer_size in hidden_layer_sizes:
            for lambda_value in lambda_values:
                for population_size in population_sizes:
                    for num_random_parent in num_random_parents:
                        for num_generation in num_generations:
                            for mutation_probability in mutation_probabilities:
                                for training_item in training_items:
                                    line = '{},{},{},{},{},{},{},{},{},{},{}\n'.format(input_layer_size,
                                                                                       hidden_layer_size,
                                                                                       num_labels,
                                                                                       lambda_value,
                                                                                       use_ga,
                                                                                       population_size,
                                                                                       num_random_parent,
                                                                                       num_generation,
                                                                                       mutation_probability,
                                                                                       training_item,
                                                                                       testing_items)
                                    f.write(line)


if __name__ == '__main__':
    generate_ga_parameters()