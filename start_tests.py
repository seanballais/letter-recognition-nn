import subprocess
import sys

def start_tests(parameters, interpreter_exec_cmd):
    parameter_labels = [
        'Input Layer Size', 'Hidden Layer Size', 'Number of Labels',
        'Punishment Lambda', 'Using Genetic Algo',
        'Population/Iters', 'Random Parents',
        'Generations/BP Tests', 'Mutation Probability',
        'Training Items', 'Testing Items' 
    ]
    for test_parameters in parameters:
        print('---')
        print('Test Parameters')
        for parameter_index in range(len(parameter_labels)):
            parameter_label = parameter_labels[parameter_index]
            parameter_value = test_parameters[parameter_index]
            print('  {}:\t{}'.format(parameter_label, parameter_value))

        command = interpreter_exec_cmd + ['-W', 'main.m'] + test_parameters       
        print('\nTest Command: `{}`'.format(' '.join(command)))
        subprocess.run(command)

        print('---')

if __name__ == '__main__':
    arguments = sys.argv[1:]
    if len(arguments) < 1:
        print('Please specify the Octave interpreter.')
        sys.exit(0)

    parameter_file = arguments[0]
    interpreter_exec_cmd = arguments[1:]

    parameters = []
    with open(parameter_file) as f:
        for line in f.readlines():
            if (line.strip() == '---'
                or line.strip() == '###'
                or line.strip() == '@@@'
                ):
                continue

            parameters.append(list(filter(None, line.strip().split(','))))

    start_tests(parameters, interpreter_exec_cmd)
