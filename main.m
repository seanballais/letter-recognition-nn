%% Initialization
clear; close all; clc;

fprintf('\nInitializing the neural network parameters...\n');

input_layer_size = 16;
hidden_layer_size = 20;
num_labels = 26;
lambda = 0.1;

useGA = 1;
numPopulation = 5;
num_random_parents = 1;
numGenerations = 2;
mutationProbability = 0.1;

% Load the data.
m_training = 300; % As per the number of training data used in the paper the
m_testing = 4000;   % data set has been originally used in.

if rows(argv()) > 0
  arg_list = argv();
  
  input_layer_size = str2num(arg_list(1){1});
  hidden_layer_size = str2num(arg_list(2){1});
  num_labels = str2num(arg_list(3){1});
  lambda = str2num(arg_list(4){1});

  useGA = str2num(arg_list(5){1});
  numPopulation = str2num(arg_list(6){1});
  num_random_parents = str2num(arg_list(7){1});
  numGenerations = str2num(arg_list(8){1});
  mutationProbability = str2num(arg_list(9){1});

  % Load the data.
  m_training = str2num(arg_list(10){1}); % As per the number of
                                         % training data used in the paper the
  m_testing = str2num(arg_list(11){1});  % data set has been originally used in.
endif

results_filename = sprintf('%d|%d|%d|%f|%d|%d|%d|%d|%f|%d|%d.csv', ...
                           input_layer_size, ...
                           hidden_layer_size, ...
                           num_labels, ...
                           lambda, ...
                           useGA, ...
                           numPopulation, ...
                           num_random_parents, ...
                           numGenerations, ...
                           mutationProbability, ...
                           m_training,
                           m_testing);

data = csvread('data/letter-recognition.csv');
training_data = data(1:m_training, :);
testing_data = data(20001 - m_testing:20000, :);

X_training = training_data(:, 2:end);
y_training = training_data(:, 1);

X_testing = testing_data(:, 2:end);
y_testing = testing_data(:, 1);
  
theta1 = zeros(hidden_layer_size, 1 + input_layer_size);
theta2 = zeros(num_labels, 1 + hidden_layer_size);

if (useGA == 1)
  [thetas1, thetas2] = generateInitialPopulation(numPopulation, ...
                                                 input_layer_size,
                                                 hidden_layer_size,
                                                 num_labels);
  populationCosts = zeros(numPopulation, 1);
  
  fprintf('===\n------------------------------\n');
  fprintf('Generation #1\n');
  fprintf('Creating new population...\n');
  for i = 1:numPopulation
    fprintf('    Specimen %d out of %d\n', i, numPopulation);
    fprintf('------------------------------\n');
    thetas1(:, :, i) = randInitializeWeights(input_layer_size, ...
                                             hidden_layer_size);
    thetas2(:, :, i) = randInitializeWeights(hidden_layer_size, ...
                                             num_labels);
    initial_nn_params = [thetas1(:, :, i)(:); thetas2(:, :, i)(:)];
    populationCosts(i) = nnCostFunction(initial_nn_params, ...
                                        input_layer_size, ...
                                        hidden_layer_size, ...
                                        num_labels, ...
                                        X_training, y_training, lambda);
  endfor
  fprintf('Current population fitness\n');
  disp(populationCosts);
  
  for g = 2:numGenerations
    fprintf('===\n------------------------------\n');
    fprintf('Generation #%d\n', g);
    fprintf('Creating new population...\n');
    [ thetas1, thetas2 ] = generateNewGeneration(populationCosts, ...
                                                 thetas1, ...
                                                 thetas2, ...
                                                 mutationProbability, ...
                                                 input_layer_size, ...
                                                 hidden_layer_size, ...
                                                 num_labels, ...
                                                 num_random_parents);
    %% Compute the cost of each weight.
    for i = 1:numPopulation
      fprintf('    Specimen %d out of %d\n', i, numPopulation);
      fprintf('------------------------------\n');
      nn_params = [thetas1(:, :, i)(:); thetas2(:, :, i)(:)];
      populationCosts(i) = nnCostFunction(nn_params, ...
                                          input_layer_size, ...
                                          hidden_layer_size, ...
                                          num_labels, ...
                                          X_training, y_training, lambda);
      pred = predict(thetas1(:, :, i), ...
                    thetas2(:, :, i), ...
                    X_testing) - 1; %% Subtract by one because the
                                    %% output layer starts with 1 
                                    %% but the alphabetical
                                    %% characters start with 0, (as
                                    %% the letter 'A' is represented
                                    %% with 0) .

      fprintf('\nSpecimen accuracy: %f\n', mean(double(pred == y_testing)) * 100); 
    endfor
    fprintf('Current population fitness\n');
    disp(populationCosts);
  endfor
  
  fprintf('-----\n');
  fprintf('Computing the accuracy of each specimen in the latest generation...\n');
  for i = 1:numPopulation
    pred = predict(thetas1(:, :, i), ...
                   thetas2(:, :, i), ...
                   X_testing) - 1; %% Subtract by one because the
                                   %% output layer starts with 1 
                                   %% but the alphabetical
                                   %% characters start with 0, (as
                                   %% the letter 'A' is represented
                                   %% with 0) .

    fprintf('\nSpecimen #%d accuracy: %f\n', ...
            i, ...
            mean(double(pred == y_testing)) * 100);
  endfor
else
  initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
  initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels);
  initial_nn_params = [initial_theta1(:); initial_theta2(:)]; % Time to unroll the weights.

  %% Train the neural network.
  fprintf('\nTraining the neural network... \n');

  options = optimset('MaxIter', 50);

  costFunction = @(params) nnCostFunction(params,
                                          input_layer_size, ...
                                          hidden_layer_size, ...
                                          num_labels, ...
                                          X_training, y_training, lambda);

  [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

  theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));
  theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));

  fprintf('Done.\n');
  
  %% Test the trained neural network.
  fprintf('\nTesting neural network...\n');
  pred = predict(theta1, theta2, X_testing) - 1; %% Subtract by one because the
                                                 %% output layer starts with 1 
                                                 %% but the alphabetical
                                                 %% characters start with 0, (as
                                                 %% the letter 'A' is represented
                                                 %% with 0) .

  fprintf('\nTraining set accuracy: %f\n', mean(double(pred == y_testing)) * 100);
endif
