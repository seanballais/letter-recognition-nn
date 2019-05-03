function [thetas1, thetas2] = generateNewGeneration(populationCosts, ...
                                                    thetas1, ...
                                                    thetas2, ...
                                                    mutationProbability, ...
                                                    input_layer_size, ...
                                                    hidden_layer_size, ...
                                                    output_layer_size, ...
                                                    num_random_parents)
    numPopulation = rows(populationCosts);
    
    parentIndices = selectParents(populationCosts, ...
                                  num_random_parents);
    fprintf('Selected Parent Costs\n');
    for i = 1:columns(parentIndices)
      fprintf('  Parent #%d Cost: %f\n', i, populationCosts(parentIndices(i)));
    endfor

    parentsTheta1 = zeros(rows(thetas1(:, :, parentIndices(1))),
                          columns(thetas1(:, :, parentIndices(1))),
                          4);
    parentsTheta1(:, :, 1) = thetas1(:, :, parentIndices(1));
    parentsTheta1(:, :, 2) = thetas1(:, :, parentIndices(2));
    parentsTheta1(:, :, 3) = thetas1(:, :, parentIndices(3));
    parentsTheta1(:, :, 4) = thetas1(:, :, parentIndices(4));
    
    parentsTheta2 = zeros(rows(thetas2(:, :, parentIndices(1))),
                          columns(thetas2(:, :, parentIndices(1))),
                          4);
    parentsTheta2(:, :, 1) = thetas2(:, :, parentIndices(1));
    parentsTheta2(:, :, 2) = thetas2(:, :, parentIndices(2));
    parentsTheta2(:, :, 3) = thetas2(:, :, parentIndices(3));
    parentsTheta2(:, :, 4) = thetas2(:, :, parentIndices(4));

    %% We divide the population into two groups. Each group will be the
    %% offsprings of two parents. If the population number is odd, then
    %% there will be a third group that whose parents will be two random
    %% parents from the four parents.
    populationGroupSize = floor(numPopulation / 2);
    for i = 1:populationGroupSize
      [ theta1, theta2 ] = crossoverParents(parentsTheta1(:, :, 1), ...
                                            parentsTheta1(:, :, 2), ...
                                            parentsTheta2(:, :, 1), ...
                                            parentsTheta2(:, :, 2), ...
                                            mutationProbability, ...
                                            input_layer_size, ...
                                            hidden_layer_size, ...
                                            output_layer_size);
      thetas1(:, :, i) = theta1;
      thetas2(:, :, i) = theta2;
      [ theta1, theta2 ] = crossoverParents(parentsTheta1(:, :, 3), ...
                                            parentsTheta1(:, :, 4), ...
                                            parentsTheta2(:, :, 3), ...
                                            parentsTheta2(:, :, 4), ...
                                            mutationProbability, ...
                                            input_layer_size, ...
                                            hidden_layer_size, ...
                                            output_layer_size);
      thetas1(:, :, i + populationGroupSize) = theta1;
      thetas2(:, :, i + populationGroupSize) = theta2;
    endfor
    
    if (mod(numPopulation, 2) == 1)
      % So there is a third group.
      randomParent1Theta1 = parentsTheta1(:, :, randi(4));
      randomParent2Theta1 = parentsTheta1(:, :, randi(4));
      randomParent1Theta2 = parentsTheta2(:, :, randi(4));
      randomParent2Theta2 = parentsTheta2(:, :, randi(4));
      [ theta1, theta2 ] = crossoverParents(randomParent1Theta1, ...
                                            randomParent2Theta1, ...
                                            randomParent1Theta2, ...
                                            randomParent2Theta2, ...
                                            mutationProbability, ...
                                            input_layer_size, ...
                                            hidden_layer_size, ...
                                            output_layer_size);
      thetas1(:, :, numPopulation) = theta1;
      thetas2(:, :, numPopulation) = theta2;
    endif
endfunction

function parentIndices = selectParents(populationCosts, ...
                                       num_random_parents)
  % 4 columns since we will be getting four parents.
  parentIndices = zeros(1, 4);
  
  for i = 1:num_random_parents
    parentIndex = 0;
    while any(parentIndices == parentIndex) == 1
      parentIndex = randi(rows(populationCosts));
    endwhile
    
    parentIndices(i) = parentIndex;
  endfor
  
  for i = (num_random_parents + 1):4
    bestCost = [ inf ];
    bestCostIndex = [ 0 ];
    for j = 1:rows(populationCosts)
      if (any(parentIndices == j) == 0)
        if (populationCosts(j) < bestCost)
          bestCostIndex = j;
        endif
      endif
    endfor

    parentIndices(i) = bestCostIndex;
  endfor
endfunction
