function [initThetas1, initThetas2] = generateInitialPopulation(numInitial, ...
                                                              numIn, ...
                                                              numHidden, ...
                                                              numOut)
  initThetas1 = zeros(numHidden,1 + numIn, numInitial);
  initThetas2 = zeros(numOut, 1 + numHidden, numInitial);
  for i = 1:numInitial
    initThetas1(:, :, i) = randInitializeWeights(numIn, numHidden);
    initThetas2(:, :, i) = randInitializeWeights(numHidden, numOut);
  endfor
end
