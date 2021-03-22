function [theta] = trainLinearReg(X, y, lambda, maxIter, alpha)
% Uses gradient descent over the iteration count to reduce the cost function value to its bare minimum.
% Iterates over theta every iteration to find the gradient and apply it

%Initialize Theta
theta = zeros(size(X, 2), 1);

for i = 1:maxIter
  [J, grad] = linearRegCostFunction(theta, X, y, lambda);
  theta = theta - alpha*grad;
endfor

end
