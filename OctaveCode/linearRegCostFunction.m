function [J, grad] = linearRegCostFunction(theta, X, y, lambda)
% Compute cost and gradient for regularized linear regression,
% so it can be used in an advanced function for gradient descent later.


m = length(y); % Number of training examples

% Compute the overall cost
J = (1/(2*m))*sum((X*theta-y).^2) + (lambda/(2*m))*sum(theta.^2);
J -= (lambda/(2*m))*(theta(1)^2); % Don't regularize the origin parameter

% Find the theta-wise gradient
grad = (1/m)*(X'*(X*theta-y)) + (lambda/m)*theta;
grad(1) -= (lambda/m)*theta(1); % Again, don't regularize the origin paramter
