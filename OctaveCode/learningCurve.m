function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda, alpha)
% Find and return the training and cross-validation errors for the dataset at
% various dataset sizes. Used to detect high bias or high variance once graphed

m = size(X, 1);

error_train = zeros(m, 1);
error_val = zeros(m, 1);

for i = 1:m
  theta = trainLinearReg(X(1:i, :), y(1:i), lambda, 200, alpha);
  error_train(i) = linearRegCostFunction(theta, X(1:i, :), y(1:i), lambda);
  error_val(i) = linearRegCostFunction(theta, Xval, yval, lambda);
endfor

end
