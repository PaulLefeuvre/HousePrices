clear; close all; clc
fprintf('Loading data ...\n');

X = load('-ascii', 'txt-data/X.txt');
y = load('-ascii', 'txt-data/y.txt');
Xtest = load('-ascii', 'txt-data/Xtest.txt');
ytest = load('-ascii', 'txt-data/ytest.txt');
Xval = load('-ascii', 'txt-data/Xcv.txt');
yval = load('-ascii', 'txt-data/ycv.txt');
X_sub = load('-ascii', 'txt-data/yfinal.txt');

fprintf('Data loaded.\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

m = length(y);

fprintf('\n');
% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n')

% Only normalize the non-one-hot encoded features - they are the ones for which this is actually useful
[X_norm mu sigma] = featuresNormalize(X(:, 1:35));
X = [ones(m, 1) X_norm X(:, 36:size(X, 2))]; % Recombine everything with the intercept term as well

Xnorm = bsxfun(@minus, Xval(:, 1:35), mu);
Xnorm = bsxfun(@rdivide, Xnorm, sigma);
Xval = [ones(size(Xval, 1), 1) Xnorm Xval(:, 36:size(Xval, 2))];

Xnorm = bsxfun(@minus, Xtest(:, 1:35), mu);
Xnorm = bsxfun(@rdivide, Xnorm, sigma);
Xtest = [ones(size(Xtest, 1), 1) Xnorm Xtest(:, 36:size(Xtest, 2))];


fprintf('Features ready\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\n')
fprintf('Finding cost for theta = [0, 0, 0, ..., 0]\n');
[J, grad] = linearRegCostFunction(zeros(size(X, 2), 1), X, y, 0);
fprintf('Cost: %f\n', J);
fprintf('Finding cost for theta = [1, 1, 1, ..., 1]\n');
[J, grad] = linearRegCostFunction(ones(size(X, 2), 1), X, y, 0);
fprintf('Cost: %f\n', J);
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\n');
fprintf('Running gradient descent on dataset without regularization ...\n');
lambda = 0;
alpha = 0.01;
[theta, ~] = trainLinearReg(X, y, lambda, 300, alpha);
[J, ~] = linearRegCostFunction(theta, X, y, lambda);
fprintf('Training cost: %f\n', J);
[J, ~] = linearRegCostFunction(theta, Xtest, ytest, lambda);
fprintf('Test cost: %f\n', J);
fprintf('Program paused. Press enter to continue.\n');
pause;
%{
lambda = 0;
alpha = 0.001;
[error_train, error_val] = ...
    learningCurve(X, y, ...
                  Xval, yval, ...
                  lambda, alpha);
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross validation')
xlabel('Number of training examples')
ylabel('Error')
%axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end



fprintf('Program paused. Press enter to continue.\n');
pause;
%}
fprintf('\n');
fprintf('Using the normal equation without regularization...\n');
lambda = 0;
normal_theta = normalEquation(X, y, lambda);
[trainCost, ~] = linearRegCostFunction(normal_theta, X, y, lambda);
[valCost, ~] = linearRegCostFunction(normal_theta, Xval, yval, lambda);

fprintf('Training cost: %f\n', trainCost);
fprintf('Cross validation cost: %f\n', valCost);

y_pred = X*normal_theta;
fprintf('Predicted y\tActual y\n');
for i = 1:20
  fprintf(' \t%d\t\t%d\n', y_pred(i), y(i));
endfor

fprintf('Program paused. Press enter to continue.\n');
pause;


% Finding the optimum rate for alpha
% using a learning curve of J with each iteration

fprintf('Now finding a suitable learning rate...\n');
%{
alpha = [0.0001, 0.0003, 0.003, 0.01, 0.03];
colors = ['magenta'; 'blue'; 'green'; 'yellow'; 'red'];
maxIter = 200;

hold on;
num_alpha = size(alpha);

for i = 1:num_alpha
  [~, J_hist] = trainLinearReg(X, y, 0, maxIter, alpha(i));
  plot(1:maxIter, J_hist, '-', 'color', colors(i));
endfor
xlabel('Number of iterations');
ylabel('Cost function');
legend('0.0001', '0.0003', '0.003', '0.01', '0.03', '0.1');
hold off;
%}
alpha = 0.03; % The optimum learning rate
fprintf('Optimum learning rate found. Alpha = %f\n', alpha)
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\n');
fprintf('Now finding the optimal regularization parameter...')

%lambda = 0.01; % [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
%num_iters = 200;
