clear; close all; clc
fprintf('Loading data ...\n');

X = csvread('C:\Users\Paul Lefeuvre\Desktop\Projects\MachineLearning\HousePrices\processed_Xtrain.csv');
y = csvread('C:\Users\Paul Lefeuvre\Desktop\Projects\MachineLearning\HousePrices\processed_ytrain.csv');
Xtest = csvread('C:\Users\Paul Lefeuvre\Desktop\Projects\MachineLearning\HousePrices\processed_Xtest.csv');
ytest = csvread('C:\Users\Paul Lefeuvre\Desktop\Projects\MachineLearning\HousePrices\processed_ytest.csv');
Xval = csvread('C:\Users\Paul Lefeuvre\Desktop\Projects\MachineLearning\HousePrices\processed_Xcv.csv');
yval = csvread('C:\Users\Paul Lefeuvre\Desktop\Projects\MachineLearning\HousePrices\processed_ycv.csv');
X_sub = csvread('C:\Users\Paul Lefeuvre\Desktop\Projects\MachineLearning\HousePrices\processed_sub_test.csv');

% NaN values seem to be returned for 38, 75, 91, 96, 107, 128, 133, 148, 168, 216, 236, 267, 307, and 323
% Probably because it can't really import csv files - change to .TXT

fprintf('Data loaded.\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

m = length(y);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n')

[X mu sigma] = featuresNormalize(X);
X = [ones(m, 1) X]; % Add intercept term

Xval = bsxfun(@minus, Xval, mu);
Xval = bsxfun(@rdivide, Xval, sigma);
Xval = [ones(size(Xval, 1), 1) Xval];

Xtest = bsxfun(@minus, Xtest, mu);
Xtest = bsxfun(@rdivide, Xtest, sigma);
Xtest = [ones(size(Xtest, 1), 1) Xtest];


fprintf('Features ready\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%J = linearRegCostFunction(ones(size(X, 2)))
fprintf('')

fprintf('Running gradient descent on dataset without regularization ...\n');

lambda = 0;
alpha = 0.1;
[error_train, error_val] = ...
    learningCurve(X, y, ...
                  Xval, yval, ...
                  lambda, alpha);
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Using the normal equation without regularization...\n');
lambda = 0;
normal_theta = normalEquation(X, y, lambda);
[trainCost, ~] = linearRegCostFunction(normal_theta, X, y, lambda);
[valCost, ~] = linearRegCostFunction(normal_theta, Xval, yval, lambda);

fprintf('Training cost: %f', trainCost);
fprintf('Cross validation cost: %f', valCost);

fprintf('Program paused. Press enter to continue.\n');
pause;

%lambda = 0.01; % [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
%num_iters = 200;
