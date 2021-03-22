function [X_norm, mu, sigma] = featuresNormalize(X)
% Normalizes the features in X using its standard deviation and average
% The normalized X-values, the std and mean are returned 

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

end
