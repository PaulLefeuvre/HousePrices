function [theta] = normalEquation(X, y, lambda)

n = size(X, 2);

I = eye(2);
I(1, 1) = 0;

theta = pinv(X'*X+lambda*I)*X'*y;

end
