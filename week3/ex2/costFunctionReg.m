function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

lambda_arr = zeros(size(theta, 1), 1);
lambda_arr(:) = lambda;
lambda_arr(1) = 0;

theta_X = X * theta;
h_theta_X = sigmoid(theta_X);

term1 = (y' * log(h_theta_X));
term2 = ((1 - y)' * log(1 - h_theta_X));
reg_term = sum((lambda_arr / (2 * m)) .* (theta .^ 2));

J = (-1/m) * (term1 + term2) + reg_term;

grad = ( 1 / m) * ((h_theta_X - y)' * X)' + ((lambda_arr / m) .* theta);





% =============================================================

end
