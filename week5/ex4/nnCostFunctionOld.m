function [J grad] = nnCostFunctionOld(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m, 1) X];
a2 = [ones(m, 1) sigmoid(a1 * Theta1')];
h = sigmoid(a2 * Theta2');

yVec = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

cost = yVec .* log(h) + (1 - yVec) .* log(1 -h);
J = - ( 1 / m ) * sum(sum(cost));

Theta1reg = Theta1(:, 2:end);
Theta2reg = Theta2(:, 2:end);

reg1 = sum(sum(Theta1reg .^ 2));
reg2 = sum(sum(Theta2reg .^ 2));

reg = (lambda / ( 2 * m)) * (reg1 + reg2);
J = J + reg;

delta1 = zeros(hidden_layer_size, input_layer_size + 1);
delta2 = zeros(num_labels, hidden_layer_size + 1);

z2 = a1 * Theta1';

for i = 1:m
    z2i = z2(i, :);
    a2i = [ones(1, 1) sigmoid(z2i)];
    
    z3i = a2i * Theta2';
    a3i = sigmoid(z3i);
    
    yi = yVec(i, :);
    d3i = a3i - yi;
    
    d2i = (d3i * Theta2) .* [1 sigmoidGradient(z2i)]; 
    
    % Delta2 = Delta2 + a[j]  * delta[i]_[3]
    delta2 = delta2 + (d3i' * a2i);
    
    % Delta1
    a1i = a1(i, :);
    d2iX = d2i(2:end);
    delta1 = delta1 + (d2iX' * a1i);
    
end;

Theta1_grad = (1 / m) * delta1;
Theta2_grad = (1 / m) * delta2;

% Regularization

theta2ZeroBias = Theta2;
theta2ZeroBias(: ,1) = 0;

theta1ZeroBias = Theta1;
theta1ZeroBias(:, 1) = 0;

Theta1_grad = Theta1_grad + (lambda / m) * theta1ZeroBias;
Theta2_grad = Theta2_grad + (lambda / m) * theta2ZeroBias;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
