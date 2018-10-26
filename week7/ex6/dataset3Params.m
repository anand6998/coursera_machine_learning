function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_try=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_try=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
k = 1;
C_size = size(C_try, 1);
sigma_size = size(sigma_try, 1);

errors = zeros(C_size * sigma_size, 3);

for i = 1:C_size
    C= C_try(i);
    for j = 1:sigma_size
       sigma = sigma_try(j);
       model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
       predictions = svmPredict(model,Xval);

       error = mean(double(predictions ~= yval));
       
       errors(k, 1) = C;
       errors(k, 2) = sigma;
       errors(k, 3) = error;
       k = k + 1;
    end
end
% =========================================================================

[min_val, min_idx] = min(errors(:, 3), [], 1);
min_C = errors(min_idx, 1);
min_sigma = errors(min_idx, 2);

C = min_C;
sigma = min_sigma;
end
