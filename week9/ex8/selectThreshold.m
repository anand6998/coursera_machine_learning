function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
scores = zeros(1, 2);

for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    predictions = (pval < epsilon);
    e = [predictions yval];
    %tp prediction = 1 yval = 1
    tps = e(e(:,1) == 1 & e(:, 2) == 1,:);
    %fp - predcition = 1 yval = 0
    fps = e(e(:,1) == 1 & e(:, 2) == 0, :);
    %fn - prediction = 0 yval = 1
    fns = e(e(:,1) == 0 & e(:, 2) == 1, :);
    
    
    tp = size(tps, 1);
    fp = size(fps, 1);
    fn = size(fns, 1);

    prec = tp / (tp + fp);
    rec = tp / (tp + fn);
    
    
    f1_score = 2 * prec * rec / (prec + rec);
    F1 = f1_score;

    scores = [scores; [epsilon, f1_score]];

    % =============================================================
% 
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

% best = max(scores, [], 1);
% bestEpsilon = best(1, 1);
% bestF1 = best(1, 2);
end
