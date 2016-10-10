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

H = sigmoid(X * theta);
for ind = 1:m
    C1 = y(ind)*(-1) * log(H(ind));
    C2 = (1-y(ind)) * log(1-H(ind));
    J = J + (C1-C2)/m;
end
J = J + (lambda * (sum(theta.^2) - theta(1)^2))/(2*m);

for j = 1:size(theta)
    if j == 1
        grad(j) = (1/m) * sum((H-y) .* X(:,j));
    else
        grad(j) = (1/m) * sum((H-y) .* X(:,j)) + (lambda * theta(j))/m;
    end
end




% =============================================================

end
