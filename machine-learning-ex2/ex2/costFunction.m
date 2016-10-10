function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
H = sigmoid(X * theta);
for ind = 1:m
    C1 = y(ind)*(-1) * log(H(ind));
    C2 = (1-y(ind)) * log(1-H(ind));
    J = J + (C1-C2)/m;
end

for j = 1:size(theta)
    grad(j) = (1/m) * sum((H-y) .* X(:,j));
end


% =============================================================

end
