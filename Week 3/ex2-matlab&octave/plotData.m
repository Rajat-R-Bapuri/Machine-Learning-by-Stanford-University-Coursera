function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

a = find(y == 1);
b = find(y == 0);
plot(X(a), y(a), 'r+', 'MarkerSize', 10);
plot(X(b), y(b), 'ro', 'MarkerSize', 10);

% =========================================================================

hold off;

end
