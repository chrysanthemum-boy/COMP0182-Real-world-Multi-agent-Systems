% readme: This code takes a set of measurements from an infrared (IR) sensor and the corresponding distances. 
% It uses a polynomial to approximate the relationship between the IR reading and the distance. 
% The approximation is done using the least squares method. 
% Both the original measurements and the approximations are plotted

% Clear all previous data and close any figures
clear all;
close all;

% Define the IR sensor readings and the corresponding distances in cm
datapoints = [399.247371, 15;
              202.418285, 30;
              131.059244, 50;
              86.047343,  80;
              77.036172,  100;
              63.934825,  120;
              62.730596,  140];

% Extract the IR sensor readings and corresponding distances
x = datapoints(:, 1);
y = datapoints(:, 2);

% Define a polynomial feature transformation for curve approximation
features = @(x) [1 x^-1 x^-2 x^-3 x^-4]; % see slides for details

% Apply the feature transformation to each IR reading
xx = arrayfun(@(x)features(x), x, 'uniformoutput', false);
xx = reshape(cell2mat(xx), length(x), length(xx{1}));

% Compute the coefficients using the pseudoinverse method
coeff = pinv(xx) * y;

% Estimate the distances using the computed coefficients
est_datapoints = xx * coeff;

% Calculate the error between estimated and actual distances
error = est_datapoints - y;
sum_errors = sum(error.^2);

% Display results
disp('Coefficient');
disp(coeff);

disp('Mean and std of error');
disp([mean(error), std(error)]);

% Given any single voltage measurement x, you can use y =  [1 x x^2 x^3 x^4]*coeff
% to calculate the distance

% Generate new test points for estimation
x2 = linspace(min(x), max(x), 50)';
xx2 = arrayfun(@(x)features(x), x2, 'uniformoutput', false);
xx2 = reshape(cell2mat(xx2), length(xx2), length(xx2{1}));
testpoints = xx2 * coeff;

% Plotting the results
% Plot original measurements and least squares approximations
figure(1);
clf; 
hold on; 
box on;
plot(x, y, 'bx-');
plot(x, est_datapoints, 'rx-');
xlabel('IR Reading');
ylabel('Actual Distance (cm)');
title(['Residual sum of squares = ' num2str(sum_errors)]);
legend('Measurement', 'Least Squares');

% Plot original measurements and estimations using new test points
figure(2);
clf; 
hold on; 
box on;
plot(x, y, 'bx-');
plot(x2, testpoints, 'gx-');
xlabel('IR Reading');
ylabel('Actual Distance (cm)');
title(['Residual sum of squares = ' num2str(sum_errors)]);
legend('Measurement', 'Estimation');

