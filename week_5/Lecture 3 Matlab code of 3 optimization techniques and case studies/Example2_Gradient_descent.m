% Create synthetic data for the function f = x^2 * sin(x)
x_data = linspace(0, 7, 400);
y_data = x_data.^2 .* sin(x_data);

% Gradient descent optimization
initial_guess = 2.5; % Starting point for search
step_size = 0.01;
max_iterations = 50;
current_point = initial_guess;
history = zeros(max_iterations, 1); % to store the history of our points

figure(1);
clf;
plot(x_data, y_data, 'g', 'LineWidth', 2); % Plotting the original function
hold on;
title('Gradient Descent Search for Local Minima of f = x^2 * sin(x)');
xlabel('x');
ylabel('y');
grid on;

for i = 1:max_iterations
    gradient = compute_gradient(current_point);
    current_point = current_point - step_size * gradient;
    
    % Visualization: Plotting the current point on the function
    plot(current_point, objective(current_point), 'ro');
    pause(0.1); % adding a pause for visualization purposes
    
    history(i) = current_point;
end

legend('Function', 'Search Points');

% Displaying the found minimum
disp(['Estimated local minimum near x = ', num2str(current_point)]);



% Define the objective function
function value = objective(x)
    value = x^2 * sin(x);
end

% Compute the gradient of the objective function with respect to x
function grad = compute_gradient(x)
    grad = 2*x*sin(x) + x^2*cos(x);
end
