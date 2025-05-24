clear all, close all
% Environment, Define the start, goal, and obstacles
start = [0, 0]; % Initial robot position
goal = [100, 50]; % Goal position
obstacles = [[30, 20, 1]; [50, 20, 1]; [70, 40, 1]]; % Each Obstacle position [x, y, radius]

% obstacles = [[38, 25, 1]; [42, 18, 1]; [45, 23, 1]]; % Obstacle positions form a trap,
% this is a hard case


% Optimize the path
num_waypoints = 50;
initial_guess = [linspace(start(1), goal(1), num_waypoints)', linspace(start(2), goal(2), num_waypoints)'];
path_x_init = initial_guess(:,1);
path_y_init = initial_guess(:,2);

options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');
path_y = fmincon(@(x) objectiveFunction(x, start, goal, obstacles, path_x_init, path_y_init), path_y_init, [], [], [], [], [], [], [], options);

path = [path_x_init, path_y];


%%
% Create a figure for the animation
clf, figure(1);
fig = gcf;
fig.Position = [200, 100, 900, 450]; % You can adjust these values as needed

axis equal
xlim([0 101]);
ylim([0 60]);

hold on; box on;
plot(start(1), start(2), 'go', 'MarkerSize', 15);
plot(goal(1), goal(2), 'ro', 'MarkerSize', 15);
for obs = obstacles'
    viscircles([obs(1), obs(2)], obs(3));
end
plot(path(:,1), path(:,2), 'b.-');
xlabel('X'); ylabel('Y'); title('Optimal Agent Path Planning with fmincon');
grid on;
axis equal;

%%

function cost = objectiveFunction(path_y, start, goal, obstacles, path_x_init, path_y_init)
    path = [path_x_init, path_y];
    % Path length component
    path_length = sum(sqrt(sum(diff([start; path; goal], 1, 1).^2, 2)));
    
    % Repulsive potential field component
    repulsive_cost = 0;
    k_rep = 50;
    rho_0 = 10; % Effective distance for the repulsive field
    
    for obs = obstacles' %  In MATLAB, when iterating over the rows of a matrix using a for-loop, the loop variable directly takes on the values of the rows
        center = obs(1:2);
        radius = obs(3);
        
        for pt = path'
            d = norm(pt - center); % Distance from current point to the obstacle center
            if d < rho_0
                repulsive_cost = repulsive_cost + k_rep * ((1/d) - (1/rho_0)).^2 * norm(pt - center');
            end
        end
    end
    
    % Penalty for lack of smoothness (based on second derivative)
    dq = diff(path, 1, 1);
    ddq = diff(dq, 1, 1);
    
    smoothness_dq = sum(sqrt(sum(dq.^2, 2)));
    smoothness_ddq = sum(sqrt(sum(ddq.^2, 2)));

    % path_deviation = sum( (path_y - path_y_init).^2 ); % deviation for
    % the whole path

    path_deviation = (path_y(1) - path_y_init(1)).^2 + (path_y(end) - path_y_init(end)).^2; % deviation for
    % only the start-end positions 

    cost = 1.0*path_length + 1*path_deviation + 1*repulsive_cost + 1*smoothness_dq + 1*smoothness_ddq;
end
