%% MATLAB Fundamentals
% A comprehensive introduction to MATLAB programming with visualization
% This script covers essential MATLAB concepts along with plotting examples
%
% Course Overview:
% 1. Variables and Data Types
% 2. Vectors and Matrices
% 3. Array Operations
% 4. Control Structures
% 5. Functions
% 6. Introduction to Plotting
% 7. Advanced Plotting Techniques
% 8. File I/O
% 9. Error Handling
% 10. Practice Exercises

%% Clear workspace and command window
clear all;
close all;
clc;

%% 1. VARIABLES AND DATA TYPES

fprintf('=== SECTION 1: VARIABLES AND DATA TYPES ===\n\n');

% Scalar variables
age = 25;
temperature = 23.5;
name = 'Alice';
is_student = true;

fprintf('Name: %s (Type: %s)\n', name, class(name));
fprintf('Age: %d (Type: %s)\n', age, class(age));
fprintf('Temperature: %.1f (Type: %s)\n', temperature, class(temperature));
fprintf('Is student: %d (Type: %s)\n', is_student, class(is_student));

% Arithmetic operations
a = 10;
b = 3;

fprintf('\n--- Arithmetic Operations ---\n');
fprintf('Addition: %d + %d = %d\n', a, b, a + b);
fprintf('Subtraction: %d - %d = %d\n', a, b, a - b);
fprintf('Multiplication: %d * %d = %d\n', a, b, a * b);
fprintf('Division: %d / %d = %.2f\n', a, b, a / b);
fprintf('Power: %d ^ %d = %d\n', a, b, a ^ b);
fprintf('Modulus: mod(%d, %d) = %d\n', a, b, mod(a, b));

% Visualize arithmetic operations
operations = {'Add', 'Sub', 'Mult', 'Div', 'Power', 'Mod'};
results = [a+b, a-b, a*b, a/b, a^b, mod(a,b)];

figure('Name', 'Arithmetic Operations', 'Position', [100, 100, 1000, 500]);
bar(results, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', operations, 'FontSize', 12);
xlabel('Operation', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Result', 'FontSize', 13, 'FontWeight', 'bold');
title(sprintf('Arithmetic Operations: %d and %d', a, b), 'FontSize', 14, 'FontWeight', 'bold');
grid on;
% Add value labels on bars
for i = 1:length(results)
    text(i, results(i), sprintf('%.2f', results(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 11, 'FontWeight', 'bold');
end

%% 2. VECTORS AND MATRICES

fprintf('\n=== SECTION 2: VECTORS AND MATRICES ===\n\n');

% Row vector
row_vector = [1, 2, 3, 4, 5];
fprintf('Row vector: ');
disp(row_vector);

% Column vector
col_vector = [1; 2; 3; 4; 5];
fprintf('Column vector:\n');
disp(col_vector);

% Matrix
matrix = [1, 2, 3; 4, 5, 6; 7, 8, 9];
fprintf('Matrix:\n');
disp(matrix);

% Matrix operations
fprintf('Matrix dimensions: %d x %d\n', size(matrix, 1), size(matrix, 2));
fprintf('Matrix transpose:\n');
disp(matrix');

% Creating special matrices
zeros_mat = zeros(3, 3);
ones_mat = ones(3, 3);
identity_mat = eye(3);
random_mat = rand(3, 3);

fprintf('Zeros matrix:\n');
disp(zeros_mat);

% Visualize matrices
figure('Name', 'Matrix Visualizations', 'Position', [100, 100, 1200, 800]);

subplot(2, 3, 1);
imagesc(matrix);
colorbar;
title('Original Matrix', 'FontSize', 12, 'FontWeight', 'bold');
axis equal tight;
colormap(gca, 'parula');

subplot(2, 3, 2);
imagesc(ones_mat);
colorbar;
title('Ones Matrix', 'FontSize', 12, 'FontWeight', 'bold');
axis equal tight;

subplot(2, 3, 3);
imagesc(identity_mat);
colorbar;
title('Identity Matrix', 'FontSize', 12, 'FontWeight', 'bold');
axis equal tight;

subplot(2, 3, 4);
imagesc(random_mat);
colorbar;
title('Random Matrix', 'FontSize', 12, 'FontWeight', 'bold');
axis equal tight;

subplot(2, 3, 5);
bar3(matrix);
title('3D Bar Plot of Matrix', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Column');
ylabel('Row');
zlabel('Value');

subplot(2, 3, 6);
surf(random_mat);
title('Surface Plot of Random Matrix', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Column');
ylabel('Row');
zlabel('Value');
colorbar;

%% 3. ARRAY OPERATIONS

fprintf('\n=== SECTION 3: ARRAY OPERATIONS ===\n\n');

% Create arrays
x = 1:10;  % Vector from 1 to 10
y = linspace(0, 2*pi, 100);  % 100 points from 0 to 2π

fprintf('Array x (1 to 10):\n');
disp(x);

% Element-wise operations
squares = x.^2;
cubes = x.^3;

fprintf('Squares: ');
disp(squares);
fprintf('Cubes: ');
disp(cubes);

% Array indexing
fprintf('\nFirst element: %d\n', x(1));
fprintf('Last element: %d\n', x(end));
fprintf('Elements 3 to 7: ');
disp(x(3:7));

% Visualize element-wise operations
figure('Name', 'Array Operations', 'Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
plot(x, squares, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'Color', [0.2 0.4 0.8]);
hold on;
plot(x, cubes, 's-', 'LineWidth', 2, 'MarkerSize', 8, 'Color', [0.8 0.2 0.2]);
xlabel('x', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('y', 'FontSize', 12, 'FontWeight', 'bold');
title('Squares vs Cubes', 'FontSize', 13, 'FontWeight', 'bold');
legend('x^2', 'x^3', 'Location', 'northwest', 'FontSize', 11);
grid on;
hold off;

subplot(1, 2, 2);
% Trigonometric functions
plot(y, sin(y), 'LineWidth', 2, 'DisplayName', 'sin(x)');
hold on;
plot(y, cos(y), 'LineWidth', 2, 'DisplayName', 'cos(x)');
xlabel('x (radians)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('y', 'FontSize', 12, 'FontWeight', 'bold');
title('Trigonometric Functions', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
grid on;
hold off;

%% 4. CONTROL STRUCTURES

fprintf('\n=== SECTION 4: CONTROL STRUCTURES ===\n\n');

% If-else statements
fprintf('--- Conditional Statements ---\n');
score = 85;

if score >= 90
    grade = 'A';
elseif score >= 80
    grade = 'B';
elseif score >= 70
    grade = 'C';
elseif score >= 60
    grade = 'D';
else
    grade = 'F';
end

fprintf('Score: %d, Grade: %s\n', score, grade);

% For loops
fprintf('\n--- For Loops ---\n');
fprintf('Counting to 5:\n');
for i = 1:5
    fprintf('%d ', i);
end
fprintf('\n');

% Generate Fibonacci sequence
n = 15;
fib = zeros(1, n);
fib(1) = 0;
fib(2) = 1;

for i = 3:n
    fib(i) = fib(i-1) + fib(i-2);
end

fprintf('Fibonacci sequence (first %d terms):\n', n);
disp(fib);

% While loops
fprintf('\n--- While Loops ---\n');
count = 1;
fprintf('Powers of 2 less than 100:\n');
while 2^count < 100
    fprintf('%d ', 2^count);
    count = count + 1;
end
fprintf('\n');

% Visualize control structures
figure('Name', 'Control Structures', 'Position', [100, 100, 1200, 500]);

% Temperature categories
subplot(1, 2, 1);
temperatures = [5, 15, 25, 35, 45];
categories = cell(1, length(temperatures));
colors_map = zeros(length(temperatures), 3);

for i = 1:length(temperatures)
    temp = temperatures(i);
    if temp < 10
        categories{i} = 'Cold';
        colors_map(i, :) = [0.2 0.4 0.8];  % Blue
    elseif temp < 20
        categories{i} = 'Cool';
        colors_map(i, :) = [0.4 0.8 0.8];  % Cyan
    elseif temp < 30
        categories{i} = 'Warm';
        colors_map(i, :) = [0.9 0.9 0.2];  % Yellow
    else
        categories{i} = 'Hot';
        colors_map(i, :) = [0.8 0.2 0.2];  % Red
    end
end

bar(temperatures, 'FaceColor', 'flat', 'CData', colors_map, 'EdgeColor', 'k', 'LineWidth', 1.5);
xlabel('Sample', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Temperature (°C)', 'FontSize', 12, 'FontWeight', 'bold');
title('Temperature Categories', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'XTickLabel', {'T1', 'T2', 'T3', 'T4', 'T5'});
grid on;

% Add labels
for i = 1:length(temperatures)
    text(i, temperatures(i), sprintf('%s\n%d°C', categories{i}, temperatures(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 10, 'FontWeight', 'bold');
end

% Fibonacci visualization
subplot(1, 2, 2);
semilogy(1:n, fib, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'Color', [0.6 0.2 0.8]);
xlabel('Index', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Value (log scale)', 'FontSize', 12, 'FontWeight', 'bold');
title('Fibonacci Sequence', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

%% 5. FUNCTIONS

fprintf('\n=== SECTION 5: FUNCTIONS ===\n\n');

% Define functions at the end of the script

% Call custom functions
fprintf('--- Custom Functions ---\n');

% Linear function
x_vals = -5:0.1:5;
y_linear = arrayfun(@(x) linear_func(x, 2, 1), x_vals);
fprintf('Linear function: y = 2x + 1\n');

% Quadratic function
y_quad = arrayfun(@(x) quadratic_func(x, 0.5, 0, 0), x_vals);
fprintf('Quadratic function: y = 0.5x^2\n');

% Exponential function
y_exp = arrayfun(@(x) exponential_func(x, 1, 0.2), x_vals);
fprintf('Exponential function: y = e^(0.2x)\n');

% Statistical functions
data = randn(1, 100) * 15 + 50;
[mean_val, median_val, std_val] = calculate_stats(data);
fprintf('\nStatistical Analysis:\n');
fprintf('Mean: %.2f\n', mean_val);
fprintf('Median: %.2f\n', median_val);
fprintf('Standard Deviation: %.2f\n', std_val);

% Visualize functions
figure('Name', 'Mathematical Functions', 'Position', [100, 100, 1400, 500]);

subplot(1, 3, 1);
plot(x_vals, y_linear, 'LineWidth', 2, 'Color', [0.2 0.4 0.8]);
xlabel('x', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('y', 'FontSize', 11, 'FontWeight', 'bold');
title('Linear: y = 2x + 1', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

subplot(1, 3, 2);
plot(x_vals, y_quad, 'LineWidth', 2, 'Color', [0.2 0.8 0.4]);
xlabel('x', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('y', 'FontSize', 11, 'FontWeight', 'bold');
title('Quadratic: y = 0.5x^2', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

subplot(1, 3, 3);
plot(x_vals, y_exp, 'LineWidth', 2, 'Color', [0.8 0.2 0.2]);
xlabel('x', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('y', 'FontSize', 11, 'FontWeight', 'bold');
title('Exponential: y = e^{0.2x}', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Visualize statistics
figure('Name', 'Statistical Analysis', 'Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
histogram(data, 20, 'FaceColor', [0.4 0.6 0.8], 'EdgeColor', 'k');
hold on;
yl = ylim;
plot([mean_val mean_val], yl, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mean: %.2f', mean_val));
plot([median_val median_val], yl, 'g--', 'LineWidth', 2, 'DisplayName', sprintf('Median: %.2f', median_val));
xlabel('Value', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 11, 'FontWeight', 'bold');
title('Data Distribution', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10);
grid on;
hold off;

subplot(1, 2, 2);
boxplot(data, 'Widths', 0.5);
ylabel('Value', 'FontSize', 11, 'FontWeight', 'bold');
title('Box Plot', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

%% 6. INTRODUCTION TO PLOTTING

fprintf('\n=== SECTION 6: INTRODUCTION TO PLOTTING ===\n\n');

% Basic 2D plots
t = 0:0.01:2*pi;
y1 = sin(t);
y2 = cos(t);

figure('Name', 'Basic 2D Plots', 'Position', [100, 100, 1400, 900]);

% Line plot
subplot(2, 3, 1);
plot(t, y1, 'LineWidth', 2);
xlabel('Time (radians)', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
title('Sine Wave', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Multiple lines
subplot(2, 3, 2);
plot(t, y1, 'b-', t, y2, 'r--', 'LineWidth', 2);
xlabel('Time (radians)', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
title('Sine and Cosine', 'FontSize', 12, 'FontWeight', 'bold');
legend('sin(t)', 'cos(t)', 'Location', 'northeast');
grid on;

% Scatter plot
subplot(2, 3, 3);
x_scatter = randn(50, 1);
y_scatter = 2*x_scatter + randn(50, 1)*0.5;
scatter(x_scatter, y_scatter, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('X', 'FontSize', 10);
ylabel('Y', 'FontSize', 10);
title('Scatter Plot', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Bar plot
subplot(2, 3, 4);
categories = {'A', 'B', 'C', 'D', 'E'};
values = [23, 45, 32, 67, 41];
bar(values, 'FaceColor', [0.3 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', categories);
ylabel('Value', 'FontSize', 10);
title('Bar Chart', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Stem plot
subplot(2, 3, 5);
stem(1:10, rand(1, 10), 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Index', 'FontSize', 10);
ylabel('Value', 'FontSize', 10);
title('Stem Plot', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Area plot
subplot(2, 3, 6);
area(t, abs(sin(t)), 'FaceColor', [0.8 0.4 0.4], 'EdgeColor', 'k', 'LineWidth', 1.5);
xlabel('Time (radians)', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', 10);
title('Area Plot', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

%% 7. ADVANCED PLOTTING TECHNIQUES

fprintf('\n=== SECTION 7: ADVANCED PLOTTING TECHNIQUES ===\n\n');

% 3D plots
figure('Name', '3D Visualizations', 'Position', [100, 100, 1400, 600]);

% 3D line plot
subplot(1, 3, 1);
t_3d = 0:pi/50:10*pi;
x_3d = sin(t_3d);
y_3d = cos(t_3d);
z_3d = t_3d;
plot3(x_3d, y_3d, z_3d, 'LineWidth', 2);
xlabel('X', 'FontSize', 10);
ylabel('Y', 'FontSize', 10);
zlabel('Z', 'FontSize', 10);
title('3D Spiral', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
view(3);

% 3D scatter
subplot(1, 3, 2);
x_scatter3 = randn(100, 1);
y_scatter3 = randn(100, 1);
z_scatter3 = x_scatter3.^2 + y_scatter3.^2;
scatter3(x_scatter3, y_scatter3, z_scatter3, 50, z_scatter3, 'filled');
colorbar;
xlabel('X', 'FontSize', 10);
ylabel('Y', 'FontSize', 10);
zlabel('Z', 'FontSize', 10);
title('3D Scatter Plot', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% 3D surface
subplot(1, 3, 3);
[X, Y] = meshgrid(-5:0.2:5, -5:0.2:5);
Z = sin(sqrt(X.^2 + Y.^2));
surf(X, Y, Z, 'EdgeColor', 'none');
colormap('jet');
colorbar;
xlabel('X', 'FontSize', 10);
ylabel('Y', 'FontSize', 10);
zlabel('Z', 'FontSize', 10);
title('3D Surface Plot', 'FontSize', 12, 'FontWeight', 'bold');
shading interp;
view(3);

% Contour plots
figure('Name', 'Contour Plots', 'Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
contour(X, Y, Z, 20);
colorbar;
xlabel('X', 'FontSize', 11);
ylabel('Y', 'FontSize', 11);
title('Contour Plot', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

subplot(1, 2, 2);
contourf(X, Y, Z, 20);
colorbar;
xlabel('X', 'FontSize', 11);
ylabel('Y', 'FontSize', 11);
title('Filled Contour Plot', 'FontSize', 13, 'FontWeight', 'bold');

%% 8. FILE I/O

fprintf('\n=== SECTION 8: FILE I/O ===\n\n');

% Write data to file
data_to_save = 1:10;
filename = 'matlab_data.txt';

fileID = fopen(filename, 'w');
fprintf(fileID, '%d\n', data_to_save);
fclose(fileID);

fprintf('Data written to %s\n', filename);

% Read data from file
fileID = fopen(filename, 'r');
data_from_file = fscanf(fileID, '%d');
fclose(fileID);

fprintf('Data read from file:\n');
disp(data_from_file');

% Plot the data
figure('Name', 'File I/O Data', 'Position', [100, 100, 800, 500]);
plot(data_from_file, 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.2 0.6 0.2]);
xlabel('Index', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Value', 'FontSize', 11, 'FontWeight', 'bold');
title('Data from File', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

%% 9. ERROR HANDLING

fprintf('\n=== SECTION 9: ERROR HANDLING ===\n\n');

fprintf('--- Safe Division Examples ---\n');

% Test safe division
test_pairs = [10, 2; 10, 0; 15, 3; 20, 4];

results = zeros(size(test_pairs, 1), 1);
for i = 1:size(test_pairs, 1)
    results(i) = safe_divide(test_pairs(i, 1), test_pairs(i, 2));
end

% Plot valid results
valid_idx = ~isnan(results);
valid_results = results(valid_idx);

figure('Name', 'Error Handling', 'Position', [100, 100, 800, 500]);
bar(find(valid_idx), valid_results, 'FaceColor', [0.3 0.5 0.7], 'EdgeColor', 'k', 'LineWidth', 1.5);
xlabel('Test Case', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Result', 'FontSize', 11, 'FontWeight', 'bold');
title('Valid Division Results', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

%% 10. PRACTICE EXERCISES

fprintf('\n=== SECTION 10: PRACTICE EXERCISES ===\n\n');
fprintf('Try these exercises:\n');
fprintf('1. Create a function to find prime numbers up to N and plot them\n');
fprintf('2. Implement temperature conversion (C, F, K) with plotting\n');
fprintf('3. Create an animated plot of a sine wave\n');
fprintf('4. Generate and visualize random walk data\n\n');

fprintf('=== END OF TUTORIAL ===\n');
fprintf('Congratulations! You''ve completed the MATLAB fundamentals course.\n\n');

%% FUNCTION DEFINITIONS

% Linear function
function y = linear_func(x, m, b)
    % y = mx + b
    y = m * x + b;
end

% Quadratic function
function y = quadratic_func(x, a, b, c)
    % y = ax^2 + bx + c
    y = a * x^2 + b * x + c;
end

% Exponential function
function y = exponential_func(x, a, b)
    % y = a * e^(bx)
    y = a * exp(b * x);
end

% Calculate statistics
function [mean_val, median_val, std_val] = calculate_stats(data)
    % Calculate mean, median, and standard deviation
    mean_val = mean(data);
    median_val = median(data);
    std_val = std(data);
end

% Safe division with error handling
function result = safe_divide(a, b)
    % Safely divide two numbers
    try
        if b == 0
            warning('Division by zero! Returning NaN.');
            result = NaN;
        else
            result = a / b;
            fprintf('%d / %d = %.2f\n', a, b, result);
        end
    catch ME
        warning('Error occurred: %s. Returning NaN.', ME.message);
        result = NaN;
    end
end
