%% 8-Bit Pixel Art Drawer - MATLAB Version
% =========================================================================
% Create retro videogame-style pixel art from simple number matrices!
% Each number in your matrix represents a different color in RGB space.
% =========================================================================

%% Setup
% Clear workspace and close all figures
clear all;
close all;
clc;

fprintf('8-Bit Pixel Art Drawer\n');
fprintf('======================\n\n');

%% Main Function Definition
% Note: In MATLAB, we define the main function at the end of the file
% For now, we'll use it inline in the examples

%% Default Color Palette
% Here are the default colors available:
% - 0: Black (background)
% - 1: White
% - 2: Red
% - 3: Green
% - 4: Blue
% - 5: Yellow
% - 6: Orange
% - 7: Purple
% - 8: Indigo
% - 9: Cyan

fprintf('Default Color Palette:\n');
fprintf('0: Black (background)\n');
fprintf('1: White\n');
fprintf('2: Red\n');
fprintf('3: Green\n');
fprintf('4: Blue\n');
fprintf('5: Yellow\n');
fprintf('6: Orange\n');
fprintf('7: Purple\n');
fprintf('8: Indigo\n');
fprintf('9: Cyan\n\n');

%% Example 1: Simple 4×4 Sprite
fprintf('Example 1: Simple 4x4 Sprite\n');
fprintf('=============================\n');

sprite = [
    0, 0, 0, 0;
    0, 2, 3, 4;
    0, 1, 1, 1;
    0, 0, 1, 1
];

% Draw the sprite
draw_pixel_art(sprite, [], 4, 'Simple Sprite');

%% Example 2: Space Invader
fprintf('\nExample 2: Space Invader\n');
fprintf('========================\n');

invader = [
    0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0;
    0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0;
    0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0;
    0, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0;
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2;
    2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2;
    2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2;
    0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0
];

draw_pixel_art(invader, [], 4, 'Space Invader');

%% Example 3: Custom Color Palette
fprintf('\nExample 3: Custom Color Palette\n');
fprintf('===============================\n');

% Define custom colors (RGB values from 0 to 1)
custom_palette = containers.Map('KeyType', 'double', 'ValueType', 'any');
custom_palette(0) = [0.1, 0.1, 0.15];  % Dark blue background
custom_palette(1) = [1.0, 0.84, 0.0];  % Gold
custom_palette(2) = [0.8, 0.2, 0.2];   % Dark red
custom_palette(3) = [0.2, 0.8, 0.2];   % Bright green

heart = [
    0, 2, 2, 0, 0, 2, 2, 0;
    2, 1, 1, 2, 2, 1, 1, 2;
    2, 1, 1, 1, 1, 1, 1, 2;
    2, 1, 1, 1, 1, 1, 1, 2;
    0, 2, 1, 1, 1, 1, 2, 0;
    0, 0, 2, 1, 1, 2, 0, 0;
    0, 0, 0, 2, 2, 0, 0, 0
];

draw_pixel_art(heart, custom_palette, 4, 'Custom Palette Heart');

%% Example 4: Mario-Style Mushroom
fprintf('\nExample 4: Mario-Style Mushroom\n');
fprintf('===============================\n');

mushroom = [
    0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0;
    0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0;
    0, 2, 2, 1, 1, 2, 1, 1, 2, 2, 0;
    2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2;
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2;
    0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0;
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0;
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0;
    0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
];

draw_pixel_art(mushroom, [], 4, 'Mushroom Power-up');

%% Create Your Own!
fprintf('\nCreate Your Own Design\n');
fprintf('======================\n');

my_sprite = [
    0, 0, 0, 0, 0;
    0, 2, 2, 2, 0;
    0, 2, 5, 2, 0;
    0, 2, 2, 2, 0;
    0, 0, 0, 0, 0
];

draw_pixel_art(my_sprite, [], 4, 'My Sprite');

%% Saving Your Pixel Art
fprintf('\nSaving Pixel Art Example\n');
fprintf('========================\n');

% Create a detailed design
my_design = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0;
    0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0;
    0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 1, 0, 0;
    0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 1, 1, 4, 4, 1, 2, 2, 2, 1, 0, 0, 0;
    0, 0, 0, 1, 2, 2, 2, 2, 1, 1, 2, 2, 4, 1, 2, 2, 2, 1, 0, 0, 0, 0;
    0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 0, 0, 0, 0, 0;
    0, 1, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 0, 0, 0, 0;
    0, 1, 1, 2, 2, 2, 2, 1, 1, 3, 3, 2, 1, 2, 2, 2, 2, 1, 0, 0, 0, 0;
    0, 1, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 1, 4, 2, 1, 0, 0, 0, 0, 0;
    0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 4, 4, 1, 0, 0, 0, 0, 0;
    0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 1, 1, 0, 0, 0, 0, 0, 0;
    0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
];

custom_palette_2 = containers.Map('KeyType', 'double', 'ValueType', 'any');
custom_palette_2(0) = [1.0, 1.0, 1.0];      % White (background)
custom_palette_2(1) = [0.0, 0.0, 0.0];      % Black
custom_palette_2(2) = [1.0, 0.949, 0.0];    % Yellow
custom_palette_2(3) = [0.929, 0.110, 0.141]; % Red
custom_palette_2(4) = [0.459, 0.298, 0.141]; % Brown

% Draw and save
fig = draw_pixel_art(my_design, custom_palette_2, 1, 'My Pixel Art');

% Save the figure
saveas(fig, 'my_pixel_art.png');
fprintf('Saved to: my_pixel_art.png\n');

%% RGB Color Helper Function
fprintf('\nRGB Color Helper\n');
fprintf('================\n');

% Example conversions
fprintf('Red (#FF0000):    [%.3f, %.3f, %.3f]\n', hex_to_rgb('#FF0000'));
fprintf('Green (#00FF00):  [%.3f, %.3f, %.3f]\n', hex_to_rgb('#00FF00'));
fprintf('Blue (#0000FF):   [%.3f, %.3f, %.3f]\n', hex_to_rgb('#0000FF'));
fprintf('Custom (#FF5733): [%.3f, %.3f, %.3f]\n', hex_to_rgb('#FF5733'));

%% Advanced: Display Multiple Sprites
fprintf('\nAdvanced: Multiple Sprites in Grid\n');
fprintf('===================================\n');

% Create multiple sprites
sprite1 = [0, 2, 0; 2, 5, 2; 0, 2, 0];
sprite2 = [2, 0, 2; 0, 5, 0; 2, 0, 2];
sprite3 = [0, 2, 2; 2, 5, 0; 2, 2, 0];
sprite4 = [2, 2, 0; 0, 5, 2; 0, 2, 2];

sprites = {sprite1, sprite2, sprite3, sprite4};
titles = {'Sprite 1', 'Sprite 2', 'Sprite 3', 'Sprite 4'};

% Create default colormap
default_cmap = [
    0.0, 0.0, 0.0;          % 0: Black
    1.0, 1.0, 1.0;          % 1: White
    0.894, 0.267, 0.204;    % 2: Red
    0.298, 0.686, 0.314;    % 3: Green
    0.247, 0.318, 0.710;    % 4: Blue
    0.984, 0.922, 0.231     % 5: Yellow
];

% Create 2x2 grid
figure('Name', 'Multiple Sprites', 'Position', [100 100 800 800]);

for i = 1:4
    subplot(2, 2, i);
    
    % Display the sprite
    imagesc(sprites{i});
    colormap(default_cmap);
    caxis([0 5]);  % Set color axis limits
    
    % Format the plot
    axis equal tight;
    axis off;
    title(titles{i}, 'FontSize', 12, 'FontWeight', 'bold');
end

fprintf('Multiple sprites displayed successfully!\n');

%% Tips for Creating Pixel Art
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('TIPS FOR CREATING PIXEL ART\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('1. Start small: 8×8 or 16×16 matrices work great for classic sprites\n');
fprintf('2. Use 0 for background: Keep empty areas as 0 (black/transparent)\n');
fprintf('3. Limit colors: Real 8-bit games used 3-4 colors per sprite\n');
fprintf('4. Symmetry: Many classic sprites are symmetrical\n');
fprintf('5. Outline: Use a dark color (like 2) for outlines\n');
fprintf('%s\n', repmat('=', 1, 70));

%% ========================================================================
%  FUNCTION DEFINITIONS
%  ========================================================================

function fig = draw_pixel_art(matrix, color_palette, pixel_size, fig_title)
    % DRAW_PIXEL_ART Draw an 8-bit style image from a matrix of color indices
    %
    % Parameters:
    % -----------
    % matrix : 2D array
    %     Matrix where each number represents a color index
    % color_palette : containers.Map or empty [], optional
    %     Map of color indices to RGB tuples (values 0-1)
    %     If empty, uses a default retro game palette
    % pixel_size : scalar, optional
    %     Size multiplier for the output figure (default: 10)
    % fig_title : string, optional
    %     Title for the figure
    %
    % Returns:
    % --------
    % fig : figure handle
    %     Handle to the created figure
    
    % Set defaults
    if nargin < 2 || isempty(color_palette)
        % Default retro color palette (RGB values from 0-1)
        color_palette = containers.Map('KeyType', 'double', 'ValueType', 'any');
        color_palette(0) = [0.0, 0.0, 0.0];       % Black (background)
        color_palette(1) = [1.0, 1.0, 1.0];       % White
        color_palette(2) = [0.894, 0.267, 0.204]; % Red
        color_palette(3) = [0.298, 0.686, 0.314]; % Green
        color_palette(4) = [0.247, 0.318, 0.710]; % Blue
        color_palette(5) = [0.984, 0.922, 0.231]; % Yellow
        color_palette(6) = [0.961, 0.490, 0.137]; % Orange
        color_palette(7) = [0.608, 0.349, 0.714]; % Purple
        color_palette(8) = [0.404, 0.227, 0.718]; % Indigo
        color_palette(9) = [0.282, 0.820, 0.800]; % Cyan
    end
    
    if nargin < 3 || isempty(pixel_size)
        pixel_size = 10;
    end
    
    if nargin < 4
        fig_title = 'Pixel Art';
    end
    
    % Get matrix dimensions
    [height, width] = size(matrix);
    
    % Get unique values and max index
    unique_vals = unique(matrix);
    max_index = max(unique_vals);
    
    % Build color list for colormap
    colors = zeros(max_index + 1, 3);
    for i = 0:max_index
        if isKey(color_palette, i)
            colors(i + 1, :) = color_palette(i);
        else
            % Default to gray for undefined colors
            colors(i + 1, :) = [0.5, 0.5, 0.5];
        end
    end
    
    % Create figure with appropriate size
    fig_width = width * pixel_size * 10;
    fig_height = height * pixel_size * 10;
    fig = figure('Name', fig_title, 'Position', [100 100 fig_width fig_height]);
    
    % Display the pixel art
    imagesc(matrix);
    colormap(colors);
    caxis([0 max_index]);
    
    % Format the plot
    axis equal tight;
    axis off;
    title(fig_title, 'FontSize', 14, 'FontWeight', 'bold');
    
    % Make background white
    set(gcf, 'Color', 'white');
end

function rgb = hex_to_rgb(hex_color)
    % HEX_TO_RGB Convert hex color to RGB tuple (0-1 range)
    %
    % Parameters:
    % -----------
    % hex_color : string
    %     Hex color string (e.g., '#FF5733')
    %
    % Returns:
    % --------
    % rgb : 1x3 array
    %     RGB values in range [0, 1]
    %
    % Example:
    %     rgb = hex_to_rgb('#FF5733')
    %     % Returns: [1.0, 0.341, 0.2]
    
    % Remove '#' if present
    if hex_color(1) == '#'
        hex_color = hex_color(2:end);
    end
    
    % Convert hex to RGB
    r = hex2dec(hex_color(1:2)) / 255.0;
    g = hex2dec(hex_color(3:4)) / 255.0;
    b = hex2dec(hex_color(5:6)) / 255.0;
    
    rgb = [r, g, b];
end