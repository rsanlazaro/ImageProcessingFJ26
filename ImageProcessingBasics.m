%% Image Loading and Basic Properties in MATLAB
% =========================================================================
% Introduction to Digital Images in MATLAB
%
% Learning Objectives:
% - Understand how images are represented as numerical arrays
% - Learn about different image file formats and their properties
% - Explore color channels and color spaces
% - Master basic image loading, display, and analysis techniques
% =========================================================================

%% 1. Setup and Initialization
% Clear workspace and command window
clear all;
close all;
clc;

fprintf('MATLAB Image Processing Tutorial\n');
fprintf('================================\n\n');

% Check for Image Processing Toolbox
if license('test', 'Image_Toolbox')
    fprintf(' Image Processing Toolbox is available\n');
else
    warning('Image Processing Toolbox not found. Some functions may not work.');
end

fprintf('MATLAB Version: %s\n\n', version);

%% 2. Helper Functions
% These functions will be used throughout the script

% Function to print image information
function print_image_info(img, name)
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('Information for: %s\n', name);
    fprintf('%s\n', repmat('=', 1, 60));
    
    % Get image properties
    [height, width, channels] = size(img);
    
    fprintf('Image size (HxW): %d x %d pixels\n', height, width);
    fprintf('Number of channels: %d\n', channels);
    fprintf('Data type: %s\n', class(img));
    fprintf('Min pixel value: %g\n', min(img(:)));
    fprintf('Max pixel value: %g\n', max(img(:)));
    fprintf('Mean pixel value: %.2f\n', mean(img(:)));
    
    % Calculate memory size
    info = whos('img');
    fprintf('Memory size: %.2f KB\n', info.bytes / 1024);
    
    % Determine image type
    if channels == 1
        fprintf('Image type: Grayscale\n');
    elseif channels == 3
        fprintf('Image type: RGB Color\n');
    elseif channels == 4
        fprintf('Image type: RGBA (with transparency)\n');
    end
    
    fprintf('%s\n\n', repmat('=', 1, 60));
end

%% 3. Loading Images from URLs or Creating Synthetic Images
fprintf('Loading images...\n\n');

% Define image sources
image_urls = struct(...
    'jpeg_photo', 'https://picsum.photos/800/600.jpg', ...
    'png_graphic', 'https://www.python.org/static/community_logos/python-logo-master-v3-TM.png');

% Try to load images from URLs (requires internet connection)
try
    fprintf('Attempting to load JPEG photo from URL...\n');
    jpeg_photo = imread(image_urls.jpeg_photo);
    fprintf(' Successfully loaded JPEG photo\n');
catch
    fprintf('! Could not load from URL, creating synthetic photo\n');
    % Create synthetic landscape image
    height = 400;
    width = 600;
    jpeg_photo = zeros(height, width, 3, 'uint8');
    
    % Sky gradient (blue)
    for i = 1:height/2
        jpeg_photo(i, :, 3) = uint8(200 - (i / (height/2)) * 50); % Blue
        jpeg_photo(i, :, 1) = uint8(100 + (i / (height/2)) * 50); % Red
    end
    
    % Ground (green/brown)
    for i = height/2+1:height
        jpeg_photo(i, :, 2) = uint8(150 - ((i - height/2) / (height/2)) * 50); % Green
        jpeg_photo(i, :, 1) = uint8(100 + ((i - height/2) / (height/2)) * 50); % Red
    end
end

try
    fprintf('Attempting to load PNG graphic from URL...\n');
    png_graphic = imread(image_urls.png_graphic);
    fprintf(' Successfully loaded PNG graphic\n');
catch
    fprintf('! Could not load from URL, creating synthetic PNG\n');
    % Create synthetic logo-style graphic
    img_size = 400;
    png_graphic = zeros(img_size, img_size, 4, 'uint8');
    
    % Create circular logo
    [X, Y] = meshgrid(1:img_size, 1:img_size);
    center = img_size / 2;
    radius = sqrt((X - center).^2 + (Y - center).^2);
    
    % Outer circle
    mask = radius <= (img_size/2 - 20);
    png_graphic(:, :, 1) = uint8(mask * 70);   % Red
    png_graphic(:, :, 2) = uint8(mask * 130);  % Green
    png_graphic(:, :, 3) = uint8(mask * 180);  % Blue
    png_graphic(:, :, 4) = uint8(mask * 255);  % Alpha
    
    % Inner circle
    inner_mask = radius <= img_size/4;
    png_graphic(inner_mask, 1) = 255;
    png_graphic(inner_mask, 2) = 200;
    png_graphic(inner_mask, 3) = 0;
    png_graphic(inner_mask, 4) = 125;
end

% Create grayscale version
fprintf('Creating grayscale version...\n');
grayscale_img = rgb2gray(jpeg_photo);

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('Image Loading Summary:\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('%-20s %10s %15s\n', 'Image Name', 'Type', 'Size');
fprintf('%s\n', repmat('-', 1, 60));

[h, w, c] = size(jpeg_photo);
fprintf('%-20s %10s %7dx%-7d\n', 'jpeg_photo', sprintf('%dch', c), h, w);

[h, w, c] = size(png_graphic);
fprintf('%-20s %10s %7dx%-7d\n', 'png_graphic', sprintf('%dch', c), h, w);

[h, w] = size(grayscale_img);
fprintf('%-20s %10s %7dx%-7d\n', 'grayscale_img', '1ch', h, w);
fprintf('%s\n\n', repmat('=', 1, 60));

%% 4. Image Information Analysis
fprintf('Analyzing color image properties...\n');
print_image_info(jpeg_photo, 'Color Photo (JPEG)');

% Display the color image
figure('Name', 'Color Image Example', 'Position', [100 100 1000 700]);
imshow(jpeg_photo);
title('Color Image Example', 'FontSize', 16, 'FontWeight', 'bold');

%% 5. PNG Image with Transparency
fprintf('Analyzing PNG image...\n');
print_image_info(png_graphic, 'PNG Graphic');

% Display PNG image
figure('Name', 'PNG Image', 'Position', [100 100 1000 600]);
imshow(png_graphic);
title('PNG Image with Transparency', 'FontSize', 14, 'FontWeight', 'bold');

% Check for alpha channel
[~, ~, channels] = size(png_graphic);
if channels == 4
    fprintf('\n This PNG has an alpha channel (transparency support)\n');
    alpha_channel = png_graphic(:, :, 4);
    fprintf('Alpha channel statistics:\n');
    fprintf('  Min: %d (fully transparent)\n', min(alpha_channel(:)));
    fprintf('  Max: %d (fully opaque)\n', max(alpha_channel(:)));
    fprintf('  Mean: %.1f\n', mean(alpha_channel(:)));
elseif channels == 3
    fprintf('\nThis PNG doesn''t have transparency (RGB mode)\n');
end

%% 6. Array Dimensions Analysis
color_array = jpeg_photo;
gray_array = grayscale_img;

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('Array Dimensions Analysis\n');
fprintf('%s\n', repmat('=', 1, 60));

fprintf('\nCOLOR IMAGE (RGB):\n');
fprintf('%s\n', repmat('-', 1, 60));
[height, width, channels] = size(color_array);
fprintf('Shape: [%d, %d, %d]\n', height, width, channels);
fprintf('  Dimension 1 (height): %d pixels\n', height);
fprintf('  Dimension 2 (width):  %d pixels\n', width);
fprintf('  Dimension 3 (channels): %d (R, G, B)\n', channels);
fprintf('Total pixels: %s\n', num2str(height * width));
fprintf('Total values: %s\n', num2str(numel(color_array)));

fprintf('\n%s\n', repmat('-', 1, 60));
fprintf('GRAYSCALE IMAGE:\n');
fprintf('%s\n', repmat('-', 1, 60));
[height, width] = size(gray_array);
fprintf('Shape: [%d, %d]\n', height, width);
fprintf('  Dimension 1 (height): %d pixels\n', height);
fprintf('  Dimension 2 (width):  %d pixels\n', width);
fprintf('  Only 1 intensity value per pixel (no separate channels)\n');
fprintf('Total pixels: %s\n', num2str(numel(gray_array)));

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('MEMORY COMPARISON:\n');
fprintf('%s\n', repmat('=', 1, 60));
color_info = whos('color_array');
gray_info = whos('gray_array');
fprintf('Color image:     %.2f KB\n', color_info.bytes / 1024);
fprintf('Grayscale image: %.2f KB\n', gray_info.bytes / 1024);
fprintf('Ratio: %.1fx larger\n', color_info.bytes / gray_info.bytes);

%% 7. Pixel Values Inspection
% Extract a small 10x10 patch from top-left corner
patch_size = 10;
patch = color_array(1:patch_size, 1:patch_size, :);

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('10×10 PIXEL PATCH (Top-Left Corner)\n');
fprintf('%s\n', repmat('=', 1, 60));

fprintf('\nRED CHANNEL VALUES:\n');
disp(patch(:, :, 1));

fprintf('\nGREEN CHANNEL VALUES:\n');
disp(patch(:, :, 2));

fprintf('\nBLUE CHANNEL VALUES:\n');
disp(patch(:, :, 3));

% Show one pixel in detail
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('SINGLE PIXEL EXAMPLE:\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('Pixel at position (5, 5):\n');
fprintf('  Red:   %3d (out of 255)\n', patch(5, 5, 1));
fprintf('  Green: %3d (out of 255)\n', patch(5, 5, 2));
fprintf('  Blue:  %3d (out of 255)\n', patch(5, 5, 3));
fprintf('  RGB: (%d, %d, %d)\n', patch(5, 5, 1), patch(5, 5, 2), patch(5, 5, 3));

% Visualize the patch
figure('Name', 'Pixel Patch', 'Position', [100 100 1200 500]);

subplot(1, 2, 1);
imshow(patch);
title('10×10 Pixel Patch (Magnified)', 'FontSize', 14, 'FontWeight', 'bold');

subplot(1, 2, 2);
imshow(patch, 'InitialMagnification', 'fit');
title('Same Patch with Pixel Grid', 'FontSize', 14, 'FontWeight', 'bold');
hold on;
% Draw grid
for i = 0.5:1:patch_size+0.5
    plot([0.5, patch_size+0.5], [i, i], 'w', 'LineWidth', 1.5);
    plot([i, i], [0.5, patch_size+0.5], 'w', 'LineWidth', 1.5);
end
hold off;

%% 8. RGB Channel Decomposition
% Extract individual color channels
red_channel = color_array(:, :, 1);
green_channel = color_array(:, :, 2);
blue_channel = color_array(:, :, 3);

% Create figure with subplots
figure('Name', 'RGB Channel Decomposition', 'Position', [50 50 1600 800]);

% Row 1: Original and channels as grayscale
subplot(2, 4, 1);
imshow(color_array);
title('Original RGB Image', 'FontSize', 12, 'FontWeight', 'bold');

subplot(2, 4, 2);
imshow(red_channel);
title({'Red Channel', '(Grayscale View)'}, 'FontSize', 11);

subplot(2, 4, 3);
imshow(green_channel);
title({'Green Channel', '(Grayscale View)'}, 'FontSize', 11);

subplot(2, 4, 4);
imshow(blue_channel);
title({'Blue Channel', '(Grayscale View)'}, 'FontSize', 11);

% Row 2: Channels in their respective colors
subplot(2, 4, 5);
axis off;

% Red channel only
red_colored = cat(3, red_channel, zeros(size(red_channel), 'like', red_channel), ...
                     zeros(size(red_channel), 'like', red_channel));
subplot(2, 4, 6);
imshow(red_colored);
title({'Red Channel Only', '(Colored View)'}, 'FontSize', 11);

% Green channel only
green_colored = cat(3, zeros(size(green_channel), 'like', green_channel), ...
                       green_channel, zeros(size(green_channel), 'like', green_channel));
subplot(2, 4, 7);
imshow(green_colored);
title({'Green Channel Only', '(Colored View)'}, 'FontSize', 11);

% Blue channel only
blue_colored = cat(3, zeros(size(blue_channel), 'like', blue_channel), ...
                      zeros(size(blue_channel), 'like', blue_channel), blue_channel);
subplot(2, 4, 8);
imshow(blue_colored);
title({'Blue Channel Only', '(Colored View)'}, 'FontSize', 11);

sgtitle('RGB Channel Decomposition', 'FontSize', 16, 'FontWeight', 'bold');

% Channel statistics
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('Channel Statistics:\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('%-10s %-10s %-10s %-6s %-6s\n', 'Channel', 'Mean', 'Std Dev', 'Min', 'Max');
fprintf('%s\n', repmat('-', 1, 60));
fprintf('%-10s %10.2f %10.2f %6d %6d\n', 'Red', mean(red_channel(:)), ...
        std(double(red_channel(:))), min(red_channel(:)), max(red_channel(:)));
fprintf('%-10s %10.2f %10.2f %6d %6d\n', 'Green', mean(green_channel(:)), ...
        std(double(green_channel(:))), min(green_channel(:)), max(green_channel(:)));
fprintf('%-10s %10.2f %10.2f %6d %6d\n', 'Blue', mean(blue_channel(:)), ...
        std(double(blue_channel(:))), min(blue_channel(:)), max(blue_channel(:)));
fprintf('%s\n', repmat('=', 1, 60));

%% 9. Grayscale Conversion Methods
% Method 1: Using MATLAB's built-in function (uses standard luminosity formula)
gray_builtin = rgb2gray(color_array);

% Method 2: Manual implementation of luminosity method
% Formula: Gray = 0.2989*R + 0.5870*G + 0.1140*B (MATLAB's weights)
gray_manual = 0.2989 * double(red_channel) + ...
              0.5870 * double(green_channel) + ...
              0.1140 * double(blue_channel);
gray_manual = uint8(gray_manual);

% Method 3: Simple average
gray_average = uint8((double(red_channel) + double(green_channel) + ...
                      double(blue_channel)) / 3);

% Display comparison
figure('Name', 'Grayscale Conversion Methods', 'Position', [100 100 1500 500]);

subplot(1, 3, 1);
imshow(color_array);
title('Original RGB', 'FontSize', 14, 'FontWeight', 'bold');

subplot(1, 3, 2);
imshow(gray_builtin);
title({'Luminosity Method (rgb2gray)', '0.2989R + 0.5870G + 0.1140B', ...
       '(Perceptually Accurate)'}, 'FontSize', 12);

subplot(1, 3, 3);
imshow(gray_average);
title({'Average Method', '(R + G + B) / 3', '(Simpler, Less Accurate)'}, ...
      'FontSize', 12);

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('Grayscale Conversion Results:\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('Luminosity method (rgb2gray):\n');
fprintf('  Shape: [%d, %d]\n', size(gray_builtin, 1), size(gray_builtin, 2));
fprintf('  dtype: %s\n', class(gray_builtin));
fprintf('  Range: [%.1f, %.1f]\n', double(min(gray_builtin(:))), double(max(gray_builtin(:))));
fprintf('\nAverage method:\n');
fprintf('  Shape: [%d, %d]\n', size(gray_average, 1), size(gray_average, 2));
fprintf('  dtype: %s\n', class(gray_average));
fprintf('  Range: [%.1f, %.1f]\n', double(min(gray_average(:))), double(max(gray_average(:))));
fprintf('\nAverage absolute difference: %.2f per pixel\n', ...
        mean(abs(double(gray_builtin(:)) - double(gray_average(:)))));
fprintf('%s\n', repmat('=', 1, 60));

%% 10. Data Type Conversions
% Original image (uint8)
img_uint8 = color_array;

% Convert to double (0.0 to 1.0 range)
img_double = im2double(img_uint8);

% Convert to single precision
img_single = im2single(img_uint8);

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('DATA TYPE COMPARISON\n');
fprintf('%s\n', repmat('=', 1, 70));

fprintf('\nuint8 (Standard Display Format):\n');
fprintf('  Data type:     %s\n', class(img_uint8));
fprintf('  Value range:   %d to %d\n', min(img_uint8(:)), max(img_uint8(:)));
info_uint8 = whos('img_uint8');
fprintf('  Memory size:   %.2f KB\n', info_uint8.bytes / 1024);
fprintf('  Bits per pixel: 8\n');
fprintf('  Total values:  256 possible values per channel\n');

fprintf('\nsingle (Processing Format):\n');
fprintf('  Data type:     %s\n', class(img_single));
fprintf('  Value range:   %.4f to %.4f\n', min(img_single(:)), max(img_single(:)));
info_single = whos('img_single');
fprintf('  Memory size:   %.2f KB\n', info_single.bytes / 1024);
fprintf('  Bits per pixel: 32\n');
fprintf('  Precision:     ~7 decimal digits\n');

fprintf('\ndouble (High Precision):\n');
fprintf('  Data type:     %s\n', class(img_double));
fprintf('  Value range:   %.4f to %.4f\n', min(img_double(:)), max(img_double(:)));
info_double = whos('img_double');
fprintf('  Memory size:   %.2f KB\n', info_double.bytes / 1024);
fprintf('  Bits per pixel: 64\n');
fprintf('  Precision:     ~15 decimal digits\n');

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('MEMORY IMPACT ANALYSIS:\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('  uint8:   %.2f KB  (baseline)\n', info_uint8.bytes / 1024);
fprintf('  single:  %.2f KB  (%.1fx larger)\n', info_single.bytes / 1024, ...
        info_single.bytes / info_uint8.bytes);
fprintf('  double:  %.2f KB  (%.1fx larger)\n', info_double.bytes / 1024, ...
        info_double.bytes / info_uint8.bytes);
fprintf('\nTip: Use uint8 for display, double/single for processing!\n');
fprintf('%s\n', repmat('=', 1, 70));

%% 11. Accessing Individual Pixels
[height, width, channels] = size(color_array);

% Access pixel at center
center_row = floor(height / 2);
center_col = floor(width / 2);
pixel_value = squeeze(color_array(center_row, center_col, :));

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('IMAGE DIMENSIONS\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('Height:   %d pixels\n', height);
fprintf('Width:    %d pixels\n', width);
fprintf('Channels: %d\n', channels);
fprintf('Total pixels: %s\n', num2str(height * width));

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('CENTER PIXEL at position (%d, %d)\n', center_row, center_col);
fprintf('%s\n', repmat('=', 1, 60));
fprintf('  Red:   %3d / 255\n', pixel_value(1));
fprintf('  Green: %3d / 255\n', pixel_value(2));
fprintf('  Blue:  %3d / 255\n', pixel_value(3));
fprintf('  RGB: (%d, %d, %d)\n', pixel_value(1), pixel_value(2), pixel_value(3));

% Access corner pixels
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('CORNER PIXELS\n');
fprintf('%s\n', repmat('=', 1, 60));

corners = struct();
corners.top_left = squeeze(color_array(1, 1, :));
corners.top_right = squeeze(color_array(1, width, :));
corners.bottom_left = squeeze(color_array(height, 1, :));
corners.bottom_right = squeeze(color_array(height, width, :));

corner_names = {'top_left', 'top_right', 'bottom_left', 'bottom_right'};
corner_labels = {'Top-left', 'Top-right', 'Bottom-left', 'Bottom-right'};

for i = 1:length(corner_names)
    rgb = corners.(corner_names{i});
    fprintf('%-15s RGB(%3d, %3d, %3d)\n', corner_labels{i}, rgb(1), rgb(2), rgb(3));
end
fprintf('%s\n', repmat('=', 1, 60));

%% 12. Region of Interest (ROI) Extraction
% Extract rectangular region from center
crop_size = min([100, floor(height/2), floor(width/2)]);
start_row = center_row - floor(crop_size/2);
end_row = center_row + floor(crop_size/2) - 1;
start_col = center_col - floor(crop_size/2);
end_col = center_col + floor(crop_size/2) - 1;

cropped_region = color_array(start_row:end_row, start_col:end_col, :);

fprintf('\nRegion Extraction Example:\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('Original image shape:  [%d, %d, %d]\n', size(color_array));
fprintf('Cropped region shape:  [%d, %d, %d]\n', size(cropped_region));
fprintf('\nExtraction coordinates:\n');
fprintf('  Rows: %d to %d\n', start_row, end_row);
fprintf('  Cols: %d to %d\n', start_col, end_col);
fprintf('%s\n', repmat('=', 1, 60));

% Visualize
figure('Name', 'ROI Extraction', 'Position', [100 100 1400 600]);

subplot(1, 2, 1);
imshow(color_array);
hold on;
rectangle('Position', [start_col, start_row, crop_size, crop_size], ...
          'EdgeColor', 'r', 'LineWidth', 3);
hold off;
title({'Original Image', 'ROI marked in red'}, 'FontSize', 14, 'FontWeight', 'bold');

subplot(1, 2, 2);
imshow(cropped_region);
title(sprintf('Extracted Region\n%d×%d pixels', crop_size, crop_size), ...
      'FontSize', 14, 'FontWeight', 'bold');

%% 13. Histograms
% Calculate and display histograms
figure('Name', 'Image Histograms', 'Position', [50 50 1000 1000]);

% Original image
subplot(3, 2, 1);
imshow(color_array);
title('Original Image', 'FontSize', 14, 'FontWeight', 'bold');

% RGB histogram combined
subplot(3, 2, 2);
hold on;
[counts_r, ~] = imhist(red_channel);
[counts_g, ~] = imhist(green_channel);
[counts_b, ~] = imhist(blue_channel);
plot(0:255, counts_r, 'r', 'LineWidth', 2);
plot(0:255, counts_g, 'g', 'LineWidth', 2);
plot(0:255, counts_b, 'b', 'LineWidth', 2);
hold off;
title('RGB Histogram (All Channels)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Pixel Intensity', 'FontSize', 11);
ylabel('Frequency (# of pixels)', 'FontSize', 11);
legend('Red', 'Green', 'Blue', 'FontSize', 10);
grid on;
xlim([0 255]);

% Individual channel histograms
subplot(3, 2, 3);
bar(0:255, counts_r, 'r', 'EdgeColor', 'none');
title('Red Channel Histogram', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Pixel Intensity', 'FontSize', 10);
ylabel('Frequency', 'FontSize', 10);
grid on;
xlim([0 255]);

subplot(3, 2, 4);
bar(0:255, counts_g, 'g', 'EdgeColor', 'none');
title('Green Channel Histogram', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Pixel Intensity', 'FontSize', 10);
ylabel('Frequency', 'FontSize', 10);
grid on;
xlim([0 255]);

subplot(3, 2, 5);
bar(0:255, counts_b, 'b', 'EdgeColor', 'none');
title('Blue Channel Histogram', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Pixel Intensity', 'FontSize', 10);
ylabel('Frequency', 'FontSize', 10);
grid on;
xlim([0 255]);

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('CHANNEL STATISTICS\n');
fprintf('%s\n', repmat('=', 1, 60));
fprintf('%-10s %-10s %-12s %-6s %-6s\n', 'Channel', 'Mean', 'Std Dev', 'Min', 'Max');
fprintf('%s\n', repmat('-', 1, 60));
fprintf('%-10s %8.2f   %10.2f   %4d   %4d\n', 'Red', ...
        mean(red_channel(:)), std(double(red_channel(:))), ...
        min(red_channel(:)), max(red_channel(:)));
fprintf('%-10s %8.2f   %10.2f   %4d   %4d\n', 'Green', ...
        mean(green_channel(:)), std(double(green_channel(:))), ...
        min(green_channel(:)), max(green_channel(:)));
fprintf('%-10s %8.2f   %10.2f   %4d   %4d\n', 'Blue', ...
        mean(blue_channel(:)), std(double(blue_channel(:))), ...
        min(blue_channel(:)), max(blue_channel(:)));
fprintf('%s\n', repmat('=', 1, 60));

%% 14. Creating Synthetic Images
fprintf('\nCreating synthetic images...\n');

img_size = 200;

% Function to create solid color
create_solid_color = @(h, w, r, g, b) cat(3, ...
    ones(h, w, 'uint8') * r, ...
    ones(h, w, 'uint8') * g, ...
    ones(h, w, 'uint8') * b);

% Create various synthetic images
red_img = create_solid_color(img_size, img_size, 255, 0, 0);
green_img = create_solid_color(img_size, img_size, 0, 255, 0);
blue_img = create_solid_color(img_size, img_size, 0, 0, 255);
yellow_img = create_solid_color(img_size, img_size, 255, 255, 0);

% Horizontal gradient
[X, Y] = meshgrid(1:img_size, 1:img_size);
gradient_h = uint8(255 * (X - 1) / (img_size - 1));
gradient_h = repmat(gradient_h, [1, 1, 3]);

% Vertical gradient
gradient_v = uint8(255 * (Y - 1) / (img_size - 1));
gradient_v = repmat(gradient_v, [1, 1, 3]);

% Checkerboard
square_size = 25;
checkerboard_img = zeros(img_size, img_size, 'uint8');
for i = 1:square_size:img_size
    for j = 1:square_size:img_size
        if mod(floor((i-1)/square_size) + floor((j-1)/square_size), 2) == 0
            i_end = min(i + square_size - 1, img_size);
            j_end = min(j + square_size - 1, img_size);
            checkerboard_img(i:i_end, j:j_end) = 255;
        end
    end
end

% Radial gradient
center = img_size / 2;
[X, Y] = meshgrid(1:img_size, 1:img_size);
radius = sqrt((X - center).^2 + (Y - center).^2);
radial = uint8(255 * (1 - min(radius / (img_size/2), 1)));

% Display all synthetic images
figure('Name', 'Synthetic Images', 'Position', [50 50 1600 800]);

images_to_show = {
    red_img, {'Pure Red', 'RGB(255, 0, 0)'};
    green_img, {'Pure Green', 'RGB(0, 255, 0)'};
    blue_img, {'Pure Blue', 'RGB(0, 0, 255)'};
    yellow_img, {'Yellow', 'RGB(255, 255, 0)'};
    gradient_h, {'Horizontal Gradient', 'Black → White'};
    gradient_v, {'Vertical Gradient', 'Black → White'};
    checkerboard_img, {'Checkerboard', '25×25 squares'};
    radial, {'Radial Gradient', 'Center → Edge'}
};

for i = 1:8
    subplot(2, 4, i);
    img_data = images_to_show{i, 1};
    img_title = images_to_show{i, 2};
    
    if size(img_data, 3) == 1
        imshow(img_data);
    else
        imshow(img_data);
    end
    title(img_title, 'FontSize', 11, 'FontWeight', 'bold');
end

sgtitle('Synthetic Images Created from MATLAB Arrays', ...
        'FontSize', 16, 'FontWeight', 'bold');

fprintf('\nKey Insight: All these images are just MATLAB arrays with different\n');
fprintf('   number patterns! Change the numbers, change the image.\n');

%% 15. Summary and Key Takeaways
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('SUMMARY AND KEY TAKEAWAYS\n');
fprintf('%s\n', repmat('=', 1, 70));

fprintf('\n1. IMAGE REPRESENTATION:\n');
fprintf('   - Images are multi-dimensional arrays of numbers\n');
fprintf('   - Grayscale: 2D array (height × width)\n');
fprintf('   - Color: 3D array (height × width × 3 channels)\n');
fprintf('   - Each pixel is just a number or set of numbers!\n');

fprintf('\n2. FILE FORMATS:\n');
fprintf('   - JPEG: Lossy compression, good for photos, no transparency\n');
fprintf('   - PNG: Lossless compression, supports transparency, good for graphics\n');
fprintf('   - Different formats = different tradeoffs in quality, size, features\n');

fprintf('\n3. COLOR CHANNELS:\n');
fprintf('   - RGB model: Red + Green + Blue\n');
fprintf('   - Each channel: 0-255 for uint8 images\n');
fprintf('   - Any color = combination of RGB values\n');
fprintf('   - Channels can be separated and manipulated independently\n');

fprintf('\n4. DATA TYPES:\n');
fprintf('   - uint8 (0-255): Standard for display, compact\n');
fprintf('   - double/single (0.0-1.0): Better for processing\n');
fprintf('   - Memory tradeoffs: double uses 8× more memory than uint8\n');

fprintf('\n5. BASIC OPERATIONS:\n');
fprintf('   - Accessing pixels: image(row, col) or image(row, col, channel)\n');
fprintf('   - Slicing regions: image(r1:r2, c1:c2, :)\n');
fprintf('   - Histograms: Show pixel intensity distribution (imhist)\n');
fprintf('   - Grayscale conversion: rgb2gray (weighted sum)\n');

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('MATLAB-SPECIFIC NOTES:\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('   - Arrays are 1-indexed (first element is index 1, not 0)\n');
fprintf('   - Use imread() to load images\n');
fprintf('   - Use imshow() to display images\n');
fprintf('   - Use imwrite() to save images\n');
fprintf('   - Image Processing Toolbox provides many useful functions\n');
fprintf('   - Use im2double/im2single for type conversion\n');
fprintf('   - Use cat() to concatenate arrays along dimensions\n');

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('NEXT STEPS:\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('   1. Practice loading and analyzing your own images\n');
fprintf('   2. Experiment with channel manipulation\n');
fprintf('   3. Create more complex synthetic images\n');
fprintf('   4. Learn about image transformations and filters\n');
fprintf('   5. Explore Image Processing Toolbox functions\n');

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('Tutorial complete!\n');
fprintf('%s\n', repmat('=', 1, 70));
