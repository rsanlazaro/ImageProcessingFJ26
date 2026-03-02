% =========================================================================
%  Arithmetic Operations in Images
%  Converted from: image_arithmetic_python.ipynb  (Python / NumPy)
%
%  Sections:
%   1. Addition      – Noise reduction by averaging k noisy images
%   2. Subtraction   – Difference between original and LSB-zeroed image
%   3. Multiplication – Applying shading to a checkerboard
%   4. Division      – Shading correction and binary masking
%
%  Requirements: MATLAB R2019b+ (uses sgtitle)
% =========================================================================

clc; clear; close all;
rng(42);         % equivalent to np.random.seed(42)
SIZE = 256;      % image size in pixels


% =========================================================================
%  1 · ADDITION — Noise Reduction by Averaging k Noisy Images
% =========================================================================
%
%  Theory
%    g_i(x,y) = f(x,y) + n_i(x,y),   n_i ~ N(0, sigma^2)
%    g_hat    = (1/k) * sum_i g_i  -->  f  as k -> inf
%    Noise std-dev in average: sigma_avg = sigma / sqrt(k)
%
%  k  | sigma_avg
%  ---|----------
%  1  | 40.0
%  4  | 20.0
%  16 | 10.0
%  64 |  5.0
% =========================================================================

% --- Parameters (Cell 3) ---
k     = 16;   % number of noisy images to average
sigma = 40;   % noise standard deviation (0-255 pixel scale)

% --- Clean image: smooth sinusoidal gradient (Cell 3) ---
x = linspace(0, 1, SIZE);
[X, Y] = meshgrid(x, x);
clean = sin(pi*X) .* sin(pi*Y) .* 200 + 28;   % float64, range [28, 228]

% --- Generate k noisy images and accumulate (Cell 3) ---
accumulator  = zeros(SIZE, SIZE);
noisy_sample = [];

for i = 1:k
    noisy_i     = clean + sigma * randn(SIZE, SIZE);
    accumulator = accumulator + noisy_i;
    if i == 1
        noisy_sample = noisy_i;   % keep first noisy image for display
    end
end

averaged = accumulator / k;

fprintf('Noise sigma in one image   : %.1f\n', sigma);
fprintf('Noise sigma after averaging: %.2f  (theoretical = sigma/sqrt(k))\n', sigma/sqrt(k));
fprintf('Measured sigma of residual : %.2f\n\n', std(averaged(:) - clean(:)));

% --- Display: show_row equivalent (Cell 4) ---
figure('Name','1 - Addition: Noise Reduction', ...
       'Units','normalized','Position',[0.05 0.55 0.9 0.40]);

subplot(1,3,1);
imshow(uint8(clean), [0 255]);
title('Clean image $f(x,y)$', 'Interpreter','latex', 'FontSize',11);

subplot(1,3,2);
imshow(uint8(max(0, min(255, noisy_sample))), [0 255]);
title(sprintf('One noisy image $g_1$\n($\\sigma$=%d)', sigma), ...
      'Interpreter','latex', 'FontSize',11);

subplot(1,3,3);
imshow(uint8(max(0, min(255, averaged))), [0 255]);
title(sprintf('Average of $k$=%d images\n($\\sigma_{\\mathrm{noise}}\\approx$%.1f)', ...
              k, sigma/sqrt(k)), 'Interpreter','latex', 'FontSize',11);

sgtitle('1 · Addition — Noise Reduction by Averaging', ...
        'FontWeight','bold', 'FontSize',13);


% =========================================================================
%  2 · SUBTRACTION — LSB Zeroing
% =========================================================================
%
%  Theory
%    f'(x,y) = bitand(f(x,y), 0xFE)   (0xFE = 11111110 binary = 254)
%    d(x,y)  = f(x,y) - f'(x,y)       d in {0, 1}
%    Scaled: d * 127  makes the binary map visible
%
%  Applications: steganography, quantisation analysis
% =========================================================================

% --- Original image: ramp + sinusoidal texture (Cell 6) ---
ramp = repmat(linspace(0, 255, SIZE), SIZE, 1);   % horizontal ramp
[Xs, Ys] = meshgrid(linspace(0, 4*pi, SIZE), linspace(0, 4*pi, SIZE));
wave     = sin(Xs) .* cos(Ys) .* 60;
original = uint8(max(0, min(255, ramp + wave)));

% --- Zero the LSB via bitwise AND with 0xFE (= 254) (Cell 6) ---
lsb_zeroed = bitand(original, uint8(254));   % 0xFE = 11111110 binary

% --- Compute difference (int16 to avoid unsigned overflow) (Cell 6) ---
diff_img    = int16(original) - int16(lsb_zeroed);
diff_scaled = uint8(abs(diff_img) * 127);   % {0,1} --> {0,127}

lsb_fraction = mean(diff_img(:) > 0) * 100;
fprintf('Pixels with LSB = 1 : %.1f%%  (expect ~50%% for natural images)\n\n', ...
        lsb_fraction);

% --- Display three panels (Cell 7) ---
figure('Name','2 - Subtraction: LSB Zeroing', ...
       'Units','normalized','Position',[0.05 0.55 0.9 0.40]);

subplot(1,3,1);
imshow(original, [0 255]);
title('Original image $f(x,y)$', 'Interpreter','latex', 'FontSize',11);

subplot(1,3,2);
imshow(lsb_zeroed, [0 255]);
title("LSB set to 0  $f'(x,y)$" + newline + "(AND 0xFE)", ...
      'Interpreter','latex', 'FontSize',11);

subplot(1,3,3);
imshow(diff_scaled, [0 255]);
title('Difference $d = f - f''$' + newline + '(scaled $\times$127)', ...
      'Interpreter','latex', 'FontSize',11);

sgtitle('2 · Subtraction — LSB Zeroing', 'FontWeight','bold', 'FontSize',13);

% --- Bit-plane analysis: all 8 planes (Cell 7) ---
figure('Name','Bit Planes', 'Units','normalized','Position',[0.05 0.05 0.9 0.25]);

for bit = 7:-1:0
    plane = bitand(bitshift(original, -bit), uint8(1)) * 255;
    subplot(1, 8, 8 - bit);
    imshow(plane);
    if     bit == 7,  lbl = 'MSB';
    elseif bit == 0,  lbl = 'LSB';
    else,             lbl = '';
    end
    title(sprintf('Bit %d\n%s', bit, lbl), 'FontSize', 8);
end

sgtitle('All 8 bit planes (MSB \rightarrow LSB)', ...
        'FontWeight','bold', 'FontSize',11);


% =========================================================================
%  3 · MULTIPLICATION — Shading Applied to a Checkerboard
% =========================================================================
%
%  Theory
%    g(x,y) = f(x,y) * s(x,y)
%
%    Shading field:
%      s(x,y) = 1 - 0.65 * [(x-cx)^2 + (y-cy)^2] / [cx^2 + cy^2]
%    clamped to [0.1, 1.0]
%
%  Simulates vignetting / non-uniform illumination (camera / microscope)
% =========================================================================

% --- Ideal checkerboard, float [0,1] (Cell 9) ---
tile = 32;
[col, row]   = meshgrid(0:SIZE-1, 0:SIZE-1);
checker_mask = mod(floor(row/tile) + floor(col/tile), 2);
checkerboard = double(checker_mask == 0) * (220/255) + ...
               double(checker_mask == 1) * (50/255);   % light / dark squares

% --- Radial shading field (Cell 9) ---
cx = SIZE/2;  cy = SIZE/2;
[Xg, Yg]    = meshgrid(0:SIZE-1, 0:SIZE-1);   % 0-based to match Python arange
shading_field = 1.0 - 0.65 * ((Xg - cx).^2 + (Yg - cy).^2) / (cx^2 + cy^2);
shading_field = max(0.1, min(1.0, shading_field));

% --- Apply shading via MULTIPLICATION (Cell 9) ---
shaded = checkerboard .* shading_field;

fprintf('Checkerboard pixel range : [%.3f, %.3f]\n', ...
        min(checkerboard(:)), max(checkerboard(:)));
fprintf('Shading field range      : [%.3f, %.3f]\n', ...
        min(shading_field(:)), max(shading_field(:)));
fprintf('Shaded image range       : [%.3f, %.3f]\n\n', ...
        min(shaded(:)), max(shaded(:)));

% --- Display three panels (Cell 10) ---
figure('Name','3 - Multiplication: Shading', ...
       'Units','normalized','Position',[0.05 0.55 0.9 0.40]);

subplot(1,3,1);
imshow(checkerboard, [0 1]);
title('Ideal checkerboard $f(x,y)$', 'Interpreter','latex', 'FontSize',11);

subplot(1,3,2);
imshow(shading_field, [0 1]);
title('Shading field $s(x,y)$', 'Interpreter','latex', 'FontSize',11);

subplot(1,3,3);
imshow(shaded, [0 1]);
title('Shaded image $g = f \cdot s$' + newline + '(multiplication)', ...
      'Interpreter','latex', 'FontSize',11);

sgtitle('3 · Multiplication — Shading Applied to Checkerboard', ...
        'FontWeight','bold', 'FontSize',13);

% --- Horizontal intensity profile at mid-row (Cell 10) ---
mid = round(SIZE/2);

figure('Name','Intensity Profile','Units','normalized','Position',[0.1 0.1 0.7 0.35]);
hold on;
plot(checkerboard(mid,:),  'LineWidth',1.5, 'Color','steelblue', ...
     'DisplayName','Ideal checkerboard');
plot(shading_field(mid,:), 'LineWidth',1.5, 'Color',[1 0.647 0], 'LineStyle','--', ...
     'DisplayName','Shading field');
plot(shaded(mid,:),        'LineWidth',1.5, 'Color','crimson', ...
     'DisplayName','Shaded image');
xlabel('Column pixel index', 'FontSize',11);
ylabel('Normalised intensity', 'FontSize',11);
title('Horizontal intensity profile at mid-row', 'FontSize',11);
legend('FontSize',10, 'Location','south');
grid on;  alpha(0.35);
ylim([-0.05, 1.05]);


% =========================================================================
%  4 · DIVISION — Shading Correction & Masking
% =========================================================================
%
%  Theory
%    (a) Shading correction (flat-field correction):
%        f_hat(x,y) = g(x,y) / s(x,y)
%        Dividing by the known shading field recovers the true image.
%        Used in: microscopy, astronomy, document scanning.
%
%    (b) Binary masking via division:
%        masked(x,y) = g(x,y) / M(x,y),  M in {0,1}
%        Isolates region of interest; zeros everything outside the mask.
% =========================================================================

% --- (a) Shading correction via DIVISION (Cell 12) ---
corrected        = zeros(SIZE, SIZE);
valid            = shading_field > 0;
corrected(valid) = shaded(valid) ./ shading_field(valid);
corrected        = max(0, min(1, corrected));

residual_error = mean(abs(corrected(:) - checkerboard(:)));
fprintf('Mean absolute error after correction: %.6f  (should be ~0)\n', ...
        residual_error);

% --- (b) Circular binary mask (Cell 12) ---
r_mask = SIZE / 3;
mask   = double((Xg - cx).^2 + (Yg - cy).^2 <= r_mask^2);   % 0 or 1
masked = zeros(SIZE, SIZE);
inside = mask > 0;
masked(inside) = shaded(inside) ./ mask(inside);
masked = max(0, min(1, masked));

fprintf('Mask covers %.1f%% of image pixels\n\n', mean(mask(:))*100);

% --- Display three panels (Cell 13) ---
figure('Name','4 - Division: Correction & Masking', ...
       'Units','normalized','Position',[0.05 0.55 0.9 0.40]);

subplot(1,3,1);
imshow(shaded, [0 1]);
title('Shaded input $g(x,y)$', 'Interpreter','latex', 'FontSize',11);

subplot(1,3,2);
imshow(corrected, [0 1]);
title('Corrected $\hat{f} = g / s$' + newline + '(division by shading field)', ...
      'Interpreter','latex', 'FontSize',11);

subplot(1,3,3);
imshow(masked, [0 1]);
title('Masked region' + newline + '(division by binary mask $M$)', ...
      'Interpreter','latex', 'FontSize',11);

sgtitle('4 · Division — Shading Correction & Masking', ...
        'FontWeight','bold', 'FontSize',13);


% =========================================================================
%  SUMMARY
% =========================================================================
fprintf('======================================================\n');
fprintf('  SUMMARY OF ARITHMETIC OPERATIONS IN IMAGES\n');
fprintf('======================================================\n');
fprintf('  Addition       g_hat = (1/k)*sum_i g_i   Noise reduction\n');
fprintf('  Subtraction    d = f - bitand(f, 0xFE)   LSB map / bit-planes\n');
fprintf('  Multiplication g = f .* s                Simulate shading\n');
fprintf('  Division       f_hat = g ./ s            Shading correction\n');
fprintf('======================================================\n');
fprintf('  Key insight: Division reverses Multiplication.\n');
fprintf('  If g = f*s (shading), then g/s = f (restored).\n');
fprintf('======================================================\n');
