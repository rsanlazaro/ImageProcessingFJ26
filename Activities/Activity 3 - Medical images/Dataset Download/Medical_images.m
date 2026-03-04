%% ============================================================
%  Medical Image Processing — Real Kaggle Datasets
%  Arithmetic · Intensity · Geometric Transformations
%  No Predefined Image Processing Functions
% =============================================================
%
%  Datasets (download manually with the Kaggle CLI before running):
%    kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset --unzip -p data/brain_tumor_mri
%    kaggle datasets download nih-chest-xrays/data --file images/images_001.zip -p data/nih_chest_xray
%    kaggle datasets download kmader/siim-medical-images --unzip -p data/siim_ct
%
%  Requirements: MATLAB R2019b+  |  Image Processing Toolbox (only for dicomread)
%  All arithmetic/intensity/geometric transforms are hand-coded — no imresize,
%  imrotate, imadd, etc. are used for the core operations.
% =============================================================

clear; clc; close all;

%% ============================================================
%  SECTION 1 — Kaggle Setup & Image Loading
% =============================================================

% ── 1.1  Paths (edit these to match where you downloaded the data) ──────────
DATA_ROOT   = fullfile(getenv('USERPROFILE'), 'kaggle_medical');   % Windows
% DATA_ROOT = fullfile(getenv('HOME'), 'kaggle_medical');           % Mac/Linux

MRI_DIR     = fullfile(DATA_ROOT, 'brain_tumor_mri');
XRAY_DIR    = fullfile(DATA_ROOT, 'nih_chest_xray');
CT_DIR      = fullfile(DATA_ROOT, 'siim_ct');

IMAGES_PER_DATASET = 50;   % how many images to work with per modality
IMG_SIZE           = 256;  % all images resized to this square

% ── 1.2  Generic helpers ──────────────────────────────────────────────────────

function img = load_image_file(path, img_size)
    % Load JPEG/PNG as float64 greyscale in [0,1], resized to img_size×img_size.
    raw = imread(path);
    if size(raw, 3) == 3
        raw = rgb2gray(raw);   % convert RGB → grey (only stdlib call allowed here)
    end
    % Manual bilinear resize (avoids imresize)
    img = bilinear_resize(double(raw) / 255.0, img_size, img_size);
end

function img = load_dicom_file(path, img_size)
    % Load DICOM via dicomread (MATLAB built-in), then normalise & resize manually.
    raw  = double(dicomread(path));
    if ndims(raw) == 4          % multi-frame: take middle frame
        mid = floor(size(raw,4)/2) + 1;
        raw = raw(:,:,1,mid);
    elseif ndims(raw) == 3
        raw = raw(:,:,1);
    end
    lo = min(raw(:)); hi = max(raw(:));
    raw = (raw - lo) / (hi - lo + 1e-8);
    img = bilinear_resize(raw, img_size, img_size);
end

function out = bilinear_resize(img, new_h, new_w)
    % Bilinear interpolation resize — no imresize used.
    [h, w] = size(img);
    [gy, gx] = ndgrid(linspace(1, h, new_h), linspace(1, w, new_w));
    y0 = floor(gy); x0 = floor(gx);
    y1 = min(y0 + 1, h); x1 = min(x0 + 1, w);
    dy = gy - y0; dx = gx - x0;
    % Clamp source indices
    y0 = max(1, min(y0, h)); x0 = max(1, min(x0, w));
    y1 = max(1, min(y1, h)); x1 = max(1, min(x1, w));
    % Four-corner bilinear blend
    idx = @(r,c) sub2ind([h w], r, c);
    out = img(idx(y0,x0)).*(1-dy).*(1-dx) ...
        + img(idx(y1,x0)).*   dy .*(1-dx) ...
        + img(idx(y0,x1)).*(1-dy).*   dx  ...
        + img(idx(y1,x1)).*   dy .*   dx;
    out = min(max(out, 0), 1);
end

function show_grid(images, titles, suptitle_str)
    % Display a row of greyscale images with colorbars.
    n   = numel(images);
    fig = figure('Units','normalized','Position',[0.05 0.1 0.9 0.35]);
    for k = 1:n
        ax = subplot(1, n, k);
        imagesc(images{k}, [0 1]);  colormap(ax, gray);  axis image off;
        colorbar;
        title(titles{k}, 'FontWeight','bold', 'FontSize', 9, ...
              'Interpreter','none');
    end
    sgtitle(suptitle_str, 'FontWeight','bold', 'FontSize', 11, ...
            'Interpreter','none');
end

% ── 1.3  Collect image paths ─────────────────────────────────────────────────
fprintf('=== Scanning Kaggle dataset directories ===\n');

% MRI: walk all subdirs, group by class folder name
mri_all  = dir(fullfile(MRI_DIR, '**', '*.jpg'));
mri_all  = [mri_all; dir(fullfile(MRI_DIR, '**', '*.png'))];
if isempty(mri_all)
    error('No MRI images found in %s\nRun: kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset --unzip -p %s', MRI_DIR, MRI_DIR);
end
fprintf('Found %d MRI images\n', numel(mri_all));

xray_all = dir(fullfile(XRAY_DIR, '**', '*.png'));
xray_all = [xray_all; dir(fullfile(XRAY_DIR, '**', '*.jpg'))];
if isempty(xray_all)
    error('No X-ray images found in %s\nRun: kaggle datasets download nih-chest-xrays/data ...', XRAY_DIR);
end
fprintf('Found %d X-ray images\n', numel(xray_all));

ct_all_dcm = dir(fullfile(CT_DIR, '**', '*.dcm'));
ct_all_png = dir(fullfile(CT_DIR, '**', '*.png'));
if isempty(ct_all_dcm) && isempty(ct_all_png)
    error('No CT images found in %s\nRun: kaggle datasets download kmader/siim-medical-images --unzip -p %s', CT_DIR, CT_DIR);
end
if ~isempty(ct_all_dcm)
    ct_all = ct_all_dcm;  use_dicom = true;
    fprintf('Found %d CT DICOM files\n', numel(ct_all));
else
    ct_all = ct_all_png;  use_dicom = false;
    fprintf('Found %d CT PNG files\n', numel(ct_all));
end

% ── 1.4  Select 50 representative images from each modality ──────────────────
rng(42);   % reproducible random seed

% MRI — group by class (parent folder name), pick evenly per class
mri_classes = containers.Map('KeyType','char','ValueType','any');
for k = 1:numel(mri_all)
    cls = mri_all(k).folder;
    [~, cls] = fileparts(cls);   % last folder name = class
    if ~isKey(mri_classes, cls)
        mri_classes(cls) = {};
    end
    mri_classes(cls) = [mri_classes(cls), {fullfile(mri_all(k).folder, mri_all(k).name)}];
end
cls_names      = keys(mri_classes);
n_cls          = numel(cls_names);
per_class      = max(1, floor(IMAGES_PER_DATASET / n_cls));
mri_paths      = {};
mri_class_labels = {};
for k = 1:n_cls
    files = mri_classes(cls_names{k});
    idx   = randperm(numel(files), min(per_class, numel(files)));
    for j = idx
        mri_paths{end+1}        = files{j};  %#ok<AGROW>
        mri_class_labels{end+1} = cls_names{k}; %#ok<AGROW>
    end
end
mri_paths = mri_paths(1:min(IMAGES_PER_DATASET, numel(mri_paths)));
mri_class_labels = mri_class_labels(1:numel(mri_paths));
fprintf('Selected %d MRI images across %d classes\n', numel(mri_paths), n_cls);

% X-ray — evenly spaced across the full list
n_xr      = numel(xray_all);
step      = max(1, floor(n_xr / IMAGES_PER_DATASET));
xr_idx    = 1:step:n_xr;
xr_idx    = xr_idx(1:min(IMAGES_PER_DATASET, numel(xr_idx)));
xray_paths = arrayfun(@(k) fullfile(xray_all(k).folder, xray_all(k).name), ...
                      xr_idx, 'UniformOutput', false);
fprintf('Selected %d X-ray images\n', numel(xray_paths));

% CT — evenly spaced
n_ct      = numel(ct_all);
step      = max(1, floor(n_ct / IMAGES_PER_DATASET));
ct_idx    = 1:step:n_ct;
ct_idx    = ct_idx(1:min(IMAGES_PER_DATASET, numel(ct_idx)));
ct_paths  = arrayfun(@(k) fullfile(ct_all(k).folder, ct_all(k).name), ...
                     ct_idx, 'UniformOutput', false);
fprintf('Selected %d CT files\n', numel(ct_paths));

% ── 1.5  Load all selected images into cell arrays ────────────────────────────
fprintf('\nLoading MRI images ...\n');
mri_imgs = cell(1, numel(mri_paths));
for k = 1:numel(mri_paths)
    mri_imgs{k} = load_image_file(mri_paths{k}, IMG_SIZE);
end

fprintf('Loading X-ray images ...\n');
xray_imgs = cell(1, numel(xray_paths));
for k = 1:numel(xray_paths)
    xray_imgs{k} = load_image_file(xray_paths{k}, IMG_SIZE);
end

fprintf('Loading CT images ...\n');
ct_imgs = cell(1, numel(ct_paths));
for k = 1:numel(ct_paths)
    if use_dicom
        ct_imgs{k} = load_dicom_file(ct_paths{k}, IMG_SIZE);
    else
        ct_imgs{k} = load_image_file(ct_paths{k}, IMG_SIZE);
    end
end

% Canonical single images used throughout the script
mri_img   = mri_imgs{1};
mri_img2  = mri_imgs{min(2, numel(mri_imgs))};
xray_img  = xray_imgs{1};
xray_img2 = xray_imgs{min(2, numel(xray_imgs))};
ct_img    = ct_imgs{1};

fprintf('All images loaded successfully.\n');

% ── 1.6  Preview grid (3 rows × 4 cols) ─────────────────────────────────────
figure('Name','Real Kaggle Medical Images','Units','normalized', ...
       'Position',[0.05 0.05 0.9 0.85]);

% Row 1: MRI per class (up to 4)
for k = 1:min(n_cls, 4)
    subplot(3, 4, k);
    imagesc(mri_imgs{k}, [0 1]);  colormap(gca, gray);  axis image off;
    [~, fn] = fileparts(mri_paths{k});
    title(sprintf('MRI\n%s', mri_class_labels{k}), 'FontWeight','bold', ...
          'FontSize',9, 'Interpreter','none');
end

% Row 2: X-rays
for k = 1:min(4, numel(xray_imgs))
    subplot(3, 4, 4+k);
    imagesc(xray_imgs{k}, [0 1]);  colormap(gca, gray);  axis image off;
    [~, fn] = fileparts(xray_paths{k});
    title(sprintf('X-ray %d\n%s', k, fn(1:min(20,end))), 'FontWeight','bold', ...
          'FontSize',9, 'Interpreter','none');
end

% Row 3: CT
for k = 1:min(4, numel(ct_imgs))
    subplot(3, 4, 8+k);
    imagesc(ct_imgs{k}, [0 1]);  colormap(gca, gray);  axis image off;
    [~, fn] = fileparts(ct_paths{k});
    title(sprintf('CT %d\n%s', k, fn(1:min(25,end))), 'FontWeight','bold', ...
          'FontSize',9, 'Interpreter','none');
end

sgtitle('Real Kaggle Medical Images — MRI (Brain Tumor) · X-ray (NIH) · CT (SIIM)', ...
        'FontWeight','bold', 'FontSize',12, 'Interpreter','none');


%% ============================================================
%  SECTION 2 — Arithmetic Operations
%  All implemented as element-wise array math — no imadd/imsubtract/etc.
% =============================================================

% ── Operation definitions ────────────────────────────────────────────────────

function out = img_sum(A, B, alpha, beta)
    % Weighted sum: out(i,j) = alpha*A(i,j) + beta*B(i,j)
    if nargin < 3, alpha = 0.5; end
    if nargin < 4, beta  = 0.5; end
    out = min(max(alpha .* A + beta .* B, 0), 1);
end

function out = img_sub(A, B, scale, offset)
    % Scaled difference with midpoint offset: out = clip(scale*(A-B)+offset, 0,1)
    if nargin < 3, scale  = 1.0; end
    if nargin < 4, offset = 0.5; end
    out = min(max(scale .* (A - B) + offset, 0), 1);
end

function out = img_mul(A, k)
    % Multiply every pixel by scalar k
    out = min(max(A .* k, 0), 1);
end

function out = img_div(A, B, eps_val)
    % Element-wise ratio, normalised to [0,1]
    if nargin < 3, eps_val = 1e-6; end
    ratio = A ./ (B + eps_val);
    out   = min(max(ratio ./ (max(ratio(:)) + eps_val), 0), 1);
end

% ── Apply to real Kaggle images ───────────────────────────────────────────────
fused_mri_xr = img_sum(mri_img,  xray_img,  0.55, 0.45);
fused_ct_mri = img_sum(ct_img,   mri_img,   0.50, 0.50);
diff_xr_xr2  = img_sub(xray_img, xray_img2, 3.0,  0.5);
diff_mri_ct  = img_sub(mri_img,  ct_img,    2.0,  0.5);
bright_xray  = img_mul(xray_img, 2.0);
dark_ct      = img_mul(ct_img,   0.4);
ratio_mri_ct = img_div(mri_img,  ct_img);

% ── Plots ─────────────────────────────────────────────────────────────────────
show_grid({mri_img, xray_img, fused_mri_xr, fused_ct_mri}, ...
    {'Brain MRI (Kaggle)', 'Chest X-ray (NIH)', ...
     'Sum: 0.55·MRI + 0.45·X-ray', 'Sum: 0.5·CT + 0.5·MRI'}, ...
    'Arithmetic — Sum: Multi-Modal Fusion (real Kaggle images)');

show_grid({xray_img, xray_img2, diff_xr_xr2, diff_mri_ct}, ...
    {'X-ray #1 (NIH)', 'X-ray #2 (NIH)', ...
     'Subtraction: X-ray1 − X-ray2', 'Subtraction: MRI − CT'}, ...
    'Arithmetic — Subtraction: Difference Imaging');

show_grid({xray_img, bright_xray, ct_img, dark_ct, ratio_mri_ct}, ...
    {'X-ray (NIH)', 'X-ray × 2.0 (contrast boost)', ...
     'CT (SIIM)', 'CT × 0.4 (low-dose sim.)', 'Ratio MRI/CT'}, ...
    'Arithmetic — Multiplication & Division');


%% ============================================================
%  SECTION 3 — Intensity Transformations
%  Negative, Log, Gamma — all pixel-wise, no histeq/imadjust
% =============================================================

% ── Transform definitions ────────────────────────────────────────────────────

function out = img_negative(img)
    % Invert: s = 1 - r
    out = 1.0 - img;
end

function out = img_log(img, c)
    % Logarithmic stretch: s = c*log(1+r)
    % c normalises output so that r=1 maps to s=1
    if nargin < 2 || isempty(c)
        c = 1.0 / log(2.0);
    end
    out = min(max(c .* log(1.0 + img), 0), 1);
end

function out = img_gamma(img, gamma)
    % Power-law: s = r^gamma
    % gamma < 1 brightens;  gamma > 1 darkens
    out = min(max(img .^ gamma, 0), 1);
end

% ── Transformation curves plot ────────────────────────────────────────────────
r_vals = linspace(0, 1, 512);
figure('Name','Intensity Transform Curves','Units','normalized', ...
       'Position',[0.1 0.2 0.55 0.5]);
plot(r_vals, img_negative(r_vals),    'r-',  'LineWidth',2.5, 'DisplayName','Negative (1−r)'); hold on;
plot(r_vals, img_log(r_vals),         'b-',  'LineWidth',2.5, 'DisplayName','Log c·log(1+r)');
plot(r_vals, img_gamma(r_vals, 0.3),  'g--', 'LineWidth',2,   'DisplayName','Gamma γ=0.3 (brighten)');
plot(r_vals, img_gamma(r_vals, 0.6),  '-',   'LineWidth',2,   'Color',[0.2 0.8 0.2], 'DisplayName','Gamma γ=0.6');
plot(r_vals, img_gamma(r_vals, 1.0),  'k:',  'LineWidth',1,   'DisplayName','Identity γ=1.0');
plot(r_vals, img_gamma(r_vals, 1.8),  '-',   'LineWidth',2,   'Color',[1 0.5 0],     'DisplayName','Gamma γ=1.8');
plot(r_vals, img_gamma(r_vals, 3.0),  '-.',  'LineWidth',2,   'Color',[0.5 0.2 0],   'DisplayName','Gamma γ=3.0 (darken)');
xlabel('Input intensity r','FontSize',12);
ylabel('Output s','FontSize',12);
title('Intensity Transformation Curves','FontWeight','bold','FontSize',13);
legend('Location','northwest','FontSize',9);  grid on;

% ── Apply to real images ──────────────────────────────────────────────────────
% MRI
mri_neg = img_negative(mri_img);
mri_log = img_log(mri_img);
mri_g04 = img_gamma(mri_img, 0.4);
mri_g25 = img_gamma(mri_img, 2.5);

% X-ray
xr_neg  = img_negative(xray_img);
xr_log  = img_log(xray_img);
xr_g03  = img_gamma(xray_img, 0.3);
xr_g30  = img_gamma(xray_img, 3.0);

% CT
ct_neg  = img_negative(ct_img);
ct_log  = img_log(ct_img);
ct_g18  = img_gamma(ct_img, 1.8);

show_grid({mri_img, mri_neg, mri_log, mri_g04, mri_g25}, ...
    {'MRI original (Brain Tumor)', 'Negative', 'Log', 'Gamma γ=0.4', 'Gamma γ=2.5'}, ...
    'Intensity Transforms — Brain Tumor MRI (Kaggle)');

show_grid({xray_img, xr_neg, xr_log, xr_g03, xr_g30}, ...
    {'X-ray original (NIH)', 'Negative', 'Log', 'Gamma γ=0.3', 'Gamma γ=3.0'}, ...
    'Intensity Transforms — NIH Chest X-ray (Kaggle)');

show_grid({ct_img, ct_neg, ct_log, ct_g18}, ...
    {'CT original (SIIM)', 'Negative', 'Log', 'Gamma γ=1.8'}, ...
    'Intensity Transforms — SIIM CT Scan (Kaggle)');

% ── Histograms — manual implementation (no imhist) ────────────────────────────
function [centers, counts] = manual_hist(img, n_bins)
    if nargin < 2, n_bins = 64; end
    edges   = linspace(0, 1, n_bins + 1);
    counts  = zeros(1, n_bins);
    flat    = img(:);
    for k = 1:n_bins
        counts(k) = sum(flat >= edges(k) & flat < edges(k+1));
    end
    centers = (edges(1:end-1) + edges(2:end)) / 2;
end

pairs  = {mri_img,'MRI original'; mri_log,'MRI Log'; ...
          xray_img,'X-ray original'; xr_g03,'X-ray γ=0.3'; ...
          ct_img,'CT original'; ct_neg,'CT Negative'};
colors = [0.27 0.51 0.71; 0.25 0.41 0.88; 0.70 0.13 0.13; ...
          1.00 0.55 0.00; 0.13 0.55 0.13; 0.58 0.44 0.86];

figure('Name','Histograms','Units','normalized','Position',[0.05 0.05 0.9 0.7]);
for k = 1:6
    subplot(2, 3, k);
    [c, h] = manual_hist(pairs{k,1});
    bar(c, h, 1.0, 'FaceColor', colors(k,:), 'EdgeColor','none', 'FaceAlpha',0.85);
    title(pairs{k,2}, 'FontWeight','bold', 'FontSize',10, 'Interpreter','none');
    xlabel('Intensity');  ylabel('Count');  grid on;
end
sgtitle('Pixel Histograms Before/After Intensity Transforms — Kaggle Datasets', ...
        'FontWeight','bold', 'FontSize',12, 'Interpreter','none');


%% ============================================================
%  SECTION 4 — Geometric Transformations
%  Scaling, Rotation, Translation — inverse mapping + bilinear interpolation
%  No imresize, imrotate, imtranslate used.
% =============================================================

% ── Shared: bilinear sampler (flat index version) ─────────────────────────────
function out = bilinear_sample(img, y_src, x_src)
    % Sample img at fractional (y_src, x_src); out-of-bounds → 0.
    % y_src, x_src are matrices of the same size as the output.
    [H, W] = size(img);
    y0 = floor(y_src);  x0 = floor(x_src);
    y1 = y0 + 1;        x1 = x0 + 1;
    dy = y_src - y0;    dx = x_src - x0;

    valid = (y0 >= 1) & (y1 <= H) & (x0 >= 1) & (x1 <= W);

    y0c = max(1, min(y0, H));  y1c = max(1, min(y1, H));
    x0c = max(1, min(x0, W));  x1c = max(1, min(x1, W));

    idx00 = sub2ind([H W], y0c, x0c);
    idx10 = sub2ind([H W], y1c, x0c);
    idx01 = sub2ind([H W], y0c, x1c);
    idx11 = sub2ind([H W], y1c, x1c);

    out = img(idx00).*(1-dy).*(1-dx) ...
        + img(idx10).*   dy .*(1-dx) ...
        + img(idx01).*(1-dy).*   dx  ...
        + img(idx11).*   dy .*   dx;
    out(~valid) = 0;
    out = min(max(out, 0), 1);
end

% ── Scale ────────────────────────────────────────────────────────────────────
function out = scale_img(img, sx, sy)
    % Scale by (sx, sy); output resized back to original dims for comparison.
    [H, W] = size(img);
    oh = round(H * sy);  ow = round(W * sx);
    [gy, gx] = ndgrid(1:oh, 1:ow);
    src_y = gy ./ sy;  src_x = gx ./ sx;   % inverse mapping
    scaled = bilinear_sample(img, src_y, src_x);
    % Resize back to original shape for fair display
    out = bilinear_resize(scaled, H, W);
end

% ── Rotate ───────────────────────────────────────────────────────────────────
function out = rotate_img(img, angle_deg)
    % Rotate CCW by angle_deg around image centre (inverse mapping).
    [H, W] = size(img);
    theta  = -deg2rad(angle_deg);   % inverse rotation
    cx = (W + 1) / 2;  cy = (H + 1) / 2;
    [gy, gx] = ndgrid(1:H, 1:W);
    gx_c = gx - cx;  gy_c = gy - cy;
    src_x = gx_c .* cos(theta) - gy_c .* sin(theta) + cx;
    src_y = gx_c .* sin(theta) + gy_c .* cos(theta) + cy;
    out = bilinear_sample(img, src_y, src_x);
end

% ── Translate ────────────────────────────────────────────────────────────────
function out = translate_img(img, tx, ty)
    % Shift right by tx and down by ty pixels (inverse mapping).
    [H, W] = size(img);
    [gy, gx] = ndgrid(1:H, 1:W);
    src_x = gx - tx;
    src_y = gy - ty;
    out = bilinear_sample(img, src_y, src_x);
end

% ── 4.1  Scaling — Brain Tumor MRI ────────────────────────────────────────────
mri_s15 = scale_img(mri_img, 1.5, 1.5);
mri_s25 = scale_img(mri_img, 2.5, 2.5);
mri_s05 = scale_img(mri_img, 0.5, 0.5);

show_grid({mri_img, mri_s15, mri_s25, mri_s05}, ...
    {'MRI original (Brain Tumor)', 'Scale ×1.5 (zoom in)', ...
     'Scale ×2.5 (close-up)', 'Scale ×0.5 (zoom out)'}, ...
    'Geometric — Scaling [Brain Tumor MRI — ROI Magnification]');

xr_s15 = scale_img(xray_img, 1.5, 1.5);
xr_s05 = scale_img(xray_img, 0.5, 0.5);
show_grid({xray_img, xr_s15, xr_s05}, ...
    {'X-ray original (NIH)', 'Scale ×1.5', 'Scale ×0.5'}, ...
    'Geometric — Scaling [NIH Chest X-ray]');

% ── 4.2  Rotation ─────────────────────────────────────────────────────────────
mri_r15  = rotate_img(mri_img,  15);
mri_r45  = rotate_img(mri_img,  45);
mri_r90  = rotate_img(mri_img,  90);
mri_r180 = rotate_img(mri_img, 180);

show_grid({mri_img, mri_r15, mri_r45, mri_r90, mri_r180}, ...
    {'MRI 0°','15°','45°','90°','180°'}, ...
    'Geometric — Rotation [Brain Tumor MRI — data augmentation / orientation correction]');

xr_r10  = rotate_img(xray_img,  10);
xr_rm10 = rotate_img(xray_img, -10);
xr_r90  = rotate_img(xray_img,  90);
show_grid({xray_img, xr_r10, xr_rm10, xr_r90}, ...
    {'X-ray 0°', '+10°', '−10°', '90°'}, ...
    'Geometric — Rotation [NIH Chest X-ray — patient tilt correction]');

ct_r20  = rotate_img(ct_img,  20);
ct_rm20 = rotate_img(ct_img, -20);
show_grid({ct_img, ct_r20, ct_rm20}, ...
    {'CT 0°', '+20°', '−20°'}, ...
    'Geometric — Rotation [SIIM CT — gantry tilt correction]');

% ── 4.3  Translation ──────────────────────────────────────────────────────────
mri_tr1 = translate_img(mri_img,  30,   0);
mri_tr2 = translate_img(mri_img,   0,  30);
mri_tr3 = translate_img(mri_img, -20, -20);

show_grid({mri_img, mri_tr1, mri_tr2, mri_tr3}, ...
    {'MRI original', 'tx=+30 (right)', 'ty=+30 (down)', 'tx=−20, ty=−20'}, ...
    'Geometric — Translation [Brain Tumor MRI]');

xr_tr = translate_img(xray_img,  25, -20);
ct_tr = translate_img(ct_img,   -20,  25);
show_grid({xray_img, xr_tr, abs(xray_img - xr_tr), ct_img, ct_tr, abs(ct_img - ct_tr)}, ...
    {'X-ray orig','X-ray translated','X-ray diff', 'CT orig','CT translated','CT diff'}, ...
    'Geometric — Translation + Difference Map [NIH X-ray · SIIM CT]');

% ── Combined affine: scale → rotate → translate ───────────────────────────────
function out = affine_transform(img, sx, sy, angle, tx, ty)
    out = scale_img(img, sx, sy);
    out = rotate_img(out, angle);
    out = translate_img(out, tx, ty);
end

ct_ref = ct_img;
ct_mis = affine_transform(ct_img,      1.05,  1.05, -7,   12,  -8);
ct_reg = affine_transform(ct_mis, 1/1.05, 1/1.05,  7,  -12,   8);

diff_before = abs(ct_ref - ct_mis);
diff_after  = abs(ct_ref - ct_reg);

show_grid({ct_ref, ct_mis, ct_reg}, ...
    {'CT Reference (SIIM)', 'CT Misaligned (scale+rot+trans)', 'CT Registered (inverse affine)'}, ...
    'Geometric — Combined Affine Pipeline: SIIM CT Registration');

show_grid({diff_before, diff_after}, ...
    {sprintf('Diff BEFORE  (mean=%.4f)', mean(diff_before(:))), ...
     sprintf('Diff AFTER   (mean=%.4f)', mean(diff_after(:)))}, ...
    'Registration Quality — lower mean = better alignment');


%% ============================================================
%  SECTION 5 — Medical Case Studies
% =============================================================

% ── Case A: All MRI tumour classes × three transforms ─────────────────────────
n_show = min(n_cls, 4);
figure('Name','Case A — MRI Tumour Classes','Units','normalized', ...
       'Position',[0.05 0.05 0.9 0.8]);
transform_fns   = {@(x) x, @img_negative, @(x) img_gamma(x, 0.4)};
transform_labels = {'Original', 'Negative (1−r)', 'Gamma γ=0.4 (brighten)'};
for col = 1:n_show
    % Pick one image per class
    cls_files = mri_classes(cls_names{col});
    img_c = load_image_file(cls_files{1}, IMG_SIZE);
    for row = 1:3
        subplot(3, n_show, (row-1)*n_show + col);
        imagesc(transform_fns{row}(img_c), [0 1]);
        colormap(gca, gray);  axis image off;
        if row == 1
            ttl = sprintf('%s\n%s', cls_names{col}, transform_labels{row});
        else
            ttl = transform_labels{row};
        end
        title(ttl, 'FontWeight','bold', 'FontSize',9, 'Interpreter','none');
    end
end
sgtitle('Case A — Brain Tumor MRI (Kaggle): All Classes × Three Intensity Transforms', ...
        'FontWeight','bold', 'FontSize',11, 'Interpreter','none');

% ── Case B: DSA simulation on NIH X-ray ──────────────────────────────────────
function out = add_vessels(img, density, n_vessels, seed_val)
    if nargin < 2, density   = 0.45; end
    if nargin < 3, n_vessels = 14;   end
    if nargin < 4, seed_val  = 99;   end
    rng(seed_val);
    out = img;
    [H, W] = size(img);
    for v = 1:n_vessels
        x0     = randi([20, W-20]);
        y0     = randi([20, H-20]);
        len    = randi([40, 110]);
        angle  = rand() * pi;
        thick  = randi([2, 4]);
        t_vals = linspace(0, len, 300);
        for t = t_vals
            xi = round(x0 + t * cos(angle));
            yi = round(y0 + t * sin(angle) + 6 * sin(t * 0.15));
            for dy = -thick:thick
                for dx = -thick:thick
                    yy = yi + dy;  xx = xi + dx;
                    if yy >= 1 && yy <= H && xx >= 1 && xx <= W
                        out(yy, xx) = min(1.0, out(yy, xx) + density);
                    end
                end
            end
        end
    end
    out = min(max(out, 0), 1);
end

mask_img     = xray_img;
contrast_img = add_vessels(xray_img, 0.50, 14, 2024);
dsa_raw      = img_sub(contrast_img, mask_img, 4.0, 0.0);
dsa_final    = img_gamma(min(max(dsa_raw, 0), 1), 0.5);

show_grid({mask_img, contrast_img, dsa_final}, ...
    {'Pre-contrast (NIH X-ray)', 'Post-contrast (vessels added)', ...
     'DSA result: subtraction + gamma'}, ...
    'Case B — DSA Simulation on NIH Chest X-ray (Kaggle)');

% ── Case C: CT windowing on SIIM data ─────────────────────────────────────────
function out = ct_window(img, center, width)
    lo  = center - width / 2.0;
    hi  = center + width / 2.0;
    out = min(max((img - lo) ./ (hi - lo + 1e-8), 0), 1);
end

brain_w  = ct_window(ct_img, 0.40, 0.35);
bone_w   = ct_window(ct_img, 0.65, 0.80);
tissue_w = ct_window(ct_img, 0.30, 0.20);

show_grid({ct_img, brain_w, bone_w, tissue_w}, ...
    {'CT original (SIIM)', 'Brain window (c=0.40, w=0.35)', ...
     'Bone window (c=0.65, w=0.80)', 'Soft-tissue window (c=0.30, w=0.20)'}, ...
    'Case C — CT Windowing on Real SIIM CT (Kaggle)');

% ── Case D: Checkerboard registration QC ─────────────────────────────────────
function out = checkerboard_overlay(A, B, tile)
    if nargin < 3, tile = 32; end
    [H, W] = size(A);
    out = zeros(H, W);
    for r = 0:tile:H-1
        for c = 0:tile:W-1
            rr = r+1 : min(r+tile, H);
            cc = c+1 : min(c+tile, W);
            if mod(floor(r/tile) + floor(c/tile), 2) == 0
                out(rr, cc) = A(rr, cc);
            else
                out(rr, cc) = B(rr, cc);
            end
        end
    end
end

checker = checkerboard_overlay(xray_img, ct_img, 32);
fc      = cat(3, xray_img, ct_img, img_sum(xray_img, ct_img));  % RGB false-colour

figure('Name','Case D — Registration QC','Units','normalized', ...
       'Position',[0.05 0.1 0.9 0.4]);
subplot(1,4,1); imagesc(xray_img,[0 1]); colormap(gca,gray); axis image off;
    title('X-ray (NIH)','FontWeight','bold','Interpreter','none');
subplot(1,4,2); imagesc(ct_img,[0 1]);   colormap(gca,gray); axis image off;
    title('CT (SIIM)','FontWeight','bold','Interpreter','none');
subplot(1,4,3); imagesc(checker,[0 1]);  colormap(gca,gray); axis image off;
    title('Checkerboard QC','FontWeight','bold','Interpreter','none');
subplot(1,4,4); imshow(fc);  axis image off;
    title('False-Colour Fusion (R=X-ray, G=CT)','FontWeight','bold','Interpreter','none');
sgtitle('Case D — Multi-Modal Registration QC: SIIM CT + NIH X-ray (Kaggle)', ...
        'FontWeight','bold', 'FontSize',11, 'Interpreter','none');

% ── Case E: Tumour boundary enhancement pipeline ──────────────────────────────
tumour_cls = cls_names{1};
for k = 1:n_cls
    if contains(lower(cls_names{k}), 'glioma') || contains(lower(cls_names{k}), 'meningioma')
        tumour_cls = cls_names{k};  break;
    end
end
cls_files = mri_classes(tumour_cls);
tumour    = load_image_file(cls_files{1}, IMG_SIZE);

step1 = img_gamma(tumour, 0.4);           % brighten dark tumour regions
step2 = img_log(step1);                   % compress bright skull/CSF
step3 = img_sub(step1, step2, 3.0, 0.5); % edge-emphasise by subtraction
step4 = rotate_img(step3, 15);            % data augmentation rotation

show_grid({tumour, step1, step2, step3, step4}, ...
    {sprintf('Original MRI (%s)', tumour_cls), ...
     'Step 1: Gamma γ=0.4 (brighten)', ...
     'Step 2: Log (compress skull/CSF)', ...
     'Step 3: Subtraction (edge enhance)', ...
     'Step 4: Rotation 15° (augment)'}, ...
    sprintf('Case E — Tumour Boundary Enhancement: %s MRI (Kaggle)', tumour_cls));

fprintf('\n=== All sections complete. ===\n');
fprintf('Figures: Section 1 Preview | Arithmetic (3) | Intensity (4+hist) | Geometric (6) | Cases (5)\n');
