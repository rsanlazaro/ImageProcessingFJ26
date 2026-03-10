%% ============================================================
%  Medical Image Loading — Real Kaggle Datasets
%  Downloads images and stores them in workspace variables.
%  No processing, no plotting.
% =============================================================
%
%  NO SETUP REQUIRED — a dialog will ask for your Kaggle credentials.
%  Get your free API key at:
%    https://www.kaggle.com/settings  →  "Create New Token"
% =============================================================

clear; clc; close all;

%% ============================================================
%  PARAMETERS
% =============================================================

DATA_ROOT          = fullfile(tempdir, 'kaggle_medical');
MRI_DIR            = fullfile(DATA_ROOT, 'brain_tumor_mri');
XRAY_DIR           = fullfile(DATA_ROOT, 'chest_xray');
CT_DIR             = fullfile(DATA_ROOT, 'siim_ct');

IMAGES_PER_DATASET = 50;   % images to load per modality
IMG_SIZE           = 256;  % resize all images to IMG_SIZE x IMG_SIZE

%% ============================================================
%  SECTION 1 — Kaggle Credentials
% =============================================================

[kg_user, kg_key] = get_kaggle_credentials();
fprintf('Credentials accepted for user: %s\n', kg_user);

%% ============================================================
%  SECTION 2 — Download Datasets
% =============================================================

fprintf('\n=== Checking / downloading Kaggle datasets ===\n');

kaggle_download(kg_user, kg_key, 'masoudnickparvar', 'brain-tumor-mri-dataset', MRI_DIR);
kaggle_download(kg_user, kg_key, 'paultimothymooney', 'chest-xray-pneumonia',   XRAY_DIR);
kaggle_download(kg_user, kg_key, 'kmader',            'siim-medical-images',     CT_DIR);

fprintf('=== All datasets ready ===\n\n');

%% ============================================================
%  SECTION 3 — Collect & Select Image Paths
% =============================================================

fprintf('=== Scanning dataset directories ===\n');

% ── MRI ───────────────────────────────────────────────────────────────────────
mri_all = collect_images(MRI_DIR, {'.jpg','.jpeg','.png'});
if isempty(mri_all)
    error('No MRI images found in %s', MRI_DIR);
end
fprintf('Found %d MRI images\n', numel(mri_all));

% Group by class (sub-folder name) and sample evenly across classes
mri_classes = containers.Map('KeyType','char','ValueType','any');
for k = 1:numel(mri_all)
    [~, cls] = fileparts(fileparts(mri_all{k}));   % parent folder = class
    if ~isKey(mri_classes, cls), mri_classes(cls) = {}; end
    mri_classes(cls) = [mri_classes(cls), mri_all(k)];
end
cls_names = keys(mri_classes);
n_cls     = numel(cls_names);
per_class = max(1, floor(IMAGES_PER_DATASET / n_cls));

rng(42);
mri_paths = {}; mri_labels = {};
for k = 1:n_cls
    files = mri_classes(cls_names{k});
    idx   = randperm(numel(files), min(per_class, numel(files)));
    for j = idx
        mri_paths{end+1}  = files{j};        %#ok<AGROW>
        mri_labels{end+1} = cls_names{k};    %#ok<AGROW>
    end
end
mri_paths  = mri_paths(1:min(IMAGES_PER_DATASET, numel(mri_paths)));
mri_labels = mri_labels(1:numel(mri_paths));
fprintf('Selected %d MRI images across %d classes: %s\n', ...
        numel(mri_paths), n_cls, strjoin(cls_names, ', '));

% ── X-ray ─────────────────────────────────────────────────────────────────────
xray_all = collect_images(XRAY_DIR, {'.jpg','.jpeg','.png'});
if isempty(xray_all)
    error('No X-ray images found in %s', XRAY_DIR);
end
fprintf('Found %d X-ray images\n', numel(xray_all));

step       = max(1, floor(numel(xray_all) / IMAGES_PER_DATASET));
xray_paths = xray_all(1:step:end);
xray_paths = xray_paths(1:min(IMAGES_PER_DATASET, numel(xray_paths)));
fprintf('Selected %d X-ray images\n', numel(xray_paths));

% ── CT ────────────────────────────────────────────────────────────────────────
ct_all_dcm = collect_images(CT_DIR, {'.dcm'});
ct_all_png = collect_images(CT_DIR, {'.png','.jpg','.jpeg'});

if ~isempty(ct_all_dcm)
    ct_all    = ct_all_dcm;
    use_dicom = true;
    fprintf('Found %d CT DICOM files\n', numel(ct_all));
elseif ~isempty(ct_all_png)
    ct_all    = ct_all_png;
    use_dicom = false;
    fprintf('Found %d CT image files\n', numel(ct_all));
else
    error('No CT images found in %s', CT_DIR);
end

step     = max(1, floor(numel(ct_all) / IMAGES_PER_DATASET));
ct_paths = ct_all(1:step:end);
ct_paths = ct_paths(1:min(IMAGES_PER_DATASET, numel(ct_paths)));
fprintf('Selected %d CT files\n', numel(ct_paths));

%% ============================================================
%  SECTION 4 — Load Images into Workspace Variables
% =============================================================

fprintf('\n=== Loading images ===\n');

% ── MRI ───────────────────────────────────────────────────────────────────────
fprintf('Loading MRI images ...\n');
mri_imgs = cell(1, numel(mri_paths));
for k = 1:numel(mri_paths)
    mri_imgs{k} = safe_load(mri_paths{k}, IMG_SIZE, false);
end
% Remove any failed loads
valid      = ~cellfun(@isempty, mri_imgs);
mri_imgs   = mri_imgs(valid);
mri_paths  = mri_paths(valid);
mri_labels = mri_labels(valid);
fprintf('  Loaded %d / %d MRI images\n', numel(mri_imgs), sum(valid | ~valid));

% ── X-ray ─────────────────────────────────────────────────────────────────────
fprintf('Loading X-ray images ...\n');
xray_imgs = cell(1, numel(xray_paths));
for k = 1:numel(xray_paths)
    xray_imgs{k} = safe_load(xray_paths{k}, IMG_SIZE, false);
end
valid      = ~cellfun(@isempty, xray_imgs);
xray_imgs  = xray_imgs(valid);
xray_paths = xray_paths(valid);
fprintf('  Loaded %d X-ray images\n', numel(xray_imgs));

% ── CT ────────────────────────────────────────────────────────────────────────
fprintf('Loading CT images ...\n');
ct_imgs = cell(1, numel(ct_paths));
for k = 1:numel(ct_paths)
    ct_imgs{k} = safe_load(ct_paths{k}, IMG_SIZE, use_dicom);
end
valid    = ~cellfun(@isempty, ct_imgs);
ct_imgs  = ct_imgs(valid);
ct_paths = ct_paths(valid);
fprintf('  Loaded %d CT images\n', numel(ct_imgs));

%% ============================================================
%  SECTION 5 — Summary
% =============================================================

fprintf('\n=== Workspace variables ready ===\n');
fprintf('  mri_imgs   : 1x%d cell — greyscale double [0,1], %dx%d each\n', numel(mri_imgs),  IMG_SIZE, IMG_SIZE);
fprintf('  mri_labels : 1x%d cell — tumour class name per image\n',         numel(mri_labels));
fprintf('  xray_imgs  : 1x%d cell — greyscale double [0,1], %dx%d each\n', numel(xray_imgs), IMG_SIZE, IMG_SIZE);
fprintf('  ct_imgs    : 1x%d cell — greyscale double [0,1], %dx%d each\n', numel(ct_imgs),   IMG_SIZE, IMG_SIZE);
fprintf('\nAll images stored as float64 in [0,1]. Ready for further processing.\n');


%% ============================================================
%  LOCAL FUNCTIONS  (must appear after all executable code)
% =============================================================

% ── Credential dialog ─────────────────────────────────────────────────────────
function [username, api_key] = get_kaggle_credentials()
    % 1. Try standard kaggle.json locations
    candidates = {
        fullfile(getenv('USERPROFILE'), '.kaggle', 'kaggle.json'),
        fullfile(getenv('HOME'),        '.kaggle', 'kaggle.json'),
    };
    for i = 1:numel(candidates)
        if isfile(candidates{i})
            [username, api_key] = parse_kaggle_json(candidates{i});
            fprintf('Loaded credentials from %s\n', candidates{i});
            return;
        end
    end

    % 2. GUI dialog
    if usejava('desktop')
        choice = questdlg( ...
            ['No kaggle.json found automatically.' newline ...
             'How would you like to provide your Kaggle credentials?' newline newline ...
             'Get your API key: https://www.kaggle.com/settings → "Create New Token"'], ...
            'Kaggle Credentials', ...
            'Browse for kaggle.json', 'Enter username & key', 'Browse for kaggle.json');

        if strcmp(choice, 'Browse for kaggle.json')
            [fname, fpath] = uigetfile('*.json', 'Select your kaggle.json file');
            if isequal(fname, 0)
                error('No file selected. Cannot continue without Kaggle credentials.');
            end
            [username, api_key] = parse_kaggle_json(fullfile(fpath, fname));
        else
            answer = inputdlg({'Kaggle Username:', 'Kaggle API Key:'}, ...
                               'Enter Kaggle Credentials', 1, {'',''});
            if isempty(answer) || isempty(strtrim(answer{1})) || isempty(strtrim(answer{2}))
                error('Credentials not provided. Cannot continue.');
            end
            username = strtrim(answer{1});
            api_key  = strtrim(answer{2});
        end
    else
        % 3. Console fallback
        fprintf('\n--- Kaggle Credentials Required ---\n');
        fprintf('Get your API key: https://www.kaggle.com/settings -> Create New Token\n');
        json_path = input('Path to kaggle.json (or press Enter to type manually): ', 's');
        if ~isempty(strtrim(json_path)) && isfile(strtrim(json_path))
            [username, api_key] = parse_kaggle_json(strtrim(json_path));
        else
            username = strtrim(input('Kaggle username: ', 's'));
            api_key  = strtrim(input('Kaggle API key : ', 's'));
        end
        if isempty(username) || isempty(api_key)
            error('Credentials not provided. Cannot continue.');
        end
    end
end

function [username, api_key] = parse_kaggle_json(json_path)
    fid = fopen(json_path, 'r');
    if fid == -1, error('Cannot open %s', json_path); end
    raw = fread(fid, '*char')';
    fclose(fid);
    tok_u = regexp(raw, '"username"\s*:\s*"([^"]+)"', 'tokens', 'once');
    tok_k = regexp(raw, '"key"\s*:\s*"([^"]+)"',      'tokens', 'once');
    if isempty(tok_u) || isempty(tok_k)
        error('Could not parse username/key from %s', json_path);
    end
    username = tok_u{1};
    api_key  = tok_k{1};
end

% ── Kaggle REST API download ───────────────────────────────────────────────────
function kaggle_download(username, api_key, owner, dataset, dest_dir)
    if ~exist(dest_dir, 'dir'), mkdir(dest_dir); end

    % Skip if already populated
    existing = collect_images(dest_dir, {'.jpg','.jpeg','.png','.dcm'});
    if numel(existing) >= 5
        fprintf('  [SKIP] %s/%s already present (%d files).\n', owner, dataset, numel(existing));
        return;
    end

    url  = sprintf('https://www.kaggle.com/api/v1/datasets/download/%s/%s', owner, dataset);
    desc = sprintf('%s/%s', owner, dataset);
    fprintf('  Downloading %s ...\n', desc);

    opts = weboptions('Username', username, 'Password', api_key, ...
                      'Timeout', 600, 'ContentType', 'raw');

    tmp_zip = fullfile(dest_dir, '_tmp.zip');
    try
        websave(tmp_zip, url, opts);
    catch ME
        error('Download failed for %s:\n%s', desc, ME.message);
    end

    % Validate zip magic bytes
    fid    = fopen(tmp_zip, 'rb');
    header = fread(fid, 2, 'uint8')';
    fclose(fid);
    if ~isequal(header, [80 75])
        delete(tmp_zip);
        error(['Downloaded file is not a valid zip.\n' ...
               'Check credentials or accept the dataset licence at:\n' ...
               'https://www.kaggle.com/datasets/%s/%s'], owner, dataset);
    end

    fprintf('  Extracting ...\n');
    unzip(tmp_zip, dest_dir);
    delete(tmp_zip);

    % Unzip any inner zips
    inner = dir(fullfile(dest_dir, '*.zip'));
    for z = 1:numel(inner)
        zp = fullfile(dest_dir, inner(z).name);
        fprintf('  Extracting inner zip: %s ...\n', inner(z).name);
        unzip(zp, dest_dir);
        delete(zp);
    end

    fprintf('  Done -> %s\n', dest_dir);
end

% ── Collect image paths recursively ───────────────────────────────────────────
function paths = collect_images(root_dir, exts)
    % Returns a cell array of full file paths matching given extensions.
    paths = {};
    for e = 1:numel(exts)
        hits = dir(fullfile(root_dir, '**', ['*' exts{e}]));
        % Keep only real files (exclude directories returned by some OS)
        for k = 1:numel(hits)
            full = fullfile(hits(k).folder, hits(k).name);
            if hits(k).bytes > 0 && ~hits(k).isdir
                paths{end+1} = full; %#ok<AGROW>
            end
        end
    end
end

% ── Safe image loader (returns [] on failure) ──────────────────────────────────
function img = safe_load(path, img_size, is_dicom)
    try
        if is_dicom
            raw = double(dicomread(path));
            if ndims(raw) == 4
                raw = raw(:,:,1, floor(size(raw,4)/2)+1);
            elseif ndims(raw) == 3
                raw = raw(:,:,1);
            end
            lo  = min(raw(:)); hi = max(raw(:));
            raw = (raw - lo) / (hi - lo + 1e-8);
        else
            raw = imread(path);
            if size(raw,3) == 3
                raw = rgb2gray(raw);
            elseif size(raw,3) == 4
                raw = rgb2gray(raw(:,:,1:3));  % drop alpha channel
            end
            raw = double(raw) / 255.0;
        end
        img = bilinear_resize(raw, img_size, img_size);
    catch
        img = [];   % skip unreadable files silently
    end
end

% ── Bilinear resize (no imresize) ─────────────────────────────────────────────
function out = bilinear_resize(img, new_h, new_w)
    [h, w] = size(img);
    [gy, gx] = ndgrid(linspace(1,h,new_h), linspace(1,w,new_w));
    y0 = floor(gy); x0 = floor(gx);
    y1 = min(y0+1, h); x1 = min(x0+1, w);
    dy = gy - y0;   dx = gx - x0;
    y0 = max(1, min(y0, h)); x0 = max(1, min(x0, w));
    y1 = max(1, min(y1, h)); x1 = max(1, min(x1, w));
    idx = @(r,c) sub2ind([h w], r, c);
    out = img(idx(y0,x0)).*(1-dy).*(1-dx) ...
        + img(idx(y1,x0)).*   dy .*(1-dx) ...
        + img(idx(y0,x1)).*(1-dy).*   dx  ...
        + img(idx(y1,x1)).*   dy .*   dx;
    out = min(max(out, 0), 1);
end