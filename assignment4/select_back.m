function [im_1d, alpha, im_sub] = select_back(im_in)

% Get image dimensions
[im_h, im_w, ~] = size(im_in);
% Compute #. nodes (pixels)
no_nodes = im_h*im_w;

% Select rectangle
imshow(im_in);
rect = getrect;
rect = int32(rect);

% Set labels
alpha_2d = zeros(im_h, im_w); % 1 for foreground, 0 for background
xmin = max(rect(1), 1);
ymin = max(rect(2), 1);
xmax = min(xmin+rect(3), im_w);
ymax = min(ymin+rect(4), im_h);

alpha_2d(ymin:ymax, xmin:xmax) = 1;

im_sub = im_in(ymin:ymax, xmin:xmax, :);

% Serialize the 2D image into 1D

im_1d = zeros(no_nodes, 3);
alpha = zeros(no_nodes, 1);
for idx = 1:size(im_in, 2)
    im_1d((idx-1)*im_h+1:idx*im_h, :) = im_in(:, idx, :);
    alpha((idx-1)*im_h+1:idx*im_h) = alpha_2d(:, idx);
end
