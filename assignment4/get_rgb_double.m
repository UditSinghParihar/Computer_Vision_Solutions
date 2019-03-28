function rgb = get_rgb_double(im, x, y)

r = im(y, x, 1);
g = im(y, x, 2);
b = im(y, x, 3);

rgb = double([r g b]);
