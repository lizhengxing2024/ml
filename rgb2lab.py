from skimage import io, color
rgb_image = io.imread('milkyway.jpg')

lab_image = color.rgb2lab(rgb_image)

# 亮度调整
lab_image[..., 0] = lab_image[..., 0] + 50

rgb_image = color.lab2rgb(lab_image)

io.imshow(rgb_image)
io.show()
