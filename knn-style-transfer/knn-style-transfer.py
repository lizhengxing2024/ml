import numpy as np
from skimage import io
from skimage.color import rgb2lab, lab2rgb
from sklearn.neighbors import KNeighborsRegressor
import os
import matplotlib.pyplot as plt

# block_size即从像素点向四周扩展几层像素，扩展1层即3*3
block_size = 1


# 读入风格图像, 得到映射 X -> Y
# X: 储存3*3像素格的灰度值
# Y: 储存中心像素格的色彩值
def read_style_image(file_name, size=block_size):
    img = io.imread(file_name)

    img = rgb2lab(img)

    # 取出图片的宽度和高度
    w, h = img.shape[:2]

    # 初始化函数的输出，即两个列表X和Y，表示3*3灰度矩阵（X）到中心点色彩值（Y）的映射
    X = []
    Y = []

    # 用for循环枚举图像中的全部可能的中心点，因为需要从中心点向外扩展1层，所以枚举范围从(0, w)变成(1, w - 1)
    for x in range(size, w - size):
        for y in range(size, h - size):
            # ------------------------------------------------------------------
            # 枚举好中心点位置，即坐标(x, y)后，分别求该中心点对应的3*3灰度矩阵（X）和中心点色彩值（Y）
            #
            # 从中心点向外扩一圈，得到一个3*3矩阵，用[x - size : x + size + 1, y - size : y + size + 1]取到这个矩阵
            # 并在第三个维度上取值为0，即l通道对应的灰度值
            # ------------------------------------------------------------------
            X.append(img[x - size: x + size + 1, y -
                     size: y + size + 1, 0].reshape(-1))

            # 取出中心点对应的色彩值，即在第三个维度上去值为1和2，即ab通道对应的色彩值
            Y.append(img[x, y, 1:])
    return X, Y

# 将所有风格图像的 X -> Y 映射整合为训练数据
def create_dataset(data_dir):
    X = []
    Y = []
    for file in os.listdir(data_dir):
        X0, Y0 = read_style_image(os.path.join(data_dir, file))
        X.extend(X0)
        Y.extend(Y0)

    return X, Y


# 输入内容图像，根据已经建立好的kNN模型，输出色彩风格迁移后的图像。
def rebuild(file_name, size=block_size):
    img = io.imread(file_name)

    img = rgb2lab(img)
    w, h = img.shape[:2]

    # 初始化输出图像对应的张量
    photo = np.zeros([w, h, 3])

    # 取出内容图像的全部3*3灰度矩阵
    X = []
    for x in range(size, w - size):
        for y in range(size, h - size):
            X.append(img[x - size: x + size + 1, y -
                     size: y + size + 1, 0].reshape(-1))

    # 调用kNN模型的predict方法，对于输入的一系列3*3灰度矩阵X，求得其各自对应的色彩的回归值
    # 调用reshape方法将输出的色彩值调整到图片对应的维度
    print("predicting...")
    p_ab = nbrs.predict(X)
    p_ab = p_ab.reshape(w - 2 * size, h - 2 * size, -1)
    print("finish predicting.", p_ab.shape)

    # 根据预测结果，调整内容图片的ab通道值
    for x in range(size, w - size):
        for y in range(size, h - size):
            # 亮度不做调整
            photo[x, y, 0] = img[x, y, 0]
            # 调整ab通道值
            photo[x, y, 1] = p_ab[x - size, y - size, 0]
            photo[x, y, 2] = p_ab[x - size, y - size, 1]

    # 最外圈无法作为中心点，因此不赋值（黑框？）
    photo = photo[size: w - size, size: h - size, :]
    return photo



X, Y = create_dataset("./vangogh-style/")

# 训练KNN回归模型
nbrs = KNeighborsRegressor(n_neighbors=4, weights='distance')
nbrs.fit(X, Y)

# 生成图像
new_photo = rebuild("./input.jpg")

# 保存输出图像
plt.imsave("output.jpg", lab2rgb(new_photo))

# 打印输出图像
plt.imshow(lab2rgb(new_photo))
plt.show()
print(new_photo.shape)
