import numpy as np

a = np.array([1,2,3,4])

# numpy 默认生成的是行向量 
print(a, a.shape) # [1 2 3 4] (4,)

# numpy 中的转置方法对于一维数组是无效的
print(a.T, a.shape) # [1 2 3 4] (4,)

# 一种比较麻烦构造列向量的方法
# [[1]
#  [2]
#  [3]
#  [4]] (4, 1)
a_T = a[:, np.newaxis]
print(a_T, a_T.shape)

# 一种比较简单的构造列向量的方法
b = np.array([[1,2,3,4]])
print(b.T, b.T.shape)

# 向量加法
u = np.array([[1,2,3]]).T
v = np.array([[5,6,7]]).T
print(u+v)

# 向量数乘
u = np.array([[1,2,3]]).T
print(3*u)

# 向量乘法：内积
# 注：使用 numpy 进行内积运算时，传入的参数必须是用一维数组表示的行向量
u = np.array([3,5,2])
v = np.array([1,4,7])
print(np.dot(u, v))

# 向量乘法：外积
# 注：在二维平面中，向量的外积表示两个向量张成的平行四边形的“面积”
# 注：在三维空间中，向量的外积表示 u 和 v 两个向量张成平面的法向量
u = np.array([3,5])
v = np.array([1,4])
print(np.cross(u, v))

# 向量线性组合