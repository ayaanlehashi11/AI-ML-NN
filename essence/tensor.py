import numpy as np

# 1D tensor
# when the tensor has only a one dimension it is called a vector
x = np.array([40 , 32 , 245])
print(x.ndim , x.shape)
# 2D tensor
#when the tensor has 2 Dimension it represents a matrix
x = np.array([[24 , 56]
          [56 , 34]
          [78 , 60]])
print(x.ndim , x.shape)
# 3D tensor
# from here it is apparent that we can use k-dimensional tensors
x = np.array([[[136 , 251 , 148] ,
               [0 , 8 , 14] ,
              [78 , 50 , 12] ]])
print(x.ndim , x.shape)

# tensor operations

X = np.array([[23 , 56],
              [45 , 50]])
Y = np.array([[70 , 23],
              [40 , 30]])

# Addition

Z = X + Y
print(Z)

# Tensor Subtraction

Z = X - Y
print(Z)

#Tensor Hadamard product

Z = X * Y
print(Z)

# Tensor Division

Z = X / Y
print(Z)