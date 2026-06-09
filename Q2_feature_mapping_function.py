# ============================================================
# Kernel Trick Practical
# Feature Mapping and Polynomial Kernel
# ============================================================

import numpy as np


# ------------------------------------------------------------
# Transform function
# Φ(x) = (x1x1, x1x2, x1x3,
#         x2x1, x2x2, x2x3,
#         x3x1, x3x2, x3x3)
# ------------------------------------------------------------

def Transform(v):
    transformed = []

    for i in range(len(v)):
        for j in range(len(v)):
            transformed.append(v[i] * v[j])

    return np.array(transformed)


# ------------------------------------------------------------
# Kernel function
# K(x, y) = (<x, y>)^2
# ------------------------------------------------------------

def kernel(x, y):
    dot_product = np.dot(x, y)
    return dot_product ** 2


# ------------------------------------------------------------
# Given vectors
# ------------------------------------------------------------

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])


# ------------------------------------------------------------
# Part A: Transform x and y to higher dimension
# ------------------------------------------------------------

phi_x = Transform(x)
phi_y = Transform(y)

print("Original x:")
print(x)

print("\nOriginal y:")
print(y)

print("\nTransformed Φ(x):")
print(phi_x)

print("\nTransformed Φ(y):")
print(phi_y)

print("\nDimension of original x:", len(x))
print("Dimension of transformed Φ(x):", len(phi_x))


# ------------------------------------------------------------
# Dot product in higher dimension
# ------------------------------------------------------------

higher_dim_dot_product = np.dot(phi_x, phi_y)

print("\nDot product in higher dimension:")
print(higher_dim_dot_product)


# ------------------------------------------------------------
# Part B: Apply kernel K(x, y) = (<x, y>)^2
# ------------------------------------------------------------

kernel_value = kernel(x, y)

print("\nKernel value K(x, y) = (<x, y>)^2:")
print(kernel_value)


# ------------------------------------------------------------
# Compare both results
# ------------------------------------------------------------

print("\nComparison:")
print("Dot product after transformation:", higher_dim_dot_product)
print("Kernel result:", kernel_value)

if higher_dim_dot_product == kernel_value:
    print("\nBoth results are SAME.")
    print("This demonstrates the kernel trick.")
else:
    print("\nResults are different.")