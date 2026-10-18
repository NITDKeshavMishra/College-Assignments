import numpy as np

def gauss_seidel_3_iterations(a, b):
    x = np.zeros(3, dtype=np.float64)

    for _ in range(3):
        for i in range(len(b)):
            #Removing the inner loop and using vectorized operations for efficiency
            x[i] = (b[i] - a[i] @ x + a[i, i] * x[i]) / a[i, i]

    return x


a = np.array([[6, 1, 2],
              [3, 5, 2],
              [2, 1, 4]], dtype=np.float64)

b = np.array([9, 10, 7], dtype=np.float64)

x = gauss_seidel_3_iterations(a, b)

print("Solution after 3 iterations:", x)
print("Residual:", a @ x - b)