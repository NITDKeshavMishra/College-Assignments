import numpy as np

a = np.array([[6, 1, 2],
              [1, 4, 0.5],
              [-1, 0.5, -4]], dtype=np.float64)

b = np.array([-2, 1, 0], dtype=np.float64)  # example RHS vector

x = np.zeros(3, dtype=np.float64)

#K = Number of iterations
for k in range(3):
    for i in range(len(a)):
        p = b[i]
        for j in range(len(a)):
            if i != j:
                p -= a[i][j] * x[j]
        x[i] = p / a[i][i]
    print(x)