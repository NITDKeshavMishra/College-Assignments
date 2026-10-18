import numpy as np

def is_diagonally_dominant(a):
    n = a.shape[0]

    for i in range(n):
        diagonal = abs(a[i, i])
        off_diagonal_sum = sum(abs(a[i, j]) for j in range(n) if j != i)

        if diagonal <= off_diagonal_sum:
            return False

    return True


a = np.array([[6, 1, 2],
              [3, 5, 2],
              [2, 1, 4]], dtype=np.float64)

print(is_diagonally_dominant(a))