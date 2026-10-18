import numpy as np


def gauss_seidel_3_iterations(a, b, tolerance=1e-6):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    x = np.zeros(len(b), dtype=np.float64)

    # Reference solution for calculating true error.
    x_reference = np.linalg.solve(a, b)

    def norm(v):
        return np.linalg.norm(v, ord=np.inf)

    def relative_error(error, reference_norm):
        if reference_norm == 0:
            return 0.0 if error == 0 else np.inf
        return error / reference_norm

    diagonal = np.abs(np.diag(a))
    off_diagonal_sum = np.sum(np.abs(a), axis=1) - diagonal

    print(
        "Strictly diagonally dominant:",
        np.all(diagonal > off_diagonal_sum)
    )

    print(
        f"\n{'Iter':>4} {'Approximate solution':>38} "
        f"{'Relative error':>16} {'Residual norm':>16}"
    )

    # Always perform exactly three iterations.
    for k in range(3):
        x_old = x.copy()

        for i in range(len(b)):
            x[i] = (
                b[i] - a[i] @ x + a[i, i] * x[i]
            ) / a[i, i]

        absolute_change = norm(x - x_old)
        relative_change = relative_error(absolute_change, norm(x))

        residual = a @ x - b
        residual_norm = norm(residual)

        x_text = np.array2string(x, precision=8, floatmode="fixed")

        print(
            f"{k + 1:4d} {x_text:>38} "
            f"{relative_change:16.6e} {residual_norm:16.6e}"
        )

    # Check the final result, but do not perform further iterations.
    converged = (
        relative_change < tolerance
        and residual_norm < tolerance
    )

    absolute_true_error = np.abs(x_reference - x)
    true_error_norm = norm(x_reference - x)
    relative_true_error = relative_error(
        true_error_norm, norm(x_reference)
    )

    print("\n--- Results after exactly 3 iterations ---")
    print("Approximate solution:", x)
    print("Reference solution:  ", x_reference)
    print("Tolerance satisfied:", converged)

    print("\n--- Approximate error ---")
    print(f"Absolute change:            {absolute_change:.8e}")
    print(f"Relative approximate error: {relative_change:.8e}")
    print(f"Percentage relative error:  {100 * relative_change:.8f}%")

    print("\n--- Residual ---")
    print("Residual vector:", residual)
    print(f"Residual infinity norm:     {residual_norm:.8e}")

    print("\n--- True error against the reference ---")
    print("Absolute error per variable:", absolute_true_error)
    print(f"True error infinity norm:   {true_error_norm:.8e}")
    print(f"Relative true error:        {relative_true_error:.8e}")
    print(f"Percentage true error:      {100 * relative_true_error:.8f}%")

    return x


a = np.array([
    [6, 1, 2],
    [1, 4, 0.5],
    [-1, 0.5, -4]
], dtype=np.float64)

b = np.array([-2, 1, 0], dtype=np.float64)

np.set_printoptions(precision=10)

x = gauss_seidel_3_iterations(a, b)