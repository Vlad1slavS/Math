def lu_decomposition(A):
    n = len(A)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1  # Единичная матрица для L

        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))

        for j in range(i + 1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U

def forward_substitution(L, b):
    n = len(L)
    y = [0] * n

    for i in range(n):
        y[i] = round(b[i] - sum(L[i][k] * y[k] for k in range(i)), 4)

    return y

def backward_substitution(U, y):
    n = len(U)
    x = [0] * n

    for i in range(n - 1, -1, -1):
        x[i] = round((y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i], 4)

    return x

def solve_linear_system(A, b):
    # LU-разложение
    L, U = lu_decomposition(A)

    # Решение Ly = b
    y = forward_substitution(L, b)

    # Решение Ux = y
    x = backward_substitution(U, y)

    return x

# Пример использования
if __name__ == '__main__':
    # Определите систему уравнений Ax = b
    A = [
        [2, 9, 7, 1, 1],
        [7, 5, 8, 1, 5],
        [3, 1, 1, 1, 3],
        [2, 4, 2, 3, 1],
        [5, 4, 4, 6, 1]
    ]

    b = [20, 26, 9, 12, 20]

    # Решение
    solution = solve_linear_system(A, b)
    print("Матрица A:")
    for x in range(len(A)):
        for y in range(len(A)):
            print(A[x][y], end=" ")
        print(b[x])
    print("Решение системы уравнений:", solution)
