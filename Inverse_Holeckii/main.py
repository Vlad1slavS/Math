def lu_decomposition(matrix):
    n = len(matrix)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            U[i][j] = matrix[i][j] - sum(L[i][k] * U[k][j] for k in range(i))

        for j in range(i + 1, n):
            L[j][i] = (matrix[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U


def forward_substitution(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y


def backward_substitution(U, y):
    n = len(U)
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = round((y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i])
    return x


def invert_matrix(matrix):
    n = len(matrix)
    L, U = lu_decomposition(matrix)
    inverse_matrix = []

    for i in range(n):
        # Create the i-th column of the identity matrix
        e = [0] * n
        e[i] = 1.0

        # Solve LY = I for Y using forward substitution
        y = forward_substitution(L, e)

        # Solve UX = Y for X using backward substitution
        x = backward_substitution(U, y)

        # Append result as a column in the inverse matrix
        inverse_matrix.append(x)

    # Transpose the result to get the inverse as a correct matrix form
    inverse_matrix = list(map(list, zip(*inverse_matrix)))
    return inverse_matrix


# Пример использования
A = [[1,4,3,6,7],[1,3,3,2,5],[6,4,7,2,4],[3,5,2,9,7],[8,5,9,5,5]]
inverse_A = invert_matrix(A)

print("Обратная матрица для A:")
for row in inverse_A:
    print(row)
