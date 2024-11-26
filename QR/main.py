import copy
import math

def transpose_matrix(A):
    return [list(row) for row in zip(*A)]

def mat_mult(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def vector_norm(v):
    return math.sqrt(sum([x**2 for x in v]))

def scalar_mult_vector(scalar, v):
    return [scalar * x for x in v]

def vector_subtract(v1, v2):
    return [x - y for x, y in zip(v1, v2)]

def vector_add(v1, v2):
    return [x + y for x, y in zip(v1, v2)]

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def qr_decomposition(A):
    n = len(A)
    m = len(A[0])

    # Инициализируем матрицы Q и R
    Q = [[0] * n for _ in range(n)]
    R = [[0] * m for _ in range(n)]

    # Транспонируем A для удобства работы со столбцами
    A_T = transpose_matrix(A)

    # Список для хранения ортогональных векторов
    u_list = []

    for i in range(n):
        ai = A_T[i]
        ui = ai.copy()

        for j in range(i):
            qj = [row[j] for row in Q]
            proj_coef = dot_product(qj, ai)
            proj = scalar_mult_vector(proj_coef, qj)
            ui = vector_subtract(ui, proj)

        norm_ui = vector_norm(ui)
        qi = [x / norm_ui for x in ui]

        # Добавляем qi в Q
        for k in range(n):
            Q[k][i] = qi[k]

        # Вычисляем элементы R
        for j in range(i, m):
            aj = A_T[j]
            R[i][j] = dot_product(qi, aj)

    return Q, R

def copy_matrix(A):
    return [row[:] for row in A]

def is_converged(A_old, A_new, tol):
    n = len(A_old)
    for i in range(n):
        for j in range(n):
            if abs(A_old[i][j] - A_new[i][j]) > tol:
                return False
    return True

def qr_algorithm(A, max_iterations=1000, tol=1e-10):
    n = len(A)
    Ak = copy_matrix(A)

    for _ in range(max_iterations):
        Q, R = qr_decomposition(Ak)
        Ak_next = mat_mult(R, Q)

        if is_converged(Ak, Ak_next, tol):
            break

        Ak = Ak_next

    # Собственные значения находятся на диагонали матрицы Ak
    eigenvalues = [Ak[i][i] for i in range(n)]
    return eigenvalues

# Пример использования
if __name__ == "__main__":
    # Определим квадратную матрицу A
    A = [
        [5, 4, 2],
        [1, 3, 1],
        [2, 1, 3]
    ]

    eigenvalues = qr_algorithm(A)
    print("Собственные значения матрицы A:")
    for idx, val in enumerate(eigenvalues):
        print(f"λ{idx+1} = {val}")
