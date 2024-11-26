import math
def print_matrix(matrix):
    res = []  # Инициализация пустого списка для хранения результата
    for i in range(n):
        ans = []  # Инициализация пустого списка для каждой строки
        for j in range(n):
            ans.append(matrix[i][j])
        print(ans)
        res.append(ans)  # Добавляем строку в результирующий список


def LY_filling(matrix):
    n = len(matrix)
    L = [[0] * n for _ in range(n)]
    Y = [[0] * n for _ in range(n)]

    L[0][0] = math.sqrt(matrix[0][0])
    for i in range(1, n):
        L[i][0] = matrix[i][0] / L[0][0]

    for k in range(1, n):
        sqSum = 0
        for m in range(k):
            sqSum += L[k][m] * L[k][m]
        # Проверка на неотрицательность
        if matrix[k][k] - sqSum < 0:
            raise ValueError("Матрица не является положительно определённой.")

        L[k][k] = round(math.sqrt(matrix[k][k] - sqSum), 1)

        if k < n - 1:
            for i in range(k + 1, n):
                pairSum = 0
                for m in range(k):
                    pairSum += L[i][m] * L[i][m]
                L[i][k] = round((matrix[i][k] - pairSum) / L[k][k], 1)

    for i in range(n):
        for j in range(n):
            if j > i:
                Y[i][j] = 0
            elif j == i and L[i][i] != 0:
                Y[i][j] = 1 / L[i][i]
            elif j < i:
                Y[i][j] = 0
                for m in range(j, i):
                    Y[i][j] += L[i][m] * Y[m][j]
                    Y[i][j] /= -L[i][i]

    return L, Y

def inv_filling(Y):
    n = len(Y)
    AInv = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for m in range(n):
                AInv[i][j] += round(Y[m][i] * Y[m][j], 2)
    return AInv

# Инициализация матрицы и вычисление
A = [[6,6,2,5,3],[6,9,5,6,1],[2,5,9,5,1],[5,6,5,8,4],[3,1,1,4,9]]
n = len(A)

L, Y = LY_filling(A)
AInv = inv_filling(Y)

print("Матрица L:")
print_matrix(L)

print("Y:")
print_matrix(Y)

print("Обратная матрица A:")
print_matrix(AInv)

