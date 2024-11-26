def solve_tridiagonal(a, b, c, d):
    n = len(d)

    # Прямой проход
    for i in range(1, n):
        factor = a[i - 1] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]

    # Обратный проход
    x = [0] * n
    x[-1] = d[-1] / b[-1]

    for i in range(n - 2, -1, -1):
        x[i] = round((d[i] - c[i] * x[i + 1]) / b[i], 5)

    return x

# Пример использования
if __name__ == '__main__':
    a = [8, 6, 7, 5]    # элементы под главной диагональю (начиная со второго)
    b = [9, 3, 3, 6, 7] # элементы главной диагонали
    c = [5, 9, 2, 9]    # элементы над главной диагональю (начиная с первого)
    d = [14, 20, 11, 22, 12] # правая часть


    # Решение
    solution = solve_tridiagonal(a, b, c, d)
    print("Решение системы уравнений:", solution)
