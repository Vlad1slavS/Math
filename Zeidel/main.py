def seidel(A, C, e=1e-100, max_iterations=1000):
    n = len(C)
    X = [0.0] * n  # Начальное приближение
    XX = [0.0] * n
    k = 0
    
    while k < max_iterations:
        k += 1

        for i in range(n):
            XX[i] = C[i]
            for j in range(n):
                if j < i:
                    XX[i] -= A[i][j] * XX[j]
                elif j > i:
                    XX[i] -= A[i][j] * X[j]
            XX[i] /= A[i][i]

        
        # Проверка изменения каждой компоненты решения
        how_many = 0
        for i in range(n):
            if abs(X[i] - XX[i]) < e:
                how_many += 1
            X[i] = round(XX[i])
        print(X)  

        if how_many == n:
            print(f"Метод сошелся за {k} итераций.")
            return X
    
    raise Exception("Метод не сошелся за заданное число итераций")

# Пример ввода данных
A = [
    [9, 2, 1, 5, 4],
    [2, 8, 2, 2, 1],
    [1, 2, 9, 3, 2],
    [5, 2, 3, 8, 1],
    [4, 1, 2, 1, 9]
]

C = [30, 29, 40, 35, 23]


solution = seidel(A, C, e=1e-100)
print("Решение:", solution)

