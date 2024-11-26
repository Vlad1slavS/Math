def iter(A, C, e=1e-10, max_iterations=1000):
    n = len(C)
    X = [0.0 for _ in range(n)]  # Начальное приближение
    k = 0
    
    while k < max_iterations:
        k += 1
        X_old = X.copy()
        
        for i in range(n):
            sum_ = C[i]
            for j in range(n):
                if i != j:
                    sum_ -= A[i][j] * X[j]
            X[i] = round(sum_ / A[i][i])
        print(X)
        
        # Проверка изменения каждой компоненты решения
        if all(abs(X[i] - X_old[i]) < e for i in range(n)):
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


solution = iter(A, C, e=1e-10)
print("Решение:", solution)

