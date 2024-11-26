export function gaussElimination(A, b) {
  const n = A.length;

  // Объединение матрицы A и вектора b в расширенную матрицу
  for (let i = 0; i < n; i++) {
    A[i].push(b[i]);
  }

  // Прямой ход (метод Гаусса)
  for (let i = 0; i < n; i++) {
    // Нормализация ведущего элемента
    const pivot = A[i][i];
    for (let j = i + 1; j < n; j++) {
      const ratio = A[j][i] / pivot;
      for (let k = i; k < n + 1; k++) {
        A[j][k] -= ratio * A[i][k];
      }
    }
  }

  // Обратный ход
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = A[i][n] / A[i][i];
    for (let j = i - 1; j >= 0; j--) {
      A[j][n] -= A[j][i] * x[i];
    }
  }

  return x;
}

export function gaussJordanElimination(A, b) {
  const n = A.length;

  // Объединение матрицы A и вектора b в расширенную матрицу
  for (let i = 0; i < n; i++) {
    A[i].push(b[i]);
  }

  // Прямой ход (метод Жордана-Гаусса)
  for (let i = 0; i < n; i++) {
    // Нормализация ведущего элемента
    const pivot = A[i][i];
    for (let k = 0; k < n + 1; k++) {
      A[i][k] /= pivot; // Делим всю строку на ведущий элемент
    }

    for (let j = 0; j < n; j++) {
      if (j !== i) {
        const ratio = A[j][i]; // Получаем коэффициент
        for (let k = 0; k < n + 1; k++) {
          A[j][k] -= ratio * A[i][k]; // Вычитаем ведущую строку, чтобы получить нули
        }
      }
    }
  }

  // Значения переменных будут находиться в правом столбце
  const x = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    x[i] = A[i][n]; // Извлекаем значения переменных с правого столбца
  }

  return x;
}
