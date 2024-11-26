import { gaussJordanElimination } from "./GausMethods.js";

var A = [];
var b = [];

var rounded = function (number) {
  return +number.toFixed(3);
};

function roundMatrix(matrix) {
  return matrix.map((row) => row.map((value) => (value ? rounded(value) : 0)));
}

function handleFile(event) {
  const file = event.target.files[0];

  const reader = new FileReader();
  reader.onload = function (e) {
    const data = new Uint8Array(e.target.result);
    const workbook = XLSX.read(data, { type: "array" });

    // Получаем первый лист
    const firstSheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[firstSheetName];

    // Преобразуем данные в массив
    const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

    // Получаем диапазон заполненных ячеек
    const range = XLSX.utils.decode_range(worksheet["!ref"]);
    const lastRow = range.e.r; // Индекс последней строки (0 по индексу)
    const lastColumn = range.e.c; // Индекс последнего заполненного столбца (0 по индексу)

    // Заполняем массив A и b
    for (let row = 0; row <= lastRow; row++) {
      const currentRow = jsonData[row];

      if (row <= lastRow) {
        // Если это не последняя строка, добавляем её в A, включая все столбцы, кроме последнего
        A.push(currentRow.slice(0, lastColumn)); // Добавляем все столбцы до последнего
      }
    }

    // Читаем данные из последнего заполненного столбца
    for (let row = range.s.r; row <= range.e.r; row++) {
      const cellAddress = XLSX.utils.encode_cell({ c: lastColumn, r: row });
      const cell = worksheet[cellAddress];
      b.push(cell ? cell.v : null); // Если ячейка пуста, добавляем null
    }

    const solution = gaussJordanElimination(A, b);
    A = roundMatrix(A); // Округляем элементы матрицы A
    printArray(A);
    printArray2(solution.map((item) => rounded(item)));
  };

  reader.readAsArrayBuffer(file);
}

gaussJordanElimination(A, b);

function printArray(arr) {
  for (let i = 0; i < arr.length; i++) {
    let row = "";
    for (let j = 0; j < arr[i].length; j++) {
      row += rounded(arr[i][j]) + "\t"; // округляем вывод
    }
    console.log(row);
  }
}

function printArray2(solution) {
  console.log("Решение системы уравнений:");
  for (let i = 0; i < solution.length; i++) {
    console.log(`x[${i}] = ${solution[i]}`); // Используем шаблонные строки
  }
}

document
  .getElementById("fileInput")
  .addEventListener("change", handleFile, false);
