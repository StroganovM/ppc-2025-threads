#include "seq/stroganov_m_multiplication_double_crs_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>

bool stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS::PreProcessingImpl() {
  // Получаем данные о размерах матриц
  A_rows_ = static_cast<int>(task_data->inputs_count[0]);
  A_cols_ = static_cast<int>(task_data->inputs_count[1]);
  B_rows_ = static_cast<int>(task_data->inputs_count[2]);
  B_cols_ = static_cast<int>(task_data->inputs_count[3]);

  // Получаем указатели на входные данные
  auto* matrixA_data = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* matrixB_data = reinterpret_cast<double*>(task_data->inputs[1]);

  // Конвертируем матрицы в формат CRS
  convertToCRS(matrixA_data, A_rows_, A_cols_, A_values_, A_columns_, A_row_ptr_, A_rows_, A_cols_);
  convertToCRS(matrixB_data, B_rows_, B_cols_, B_values_, B_columns_, B_row_ptr_, B_rows_, B_cols_);

  // Инициализируем результирующую матрицу
  res_rows_ = A_rows_;
  res_cols_ = B_cols_;
  res_row_ptr_.resize(res_rows_ + 1, 0);

  return true;
}

bool stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS::ValidationImpl() {
  return task_data->inputs_count[1] == task_data->inputs_count[2];
}

bool stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS::RunImpl() {
  res_values_.clear();
  res_columns_.clear();
  res_row_ptr_.assign(res_rows_ + 1, 0);

  std::vector<double> row_result_(res_cols_, 0.0);

  for (int i = 0; i < res_rows_; ++i) {
    std::fill(row_result_.begin(), row_result_.end(), 0.0);

    for (int k = A_row_ptr_[i]; k < A_row_ptr_[i + 1]; ++k) {
      const int a_col_ = A_columns_[k];
      const double a_val_ = A_values_[k];

      for (int j = B_row_ptr_[a_col_]; j < B_row_ptr_[a_col_ + 1]; ++j) {
        const int b_col_ = B_columns_[j];
        row_result_[b_col_] += a_val_ * B_values_[j];
      }
    }

    // Сохраняем ненулевые элементы
    for (int j = 0; j < res_cols_; ++j) {
      if (std::abs(row_result_[j]) > 1e-9) {
        res_values_.push_back(row_result_[j]);
        res_columns_.push_back(j);
        res_row_ptr_[i + 1]++;
      }
    }
  }

  // Преобразуем row_ptr в кумулятивную сумму
  for (int i = 1; i <= res_rows_; ++i) {
    res_row_ptr_[i] += res_row_ptr_[i - 1];
  }

  return true;
}

bool stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS::PostProcessingImpl() {
  // Проверка на положительные размеры матрицы
  if (res_rows_ <= 0 || res_cols_ <= 0) {
    return false;  // или throw std::invalid_argument("Invalid matrix dimensions");
  }

  // Проверка на переполнение при умножении размеров
  if (res_rows_ > INT_MAX / res_cols_) {
    return false;  // или throw std::overflow_error("Matrix size too large");
  }

  const int output_size_ = res_rows_ * res_cols_;
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);

  // Инициализация выходного массива нулями
  std::fill(out_ptr, out_ptr + output_size_, 0.0);

  // Заполнение ненулевых элементов
  for (int i = 0; i < res_rows_; ++i) {
    for (int j = res_row_ptr_[i]; j < res_row_ptr_[i + 1]; ++j) {
      const int col = res_columns_[j];
      out_ptr[i * res_cols_ + col] = res_values_[j];
    }
  }

  return true;
}

void stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS::convertToCRS(
    const double* input_, int rows_, int cols_, std::vector<double>& values_, std::vector<int>& columns_,
    std::vector<int>& row_ptr_, int& out_rows_, int& out_cols_) {
  out_rows_ = rows_;
  out_cols_ = cols_;
  row_ptr_.resize(rows_ + 1, 0);

  // Первый проход: подсчет ненулевых элементов в каждой строке
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (input_[i * cols_ + j] != 0.0) {
        row_ptr_[i + 1]++;
      }
    }
  }

  // Преобразование в кумулятивные суммы
  for (int i = 1; i <= rows_; ++i) {
    row_ptr_[i] += row_ptr_[i - 1];
  }

  // Второй проход: заполнение values и columns
  values_.resize(row_ptr_[rows_]);
  columns_.resize(row_ptr_[rows_]);

  std::vector<int> current_pos_(rows_, 0);

  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      double val = input_[i * cols_ + j];
      if (val != 0.0) {
        int offset = row_ptr_[i] + current_pos_[i];
        values_[offset] = val;
        columns_[offset] = j;
        current_pos_[i]++;
      }
    }
  }
}