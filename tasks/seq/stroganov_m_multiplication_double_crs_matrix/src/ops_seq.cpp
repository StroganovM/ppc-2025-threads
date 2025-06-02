#include "seq/stroganov_m_multiplication_double_crs_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>

bool stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS::PreProcessingImpl() {
  A_rows_ = static_cast<int>(task_data->inputs_count[0]);
  A_cols_ = static_cast<int>(task_data->inputs_count[1]);
  B_rows_ = static_cast<int>(task_data->inputs_count[2]);
  B_cols_ = static_cast<int>(task_data->inputs_count[3]);

  // Прямое чтение CRS-структур из inputs
  auto* A_values_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* A_columns_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
  auto* A_row_ptr_ptr = reinterpret_cast<int*>(task_data->inputs[2]);

  auto* B_values_ptr = reinterpret_cast<double*>(task_data->inputs[3]);
  auto* B_columns_ptr = reinterpret_cast<int*>(task_data->inputs[4]);
  auto* B_row_ptr_ptr = reinterpret_cast<int*>(task_data->inputs[5]);

  size_t A_nnz = static_cast<size_t>(task_data->inputs_count[4]);
  size_t B_nnz = static_cast<size_t>(task_data->inputs_count[5]);

  A_values_ = std::vector<double>(A_values_ptr, A_values_ptr + A_nnz);
  A_columns_ = std::vector<int>(A_columns_ptr, A_columns_ptr + A_nnz);
  A_row_ptr_ = std::vector<int>(A_row_ptr_ptr, A_row_ptr_ptr + A_rows_ + 1);

  B_values_ = std::vector<double>(B_values_ptr, B_values_ptr + B_nnz);
  B_columns_ = std::vector<int>(B_columns_ptr, B_columns_ptr + B_nnz);
  B_row_ptr_ = std::vector<int>(B_row_ptr_ptr, B_row_ptr_ptr + B_rows_ + 1);

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
