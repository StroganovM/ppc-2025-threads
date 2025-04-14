#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/stroganov_m_multiplication_double_crs_matrix/include/ops_seq.hpp"  // Путь к вашему файлу с задачей

namespace stroganov_m_multiplication_double_crs_matrix_seq {

std::vector<double> GetRandomMatrix(unsigned int m, unsigned int n) {
  if (m * n == 0) {
    throw std::invalid_argument("Can't create matrix with 0 rows or columns");
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distrib(-100.0, 100.0);
  std::vector<double> res(m * n);
  for (unsigned int i = 0; i < m; i++) {
    for (unsigned int j = 0; j < n; j++) {
      res[(i * n) + j] = distrib(gen);
    }
  }
  return res;
}

void convertToCRS(const double* input_, int rows_, int cols_, std::vector<double>& values_, std::vector<int>& columns_,
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
}  // namespace stroganov_m_multiplication_double_crs_matrix_seq

TEST(stroganov_m_sparse_matrix_seq, test_sparse_matmul_identity) {
  constexpr int kSize = 5;
  constexpr double kTolerance = 1e-9;

  // Создаём плотную единичную матрицу
  std::vector<double> dense_identity(kSize * kSize, 0.0);
  for (int i = 0; i < kSize; ++i) {
    dense_identity[i * kSize + i] = 1.0;
  }

  // Конвертируем обе матрицы (A и B) в CRS
  std::vector<double> A_values, B_values;
  std::vector<int> A_columns, A_row_ptr, B_columns, B_row_ptr;
  int A_rows, A_cols, B_rows, B_cols;

  stroganov_m_multiplication_double_crs_matrix_seq::convertToCRS(dense_identity.data(), kSize, kSize, A_values,
                                                                 A_columns, A_row_ptr, A_rows, A_cols);

  stroganov_m_multiplication_double_crs_matrix_seq::convertToCRS(dense_identity.data(), kSize, kSize, B_values,
                                                                 B_columns, B_row_ptr, B_rows, B_cols);

  std::vector<double> out(kSize * kSize, -1.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(A_values.data()), reinterpret_cast<uint8_t*>(A_columns.data()),
                       reinterpret_cast<uint8_t*>(A_row_ptr.data()), reinterpret_cast<uint8_t*>(B_values.data()),
                       reinterpret_cast<uint8_t*>(B_columns.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};

  task_data->inputs_count = {static_cast<std::uint32_t>(A_rows), static_cast<std::uint32_t>(A_cols),
                             static_cast<std::uint32_t>(B_rows), static_cast<std::uint32_t>(B_cols),
                             static_cast<std::uint32_t>(A_values.size()), static_cast<std::uint32_t>(B_values.size())};

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(static_cast<std::uint32_t>(kSize * kSize));

  stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS task(task_data);

  EXPECT_TRUE(task.Validation()) << "Validation failed";
  EXPECT_TRUE(task.PreProcessing()) << "PreProcessing failed";
  EXPECT_TRUE(task.Run()) << "Run failed";
  EXPECT_TRUE(task.PostProcessing()) << "PostProcessing failed";

  for (int i = 0; i < kSize; ++i) {
    for (int j = 0; j < kSize; ++j) {
      double expected = (i == j) ? 1.0 : 0.0;
      EXPECT_NEAR(out[i * kSize + j], expected, kTolerance)
          << "Mismatch at (" << i << ", " << j << "): expected " << expected << ", got " << out[i * kSize + j];
    }
  }

  // Дополнительная проверка: результат == исходной плотной матрице
  for (int i = 0; i < kSize * kSize; ++i) {
    EXPECT_NEAR(out[i], dense_identity[i], kTolerance) << "Mismatch at index " << i;
  }
}

TEST(stroganov_m_sparse_matrix_seq, test_sparse_matmul_random_sparse_matrices) {
  constexpr int kRowsA = 10;
  constexpr int kColsA = 20;
  constexpr int kRowsB = 20;
  constexpr int kColsB = 15;
  constexpr double kSparsity = 0.1;
  constexpr double kTolerance = 1e-6;

  // Генерация случайной разреженной плотной матрицы
  auto generate_sparse_matrix = [](int rows, int cols, double sparsity) {
    std::vector<double> matrix(rows * cols, 0.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> value_dist(-100.0, 100.0);
    std::uniform_real_distribution<double> sparsity_dist(0.0, 1.0);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (sparsity_dist(gen) < sparsity) {
          matrix[i * cols + j] = value_dist(gen);
        }
      }
    }
    return matrix;
  };

  std::vector<double> A_dense = generate_sparse_matrix(kRowsA, kColsA, kSparsity);
  std::vector<double> B_dense = generate_sparse_matrix(kRowsB, kColsB, kSparsity);
  std::vector<double> out(kRowsA * kColsB, 0.0);

  // Ожидаемый результат
  std::vector<double> expected(kRowsA * kColsB, 0.0);
  for (int i = 0; i < kRowsA; ++i) {
    for (int j = 0; j < kColsB; ++j) {
      for (int k = 0; k < kColsA; ++k) {
        expected[i * kColsB + j] += A_dense[i * kColsA + k] * B_dense[k * kColsB + j];
      }
    }
  }

  // Преобразуем A и B в CRS
  std::vector<double> A_values, B_values;
  std::vector<int> A_columns, A_row_ptr, B_columns, B_row_ptr;
  int A_rows, A_cols, B_rows, B_cols;

  stroganov_m_multiplication_double_crs_matrix_seq::convertToCRS(A_dense.data(), kRowsA, kColsA, A_values,
                                                                 A_columns, A_row_ptr, A_rows, A_cols);

  stroganov_m_multiplication_double_crs_matrix_seq::convertToCRS(B_dense.data(), kRowsB, kColsB, B_values,
                                                                 B_columns, B_row_ptr, B_rows, B_cols);

  // Настройка task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(A_values.data()), reinterpret_cast<uint8_t*>(A_columns.data()),
                       reinterpret_cast<uint8_t*>(A_row_ptr.data()), reinterpret_cast<uint8_t*>(B_values.data()),
                       reinterpret_cast<uint8_t*>(B_columns.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};

  task_data->inputs_count = {static_cast<std::uint32_t>(A_rows), static_cast<std::uint32_t>(A_cols),
                             static_cast<std::uint32_t>(B_rows), static_cast<std::uint32_t>(B_cols),
                             static_cast<std::uint32_t>(A_values.size()), static_cast<std::uint32_t>(B_values.size())};

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(static_cast<std::uint32_t>(kRowsA * kColsB));

  stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS task(task_data);

  EXPECT_TRUE(task.Validation()) << "Validation failed";
  EXPECT_TRUE(task.PreProcessing()) << "PreProcessing failed";
  EXPECT_TRUE(task.Run()) << "Run failed";
  EXPECT_TRUE(task.PostProcessing()) << "PostProcessing failed";

  // Сравнение с ожидаемым результатом
  for (int i = 0; i < kRowsA * kColsB; ++i) {
    EXPECT_NEAR(out[i], expected[i], kTolerance)
        << "Mismatch at index " << i << ", expected: " << expected[i] << ", got: " << out[i];
  }
}

TEST(stroganov_m_sparse_matrix_seq, test_sparse_matmul_rectangular_matrices) {
  constexpr int kRowsA = 3;
  constexpr int kColsA = 4;
  constexpr int kRowsB = 4;
  constexpr int kColsB = 2;
  constexpr double kTolerance = 1e-9;

  std::vector<double> A_dense = {1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0};
  std::vector<double> B_dense = {0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0};
  std::vector<double> expected = {8.0, 1.0, 6.0, 0.0, 0.0, 12.0};
  std::vector<double> out(kRowsA * kColsB, -1.0);

  // Конвертация в CRS
  std::vector<double> A_values, B_values;
  std::vector<int> A_columns, A_row_ptr, B_columns, B_row_ptr;
  int A_rows, A_cols, B_rows, B_cols;

  stroganov_m_multiplication_double_crs_matrix_seq::convertToCRS(A_dense.data(), kRowsA, kColsA, A_values,
                                                                 A_columns, A_row_ptr, A_rows, A_cols);
  stroganov_m_multiplication_double_crs_matrix_seq::convertToCRS(B_dense.data(), kRowsB, kColsB, B_values,
                                                                 B_columns, B_row_ptr, B_rows, B_cols);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(A_values.data()), reinterpret_cast<uint8_t*>(A_columns.data()),
                       reinterpret_cast<uint8_t*>(A_row_ptr.data()), reinterpret_cast<uint8_t*>(B_values.data()),
                       reinterpret_cast<uint8_t*>(B_columns.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())};

  task_data->inputs_count = {static_cast<std::uint32_t>(A_rows), static_cast<std::uint32_t>(A_cols),
                             static_cast<std::uint32_t>(B_rows), static_cast<std::uint32_t>(B_cols),
                             static_cast<std::uint32_t>(A_values.size()), static_cast<std::uint32_t>(B_values.size())};

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(static_cast<std::uint32_t>(kRowsA * kColsB));

  stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS task(task_data);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  for (int i = 0; i < kRowsA; ++i) {
    for (int j = 0; j < kColsB; ++j) {
      EXPECT_NEAR(out[i * kColsB + j], expected[i * kColsB + j], kTolerance)
          << "Mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(stroganov_m_sparse_matrix_seq, test_sparse_matmul_invalid_dimensions) {
  constexpr int kRowsA = 3;
  constexpr int kColsA = 4;
  constexpr int kRowsB = 5;  // Несовместимо
  constexpr int kColsB = 2;

  std::vector<double> A_dense(kRowsA * kColsA, 1.0);
  std::vector<double> B_dense(kRowsB * kColsB, 1.0);
  std::vector<double> out(kRowsA * kColsB, 0.0);

  std::vector<double> A_values, B_values;
  std::vector<int> A_columns, A_row_ptr, B_columns, B_row_ptr;
  int A_rows, A_cols, B_rows, B_cols;

  stroganov_m_multiplication_double_crs_matrix_seq::convertToCRS(A_dense.data(), kRowsA, kColsA, A_values,
                                                                 A_columns, A_row_ptr, A_rows, A_cols);
  stroganov_m_multiplication_double_crs_matrix_seq::convertToCRS(B_dense.data(), kRowsB, kColsB, B_values,
                                                                 B_columns, B_row_ptr, B_rows, B_cols);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(A_values.data()), reinterpret_cast<uint8_t*>(A_columns.data()),
                       reinterpret_cast<uint8_t*>(A_row_ptr.data()), reinterpret_cast<uint8_t*>(B_values.data()),
                       reinterpret_cast<uint8_t*>(B_columns.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())
  };

  task_data->inputs_count = {static_cast<std::uint32_t>(A_rows),
                             static_cast<std::uint32_t>(A_cols),
                             static_cast<std::uint32_t>(B_rows),
                             static_cast<std::uint32_t>(B_cols),
                             static_cast<std::uint32_t>(A_values.size()),
                             static_cast<std::uint32_t>(B_values.size())};

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(static_cast<std::uint32_t>(kRowsA * kColsB));

  stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS task(task_data);

  EXPECT_FALSE(task.Validation()) << "Validation should fail for incompatible sizes";
}

TEST(stroganov_m_sparse_matrix_seq, test_sparse_matmul_zero_matrix) {
  constexpr int kSize = 5;
  constexpr double kTolerance = 1e-9;

  std::vector<double> zero_dense(kSize * kSize, 0.0);
  std::vector<double> out(kSize * kSize, -1.0);

  std::vector<double> A_values, B_values;
  std::vector<int> A_columns, A_row_ptr, B_columns, B_row_ptr;
  int A_rows, A_cols, B_rows, B_cols;

  stroganov_m_multiplication_double_crs_matrix_seq::convertToCRS(zero_dense.data(), kSize, kSize, A_values,
                                                                 A_columns, A_row_ptr, A_rows, A_cols);
  stroganov_m_multiplication_double_crs_matrix_seq::convertToCRS(zero_dense.data(), kSize, kSize, B_values,
                                                                 B_columns, B_row_ptr, B_rows, B_cols);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(A_values.data()), reinterpret_cast<uint8_t*>(A_columns.data()),
                       reinterpret_cast<uint8_t*>(A_row_ptr.data()), reinterpret_cast<uint8_t*>(B_values.data()),
                       reinterpret_cast<uint8_t*>(B_columns.data()), reinterpret_cast<uint8_t*>(B_row_ptr.data())
  };

  task_data->inputs_count = {static_cast<std::uint32_t>(A_rows),
                             static_cast<std::uint32_t>(A_cols),
                             static_cast<std::uint32_t>(B_rows),
                             static_cast<std::uint32_t>(B_cols),
                             static_cast<std::uint32_t>(A_values.size()),
                             static_cast<std::uint32_t>(B_values.size())};

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(static_cast<std::uint32_t>(kSize * kSize));

  stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS task(task_data);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  for (int i = 0; i < kSize * kSize; ++i) {
    EXPECT_NEAR(out[i], 0.0, kTolerance) << "Non-zero element at index " << i;
  }
}