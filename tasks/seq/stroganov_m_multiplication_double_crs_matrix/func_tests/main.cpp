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
}  // namespace stroganov_m_multiplication_double_crs_matrix_seq

TEST(stroganov_m_sparse_matrix_seq, test_sparse_matmul_identity) {
  // Тест с единичной матрицей 5x5 (разреженная)
  constexpr int kSize = 5;
  constexpr double kTolerance = 1e-9;  // Допустимая погрешность для double

  // Создаем входные данные (единичная матрица в плотном формате)
  std::vector<double> in(kSize * kSize, 0.0);
  std::vector<double> out(kSize * kSize, -1.0);  // Инициализируем не нулями для проверки

  // Заполняем диагональ единицами
  for (int i = 0; i < kSize; i++) {
    in[i * kSize + i] = 1.0;
  }

  // Создаем task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));  // Умножаем на себя
  task_data->inputs_count.emplace_back(kSize);                            // rows A
  task_data->inputs_count.emplace_back(kSize);                            // cols A
  task_data->inputs_count.emplace_back(kSize);                            // rows B
  task_data->inputs_count.emplace_back(kSize);                            // cols B
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(kSize * kSize);

  // Создаем и выполняем задачу
  stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS task(task_data);

  // Проверяем все этапы выполнения
  EXPECT_TRUE(task.Validation()) << "Validation failed";
  EXPECT_TRUE(task.PreProcessing()) << "PreProcessing failed";
  EXPECT_TRUE(task.Run()) << "Run failed";
  EXPECT_TRUE(task.PostProcessing()) << "PostProcessing failed";

  // Проверяем результат (A * I = A)
  for (int i = 0; i < kSize; i++) {
    for (int j = 0; j < kSize; j++) {
      double expected_value = (i == j) ? 1.0 : 0.0;
      EXPECT_NEAR(out[i * kSize + j], expected_value, kTolerance)
          << "Mismatch at position (" << i << ", " << j << "). "
          << "Expected: " << expected_value << ", got: " << out[i * kSize + j];
    }
  }

  // Дополнительная проверка - результат должен быть равен входной матрице
  for (int i = 0; i < kSize * kSize; i++) {
    EXPECT_NEAR(out[i], in[i], kTolerance) << "Full matrix mismatch at index " << i;
  }
}

TEST(stroganov_m_sparse_matrix_seq, test_sparse_matmul_random_sparse_matrices) {
  // Тест со случайными разреженными матрицами
  constexpr int kRowsA = 10;
  constexpr int kColsA = 20;
  constexpr int kRowsB = 20;
  constexpr int kColsB = 15;
  constexpr double kSparsity = 0.1;  // 10% ненулевых элементов
  constexpr double kTolerance = 1e-6;

  // Генерируем случайные разреженные матрицы
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

  std::vector<double> A = generate_sparse_matrix(kRowsA, kColsA, kSparsity);
  std::vector<double> B = generate_sparse_matrix(kRowsB, kColsB, kSparsity);
  std::vector<double> out(kRowsA * kColsB, 0.0);

  // Вычисляем ожидаемый результат наивным методом
  std::vector<double> expected(kRowsA * kColsB, 0.0);
  for (int i = 0; i < kRowsA; i++) {
    for (int j = 0; j < kColsB; j++) {
      for (int k = 0; k < kColsA; k++) {
        expected[i * kColsB + j] += A[i * kColsA + k] * B[k * kColsB + j];
      }
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data->inputs_count.emplace_back(kRowsA);
  task_data->inputs_count.emplace_back(kColsA);
  task_data->inputs_count.emplace_back(kRowsB);
  task_data->inputs_count.emplace_back(kColsB);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(kRowsA * kColsB);

  stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS task(task_data);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  // Проверяем результат
  for (int i = 0; i < kRowsA * kColsB; i++) {
    EXPECT_NEAR(out[i], expected[i], kTolerance)
        << "Mismatch at index " << i << ", expected: " << expected[i] << ", got: " << out[i];
  }
}

TEST(stroganov_m_sparse_matrix_seq, test_sparse_matmul_rectangular_matrices) {
  // Тест с прямоугольными матрицами
  constexpr int kRowsA = 3;
  constexpr int kColsA = 4;
  constexpr int kRowsB = 4;
  constexpr int kColsB = 2;
  constexpr double kTolerance = 1e-9;

  // Создаем тестовые матрицы
  std::vector<double> A = {1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0};

  std::vector<double> B = {0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0};

  std::vector<double> expected = {8.0, 1.0, 6.0, 0.0, 0.0, 12.0};

  std::vector<double> out(kRowsA * kColsB, -1.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data->inputs_count.emplace_back(kRowsA);
  task_data->inputs_count.emplace_back(kColsA);
  task_data->inputs_count.emplace_back(kRowsB);
  task_data->inputs_count.emplace_back(kColsB);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(kRowsA * kColsB);

  stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS task(task_data);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  // Проверяем результат
  for (int i = 0; i < kRowsA; i++) {
    for (int j = 0; j < kColsB; j++) {
      EXPECT_NEAR(out[i * kColsB + j], expected[i * kColsB + j], kTolerance)
          << "Mismatch at position (" << i << ", " << j << ")";
    }
  }
}

TEST(stroganov_m_sparse_matrix_seq, test_sparse_matmul_invalid_dimensions) {
  // Тест с несовместимыми размерами матриц
  constexpr int kRowsA = 3;
  constexpr int kColsA = 4;
  constexpr int kRowsB = 5;  // Несовместимое количество строк
  constexpr int kColsB = 2;

  std::vector<double> A(kRowsA * kColsA, 1.0);
  std::vector<double> B(kRowsB * kColsB, 1.0);
  std::vector<double> out(kRowsA * kColsB, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  task_data->inputs_count.emplace_back(kRowsA);
  task_data->inputs_count.emplace_back(kColsA);
  task_data->inputs_count.emplace_back(kRowsB);
  task_data->inputs_count.emplace_back(kColsB);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(kRowsA * kColsB);

  stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS task(task_data);

  // Проверка должна завершиться неудачей, так как размеры несовместимы
  EXPECT_FALSE(task.Validation());
}

TEST(stroganov_m_sparse_matrix_seq, test_sparse_matmul_zero_matrix) {
  // Тест с нулевой матрицей
  constexpr int kSize = 5;
  constexpr double kTolerance = 1e-9;

  std::vector<double> in(kSize * kSize, 0.0);
  std::vector<double> out(kSize * kSize, -1.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs_count.emplace_back(kSize);
  task_data->inputs_count.emplace_back(kSize);
  task_data->inputs_count.emplace_back(kSize);
  task_data->inputs_count.emplace_back(kSize);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(kSize * kSize);

  stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS task(task_data);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  // Результат умножения нулевой матрицы на себя должен быть нулевой матрицей
  for (int i = 0; i < kSize * kSize; i++) {
    EXPECT_NEAR(out[i], 0.0, kTolerance) << "Non-zero element at index " << i;
  }
}