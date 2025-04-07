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
  constexpr double kTolerance = 1e-9; // Допустимая погрешность для double

  // Создаем входные данные (единичная матрица в плотном формате)
  std::vector<double> in(kSize * kSize, 0.0);
  std::vector<double> out(kSize * kSize, -1.0); // Инициализируем не нулями для проверки

  // Заполняем диагональ единицами
  for (int i = 0; i < kSize; i++) {
    in[i * kSize + i] = 1.0;
  }

  // Создаем task_data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data())); // Умножаем на себя
  task_data->inputs_count.emplace_back(kSize);  // rows A
  task_data->inputs_count.emplace_back(kSize);  // cols A
  task_data->inputs_count.emplace_back(kSize);  // rows B
  task_data->inputs_count.emplace_back(kSize);  // cols B
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
    EXPECT_NEAR(out[i], in[i], kTolerance)
        << "Full matrix mismatch at index " << i;
  }
}