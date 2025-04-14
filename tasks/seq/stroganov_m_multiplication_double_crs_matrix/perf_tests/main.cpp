#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/stroganov_m_multiplication_double_crs_matrix/include/ops_seq.hpp"

TEST(stroganov_m_multiplication_double_crs_matrix_seq, test_pipeline_run) {
  const int n = 800;

  // CRS-данные для A и B (единичные значения)
  std::vector<double> a_val(n * n, 1.0);
  std::vector<double> b_val(n * n, 1.0);
  std::vector<int> a_col(n * n), b_col(n * n);
  std::vector<int> a_ri(n + 1, 0), b_ri(n + 1, 0);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a_col[i * n + j] = j;
      b_col[i * n + j] = j;
    }
    a_ri[i + 1] = n * (i + 1);
    b_ri[i + 1] = n * (i + 1);
  }

  // Выходные данные
  std::vector<double> out(n * n, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_val.data()), reinterpret_cast<uint8_t*>(a_col.data()),
                       reinterpret_cast<uint8_t*>(a_ri.data()), reinterpret_cast<uint8_t*>(b_val.data()),
                       reinterpret_cast<uint8_t*>(b_col.data()), reinterpret_cast<uint8_t*>(b_ri.data())};

  task_data->inputs_count = {static_cast<std::uint32_t>(n),
                             static_cast<std::uint32_t>(n),
                             static_cast<std::uint32_t>(n),
                             static_cast<std::uint32_t>(n),
                             static_cast<std::uint32_t>(a_val.size()),
                             static_cast<std::uint32_t>(b_val.size())};

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(static_cast<std::uint32_t>(n * n));

  auto task =
      std::make_shared<stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS>(task_data);

  // Производительность
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(dur) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Проверка: A и B из единиц, значит каждый элемент результата = n
  for (int i = 0; i < n * n; ++i) {
    ASSERT_NEAR(out[i], static_cast<double>(n), 1e-9)
        << "Mismatch at index " << i << ": expected " << n << ", got " << out[i];
  }
}

TEST(stroganov_m_multiplication_double_crs_matrix_seq, test_task_run) {
  using namespace stroganov_m_multiplication_double_crs_matrix_seq;

  const int n = 800;

  // CRS-матрицы A и B: единицы на всех позициях
  std::vector<double> a_val(n * n, 1.0);
  std::vector<double> b_val(n * n, 1.0);
  std::vector<int> a_col(n * n), a_ri(n + 1, 0);
  std::vector<int> b_col(n * n), b_ri(n + 1, 0);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a_col[i * n + j] = j;
      b_col[i * n + j] = j;
    }
    a_ri[i + 1] = n * (i + 1);
    b_ri[i + 1] = n * (i + 1);
  }

  std::vector<double> out(n * n, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(a_val.data()), reinterpret_cast<uint8_t*>(a_col.data()),
                       reinterpret_cast<uint8_t*>(a_ri.data()), reinterpret_cast<uint8_t*>(b_val.data()),
                       reinterpret_cast<uint8_t*>(b_col.data()), reinterpret_cast<uint8_t*>(b_ri.data())};

  task_data->inputs_count = {static_cast<std::uint32_t>(n),
                             static_cast<std::uint32_t>(n),
                             static_cast<std::uint32_t>(n),
                             static_cast<std::uint32_t>(n),
                             static_cast<std::uint32_t>(a_val.size()),
                             static_cast<std::uint32_t>(b_val.size())};

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(static_cast<std::uint32_t>(n * n));

  // Создание задачи
  auto test_task =
      std::make_shared<stroganov_m_multiplication_double_crs_matrix_seq::SparseMatrixMultiplicationCRS>(task_data);

  // Настройка замера времени
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(dur) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(test_task);

  perf->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Проверка: каждый элемент результата должен быть равен n
  for (int i = 0; i < n * n; ++i) {
    ASSERT_NEAR(out[i], static_cast<double>(n), 1e-9)
        << "Mismatch at index " << i << ": expected " << n << ", got " << out[i];
  }
}