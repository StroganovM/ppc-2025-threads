#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace stroganov_m_multiplication_double_crs_matrix_seq {

class SparseMatrixMultiplicationCRS : public ppc::core::Task {
 public:
  explicit SparseMatrixMultiplicationCRS(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  // Данные для матрицы A в формате CRS
  std::vector<double> A_values_;
  std::vector<int> A_columns_;
  std::vector<int> A_row_ptr_;
  int A_rows_, A_cols_;

  // Данные для матрицы B в формате CRS
  std::vector<double> B_values_;
  std::vector<int> B_columns_;
  std::vector<int> B_row_ptr_;
  int B_rows_, B_cols_;

  // Данные для результирующей матрицы в формате CRS
  std::vector<double> res_values_;
  std::vector<int> res_columns_;
  std::vector<int> res_row_ptr_;
  int res_rows_, res_cols_;

};
void convertToCRS(const double* input, int rows, int cols, std::vector<double>& values, std::vector<int>& columns,
                    std::vector<int>& row_ptr, int& out_rows, int& out_cols);

std::vector<double> GetRandomMatrix(unsigned int m, unsigned int n);

}  // namespace stroganov_m_multiplication_double_crs_matrix_seq