#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace stroganov_m_multiplication_double_crs_matrix_seq {

class MuitiplicationCrsMatrixSeq : public ppc::core::Task {
 public:
  explicit MuitiplicationCrsMatrixSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_val_, B_val_, output_;
  std::vector<unsigned int> col_A_, A_rI_, col_B_, B_rI_, output_col_, output_rI_;
  unsigned int A_count_rows_, A_count_non_zero_, B_count_rows_, B_count_non_zero_;
};

std::vector<double> GetRandomMatrix(unsigned int m, unsigned int n);

void MakeCRS(std::vector<unsigned int> &r_i, std::vector<unsigned int> &col, std::vector<double> &val,
             const std::vector<double> &src, unsigned int m, unsigned int n);

void MatrixMultiplication(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                          unsigned int m, unsigned int n, unsigned int p);

}  // stroganov_m_multiplication_double_crs_matrix_seq