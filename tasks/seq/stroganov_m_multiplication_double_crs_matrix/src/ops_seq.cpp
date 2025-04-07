#include "seq/stroganov_m_multiplication_double_crs_matrix/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <algorithm>


bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::PreProcessingImpl() {
  A_count_rows_ = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[0]);
  A_rI_ = std::vector<unsigned int>(in_ptr, in_ptr + A_count_rows_);

  A_count_non_zero_ = task_data->inputs_count[1];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[1]);
  col_A_ = std::vector<unsigned int>(in_ptr, in_ptr +  A_count_non_zero_);

  auto *val_ptr = reinterpret_cast<double *>(task_data->inputs[2]);
  A_val_ = std::vector<double>(val_ptr, val_ptr + A_count_non_zero_);

  B_count_rows_ = task_data->inputs_count[3];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[3]);
  B_rI_ = std::vector<unsigned int>(in_ptr, in_ptr + B_count_rows_);

  B_count_non_zero_ = task_data->inputs_count[4];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[4]);
  col_B_ = std::vector<unsigned int>(in_ptr, in_ptr + B_count_non_zero_);

  val_ptr = reinterpret_cast<double *>(task_data->inputs[5]);
  B_val_ = std::vector<double>(val_ptr, val_ptr + B_count_non_zero_);

  unsigned int output_size = task_data->outputs_count[0];
  output_rI_ = std::vector<unsigned int>(output_size, 0);
  return true;
}

bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::ValidationImpl() {
  if (task_data->inputs_count[1] != task_data->inputs_count[2]) {
    return false;
  }
  if (task_data->inputs_count[4] != task_data->inputs_count[5]) {
    return false;
  }
  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }
  if (A_count_rows_ > 0 && !col_A_.empty()) {
    unsigned int max_col_A = *std::max_element(col_A_.begin(), col_A_.end());
    if (max_col_A >= B_count_rows_) {
      return false;
    }
  }
  if (!A_rI_.empty() && A_rI_.back() != A_count_non_zero_) {
    return false;
  }
  if (!B_rI_.empty() && B_rI_.back() != B_count_non_zero_) {
    return false;
  }
  return true;
}

bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::RunImpl() {
  output_rI_.resize(A_count_rows_ + 1, 0);
  std::vector<std::unordered_map<unsigned int, double>> temp_result(A_count_rows_);
  for (unsigned int i = 0; i < A_count_rows_; ++i) {
    if (i >= A_rI_.size() - 1) continue;
    for (unsigned int j = A_rI_[i]; j < A_rI_[i + 1]; ++j) {
      if (j >= col_A_.size()) continue;
      unsigned int col = col_A_[j];
      double val = A_val_[j];
      if (col >= B_rI_.size() - 1) continue;
      for (unsigned int k = B_rI_[col]; k < B_rI_[col + 1]; ++k) {
        if (k >= col_B_.size() || k >= B_val_.size()) continue;
        temp_result[i][col_B_[k]] += val * B_val_[k];
      }
    }
  }
  output_rI_[0] = 0;
  for (unsigned int i = 0; i < A_count_rows_; ++i) {
    output_rI_[i + 1] = output_rI_[i] + static_cast<unsigned int>(temp_result[i].size());
    std::vector<std::pair<unsigned int, double>> sorted_row(temp_result[i].begin(), temp_result[i].end());
    std::sort(sorted_row.begin(), sorted_row.end());
    for (const auto& [col, val] : sorted_row) {
      output_col_.push_back(col);
      output_.push_back(val);
    }
  }
  return true;
}

bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::PostProcessingImpl() {
  std::copy(output_rI_.begin(), output_rI_.end(), reinterpret_cast<unsigned int *>(task_data->outputs[0]));
  std::copy(output_col_.begin(), output_col_.end(), reinterpret_cast<unsigned int *>(task_data->outputs[1]));
  std::copy(output_.begin(), output_.end(), reinterpret_cast<double *>(task_data->outputs[2]));
  return true;
}