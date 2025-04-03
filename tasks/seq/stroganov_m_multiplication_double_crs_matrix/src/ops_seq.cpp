#include "seq/stroganov_m_multiplication_double_crs_matrix/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>

bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::PreProcessingImpl() {
  A_count_rows_ = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[0]);
  A_rI_ = std::vector<unsigned int>(in_ptr, in_ptr + A_count_rows_);

  A_count_non_zero_ = task_data->inputs_count[1];
  in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[1]);
  col_A_ = std::vector<unsigned int>(in_ptr, in_ptr + A_count_non_zero_);

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
  output_col_.clear();
  output_.clear();

  return true;
}

bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::ValidationImpl() {
  return task_data->inputs_count[1] == task_data->inputs_count[2] &&
         task_data->inputs_count[4] == task_data->inputs_count[5] &&
         task_data->inputs_count[0] == task_data->outputs_count[0] &&
         *std::max_element(reinterpret_cast<unsigned int *>(task_data->inputs[1]),
                           reinterpret_cast<unsigned int *>(task_data->inputs[1]) + task_data->inputs_count[1]) <=
             task_data->inputs_count[3] - 2;
}

/*
bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::RunImpl() {
  std::vector<unsigned int> tr_i(*std::ranges::max_element(col_B_.begin(), col_B_.end()) + 2, 0);
  for (unsigned int i = 0; i < B_count_non_zero_; ++i) {
    ++tr_i[col_B_[i] + 1];
  }
  for (unsigned int i = 1; i < tr_i.size(); ++i) {
    tr_i[i] += tr_i[i - 1];
  }
  std::vector<unsigned int> tcol(B_count_non_zero_);
  std::vector<double> tval(B_count_non_zero_);
  for (unsigned int i = 0; i < B_count_rows_ - 1; ++i) {
    for (unsigned int j = B_rI_[i]; j < B_rI_[i + 1]; ++j) {
      unsigned int idx = tr_i[col_B_[j]]++;
      tval[idx] = B_val_[j];
      tcol[idx] = i;
    }
  }
  std::rotate(tr_i.rbegin(), tr_i.rbegin() + 1, tr_i.rend());
  tr_i[0] = 0;
  for (unsigned int i = 0; i < A_count_rows_ - 1; ++i) {
    for (unsigned int j = 0; j < tr_i.size() - 1; ++j) {
      double sum = 0;
      unsigned int ai = A_rI_[i], bt = tr_i[j];
      while (ai < A_rI_[i + 1] && bt < tr_i[j + 1]) {
        if (col_A_[ai] == tcol[bt]) {
          sum += A_val_[ai] * tval[bt];
          ++ai;
          ++bt;
        } else if (col_A_[ai] < tcol[bt]) {
          ++ai;
        } else {
          ++bt;
        }
      }
      if (sum != 0) {
        output_.push_back(sum);
        output_col_.push_back(j);
        ++output_rI_[i + 1];
      }
    }
  }
  for (unsigned int i = 1; i < A_count_rows_; ++i) {
    output_rI_[i] += output_rI_[i - 1];
  }
  return true;
}
*/

bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::RunImpl() {
  const unsigned int B_cols = *std::max_element(col_B_.begin(), col_B_.end()) + 1;
  std::vector<unsigned int> tr_i(B_cols + 1, 0);
  for (unsigned int i = 0; i < B_count_non_zero_; ++i) {
    ++tr_i[col_B_[i] + 1];
  }
  for (unsigned int i = 1; i <= B_cols; ++i) {
    tr_i[i] += tr_i[i - 1];
  }
  std::vector<unsigned int> tcol(B_count_non_zero_);
  std::vector<double> tval(B_count_non_zero_);
  for (unsigned int i = 0; i < B_count_rows_; ++i) {
    for (unsigned int j = B_rI_[i]; j < B_rI_[i + 1]; ++j) {
      const unsigned int col = col_B_[j];
      const unsigned int idx = tr_i[col]++;
      tval[idx] = B_val_[j];
      tcol[idx] = i;
    }
  }
  std::rotate(tr_i.begin(), tr_i.end() - 1, tr_i.end());
  tr_i[0] = 0;
  output_rI_.resize(A_count_rows_ + 1, 0);
  for (unsigned int i = 0; i < A_count_rows_; ++i) {
    for (unsigned int j = 0; j < B_cols; ++j) {
      double sum = 0.0;
      unsigned int a_pos = A_rI_[i];
      unsigned int b_pos = tr_i[j];
      const unsigned int a_end = A_rI_[i + 1];
      const unsigned int b_end = tr_i[j + 1];
      while (a_pos < a_end && b_pos < b_end) {
        if (col_A_[a_pos] == tcol[b_pos]) {
          sum += A_val_[a_pos] * tval[b_pos];
          ++a_pos;
          ++b_pos;
        } else if (col_A_[a_pos] < tcol[b_pos]) {
          ++a_pos;
        } else {
          ++b_pos;
        }
      }
      if (sum != 0.0) {
        output_.push_back(sum);
        output_col_.push_back(j);
        ++output_rI_[i + 1];
      }
    }
  }
  for (unsigned int i = 1; i <= A_count_rows_; ++i) {
    output_rI_[i] += output_rI_[i - 1];
  }
  return true;
}

bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::PostProcessingImpl() {
  std::copy(output_rI_.begin(), output_rI_.end(), reinterpret_cast<unsigned int *>(task_data->outputs[0]));
  std::copy(output_col_.begin(), output_col_.end(), reinterpret_cast<unsigned int *>(task_data->outputs[1]));
  std::copy(output_.begin(), output_.end(), reinterpret_cast<double *>(task_data->outputs[2]));
  return true;
}