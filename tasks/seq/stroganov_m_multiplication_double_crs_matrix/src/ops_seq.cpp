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
  /*
  return task_data->inputs_count[1] == task_data->inputs_count[2] &&
         task_data->inputs_count[4] == task_data->inputs_count[5] &&
         task_data->inputs_count[0] == task_data->outputs_count[0] &&
         *std::max_element(reinterpret_cast<unsigned int *>(task_data->inputs[1]),
                           reinterpret_cast<unsigned int *>(task_data->inputs[1]) + task_data->inputs_count[1]) <=
             task_data->inputs_count[3] - 2;
  */
}

/*
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
    for (const auto &[col, val] : sorted_row) {
      output_col_.push_back(col);
      output_.push_back(val);
    }
  }
  return true;
}


bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::RunImpl() {
  output_rI_.resize(A_count_rows_ + 1, 0);
  std::vector<std::unordered_map<unsigned int, double>> temp_result(A_count_rows_);
  for (unsigned int i = 0; i < A_count_rows_; ++i) {
    for (unsigned int j = A_rI_[i]; j < A_rI_[i + 1]; ++j) {
      unsigned int col_A = col_A_[j];
      double val_A = A_val_[j];

      for (unsigned int k = B_rI_[col_A]; k < B_rI_[col_A + 1]; ++k) {
        unsigned int col_B = col_B_[k];
        double val_B = B_val_[k];
        temp_result[i][col_B] += val_A * val_B;
      }
    }
  }
  output_rI_[0] = 0;
  for (unsigned int i = 0; i < A_count_rows_; ++i) {
    output_rI_[i + 1] = output_rI_[i] + temp_result[i].size();
    std::vector<std::pair<unsigned int, double>> sorted_row;
    for (const auto &elem : temp_result[i]) {
      if (elem.second != 0.0) {
        sorted_row.emplace_back(elem);
      }
    }
    std::sort(sorted_row.begin(), sorted_row.end());
    for (const auto &elem : sorted_row) {
      output_col_.push_back(elem.first);
      output_.push_back(elem.second);
    }
  }

  return true;
}


bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::RunImpl() {
  output_rI_.clear();
  output_col_.clear();
  output_.clear();
  output_rI_.resize(A_count_rows_ + 1, 0);
  output_col_.reserve(A_count_rows_ * 10);
  output_.reserve(A_count_rows_ * 10);
  std::vector<std::unordered_map<unsigned int, double>> temp_result(A_count_rows_);
  for (unsigned int i = 0; i < A_count_rows_; ++i) {
    if (i >= A_rI_.size() - 1) continue;
    for (unsigned int j = A_rI_[i]; j < A_rI_[i + 1]; ++j) {
      if (j >= col_A_.size() || j >= A_val_.size()) continue;
      unsigned int col = col_A_[j];
      double val = A_val_[j];
      if (col >= B_rI_.size() - 1) continue;
      for (unsigned int k = B_rI_[col]; k < B_rI_[col + 1]; ++k) {
        if (k >= col_B_.size() || k >= B_val_.size()) continue;
        unsigned int col_b = col_B_[k];
        temp_result[i][col_b] += val * B_val_[k];
      }
    }
  }
  output_rI_[0] = 0;
  for (unsigned int i = 0; i < A_count_rows_; ++i) {
    output_rI_[i + 1] = output_rI_[i] + static_cast<unsigned int>(temp_result[i].size());
    std::vector<std::pair<unsigned int, double>> sorted_row(temp_result[i].begin(), temp_result[i].end());
    std::sort(sorted_row.begin(), sorted_row.end());
    for (const auto &[col, val] : sorted_row) {
      output_col_.push_back(col);
      output_.push_back(val);
    }
  }
  return true;
}
*/

bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::RunImpl() {
  std::vector<unsigned int> tr_i(*std::ranges::max_element(col_B_.begin(), col_B_.end()) + 2, 0);
  unsigned int i = 0;
  unsigned int j = 0;
  for (i = 0; i < B_count_non_zero_; i++) {
    tr_i[col_B_[i] + 1]++;
  }
  for (i = 1; i < tr_i.size(); i++) {
    tr_i[i] += tr_i[i - 1];
  }
  std::vector<unsigned int> tcol(B_count_non_zero_, 0);
  std::vector<double> tval(B_count_non_zero_, 0);
  for (i = 0; i < B_count_rows_ - 1; i++) {
    for (j = B_rI_[i]; j < B_rI_[i + 1]; j++) {
      tval[tr_i[col_B_[j]]] = B_val_[j];
      tcol[tr_i[col_B_[j]]] = i;
      tr_i[col_B_[j]]++;
    }
  }
  for (i = tr_i.size() - 1; i > 0; i--) {
    tr_i[i] = tr_i[i - 1];
  }
  tr_i[0] = 0;
  unsigned int ai = 0;
  unsigned int bt = 0;
  double sum = 0;
  for (i = 0; i < A_count_rows_ - 1; i++) {
    for (j = 0; j < tr_i.size() - 1; j++) {
      sum = 0;
      ai = A_rI_[i];
      bt = tr_i[j];
      while (ai < A_rI_[i + 1] && bt < tr_i[j + 1]) {
        if (col_A_[ai] == tcol[bt]) {
          sum += A_val_[ai] * tval[bt];
          ai++;
          bt++;
        } else if (col_A_[ai] < tcol[bt]) {
          ai++;
        } else {
          bt++;
        }
      }
      if (sum != 0) {
        output_.push_back(sum);
        output_col_.push_back(j);
        output_rI_[i + 1]++;
      }
    }
  }
  for (i = 1; i < A_count_rows_; i++) {
    output_rI_[i] += output_rI_[i - 1];
  }
  return true;
}

/*
bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::PostProcessingImpl() {
  std::copy(output_rI_.begin(), output_rI_.end(), reinterpret_cast<unsigned int *>(task_data->outputs[0]));
  std::copy(output_col_.begin(), output_col_.end(), reinterpret_cast<unsigned int *>(task_data->outputs[1]));
  std::copy(output_.begin(), output_.end(), reinterpret_cast<double *>(task_data->outputs[2]));
  return true;
}


bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::PostProcessingImpl() {
  // Получаем указатели на выходные буферы
  auto *out_rI_ptr = reinterpret_cast<unsigned int *>(task_data->outputs[0]);
  auto *out_col_ptr = reinterpret_cast<unsigned int *>(task_data->outputs[1]);
  auto *out_val_ptr = reinterpret_cast<double *>(task_data->outputs[2]);

  // Явно приводим типы к size_t перед сравнением
  size_t rI_size = std::min(output_rI_.size(), static_cast<size_t>(task_data->outputs_count[0]));
  if (rI_size > 0) {
    std::copy(output_rI_.begin(), output_rI_.begin() + rI_size, out_rI_ptr);
  }

  size_t col_size = std::min(output_col_.size(), static_cast<size_t>(task_data->outputs_count[1]));
  if (col_size > 0) {
    std::copy(output_col_.begin(), output_col_.begin() + col_size, out_col_ptr);
  }

  size_t val_size = std::min(output_.size(), static_cast<size_t>(task_data->outputs_count[2]));
  if (val_size > 0) {
    std::copy(output_.begin(), output_.begin() + val_size, out_val_ptr);
  }

  return true;
}

*/

bool stroganov_m_multiplication_double_crs_matrix_seq::MuitiplicationCrsMatrixSeq::PostProcessingImpl() {
  for (size_t i = 0; i < output_rI_.size(); i++) {
    reinterpret_cast<unsigned int *>(task_data->outputs[0])[i] = output_rI_[i];
  }
  for (size_t i = 0; i < output_col_.size(); i++) {
    reinterpret_cast<unsigned int *>(task_data->outputs[1])[i] = output_col_[i];
    reinterpret_cast<double *>(task_data->outputs[2])[i] = output_[i];
  }
  task_data->outputs_count.emplace_back(output_col_.size());
  task_data->outputs_count.emplace_back(output_.size());
  return true;
}
