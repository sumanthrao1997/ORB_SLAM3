#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <opencv2/core/hal/interface.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include "ImuTypes.h"
#include "math.h"

namespace py = pybind11;

namespace ORB_SLAM3 {

std::map<std::string, int> np_cv{
    {py::format_descriptor<uint8_t>::format(), CV_8U},
    {py::format_descriptor<int8_t>::format(), CV_8S},
    {py::format_descriptor<uint16_t>::format(), CV_16U},
    {py::format_descriptor<int16_t>::format(), CV_16S},
    {py::format_descriptor<int32_t>::format(), CV_32S},
    {py::format_descriptor<float>::format(), CV_32F},
    {py::format_descriptor<double>::format(), CV_64F}};

bool is_array_contiguous(const pybind11::array &a) {
  py::ssize_t expected_stride = a.itemsize();
  for (int i = a.ndim() - 1; i >= 0; --i) {
    pybind11::ssize_t current_stride = a.strides()[i];
    if (current_stride != expected_stride) {
      return false;
    }
    expected_stride = expected_stride * a.shape()[i];
  }
  return true;
}

int determine_cv_type(const py::array &np_array) {
  const int ndim = np_array.ndim();
  std::string np_type;
  np_type += np_array.dtype().char_();
  int cv_type = 0;

  if (auto search = np_cv.find(np_type); search != np_cv.end()) {
    cv_type = search->second;
  } else {
    cv_type = -1;
  }
  if (ndim < 2) {
    throw std::invalid_argument(
        "determine_cv_type needs at least two dimensions");
  }
  if (ndim > 3) {
    throw std::invalid_argument(
        "determine_cv_type needs at most three dimensions");
  }
  if (ndim == 2) {
    return CV_MAKETYPE(cv_type, 1);
  }
  return CV_MAKETYPE(cv_type, np_array.shape(2));
}

cv::Mat py_array_to_mat(py::array &np_array) {
  bool is_contiguous = is_array_contiguous(np_array);
  bool is_empty = np_array.size() == 0;
  if (!is_contiguous) {
    throw std::invalid_argument("is not contiguous array; try np.contiguous");
  }
  if (is_empty) {
    throw std::invalid_argument("numpy array is empty");
  }
  int mat_type = determine_cv_type(np_array);
  cv::Mat im(np_array.shape(0), np_array.shape(1), mat_type,
             np_array.mutable_data(), np_array.strides(0));
  return im;
}

std::vector<IMU::Point> py_array_to_vector_imu_points(
    py::array_t<double> &array) {
  // Imu::point size must be 7 (ax, ay, az, vx,vy,vz, t)
  if (array.ndim() != 2 || array.shape(1) != 7) {
    throw py::cast_error();
  }
  std::vector<IMU::Point> imu_vectors{};
  imu_vectors.reserve(array.shape(0));
  auto array_unchecked = array.unchecked<2>();
  for (auto i = 0; i < array_unchecked.shape(0); ++i) {
    auto acc_x = static_cast<float>(array_unchecked(i, 0));
    auto acc_y = static_cast<float>(array_unchecked(i, 1));
    auto acc_z = static_cast<float>(array_unchecked(i, 2));
    auto ang_vel_x = static_cast<float>(array_unchecked(i, 3));
    auto ang_vel_y = static_cast<float>(array_unchecked(i, 4));
    auto ang_vel_z = static_cast<float>(array_unchecked(i, 5));
    double timestamp = array_unchecked(i, 6);
    imu_vectors.emplace_back(IMU::Point(acc_x, acc_y, acc_z, ang_vel_x,
                                        ang_vel_y, ang_vel_z, timestamp));
  }
  return imu_vectors;
}
}  // namespace ORB_SLAM3
