//#include "orb_slam3_pybind.h"

#include <System.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "ImuTypes.h"
#include "sophus/se3.hpp"

namespace py = pybind11;
using namespace py::literals;
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
PYBIND11_MODULE(orb_slam_pybind, m) {
  py::class_<System> system(m, "_System",
        "This is the low level C++ bindings, all the methods and "
        "constructor defined within this module (starting with a ``_`` "
        "should not be used. Please reffer to the python Procesor class to "
        "check how to use the API");
  system
      .def(py::init<const std::string &, const std::string &,
                    const System::eSensor, bool, const int,
                    const std::string &>(),
           "strVocFile"_a, "strSettingsFile"_a, "sensor"_a,
           "bUseViewer"_a = true, "initFr"_a = 0, "strSequence"_a = "")
      .def("_ActivateLocalizationMode", &System::ActivateLocalizationMode)
      .def("_DeactivateLocalizationMode", &System::DeactivateLocalizationMode)
      .def("_GetImageScale", &System::GetImageScale)
      .def("_MapChanged", &System::MapChanged)
      .def("_Reset", &System::Reset)
      .def("_ResetActiveMap", &System::ResetActiveMap)
      .def(
          "_SaveKeyFrameTrajectoryEuroC",
          [](System &self, const std::string &filename) {
            self.SaveKeyFrameTrajectoryEuRoC(filename);
          },
          "filename"_a)
      .def(
          "_SaveTrajectoryEuroC",
          [](System &self, const std::string &filename) {
            self.SaveTrajectoryEuRoC(filename);
          },
          "filename"_a)
      .def("_SaveKeyFrameTrajectoryTUM", &System::SaveKeyFrameTrajectoryTUM,
           "filename"_a)
      .def("_SaveTrajectoryKITTI", &System::SaveTrajectoryKITTI, "filename"_a)
      .def("_SaveTrajectoryTUM", &System::SaveTrajectoryTUM, "filename"_a)
      .def("_Shutdown", &System::Shutdown)
      .def("_isFinished", &System::isFinished)
      .def("_isLost", &System::isLost)
      .def("_isShutDown", &System::isShutDown)
      .def(
          "_TrackMonocular",
          [](System &self, py::array &image, const double timestamp,
             py::array_t<double> &vImuMeas, const std::string &filename) {
            cv::Mat im = py_array_to_mat(image);
            std::vector<IMU::Point> vector_imu{};
            if (vImuMeas.size() != 0) {
              vector_imu = py_array_to_vector_imu_points(vImuMeas);
            }
            return self.TrackMonocular(im, timestamp, vector_imu, filename)
                .matrix();
          },
          "im"_a, "timestamp"_a, "vImuMeas"_a = py::array_t<double>(),
          "filename"_a = "")

      .def(
          "_TrackRGBD",
          [](System &self, py::array &image, py::array &depth,
             const double &timestamp, py::array_t<double> vImuMeas,
             const std::string &filename) {
            cv::Mat im = py_array_to_mat(image);
            cv::Mat depthmap = py_array_to_mat(depth);
            std::vector<IMU::Point> vector_imu{};
            if (vImuMeas.size() != 0) {
              vector_imu = py_array_to_vector_imu_points(vImuMeas);
            }

            return self.TrackRGBD(im, depthmap, timestamp, vector_imu, filename)
                .matrix();
          },
          "im"_a, "depthmap"_a, "timestamp"_a,
          "vImuMeas"_a = py::array_t<double>(), "filename"_a = "")

      .def(
          "_TrackStereo",
          [](System &self, py::array &imLeft, py::array &imRight,
             const double &timestamp, py::array_t<double> vImuMeas,
             const std::string &filename) {
            cv::Mat imL = py_array_to_mat(imLeft);
            cv::Mat imR = py_array_to_mat(imRight);
            std::vector<IMU::Point> vector_imu{};
            if (vImuMeas.size() != 0) {
              vector_imu = py_array_to_vector_imu_points(vImuMeas);
            }

            return self.TrackStereo(imL, imR, timestamp, vector_imu, filename)
                .matrix();
          },
          "imLeft"_a, "imRight"_a, "timestamp"_a,
          "vImuMeas"_a = py::array_t<double>(), "filename"_a = "");

  py::enum_<System::eSensor>(system, "eSensor")
      .value("MONOCULAR", System::eSensor::MONOCULAR)
      .value("STEREO", System::eSensor::STEREO)
      .value("RGBD", System::RGBD)
      .value("IMU_MONOCULAR", System::eSensor::IMU_MONOCULAR)
      .value("IMU_STEREO", System::eSensor::IMU_STEREO)
      .value("IMU_RGBD", System::eSensor::IMU_RGBD);

  py::class_<IMU::Point> imu_point(m, "Point");
  imu_point.def(
      py::init<const float &, const float &, const float &, const float &,
               const float &, const float &, const double &>(),
      "acc_x"_a, "acc_y"_a, "acc_z"_a, "ang_vel_x"_a, "ang_vel_y"_a,
      "ang_vel_z"_a, "timestamp"_a);
}
}  // namespace ORB_SLAM3
