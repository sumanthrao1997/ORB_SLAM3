#include <System.h>
#include <bits/stdint-uintn.h>
#include <opencv2/core/hal/interface.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

//#include "ndarray_converter.h"
#include "ImuTypes.h"
#include "orb_slam3_pybind.h"
#include "sophus/se3.hpp"

namespace py = pybind11;
using namespace py::literals;
namespace ORB_SLAM3 {

PYBIND11_MODULE(orb_slam_pybind, m) {
  py::class_<System> system(m, "System");
  system
      .def(py::init<const std::string &, const std::string &,
                    const System::eSensor, bool, const int,
                    const std::string &>(),
           "strVocFile"_a, "strSettingsFile"_a, "sensor"_a,
           "bUseViewer"_a = true, "initFr"_a = 0, "strSequence"_a = "")
      .def("GetImageScale", &System::GetImageScale)
      .def("SaveTrajectoryTUM", &System::SaveTrajectoryTUM, "filename"_a)
      .def("SaveKeyFrameTrajectoryTUM", &System::SaveKeyFrameTrajectoryTUM,
           "filename"_a)
      .def(
          "TrackRGBD",
          [](System &self, py::array &image, py::array &depth,
             const double &timestamp) {
            cv::Mat im = py_array_to_mat(image);
            cv::Mat depthmap = py_array_to_mat(depth);
            return self.TrackRGBD(im, depthmap, timestamp).matrix();
          },
          "im"_a, "depthmap"_a, "timestamp"_a)

      .def(
          "TrackRGBD",
          [](System &self, py::array &image, py::array &depth,
             const double &timestamp, py::array_t<double> vImuMeas,
             std::string filename) {
            std::vector<IMU::Point> vector_imu{};
            if (!vImuMeas.is_none()) {
              vector_imu = py_array_to_vector_imu_points(vImuMeas);
            }

            cv::Mat im = py_array_to_mat(image);
            cv::Mat depthmap = py_array_to_mat(depth);
            return self.TrackRGBD(im, depthmap, timestamp, vector_imu, filename)
                .matrix();
          },
          "im"_a, "depthmap"_a, "timestamp"_a,
          "vImuMeas"_a = py::array_t<double>(), "filename"_a = "")

      .def(
          "TrackRGBD",
          [](System &self, py::array &image, py::array &depth,
             const double &timestamp, std::string filename) {
            std::vector<IMU::Point> vector_imu{};

            cv::Mat im = py_array_to_mat(image);
            cv::Mat depthmap = py_array_to_mat(depth);

            return self.TrackRGBD(im, depthmap, timestamp, vector_imu, filename)
                .matrix();
          },
          "im"_a, "depthmap"_a, "timestamp"_a, "filename"_a)

      .def("Shutdown", &System::Shutdown);

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
