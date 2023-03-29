#include "orb_slam3_pybind.h"

#include <System.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ImuTypes.h"
#include "sophus/se3.hpp"

namespace py = pybind11;
using namespace py::literals;
namespace ORB_SLAM3 {

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
