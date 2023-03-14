#include <System.h>
#include <bits/stdint-uintn.h>
#include <opencv2/core/hal/interface.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

//#include "ndarray_converter.h"
#include "sophus/se3.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace ORB_SLAM3 {

PYBIND11_MODULE(orb_slam_pybind, m) {
  //NDArrayConverter::init_numpy();
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
          [](System &self, const std::string& rgb_filename,
             const std::string& depth_filename, const double& timestamp) {
          auto im = cv::imread(rgb_filename, cv::IMREAD_UNCHANGED);
          auto depth = cv::imread(depth_filename, cv::IMREAD_UNCHANGED);
        
            std::cout << "depth image shape,type, channels, elemsize is: "<<
            depth.size()<<std::endl; std::cout << depth.type()<<std::endl;
            std::cout << depth.channels()<<std::endl;
            std::cout << depth.elemSize()<<std::endl;
          //cv::imshow("depth",depth);
          //cv::waitKey(0);
          return self.TrackRGBD(im, depth, timestamp).matrix();
          },
          "im_filename"_a, "depthmap"_a, "timestamp"_a)
      //.def(
      //"TrackRGBD",
      //[](System &self, py::array_t<uint8_t> &image,
      // py::array_t<uint8_t> &depth, const double &timestamp) {
      // NDArrayConverter::toMat(PyObject *o, cv::Mat &m)
      // https://stackoverflow.com/questions/60917800/how-to-get-the-opencv-image-from-python-and-use-it-in-c-in-pybind11
      // cv::Mat im(image.shape(0), image.shape(1),
      // CV_MAKE_TYPE(CV_8U, image.shape(2)),
      // const_cast<uint8_t *>(image.data()), image.strides(0));
      // cv::Mat depthmap(depth.shape(0),depth.shape(1),CV_8U,
      // const_cast<uint8_t *>(depth.data()), depth.strides(0));

      // cv::imshow("depth",depthmap);
      // cv::waitKey(0);

      // std::cout << "color image shape, type, channels, elemsize is: "<<
      // im.size()<<std::endl; std::cout<< im.type()<<std::endl; std::cout  <<
      // im.channels()<<std::endl; std::cout << im.elemSize() <<std::endl;
      // std::cout << "depth image shape,type, channels, elemsize is: "<<
      // depthmap.size()<<std::endl; std::cout << depthmap.type()<<std::endl;
      // std::cout << depthmap.channels()<<std::endl;
      // std::cout << depthmap.elemSize()<<std::endl;
      ////return self.TrackRGBD(im, depthmap, timestamp).matrix();
      //},
      //"im"_a, "depthmap"_a, "timestamp"_a)
      .def("Shutdown", &System::Shutdown);

  py::enum_<System::eSensor>(system, "eSensor")
      .value("MONOCULAR", System::eSensor::MONOCULAR)
      .value("STEREO", System::eSensor::STEREO)
      .value("RGBD", System::RGBD)
      .value("IMU_MONOCULAR", System::eSensor::IMU_MONOCULAR)
      .value("IMU_STEREO", System::eSensor::IMU_STEREO)
      .value("IMU_RGBD", System::eSensor::IMU_RGBD);
  //.export_values();
}
}  // namespace ORB_SLAM3
