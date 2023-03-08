find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(orb_slam_pybind MODULE orb_slam3_pybind.cpp)
target_link_libraries(orb_slam_pybind PUBLIC 
ORB_SLAM3 
g2o
DBoW2
)
MESSAGE("..............."${PROJECT_SOURCE_DIR})
install(TARGETS orb_slam_pybind DESTINATION .)
