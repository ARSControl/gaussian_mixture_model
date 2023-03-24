cmake_minimum_required(VERSION 3.0.2)
project(gaussian_mixture_model)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES gaussian_mixture_model
)

# let the compiler search for headers in include folder
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# define a library target
add_library(gaussian_mixture_model
  src/gaussian_mixture_model.cpp
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gaussian_mixture_model ${catkin_LIBRARIES})

add_executable(gmm_node src/gmm_node.cpp)
add_dependencies(gmm_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gmm_node ${catkin_LIBRARIES} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${PROJECT_NAME})

# install
install(
  DIRECTORY include/gaussian_mixture_model
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(
  TARGETS gaussian_mixture_model
  EXPORT gaussian_mixture_model
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
