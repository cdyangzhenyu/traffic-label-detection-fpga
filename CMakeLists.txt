cmake_minimum_required(VERSION 3.5.1)
project(edgeboard_samples)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
set(EB_LIB_DIR "${PROJECT_SOURCE_DIR}/lib" )
set(EB_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" )

set(COMMON_SOURCES
        src/blocking_queue.h
        src/camera.hpp
        src/camera.cpp
        src/json.hpp
        )

find_package(OpenCV QUIET COMPONENTS core videoio highgui imgproc imgcodecs ml video)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
#list(APPEND Caffe_LINKER_LIBS ${OpenCV_LIBS})
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})，${OpenCV_LIBS}")
#add_definitions(-DUSE_OPENCV)

include_directories(${EB_INCLUDE_DIR})
LINK_DIRECTORIES(${EB_LIB_DIR})

find_package(PaddleLite QUIET)

if (PADDLELITE_FOUND)
	message("PADDLELITE_FOUND")
	include_directories(${PADDLELITE_INCLUDE_DIR})
	LINK_DIRECTORIES(${PADDLELITE_LIBRARY})

	set(DETECTION_SERVER "detection_server" )
	add_executable(${DETECTION_SERVER} ${COMMON_SOURCES} src/detection_server.cpp)
	target_link_libraries(${DETECTION_SERVER} ${PADDLELITE_LIBRARY})
	target_link_libraries(${DETECTION_SERVER} ${OpenCV_LIBS} ) 
else ()
	message(FATAL_ERROR "PADDLELITE NOT FOUND")
endif ()

# set(IMAGE_DETECTION "image_detection" )
# add_executable(${IMAGE_DETECTION} ${COMMON_SOURCES} src/image_detection.cpp)
# target_link_libraries(${IMAGE_DETECTION} paddle_full_api_shared)
# target_link_libraries(${IMAGE_DETECTION} ${OpenCV_LIBS} ) 

# set(VIDEO_DETECTION "video_detection" )
# add_executable(${VIDEO_DETECTION} ${COMMON_SOURCES} src/video_detection.cpp)
# target_link_libraries(${VIDEO_DETECTION} paddle_full_api_shared)
# target_link_libraries(${VIDEO_DETECTION} ${OpenCV_LIBS} ) 

