set(TARGET server)
add_executable(${TARGET} server.cpp httplib.h)

include(DefaultTargetOptions)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})
link_directories(${JSONCPP_LIBRARY_DIRS})

# Change json_cpp to jsoncpp
target_link_libraries(${TARGET} PRIVATE common ${JSONCPP_LIBRARIES} whisper ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()
