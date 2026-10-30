cmake_minimum_required(VERSION 3.18)
project(person_detector_demo CXX)

# PersonDetector(ONNX Runtime 추론)만으로 빌드되는 데모.
# RC카 전체 앱(main.cpp, I2C/PWM/GPS 등 하드웨어 제어)은 이 저장소에 포함되지 않는다.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# ONNX Runtime는 Windows/Linux-x64 공식 배포본에 CMake config가 없어 find_package로
# 못 찾는다 - include/lib 경로를 직접 지정해야 한다.
# ONNXRUNTIME_ROOT 환경변수(또는 -DONNXRUNTIME_ROOT=<path>)로 SDK 압축 해제 경로를
# 알려줄 것. 공식 배포: https://github.com/microsoft/onnxruntime/releases
#   Windows: onnxruntime-win-x64-<ver>.zip
#   RPi4(Linux aarch64): onnxruntime-linux-aarch64-<ver>.tgz (LibTorch와 달리
#   공식 prebuilt가 있음 - 이게 LibTorch에서 ONNX Runtime으로 바꾼 이유)
if(DEFINED ENV{ONNXRUNTIME_ROOT} AND NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()
if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOT가 설정되지 않았습니다. ONNX Runtime C/C++ SDK를 받아서 "
        "-DONNXRUNTIME_ROOT=<path> 또는 환경변수 ONNXRUNTIME_ROOT로 압축 해제 경로를 지정하세요. "
        "(https://github.com/microsoft/onnxruntime/releases)")
endif()

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT}/lib
    NO_DEFAULT_PATH
    REQUIRED
)

add_executable(person_detector_demo
    demo_main.cpp
    PersonDetector.cpp
)

target_include_directories(person_detector_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(person_detector_demo PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
)

if(MSVC)
    # 소스 주석이 UTF-8 한글이라 /utf-8 없으면 MSVC(CP949)에서 경고/오류가 난다
    target_compile_options(person_detector_demo PRIVATE /utf-8 /EHsc)
endif()

if(WIN32)
    # 실행 파일 옆에 onnxruntime DLL을 복사해 PATH 설정 없이 바로 실행되게 한다
    add_custom_command(TARGET person_detector_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll
            ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll
            $<TARGET_FILE_DIR:person_detector_demo>
    )
else()
    # 실행 시점에 libonnxruntime.so를 찾도록 rpath에 심어둔다 - 안 하면 배포 시
    # LD_LIBRARY_PATH를 매번 손으로 맞춰야 한다.
    set_target_properties(person_detector_demo PROPERTIES
        BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
        INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib
    )
endif()
