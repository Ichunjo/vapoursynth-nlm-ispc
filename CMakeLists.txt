cmake_minimum_required(VERSION 3.20.0)

project(vs-nlm-ispc VERSION 0.1 LANGUAGES CXX ISPC)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CLANG_TIDY "Enable clang-tidy analysis during compilation" OFF)

add_library(vsnlm_ispc SHARED source/vsnlm.cpp source/nlm.ispc)

set_target_properties(
  vsnlm_ispc
  PROPERTIES CXX_EXTENSIONS OFF CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXE}")
    set_target_properties(vsnlm_ispc PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  else()
    message(WARNING "ENABLE_CLANG_TIDY is ON, but clang-tidy executable was not found.")
  endif()
endif()

if(NOT DEFINED CMAKE_ISPC_INSTRUCTION_SETS)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set_target_properties(vsnlm_ispc PROPERTIES ISPC_INSTRUCTION_SETS "neon-i32x4")
  else()
    set_target_properties(
      vsnlm_ispc
      PROPERTIES ISPC_INSTRUCTION_SETS "sse2-i32x4;avx1-i32x4;avx2-i32x8;avx512skx-i32x16"
    )
  endif()
endif()

if(MSVC)
  target_compile_options(
    vsnlm_ispc
    PRIVATE
      "$<$<COMPILE_LANGUAGE:ISPC>:-O3>"
      "$<$<COMPILE_LANGUAGE:ISPC>:--math-lib=fast>"
      "$<$<COMPILE_LANGUAGE:ISPC>:--opt=fast-math:aggressive>"
      "$<$<COMPILE_LANGUAGE:ISPC>:--opt=disable-assertions>"
      "$<$<COMPILE_LANGUAGE:CXX>:/O2>"
      "$<$<COMPILE_LANGUAGE:CXX>:/Oi>"
      "$<$<COMPILE_LANGUAGE:CXX>:/fp:fast>"
      "$<$<COMPILE_LANGUAGE:CXX>:/GS->"
  )
else()
  target_compile_options(
    vsnlm_ispc
    PRIVATE
      "$<$<COMPILE_LANGUAGE:ISPC>:-O3>"
      "$<$<COMPILE_LANGUAGE:ISPC>:--math-lib=fast>"
      "$<$<COMPILE_LANGUAGE:ISPC>:--opt=fast-math:aggressive>"
      "$<$<COMPILE_LANGUAGE:ISPC>:--opt=disable-assertions>"
      "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
      "$<$<COMPILE_LANGUAGE:CXX>:-ffast-math>"
  )
endif()

# 1. Look for VapourSynth headers
if(NOT VAPOURSYNTH_INCLUDE_DIR AND VS_INCLUDE_DIR)
  set(VAPOURSYNTH_INCLUDE_DIR "${VS_INCLUDE_DIR}")
endif()

find_path(
  VAPOURSYNTH_INCLUDE_DIR
  NAMES VapourSynth4.h VapourSynth.h
  HINTS
    "${CMAKE_CURRENT_SOURCE_DIR}/vapoursynth/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/vapoursynth"
    "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Lib/site-packages/vapoursynth/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/.venv/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Lib/site-packages/vapoursynth"
  PATH_SUFFIXES vapoursynth/include vapoursynth include
  DOC "Path to VapourSynth headers"
)

if(VAPOURSYNTH_INCLUDE_DIR)
  message(STATUS "Found VapourSynth headers: ${VAPOURSYNTH_INCLUDE_DIR}")
  target_include_directories(vsnlm_ispc PRIVATE ${VAPOURSYNTH_INCLUDE_DIR})
else()
  message(WARNING "VapourSynth headers not found")
endif()

# 2. Version handling
find_package(Git QUIET)

if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE VCS_TAG
  )
  if(VCS_TAG)
    string(STRIP ${VCS_TAG} VCS_TAG)
  endif()
endif()

if(NOT VCS_TAG AND SKBUILD_PROJECT_VERSION)
  set(VCS_TAG "v${SKBUILD_PROJECT_VERSION}")
endif()

if(VCS_TAG)
  message(STATUS "vs-nlm-ispc ${VCS_TAG}")
else()
  message(WARNING "unknown plugin version")
  set(VCS_TAG "unknown")
endif()

configure_file(source/config.h.in config.h)
target_include_directories(vsnlm_ispc PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# 3. Installation into wheel structure
install(TARGETS vsnlm_ispc LIBRARY DESTINATION vapoursynth/plugins RUNTIME DESTINATION vapoursynth/plugins)
