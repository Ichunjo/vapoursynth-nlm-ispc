cmake_minimum_required(VERSION 3.20.0)

if(NOT DEFINED SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION "0.0.0")
endif()

project(vs-nlm-ispc VERSION "${SKBUILD_PROJECT_VERSION}" LANGUAGES CXX ISPC)

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

target_compile_options(
  vsnlm_ispc
  PRIVATE
    "$<$<COMPILE_LANGUAGE:ISPC>:-O3;--math-lib=fast;--opt=fast-math:aggressive;--opt=disable-assertions>"
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/O2;/Oi;/fp:fast;/GS->>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3;-ffast-math>>
)

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
else()
  message(WARNING "VapourSynth headers not found")
endif()

# 2. Version handling and include directories
target_compile_definitions(
  vsnlm_ispc
  PRIVATE PLUGIN_VERSION_MAJOR=${PROJECT_VERSION_MAJOR} PLUGIN_VERSION_MINOR=${PROJECT_VERSION_MINOR}
)

target_include_directories(vsnlm_ispc PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${VAPOURSYNTH_INCLUDE_DIR})

# 3. Installation into wheel structure
install(TARGETS vsnlm_ispc LIBRARY DESTINATION vapoursynth/plugins RUNTIME DESTINATION vapoursynth/plugins)
