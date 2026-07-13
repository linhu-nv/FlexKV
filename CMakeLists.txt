# =============================================================================
# FlexKV native C++ dependency resolver
# =============================================================================
# FlexKV's Python extension (flexkv.c_ext) is compiled by setup.py via torch's
# cpp_extension, NOT by this file. The sole job of this CMakeLists is to resolve
# every native C++ dependency FlexKV links against -- discovering an installed
# copy when one exists, otherwise fetching and compiling it from source -- and
# to record the resolved include/library/link inputs in a JSON manifest
# (build/flexkv_pyext.json) that setup.py consumes.
#
# CMake is therefore the single source of truth for where xxHash / liburing /
# hiredis / prometheus-cpp live, even when they were fetched from the network.
# NOTHING here assumes a distro package manager pre-installed the dependencies:
# every block is "find locally, else fetch + build".
#
# Dependencies handled here (gated by the same FLEXKV_ENABLE_* switches setup.py
# reads):
#   * xxHash          - always      (csrc/hash.cpp, csrc/cache_utils.h)
#   * liburing        - always      (csrc/transfer_ssd.{h,cpp})
#   * hiredis         - P2P only     (csrc/dist/redis_meta_channel.cpp)
#   * prometheus-cpp  - metrics only (csrc/monitoring/metrics_manager.cpp)
#
# Deps NOT handled here (and why): CUDA driver (-lcuda), pthread/rt (libc),
# cuFile (-lcufile, GDS) and hifs (-lhifs_client_sdk, CFS) are toolchain / driver
# / proprietary libraries with no fetchable source, so setup.py links them
# directly as it always has.
# =============================================================================
cmake_minimum_required(VERSION 3.18)   # FetchContent SOURCE_SUBDIR (xxHash)
project(FlexKVDeps VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Every fetched dependency is statically linked into the shared flexkv.c_ext
# (or is itself a shared lib we bundle), so all object code must be PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# All artefacts (fetched static/shared libs) land under build/lib so setup.py
# has a single, predictable place to find and bundle them.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Staging dir for headers we copy out of fetched source trees (hiredis shim,
# prometheus generated headers). setup.py adds this to the extension's -I set.
set(FLEXKV_INCLUDE_STAGE ${CMAKE_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${FLEXKV_INCLUDE_STAGE})

# Feature switches -- mirror the FLEXKV_ENABLE_* env vars setup.py reads so a
# dependency is only resolved when the feature that needs it is actually built.
option(FLEXKV_ENABLE_MONITORING "Prometheus monitoring (needs prometheus-cpp)" OFF)
option(FLEXKV_ENABLE_P2P        "Distributed P2P/Redis (needs hiredis)"        OFF)

include(FetchContent)
include(ExternalProject)
find_package(PkgConfig QUIET)

# -----------------------------------------------------------------------------
# Manifest accumulators. As each dependency is resolved below we record what
# CMake actually settled on. setup.py (which compiles the extension itself and
# cannot read CMake targets) reads these back from build/flexkv_pyext.json.
# -----------------------------------------------------------------------------
set(FLEXKV_PYEXT_INCLUDE_DIRS  "")   # -I ... for the extension compile
set(FLEXKV_PYEXT_LIBRARY_DIRS  "")   # -L ... for the extension link
set(FLEXKV_PYEXT_LIBRARIES     "")   # -l ... link names
set(FLEXKV_PYEXT_BUILD_TARGETS "")   # cmake targets setup.py must build first
                                     # (fetched deps only; a static archive does
                                     #  not pull them in on its own)
set(FLEXKV_PYEXT_BUNDLE_LIBS   "")   # self-built .so's with no stable home:
                                     # setup.py copies each next to c_ext and
                                     # relies on an $ORIGIN RPATH to load it.

# =============================================================================
# xxHash  (always)  --  <xxhash.h>, link "xxhash"
# =============================================================================
find_package(xxHash 0.8 QUIET CONFIG)
if(NOT xxHash_FOUND)
  find_path(XXHASH_INCLUDE_DIR xxhash.h)
  find_library(XXHASH_LIBRARY NAMES xxhash xxHash)
endif()

if(xxHash_FOUND)
  message(STATUS "xxHash: found via CMake config package")
  get_target_property(_xxh_inc xxHash::xxhash INTERFACE_INCLUDE_DIRECTORIES)
  if(_xxh_inc)
    list(APPEND FLEXKV_PYEXT_INCLUDE_DIRS ${_xxh_inc})
  endif()
  list(APPEND FLEXKV_PYEXT_LIBRARIES xxhash)
elseif(XXHASH_INCLUDE_DIR AND XXHASH_LIBRARY)
  message(STATUS "xxHash: found system copy (${XXHASH_LIBRARY})")
  list(APPEND FLEXKV_PYEXT_INCLUDE_DIRS ${XXHASH_INCLUDE_DIR})
  get_filename_component(_xxh_libdir "${XXHASH_LIBRARY}" DIRECTORY)
  list(APPEND FLEXKV_PYEXT_LIBRARY_DIRS ${_xxh_libdir})
  list(APPEND FLEXKV_PYEXT_LIBRARIES xxhash)
else()
  message(STATUS "xxHash: not found locally -- fetching v0.8.2 from GitHub")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(XXHASH_BUILD_XXHSUM OFF CACHE BOOL "" FORCE)
  # xxHash v0.8.2 still declares cmake_minimum_required(VERSION 2.8.12); CMake
  # >= 4.0 rejects that, so pin a floor policy version for the fetched project.
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
  FetchContent_Declare(
    xxHash
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG        v0.8.2
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  cmake_unofficial
  )
  FetchContent_MakeAvailable(xxHash)
  set_target_properties(xxhash PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  list(APPEND FLEXKV_PYEXT_INCLUDE_DIRS ${xxhash_SOURCE_DIR})
  list(APPEND FLEXKV_PYEXT_LIBRARY_DIRS ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  list(APPEND FLEXKV_PYEXT_LIBRARIES xxhash)
  list(APPEND FLEXKV_PYEXT_BUILD_TARGETS xxhash)
endif()

# =============================================================================
# liburing  (always)  --  <liburing.h>, link "uring"
# =============================================================================
# liburing ships a hand-written ./configure + Makefiles (NO CMake), so it cannot
# be pulled in via FetchContent_MakeAvailable. Discover an installed copy first;
# only if that fails do we fetch it and drive its native build via
# ExternalProject. The fetch path builds the shared library (already PIC) and
# bundles its soname next to the extension.
find_path(URING_INCLUDE_DIR liburing.h)
find_library(URING_LIBRARY NAMES uring)
if(URING_INCLUDE_DIR AND URING_LIBRARY)
  message(STATUS "liburing: found system copy (${URING_LIBRARY})")
  list(APPEND FLEXKV_PYEXT_INCLUDE_DIRS ${URING_INCLUDE_DIR})
  get_filename_component(_uring_libdir "${URING_LIBRARY}" DIRECTORY)
  list(APPEND FLEXKV_PYEXT_LIBRARY_DIRS ${_uring_libdir})
  list(APPEND FLEXKV_PYEXT_LIBRARIES uring)
else()
  message(STATUS "liburing: not found locally -- fetching from GitHub and building")
  set(_uring_root    ${CMAKE_BINARY_DIR}/liburing)
  set(_uring_install ${_uring_root}/install)
  ExternalProject_Add(liburing_ext
    GIT_REPOSITORY      https://github.com/axboe/liburing.git
    GIT_TAG             liburing-2.6
    GIT_SHALLOW         TRUE
    PREFIX              ${_uring_root}
    INSTALL_DIR         ${_uring_install}
    CONFIGURE_COMMAND   <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
    BUILD_IN_SOURCE     1
    BUILD_COMMAND       make -C <SOURCE_DIR>/src ENABLE_SHARED=1
    INSTALL_COMMAND     make -C <SOURCE_DIR>/src install prefix=<INSTALL_DIR>
                            includedir=<INSTALL_DIR>/include libdir=<INSTALL_DIR>/lib
    UPDATE_DISCONNECTED 1
    BUILD_BYPRODUCTS    ${_uring_install}/lib/liburing.so.2
  )
  list(APPEND FLEXKV_PYEXT_INCLUDE_DIRS ${_uring_install}/include)
  list(APPEND FLEXKV_PYEXT_LIBRARY_DIRS ${_uring_install}/lib)
  list(APPEND FLEXKV_PYEXT_LIBRARIES uring)
  list(APPEND FLEXKV_PYEXT_BUILD_TARGETS liburing_ext)
  list(APPEND FLEXKV_PYEXT_BUNDLE_LIBS ${_uring_install}/lib/liburing.so.2)
endif()

# =============================================================================
# hiredis  (P2P only)  --  <hiredis/hiredis.h>, link "hiredis"
# =============================================================================
if(FLEXKV_ENABLE_P2P)
  set(_hiredis_resolved OFF)

  if(PkgConfig_FOUND)
    pkg_check_modules(HIREDIS QUIET IMPORTED_TARGET hiredis)
    if(TARGET PkgConfig::HIREDIS)
      message(STATUS "hiredis: found via pkg-config (${HIREDIS_VERSION})")
      list(APPEND FLEXKV_PYEXT_INCLUDE_DIRS ${HIREDIS_INCLUDE_DIRS})
      list(APPEND FLEXKV_PYEXT_LIBRARY_DIRS ${HIREDIS_LIBRARY_DIRS})
      list(APPEND FLEXKV_PYEXT_LIBRARIES   ${HIREDIS_LIBRARIES})
      set(_hiredis_resolved ON)
    endif()
  endif()

  if(NOT _hiredis_resolved)
    find_path(HIREDIS_INCLUDE_DIR hiredis/hiredis.h)
    find_library(HIREDIS_LIBRARY NAMES hiredis)
    if(HIREDIS_INCLUDE_DIR AND HIREDIS_LIBRARY)
      message(STATUS "hiredis: found system copy (${HIREDIS_LIBRARY})")
      list(APPEND FLEXKV_PYEXT_INCLUDE_DIRS ${HIREDIS_INCLUDE_DIR})
      get_filename_component(_hiredis_libdir "${HIREDIS_LIBRARY}" DIRECTORY)
      list(APPEND FLEXKV_PYEXT_LIBRARY_DIRS ${_hiredis_libdir})
      list(APPEND FLEXKV_PYEXT_LIBRARIES hiredis)
      set(_hiredis_resolved ON)
    endif()
  endif()

  if(NOT _hiredis_resolved)
    message(STATUS "hiredis: not found locally -- fetching v1.2.0 from GitHub")
    set(DISABLE_TESTS ON CACHE BOOL "" FORCE)
    set(ENABLE_SSL OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
    FetchContent_Declare(
      hiredis
      GIT_REPOSITORY https://github.com/redis/hiredis.git
      GIT_TAG        v1.2.0
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(hiredis)
    if(TARGET hiredis_static)
      set_target_properties(hiredis_static PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
      list(APPEND FLEXKV_PYEXT_LIBRARIES hiredis_static)
      list(APPEND FLEXKV_PYEXT_BUILD_TARGETS hiredis_static)
    else()
      set_target_properties(hiredis PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
      list(APPEND FLEXKV_PYEXT_LIBRARIES hiredis)
      list(APPEND FLEXKV_PYEXT_BUILD_TARGETS hiredis)
    endif()
    # hiredis lays its headers flat (hiredis.h at repo root), but our code
    # includes <hiredis/hiredis.h>; stage a copy under build/include/hiredis/.
    file(MAKE_DIRECTORY ${FLEXKV_INCLUDE_STAGE}/hiredis)
    file(GLOB _hiredis_headers ${hiredis_SOURCE_DIR}/*.h)
    file(COPY ${_hiredis_headers} DESTINATION ${FLEXKV_INCLUDE_STAGE}/hiredis)
    list(APPEND FLEXKV_PYEXT_INCLUDE_DIRS ${FLEXKV_INCLUDE_STAGE})
    list(APPEND FLEXKV_PYEXT_LIBRARY_DIRS ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  endif()
endif()

# =============================================================================
# prometheus-cpp  (metrics only)  --  link "prometheus-cpp-pull" + "-core"
# =============================================================================
if(FLEXKV_ENABLE_MONITORING)
  message(STATUS "Prometheus monitoring: ENABLED")
  set(ENABLE_PUSH        OFF CACHE BOOL "" FORCE)
  set(ENABLE_COMPRESSION OFF CACHE BOOL "" FORCE)
  set(ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
  set(GENERATE_PKGCONFIG OFF CACHE BOOL "" FORCE)

  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/prometheus-cpp/CMakeLists.txt)
    message(STATUS "prometheus-cpp: using vendored submodule")
    add_subdirectory(third_party/prometheus-cpp EXCLUDE_FROM_ALL)
    set(_prom_core_inc
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/prometheus-cpp/core/include
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/prometheus-cpp/pull/include)
  else()
    message(STATUS "prometheus-cpp: submodule absent -- fetching v1.2.4 from GitHub")
    FetchContent_Declare(
      prometheus-cpp
      GIT_REPOSITORY https://github.com/jupp0r/prometheus-cpp.git
      GIT_TAG        v1.2.4
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(prometheus-cpp)
    set(_prom_core_inc
      ${prometheus-cpp_SOURCE_DIR}/core/include
      ${prometheus-cpp_SOURCE_DIR}/pull/include)
  endif()

  foreach(_pt core pull)
    if(TARGET ${_pt})
      set_target_properties(${_pt} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
    endif()
  endforeach()
  list(APPEND FLEXKV_PYEXT_INCLUDE_DIRS
    ${_prom_core_inc}
    ${CMAKE_BINARY_DIR}/_deps/prometheus-cpp-build/core/include
    ${CMAKE_BINARY_DIR}/_deps/prometheus-cpp-build/pull/include
    ${CMAKE_BINARY_DIR}/third_party/prometheus-cpp/core/include
    ${CMAKE_BINARY_DIR}/third_party/prometheus-cpp/pull/include)
  list(APPEND FLEXKV_PYEXT_LIBRARY_DIRS ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  list(APPEND FLEXKV_PYEXT_LIBRARIES prometheus-cpp-pull prometheus-cpp-core)
  list(APPEND FLEXKV_PYEXT_BUILD_TARGETS pull core)
else()
  message(STATUS "Prometheus monitoring: DISABLED")
endif()

# =============================================================================
# Emit the dependency manifest for setup.py
# =============================================================================
if(FLEXKV_PYEXT_INCLUDE_DIRS)
  list(REMOVE_DUPLICATES FLEXKV_PYEXT_INCLUDE_DIRS)
endif()
if(FLEXKV_PYEXT_LIBRARY_DIRS)
  list(REMOVE_DUPLICATES FLEXKV_PYEXT_LIBRARY_DIRS)
endif()
if(FLEXKV_PYEXT_LIBRARIES)
  list(REMOVE_DUPLICATES FLEXKV_PYEXT_LIBRARIES)
endif()
if(FLEXKV_PYEXT_BUILD_TARGETS)
  list(REMOVE_DUPLICATES FLEXKV_PYEXT_BUILD_TARGETS)
endif()
if(FLEXKV_PYEXT_BUNDLE_LIBS)
  list(REMOVE_DUPLICATES FLEXKV_PYEXT_BUNDLE_LIBS)
endif()

set(_json_inc "")
foreach(_d ${FLEXKV_PYEXT_INCLUDE_DIRS})
  string(APPEND _json_inc "    \"${_d}\",\n")
endforeach()
set(_json_libdir "")
foreach(_d ${FLEXKV_PYEXT_LIBRARY_DIRS})
  string(APPEND _json_libdir "    \"${_d}\",\n")
endforeach()
set(_json_libs "")
foreach(_l ${FLEXKV_PYEXT_LIBRARIES})
  string(APPEND _json_libs "    \"${_l}\",\n")
endforeach()
set(_json_targets "")
foreach(_t ${FLEXKV_PYEXT_BUILD_TARGETS})
  string(APPEND _json_targets "    \"${_t}\",\n")
endforeach()
set(_json_bundle "")
foreach(_b ${FLEXKV_PYEXT_BUNDLE_LIBS})
  string(APPEND _json_bundle "    \"${_b}\",\n")
endforeach()

string(REGEX REPLACE ",\n$" "\n" _json_inc    "${_json_inc}")
string(REGEX REPLACE ",\n$" "\n" _json_libdir "${_json_libdir}")
string(REGEX REPLACE ",\n$" "\n" _json_libs   "${_json_libs}")
string(REGEX REPLACE ",\n$" "\n" _json_targets "${_json_targets}")
string(REGEX REPLACE ",\n$" "\n" _json_bundle "${_json_bundle}")

file(WRITE ${CMAKE_BINARY_DIR}/flexkv_pyext.json
"{\n  \"include_dirs\": [\n${_json_inc}  ],\n  \"library_dirs\": [\n${_json_libdir}  ],\n  \"libraries\": [\n${_json_libs}  ],\n  \"build_targets\": [\n${_json_targets}  ],\n  \"bundle_libs\": [\n${_json_bundle}  ]\n}\n")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Wrote dependency manifest: ${CMAKE_BINARY_DIR}/flexkv_pyext.json")
