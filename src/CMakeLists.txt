# source files to compile
set(SRCS pumiPETSc.cc)
# any headers which need to be public/installed
#  this does not need to be ALL headers
set(HDRS pumiPETSc.h)

set(LIB_NAME pumi_petsc)

find_package(PETSc REQUIRED)
set(INCLUDE_DIRS ${PETSC_INCLUDE_DIRS})
set(LINK_LIBS ${PETSC_LIBRARIES})

add_library(${LIB_NAME} ${SRCS})
target_include_directories(${LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  ${INCLUDE_DIRS})

target_link_libraries(${LIB_NAME} ${LINK_LIBS})

install(FILES ${HDRS} DESTINATION include)
install(TARGETS ${LIB_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
