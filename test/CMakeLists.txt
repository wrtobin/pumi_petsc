add_executable(fea_skeleton fea_skeleton.cc)
target_include_directories(fea_skeleton PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>)

#set(test_params -np 4 "${CMAKE_CURRENT_BINARY_DIR}/fea_skeleton)
#add_test(NAME fea_skeleton COMMAND "mpirun" ${test_params})

