
add_subdirectory(baspacho)

if(BASPACHO_BUILD_TESTS OR BASPACHO_BUILD_EXAMPLES)
  add_subdirectory(testing)
endif()

if(BASPACHO_BUILD_EXAMPLES)
  add_subdirectory(benchmarking)
  add_subdirectory(examples)
endif()

if(BASPACHO_BUILD_TESTS)
  add_subdirectory(tests)
endif()