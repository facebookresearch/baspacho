
add_library(BAL_utils BaAtLarge.cpp)

set(BAL_bench_sources BaAtLargeBench.cpp)
set(BAL_opt_sources BaAtLargeOptimizer.cpp)
set(bench_sources Bench.cpp)

if(HAVE_CHOLMOD)
    list(APPEND BAL_bench_sources
         BenchCholmod.cpp)
    list(APPEND bench_sources
         BenchCholmod.cpp)
endif()

add_executable(BAL_bench ${BAL_bench_sources})
target_link_libraries (BAL_bench BaSpaCho testing BAL_utils)

add_executable(BAL_opt ${BAL_opt_sources})
target_link_libraries (BAL_opt BaSpaCho testing BAL_utils)

add_executable(bench ${bench_sources})
target_link_libraries (bench BaSpaCho testing)

if(HAVE_CHOLMOD)
  target_link_libraries (BAL_bench ${CHOLMOD_LIBRARIES})
  target_link_libraries (bench ${CHOLMOD_LIBRARIES})
endif()