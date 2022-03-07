
set(opt_sources
    Optimizer.cpp)

add_executable(opt ${opt_sources})
target_link_libraries (opt BaSpaCho testing BAL_utils)


set(simple_opt_sources Simple.cpp)

add_executable(simple_opt ${simple_opt_sources})
target_link_libraries (simple_opt BaSpaCho testing)