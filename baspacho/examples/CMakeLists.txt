
set(opt_sources
    Optimizer.cpp)

add_executable(opt ${opt_sources})
target_link_libraries (opt BaSpaCho testing BAL_utils)
