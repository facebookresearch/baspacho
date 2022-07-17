
set(pcg_sample_sources 
    PCG.cpp
    PCG_Sample.cpp)

add_executable(pcg_sample ${pcg_sample_sources})
target_link_libraries (pcg_sample BaSpaCho testing)

set(opt_sources
    PCG.cpp
    Optimizer.cpp)

add_executable(opt ${opt_sources})
target_link_libraries (opt BaSpaCho testing BAL_utils)


set(simple_opt_sources 
    PCG.cpp
    Simple.cpp)

add_executable(simple_opt ${simple_opt_sources})
target_link_libraries (simple_opt BaSpaCho testing)

