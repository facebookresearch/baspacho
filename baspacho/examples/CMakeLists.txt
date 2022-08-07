
set(pcg_sample_sources 
    PCG.cpp
    PCG_Sample.cpp)

add_executable(pcg_sample ${pcg_sample_sources})
target_link_libraries (pcg_sample BaSpaCho testing)


set(opt_ba_at_large_sources
    PCG.cpp
    Utils.cpp
    OptimizeBaAtLarge.cpp)

add_executable(opt_ba_at_large ${opt_ba_at_large_sources})
target_link_libraries (opt_ba_at_large BaSpaCho testing BAL_utils)


set(opt_simple_sources 
    PCG.cpp
    Utils.cpp
    OptimizeSimple.cpp)

add_executable(opt_simple ${opt_simple_sources})
target_link_libraries (opt_simple BaSpaCho)


set(opt_comp_model_sources 
    PCG.cpp
    Utils.cpp
    OptimizeCompModel.cpp)

add_executable(opt_comp_model ${opt_comp_model_sources})
target_link_libraries (opt_comp_model BaSpaCho testing)