
set(pcg_sample_sources 
    PCG.cpp
    PCG_Sample.cpp)

add_executable(pcg_sample ${pcg_sample_sources})
target_link_libraries (pcg_sample BaSpaCho testing)
