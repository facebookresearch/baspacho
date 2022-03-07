
add_library(testing
            TestingMatGen.cpp
            TestingUtils.cpp)
set_property(TARGET testing PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries (testing BaSpaCho)
