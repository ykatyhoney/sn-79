file(GLOB matching_sources CONFIGURE_DEPENDS "*.cpp")
add_library(matching STATIC ${matching_sources})
add_library(taosim::matching ALIAS matching)

target_include_directories(matching PUBLIC include)

target_link_libraries(matching
    PUBLIC
        taosim::accounting
        taosim::book
        taosim::decimal
        taosim::event
        taosim::logging
        taosim::message
        taosim::process
        taosim::replay
        taosim::serialization
        taosim::simulation
)
