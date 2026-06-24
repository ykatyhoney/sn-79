add_library(net STATIC
    net.cpp
    server.cpp
)
add_library(taosim::net ALIAS net)

target_include_directories(net PUBLIC include)

target_link_libraries(net
    PUBLIC
        taosim::serialization
    PRIVATE
        taosim::decimal
        Boost::boost
)
