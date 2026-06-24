add_library(fp INTERFACE)
add_library(taosim::fp ALIAS fp)

target_include_directories(fp INTERFACE include)