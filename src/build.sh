# Sources of libharmony.a (called by the root build.sh).

HARMONY_SOURCES="
    src/runtime.cpp
    src/detail/format.cpp
    src/detail/load_math.cpp
    src/detail/opencl_resources.cpp
    src/impl/balancing.cpp
    src/impl/communication.cpp
    src/impl/devices.cpp
    src/impl/events.cpp
    src/impl/execution.cpp
    src/impl/halo.cpp
    src/impl/partitioning.cpp
    src/impl/redistribution.cpp
    src/impl/resources.cpp
    src/impl/timing.cpp
"

COMPILE_STATIC_LIB harmony "$HARMONY_SOURCES" "-Iinclude -Isrc"
