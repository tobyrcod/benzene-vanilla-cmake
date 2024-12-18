# Gather all source files
file(GLOB mohex_SRC *.cpp *.hpp)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${top_srcdir}/src
        ${top_srcdir}/src/book
        ${top_srcdir}/src/hex
        ${top_srcdir}/src/util
        ${top_srcdir}/src/commonengine
        ${top_srcdir}/src/simpleplayers
        ${top_srcdir}/src/solver
        ${top_srcdir}/src/smartgame
        ${top_srcdir}/src/gtpengine)

# Create executable
add_executable(mohex ${mohex_SRC})

message(STATUS "DB_LIBRARY value: ${DB_LIBRARY}")

# Link libraries
target_link_libraries(mohex
        hex
        util
        book
        fuego_smartgame
        fuego_gtpengine
        solver
        commonengine
        simpleplayers
        ${Boost_LIBRARIES}
        ${DB_LIBRARY}  # Use the full path to Berkeley DB library
        pthread
)