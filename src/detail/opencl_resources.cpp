#include "detail/opencl_resources.hpp"

#include <fstream>
#include <sstream>

namespace harmony::detail {

std::string slurp_file(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        throw Error("Could not open kernel source file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace harmony::detail
