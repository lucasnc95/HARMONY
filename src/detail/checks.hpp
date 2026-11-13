// Error checking for OpenCL and MPI calls. Failures throw harmony::Error.

#ifndef HARMONY_DETAIL_CHECKS_HPP
#define HARMONY_DETAIL_CHECKS_HPP

#include "harmony/types.hpp"

#include <CL/cl.h>
#include <mpi.h>

#include <sstream>
#include <string>

namespace harmony::detail {

inline void check_mpi(int code, const char* what) {
    if (code == MPI_SUCCESS) return;
    char err_string[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(code, err_string, &len);
    std::ostringstream oss;
    oss << what << " failed: " << std::string(err_string, len);
    throw Error(oss.str());
}

inline void check_cl(cl_int code, const char* what) {
    if (code == CL_SUCCESS) return;
    std::ostringstream oss;
    oss << "OpenCL failure at " << what << " code=" << code;
    throw Error(oss.str());
}

} // namespace harmony::detail

#endif // HARMONY_DETAIL_CHECKS_HPP
