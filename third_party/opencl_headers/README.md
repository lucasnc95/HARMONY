# OpenCL C headers

These Khronos OpenCL 3.0 C headers are copied from Debian's
`opencl-c-headers` package, version `3.0~2023.12.14-1`. They are licensed
under Apache 2.0; see [LICENSE](LICENSE). The upstream project is
[KhronosGroup/OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers).

The cluster build uses these headers because its frontend provides an OpenCL
runtime library but lacks the OpenCL development headers in the default
compiler search path. The headers do not provide a GPU driver or an OpenCL
implementation; device availability is checked by the PBS jobs.
