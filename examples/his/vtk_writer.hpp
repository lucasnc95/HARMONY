// Minimal writer of legacy VTK files (STRUCTURED_POINTS, binary) for a
// regular nx x ny x nz grid. ParaView and VisIt open them directly, and a
// numbered series (name_0000.vtk, name_0001.vtk, ...) is loaded as a time
// series.
//
//   vtk::StructuredPoints out("tissue_0003.vtk", "t = 3 ms", nx, ny, nz, dx);
//   out.add("Vm", vm);            // one value per point, x fastest, then y, then z
//   out.add("device", owner);     // int values
//   out.close();
//
// The legacy binary format is big-endian; values are converted when written.

#ifndef HARMONY_EXAMPLES_VTK_WRITER_HPP
#define HARMONY_EXAMPLES_VTK_WRITER_HPP

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vtk {

class StructuredPoints {
public:
    StructuredPoints(const std::string& path, const std::string& title, int nx, int ny, int nz, double spacing)
        : out_(path, std::ios::binary), points_(static_cast<std::size_t>(nx) * ny * nz) {
        if (!out_) throw std::runtime_error("cannot write " + path);
        out_ << "# vtk DataFile Version 3.0\n"
             << title.substr(0, 255) << "\n"
             << "BINARY\n"
             << "DATASET STRUCTURED_POINTS\n"
             << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n"
             << "ORIGIN 0 0 0\n"
             << "SPACING " << spacing << " " << spacing << " " << spacing << "\n"
             << "POINT_DATA " << points_ << "\n";
    }

    // Scalar field written as float32 (any arithmetic input type).
    template <typename T>
    void add(const std::string& name, const std::vector<T>& values) {
        check(values.size());
        out_ << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
        std::vector<char> buf(values.size() * 4);
        for (std::size_t i = 0; i < values.size(); ++i) put_big_endian(&buf[i * 4], static_cast<float>(values[i]));
        out_.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        out_ << "\n";
    }

    // Scalar field written as int32 (e.g. the device that owns each point).
    void add(const std::string& name, const std::vector<int>& values) {
        check(values.size());
        out_ << "SCALARS " << name << " int 1\nLOOKUP_TABLE default\n";
        std::vector<char> buf(values.size() * 4);
        for (std::size_t i = 0; i < values.size(); ++i) put_big_endian(&buf[i * 4], values[i]);
        out_.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        out_ << "\n";
    }

    void close() { out_.close(); }

private:
    void check(std::size_t n) const {
        if (n != points_) throw std::runtime_error("vtk: field size does not match the grid");
    }

    template <typename V>
    static void put_big_endian(char* dst, V value) {
        static_assert(sizeof(V) == 4, "4-byte values only");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, 4);
        dst[0] = static_cast<char>((bits >> 24) & 0xFF);
        dst[1] = static_cast<char>((bits >> 16) & 0xFF);
        dst[2] = static_cast<char>((bits >> 8) & 0xFF);
        dst[3] = static_cast<char>(bits & 0xFF);
    }

    std::ofstream out_;
    std::size_t points_;
};

} // namespace vtk

#endif // HARMONY_EXAMPLES_VTK_WRITER_HPP
