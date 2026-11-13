// Text formatting for the tables printed by the runtime.

#ifndef HARMONY_DETAIL_FORMAT_HPP
#define HARMONY_DETAIL_FORMAT_HPP

#include "harmony/types.hpp"

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace harmony::detail {

// 0.000123 -> "123.0 us", 0.0456 -> "45.60 ms", 12.3 -> "12.300 s"
std::string format_seconds(double seconds);
// 1536 -> "1.5 KiB"
std::string format_bytes(std::size_t bytes);
// 1234567 -> "1,234,567"
std::string format_count(std::size_t value);
// 0.253 -> "25.3 %"
std::string format_percent(double fraction);
// value with `significant` digits: 0.0123456, 3 -> "0.0123"
std::string format_number(double value, int significant);
// 1234 -> "+1,234", -12 -> "-12"
std::string format_signed_count(long long value);

std::string truncate(const std::string& text, std::size_t width);
// Trims and turns every run of spaces into one space.
std::string collapse_spaces(const std::string& text);
const char* device_kind_name(DeviceKind kind);

// Plain-text table with automatic column widths.
class Table {
public:
    struct Column {
        std::string header;
        bool right_aligned;
    };

    explicit Table(std::vector<Column> columns);
    void add_row(std::vector<std::string> cells);
    void print(std::ostream& os, const std::string& indent) const;

private:
    std::vector<Column> columns_;
    std::vector<std::vector<std::string>> rows_;
};

} // namespace harmony::detail

#endif // HARMONY_DETAIL_FORMAT_HPP
