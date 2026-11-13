#include "detail/format.hpp"

#include <algorithm>
#include <cstdio>

namespace harmony::detail {

namespace {

// Characters on screen (UTF-8 continuation bytes do not count).
std::size_t display_width(const std::string& text) {
    std::size_t n = 0;
    for (unsigned char c : text) n += ((c & 0xC0) != 0x80) ? 1 : 0;
    return n;
}

} // namespace

std::string format_seconds(double seconds) {
    char buf[32];
    if (seconds >= 1.0) {
        std::snprintf(buf, sizeof(buf), "%.3f s", seconds);
    } else if (seconds >= 1.0e-3) {
        std::snprintf(buf, sizeof(buf), "%.2f ms", seconds * 1.0e3);
    } else if (seconds >= 0.05e-6) {
        std::snprintf(buf, sizeof(buf), "%.1f us", seconds * 1.0e6);
    } else {
        return "0 s";
    }
    return buf;
}

std::string format_bytes(std::size_t bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), unit == 0 ? "%.0f %s" : "%.1f %s", value, units[unit]);
    return buf;
}

std::string format_count(std::size_t value) {
    std::string digits = std::to_string(value);
    std::string out;
    for (std::size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out.push_back(',');
        out.push_back(digits[i]);
    }
    return out;
}

std::string format_signed_count(long long value) {
    if (value == 0) return "0";
    const std::string magnitude = format_count(static_cast<std::size_t>(value < 0 ? -value : value));
    return (value < 0 ? "-" : "+") + magnitude;
}

std::string format_percent(double fraction) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f %%", fraction * 100.0);
    return buf;
}

std::string format_number(double value, int significant) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", significant, value);
    return buf;
}

std::string truncate(const std::string& text, std::size_t width) {
    if (text.size() <= width) return text;
    return text.substr(0, width - 3) + "...";
}

std::string collapse_spaces(const std::string& text) {
    std::string out;
    for (char c : text) {
        const bool space = c == ' ' || c == '\t';
        if (space && (out.empty() || out.back() == ' ')) continue;
        out.push_back(space ? ' ' : c);
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

const char* device_kind_name(DeviceKind kind) {
    switch (kind) {
        case DeviceKind::gpu: return "GPU";
        case DeviceKind::cpu: return "CPU";
        case DeviceKind::accelerator: return "ACC";
        case DeviceKind::all: return "?";
    }
    return "?";
}

Table::Table(std::vector<Column> columns) : columns_(std::move(columns)) {}

void Table::add_row(std::vector<std::string> cells) {
    cells.resize(columns_.size());
    rows_.push_back(std::move(cells));
}

void Table::print(std::ostream& os, const std::string& indent) const {
    std::vector<std::size_t> width(columns_.size());
    for (std::size_t c = 0; c < columns_.size(); ++c) {
        width[c] = display_width(columns_[c].header);
        for (const auto& row : rows_) width[c] = std::max(width[c], display_width(row[c]));
    }

    auto print_row = [&](const std::vector<std::string>& cells) {
        std::string line = indent;
        for (std::size_t c = 0; c < columns_.size(); ++c) {
            const std::string& cell = cells[c];
            const std::string pad(width[c] - display_width(cell), ' ');
            line += columns_[c].right_aligned ? pad + cell : cell + pad;
            if (c + 1 < columns_.size()) line += "  ";
        }
        while (!line.empty() && line.back() == ' ') line.pop_back();
        os << line << "\n";
    };

    std::vector<std::string> header;
    std::vector<std::string> rule;
    for (std::size_t c = 0; c < columns_.size(); ++c) {
        header.push_back(columns_[c].header);
        rule.push_back(std::string(width[c], '-'));
    }
    print_row(header);
    print_row(rule);
    for (const auto& row : rows_) print_row(row);
}

} // namespace harmony::detail
