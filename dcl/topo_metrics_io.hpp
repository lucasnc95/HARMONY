#ifndef DCL_TOPO_METRICS_IO_HPP
#define DCL_TOPO_METRICS_IO_HPP

#include "types.hpp"
#include <cmath>
#include <cctype>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace dcl {

namespace detail {

inline void skip_whitespace(const std::string& str, std::size_t& pos) noexcept {
    while (pos < str.size() && (std::isspace(static_cast<unsigned char>(str[pos])) != 0)) {
        ++pos;
    }
}

inline bool consume_char(const std::string& str, std::size_t& pos, char c) noexcept {
    skip_whitespace(str, pos);
    if (pos < str.size() && str[pos] == c) {
        ++pos;
        return true;
    }
    return false;
}

inline std::string parse_string(const std::string& str, std::size_t& pos) {
    skip_whitespace(str, pos);
    if (pos >= str.size() || str[pos] != '"') {
        throw dcl::Error("JSON parse error: expected double quote at position " + std::to_string(pos));
    }
    ++pos; // skip opening quote
    std::string result;
    while (pos < str.size()) {
        char c = str[pos++];
        if (c == '"') {
            return result;
        }
        if (static_cast<unsigned char>(c) < 0x20) {
            throw dcl::Error("JSON parse error: unescaped control character");
        }
        if (c == '\\') {
            if (pos >= str.size()) {
                throw dcl::Error("JSON parse error: trailing escape backslash");
            }
            c = str[pos++];
            switch (c) {
                case '"': case '\\': case '/': break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                default: throw dcl::Error("JSON parse error: invalid string escape");
            }
        }
        result.push_back(c);
    }
    throw dcl::Error("JSON parse error: unterminated string");
}

template <typename T>
inline std::vector<T> parse_number_array(const std::string& str, std::size_t& pos) {
    if (!consume_char(str, pos, '[')) {
        throw dcl::Error("JSON parse error: expected '[' at position " + std::to_string(pos));
    }
    std::vector<T> result;
    skip_whitespace(str, pos);
    if (pos < str.size() && str[pos] == ']') {
        ++pos;
        return result;
    }

    while (pos < str.size()) {
        skip_whitespace(str, pos);
        std::size_t start = pos;
        if (pos < str.size() && str[pos] == '-') ++pos;
        if (pos >= str.size()) throw dcl::Error("JSON parse error: missing number");
        if (str[pos] == '0') {
            ++pos;
            if (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos]))) {
                throw dcl::Error("JSON parse error: leading zero");
            }
        } else {
            if (str[pos] < '1' || str[pos] > '9') {
                throw dcl::Error("JSON parse error: invalid number");
            }
            while (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos]))) ++pos;
        }
        bool fractional = false;
        if (pos < str.size() && str[pos] == '.') {
            fractional = true;
            ++pos;
            if (pos >= str.size() || !std::isdigit(static_cast<unsigned char>(str[pos]))) {
                throw dcl::Error("JSON parse error: invalid fraction");
            }
            while (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos]))) ++pos;
        }
        if (pos < str.size() && (str[pos] == 'e' || str[pos] == 'E')) {
            fractional = true;
            ++pos;
            if (pos < str.size() && (str[pos] == '+' || str[pos] == '-')) ++pos;
            if (pos >= str.size() || !std::isdigit(static_cast<unsigned char>(str[pos]))) {
                throw dcl::Error("JSON parse error: invalid exponent");
            }
            while (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos]))) ++pos;
        }
        if constexpr (std::is_integral_v<T>) {
            if (fractional) throw dcl::Error("JSON parse error: expected integer");
        }
        T val{};
        const auto parsed = std::from_chars(str.data() + start, str.data() + pos, val);
        if (parsed.ec != std::errc{} || parsed.ptr != str.data() + pos) {
            throw dcl::Error("JSON parse error: number out of range");
        }
        if (!std::isfinite(static_cast<double>(val))) {
            throw dcl::Error("JSON parse error: non-finite number");
        }
        result.push_back(val);

        skip_whitespace(str, pos);
        if (pos < str.size() && str[pos] == ',') {
            ++pos;
            skip_whitespace(str, pos);
            if (pos >= str.size() || str[pos] == ']') {
                throw dcl::Error("JSON parse error: trailing array comma");
            }
            continue;
        }
        if (pos < str.size() && str[pos] == ']') {
            ++pos;
            return result;
        }
        throw dcl::Error("JSON parse error: expected ',' or ']' at position " + std::to_string(pos));
    }
    throw dcl::Error("JSON parse error: missing closing ']'");
}

inline void skip_json_value(const std::string& str, std::size_t& pos, int depth = 0) {
    if (depth > 64) throw dcl::Error("JSON parse error: excessive nesting");
    skip_whitespace(str, pos);
    if (pos >= str.size()) throw dcl::Error("JSON parse error: missing value");
    if (str[pos] == '"') {
        (void)parse_string(str, pos);
        return;
    }
    if (str[pos] == '[' || str[pos] == '{') {
        const bool object = str[pos] == '{';
        ++pos;
        skip_whitespace(str, pos);
        const char closing = object ? '}' : ']';
        if (pos < str.size() && str[pos] == closing) { ++pos; return; }
        while (true) {
            if (object) {
                (void)parse_string(str, pos);
                if (!consume_char(str, pos, ':')) {
                    throw dcl::Error("JSON parse error: expected ':'");
                }
            }
            skip_json_value(str, pos, depth + 1);
            if (consume_char(str, pos, closing)) return;
            if (!consume_char(str, pos, ',')) {
                throw dcl::Error("JSON parse error: expected comma");
            }
            skip_whitespace(str, pos);
            if (pos >= str.size() || str[pos] == closing) {
                throw dcl::Error("JSON parse error: trailing comma");
            }
        }
    }
    for (const char* literal : {"true", "false", "null"}) {
        const std::size_t length = std::char_traits<char>::length(literal);
        if (str.compare(pos, length, literal) == 0) { pos += length; return; }
    }
    const std::size_t start = pos;
    if (str[pos] == '-') ++pos;
    if (pos >= str.size()) throw dcl::Error("JSON parse error: invalid number");
    if (str[pos] == '0') {
        ++pos;
        if (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos]))) {
            throw dcl::Error("JSON parse error: leading zero");
        }
    } else {
        if (str[pos] < '1' || str[pos] > '9') {
            throw dcl::Error("JSON parse error: invalid value");
        }
        while (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos]))) ++pos;
    }
    if (pos < str.size() && str[pos] == '.') {
        ++pos;
        if (pos >= str.size() || !std::isdigit(static_cast<unsigned char>(str[pos]))) {
            throw dcl::Error("JSON parse error: invalid fraction");
        }
        while (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos]))) ++pos;
    }
    if (pos < str.size() && (str[pos] == 'e' || str[pos] == 'E')) {
        ++pos;
        if (pos < str.size() && (str[pos] == '+' || str[pos] == '-')) ++pos;
        if (pos >= str.size() || !std::isdigit(static_cast<unsigned char>(str[pos]))) {
            throw dcl::Error("JSON parse error: invalid exponent");
        }
        while (pos < str.size() && std::isdigit(static_cast<unsigned char>(str[pos]))) ++pos;
    }
    double value = 0.0;
    const auto parsed = std::from_chars(str.data() + start, str.data() + pos, value);
    if (parsed.ec != std::errc{} || parsed.ptr != str.data() + pos || !std::isfinite(value)) {
        throw dcl::Error("JSON parse error: invalid number");
    }
}

template <typename T>
inline void serialize_array(std::ostream& os, const std::string& key, const std::vector<T>& vec, bool is_last = false) {
    os << "  \"" << key << "\": [";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) os << ", ";
        os << std::setprecision(10) << vec[i];
    }
    os << "]";
    if (!is_last) os << ",";
    os << "\n";
}

} // namespace detail

inline TopoMetrics load_topo_metrics(const std::string& json_path) {
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw dcl::Error("Cannot open file: " + json_path);
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string content = buffer.str();

    TopoMetrics metrics;
    std::size_t pos = 0;
    if (!detail::consume_char(content, pos, '{')) {
        throw dcl::Error("JSON parse error: expected '{' at start of file: " + json_path);
    }

    bool closed = false;
    std::unordered_set<std::string> seen_keys;
    while (pos < content.size()) {
        detail::skip_whitespace(content, pos);
        if (pos < content.size() && content[pos] == '}') {
            ++pos;
            closed = true;
            break;
        }
        std::string key = detail::parse_string(content, pos);
        if (!seen_keys.insert(key).second) {
            throw dcl::Error("JSON parse error: duplicate key '" + key + "'");
        }
        if (!detail::consume_char(content, pos, ':')) {
            throw dcl::Error("JSON parse error: expected ':' after key '" + key + "'");
        }
        detail::skip_whitespace(content, pos);

        if (key == "pcie_latency_ns") {
            metrics.pcie_latency_ns = detail::parse_number_array<double>(content, pos);
        } else if (key == "pcie_bandwidth_gbps") {
            metrics.pcie_bandwidth_gbps = detail::parse_number_array<double>(content, pos);
        } else if (key == "mpi_latency_ns") {
            metrics.mpi_latency_ns = detail::parse_number_array<double>(content, pos);
        } else if (key == "mpi_bandwidth_gbps") {
            metrics.mpi_bandwidth_gbps = detail::parse_number_array<double>(content, pos);
        } else if (key == "numa_distance") {
            metrics.numa_distance = detail::parse_number_array<int>(content, pos);
        } else if (key == "device_numa_node") {
            metrics.device_numa_node = detail::parse_number_array<int>(content, pos);
        } else if (key == "memory_contention_factor") {
            metrics.memory_contention_factor = detail::parse_number_array<double>(content, pos);
        } else if (key == "thermal_tdp_watts") {
            metrics.thermal_tdp_watts = detail::parse_number_array<double>(content, pos);
        } else if (key == "current_power_watts") {
            metrics.current_power_watts = detail::parse_number_array<double>(content, pos);
        } else if (key == "synthetic") {
            if (content.compare(pos, 4, "true") == 0) {
                metrics.synthetic = true;
                pos += 4;
            } else if (content.compare(pos, 5, "false") == 0) {
                metrics.synthetic = false;
                pos += 5;
            } else {
                throw dcl::Error("JSON parse error: synthetic must be boolean");
            }
        } else {
            detail::skip_json_value(content, pos);
        }

        detail::skip_whitespace(content, pos);
        if (pos < content.size() && content[pos] == ',') {
            ++pos;
            detail::skip_whitespace(content, pos);
            if (pos < content.size() && content[pos] == '}') {
                throw dcl::Error("JSON parse error: trailing comma before '}'");
            }
            continue;
        }
        if (pos < content.size() && content[pos] == '}') {
            ++pos;
            closed = true;
            break;
        }
        throw dcl::Error("JSON parse error: expected ',' or '}' at position " + std::to_string(pos));
    }

    if (!closed) {
        throw dcl::Error("JSON parse error: missing closing '}'");
    }
    detail::skip_whitespace(content, pos);
    if (pos != content.size()) {
        throw dcl::Error("JSON parse error: trailing characters after object");
    }

    return metrics;
}

inline void save_topo_metrics(const TopoMetrics& metrics, const std::string& json_path) {
    std::ofstream ofs(json_path);
    if (!ofs.is_open()) {
        throw dcl::Error("Cannot open file for writing: " + json_path);
    }
    ofs << "{\n";
    detail::serialize_array(ofs, "pcie_latency_ns", metrics.pcie_latency_ns);
    detail::serialize_array(ofs, "pcie_bandwidth_gbps", metrics.pcie_bandwidth_gbps);
    detail::serialize_array(ofs, "mpi_latency_ns", metrics.mpi_latency_ns);
    detail::serialize_array(ofs, "mpi_bandwidth_gbps", metrics.mpi_bandwidth_gbps);
    detail::serialize_array(ofs, "numa_distance", metrics.numa_distance);
    detail::serialize_array(ofs, "device_numa_node", metrics.device_numa_node);
    detail::serialize_array(ofs, "memory_contention_factor", metrics.memory_contention_factor);
    detail::serialize_array(ofs, "thermal_tdp_watts", metrics.thermal_tdp_watts);
    detail::serialize_array(ofs, "current_power_watts", metrics.current_power_watts);
    ofs << "  \"synthetic\": " << (metrics.synthetic ? "true" : "false") << "\n";
    ofs << "}\n";
}

inline int numa_distance_between(const TopoMetrics& metrics, int node_a, int node_b) {
    if (metrics.numa_distance.empty()) {
        throw dcl::Error("NUMA distance matrix is empty");
    }
    const std::size_t n_dist = metrics.numa_distance.size();
    const std::size_t num_nodes = static_cast<std::size_t>(std::round(std::sqrt(static_cast<double>(n_dist))));
    if (num_nodes == 0 || num_nodes * num_nodes != n_dist) {
        throw dcl::Error("Invalid NUMA distance matrix dimension");
    }
    if (node_a < 0 || static_cast<std::size_t>(node_a) >= num_nodes ||
        node_b < 0 || static_cast<std::size_t>(node_b) >= num_nodes) {
        throw dcl::Error("Invalid NUMA node index");
    }
    const std::size_t idx_ab = static_cast<std::size_t>(node_a) * num_nodes + static_cast<std::size_t>(node_b);
    const std::size_t idx_ba = static_cast<std::size_t>(node_b) * num_nodes + static_cast<std::size_t>(node_a);
    const int dist_ab = metrics.numa_distance[idx_ab];
    const int dist_ba = metrics.numa_distance[idx_ba];
    int dist = std::max(dist_ab, dist_ba);

    const std::size_t diag_a_idx = static_cast<std::size_t>(node_a) * num_nodes + static_cast<std::size_t>(node_a);
    const std::size_t diag_b_idx = static_cast<std::size_t>(node_b) * num_nodes + static_cast<std::size_t>(node_b);
    const int diag_min = std::min(metrics.numa_distance[diag_a_idx], metrics.numa_distance[diag_b_idx]);
    if (dist < diag_min) {
        dist = diag_min;
    }
    return dist;
}

inline double estimate_migration_cost_bytes(const TopoMetrics& metrics,
                                           int src_global_device,
                                           int dst_global_device,
                                           int src_rank,
                                           int dst_rank,
                                           std::size_t bytes,
                                           double numa_penalty_ns_per_hop = 50.0) {
    if (src_global_device < 0 || dst_global_device < 0) {
        throw dcl::Error("Device index cannot be negative");
    }
    const std::size_t s_idx = static_cast<std::size_t>(src_global_device);
    const double pcie_lat = (s_idx < metrics.pcie_latency_ns.size()) ? metrics.pcie_latency_ns[s_idx] : 5000.0;
    const double pcie_bw = (s_idx < metrics.pcie_bandwidth_gbps.size() && metrics.pcie_bandwidth_gbps[s_idx] > 0.0)
                               ? metrics.pcie_bandwidth_gbps[s_idx]
                               : 10.0;

    const std::size_t d_idx = static_cast<std::size_t>(dst_global_device);
    const double pcie_lat_dst = (d_idx < metrics.pcie_latency_ns.size()) ? metrics.pcie_latency_ns[d_idx] : 5000.0;
    const double pcie_bw_dst = (d_idx < metrics.pcie_bandwidth_gbps.size() && metrics.pcie_bandwidth_gbps[d_idx] > 0.0)
                               ? metrics.pcie_bandwidth_gbps[d_idx]
                               : 10.0;

    int src_node = 0;
    int dst_node = 0;
    if (s_idx < metrics.device_numa_node.size()) {
        src_node = metrics.device_numa_node[s_idx];
    }
    if (d_idx < metrics.device_numa_node.size()) {
        dst_node = metrics.device_numa_node[d_idx];
    }

    double latency_s = 0.0;
    double transfer_s = 0.0;

    if (src_rank != dst_rank && src_rank >= 0 && dst_rank >= 0) {
        // Inter-rank migration: PCIe src + MPI network + PCIe dst
        latency_s = (pcie_lat + pcie_lat_dst) * 1e-9;
        transfer_s = (static_cast<double>(bytes) / (pcie_bw * 1e9)) +
                     (static_cast<double>(bytes) / (pcie_bw_dst * 1e9));

        const std::size_t num_ranks =
            static_cast<std::size_t>(std::sqrt(metrics.mpi_latency_ns.size()));
        if (num_ranks == 0 || num_ranks * num_ranks != metrics.mpi_latency_ns.size() ||
            metrics.mpi_bandwidth_gbps.size() != metrics.mpi_latency_ns.size() ||
            static_cast<std::size_t>(src_rank) >= num_ranks ||
            static_cast<std::size_t>(dst_rank) >= num_ranks) {
            throw dcl::Error("MPI migration metrics do not cover the requested ranks");
        }
        const std::size_t mpi_idx =
            static_cast<std::size_t>(src_rank) * num_ranks +
            static_cast<std::size_t>(dst_rank);
        const double mpi_bw = metrics.mpi_bandwidth_gbps[mpi_idx];
        if (!std::isfinite(mpi_bw) || mpi_bw <= 0.0 ||
            !std::isfinite(metrics.mpi_latency_ns[mpi_idx]) ||
            metrics.mpi_latency_ns[mpi_idx] < 0.0) {
            throw dcl::Error("Invalid MPI migration metrics");
        }
        latency_s += metrics.mpi_latency_ns[mpi_idx] * 1e-9;
        transfer_s += static_cast<double>(bytes) / (mpi_bw * 1e9);
    } else {
        // Intra-rank migration: NUMA + PCIe
        if (src_node == dst_node) {
            latency_s = pcie_lat * 1e-9;
        } else {
            const double extra_latency_ns =
                static_cast<double>(numa_distance_between(metrics, src_node, dst_node)) * numa_penalty_ns_per_hop;
            latency_s = (pcie_lat + extra_latency_ns) * 1e-9;
        }
        transfer_s = static_cast<double>(bytes) / (pcie_bw * 1e9);
    }

    return latency_s + transfer_s;
}

inline double estimate_migration_cost_bytes(const TopoMetrics& metrics,
                                           int src_global_device,
                                           int dst_global_device,
                                           std::size_t bytes,
                                           double numa_penalty_ns_per_hop = 50.0) {
    return estimate_migration_cost_bytes(metrics, src_global_device, dst_global_device, -1, -1, bytes, numa_penalty_ns_per_hop);
}

inline double adjusted_capacity(const TopoMetrics& metrics, int global_device, double raw_throughput) {
    if (global_device < 0 || static_cast<std::size_t>(global_device) >= metrics.memory_contention_factor.size()) {
        throw dcl::Error("Device index out of range for memory_contention_factor");
    }
    const double factor = metrics.memory_contention_factor[static_cast<std::size_t>(global_device)];
    if (factor < 1.0) {
        throw dcl::Error("memory_contention_factor must be >= 1.0");
    }
    return raw_throughput / factor;
}

} // namespace dcl

using dcl::load_topo_metrics;
using dcl::save_topo_metrics;
using dcl::numa_distance_between;
using dcl::estimate_migration_cost_bytes;
using dcl::adjusted_capacity;

#endif
