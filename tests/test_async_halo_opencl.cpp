#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif
#include "../dcl/runtime.hpp"
#include "../dcl/runtime.cpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {
std::vector<float> stencil(const std::vector<float>& input, int radius) {
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        float sum = 0.0f;
        int count = 0;
        for (int offset = -radius; offset <= radius; ++offset) {
            const auto neighbor = static_cast<long long>(i) + offset;
            if (neighbor >= 0 && neighbor < static_cast<long long>(input.size())) {
                sum += input[static_cast<std::size_t>(neighbor)];
                ++count;
            }
        }
        output[i] = sum / static_cast<float>(count);
    }
    return output;
}
}

int main(int argc, char** argv) {
    dcl::Runtime runtime = dcl::Runtime::create(argc, argv);
    runtime.discover_devices({dcl::DeviceKind::gpu, 0});
    const int local_count = static_cast<int>(runtime.devices().size());
    const int local_ready = runtime.size() == 2 && local_count == 1 ? 1 : 0;
    int all_ready = 0;
    if (MPI_Allreduce(&local_ready, &all_ready, 1, MPI_INT, MPI_MIN,
                      runtime.communicator()) != MPI_SUCCESS) {
        std::cerr << "Failed to verify two-rank OpenCL GPU allocation\n";
        MPI_Abort(runtime.communicator(), 1);
    }
    if (!all_ready) {
        if (runtime.rank() == 0) {
            std::cout << "SKIP: asynchronous halo test needs exactly one OpenCL GPU on each of two ranks\n";
        }
        MPI_Finalize();
        return 77;
    }

    constexpr std::size_t elements = 1024;
    constexpr int radius = 3;
    std::vector<float> input(elements), output(elements, 0.0f);
    for (std::size_t i = 0; i < elements; ++i) {
        input[i] = static_cast<float>(i % 37) * 0.03125f;
    }
    dcl::PartitionSpec partition;
    partition.global_elements = elements;
    partition.units_per_element = 1;
    partition.bytes_per_unit = sizeof(float);
    runtime.set_partition(partition);

    dcl::FieldSpec in_spec{"input", elements, 1, sizeof(float),
                           dcl::BufferUsage::read_write, input.data(),
                           dcl::RedistributionDependency::proportional};
    dcl::FieldSpec out_spec{"output", elements, 1, sizeof(float),
                            dcl::BufferUsage::read_write, output.data(),
                            dcl::RedistributionDependency::proportional};
    const auto in_field = runtime.create_field(in_spec);
    const auto out_field = runtime.create_field(out_spec);
    const auto kernel = runtime.create_kernel(
        {"benchmarks/kneighbor.cl", "kneighbor_stencil", ""});
    const auto first_binding = runtime.bind(kernel)
        .arg(0, in_field).arg(1, out_field)
        .arg(2, dcl::ScalarArg(static_cast<int>(elements)))
        .arg(3, dcl::ScalarArg(radius)).build();
    const auto second_binding = runtime.bind(kernel)
        .arg(0, out_field).arg(1, in_field)
        .arg(2, dcl::ScalarArg(static_cast<int>(elements)))
        .arg(3, dcl::ScalarArg(radius)).build();

    auto first = runtime.step("async_first");
    first.tag_field(in_field, dcl::StepFieldRole::read_source);
    first.tag_field(out_field, dcl::StepFieldRole::write_target);
    first.invoke(first_binding, {0, elements, std::nullopt});
    first.synchronize_at_end(false);
    runtime.execute(first.build());

    auto second = runtime.step("halo_second");
    second.tag_field(out_field, dcl::StepFieldRole::read_source);
    second.tag_field(in_field, dcl::StepFieldRole::write_target);
    second.with_halo_exchange({static_cast<std::size_t>(radius), {out_field}});
    second.invoke(second_binding, {0, elements, std::nullopt});
    second.synchronize_at_end(true);
    runtime.execute(second.build());

    runtime.gather(in_field, input.data(), elements * sizeof(float));
    int mismatch = 0;
    if (runtime.rank() == 0) {
        std::vector<float> initial(elements);
        for (std::size_t i = 0; i < elements; ++i) {
            initial[i] = static_cast<float>(i % 37) * 0.03125f;
        }
        const auto expected = stencil(stencil(initial, radius), radius);
        for (std::size_t i = 0; i < elements; ++i) {
            if (!std::isfinite(input[i]) ||
                std::fabs(input[i] - expected[i]) > 1e-5f) {
                std::cerr << "Async halo mismatch at index " << i << '\n';
                mismatch = 1;
                break;
            }
        }
    }
    MPI_Bcast(&mismatch, 1, MPI_INT, 0, runtime.communicator());
    if (runtime.rank() == 0 && mismatch == 0) {
        std::cout << "PASS: asynchronous OpenCL halo matches two-step CPU reference\n";
    }
    MPI_Finalize();
    return mismatch;
}
