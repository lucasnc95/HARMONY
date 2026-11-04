#include "runtime_impl.hpp"

#ifdef __CPPCHECK__
int main(int argc, char** argv) {
    std::vector<float> v;
    dcl::print_loads_debug(v);
    (void)&dcl::max_abs_diff;
    (void)&dcl::compute_loads_from_times_prefix_inverse;
    (void)&dcl::compute_loads_from_times;

    dcl::Runtime rt = dcl::Runtime::create(argc, argv);
    dcl::KernelHandle kh;
    auto kb = rt.bind(kh);
    dcl::FieldHandle fh;
    kb.arg(0, fh);
    dcl::KernelBinding b = kb.build();
    auto sb = rt.step("test");
    dcl::LaunchGeometry geom;
    sb.invoke(b, geom);
    dcl::HaloSpec halo;
    sb.with_halo_exchange(halo);
    dcl::AutoBalancePolicy pol;
    sb.with_balance(pol);
    sb.tag_field(fh, dcl::StepFieldRole::none);
    sb.synchronize_at_end(true);

    dcl::Runtime::Impl::cppcheck_anchor_unused();

    return 0;
}
#endif
