#include "IpoptConfig.h"
#include "IpSolveKernelTimer.hpp"
#include <array>
#include <limits>

namespace {
thread_local std::array<double, 4> kernel_seconds{};
thread_local std::array<unsigned long long, 4> kernel_counts{};
thread_local int arithmetic_context = -1;
thread_local unsigned int kernel_depth = 0;
}

extern "C" IPOPTLIB_EXPORT const char* frs_ipopt_solve_timer_protocol_v4() {
    return "mumps-numeric-and-pd-arithmetic-v4";
}
extern "C" IPOPTLIB_EXPORT void frs_ipopt_solve_timer_reset_v4() {
    kernel_seconds.fill(0.0); kernel_counts.fill(0);
    arithmetic_context = -1; kernel_depth = 0;
}
extern "C" IPOPTLIB_EXPORT double frs_ipopt_solve_timer_read_v4() {
    double total = 0.0;
    for (double value : kernel_seconds) total += value;
    return total;
}
extern "C" IPOPTLIB_EXPORT double frs_ipopt_solve_timer_seconds_v4(int kind) {
    return kind >= 0 && kind < 4 ? kernel_seconds[kind] : std::numeric_limits<double>::quiet_NaN();
}
extern "C" IPOPTLIB_EXPORT unsigned long long frs_ipopt_solve_timer_count_v4(int kind) {
    return kind >= 0 && kind < 4 ? kernel_counts[kind] : 0;
}
extern "C" IPOPTLIB_EXPORT void frs_ipopt_solve_timer_add_v4(int kind, double seconds) {
    if (kind >= 0 && kind < 4) { kernel_seconds[kind] += seconds; ++kernel_counts[kind]; }
}
extern "C" IPOPTLIB_EXPORT int frs_ipopt_solve_timer_context_v4() {
    return arithmetic_context;
}
extern "C" IPOPTLIB_EXPORT int frs_ipopt_solve_timer_exchange_context_v4(int kind) {
    const int previous = arithmetic_context;
    arithmetic_context = kind;
    return previous;
}
extern "C" IPOPTLIB_EXPORT bool frs_ipopt_solve_timer_enter_v4() {
    return kernel_depth++ == 0;
}
extern "C" IPOPTLIB_EXPORT void frs_ipopt_solve_timer_leave_v4() {
    --kernel_depth;
}

extern "C" IPOPTLIB_EXPORT const char* frs_ipopt_timing_clock()
{
    return "steady-clock-native-tasks-v1";
}
