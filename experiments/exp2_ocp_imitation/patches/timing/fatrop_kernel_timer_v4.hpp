#pragma once
#include <utility>
#ifdef _WIN32
#define FRS_FATROP_EXPORT extern "C" __declspec(dllexport)
#else
#define FRS_FATROP_EXPORT extern "C" __attribute__((visibility("default")))
#endif
FRS_FATROP_EXPORT void frs_fatrop_kernel_timer_reset_v4();
FRS_FATROP_EXPORT double frs_fatrop_kernel_timer_read_v4();
FRS_FATROP_EXPORT unsigned long long frs_fatrop_kernel_timer_count_v4(int);
FRS_FATROP_EXPORT int frs_fatrop_kernel_timer_enabled_v4();
namespace frs_fatrop_v4 {
enum Kind { Factor, Backsolve, Riccati, Reduction, Residual, Count };
void begin(Kind, bool);
void end();
void residual_enter();
void residual_leave();
bool in_residual();
struct Kernel {
    bool active;
    explicit Kernel(Kind k, bool condition=true) : active(condition) { if(active) begin(k, true); }
    ~Kernel() { if(active) end(); }
};
struct ResidualContext {
    ResidualContext() { residual_enter(); }
    ~ResidualContext() { residual_leave(); }
};
template<class F> decltype(auto) run(Kind kind, bool condition, F&& f) {
    Kernel timer(kind, condition);
    return std::forward<F>(f)();
}
}
#define FRS_FATROP_KERNEL(kind, condition, ...) frs_fatrop_v4::run(frs_fatrop_v4::kind, (condition), [&]() { __VA_ARGS__; })
#define FRS_FATROP_VALUE(kind, expression) frs_fatrop_v4::run(frs_fatrop_v4::kind, true, [&]() { return (expression); })
