#include "fatrop_kernel_timer_v4.hpp"
#include <chrono>
#include <limits>
namespace {
using Clock = std::chrono::steady_clock;
thread_local bool enabled=false;
thread_local int depth=0, residual_depth=0;
thread_local Clock::time_point started;
thread_local Clock::duration elapsed=Clock::duration::zero();
thread_local unsigned long long calls[frs_fatrop_v4::Count]={};
}
void frs_fatrop_kernel_timer_reset_v4() {
    enabled=true; depth=residual_depth=0; elapsed=Clock::duration::zero();
    for(auto& call:calls) call=0;
}
double frs_fatrop_kernel_timer_read_v4() {
    enabled=false;
    unsigned long long total=0;
    for(auto call:calls) total+=call;
    return total && depth==0 && residual_depth==0
        ? std::chrono::duration<double>(elapsed).count()
        : std::numeric_limits<double>::quiet_NaN();
}
unsigned long long frs_fatrop_kernel_timer_count_v4(int kind) {
    return kind>=0 && kind<frs_fatrop_v4::Count ? calls[kind] : 0;
}
int frs_fatrop_kernel_timer_enabled_v4() { return 1; }
namespace frs_fatrop_v4 {
void begin(Kind k, bool) {
    if(!enabled) return;
    ++calls[k];
    if(depth++==0) started=Clock::now();
}
void end() {
    if(enabled && --depth==0) elapsed+=Clock::now()-started;
}
void residual_enter() { ++residual_depth; }
void residual_leave() { --residual_depth; }
bool in_residual() { return residual_depth>0; }
}
