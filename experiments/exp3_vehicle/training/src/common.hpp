#pragma once
#include "f8.h"
#include "model_config.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace f8 {
using Vec = std::vector<double>;
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
constexpr int nz = F8_N*7+5;
constexpr int neq = (F8_N+1)*5;
inline double nan() { return std::numeric_limits<double>::quiet_NaN(); }
inline void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
inline bool finite(const double *v, size_t n) {
    return std::all_of(v, v+n, [](double x) { return std::isfinite(x); });
}
inline F8Stats empty_stats() {
    return {nan(),nan(),nan(),nan(),nan(),nan(),-1,-1,-1,0};
}
inline void bounds(Vec& lo, Vec& hi) {
    lo.assign(nz, -std::numeric_limits<double>::infinity());
    hi.assign(nz, std::numeric_limits<double>::infinity());
    for (int k=0;k<=F8_N;++k) {
        if (k>0) { lo[k*7+3]=F8_SPEED_MIN; hi[k*7+3]=F8_SPEED_MAX; }
        if (k<F8_N) {
            lo[k*7+5]=-F8_ACCELERATION_MAX; hi[k*7+5]=F8_ACCELERATION_MAX;
            lo[k*7+6]=-F8_STEERING_MAX; hi[k*7+6]=F8_STEERING_MAX;
        }
    }
}
struct Backend {
    bool valid=false;
    F8Stats stats=empty_stats();
    std::map<std::string,Vec> trace;
    virtual ~Backend()=default;
    virtual void forward(const double*, const double*, const double*, double*)=0;
    virtual void backward(const double*, double*)=0;
};
std::unique_ptr<Backend> make_backend(const std::string& assets);
const char* backend_name();
const char* backend_details();
}
