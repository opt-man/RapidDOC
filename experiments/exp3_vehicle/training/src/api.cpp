#include "common.hpp"
struct F8Solver { std::unique_ptr<f8::Backend> backend; };
namespace { thread_local std::string error; }
extern "C" {
const char* f8_backend() { return f8::backend_name(); }
const char* f8_backend_details() { return f8::backend_details(); }
const char* f8_model_id() { return F8_MODEL_ID; }
const char* f8_last_error() { return error.c_str(); }
int f8_horizon() { return F8_N; }
int f8_trajectory_size() { return f8::nz; }
double f8_dt() { return F8_DT; }
F8Solver* f8_create(const char* assets) {
    try { error.clear(); return new F8Solver{f8::make_backend(assets?assets:"")}; }
    catch(const std::exception& e) { error=e.what(); return nullptr; }
}
void f8_destroy(F8Solver* s) { delete s; }
int f8_forward(F8Solver* s,const double* theta,const double* initial,const double* guess,
               double* output,F8Stats* stats) {
    if (stats) *stats=f8::empty_stats();
    try {
        f8::require(s&&theta&&initial&&guess&&output&&stats,"null forward argument");
        auto& b=*s->backend; b.valid=false; b.stats=f8::empty_stats(); b.trace.clear();
        std::fill_n(output,f8::nz,f8::nan());
        b.trace["theta"]={theta,theta+4}; b.trace["initial"]={initial,initial+5};
        b.trace["guess"]={guess,guess+f8::nz};
        f8::require(f8::finite(theta,4)&&f8::finite(initial,5)&&f8::finite(guess,f8::nz),"nonfinite forward input");
        f8::require(initial[3]>=F8_SPEED_MIN-1e-8&&initial[3]<=F8_SPEED_MAX+1e-8,"initial speed outside model bounds");
        b.forward(theta,initial,guess,output);
        f8::require(f8::finite(output,f8::nz),"nonfinite forward result");
        b.trace["trajectory"]={output,output+f8::nz};
        b.valid=true; *stats=b.stats; error.clear(); return 0;
    } catch(const std::exception& e) {
        error=e.what(); if(s) { s->backend->stats.status=1; if(stats)*stats=s->backend->stats; } return 1;
    }
}
int f8_backward(F8Solver* s,const double* seed,double* gradient,F8Stats* stats) {
    if(gradient)std::fill_n(gradient,4,f8::nan());
    try {
        f8::require(s&&seed&&gradient&&stats,"null backward argument");
        auto& b=*s->backend; f8::require(b.valid,"backward requires its matching successful forward");
        b.trace["loss_seed"]={seed,seed+f8::nz};
        f8::require(f8::finite(seed,f8::nz),"nonfinite loss seed");
        b.backward(seed,gradient);
        f8::require(f8::finite(gradient,4),"nonfinite parameter gradient");
        b.trace["gradient"]={gradient,gradient+4}; *stats=b.stats; error.clear(); return 0;
    } catch(const std::exception& e) {
        error=e.what(); if(s) {s->backend->stats.status=1; if(stats)*stats=s->backend->stats;} return 1;
    }
}
int f8_trace_count(F8Solver* s) {return s?static_cast<int>(s->backend->trace.size()):0;}
const char* f8_trace_name(F8Solver* s,int i) {
    if(!s||i<0||i>=f8_trace_count(s))return nullptr;
    auto it=s->backend->trace.begin();std::advance(it,i);return it->first.c_str();
}
const double* f8_trace_values(F8Solver* s,int i,size_t* n) {
    if(n)*n=0;if(!s||!n||i<0||i>=f8_trace_count(s))return nullptr;
    auto it=s->backend->trace.begin();std::advance(it,i);*n=it->second.size();return it->second.data();
}
}
