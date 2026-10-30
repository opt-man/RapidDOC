#pragma once
#include <chrono>
#include <limits>
#include <string>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace f8 {
// Kernel wall time, not hardware FLOP cycles. Separable preparation is paused;
// memory access inside a numerical library kernel remains inseparable.
class NumericalTimer;
inline thread_local NumericalTimer* active_numerical_timer=nullptr;
class NumericalTimer {
    using Clock=std::chrono::steady_clock;
    Clock::time_point since_=Clock::now();
    double elapsed_=0.;
    int pauses_=0;
    bool finished_=false;
    NumericalTimer* previous_=active_numerical_timer;
public:
    NumericalTimer(){active_numerical_timer=this;}
    NumericalTimer(const NumericalTimer&)=delete;
    ~NumericalTimer(){finish();active_numerical_timer=previous_;}
    void pause(){
        if(!finished_ && pauses_++==0)
            elapsed_+=std::chrono::duration<double>(Clock::now()-since_).count();
    }
    void resume(){if(!finished_ && --pauses_==0)since_=Clock::now();}
    double finish(){
        if(!finished_ && pauses_==0)
            elapsed_+=std::chrono::duration<double>(Clock::now()-since_).count();
        finished_=true;return elapsed_;
    }
};
inline void pause_preparation(){if(active_numerical_timer)active_numerical_timer->pause();}
inline void resume_preparation(){if(active_numerical_timer)active_numerical_timer->resume();}
struct ExcludePreparation {
    NumericalTimer* timer=active_numerical_timer;
    ExcludePreparation(){if(timer)timer->pause();}
    ~ExcludePreparation(){if(timer)timer->resume();}
};
template<class F> auto preparation(F&& f)->decltype(f()){
    ExcludePreparation excluded;return std::forward<F>(f)();
}

// Optional hooks in the actual dependency, never a repeated surrogate solve.
// Missing instrumentation is NaN, never zero or inclusive native time.
class ExternalSolveTimer {
    using Reset=void(*)();
    using Read=double(*)();
    Reset reset_=nullptr;
    Read read_=nullptr;
    static void* symbol(const char* module,const std::string& name){
#ifdef _WIN32
        auto h=GetModuleHandleA(module);
        if(!h){std::string alt="lib";alt+=module;h=GetModuleHandleA(alt.c_str());}
        return h?reinterpret_cast<void*>(GetProcAddress(h,name.c_str())):nullptr;
#else
        (void)module;return dlsym(RTLD_DEFAULT,name.c_str());
#endif
    }
public:
    ExternalSolveTimer(const char* prefix,const char* module){
        reset_=reinterpret_cast<Reset>(symbol(module,std::string(prefix)+"_reset_v3"));
        read_=reinterpret_cast<Read>(symbol(module,std::string(prefix)+"_read_v3"));
    }
    void reset()const{if(reset_&&read_)reset_();}
    double read()const{return reset_&&read_?read_():std::numeric_limits<double>::quiet_NaN();}
};
}
