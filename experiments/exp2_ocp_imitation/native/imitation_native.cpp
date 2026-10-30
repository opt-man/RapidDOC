// Native nonlinear forward and actual forward-cache reuse for imitation learning.
// The existing solver and vendor sources are not modified by this adapter.
#include <fatrop/fatrop.hpp>
#include <fatrop/ocp/pd_solver_orig.hpp>
#include "factor_reuse_solver.h"
#include "imitation_models.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#define IL_API extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#define IL_API extern "C"
extern "C" bool fatrop_export_factor_cache(
    const fatrop::PdSolverOrig<fatrop::OcpType>*,
    const fatrop::ProblemInfo<fatrop::OcpType>&,
    double*, double*, double*, double*, double*, double*);
#endif

using namespace fatrop;
namespace {
thread_local std::string last_error;
using Eval = int (*)(const double**, double**, casadi_int*, double*, int);
using Work = int (*)(casadi_int*, casadi_int*, casadi_int*, casadi_int*);
using Export = bool (*)(const PdSolverOrig<OcpType>*, const ProblemInfo<OcpType>&,
                       double*, double*, double*, double*, double*, double*);
double seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
Export exporter() {
#ifdef _WIN32
    HMODULE mod = GetModuleHandleA("fatrop.dll");
    auto fn = mod ? reinterpret_cast<Export>(GetProcAddress(mod, "fatrop_export_factor_cache")) : nullptr;
    if (!fn) throw std::runtime_error("FATROP runtime lacks fatrop_export_factor_cache");
    return fn;
#else
    return &fatrop_export_factor_cache;
#endif
}

// The runtime owns the accumulator because the numerical kernels execute in
// fatrop.dll. A missing instrumented dependency must never fall back to wall
// time around optimize(), which includes model/KKT construction and caches.
struct NativeKernelTimer {
    using Reset = void (*)();
    using Read = double (*)();
    using Count = unsigned long long (*)(int);
    Reset reset;
    Read read;
    Count count;
    static void* symbol(const char* name) {
#ifdef _WIN32
        const auto module = GetModuleHandleA("fatrop.dll");
        return module ? reinterpret_cast<void*>(GetProcAddress(module, name)) : nullptr;
#else
        return dlsym(RTLD_DEFAULT, name);
#endif
    }
    NativeKernelTimer()
        : reset(reinterpret_cast<Reset>(symbol("frs_fatrop_kernel_timer_reset_v4"))),
          read(reinterpret_cast<Read>(symbol("frs_fatrop_kernel_timer_read_v4"))),
          count(reinterpret_cast<Count>(symbol("frs_fatrop_kernel_timer_count_v4"))) {
        const auto capability = reinterpret_cast<int (*)()>(symbol("frs_fatrop_kernel_timer_enabled_v4"));
        if(!reset || !read || !count || !capability || capability()!=1)
            throw std::runtime_error("FATROP runtime lacks solve-kernels-v4 numerical instrumentation; build_imitation_fatrop_v4.ps1 is required");
    }
};
struct Function {
    Eval eval = nullptr;
    std::vector<const double*> args;
    std::vector<double*> results;
    std::vector<casadi_int> iw;
    std::vector<double> w, out;
    void initialize(Eval e, Work work, int n) {
        eval=e;
        casadi_int a=0,r=0,i=0,v=0;
        work(&a,&r,&i,&v);
        args.resize(std::max<casadi_int>(5,a));
        results.resize(std::max<casadi_int>(1,r));
        iw.resize(i); w.resize(v); out.resize(n);
    }
    const std::vector<double>& operator()(const double* x,const double* u,
            const double* p,const double* lambda,const double* scale) {
        args[0]=x; args[1]=u; args[2]=p; args[3]=lambda; args[4]=scale;
        results[0]=out.data();
        if (eval(args.data(),results.data(),iw.data(),w.data(),0))
            throw std::runtime_error("generated model evaluation failed");
        if (!std::all_of(out.begin(),out.end(),[](double v){return std::isfinite(v);}))
            throw std::runtime_error("nonfinite nonlinear model evaluation");
        return out;
    }
};

class Problem final : public OcpAbstract {
public:
    int N,nx,nu,d,nineq,nineqT;
    std::vector<double> parameters,initial,guess,zeros;
    Function f,jac,hess,grad,cost,hessT,gradT,costT,mixed,mixedT;
    Function ineq,ineqJac,ineqHess,ineqT,ineqJacT,ineqHessT;
    Problem(int kind,int horizon,double dt):N(horizon) {
        nx=kind==0?4:(kind==1?6:2); nu=kind==0?1:2;
        d=kind==0?5:(kind==1?6:5);
        nineq=kind==0?4:(kind==1?8:9);
        nineqT=kind==0?2:(kind==1?4:5);
        parameters.resize(d+1); parameters[d]=dt;
        initial.resize(nx); guess.resize(N*(nx+nu)+nx);
        zeros.resize(std::max({nx+nu,nineq,nineqT}));
#define IL_INIT(name,field,count) field.initialize(il_##name##_##field,il_##name##_##field##_work,count)
#define IL_MODEL(name) \
        IL_INIT(name,f,nx); IL_INIT(name,jac,nx*(nx+nu)); \
        IL_INIT(name,hess,(nx+nu)*(nx+nu)); IL_INIT(name,grad,nx+nu); IL_INIT(name,cost,1); \
        IL_INIT(name,hessT,nx*nx); IL_INIT(name,gradT,nx); IL_INIT(name,costT,1); \
        IL_INIT(name,mixed,(nx+nu)*d); IL_INIT(name,mixedT,nx*d); \
        IL_INIT(name,ineq,nineq); IL_INIT(name,ineqJac,nineq*(nx+nu)); \
        IL_INIT(name,ineqHess,(nx+nu)*(nx+nu)); IL_INIT(name,ineqT,nineqT); \
        IL_INIT(name,ineqJacT,nineqT*nx); IL_INIT(name,ineqHessT,nx*nx)
        if(kind==0){ IL_MODEL(cartpole); }
        else if(kind==1){ IL_MODEL(quadrotor); }
        else { IL_MODEL(robotarm); }
#undef IL_MODEL
#undef IL_INIT
    }
    Index get_nx(Index) const override {return nx;}
    Index get_nu(Index k) const override {return k<N?nu:0;}
    Index get_ng(Index k) const override {return k==0?nx:0;}
    Index get_ng_ineq(Index k) const override {return k<N?nineq:nineqT;}
    Index get_horizon_length() const override {return N+1;}
    static void zero(MAT* m) {
        for(int j=0;j<m->n;++j) for(int i=0;i<m->m;++i) blasfeo_matel_wrap(m,i,j)=0;
    }
    int external(int i,int controls) const {return i<controls?nx+i:i-controls;}
    Index eval_BAbt(const Scalar*,const Scalar* u,const Scalar* x,MAT* out,Index) override {
        zero(out); double one=1;
        const auto& J=jac(x,u,parameters.data(),zeros.data(),&one);
        for(int i=0;i<nx+nu;++i) for(int j=0;j<nx;++j)
            blasfeo_matel_wrap(out,i,j)=J[j+nx*external(i,nu)];
        return 0;
    }
    Index eval_RSQrqt(const Scalar* scale,const Scalar* u,const Scalar* x,
            const Scalar* lam,const Scalar*,const Scalar* lam_ineq,MAT* out,Index k) override {
        zero(out); const int controls=get_nu(k),s=nx+controls;
        const auto& H=(k<N?hess:hessT)(x,u,parameters.data(),k<N?lam:zeros.data(),scale);
        double one=1.0;
        const auto& HI=(k<N?ineqHess:ineqHessT)(
            x,u,parameters.data(),lam_ineq?lam_ineq:zeros.data(),&one);
        for(int i=0;i<s;++i) for(int j=0;j<s;++j)
            blasfeo_matel_wrap(out,i,j)=
                H[external(i,controls)+s*external(j,controls)]
                +HI[external(i,controls)+s*external(j,controls)];
        return 0;
    }
    Index eval_Ggt(const Scalar*,const Scalar*,MAT* out,Index k) override {
        zero(out); if(k==0) for(int i=0;i<nx;++i) blasfeo_matel_wrap(out,nu+i,i)=1;
        return 0;
    }
    Index eval_Ggt_ineq(const Scalar* u,const Scalar* x,MAT* out,Index k) override {
        zero(out); double one=1.0;
        const int controls=get_nu(k),count=get_ng_ineq(k);
        const auto& J=(k<N?ineqJac:ineqJacT)(
            x,u,parameters.data(),zeros.data(),&one);
        for(int variable=0;variable<nx+controls;++variable)
            for(int constraint=0;constraint<count;++constraint)
                blasfeo_matel_wrap(out,variable,constraint)=
                    J[constraint+count*external(variable,controls)];
        return 0;
    }
    Index eval_b(const Scalar* next,const Scalar* u,const Scalar* x,Scalar* out,Index) override {
        double one=1; const auto& value=f(x,u,parameters.data(),zeros.data(),&one);
        for(int i=0;i<nx;++i) out[i]=value[i]-next[i]; return 0;
    }
    Index eval_g(const Scalar*,const Scalar* x,Scalar* out,Index k) override {
        if(k==0) for(int i=0;i<nx;++i) out[i]=x[i]-initial[i]; return 0;
    }
    Index eval_gineq(const Scalar* u,const Scalar* x,Scalar* out,Index k) override {
        double one=1.0; const int count=get_ng_ineq(k);
        const auto& value=(k<N?ineq:ineqT)(
            x,u,parameters.data(),zeros.data(),&one);
        std::copy_n(value.data(),count,out); return 0;
    }
    Index eval_rq(const Scalar* scale,const Scalar* u,const Scalar* x,Scalar* out,Index k) override {
        const int controls=get_nu(k);
        const auto& value=(k<N?grad:gradT)(x,u,parameters.data(),zeros.data(),scale);
        for(int i=0;i<nx+controls;++i) out[i]=value[external(i,controls)]; return 0;
    }
    Index eval_L(const Scalar* scale,const Scalar* u,const Scalar* x,Scalar* out,Index k) override {
        out[0]=(k<N?cost:costT)(x,u,parameters.data(),zeros.data(),scale)[0]; return 0;
    }
    Index get_bounds(Scalar* lower,Scalar* upper,Index k) const override {
        for(int i=0;i<get_ng_ineq(k);++i) {
            lower[i]=0.0; upper[i]=std::numeric_limits<Scalar>::infinity();
        }
        return 0;
    }
    Index get_initial_xk(Scalar* x,Index k) const override {
        std::copy_n(guess.data()+k*(nx+nu),nx,x); return 0;
    }
    Index get_initial_uk(Scalar* u,Index k) const override {
        if(k<N) std::copy_n(guess.data()+k*(nx+nu)+nx,nu,u); return 0;
    }
};

struct Session {
    std::shared_ptr<Problem> problem;
    std::unique_ptr<OptionRegistry> options;
    std::unique_ptr<IpAlgBuilder<OcpType>> builder;
    std::shared_ptr<IpAlgorithm<OcpType>> algorithm;
    Export export_cache;
    NativeKernelTimer forward_timer;
    // Kernel diagnostics, five call counts, and the complete optimize() call.
    std::array<double,7> forward_timing{};
    bool valid=false;
    FrsHandle plain_handle=nullptr;
    std::vector<double> H,HT,A,P,R,S,trajectory;
    std::vector<double> target_H,target_HT,target_A,target_mixed,target_mixedT;
    Session(int kind,int N,double dt,double tolerance,int max_iter)
        :problem(std::make_shared<Problem>(kind,N,dt)),options(std::make_unique<OptionRegistry>()),
         builder(std::make_unique<IpAlgBuilder<OcpType>>(std::make_shared<NlpOcp>(problem))),
         algorithm(builder->with_options_registry(options.get()).build()),export_cache(exporter()) {
        options->set_option<Index>("print_level",0);
        options->set_option<Scalar>("tolerance",tolerance);
        options->set_option<Scalar>("tol_acceptable",tolerance);
        options->set_option<Scalar>("constr_viol_tol",tolerance);
        options->set_option<Index>("max_iter",max_iter);
        int nx=problem->nx,nu=problem->nu,s=nx+nu;
        H.resize(N*s*s); HT.resize(nx*nx); A.resize(N*nx*s);
        P.resize((N+1)*nx*nx); R.resize(N*nu*nu); S.resize(N*nx*nu);
        trajectory.resize(N*s+nx);
        target_H.resize(N*s*s); target_HT.resize(nx*nx); target_A.resize(N*nx*s);
        target_mixed.resize(N*s*problem->d); target_mixedT.resize(nx*problem->d);
    }
    ~Session() { if(plain_handle) frs_destroy(plain_handle); }
};

// The plain solver needs only dimensions and a workspace. Bind a lightweight
// handle without exporting or factoring matrices; accidental preconditioning
// must fail rather than silently turn the ablation into the reuse method.
int forbidden_plain_cached_solve(void*,int,const double*,double*) { return 1; }

void copy_matrix_to_row_major(const std::vector<double>& source,double* destination,
                              int rows,int columns) {
    for(int row=0;row<rows;++row) for(int column=0;column<columns;++column)
        destination[row*columns+column]=source[row+rows*column];
}

void build_target(Session& session) {
    auto& p=*session.problem;
    const int nx=p.nx,nu=p.nu,d=p.d,N=p.N,stage_dimension=nx+nu;
    auto& iterate=session.builder->get_ipdata()->current_iterate();
    auto& hessian=iterate.hessian();
    auto& jacobian=iterate.jacobian();
    const auto& info=session.algorithm->info();
    const auto& dx=iterate.Dx();
    const auto& de=iterate.De();
    const bool de_is_zero=iterate.De_is_zero();
    const auto& slacks=iterate.primal_s();
    const auto& lower=iterate.lower_bounds();
    const auto& upper=iterate.upper_bounds();
    const auto& dual_lower=iterate.dual_bounds_l();
    const auto& dual_upper=iterate.dual_bounds_u();
    double one=1.0;

    for(int stage=0;stage<=N;++stage) {
        const int controls=stage<N?nu:0;
        const int variables=nx+controls;
        const auto& matrix=hessian.RSQrqt[stage].mat();
        double* target=stage<N
            ?session.target_H.data()+stage*stage_dimension*stage_dimension
            :session.target_HT.data();
        for(int external_row=0;external_row<variables;++external_row) {
            const int internal_row=external_row<nx?controls+external_row:external_row-nx;
            for(int external_col=0;external_col<variables;++external_col) {
                const int internal_col=external_col<nx?controls+external_col:external_col-nx;
                double value=blasfeo_matel_wrap(
                    &matrix,std::max(internal_row,internal_col),
                    std::min(internal_row,internal_col));
                if(internal_row==internal_col)
                    value+=dx(info.offsets_tangent_u[stage]+internal_row);
                const int count=info.dims.number_of_ineq_constraints[stage];
                const auto& G=jacobian.Gg_ineqt[stage].mat();
                for(int constraint=0;constraint<count;++constraint) {
                    const int slack=info.offsets_slack[stage]+constraint;
                    const double dl=std::max(1e-16,slacks(slack)-lower(slack));
                    const double du=std::max(1e-16,upper(slack)-slacks(slack));
                    const double sigma_inverse=1.0/
                        (dx(info.offset_slack+slack)+dual_lower(slack)/dl+dual_upper(slack)/du);
                    const int equation=info.offsets_g_eq_slack[stage]+constraint;
                    const double diagonal=sigma_inverse+de(equation);
                    value+=blasfeo_matel_wrap(&G,internal_row,constraint)*
                           blasfeo_matel_wrap(&G,internal_col,constraint)/diagonal;
                }
                if(!de_is_zero) {
                    const int count=info.dims.number_of_eq_constraints[stage];
                    const auto& G=jacobian.Gg_eqt[stage].mat();
                    for(int constraint=0;constraint<count;++constraint) {
                        const int equation=info.offsets_g_eq_path[stage]+constraint;
                        value+=blasfeo_matel_wrap(&G,internal_row,constraint)*
                               blasfeo_matel_wrap(&G,internal_col,constraint)/de(equation);
                    }
                }
                target[external_row*variables+external_col]=value;
            }
        }

        const double* x=session.trajectory.data()+stage*stage_dimension;
        const double* u=stage<N?x+nx:p.zeros.data();
        if(stage<N) {
            const auto& dynamics=jacobian.BAbt[stage].mat();
            double* target_A=session.target_A.data()+stage*nx*stage_dimension;
            for(int row=0;row<nx;++row)
                for(int external_col=0;external_col<stage_dimension;++external_col) {
                    const int internal_col=external_col<nx?nu+external_col:external_col-nx;
                    target_A[row*stage_dimension+external_col]=
                        -blasfeo_matel_wrap(&dynamics,internal_col,row);
                }
            copy_matrix_to_row_major(
                p.mixed(x,u,p.parameters.data(),p.zeros.data(),&one),
                session.target_mixed.data()+stage*stage_dimension*d,
                stage_dimension,d);
        } else {
            copy_matrix_to_row_major(
                p.mixedT(x,u,p.parameters.data(),p.zeros.data(),&one),
                session.target_mixedT.data(),nx,d);
        }
    }
}

// Independent common certificate, with exactly the KKT ordering and signs of
// imitation_learning.apply_target: all stage/terminal primals, then lambda_0..N.
// Evaluate directly from the target blocks, not the cached factorization or a
// recursive GMRES estimate. No assembled KKT matrix or temporary vectors needed.
double target_relative_residual(const Session& session,const double* rhs,
                                const double* adjoint) {
    const auto& p=*session.problem;
    const int nx=p.nx,sd=nx+p.nu,N=p.N;
    const int primal=N*sd+nx;
    const double* lambda=adjoint+primal;
    double rhs_norm=0.0,residual_norm=0.0;
    const auto accumulate=[&](int index,double applied) {
        rhs_norm=std::hypot(rhs_norm,rhs[index]);
        residual_norm=std::hypot(residual_norm,rhs[index]-applied);
    };
    for(int stage=0;stage<N;++stage) {
        const double* z=adjoint+stage*sd;
        const double* H=session.target_H.data()+stage*sd*sd;
        const double* A=session.target_A.data()+stage*nx*sd;
        for(int row=0;row<sd;++row) {
            double applied=0.0;
            for(int column=0;column<sd;++column)
                applied+=H[row*sd+column]*z[column];
            for(int equation=0;equation<nx;++equation)
                applied+=A[equation*sd+row]*lambda[(stage+1)*nx+equation];
            if(row<nx) applied+=lambda[stage*nx+row];
            accumulate(stage*sd+row,applied);
        }
        for(int row=0;row<nx;++row) {
            double applied=0.0;
            for(int column=0;column<sd;++column)
                applied+=A[row*sd+column]*z[column];
            applied+=adjoint[(stage+1)*sd+row];
            accumulate(primal+(stage+1)*nx+row,applied);
        }
    }
    for(int row=0;row<nx;++row) {
        double applied=0.0;
        for(int column=0;column<nx;++column)
            applied+=session.target_HT[row*nx+column]*adjoint[N*sd+column];
        applied+=lambda[N*nx+row];
        accumulate(N*sd+row,applied);
        accumulate(primal+row,adjoint[row]);
    }
    if(!std::isfinite(rhs_norm)||!std::isfinite(residual_norm))
        return std::numeric_limits<double>::infinity();
    return residual_norm/std::max(rhs_norm,std::numeric_limits<double>::min());
}
} // namespace

IL_API const char* il_last_error() {return last_error.c_str();}
IL_API void* il_create(int kind,int N,double dt,double tolerance,int max_iter) {
    try {
        if(kind<0||kind>2||N<1||dt<=0||tolerance<=0||max_iter<1)
            throw std::runtime_error("invalid imitation solver configuration");
        return new Session(kind,N,dt,tolerance,max_iter);
    } catch(const std::exception& e) {last_error=e.what();return nullptr;}
}
IL_API void il_destroy(void* ptr) {delete static_cast<Session*>(ptr);}
IL_API int il_forward(void* ptr,const double* theta,const double* x0,
                      const double* guess,double* trajectory,double* elapsed) {
    try {
        auto& s=*static_cast<Session*>(ptr); auto& p=*s.problem; s.valid=false;
        std::copy_n(theta,p.d,p.parameters.data());
        std::copy_n(x0,p.nx,p.initial.data());
        std::copy_n(guess,p.guess.size(),p.guess.data());
        s.forward_timer.reset();
        const auto native_start=std::chrono::steady_clock::now();
        IpSolverReturnFlag status;
        try { status=s.algorithm->optimize(); }
        catch(...) { s.forward_timer.read(); throw; }
        const double native_seconds=seconds(native_start);
        const double solve_seconds=s.forward_timer.read();
        if(status!=IpSolverReturnFlag::Success)
            throw std::runtime_error("FATROP nonlinear forward status "+std::to_string(static_cast<int>(status)));
        if(!std::isfinite(solve_seconds) || solve_seconds<=0.0 || s.forward_timer.count(0)==0)
            throw std::runtime_error("FATROP forward numerical timing is missing or has no factorization");
        s.forward_timing[0]=solve_seconds;
        for(int kind=0;kind<5;++kind)
            s.forward_timing[1+kind]=static_cast<double>(s.forward_timer.count(kind));
        s.forward_timing[6]=native_seconds;
        const auto& primal=s.algorithm->solution_primal(); const auto& info=s.algorithm->info();
        for(int k=0;k<=p.N;++k) {
            int offset=k*(p.nx+p.nu);
            for(int i=0;i<p.nx;++i) trajectory[offset+i]=primal(info.offsets_primal_x[k]+i);
            if(k<p.N) for(int i=0;i<p.nu;++i)
                trajectory[offset+p.nx+i]=primal(info.offsets_primal_u[k]+i);
        }
        std::copy_n(trajectory,s.trajectory.size(),s.trajectory.data());
        s.valid=true; *elapsed=solve_seconds; return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
}
IL_API int il_forward_timing(void* ptr,double* diagnostics) {
    if(!ptr || !diagnostics || !static_cast<Session*>(ptr)->valid) return 1;
    const auto& timing=static_cast<Session*>(ptr)->forward_timing;
    std::copy(timing.begin(),timing.end(),diagnostics);
    return 0;
}
IL_API const char* il_timing_protocol() { return "solve-kernels-v4"; }
namespace {
int backward_impl(void* ptr,const double* rhs,double tolerance,int requested_max_iter,
        int reuse_forward_factors,bool allow_full_dimension,double* output,
        double* gradient,FrsStats* stats,double* cache_seconds) {
    FrsHandle transient_handle=nullptr;
    try {
        if(!ptr||!rhs||!output||!gradient||!stats||!cache_seconds||
                !std::isfinite(tolerance)||tolerance<=0||
                requested_max_iter<(allow_full_dimension?0:1)||
                (reuse_forward_factors!=0&&reuse_forward_factors!=1))
            throw std::runtime_error("invalid native backward arguments");
        auto& s=*static_cast<Session*>(ptr); auto& p=*s.problem;
        if(!s.valid) throw std::runtime_error("backward requires a successful matching forward call");
        if(!reuse_forward_factors&&!s.plain_handle)
            throw std::runtime_error("plain backward requires il_prepare_no_reuse after a successful forward call");
        const long long dimension=static_cast<long long>(p.N)*(2LL*p.nx+p.nu)+2LL*p.nx;
        if(dimension<=0||dimension>std::numeric_limits<int>::max())
            throw std::runtime_error("native backward KKT dimension exceeds the supported integer range");
        const int max_iter=requested_max_iter==0?static_cast<int>(dimension):requested_max_iter;
        *stats={};
        *cache_seconds=0.0;
        auto target_start=std::chrono::steady_clock::now();
        build_target(s);
        const double target_seconds=seconds(target_start);
        FrsHandle handle=s.plain_handle;
        if(reuse_forward_factors) {
            auto start=std::chrono::steady_clock::now();
            if(!s.export_cache(s.builder->get_pd_solver().get(),s.algorithm->info(),
                    s.H.data(),s.HT.data(),s.A.data(),s.P.data(),s.R.data(),s.S.data()))
                throw std::runtime_error("no valid actual FATROP forward factorization is available");
            int code=frs_create_from_factors(p.N,p.nx,p.nu,s.H.data(),s.HT.data(),s.A.data(),
                                            s.P.data(),s.R.data(),s.S.data(),&transient_handle);
            if(code) throw std::runtime_error(frs_last_error());
            handle=transient_handle;
            *cache_seconds=seconds(start);
        }
        // Same solve, segmented timer: exclude separate preparation/copies.
        double solve_call_seconds=0.0;
        int code=frs_solve_ablation(handle,s.target_H.data(),s.target_HT.data(),s.target_A.data(),
                       rhs,tolerance,max_iter,reuse_forward_factors,output,stats,&solve_call_seconds);
        stats->target_build_seconds=target_seconds;
        stats->sensitivity_linearization_seconds=0.0;
        if(code) throw std::runtime_error(frs_last_error());
        const auto certificate_start=std::chrono::steady_clock::now();
        stats->relative_residual=target_relative_residual(s,rhs,output);
        const double certificate_seconds=seconds(certificate_start);
        stats->residual_seconds+=certificate_seconds;
        if(!std::isfinite(stats->relative_residual)||stats->relative_residual>tolerance) {
            std::ostringstream message;
            message << std::scientific << std::setprecision(10)
                    << "independent target adjoint residual exceeds the requested tolerance; "
                    << "relative_residual=" << stats->relative_residual
                    << ", tolerance=" << tolerance;
            throw std::runtime_error(message.str());
        }
        std::fill_n(gradient,p.d,0.0);
        const int stage_dimension=p.nx+p.nu;
        for(int stage=0;stage<p.N;++stage) {
            const double* direction=output+stage*stage_dimension;
            const double* mixed=s.target_mixed.data()+stage*stage_dimension*p.d;
            for(int parameter=0;parameter<p.d;++parameter)
                for(int component=0;component<stage_dimension;++component)
                    gradient[parameter]-=direction[component]*mixed[component*p.d+parameter];
        }
        const double* terminal_direction=output+p.N*stage_dimension;
        for(int parameter=0;parameter<p.d;++parameter)
            for(int state=0;state<p.nx;++state)
                gradient[parameter]-=terminal_direction[state]*
                    s.target_mixedT[state*p.d+parameter];
        if(!std::all_of(gradient,gradient+p.d,[](double value){return std::isfinite(value);}))
            throw std::runtime_error("nonfinite native parameter gradient");
        if(transient_handle) frs_destroy(transient_handle);
        return 0;
    } catch(const std::exception& e) {
        if(transient_handle) frs_destroy(transient_handle);
        last_error=e.what();return 1;
    } catch(...) {
        if(transient_handle) frs_destroy(transient_handle);
        last_error="unknown exception during native backward";return 1;
    }
}
} // namespace

IL_API int il_backward(void* ptr,const double* rhs,double tolerance,int max_iter,
        double* output,double* gradient,FrsStats* stats,double* cache_seconds) {
    return backward_impl(ptr,rhs,tolerance,max_iter,1,false,
                         output,gradient,stats,cache_seconds);
}

// Call outside the measured backward interval. This idempotent workspace is
// dimension-only and remains usable across subsequent successful forwards.
IL_API int il_prepare_no_reuse(void* ptr) {
    try {
        if(!ptr) throw std::runtime_error("invalid plain workspace session");
        auto& s=*static_cast<Session*>(ptr); auto& p=*s.problem;
        if(!s.valid) throw std::runtime_error("plain workspace requires a successful forward call");
        if(!s.plain_handle) {
            const int code=frs_create_from_cached_solve(
                p.N,p.nx,p.nu,forbidden_plain_cached_solve,&s,&s.plain_handle);
            if(code) throw std::runtime_error(frs_last_error());
        }
        return 0;
    } catch(const std::exception& e) {last_error=e.what();return 1;}
    catch(...) {last_error="unknown exception preparing plain workspace";return 1;}
}

// requested_max_iter==0 means a full KKT-dimensional Krylov budget, not an
// unbounded sequence of restarts. Explicit positive budgets retain their meaning.
IL_API int il_backward_ablation(void* ptr,const double* rhs,double tolerance,
        int requested_max_iter,int reuse_forward_factors,double* output,
        double* gradient,FrsStats* stats,double* cache_seconds) {
    return backward_impl(ptr,rhs,tolerance,requested_max_iter,reuse_forward_factors,
                         true,output,gradient,stats,cache_seconds);
}
