#include "common.hpp"
#include "gmres.hpp"
#include "model_generated.h"
#include <fatrop/fatrop.hpp>
#include <fatrop/ocp/pd_solver_orig.hpp>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace f8 {
using namespace fatrop;
using Eval=int(*)(const double**,double**,casadi_int*,double*,int);
using Work=int(*)(casadi_int*,casadi_int*,casadi_int*,casadi_int*);
struct Function {
    Eval eval;std::vector<const double*> args;std::vector<double*> results;
    std::vector<casadi_int> iw;Vec work,out;
    Function(Eval e,Work w,int n):eval(e),out(n){
        casadi_int a=0,r=0,i=0,v=0;w(&a,&r,&i,&v);
        args.resize(std::max<casadi_int>(5,a));results.resize(std::max<casadi_int>(1,r));iw.resize(i);work.resize(v);
    }
    const Vec& operator()(const double* x,const double* u,const double* p,const double* lam,const double* scale){
        args[0]=x;args[1]=u;args[2]=p;args[3]=lam;args[4]=scale;results[0]=out.data();
        require(eval(args.data(),results.data(),iw.data(),work.data(),0)==0,"generated model evaluation failed");
        require(finite(out.data(),out.size()),"nonfinite generated model");return out;
    }
};
#define FN(name,size) Function name{f8m_##name,f8m_##name##_work,size}
class Problem final:public OcpAbstract {
public:
    Vec theta=Vec(4),initial=Vec(5),guess=Vec(nz),zero=Vec(7);
    FN(dynamics,5);FN(jacobian,35);FN(hessian,49);FN(gradient,7);FN(cost,1);
    FN(terminal_hessian,25);FN(terminal_gradient,5);FN(terminal_cost,1);
    FN(mixed,28);FN(terminal_mixed,20);
    Index get_nx(Index)const override{return 5;}
    Index get_nu(Index k)const override{return k<F8_N?2:0;}
    Index get_ng(Index k)const override{return k==0?5:0;}
    Index get_ng_ineq(Index k)const override{return k==0?2:(k==F8_N?1:3);}
    Index get_horizon_length()const override{return F8_N+1;}
    static int ext(int i,int nu){return i<nu?5+i:i-nu;}
    static void zero_matrix(MAT* m){for(int j=0;j<m->n;++j)for(int i=0;i<m->m;++i)blasfeo_matel_wrap(m,i,j)=0;}
    Index eval_BAbt(const Scalar*,const Scalar* u,const Scalar* x,MAT* out,Index)override{
        zero_matrix(out);double one=1;const auto& J=jacobian(x,u,theta.data(),zero.data(),&one);
        for(int i=0;i<7;++i)for(int j=0;j<5;++j)blasfeo_matel_wrap(out,i,j)=J[j+5*ext(i,2)];return 0;
    }
    Index eval_RSQrqt(const Scalar* scale,const Scalar* u,const Scalar* x,const Scalar* lam,
                      const Scalar*,const Scalar*,MAT* out,Index k)override{
        zero_matrix(out);int nu=get_nu(k),n=5+nu;
        const auto& H=(k<F8_N?hessian:terminal_hessian)(x,u?u:zero.data(),theta.data(),lam?lam:zero.data(),scale);
        for(int i=0;i<n;++i)for(int j=0;j<n;++j)blasfeo_matel_wrap(out,i,j)=H[ext(i,nu)+n*ext(j,nu)];return 0;
    }
    Index eval_Ggt(const Scalar*,const Scalar*,MAT* out,Index k)override{
        zero_matrix(out);if(k==0)for(int j=0;j<5;++j)blasfeo_matel_wrap(out,2+j,j)=1;return 0;
    }
    static int bound_external(int k,int j){return k==0?5+j:(j==0?3:4+j);}
    Index eval_Ggt_ineq(const Scalar*,const Scalar*,MAT* out,Index k)override{
        zero_matrix(out);int nu=get_nu(k);
        for(int j=0;j<get_ng_ineq(k);++j){int v=bound_external(k,j);int internal=v<5?nu+v:v-5;
            blasfeo_matel_wrap(out,internal,j)=1;}return 0;
    }
    Index eval_b(const Scalar* next,const Scalar* u,const Scalar* x,Scalar* out,Index)override{
        double one=1;const auto& f=dynamics(x,u,theta.data(),zero.data(),&one);
        for(int i=0;i<5;++i)out[i]=f[i]-next[i];return 0;
    }
    Index eval_g(const Scalar*,const Scalar* x,Scalar* out,Index k)override{
        if(k==0)for(int i=0;i<5;++i)out[i]=x[i]-initial[i];return 0;
    }
    Index eval_gineq(const Scalar* u,const Scalar* x,Scalar* out,Index k)override{
        for(int j=0;j<get_ng_ineq(k);++j){int v=bound_external(k,j);out[j]=v<5?x[v]:u[v-5];}return 0;
    }
    Index get_bounds(Scalar* lo,Scalar* hi,Index k)const override{
        for(int j=0;j<get_ng_ineq(k);++j){int v=bound_external(k,j);
            lo[j]=v==3?F8_SPEED_MIN:(v==5?-F8_ACCELERATION_MAX:-F8_STEERING_MAX);
            hi[j]=v==3?F8_SPEED_MAX:(v==5?F8_ACCELERATION_MAX:F8_STEERING_MAX);}return 0;
    }
    Index eval_rq(const Scalar* scale,const Scalar* u,const Scalar* x,Scalar* out,Index k)override{
        int nu=get_nu(k);const auto& g=(k<F8_N?gradient:terminal_gradient)(x,u?u:zero.data(),theta.data(),zero.data(),scale);
        for(int i=0;i<5+nu;++i)out[i]=g[ext(i,nu)];return 0;
    }
    Index eval_L(const Scalar* scale,const Scalar* u,const Scalar* x,Scalar* out,Index k)override{
        out[0]=(k<F8_N?cost:terminal_cost)(x,u?u:zero.data(),theta.data(),zero.data(),scale)[0];return 0;
    }
    Index get_initial_xk(Scalar* out,Index k)const override{std::copy_n(guess.data()+7*k,5,out);return 0;}
    Index get_initial_uk(Scalar* out,Index k)const override{if(k<F8_N)std::copy_n(guess.data()+7*k+5,2,out);return 0;}
};
#undef FN

using Cached=bool(*)(PdSolverOrig<OcpType>*,const double*,const double*,double*,double*);
static Cached cached_symbol(){
#ifdef _WIN32
    auto module=GetModuleHandleA("fatrop.dll");
    auto p=module?reinterpret_cast<Cached>(GetProcAddress(module,"fatrop_solve_cached_reduced_rhs")):nullptr;
#else
    auto p=reinterpret_cast<Cached>(dlsym(RTLD_DEFAULT,"fatrop_solve_cached_reduced_rhs"));
#endif
    require(p!=nullptr,"modified FATROP with fatrop_solve_cached_reduced_rhs is required");return p;
}

struct Ours final:Backend {
    std::shared_ptr<Problem> problem=std::make_shared<Problem>();
    std::unique_ptr<OptionRegistry> options=std::make_unique<OptionRegistry>();
    std::unique_ptr<IpAlgBuilder<OcpType>> builder;
    std::shared_ptr<IpAlgorithm<OcpType>> algorithm;
    Cached cached;
    Vec trajectory=Vec(nz),H=Vec(F8_N*49+25),J=Vec(F8_N*35),mixed=Vec(nz*4);
    Vec rf,rg,xf,xg;std::vector<int> active;Vec signs;
    Ours():builder(std::make_unique<IpAlgBuilder<OcpType>>(std::make_shared<NlpOcp>(problem))),
           algorithm(builder->with_options_registry(options.get()).build()),cached(cached_symbol()){
        options->set_option<Index>("print_level",0);
        options->set_option<Scalar>("tolerance",F8_FORWARD_TOLERANCE);
        options->set_option<Scalar>("tol_acceptable",F8_FORWARD_TOLERANCE);
        options->set_option<Scalar>("constr_viol_tol",F8_FORWARD_TOLERANCE);
        options->set_option<Index>("max_iter",F8_FORWARD_MAX_ITERATIONS);
        const auto& info=algorithm->info();
        rf.resize(info.number_of_tangent_variables+info.number_of_slack_variables);
        xf.resize(rf.size());rg.resize(info.number_of_eq_constraints);xg.resize(rg.size());
    }
    void forward(const double* theta,const double* x0,const double* guess,double* out)override{
        std::copy_n(theta,4,problem->theta.data());std::copy_n(x0,5,problem->initial.data());
        std::copy_n(guess,nz,problem->guess.data());
        auto start=Clock::now();auto status=algorithm->optimize();stats.forward_solve_s=seconds(start);
        stats.forward_iterations=static_cast<int>(builder->get_ipdata()->iteration_number());
        const auto& p=algorithm->solution_primal();const auto& info=algorithm->info();
        for(int k=0;k<=F8_N;++k){for(int j=0;j<5;++j)out[k*7+j]=p(info.offsets_primal_x[k]+j);
            if(k<F8_N)for(int j=0;j<2;++j)out[k*7+5+j]=p(info.offsets_primal_u[k]+j);}
        trajectory.assign(out,out+nz);trace["trajectory"]=trajectory;
        const auto& it=builder->get_ipdata()->current_iterate();
        const auto& dual=it.dual_eq();Vec v(dual.m());for(int i=0;i<dual.m();++i)v[i]=dual(i);trace["native_dual_eq"]=v;
        const auto& sl=it.primal_s();v.resize(sl.m());for(int i=0;i<sl.m();++i)v[i]=sl(i);trace["native_slacks"]=v;
        const auto& zl=it.dual_bounds_l();v.resize(zl.m());for(int i=0;i<zl.m();++i)v[i]=zl(i);trace["native_dual_lower"]=v;
        const auto& zu=it.dual_bounds_u();v.resize(zu.m());for(int i=0;i<zu.m();++i)v[i]=zu(i);trace["native_dual_upper"]=v;
        require(status==IpSolverReturnFlag::Success,"FATROP forward status "+std::to_string(static_cast<int>(status)));
    }
    void build_target(){
        active.clear();signs.clear();double one=1;
        auto& it=builder->get_ipdata()->current_iterate();const auto& info=algorithm->info();
        // Lazy evaluation is at the returned iterate, not the last cached Newton point.
        auto& hessian=it.hessian();
        require(it.objective_scale>0,"invalid FATROP objective scale");
        Vec lo,hi;bounds(lo,hi);
        for(int k=0;k<=F8_N;++k){
            int nu=k<F8_N?2:0,n=5+nu;const auto& mat=hessian.RSQrqt[k].mat();
            double* block=H.data()+k*49;
            for(int i=0;i<n;++i)for(int j=0;j<n;++j){
                int ii=i<5?nu+i:i-5,jj=j<5?nu+j:j-5;
                block[i*n+j]=blasfeo_matel_wrap(&mat,std::max(ii,jj),std::min(ii,jj))/it.objective_scale;
            }
            const double* x=trajectory.data()+k*7;const double* u=k<F8_N?x+5:problem->zero.data();
            const auto& m=(k<F8_N?problem->mixed:problem->terminal_mixed)(x,u,problem->theta.data(),problem->zero.data(),&one);
            for(int i=0;i<n;++i)for(int p=0;p<4;++p)mixed[(k*7+i)*4+p]=m[i+n*p];
            if(k<F8_N){const auto& j=problem->jacobian(x,u,problem->theta.data(),problem->zero.data(),&one);
                for(int i=0;i<5;++i)for(int z=0;z<7;++z)J[k*35+i*7+z]=j[i+5*z];}
            for(int j=0;j<problem->get_ng_ineq(k);++j){
                int v=k*7+Problem::bound_external(k,j),s=info.offsets_slack[k]+j;
                double dl=trajectory[v]-lo[v],du=hi[v]-trajectory[v];
                require(std::min(dl,du)>=-10*F8_FORWARD_TOLERANCE,"returned bound violation");
                if(std::min(dl,du)<=F8_ACTIVE_TOLERANCE){
                    double multiplier=(dl<du?it.dual_bounds_l()(s):it.dual_bounds_u()(s))/it.objective_scale;
                    require(multiplier>F8_DUAL_TOLERANCE,"weakly active bound: classical local derivative is ambiguous");
                    active.push_back(v);signs.push_back(dl<du?-1.:1.);
                }
            }
        }
        stats.active_constraints=static_cast<int>(active.size());
        trace["active_indices"]=Vec(active.begin(),active.end());trace["active_signs"]=signs;
        trace["target_hessian_blocks_row_major"]=H;trace["dynamics_jacobians_row_major"]=J;
        trace["cost_mixed_row_major"]=mixed;
    }
    void target(const Vec& x,Vec& y){
        preparation([&]{y.assign(x.size(),0.);});
        for(int k=0;k<=F8_N;++k){int n=k<F8_N?7:5;const double* h=H.data()+49*k;
            for(int i=0;i<n;++i)for(int j=0;j<n;++j)y[k*7+i]+=h[i*n+j]*x[k*7+j];}
        for(int j=0;j<5;++j){y[j]+=x[nz+j];y[nz+j]=x[j];}
        for(int k=0;k<F8_N;++k)for(int i=0;i<5;++i){
            int e=nz+5*(k+1)+i,next=7*(k+1)+i;
            y[e]=x[next];y[next]+=x[e];
            for(int j=0;j<7;++j){double a=J[k*35+i*7+j];y[e]-=a*x[k*7+j];y[k*7+j]-=a*x[e];}
        }
        for(size_t i=0;i<active.size();++i){int e=nz+neq+static_cast<int>(i),v=active[i];
            y[e]=signs[i]*x[v];y[v]+=signs[i]*x[e];}
    }
    void precondition(const Vec& b,Vec& x){
        pause_preparation();
        const auto& info=algorithm->info();std::fill(rf.begin(),rf.end(),0);std::fill(rg.begin(),rg.end(),0);
        for(int k=0;k<=F8_N;++k){for(int i=0;i<5;++i)rf[info.offsets_tangent_x[k]+i]=b[k*7+i];
            if(k<F8_N)for(int i=0;i<2;++i)rf[info.offsets_tangent_u[k]+i]=b[k*7+5+i];}
        for(int i=0;i<5;++i)rg[info.offsets_g_eq_path[0]+i]=b[nz+i];
        for(int k=0;k<F8_N;++k)for(int i=0;i<5;++i)rg[info.offsets_g_eq_dyn[k]+i]=b[nz+5*(k+1)+i];
        auto* pd=builder->get_pd_solver().get();
        resume_preparation();
        require(cached(pd,rf.data(),rg.data(),xf.data(),xg.data()),"retained FATROP solve failed");
        ExcludePreparation unpacking;
        x.assign(b.size(),0.);
        for(int k=0;k<=F8_N;++k){for(int i=0;i<5;++i)x[k*7+i]=xf[info.offsets_tangent_x[k]+i];
            if(k<F8_N)for(int i=0;i<2;++i)x[k*7+5+i]=xf[info.offsets_tangent_u[k]+i];}
        for(int i=0;i<5;++i)x[nz+i]=xg[info.offsets_g_eq_path[0]+i];
        for(int k=0;k<F8_N;++k)for(int i=0;i<5;++i)x[nz+5*(k+1)+i]=xg[info.offsets_g_eq_dyn[k]+i];
        // Active-set multiplier coordinates are not in the reduced cache.
        // Identity extension is explicit; it does not claim their factors were retained.
        for(size_t i=0;i<active.size();++i)x[nz+neq+i]=b[nz+neq+i];
    }
    void backward(const double* seed,double* gradient)override{
        const auto native_start=Clock::now(),build_start=Clock::now();
        build_target();stats.target_build_s=seconds(build_start);
        Vec rhs(nz+neq+active.size());std::copy_n(seed,nz,rhs.data());
        trace["adjoint_rhs"]=rhs;Vec history,a;int iterations=0;
        NumericalTimer numerical_timer;
        try {
            a=reused_gmres([this](const Vec& x,Vec& y){target(x,y);},
                          [this](const Vec& x,Vec& y){precondition(x,y);},rhs,history,iterations);
        }catch(...){
            stats.backward_solve_s=numerical_timer.finish();stats.backward_native_s=seconds(native_start);
            stats.backward_iterations=iterations;trace["true_relative_residual_history"]=history;
            if(!history.empty()){stats.initial_relative_residual=history.front();stats.final_relative_residual=history.back();}
            throw;
        }
        stats.backward_solve_s=numerical_timer.finish();stats.backward_iterations=iterations;
        stats.initial_relative_residual=history.front();stats.final_relative_residual=history.back();
        std::fill_n(gradient,4,0.);for(int i=0;i<nz;++i)for(int p=0;p<4;++p)gradient[p]-=a[i]*mixed[i*4+p];
        trace["adjoint"]=a;trace["initial_state_gradient"]=Vec(a.begin()+nz,a.begin()+nz+5);trace["true_relative_residual_history"]=history;
        stats.backward_native_s=seconds(native_start);
    }
};
std::unique_ptr<Backend> make_backend(const std::string&){return std::make_unique<Ours>();}
const char* backend_name(){return "Ours";}
const char* backend_details(){return "FATROP explicit bounds; retained reduced Riccati solves; active-set target; identity extension on active multipliers; warm-started right-GMRES; timing=solve-kernels-v3";}
}
