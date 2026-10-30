#include "common.hpp"
#include "solve_timing.hpp"
extern "C" {
#include "acados_solver_f8_forward.h"
#include "acados_solver_f8_sensitivity.h"
#include "acados_c/ocp_nlp_interface.h"
}
namespace f8 {
struct Acados final:Backend {
    ExternalSolveTimer linear_timer{"frs_acados_solve_timer","acados.dll"};
    f8_forward_solver_capsule* f=nullptr;
    f8_sensitivity_solver_capsule* s=nullptr;
    Acados(){
        f=f8_forward_acados_create_capsule();s=f8_sensitivity_acados_create_capsule();
        require(f&&s,"acados capsule allocation failed");
        require(f8_forward_acados_create(f)==0,"acados forward creation failed");
        require(f8_sensitivity_acados_create(s)==0,"acados sensitivity creation failed");
        // A QP solved only to the NLP tolerance can stall nonlinear stationarity.
        // Keep the requested NLP tolerance unchanged and solve its QPs more accurately.
        double qp_tolerance=1e-10;
        for(const char* field:{"qp_tol_stat","qp_tol_eq","qp_tol_ineq","qp_tol_comp"})
            ocp_nlp_solver_opts_set(f->nlp_config,f->nlp_opts,field,&qp_tolerance);
    }
    ~Acados()override{
        if(f){f8_forward_acados_free(f);f8_forward_acados_free_capsule(f);}
        if(s){f8_sensitivity_acados_free(s);f8_sensitivity_acados_free_capsule(s);}
    }
    static void set(ocp_nlp_config* cfg,ocp_nlp_dims* dims,ocp_nlp_out* out,ocp_nlp_in* in,
                    int k,const char* field,const double* value){
        ocp_nlp_out_set(cfg,dims,out,in,k,field,const_cast<double*>(value));
    }
    static Vec get(ocp_nlp_config* cfg,ocp_nlp_dims* dims,ocp_nlp_out* out,int k,const char* field){
        int n=ocp_nlp_dims_get_from_attr(cfg,dims,out,k,field);Vec v(n);
        if(n)ocp_nlp_out_get(cfg,dims,out,k,field,v.data());return v;
    }
    void forward(const double* theta,const double* x0,const double* guess,double* out)override{
        require(f8_forward_acados_set_p_global_and_precompute_dependencies(f,const_cast<double*>(theta),4)==0,"acados p_global update failed");
        require(f8_sensitivity_acados_set_p_global_and_precompute_dependencies(s,const_cast<double*>(theta),4)==0,"acados sensitivity p_global update failed");
        ocp_nlp_out_set_values_to_zero(f->nlp_config,f->nlp_dims,f->nlp_out);
        for(const char* field:{"lbx","ubx"}){
            require(ocp_nlp_constraints_model_set(f->nlp_config,f->nlp_dims,f->nlp_in,f->nlp_out,0,field,const_cast<double*>(x0))==0,"acados x0 update failed");
            require(ocp_nlp_constraints_model_set(s->nlp_config,s->nlp_dims,s->nlp_in,s->nlp_out,0,field,const_cast<double*>(x0))==0,"acados sensitivity x0 update failed");
        }
        for(int k=0;k<=F8_N;++k){set(f->nlp_config,f->nlp_dims,f->nlp_out,f->nlp_in,k,"x",guess+7*k);
            if(k<F8_N)set(f->nlp_config,f->nlp_dims,f->nlp_out,f->nlp_in,k,"u",guess+7*k+5);}
        auto start=Clock::now();int status=f8_forward_acados_solve(f);stats.forward_solve_s=seconds(start);
        ocp_nlp_get(f->nlp_solver,"sqp_iter",&stats.forward_iterations);
        Vec multipliers;
        for(int k=0;k<=F8_N;++k){
            Vec x=get(f->nlp_config,f->nlp_dims,f->nlp_out,k,"x");std::copy_n(x.data(),5,out+7*k);
            if(k<F8_N){Vec u=get(f->nlp_config,f->nlp_dims,f->nlp_out,k,"u");std::copy_n(u.data(),2,out+7*k+5);}
            for(const char* field:{"lam","pi"}){
                if(k==F8_N&&std::string(field)=="pi")continue;
                Vec v=get(f->nlp_config,f->nlp_dims,f->nlp_out,k,field);
                trace[std::string("native_")+field+"_stage_"+std::to_string(k)]=v;
            }
        }
        trace["trajectory"]=Vec(out,out+nz);require(status==0,"acados SQP status "+std::to_string(status));
    }
    void backward(const double* seed,double* gradient)override{
        // Transfer accepted primal AND dual iterates to the exact, unregularized solver.
        // A MIRROR-regularized forward factorization is NOT differentiated as exact.
        for(int k=0;k<=F8_N;++k)for(const char* field:{"x","u","pi","lam","sl","su"}){
            if(k==F8_N&&(std::string(field)=="u"||std::string(field)=="pi"))continue;
            Vec v=get(f->nlp_config,f->nlp_dims,f->nlp_out,k,field);
            if(!v.empty())set(s->nlp_config,s->nlp_dims,s->nlp_out,s->nlp_in,k,field,v.data());
        }
        ocp_nlp_out_set_values_to_zero(s->nlp_config,s->nlp_dims,s->sens_out);
        for(int k=0;k<=F8_N;++k){set(s->nlp_config,s->nlp_dims,s->sens_out,s->nlp_in,k,"x",seed+7*k);
            if(k<F8_N)set(s->nlp_config,s->nlp_dims,s->sens_out,s->nlp_in,k,"u",seed+7*k+5);}
        linear_timer.reset();
        auto start=Clock::now();
        int status=ocp_nlp_setup_qp_matrices_and_factorize(s->nlp_solver,s->nlp_in,s->nlp_out);
        if(status){stats.backward_solve_s=linear_timer.read();stats.backward_native_s=seconds(start);throw std::runtime_error("acados exact sensitivity factorization failed: "+std::to_string(status));}
        // Match AcadosOcpSolver.eval_adjoint_solution_sensitivity: update the
        // parameter Jacobians at the transferred solution before the adjoint.
        // Keep this required work inside the native backward timer.
        ocp_nlp_eval_params_jac(s->nlp_solver,s->nlp_in,s->nlp_out);
        ocp_nlp_eval_solution_sens_adj_p(s->nlp_solver,s->nlp_in,s->sens_out,"p_global",0,gradient);
        // Native initial-state sensitivities sharing the exact factorization.
        // Include the five extra solves in backward timings. No finite differences.
        Vec gx(5,0.);char field[]="ex";
        for(int j=0;j<5;++j){
            ocp_nlp_eval_param_sens(s->nlp_solver,field,0,j,s->sens_out);
            for(int k=0;k<=F8_N;++k){
                Vec dx=get(s->nlp_config,s->nlp_dims,s->sens_out,k,"x");
                for(int i=0;i<5;++i)gx[j]+=seed[7*k+i]*dx[i];
                if(k<F8_N){Vec du=get(s->nlp_config,s->nlp_dims,s->sens_out,k,"u");
                    for(int i=0;i<2;++i)gx[j]+=seed[7*k+5+i]*du[i];}
            }
        }
        trace["initial_state_gradient"]=gx;
        stats.backward_native_s=seconds(start);
        // setup_qp_matrices_and_factorize mixes assembly and factorization. Keep both
        // in native timing; do not hide the new factorization as untimed setup.
        stats.backward_solve_s=linear_timer.read();stats.target_build_s=nan();
    }
};
std::unique_ptr<Backend> make_backend(const std::string&){return std::make_unique<Acados>();}
const char* backend_name(){return "acados";}
const char* backend_details(){return "acados SQP/HPIPM; exact sensitivity; timing=solve-kernels-v3; QP factorization and adjoint solve hooks; missing dependency hooks yield NaN";}
}
