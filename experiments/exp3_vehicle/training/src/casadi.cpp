#include "common.hpp"
#include "solve_timing.hpp"
#include <casadi/casadi.hpp>
#include <filesystem>
namespace f8 {
using casadi::DM;using casadi::Function;using casadi::DMDict;using casadi::Dict;
struct Casadi final:Backend {
    Function solver,reverse;
    ExternalSolveTimer linear_timer{"frs_casadi_solve_timer","casadi.dll"};
    DMDict args,result;
    explicit Casadi(const std::string& assets){
        Function oracle=Function::load((std::filesystem::path(assets)/"ocp.casadi").string());
        Dict options;options["print_time"]=false;options["record_time"]=true;
        options["ipopt.print_level"]=0;options["ipopt.sb"]="yes";
        options["ipopt.tol"]=F8_FORWARD_TOLERANCE;options["ipopt.constr_viol_tol"]=F8_FORWARD_TOLERANCE;
        options["ipopt.max_iter"]=F8_FORWARD_MAX_ITERATIONS;
        options["ipopt.nlp_scaling_method"]="none";
        options["ipopt.bound_relax_factor"]=0.0;
        // Native NLP sensitivities infer active bounds from multipliers.
        // IPOPT can leave tiny nonzero multipliers on inactive bounds;
        // let its CasADi interface clear them using the shared activity tolerance.
        // https://web.casadi.org/api/html/d4/d89/group__nlpsol.html
        // Keep solver.reverse(1): no custom derivative or extra SQP solve.
        options["clip_inactive_lam"]=true;
        options["inactive_lam_strategy"]="abstol";
        options["inactive_lam_value"]=F8_ACTIVE_TOLERANCE;
        // CasADi constructs its own active-set derivative and sensitivity linear solver.
        solver=casadi::nlpsol("figure8_ipopt","ipopt",oracle,options);
        reverse=solver.reverse(1); // No sIPOPT and no hand-built KKT differentiation here.
        Vec lo,hi;bounds(lo,hi);args["lbx"]=DM(lo);args["ubx"]=DM(hi);
        args["lbg"]=DM::zeros(neq);args["ubg"]=DM::zeros(neq);
    }
    void forward(const double* theta,const double* initial,const double* guess,double* out)override{
        Vec p(theta,theta+4);p.insert(p.end(),initial,initial+5);
        args["p"]=DM(p);args["x0"]=DM(Vec(guess,guess+nz));
        args["lam_x0"]=DM::zeros(nz);args["lam_g0"]=DM::zeros(neq);
        auto start=Clock::now();result=solver(args);stats.forward_solve_s=seconds(start);
        const auto s=solver.stats();
        auto copy=[&](const char* key){DM v=DM::densify(result.at(key));return v.nonzeros();};
        Vec trajectory=copy("x");std::copy_n(trajectory.data(),nz,out);trace["trajectory"]=trajectory;
        trace["native_lam_x"]=copy("lam_x");trace["native_lam_g"]=copy("lam_g");
        if(s.count("iter_count"))stats.forward_iterations=static_cast<int>(s.at("iter_count").to_int());
        require(s.count("success")&&s.at("success").to_bool(),"CasADi-IPOPT forward did not report success");
    }
    void backward(const double* seed,double* gradient)override{
        DMDict input;
        for(casadi_int i=0;i<reverse.n_in();++i){
            std::string key=reverse.name_in(i);
            if(key=="adj_x")input[key]=DM(Vec(seed,seed+nz));
            else if(key.rfind("adj_",0)==0)input[key]=DM::zeros(reverse.sparsity_in(i));
            else if(key.rfind("out_",0)==0)input[key]=result.at(key.substr(4));
            else if(args.count(key))input[key]=args.at(key);
            else input[key]=DM::zeros(reverse.sparsity_in(i));
        }
        linear_timer.reset();
        const auto start=Clock::now();DMDict answer=reverse(input);stats.backward_native_s=seconds(start);
        // The public reverse Function timer includes derivative evaluation and contraction.
        // Do not label it as an isolated numerical linear solve.
        stats.backward_solve_s=linear_timer.read();stats.target_build_s=nan();
        Vec g=DM::densify(answer.at("adj_p")).nonzeros();require(g.size()==9,"unexpected CasADi adj_p size");
        std::copy_n(g.data(),4,gradient);trace["native_adj_p_including_initial_state"]=g;
        trace["initial_state_gradient"]=Vec(g.begin()+4,g.end());
    }
};
std::unique_ptr<Backend> make_backend(const std::string& assets){return std::make_unique<Casadi>(assets);}
const char* backend_name(){return "CasADi-IPOPT";}
const char* backend_details(){return "CasADi nlpsol(ipopt).reverse(1); timing=solve-kernels-v3; numerical factorization and solve hooks; missing dependency hooks yield NaN";}
}
