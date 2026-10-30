#include "f8.h"
#include "model_config.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
using V=std::vector<double>; using X=std::array<double,5>;using T=std::array<double,4>;
using Clock=std::chrono::steady_clock;
constexpr double heading_weight=.01,speed_weight=.1;
struct Step {X x;std::array<double,2> u;V guess;};
struct Run {double loss=0,rmse=0,maxerror=0,inward=0,inside_fraction=0,maxx=0,maxy=0,forward_s=0,backward_s=0,recompute_s=0;int count=0;std::vector<Step> steps;};
void require(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
using Solver=std::unique_ptr<F8Solver,decltype(&f8_destroy)>;
Solver solver(){const char* assets=std::getenv("F8_ASSETS");Solver s(f8_create(assets?assets:F8_ASSETS_DIRECTORY),f8_destroy);require(bool(s),f8_last_error());return s;}
void theta_file(const std::filesystem::path& p,const T& theta){std::ofstream f(p);f<<f8_model_id()<<'\n'<<std::setprecision(17);for(double t:theta)f<<t<<' ';f<<'\n';}
double phase_arc_center(double p){
 const double pi=acos(-1.),c=F8_PATH_A,r=F8_PATH_B,d=sqrt(c*c-r*r),a=r*(pi+2*asin(r/c)),len=4*d+2*a;
 double s=(p-2*pi*floor(p/(2*pi)))*len/(2*pi);
 if(s>d+.15 && s<d+a-.15)return c;
 if(s>3*d+a+.15 && s<3*d+2*a-.15)return -c;
 return 0;
}
Run rollout(const T& theta,int n,const std::filesystem::path& csv={},bool tape=false){
 auto sol=solver();Run run;run.count=n;X x;double ref[5],ur[2];f8_reference(0,x.data(),ur);x[3]=0;
 V w(f8_trajectory_size()),guess(w.size());bool warm=false;int arcs=0,inside=0;
 std::ofstream f;if(!csv.empty()){f.open(csv);f<<std::setprecision(17)<<"step,time,x,y,yaw,speed,phase,ref_x,ref_y,ref_yaw,ref_speed,a,delta,error,inward,solve_s\n";}
 for(int k=0;k<n;++k){
  f8_reference(x[4],ref,ur);double ex=x[0]-ref[0],ey=x[1]-ref[1],eh=x[2]-ref[2],ev=x[3]-ref[3];
  double e2=ex*ex+ey*ey;run.loss+=(.5*e2+heading_weight*(1-cos(eh))+.5*speed_weight*ev*ev)/n;
  run.rmse+=e2/n;run.maxerror=std::max(run.maxerror,sqrt(e2));run.maxx=std::max(run.maxx,std::abs(x[0]));run.maxy=std::max(run.maxy,std::abs(x[1]));
  double center=phase_arc_center(x[4]),in=0;if(center!=0){in=F8_PATH_B-hypot(x[0]-center,x[1]);run.inward+=in;++arcs;if(in>.01)++inside;}
  if(warm)f8_shift_guess(w.data(),x.data(),guess.data());else f8_initial_guess(x.data(),guess.data());
  F8Stats stats;int status=f8_forward(sol.get(),theta.data(),x.data(),guess.data(),w.data(),&stats);require(!status,"forward step="+std::to_string(k)+": "+f8_last_error());
  run.forward_s+=stats.forward_solve_s;double u[2]={w[5],w[6]};
  if(tape)run.steps.push_back({x,{u[0],u[1]},guess});
  if(f)f<<k<<','<<k*F8_DT<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<x[3]<<','<<x[4]<<','<<ref[0]<<','<<ref[1]<<','<<ref[2]<<','<<ref[3]<<','<<u[0]<<','<<u[1]<<','<<sqrt(e2)<<','<<in<<','<<stats.forward_solve_s<<'\n';
  X next;f8_dynamics(x.data(),u,next.data());x=next;warm=true;
 }
 run.rmse=sqrt(run.rmse);if(arcs){run.inward/=arcs;run.inside_fraction=double(inside)/arcs;}return run;
}
// Full closed-loop reverse accumulation. Recompute each MPC at its recorded
// initial state and warm start, then use each backend's implicit derivative
// with respect to both the weights and the measured initial state.
T gradient(const T& theta,Run& run,const std::filesystem::path& csv={}){
 std::ofstream timing;if(!csv.empty()){timing.open(csv);timing<<std::setprecision(17)<<"step,recompute_forward_s,backward_solve_s,backward_native_s,backward_iterations,residual\n";}
 auto sol=solver();T g{};X lambda{};V w(f8_trajectory_size()),seed(w.size());
 for(int k=run.count-1;k>=0;--k){
  const auto& st=run.steps.at(k);F8Stats stats;
  require(!f8_forward(sol.get(),theta.data(),st.x.data(),st.guess.data(),w.data(),&stats),"backward recompute: "+std::string(f8_last_error()));
  const double recompute=stats.forward_solve_s;run.recompute_s+=recompute;
  require(std::abs(w[5]-st.u[0])+std::abs(w[6]-st.u[1])<1e-5,"recomputed MPC differs from original branch");
  const double v=st.x[3],yaw=st.x[2],delta=st.u[1];
  std::fill(seed.begin(),seed.end(),0.);
  seed[5]=F8_DT*lambda[3];
  seed[6]=F8_DT*v/(F8_WHEELBASE*cos(delta)*cos(delta))*lambda[2];
  double dg[4];require(!f8_backward(sol.get(),seed.data(),dg,&stats),"backward step="+std::to_string(k)+": "+f8_last_error());run.backward_s+=stats.backward_solve_s;
  if(timing.is_open())timing<<k<<','<<recompute<<','<<stats.backward_solve_s<<','<<stats.backward_native_s<<','<<stats.backward_iterations<<','<<stats.final_relative_residual<<'\n';
  const double* adj=nullptr;size_t count=0;
  for(int j=0;j<f8_trace_count(sol.get());++j)if(std::string(f8_trace_name(sol.get(),j))=="initial_state_gradient")adj=f8_trace_values(sol.get(),j,&count);
  require(adj && count==5,"missing initial-state sensitivity");
  require(std::all_of(adj,adj+5,[](double v){return std::isfinite(v);}),"non-finite initial-state sensitivity");
  X next=lambda;
  next[2]+=F8_DT*v*(-sin(yaw)*lambda[0]+cos(yaw)*lambda[1]);
  next[3]+=F8_DT*(cos(yaw)*lambda[0]+sin(yaw)*lambda[1]+tan(delta)/F8_WHEELBASE*lambda[2]);
  double ref[5],u[2];f8_reference(st.x[4],ref,u);
  next[0]+=(st.x[0]-ref[0])/run.count;next[1]+=(st.x[1]-ref[1])/run.count;
  next[2]+=heading_weight*sin(st.x[2]-ref[2])/run.count;
  next[3]+=speed_weight*(st.x[3]-ref[3])/run.count;
  for(int j=0;j<4;++j){g[j]+=dg[j];next[j]+=adj[j];}
  // Phase is deterministic and independent of theta; its cotangent cannot
  // affect the first four states or controls through the plant dynamics.
  next[4]=0;lambda=next;
 }
 return g;
}
void metric(std::ostream& f,int id,const T& t,const Run& r,double seconds,const std::string& status="ok"){
 f<<id;for(double v:t)f<<','<<exp(v);f<<','<<r.loss<<','<<r.rmse<<','<<r.maxerror<<','<<r.inward<<','<<r.inside_fraction<<','<<r.maxx<<','<<r.maxy<<','<<r.forward_s<<','<<r.backward_s<<','<<r.recompute_s<<','<<seconds<<','<<status<<'\n';f.flush();
}
void header(std::ostream& f){f<<std::setprecision(17)<<"id,w_position,w_heading,w_speed,w_steering,loss,rmse,max_error,mean_inward,inside_fraction,max_abs_x,max_abs_y,forward_s,backward_s,recompute_forward_s,wall_s,status\n";}
int main(int argc,char** argv){try{
 require(argc>=3,"Usage: reference_learning search|train|check|eval NEW_OUT [w_position w_heading w_speed w_steering] [epochs] [learning_rate]");
 std::string mode=argv[1];std::filesystem::path out=argv[2];require(!std::filesystem::exists(out),"output already exists");std::filesystem::create_directories(out);
 int n=int(ceil(2*acos(-1.)/F8_PHASE_RATE/F8_DT));
 T t;f8_initial_theta(t.data());if(argc>=7)for(int j=0;j<4;++j){double v=std::stod(argv[3+j]);require(std::isfinite(v)&&v>0&&log(v)>=-6&&log(v)<=6,"weights must be finite, positive, and have log values in [-6,6]");t[j]=log(v);}
 if(mode=="search"){
  std::ofstream f(out/"search.csv");header(f);std::mt19937 gen(20260911);std::uniform_real_distribution<double> dist(-5.5,5.5);
  std::vector<T> candidates;
  for(double p:{.01,.1,1.})for(double h:{.01,1.,50.})for(double d:{.003,.1,10.})candidates.push_back({log(p),log(h),log(1.),log(d)});
  for(int i=0;i<65;++i){T q;for(double& v:q)v=dist(gen);candidates.push_back(q);}
  int id=0;for(const auto& q:candidates){auto start=Clock::now();try{auto r=rollout(q,n,out/("candidate_"+std::to_string(id)+".csv"));metric(f,id,q,r,std::chrono::duration<double>(Clock::now()-start).count());if(id%5==0)std::cout<<"candidate="<<id<<" rmse="<<r.rmse<<" inward="<<r.inward<<std::endl;}catch(const std::exception& e){Run r;metric(f,id,q,r,0,"FAILED");std::cerr<<id<<": "<<e.what()<<std::endl;}++id;}return 0;
 }
 if(mode=="eval"){auto r=rollout(t,n,out/"rollout.csv");std::ofstream f(out/"metrics.csv");header(f);metric(f,0,t,r,0);theta_file(out/"theta.txt",t);return 0;}
 if(mode=="check"){
  auto r=rollout(t,n,{},true);auto g=gradient(t,r);std::ofstream f(out/"gradient_check.csv");f<<std::setprecision(17)<<"parameter,analytic,finite_difference,relative_error\n";
  double worst=0;for(int j=0;j<4;++j){T p=t,m=t;double eps=1e-3;p[j]+=eps;m[j]-=eps;double fd=(rollout(p,n).loss-rollout(m,n).loss)/(2*eps);double err=std::abs(fd-g[j])/std::max(1e-7,std::max(std::abs(fd),std::abs(g[j])));f<<j<<','<<g[j]<<','<<fd<<','<<err<<'\n';std::cout<<j<<" analytic="<<g[j]<<" fd="<<fd<<" rel="<<err<<std::endl;require(std::isfinite(err),"non-finite gradient check");worst=std::max(worst,err);}require(worst<.02,"closed-loop gradient check exceeded 2% relative tolerance");return 0;
 }
 require(mode=="train","unknown mode");int epochs=argc>7?std::stoi(argv[7]):500;double rate=argc>8?std::stod(argv[8]):.008;require(epochs>0&&std::isfinite(rate)&&rate>0,"epochs and learning rate must be positive");T m{},v{};std::ofstream f(out/"learning.csv");header(f);theta_file(out/"theta_initial.txt",t);
 std::ofstream optimizer(out/"optimizer.csv");optimizer<<std::setprecision(17)<<"epoch,gradient0,gradient1,gradient2,gradient3,theta0,theta1,theta2,theta3\n";
 std::ofstream manifest(out/"manifest.json");manifest<<std::setprecision(17)<<"{\n  \"model_id\": \""<<f8_model_id()<<"\",\n  \"backend\": \""<<f8_backend()<<"\",\n  \"simulation_only\": true,\n  \"epochs\": "<<epochs<<",\n  \"steps\": "<<n<<",\n  \"learning_rate\": "<<rate<<",\n  \"gradient\": \"full closed-loop implicit reverse accumulation\",\n  \"objective\": \"mean(0.5 position_error_squared + 0.01 (1-cos(heading_error)) + 0.05 speed_error_squared)\"\n}\n";manifest.close();
 for(int epoch=0;epoch<=epochs;++epoch){auto start=Clock::now();auto r=rollout(t,n,out/("epoch_"+std::to_string(epoch)+".csv"),epoch<epochs);T g{};if(epoch<epochs)g=gradient(t,r,out/("backward_epoch_"+std::to_string(epoch)+".csv"));metric(f,epoch,t,r,std::chrono::duration<double>(Clock::now()-start).count());std::cout<<"epoch="<<epoch<<" loss="<<r.loss<<" rmse="<<r.rmse<<" inward="<<r.inward<<std::endl;theta_file(out/("theta_"+std::to_string(epoch)+".txt"),t);if(epoch==epochs)break;
  optimizer<<epoch;for(double z:g)optimizer<<','<<z;for(double z:t)optimizer<<','<<z;optimizer<<'\n';optimizer.flush();
  for(int j=0;j<4;++j){m[j]=.9*m[j]+.1*g[j];v[j]=.999*v[j]+.001*g[j]*g[j];t[j]=std::clamp(t[j]-rate*(m[j]/(1-pow(.9,epoch+1)))/(sqrt(v[j]/(1-pow(.999,epoch+1)))+1e-10),-6.,6.);}
 }theta_file(out/"theta_final.txt",t);std::ofstream done(out/"status.json");done<<"{\"status\":\"complete\"}\n";return 0;
 }catch(const std::exception& e){std::cerr<<"FAILED: "<<e.what()<<std::endl;return 1;}}
