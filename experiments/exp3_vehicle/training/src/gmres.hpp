#pragma once
#include "common.hpp"
#include "solve_timing.hpp"
#include <functional>
#include <numeric>

namespace f8 {
using Action=std::function<void(const Vec&,Vec&)>;
inline double norm(const Vec& x){double s=0;for(double v:x)s=std::hypot(s,v);return s;}
/* Fixed right preconditioning, cached initial solution, two-pass MGS.
   Every reported residual is explicitly evaluated against the target operator. */
inline Vec reused_gmres(const Action& A,const Action& M,const Vec& b,Vec& history,int& iterations) {
    const int n=static_cast<int>(b.size()), m=std::min(n,F8_BACKWARD_MAX_ITERATIONS);
    pause_preparation();
    Vec x(n),r(n),Ax(n);
    resume_preparation();
    iterations=0;double bn=norm(b);
    if(bn==0){preparation([&]{history.push_back(0);});return x;}
    M(b,x);A(x,Ax);for(int i=0;i<n;++i)r[i]=b[i]-Ax[i];
    double beta=norm(r);const double initial_relative=beta/bn;
    preparation([&]{history.push_back(initial_relative);});
    if(beta<=F8_BACKWARD_TOLERANCE*bn)return x;
    pause_preparation();
    std::vector<Vec> V(m+1,Vec(n)),P(m,Vec(n));
    Vec H((m+1)*m),cs(m),sn(m),q(m+1),y(m),candidate(n),w(n);
    resume_preparation();
    q[0]=beta;for(int i=0;i<n;++i)V[0][i]=r[i]/beta;
    auto h=[&](int row,int col)->double&{return H[col*(m+1)+row];};
    for(int j=0;j<m;++j){
        M(V[j],P[j]);A(P[j],w);const double before=norm(w);
        for(int pass=0;pass<2;++pass)for(int i=0;i<=j;++i){
            double t=std::inner_product(w.begin(),w.end(),V[i].begin(),0.0);h(i,j)+=t;
            for(int z=0;z<n;++z)w[z]-=t*V[i][z];
        }
        h(j+1,j)=norm(w);
        bool breakdown=h(j+1,j)<=64*std::numeric_limits<double>::epsilon()*std::max(before,1e-300);
        if(!breakdown)for(int z=0;z<n;++z)V[j+1][z]=w[z]/h(j+1,j);
        for(int i=0;i<j;++i){double a=h(i,j),b1=h(i+1,j);h(i,j)=cs[i]*a+sn[i]*b1;h(i+1,j)=-sn[i]*a+cs[i]*b1;}
        double den=std::hypot(h(j,j),h(j+1,j));
        require(std::isfinite(den)&&den>0,"GMRES singular projected system");
        cs[j]=h(j,j)/den;sn[j]=h(j+1,j)/den;h(j,j)=den;h(j+1,j)=0;
        q[j+1]=-sn[j]*q[j];q[j]*=cs[j];
        for(int i=j;i>=0;--i){double v=q[i];for(int z=i+1;z<=j;++z)v-=h(i,z)*y[z];
            require(std::abs(h(i,i))>0,"GMRES triangular breakdown");y[i]=v/h(i,i);}
        preparation([&]{candidate=x;});
        for(int i=0;i<=j;++i)for(int z=0;z<n;++z)candidate[z]+=P[i][z]*y[i];
        A(candidate,Ax);for(int z=0;z<n;++z)r[z]=b[z]-Ax[z];
        double relative=norm(r)/bn;
        preparation([&]{history.push_back(relative);});iterations=j+1;
        require(std::isfinite(relative),"GMRES nonfinite true residual");
        if(relative<=F8_BACKWARD_TOLERANCE)return candidate;
        require(!breakdown,"GMRES Arnoldi breakdown before true target convergence");
    }
    throw std::runtime_error("GMRES iteration limit before true target convergence");
}
}
