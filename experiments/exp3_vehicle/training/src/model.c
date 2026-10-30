#include "f8.h"
#include "model_config.h"
#include <math.h>
#include <string.h>

void f8_reference(double p,double x[5],double u[2]) {
    const double pi=acos(-1.0),c=F8_PATH_A,r=F8_PATH_B;
    const double alpha=asin(r/c),d=sqrt(c*c-r*r),arc=r*(pi+2*alpha);
    const double length=4*d+2*arc,co=cos(alpha),si=sin(alpha);
    const double s=(p-2*pi*floor(p/(2*pi)))*length/(2*pi);
    double curvature=0;
    if(s<d) {
        x[0]=s*co;x[1]=s*si;x[2]=alpha;
    } else if(s<d+arc) {
        double th=pi/2+alpha-(s-d)/r;
        x[0]=c+r*cos(th);x[1]=r*sin(th);x[2]=th-pi/2;curvature=-1/r;
    } else if(s<3*d+arc) {
        x[0]=(2*d+arc-s)*co;x[1]=(s-2*d-arc)*si;x[2]=pi-alpha;
    } else if(s<3*d+2*arc) {
        double th=pi/2-alpha+(s-3*d-arc)/r;
        x[0]=-c+r*cos(th);x[1]=r*sin(th);x[2]=th+pi/2;curvature=1/r;
    } else {
        x[0]=(s-length)*co;x[1]=(s-length)*si;x[2]=alpha;
    }
    x[2]=atan2(sin(x[2]),cos(x[2]));
    x[3]=F8_PHASE_RATE*length/(2*pi);x[4]=p;
    u[0]=0;u[1]=atan(F8_WHEELBASE*curvature);
}
void f8_dynamics(const double x[5],const double u[2],double y[5]) {
    double tmp[5]={x[0]+F8_DT*x[3]*cos(x[2]),x[1]+F8_DT*x[3]*sin(x[2]),
        x[2]+F8_DT*x[3]*tan(u[1])/F8_WHEELBASE,x[3]+F8_DT*u[0],x[4]+F8_DT*F8_PHASE_RATE};
    memcpy(y,tmp,sizeof(tmp));
}
static double clip(double x,double lo,double hi){return fmax(lo,fmin(hi,x));}
void f8_initial_guess(const double initial[5],double* w) {
    double x[5],r[5],u[2];memcpy(x,initial,5*sizeof(double));
    for(int k=0;k<=F8_N;++k) {
        memcpy(w+7*k,x,5*sizeof(double));if(k==F8_N)break;
        f8_reference(x[4],r,u);
        u[0]=clip((r[3]-x[3])/F8_DT,-F8_ACCELERATION_MAX,F8_ACCELERATION_MAX);
        u[1]=clip(u[1],-F8_STEERING_MAX,F8_STEERING_MAX);
        memcpy(w+7*k+5,u,2*sizeof(double));f8_dynamics(x,u,x);
    }
}
void f8_shift_guess(const double* previous,const double initial[5],double* w) {
    double x[5],u[2];memcpy(x,initial,5*sizeof(double));
    for(int k=0;k<F8_N;++k) {
        int j=k+1<F8_N?k+1:F8_N-1;
        u[0]=previous[7*j+5];u[1]=previous[7*j+6];
        u[0]=clip(u[0],fmax(-F8_ACCELERATION_MAX,(F8_SPEED_MIN-x[3])/F8_DT),
                         fmin(F8_ACCELERATION_MAX,(F8_SPEED_MAX-x[3])/F8_DT));
        memcpy(w+7*k,x,5*sizeof(double));memcpy(w+7*k+5,u,2*sizeof(double));f8_dynamics(x,u,x);
    }
    memcpy(w+7*F8_N,x,5*sizeof(double));
}
void f8_initial_theta(double* theta) {const double w[4]=F8_INITIAL_WEIGHTS;for(int i=0;i<4;++i)theta[i]=log(w[i]);}
void f8_teacher_theta(double* theta) {const double w[4]=F8_TEACHER_WEIGHTS;for(int i=0;i<4;++i)theta[i]=log(w[i]);}
double f8_imitation_loss(const double* w,const double* teacher,double* g) {
    double loss=0;memset(g,0,(F8_N*7+5)*sizeof(double));
    for(int k=0;k<=F8_N;++k) {
        const double scale=1.0/(F8_N+1),weight[4]={1,1,0.25,0.25};
        for(int j=0;j<4;++j) {
            int i=7*k+j;double e=w[i]-teacher[i];
            if(j==2){loss+=scale*weight[j]*(1-cos(e));g[i]=scale*weight[j]*sin(e);}
            else{loss+=0.5*scale*weight[j]*e*e;g[i]=scale*weight[j]*e;}
        }
        if(k<F8_N)for(int j=0;j<2;++j){int i=7*k+5+j;double e=w[i]-teacher[i];
            double s=(j==0?0.05:0.02)/F8_N;loss+=0.5*s*e*e;g[i]=s*e;}
    }
    return loss;
}
