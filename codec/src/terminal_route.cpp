#include "coded_class.h"
#include "coded_router.h"
#include "view_router.h"
#include "terminal_route.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
namespace terminal_route {
static double bounded(double p){return std::max(1./65536,std::min(65535./65536,p));}
unsigned Router::predict(unsigned base,int bp,bool eligible,bool live,double oldp,double newp){
 pending=false;q0=q1=qmix=base;
 if(!enabled||!eligible||!live)return base;
 if(bp<0||bp>=8||base<1||base>65534||!std::isfinite(oldp)||!std::isfinite(newp))std::exit(201);
 phase=bp;pending=true;delta=0;
 double a=bounded(oldp),b=bounded(newp);
 if(a!=b)delta=std::max(-4.,std::min(4.,std::log(b)-std::log1p(-b)-std::log(a)+std::log1p(-a)));
 alpha=std::max(-1.,std::min(1.,-score[bp]/(1.+curvature[bp])));
 const double p0=double(base)/65536;
 probability=p0;
 if(delta!=0&&alpha!=0){double r=std::exp(alpha*delta);probability=p0*r/(1.-p0+p0*r);qmix=unsigned(std::max(1.,std::min(65534.,std::floor(65536.*probability+.5))));}
 q1=qmix;nonzero+=delta!=0;changed+=qmix!=base;++n[bp];coefficient_sum[bp]+=alpha;
 return qmix;
}
void Router::observe(int bit){
 if(!pending)return;pending=false;if(bit!=0&&bit!=1)std::exit(202);
 auto prob=[&](unsigned q){return double(bit?q:65536-q)/65536;};
 baseline_bits-=std::log2(prob(q0));coded_bits-=std::log2(prob(qmix));
 // Local quadratic FTRL-style calibration of final-bit logistic loss.
 // Derivatives are for continuous pre-round probability, NOT an exact
 // derivative of the discrete16-bit range coder. All scoring uses hard qmix.
 const double g=(probability-double(bit))*delta;
 const double h=probability*(1.-probability)*delta*delta;
 score[phase]+=g-h*alpha;curvature[phase]+=h;
 if(!std::isfinite(score[phase])||!std::isfinite(curvature[phase])||curvature[phase]<0)std::exit(203);
}
void Coupled::coefficients(int bp,double*out)const{
 const double a=1.+curvature[bp][0],b=curvature[bp][1],c=1.+curvature[bp][2],det=a*c-b*b;
 if(!(det>0)||!std::isfinite(det))std::exit(211);
 out[0]=-(c*score[bp][0]-b*score[bp][1])/det;
 out[1]=-(a*score[bp][1]-b*score[bp][0])/det;
 const double scale=std::abs(out[0])+std::abs(out[1]);
 if(scale>1){out[0]/=scale;out[1]/=scale;}
 if(!std::isfinite(out[0])||!std::isfinite(out[1]))std::exit(212);
}
unsigned Coupled::predict(unsigned base,int bp,bool eligible,bool live,double oldp,double newp){
 legacy.enabled=enabled;
 if(scalar_only)return legacy.predict(base,bp,eligible,live,oldp,newp);
 pending=false;q0=qmix=base;if(!enabled||!eligible||!live)return base;
 if(bp<0||bp>=8||base<1||base>65534||!std::isfinite(oldp)||!std::isfinite(newp))std::exit(213);
 phase=bp;pending=true;const double a=bounded(oldp),b=bounded(newp),p0=double(base)/65536;
 const double z0=std::log(p0)-std::log1p(-p0),za=std::log(a)-std::log1p(-a);
 phi[0]=a==b?0:std::max(-4.,std::min(4.,std::log(b)-std::log1p(-b)-std::log(a)+std::log1p(-a)));
 phi[1]=std::max(-4.,std::min(4.,za-z0));coefficients(bp,theta);
 const double shift=theta[0]*phi[0]+theta[1]*phi[1];probability=p0;
 if(shift!=0){const double ratio=std::exp(shift);probability=p0*ratio/(1.-p0+p0*ratio);qmix=unsigned(std::max(1.,std::min(65534.,std::floor(65536.*probability+.5))));}
 nonzero+=phi[0]!=0||phi[1]!=0;changed+=qmix!=base;++n[bp];for(int j=0;j<2;++j)coefficient_sum[bp][j]+=theta[j];return qmix;
}
void Coupled::observe(int bit){
 if(scalar_only){legacy.observe(bit);return;}
 if(!pending)return;pending=false;if(bit!=0&&bit!=1)std::exit(214);
 auto prob=[&](unsigned q){return double(bit?q:65536-q)/65536;};baseline_bits-=std::log2(prob(q0));coded_bits-=std::log2(prob(qmix));
 const double v=probability*(1.-probability),h00=v*phi[0]*phi[0],h01=v*phi[0]*phi[1],h11=v*phi[1]*phi[1],error=probability-double(bit);
 score[phase][0]+=error*phi[0]-h00*theta[0]-h01*theta[1];score[phase][1]+=error*phi[1]-h01*theta[0]-h11*theta[1];
 curvature[phase][0]+=h00;curvature[phase][1]+=h01;curvature[phase][2]+=h11;
 for(double x:score[phase])if(!std::isfinite(x))std::exit(215);
 for(double x:curvature[phase])if(!std::isfinite(x))std::exit(216);
}
#ifndef EXGF_ELIDE_LEGACY_COUPLED
#define EXGF_ELIDE_LEGACY_COUPLED 0
#endif
#if EXGF_ELIDE_LEGACY_COUPLED
static winner_views::Router views;static coded_router::Router coded;
static float old_probability=.5,new_probability=.5;static bool row_live=false;
void init_trace(const char*p){views.init(p);coded.init(p);}
void feature(float a,float b,bool live){old_probability=a;new_probability=b;row_live=live;}
unsigned predict(unsigned base,int bp,bool eligible){
 views.enabled=true;
 const unsigned parent=views.predict(base,base,bp,eligible,row_live,old_probability,new_probability,winner_views::bit_p);
 coded_class::expert.predict(parent,coded.enabled&&eligible&&row_live&&coded_class::row_available);
 return coded.predict(base,parent,bp,eligible,row_live,old_probability,new_probability,winner_views::bit_p,coded_class::bit_probability);
}
void observe(int bit){views.observe(bit);coded_class::expert.observe(bit);coded.observe(bit);}
void close(){views.close();coded.close();std::fprintf(stderr,"\nLEGACY_COUPLED_RETIRED {\"retired\":true,\"coded_dependency\":false}\n");}
#else
static Coupled router;static winner_views::Router views;static coded_router::Router coded;
void init_trace(const char*p){views.init(p);coded.init(p);}static float old_probability=.5,new_probability=.5;static bool row_live=false;
void feature(float a,float b,bool live){old_probability=a;new_probability=b;row_live=live;}
unsigned predict(unsigned base,int bp,bool eligible){router.enabled=nullptr==nullptr;router.scalar_only=nullptr!=nullptr;const unsigned old=router.predict(base,bp,eligible,row_live,old_probability,new_probability);views.enabled=router.enabled&&nullptr==nullptr;const unsigned parent=views.predict(base,old,bp,eligible,row_live,old_probability,new_probability,winner_views::bit_p);coded_class::expert.predict(parent,coded.enabled&&eligible&&row_live&&coded_class::row_available);return coded.predict(base,parent,bp,eligible,row_live,old_probability,new_probability,winner_views::bit_p,coded_class::bit_probability);}
void observe(int bit){router.observe(bit);views.observe(bit);coded_class::expert.observe(bit);coded.observe(bit);}
void close(){
 views.close();coded.close();
 double alpha[8]{},beta[8]{};uint64_t*count=router.n;double base=router.baseline_bits,coded=router.coded_bits;uint64_t proposals=router.nonzero,changes=router.changed;
 for(int j=0;j<8;++j){double t[2];router.coefficients(j,t);alpha[j]=t[0];beta[j]=t[1];}
 if(router.scalar_only){count=router.legacy.n;base=router.legacy.baseline_bits;coded=router.legacy.coded_bits;proposals=router.legacy.nonzero;changes=router.legacy.changed;for(int j=0;j<8;++j){alpha[j]=std::max(-1.,std::min(1.,-router.legacy.score[j]/(1.+router.legacy.curvature[j])));beta[j]=0;}}
 std::fprintf(stderr,"\nCOUPLED_CALIBRATION {\"nonzero_proposals\":%llu,\"changed_q\":%llu,\"baseline_bits\":%.12f,\"coded_bits\":%.12f,\"phase_events\":[",(unsigned long long)proposals,(unsigned long long)changes,base,coded);
 for(int j=0;j<8;++j)std::fprintf(stderr,"%s%llu",j?",":"",(unsigned long long)count[j]);
 std::fprintf(stderr,"],\"final_phase_coefficients\":[");for(int j=0;j<8;++j)std::fprintf(stderr,"%s%.12f",j?",":"",alpha[j]);
 std::fprintf(stderr,"],\"final_reference_coefficients\":[");for(int j=0;j<8;++j)std::fprintf(stderr,"%s%.12f",j?",":"",beta[j]);std::fprintf(stderr,"]}\n");
}
#endif
}
