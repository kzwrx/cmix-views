#include "q16_logit_cache.h"
#pragma once
#include "cubic_view.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
namespace cubic_overlay {
inline double cue=0;inline bool cue_live=false;
struct Expert {
 double score=0,curvature=0,alpha=0,phi=0,probability=.5,log_odds=0,continuous=.5;
 unsigned base=32768,peer=32768,output=32768;bool pending=false;
 uint64_t reads=0,updates=0,nonzero=0,disagreements=0,changed=0;
 double maximum_cue=0,sum_abs_alpha=0;
 static double weight(double z){if(z>=0)return 1./(1.+std::exp(-z));const double e=std::exp(z);return e/(1.+e);}
 unsigned predict(unsigned q,double u,bool active){
  pending=false;base=peer=output=q;if(!active)return q;
  if(q<1||q>65534||!std::isfinite(u)||std::abs(u)>4.)std::exit(266);
  phi=u;alpha=std::max(-1.,std::min(1.,-score/(1.+curvature)));probability=double(q)/65536.;
  if(alpha!=0&&phi!=0){const double z=cubic_q16_cache::get(q,probability)+alpha*phi;if(z>=0)probability=1./(1.+std::exp(-z));else{const double e=std::exp(z);probability=e/(1.+e);}peer=unsigned(std::max(1.,std::min(65534.,std::floor(65536.*probability+.5))));}
  continuous=double(q)/65536.;
  if(peer!=q){const double mixed=double(q)+weight(log_odds)*(double(peer)-double(q));continuous=mixed/65536.;output=unsigned(std::max(1.,std::min(65534.,std::floor(mixed+.5))));}
  pending=true;++reads;nonzero+=phi!=0;disagreements+=peer!=q;changed+=output!=q;maximum_cue=std::max(maximum_cue,std::abs(phi));sum_abs_alpha+=std::abs(alpha);return output;
 }
 void observe(int bit){
  if(bit!=0&&bit!=1)std::exit(267);if(!pending)return;pending=false;
  const double v=probability*(1.-probability),g=probability-double(bit),h=v*phi*phi;
  score+=g*phi-h*alpha;curvature+=h;
  if(peer!=base){const unsigned a=bit?base:65536-base,b=bit?peer:65536-peer;log_odds+=std::log(double(b)/double(a));}
  if(!std::isfinite(score)||!std::isfinite(curvature)||curvature<0||!std::isfinite(log_odds))std::exit(268);++updates;
 }
 void report()const{std::fprintf(stderr,"\nCUBIC_OVERLAY {\"reads\":%llu,\"updates\":%llu,\"nonzero_cue\":%llu,\"disagreements\":%llu,\"changed_q\":%llu,\"maximum_cue\":%.12f,\"sum_abs_alpha\":%.12f,\"score\":%.12f,\"curvature\":%.12f,\"log_odds\":%.12f}\n",(unsigned long long)reads,(unsigned long long)updates,(unsigned long long)nonzero,(unsigned long long)disagreements,(unsigned long long)changed,maximum_cue,sum_abs_alpha,score,curvature,log_odds);}
};
inline Expert expert;struct Reporter{~Reporter(){expert.report();}};inline Reporter reporter;
}
