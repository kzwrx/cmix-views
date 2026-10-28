#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "page_registry.h"
namespace word_order {
inline uint64_t models=0,steps=0,backward_steps=0,out_updates=0,gate_cell_updates=0,features=0,nonzero_features=0,cancels=0;
inline uint64_t backprop_steps=0;inline unsigned maximum_credit_steps=0;inline double propagated_absolute_gradient=0.;
inline double maximum_hidden=0.,maximum_feature=0.,feature=0.;inline bool feature_live=false;
inline void ClearFeature(){feature_live=false;feature=0.;}
struct Model {
 static constexpr unsigned D=32,I=96,M=I+D+1,NODES=131072;
 float weights[3][D][M]{};
 float h[D]{},previous[D]{},x[I]{},z[D]{},r[D]{},candidate[D]{};
 double gradient[D]{};
 static constexpr unsigned CREDIT=16;
 struct Trace{float previous[D],x[I],z[D],r[D],candidate[D],recurrent[3][D][D];};
 Trace tape[CREDIT]{};unsigned tape_head=0,tape_size=0;
 float*out=nullptr;unsigned node=0;double probability=.5;bool pending=false,cached=false;
 static double squash(double v){if(v>=0)return 1./(1.+std::exp(-v));double e=std::exp(v);return e/(1.+e);}
 Model(){
  out=static_cast<float*>(std::calloc(size_t(NODES)*(D+1),sizeof(float)));if(!out)std::abort();
  PAGE_REGISTRY_ADD(out,size_t(NODES)*(D+1)*sizeof(float));
  uint64_t seed=0x6a09e667f3bcc909ULL;
  for(unsigned g=0;g<3;++g)for(unsigned j=0;j<D;++j)for(unsigned k=0;k<I+D;++k){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;weights[g][j][k]=float(double(seed>>11)/9007199254740992.-.5);}
  ++models;
 }
 ~Model(){std::free(out);}
 void Abstain(){pending=false;}
 void Cancel(){pending=false;for(double&v:gradient)v=0.;++cancels;}
 void Predict(unsigned n,float old_logit){
  if(n>=NODES||!std::isfinite(old_logit))std::abort();node=n;
  const float*row=out+size_t(node)*(D+1);double value=row[D];
  for(unsigned j=0;j<D;++j)value+=double(row[j])*h[j];
  if(!std::isfinite(value))std::abort();value=std::max(-20.,std::min(20.,value));probability=squash(value);pending=true;
  if(cached){feature=std::max(-4.,std::min(4.,value-double(old_logit)));feature_live=true;++features;nonzero_features+=feature!=0.;maximum_feature=std::max(maximum_feature,std::abs(feature));}
 }
 void Observe(int bit){
  if(!pending)return;const double g=probability-double(bit);float*row=out+size_t(node)*(D+1);
  // Read the old row into the state gradient before modifying that row.
  for(unsigned j=0;j<D;++j){gradient[j]+=g*double(row[j]);row[j]-=float(.06*g*double(h[j]));}
  row[D]-=float(.06*g);pending=false;++out_updates;
 }
 void SaveTrace(){
  Trace&t=tape[tape_head];
  std::memcpy(t.previous,previous,sizeof(previous));std::memcpy(t.x,x,sizeof(x));
  std::memcpy(t.z,z,sizeof(z));std::memcpy(t.r,r,sizeof(r));std::memcpy(t.candidate,candidate,sizeof(candidate));
  for(unsigned g=0;g<3;++g)for(unsigned j=0;j<D;++j)for(unsigned k=0;k<D;++k)t.recurrent[g][j][k]=weights[g][j][I+k];
  tape_head=(tape_head+1)%CREDIT;tape_size=std::min(CREDIT,tape_size+1);
 }
 void Backward(){
  if(!cached)return;if(!tape_size)std::abort();
  double dh[D],accum[3][D][M]{};
  for(unsigned j=0;j<D;++j)dh[j]=gradient[j];
  for(unsigned depth=0;depth<tape_size;++depth){
   const Trace&t=tape[(tape_head+CREDIT-1-depth)%CREDIT];
   double gn[D],gz[D],gr[D],through_reset[D]{},dh_previous[D];
   for(unsigned j=0;j<D;++j){
    if(!std::isfinite(dh[j]))std::abort();dh[j]=std::max(-.5,std::min(.5,dh[j]));
    gn[j]=dh[j]*(1.-t.z[j])*(1.-double(t.candidate[j])*t.candidate[j]);
    gz[j]=dh[j]*(double(t.previous[j])-t.candidate[j])*t.z[j]*(1.-t.z[j]);
   }
   for(unsigned j=0;j<D;++j)for(unsigned k=0;k<D;++k)through_reset[k]+=double(t.recurrent[2][j][k])*gn[j];
   for(unsigned j=0;j<D;++j)gr[j]=through_reset[j]*t.previous[j]*t.r[j]*(1.-t.r[j]);
   for(unsigned k=0;k<D;++k){
    dh_previous[k]=dh[k]*t.z[k]+through_reset[k]*t.r[k];
    for(unsigned j=0;j<D;++j)dh_previous[k]+=double(t.recurrent[0][j][k])*gz[j]+double(t.recurrent[1][j][k])*gr[j];
    if(!std::isfinite(dh_previous[k]))std::abort();
   }
   for(unsigned g=0;g<3;++g)for(unsigned j=0;j<D;++j){
    const double delta=(g==0?gz[j]:(g==1?gr[j]:gn[j]));
    for(unsigned k=0;k<I;++k)accum[g][j][k]+=delta*t.x[k];
    for(unsigned k=0;k<D;++k)accum[g][j][I+k]+=delta*t.previous[k]*(g==2?t.r[k]:1.f);
    accum[g][j][I+D]+=delta;
   }
   for(unsigned j=0;j<D;++j){dh[j]=dh_previous[j];if(depth+1<tape_size)propagated_absolute_gradient+=std::abs(dh[j]);}
   ++backprop_steps;maximum_credit_steps=std::max(maximum_credit_steps,depth+1);
  }
  // Delayed gradients for the recorded forward graph are tied onto CURRENT weights.
  // Past gates use their own recorded recurrent matrices; no current-weight replay.
  for(unsigned g=0;g<3;++g)for(unsigned j=0;j<D;++j)for(unsigned k=0;k<M;++k){
   if(!std::isfinite(accum[g][j][k]))std::abort();weights[g][j][k]-=float(.03*accum[g][j][k]);++gate_cell_updates;
  }
  ++backward_steps;
 }
 void Finish(const float*embedding){
  Backward();for(double&v:gradient)v=0.;
  double norm=1e-12;for(unsigned k=0;k<I;++k){if(!std::isfinite(embedding[k]))std::abort();norm+=double(embedding[k])*embedding[k];}
  norm=std::sqrt(norm);for(unsigned k=0;k<I;++k)x[k]=float(double(embedding[k])/norm);
  std::memcpy(previous,h,sizeof(h));
  for(unsigned g=0;g<2;++g)for(unsigned j=0;j<D;++j){double value=weights[g][j][I+D];
   for(unsigned k=0;k<I;++k)value+=double(weights[g][j][k])*x[k];
   for(unsigned k=0;k<D;++k)value+=double(weights[g][j][I+k])*previous[k];
   (g==0?z[j]:r[j])=float(squash(value));
  }
  for(unsigned j=0;j<D;++j){double value=weights[2][j][I+D];
   for(unsigned k=0;k<I;++k)value+=double(weights[2][j][k])*x[k];
   for(unsigned k=0;k<D;++k)value+=double(weights[2][j][I+k])*r[k]*previous[k];
   candidate[j]=float(std::tanh(value));h[j]=z[j]*previous[j]+(1.f-z[j])*candidate[j];
   if(!std::isfinite(h[j])||std::abs(h[j])>1.000001f)std::abort();maximum_hidden=std::max(maximum_hidden,std::abs(double(h[j])));
  }
  SaveTrace();cached=true;++steps;
 }
};
}
