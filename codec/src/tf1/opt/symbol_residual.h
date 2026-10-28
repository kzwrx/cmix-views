#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

// A prospective challenger, not a modification of any prior frozen attempt.
// Original four-layer FW probabilities and their backward/update scratch stay
// untouched. This independent full-feature expert is mixed only at its output.
namespace symbol_residual {
struct Expert {
 static constexpr int B=4,H=1024,O=208,N=B*H;
 bool enabled=nullptr==nullptr;
 std::vector<float> w;
 uint16_t index[N]{};
 float u[N]{},base[O]{},alt[O]{};
 int nnz=0;
 uint64_t reads=0,updates=0,writes=0;
 double log_odds=0,base_bits=0,alt_bits=0,mix_bits=0,weight_sum=0;
 double pending_weight=.5;
 bool pending=false;
 static double weight(double z){if(z>=0)return 1./(1.+std::exp(-z));double a=std::exp(z);return a/(1.+a);}
 void read(const float*p,const int*count,const uint16_t ix[B][H],const float h[B][H],float*out){
  if(!enabled)return; // OFF is an exact pointer/no-write control.
  if(w.empty())w.resize(size_t(N)*O,0.f);
  nnz=0;
  for(int b=0;b<B;++b){
   double norm=1.;for(int k=0;k<count[b];++k)norm+=double(h[b][k])*h[b][k];
   const float scale=float(.5/std::sqrt(norm)); // equal block weighting, ||u|| <= 1
   for(int k=0;k<count[b];++k){index[nnz]=uint16_t(b*H+ix[b][k]);u[nnz]=h[b][k]*scale;++nnz;}
  }
  std::memcpy(base,p,sizeof(base));float d[O]{};
  for(int k=0;k<nnz;++k){const float*col=w.data()+size_t(index[k])*O;const float v=u[k];for(int o=0;o<O;++o)d[o]+=col[o]*v;}
  if(updates==0)std::memcpy(alt,base,sizeof(alt));
  else{
   double logits[O],maxlog=-1e300,z=0;
   for(int o=0;o<O;++o){if(!(base[o]>0)||!std::isfinite(d[o]))std::exit(181);logits[o]=std::log(double(base[o]))+double(d[o]);maxlog=std::max(maxlog,logits[o]);}
   for(int o=0;o<O;++o){logits[o]=std::exp(logits[o]-maxlog);z+=logits[o];}
   for(int o=0;o<O;++o)alt[o]=float(logits[o]/z);
  }
  pending_weight=weight(log_odds);
  // At zero state, preserve native baseline probabilities exactly.
  if(updates)for(int o=0;o<O;++o)out[o]=float((1.-pending_weight)*double(base[o])+pending_weight*double(alt[o]));
  pending=true;++reads;weight_sum+=pending_weight;
 }
 void observe(uint8_t target){
  if(!enabled)return;
  if(!pending||target>=O)std::exit(182);
  const double p0=std::max(1e-300,double(base[target])),p1=std::max(1e-300,double(alt[target]));
  const double pm=(1.-pending_weight)*p0+pending_weight*p1;
  base_bits-=std::log2(p0);alt_bits-=std::log2(p1);mix_bits-=std::log2(pm);
  log_odds+=std::log(p1)-std::log(p0);
  if(!std::isfinite(log_odds))std::exit(183);
  // Exact CE gradient of log(base) + W*u, with the offset held fixed.
  // Unit learning step; ||u||<=1 bounds the current-row logit displacement.
  float error[O];for(int o=0;o<O;++o)error[o]=alt[o]-(o==int(target)?1.f:0.f);
  for(int k=0;k<nnz;++k){float*col=w.data()+size_t(index[k])*O;const float v=u[k];for(int o=0;o<O;++o)col[o]-=v*error[o];}
  ++updates;writes+=nnz;pending=false;
 }
 ~Expert(){if(enabled)std::fprintf(stderr,"\nSYMBOL_RESIDUAL {\"reads\":%llu,\"updates\":%llu,\"column_writes\":%llu,\"state_weight_bytes\":%zu,\"base_symbol_bits\":%.12f,\"alt_symbol_bits\":%.12f,\"mix_symbol_bits\":%.12f,\"mean_alt_weight\":%.12f,\"final_alt_weight\":%.12f}\n",(unsigned long long)reads,(unsigned long long)updates,(unsigned long long)writes,w.size()*sizeof(float),base_bits,alt_bits,mix_bits,reads?weight_sum/reads:0.,weight(log_odds));}
};
}
