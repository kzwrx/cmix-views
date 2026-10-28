#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
// Four independent views of the SAME existing four-block features.
// No old weight, normalization, read or gradient is modified.
namespace winner_views {
struct Expert {
 static constexpr int B=4,H=1024,O=208,N=B*H;
 std::vector<float>w;
 float base[O]{},alt[B][O]{},u[N]{};uint16_t ix[N]{};int nnz=0;
 uint64_t reads=0,updates=0;double base_bits=0,view_bits[B]{};bool pending=false;
 void read(const float*p,const uint16_t*index,const float*features,int n){
  if(n<0||n>N)std::exit(240);if(w.empty())w.resize(size_t(N)*O,0.f);
  nnz=n;std::memcpy(base,p,sizeof(base));float d[B][O]{};
  for(int k=0;k<n;++k){if(index[k]>=N||!std::isfinite(features[k]))std::exit(241);ix[k]=index[k];u[k]=2.f*features[k];const int b=ix[k]/H;const float*col=w.data()+size_t(ix[k])*O;for(int o=0;o<O;++o)d[b][o]+=col[o]*u[k];}
  for(int b=0;b<B;++b){
   if(!updates)std::memcpy(alt[b],base,sizeof(base));
   else {double l[O],m=-1e300,z=0;for(int o=0;o<O;++o){if(!(base[o]>0)||!std::isfinite(d[b][o]))std::exit(242);l[o]=std::log(double(base[o]))+double(d[b][o]);m=std::max(m,l[o]);}for(double&v:l){v=std::exp(v-m);z+=v;}for(int o=0;o<O;++o)alt[b][o]=float(l[o]/z);}
  }pending=true;++reads;
 }
 void observe(uint8_t target){
  if(!pending||target>=O)std::exit(243);base_bits-=std::log2(std::max(1e-300,double(base[target])));
  float error[B][O];for(int b=0;b<B;++b){view_bits[b]-=std::log2(std::max(1e-300,double(alt[b][target])));for(int o=0;o<O;++o)error[b][o]=alt[b][o]-(o==target?1.f:0.f);}
  for(int k=0;k<nnz;++k){float*col=w.data()+size_t(ix[k])*O;const int b=ix[k]/H;for(int o=0;o<O;++o)col[o]-=u[k]*error[b][o];}pending=false;++updates;
 }
 ~Expert(){std::fprintf(stderr,"\nVIEW_EXPERTS {\"reads\":%llu,\"updates\":%llu,\"runtime_weight_bytes\":%zu,\"base_bits\":%.12f,\"view_bits\":[%.12f,%.12f,%.12f,%.12f]}\n",(unsigned long long)reads,(unsigned long long)updates,w.size()*sizeof(float),base_bits,view_bits[0],view_bits[1],view_bits[2],view_bits[3]);}
};
}
