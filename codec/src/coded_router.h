#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
namespace coded_router {
inline float bit_p[4]={.5f,.5f,.5f,.5f};
struct Record {uint16_t q0,qwinner,qcoded;uint8_t phase,bit,active,reserved[7];double phi[7];};
static_assert(sizeof(Record)==72,"fixed6feature causal replay sidecar");
struct Router {
 static constexpr int K=7;
 double score[8][K]{},H[8][K][K]{},phi[K]{},theta[K]{};
 double probability=.5,base_bits=0,winner_bits=0,final_bits=0;
 unsigned base=32768,winner=32768,output=32768;int phase=0;bool pending=false;static constexpr bool enabled=false;
 uint64_t events=0,changed=0,records=0;static constexpr FILE*trace=nullptr;
 static double logit(double p){p=std::max(1./65536,std::min(65535./65536,p));return std::log(p)-std::log1p(-p);}
 static double clip(double v){return std::max(-4.,std::min(4.,v));}
 void coefficients(int bp,double*out)const{
  double L[K][K]{},y[K]{};
  for(int i=0;i<K;++i)for(int j=0;j<=i;++j){double v=H[bp][i][j]+(i==j?1.:0.);for(int k=0;k<j;++k)v-=L[i][k]*L[j][k];if(i==j){if(!(v>0)||!std::isfinite(v))std::exit(244);L[i][j]=std::sqrt(v);}else L[i][j]=v/L[j][j];}
  for(int i=0;i<K;++i){double v=-score[bp][i];for(int j=0;j<i;++j)v-=L[i][j]*y[j];y[i]=v/L[i][i];}
  for(int i=K-1;i>=0;--i){double v=y[i];for(int j=i+1;j<K;++j)v-=L[j][i]*out[j];out[i]=v/L[i][i];}
  double norm=0;for(int i=0;i<K;++i)norm+=std::abs(out[i]);if(norm>1)for(int i=0;i<K;++i)out[i]/=norm;
 }
 unsigned predict(unsigned q,unsigned old,int bp,bool eligible,bool live,double fw,double pooled,const float*views,float coded){
  base=q;winner=output=old;phase=bp;pending=false;std::memset(phi,0,sizeof(phi));
  if(!enabled||!eligible||!live)return output;
  if(bp<0||bp>=8||q<1||q>65534)std::exit(245);
  const double z0=logit(double(q)/65536),zf=logit(fw);
  phi[0]=clip(logit(pooled)-zf);phi[1]=clip(zf-z0);for(int b=0;b<4;++b)phi[b+2]=clip(logit(views[b])-zf);phi[6]=clip(logit(coded)-logit(pooled));
  coefficients(bp,theta);double shift=0;for(int i=0;i<K;++i)shift+=theta[i]*phi[i];probability=double(q)/65536;output=q;
  if(shift!=0){const double s=z0+shift;if(s>=0)probability=1/(1+std::exp(-s));else{double e=std::exp(s);probability=e/(1+e);}output=unsigned(std::max(1.,std::min(65534.,std::floor(65536*probability+.5))));}
  pending=true;++events;changed+=output!=winner;return output;
 }
 void observe(int bit){
  if(bit!=0&&bit!=1)std::exit(246);const bool active=pending;
  if(pending){
   const double v=probability*(1-probability),error=probability-bit;
   for(int i=0;i<K;++i){double ht=0;for(int j=0;j<K;++j)ht+=v*phi[i]*phi[j]*theta[j];score[phase][i]+=error*phi[i]-ht;for(int j=0;j<K;++j)H[phase][i][j]+=v*phi[i]*phi[j];if(!std::isfinite(score[phase][i]))std::exit(247);}
   auto cost=[&](unsigned q){return -std::log2(double(bit?q:65536-q)/65536);};base_bits+=cost(base);winner_bits+=cost(winner);final_bits+=cost(output);pending=false;
  }
  if(trace){Record r{};r.q0=uint16_t(base);r.qwinner=uint16_t(winner);r.qcoded=uint16_t(output);r.phase=uint8_t(phase);r.bit=uint8_t(bit);r.active=active;std::memcpy(r.phi,phi,sizeof(phi));if(fwrite(&r,sizeof(r),1,trace)!=1)std::exit(248);}++records;
 }
 void init(const char*path){if(path&&std::strcmp(path,"-"))std::exit(249);}
 void close(){if(trace){Record r{};r.bit=2;if(fwrite(&r,sizeof(r),1,trace)!=1||fclose(trace))std::exit(250);}std::fprintf(stderr,"\nCODED_CLASS_CALIBRATION {\"events\":%llu,\"changed_q\":%llu,\"records\":%llu,\"base_bits\":%.12f,\"winner_bits\":%.12f,\"final_bits\":%.12f}\n",(unsigned long long)events,(unsigned long long)changed,(unsigned long long)records,base_bits,winner_bits,final_bits);}
};
}
