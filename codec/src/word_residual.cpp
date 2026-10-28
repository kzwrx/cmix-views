#ifndef WR_AFD_CONTROL_OFF
#define WR_AFD_CONTROL_OFF 0
#endif
#include "word_residual.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <unordered_map>
// ONE extra pre-label feature in the WR-1 expert: signed parent confidence
// x=2*p-1. Solve a fixed unit-prior two-coordinate Fisher quadratic, not a
// logistic optimum or a derivative/loss guarantee for the hard range coder.
namespace word_residual {
struct Key {uint64_t c;uint32_t n;bool operator==(const Key&b)const{return c==b.c&&n==b.n;}};
struct Hash {size_t operator()(const Key&k)const{uint64_t v=k.c^(uint64_t(k.n)*0x9e3779b97f4a7c15ull);v^=v>>30;v*=0xbf58476d1ce4e5b9ull;return v^(v>>27);}};
using I128=__int128_t;using U128=__uint128_t;
struct Cell {int64_t r=0;uint64_t v=0;I128 r1=0,v1=0;U128 v2=0;};
static std::unordered_map<Key,Cell,Hash> cells;
static std::unordered_map<uint64_t,double> contexts;
static bool live=false,pending=false;static Key key{};
static uint64_t context=0,events=0,changed=0;static uint32_t node=0;
static unsigned original=0;static double base=0,expert=0;
static int64_t signed_q=0;
void clear_feature(){live=false;}
void feature(bool l,uint64_t c,uint32_t n){live=l;context=c;node=n;}
unsigned predict(unsigned q,bool eligible){
 pending=false;if(WR_AFD_CONTROL_OFF||!eligible||!live)return q;
 assert(q>=1&&q<=65535);key={context,node};auto it=cells.find(key);
 if(it==cells.end()&&!(cells.size()<1048576&&(contexts.count(context)||contexts.size()<262144)))return q;
 original=q;signed_q=2*int64_t(q)-65536;base=double(q)/65536.;expert=base;
 if(it!=cells.end()&&(it->second.r||it->second.r1)){
  const auto&c=it->second;
  // Exact integer units until this fixed, deterministic double solve.
  double A=1.+double(c.v)/4294967296.,B=double(c.v1)/281474976710656.,C=1.+double(c.v2)/18446744073709551616.;
  double b0=double(c.r)/65536.,b1=double(c.r1)/4294967296.;
  double ratio=B/A,D=C-B*ratio;assert(A>0&&D>0&&std::isfinite(D));
  double beta1=(b1-ratio*b0)/D,beta0=(b0-B*beta1)/A;
  double shift=beta0+beta1*(double(signed_q)/65536.);assert(std::isfinite(shift));
  double z=std::log(base)-std::log1p(-base)+shift;double pe;if(z>=0)pe=1./(1.+std::exp(-z));else{double e=std::exp(z);pe=e/(1.+e);}unsigned eq=unsigned(std::max(1.,std::min(65534.,std::floor(pe*65536.+.5))));expert=double(eq)/65536.;
 }
 auto st=contexts.find(context);double lr=st==contexts.end()?0:st->second;double w;if(lr>=0)w=1./(1.+std::exp(-lr));else{double e=std::exp(lr);w=e/(1.+e);}double mixed=base+w*(expert-base);
 assert(mixed>0&&mixed<1&&std::isfinite(mixed));unsigned out=q;if(mixed!=base)out=unsigned(std::max(1.,std::min(65534.,std::floor(mixed*65536.+.5))));pending=true;changed+=out!=q;return out;
}
void observe(int bit){
 if(!pending)return;assert(bit==0||bit==1);auto&c=cells[key];auto&lr=contexts[key.c];double pb=bit?base:1-base,pe=bit?expert:1-expert;lr+=std::log(pe)-std::log(pb);
 int64_t r=65536ll*bit-int64_t(original);assert((r>=0&&c.r<=std::numeric_limits<int64_t>::max()-r)||(r<0&&c.r>=std::numeric_limits<int64_t>::min()-r));c.r+=r;uint64_t v=uint64_t(original)*(65536u-original);assert(c.v<=std::numeric_limits<uint64_t>::max()-v);c.v+=v;
 c.r1+=I128(r)*signed_q;c.v1+=I128(v)*signed_q;c.v2+=U128(v)*U128(signed_q*signed_q);
 ++events;pending=false;
}
void report(){std::fprintf(stderr,"\nWORD_RESIDUAL {\"events\":%llu,\"changed_q\":%llu,\"cells\":%zu,\"contexts\":%zu}\n",(unsigned long long)events,(unsigned long long)changed,cells.size(),contexts.size());}
}
