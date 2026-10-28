#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
namespace cubic_q16_cache {
inline unsigned short x87_control(){unsigned short v;__asm__ __volatile__("fnstcw %0":"=m"(v));return v;}
inline unsigned short x87_status(){unsigned short v;__asm__ __volatile__("fnstsw %0":"=a"(v));return v;}
inline double original(double p){return std::log(p)-std::log1p(-p);}
struct Cache {
 double values[65536]{};
 unsigned char sse_delta[65536]{},x87_delta[65536]{},valid[8192]{};
 unsigned control=~0u;unsigned short cw=0;
 uint64_t hits=0,misses=0,flag_fallbacks=0,unsafe_fallbacks=0,invalidations=0;
 double get(unsigned q,double p){
  const unsigned before=_mm_getcsr();const unsigned short xc=x87_control();
  // No shortcut in trapping environments. The original expression remains
  // authority, including floating exception delivery.
  if((before&0x1f80u)!=0x1f80u||(xc&0x3fu)!=0x3fu){++unsafe_fallbacks;return original(p);}
  const unsigned mode=before&~0x3fu;
  if(mode!=control||xc!=cw){std::memset(valid,0,sizeof(valid));control=mode;cw=xc;++invalidations;}
  const unsigned short xs=x87_status();const unsigned byte=q>>3,bit=1u<<(q&7);
  if(valid[byte]&bit){
   if((before&sse_delta[q])==sse_delta[q]&&(xs&x87_delta[q])==x87_delta[q]){++hits;return values[q];}
   ++flag_fallbacks;return original(p);
  }
  const double value=original(p);
  const unsigned after=_mm_getcsr();const unsigned short xa=x87_status();
  values[q]=value;sse_delta[q]=static_cast<unsigned char>((after&~before)&0x3fu);
  x87_delta[q]=static_cast<unsigned char>((xa&~xs)&0x3fu);valid[byte]|=bit;++misses;return value;
 }
};
inline Cache cache;
inline double get(unsigned q,double p){return cache.get(q,p);}
struct Reporter {~Reporter(){std::fprintf(stderr,"\nCUBIC_Q16_CACHE {\"hits\":%llu,\"misses\":%llu,\"flag_fallbacks\":%llu,\"unsafe_fallbacks\":%llu,\"invalidations\":%llu}\n",(unsigned long long)cache.hits,(unsigned long long)cache.misses,(unsigned long long)cache.flag_fallbacks,(unsigned long long)cache.unsafe_fallbacks,(unsigned long long)cache.invalidations);}};
inline Reporter reporter;
}
