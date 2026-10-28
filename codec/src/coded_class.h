#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include "metric_rowstream.h"
namespace coded_class {
inline float bit_probability=.5f;inline bool row_available=false;
// Production's fixed-OFF contract still transports the original live alt row.
struct Expert{
 static constexpr int N=4096,D=4096,O=208;
 static constexpr bool enabled=false,pending=false;
 float alt[O]{};int context=1;bool bit_pending=false;
 void configure(int i,int b,bool live){if(i<0||i>=O||b<0||b>255)std::exit(320);}
 void cache(const float*p,const uint16_t*,const float*,int){std::memcpy(alt,p,sizeof(alt));}
 void symbol_update(uint8_t){}
 void begin(){row_available=false;}
 void abandon(){row_available=false;bit_pending=false;}
 void predict(unsigned,bool){bit_pending=false;}
 void observe(int){bit_pending=false;}
 void commit(uint8_t){std::exit(327);}
};
inline Expert expert;
}
