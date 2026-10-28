#pragma once
#include <cstdint>
namespace terminal_route {
struct Router {
 bool enabled=true,pending=false;
 int phase=0;unsigned q0=32768,q1=32768,qmix=32768;
 double score[8]{},curvature[8]{},coefficient_sum[8]{};
 double delta=0,alpha=0,probability=.5;
 uint64_t n[8]{},changed=0,nonzero=0;
 double baseline_bits=0,coded_bits=0;
 unsigned predict(unsigned,int,bool,bool,double,double);
 void observe(int);
};
struct Coupled {
 Router legacy;
 bool enabled=true,scalar_only=false,pending=false;
 int phase=0;unsigned q0=32768,qmix=32768;
 double score[8][2]{},curvature[8][3]{},coefficient_sum[8][2]{};
 double phi[2]{},theta[2]{},probability=.5;
 uint64_t n[8]{},changed=0,nonzero=0;
 double baseline_bits=0,coded_bits=0;
 void coefficients(int,double*)const;
 unsigned predict(unsigned,int,bool,bool,double,double);
 void observe(int);
};
void feature(float,float,bool);
unsigned predict(unsigned,int,bool);
void observe(int);
void close();
void init_trace(const char*);
}
