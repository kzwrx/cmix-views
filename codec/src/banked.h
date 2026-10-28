#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Decoder-causal POC only. Model/predictor/head state is never writable here.
namespace banked {
constexpr int NB=64, NP=8, NF=16, NT=15;
struct Input { uint32_t offset; uint16_t q; uint8_t phase, bank, active, pad[3]; int16_t x[NF]; };
struct Record { Input in; uint8_t bit, pad; uint16_t coded_q; };
static_assert(sizeof(Input)==44 && sizeof(Record)==48,"pinned trace ABI");
struct Decision { unsigned q; float error_base, trust; };
struct Memory {
 float f[NB][NP][NF]{}, s[NB][NP][NF]{};
 float grad[NB][NP][NF]{}, conf[NB][NP]{}, recent[NB]{};
 uint8_t touched[NB][NP]{};
 float theta[NT]{};
 Memory();
 explicit Memory(const float* t);
 Decision predict(const Input& a) const;
 void observe(const Input& a,const Decision& d,int bit);
 void commit();
};
float squash(float z);
extern const float initial[NT];
extern unsigned long long fw_updates;
extern uint32_t offset, previous;
extern int phase;
void init(const char* trace,const char* parameters);
void begin_byte(unsigned long long p);
void end_byte(unsigned char c);
Input make_input(unsigned q,bool active,const float* sketch);
unsigned encode_event(const Input&,int bit);
unsigned predict_event(const Input&);
void decode_observe(const Input&,int bit);
void close();
}
