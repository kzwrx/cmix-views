#include "cubic_overlay.h"
#include "reference_transfer.h"
#include "lstm_reference_consumer.h"
#include "word_order_consumer.h"
#include "word_credit_report.h"
#include "word_residual.h"
#include "terminal_route.h"
#include "banked.h"
#include <memory>

namespace banked {
const float initial[NT]={-1,0,0, -3,0,0, -3,0,0, -5,0,0, 0,0,0};
float squash(float z){return .5f+.5f*z/(1.0f+std::fabs(z));}
Memory::Memory(){std::memcpy(theta,initial,sizeof(theta));}
Memory::Memory(const float* t){std::memcpy(theta,t,sizeof(theta));}
static float gate(const float* t,int row,float confidence,float recent){
 return squash(t[row*3]+t[row*3+1]*confidence+t[row*3+2]*recent);
}
Decision Memory::predict(const Input& a)const{
 if(!a.active)return {a.q,0,0};
 float confidence=std::fabs(float(a.q)/32768.f-1.f);
 float trust=.75f*gate(theta,4,confidence,recent[a.bank]);
 float d=0;
 for(int i=0;i<NF;++i)d+=(f[a.bank][a.phase][i]+s[a.bank][a.phase][i])*(float(a.x[i])/256.f);
 float r=std::max(-.75f,std::min(.75f,trust*d));
 unsigned q=a.q;
 if(r!=0){
  double shifted=double(a.q)*(1.0+double(r))/(1.0+double(a.q)*double(r)/65536.0);
  q=unsigned(std::max(1.0,std::min(65534.0,std::floor(shifted+.5))));
 }
 return {q,float(a.q)/65536.f,trust};
}
void Memory::observe(const Input&a,const Decision&d,int bit){
 if(!a.active)return;
 float norm=1.f;
 for(int i=0;i<NF;++i){float v=float(a.x[i])/256.f;norm+=v*v;}
 float e=(float(bit)-float(d.q)/65536.f)/norm;
 for(int i=0;i<NF;++i)grad[a.bank][a.phase][i]+=e*(float(a.x[i])/256.f);
 conf[a.bank][a.phase]=std::fabs(float(a.q)/32768.f-1.f);
 touched[a.bank][a.phase]=1;
 recent[a.bank]=.99f*recent[a.bank]+.01f*std::fabs(float(bit)-float(d.q)/65536.f);
}
void Memory::commit(){
 for(int b=0;b<NB;++b)for(int p=0;p<NP;++p)if(touched[b][p]){
  float c=conf[b][p],r=recent[b];
  float ef=.1f*gate(theta,0,c,r),es=.005f*gate(theta,1,c,r);
  float keepf=1.f-.02f*gate(theta,2,c,r),keeps=1.f-.0001f*gate(theta,3,c,r);
  for(int i=0;i<NF;++i){
   float v=keepf*f[b][p][i]+ef*grad[b][p][i];
   f[b][p][i]=std::max(-2.f,std::min(2.f,v));
   s[b][p][i]=std::max(-2.f,std::min(2.f,keeps*s[b][p][i]+es*(f[b][p][i]-s[b][p][i])));
   grad[b][p][i]=0;
  }
  touched[b][p]=0;
 }
}
unsigned long long fw_updates=0;
uint32_t offset=0,previous=0;
int phase=0;
static constexpr FILE* trace_file=nullptr;
static constexpr Memory* learner=nullptr;
static Decision pending{},parent_pending{};
static unsigned long long records=0,active_records=0;
void init(const char* path,const char* params){
 // Fixed production does not accept research traces or optional bank parameters.
 if(path&&std::strcmp(path,"-"))std::exit(110);
 if(params&&std::strcmp(params,"-"))std::exit(111);
 terminal_route::init_trace(path);
}
void begin_byte(unsigned long long p){offset=uint32_t(p);}
void end_byte(unsigned char c){if(learner)learner->commit();previous=(previous<<8)|c;}
Input make_input(unsigned q,bool active,const float* sketch){
 Input a{};a.offset=offset;a.q=uint16_t(q);a.phase=uint8_t(phase);a.bank=uint8_t((previous^(previous>>7)^(previous>>17))&63);a.active=active;
 if(active){a.x[0]=256;for(int i=1;i<15;++i){float v=sketch[i-1];v=std::max(-4.f,std::min(4.f,v));a.x[i]=int16_t(std::round(v*256.f));}a.x[15]=int16_t(std::round((float(q)/32768.f-1.f)*256.f));}
 return a;
}
unsigned predict_event(const Input&a){pending=learner?learner->predict(a):Decision{a.q,0,0};pending.q=terminal_route::predict(pending.q,a.phase,a.active);parent_pending=pending;pending.q=word_residual::predict(pending.q,a.active);pending.q=cubic_overlay::expert.predict(pending.q,cubic_overlay::cue,a.active&&cubic_overlay::cue_live);pending.q=reference_transfer::predict(pending.q,a.active);pending.q=lstm_reference::predict(pending.q,a.active);pending.q=word_order::PredictBit(pending.q,a.active);return pending.q;}
void decode_observe(const Input&a,int bit){
 terminal_route::observe(bit);
 if(learner)learner->observe(a,parent_pending,bit);
 word_residual::observe(bit);
 cubic_overlay::expert.observe(bit);
 reference_transfer::expert.observe(bit);
 lstm_reference::expert.observe(bit);
 word_order::expert.observe(bit);
 if(trace_file){Record r{};r.in=a;r.bit=uint8_t(bit);r.coded_q=uint16_t(pending.q);if(std::fwrite(&r,sizeof(r),1,trace_file)!=1)std::exit(113);}
 ++records;active_records+=a.active;
}
unsigned encode_event(const Input&a,int bit){
 unsigned q=predict_event(a);decode_observe(a,bit);
 return q;
}
void close(){reference_transfer::report();lstm_reference::report();word_order::Report();word_credit::Report();word_residual::report();terminal_route::close();if(trace_file){Record end{};end.in.active=2;end.in.offset=offset+1;if(fwrite(&end,sizeof(end),1,trace_file)!=1)std::exit(115);if(std::fclose(trace_file))std::exit(114);}std::fprintf(stderr,"\nPOC records=%llu active=%llu fw_updates=%llu state_bytes=%zu\n",records,active_records,fw_updates,sizeof(Memory));}
}

// Reference-row arithmetic uses this precise, contraction-disabled translation unit.
namespace lstm_reference {
void Rows::set(const float*neural,const float*base,bool available){
 live=false;if(!available){++disabled;return;}
 double maximum=-1e300,lo=1e300,hi=-1e300;unsigned count=0;
 for(unsigned b=0;b<256;++b){
  if(!std::isfinite(base[b])||base[b]<0)std::exit(340);teacher[b]=double(base[b]);
  if(base[b]>0){if(!std::isfinite(neural[b]))std::exit(340);const double n=double(neural[b]);transferred[b]=std::log(double(base[b]))+n;maximum=std::max(maximum,transferred[b]);lo=std::min(lo,n);hi=std::max(hi,n);++count;}
 }
 if(count==0)std::exit(340);double total=0;
 for(unsigned b=0;b<256;++b){transferred[b]=teacher[b]>0?std::exp(transferred[b]-maximum):0.;total+=transferred[b];}
 if(!(total>0)||!std::isfinite(total))std::exit(340);
 for(unsigned b=0;b<256;++b)transferred[b]/=total;
 maximum_neural_span=std::max(maximum_neural_span,hi-lo);live=true;++rows;
}
void Rows::feature(unsigned prefix)const{
 bit_live=false;cue=0;++features;if(!live)return;if(prefix<1||prefix>255)std::exit(341);
 const unsigned depth=31u-unsigned(__builtin_clz(prefix)),size=1u<<(8u-depth),start=(prefix-(1u<<depth))<<(8u-depth),mid=start+size/2,end=start+size;
 double t0=0,t1=0,r0=0,r1=0;
 for(unsigned b=start;b<mid;++b){t0+=teacher[b];r0+=transferred[b];}
 for(unsigned b=mid;b<end;++b){t1+=teacher[b];r1+=transferred[b];}
 if(!(t0>0&&t1>0&&r0>0&&r1>0))return;
 const double raw=std::log(r1)-std::log(r0)-std::log(t1)+std::log(t0);
 if(!std::isfinite(raw))std::exit(341);cue=std::max(-4.,std::min(4.,raw));bit_live=true;++exposed;nonzero+=cue!=0;maximum_cue=std::max(maximum_cue,std::abs(cue));
}
}
