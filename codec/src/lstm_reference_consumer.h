#pragma once
#include "reference_transfer.h"
#include "lstm_reference.h"
namespace lstm_reference {
inline cubic_overlay::Expert expert;inline bool bit_live=false;inline double cue=0,maximum_neural_span=0,maximum_cue=0;
inline uint64_t rows=0,disabled=0,features=0,exposed=0,nonzero=0;
inline unsigned predict(unsigned q,bool active){const bool ok=active&&bit_live;return expert.predict(q,ok?cue:0.,ok);}
inline void report(){std::fprintf(stderr,"\nLSTM_REFERENCE {\"rows\":%llu,\"disabled\":%llu,\"features\":%llu,\"exposed\":%llu,\"nonzero\":%llu,\"maximum_neural_span\":%.12f,\"maximum_cue\":%.12f,\"updates\":%llu,\"changed_q\":%llu,\"log_odds\":%.12f}\n",(unsigned long long)rows,(unsigned long long)disabled,(unsigned long long)features,(unsigned long long)exposed,(unsigned long long)nonzero,maximum_neural_span,maximum_cue,(unsigned long long)expert.updates,(unsigned long long)expert.changed,expert.log_odds);}
}
