#pragma once
#include "reference_transfer.h"
#include "word_order.h"
namespace word_order {
inline cubic_overlay::Expert expert;
inline unsigned PredictBit(unsigned q,bool active){const bool ok=active&&feature_live;return expert.predict(q,ok?feature:0.,ok);}
inline void Report(){std::fprintf(stderr,"\nWORD_ORDER {\"models\":%llu,\"steps\":%llu,\"backward_steps\":%llu,\"out_updates\":%llu,\"gate_cell_updates\":%llu,\"features\":%llu,\"nonzero_features\":%llu,\"cancels\":%llu,\"output_state_B_per_model\":%zu,\"maximum_hidden\":%.12f,\"maximum_feature\":%.12f,\"updates\":%llu,\"changed_q\":%llu,\"log_odds\":%.12f}\n",(unsigned long long)models,(unsigned long long)steps,(unsigned long long)backward_steps,(unsigned long long)out_updates,(unsigned long long)gate_cell_updates,(unsigned long long)features,(unsigned long long)nonzero_features,(unsigned long long)cancels,size_t(Model::NODES)*(Model::D+1)*sizeof(float),maximum_hidden,maximum_feature,(unsigned long long)expert.updates,(unsigned long long)expert.changed,expert.log_odds);}
}
