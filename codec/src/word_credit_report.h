#pragma once
#include <cstdio>
#include "word_order.h"
namespace word_credit {
inline void Report(){std::fprintf(stderr,"\nWORD_CREDIT {\"word_steps\":%llu,\"backward_calls\":%llu,\"backprop_steps\":%llu,\"maximum_credit_steps\":%u,\"propagated_absolute_gradient\":%.12f,\"features\":%llu,\"changed_q\":%llu,\"log_odds\":%.12f}\n",(unsigned long long)word_order::steps,(unsigned long long)word_order::backward_steps,(unsigned long long)word_order::backprop_steps,word_order::maximum_credit_steps,word_order::propagated_absolute_gradient,(unsigned long long)word_order::features,(unsigned long long)word_order::expert.changed,word_order::expert.log_odds);}
}
