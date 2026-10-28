#pragma once
#include <cstdint>
namespace word_label {struct State;bool ListenLive(State*);uint64_t ExactContext(State*);uint32_t ExactCodeNode(State*);}
namespace word_residual {
void clear_feature();void feature(bool,uint64_t,uint32_t);
unsigned predict(unsigned,bool);void observe(int);void report();
}
