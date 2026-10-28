#pragma once
#include "view_router.h"
// Stable-reference transfer consumer; upstream learners remain unchanged.
namespace reference_transfer {
inline cubic_overlay::Expert expert;
inline float transferred_bit=.5f,corrected_bit=.5f;inline bool live=false;
inline void feature(float pair,float fw,bool available){transferred_bit=pair;corrected_bit=fw;live=available;}
inline double cue(){return winner_views::Router::clip(winner_views::Router::logit(transferred_bit)-winner_views::Router::logit(corrected_bit));}
inline unsigned predict(unsigned q,bool active){bool eligible=active&&live&&q>=1&&q<=65534;return expert.predict(q,eligible?cue():0.,eligible);}
inline void report(){std::fprintf(stderr,"\nREFERENCE_TRANSFER {\"updates\":%llu,\"changed_q\":%llu}\n",(unsigned long long)expert.updates,(unsigned long long)expert.changed);}
}
