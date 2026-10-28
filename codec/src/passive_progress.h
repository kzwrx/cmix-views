#pragma once
#include <cstdint>
#include <cstdio>
// Observational counters ONLY. No coder/input flush, model call, RNG or clock.
namespace passive_progress {
struct Snapshot {
  bool parser_active,metadata_open,metadata_done;
  unsigned long long metadata_start,metadata_end;
  long long page;
  bool redirect;
};
class Counter {
  const char* side_;bool encode_,quiet_;
  FILE* log_;unsigned long long last_=0,next_=10000;
  long long page_seen_=0;
  bool opened_=false,done_=false,main_=false;
  void record(const char* event,unsigned long long at,unsigned long long processed,
              unsigned long long total,unsigned long long generated,const Snapshot&s) {
    if(quiet_)return;
    std::fprintf(log_,"\nPREPARED_PROGRESS {\"side\":\"%s\",\"event\":\"%s\",\"boundary_prepared_B\":%llu,\"processed_prepared_B\":%llu,\"total_prepared_B\":%llu,\"parser_page\":%lld,\"archive_generated_including_buffer_B\":",
      side_,event,at,processed,total,s.page);
    if(encode_)std::fprintf(log_,"%llu,\"archive_header_B\":5,\"logical_prepared_output_B\":null",generated);
    else std::fprintf(log_,"null,\"archive_header_B\":null,\"logical_prepared_output_B\":%llu",processed);
    std::fprintf(log_,",\"not_raw_coverage\":true,\"not_physical_file_bytes\":true}\n");
    std::fflush(log_); // stderr/private test log ONLY, never coded stream
  }
  void page_record(unsigned long long processed,unsigned long long generated,const Snapshot&s) {
    if(quiet_)return;
    std::fprintf(log_,"\nPAGE_COUNTER {\"s\":\"%c\",\"p\":%llu,\"ord\":%lld,\"g\":",
      encode_?'E':'D',processed,s.page-1);
    if(encode_)std::fprintf(log_,"%llu,\"h\":5}\n",generated);
    else std::fprintf(log_,"null,\"h\":null}\n");
    std::fflush(log_);
  }
 public:
  Counter(const char*side,bool encode,bool quiet,FILE*log=stderr):side_(side),encode_(encode),quiet_(quiet),log_(log){}
  void observe(unsigned long long processed,unsigned long long total,
               unsigned long long generated,const Snapshot&s) {
    if(quiet_||processed<=last_)return;last_=processed;bool emitted=false;
    if(s.parser_active&&s.metadata_open&&!opened_){opened_=true;record("metadata_start",s.metadata_start,processed,total,generated,s);emitted=true;}
    if(s.parser_active&&s.metadata_done&&!done_){done_=true;record("redirect_start",s.metadata_end,processed,total,generated,s);emitted=true;}
    if(s.parser_active&&done_&&s.page>page_seen_){page_record(processed,generated,s);page_seen_=s.page;}
    // Author parser counts CLOSED pages. This is the next-byte offset after
    // the final redirect page closes, not a raw corpus geography inference.
    if(s.parser_active&&done_&&!s.redirect&&s.page>=0&&!main_){main_=true;record("MAIN_start_by_parser",processed,processed,total,generated,s);emitted=true;}
    if(processed>=next_){if(!emitted)record("tick",processed,processed,total,generated,s);next_=(processed/10000+1)*10000;}
  }
  void finish(unsigned long long processed,unsigned long long total,
              unsigned long long generated,const Snapshot&s){record("eof",processed,processed,total,generated,s);}
};
}
