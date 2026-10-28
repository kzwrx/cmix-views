#pragma once
namespace lstm_reference {
struct Rows{double transferred[256]{},teacher[256]{};bool live=false;void set(const float*,const float*,bool);void feature(unsigned)const;};
}
