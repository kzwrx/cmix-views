#pragma once
#include "metric_fast.h"
// ONE prospective address traversal. Existing metric storage, final reductions,
// key scaling and commit are inherited unchanged from the pinned reference.
template<int N> struct WriteMetricRowStream : WriteMetricFast<N> {
  using Base = WriteMetricFast<N>;
  void address(const uint16_t* ix, const float* u, int nz) {
    // Preserve the public reference operation for generic non-sorted inputs.
    if (nz < 0 || nz > N) { Base::address(ix,u,nz); return; }
    for (int j=1;j<nz;++j)
      if (ix[j] <= ix[j-1]) { Base::address(ix,u,nz); return; }
    this->init(); this->norm2=0;
    for(int j=0;j<nz;++j) {
      if(ix[j]>=N || !std::isfinite(u[j])) std::abort();
      this->norm2 += double(u[j])*u[j];
    }
    this->dot=0; for(auto& x:this->v)x=0;
    int upper=0;
    for(int r=0;r<N;++r) {
      const double* row=this->a.data()+size_t(r)*(r+1)/2;
      while(upper<nz && ix[upper]<=r)++upper;
      // Original low/diagonal contributions, in original active-index order.
      for(int j=0;j<upper;++j)
        this->v[r] += row[ix[j]]*double(u[j]);
      // Higher input coordinates are appended later, in increasing r order.
      // Present zero coefficients still execute their original operations.
      if(upper && ix[upper-1]==r) {
        const double f=u[upper-1];
        for(int i=0;i<r;++i)this->v[i] += row[i]*f;
      }
    }
    for(int j=0;j<nz;++j)this->dot += double(u[j])*this->v[ix[j]];
    if(this->norm2==0) { for(auto& x:this->key)x=0; return; }
    if(!(this->dot>0)||!std::isfinite(this->dot)) {
      std::fprintf(stderr,"METRIC_NONPOSITIVE q=%.17g norm=%.17g writes=%llu\n",
        this->dot,this->norm2,(unsigned long long)this->writes); std::abort();
    }
    const double scale=this->norm2/this->dot;
    for(int i=0;i<N;++i) {
      this->key[i]=scale*this->v[i];
      if(!std::isfinite(this->key[i]))std::abort();
    }
  }
};
