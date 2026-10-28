#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
// Full causal inverse Gram metric. No target, class probability, or future key
// is an input to the addressing operation. Numerical failures are not clipped.
template<int N>struct WriteMetricFast{
 std::vector<double>a,v,key;double dot=0,norm2=0;uint64_t writes=0;
 static constexpr size_t cells=size_t(N)*(N+1)/2;
 static size_t at(int i,int j){if(i<j){int t=i;i=j;j=t;}return size_t(i)*(i+1)/2+j;}
 void init(){if(!a.empty())return;a.assign(cells,0.);v.resize(N);key.resize(N);for(int i=0;i<N;++i)a[at(i,i)]=1.;}
 void address(const uint16_t*ix,const float*u,int nz){init();if(nz<0||nz>N)std::abort();norm2=0;for(int j=0;j<nz;++j){if(ix[j]>=N||!std::isfinite(u[j]))std::abort();norm2+=double(u[j])*u[j];}
  dot=0;for(auto&x:v)x=0;for(int j=0;j<nz;++j){const int n=ix[j];const double f=u[j];const double*row=a.data()+size_t(n)*(n+1)/2;for(int i=0;i<n;++i)v[i]+=row[i]*f;for(int i=n;i<N;++i)v[i]+=a[size_t(i)*(i+1)/2+n]*f;}
  for(int j=0;j<nz;++j)dot+=double(u[j])*v[ix[j]];
  if(norm2==0){for(auto&x:key)x=0;return;}
  if(!(dot>0)||!std::isfinite(dot)){std::fprintf(stderr,"METRIC_NONPOSITIVE q=%.17g norm=%.17g writes=%llu\n",dot,norm2,(unsigned long long)writes);std::abort();}
  const double scale=norm2/dot;for(int i=0;i<N;++i){key[i]=scale*v[i];if(!std::isfinite(key[i]))std::abort();}
 }
 void commit(){if(norm2>0){const double den=1+dot;for(int i=0;i<N;++i)if(v[i]!=0){const double t=v[i]/den;double*row=a.data()+size_t(i)*(i+1)/2;for(int j=0;j<=i;++j)row[j]-=t*v[j];}for(int i=0;i<N;++i)if(!(a[at(i,i)]>0)||!std::isfinite(a[at(i,i)]))std::abort();}++writes;}
};
