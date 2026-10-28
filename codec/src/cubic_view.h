#pragma once
#include <cmath>
#include <cstdlib>
namespace cubic_view {
inline double feature(double x0,double x1,double x2,double x3){
 const double x[4]={x0,x1,x2,x3};for(double v:x)if(!std::isfinite(v)||std::abs(v)>4.)std::exit(264);
 const double a=x0*.25,b=x1*.25,c=x2*.25,d=x3*.25;
 const double value=((a*b*c+a*b*d)+a*c*d)+b*c*d;
 if(!std::isfinite(value)||std::abs(value)>4.)std::exit(264);return value;
}
}
