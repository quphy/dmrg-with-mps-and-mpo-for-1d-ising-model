#ifndef mpo_h
#define mpo_h

#include <armadillo>
namespace dmrg::mpo{
class isingmpo{
public:
int L;
int d;
double h;
double J;
arma::field<arma::field<arma::mat>>Ms;
isingmpo(int L_,int d_,double h_,double J_);
void init_Ms();
};
}
#endif