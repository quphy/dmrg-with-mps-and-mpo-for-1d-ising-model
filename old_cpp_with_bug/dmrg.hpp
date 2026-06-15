#ifndef dmrg_h
#define dmrg_h
#include "mps.hpp"
#include "mpo.hpp"

#include <armadillo>

namespace dmrg{
class DMRG{
public:
mps::random_mps MPS;
mpo::isingmpo MPO;
int L;
double eps;
int chi_max;
arma::field<arma::cube> LPs;
arma::field<arma::cube> RPs;
arma::vec energies;
DMRG(const mps::random_mps &mps, const mpo::isingmpo &mpo,
     double eps_=1e-10, int chi_max_=-1);
arma::cube site_update(int i, double &outEnergy);
void left_to_right();
void right_to_left();
arma::cube next_LP(int i);
arma::cube next_RP(int i);
double expect_mpo() const;
};
}
#endif