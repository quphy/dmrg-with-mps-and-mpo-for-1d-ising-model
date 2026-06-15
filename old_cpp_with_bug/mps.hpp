#ifndef mps_h
#define mps_h
#include <armadillo>
#include <string>


namespace dmrg::mps{
arma::ivec chi_list(int L,int d,int chi_max);
arma::field<arma::cube> create_random_mps(int L, int d, int chi_max);

class random_mps{
public:
 int L;
 int d;
 int chi_max;
 arma::ivec chi;                    
 arma::field<arma::cube> mps;  
 std::string normway; 

 random_mps(int L_,int d_,int chi_max_);
 double self_norm() const;
 void right_normalize();
 void left_normalize();
};

}
#endif