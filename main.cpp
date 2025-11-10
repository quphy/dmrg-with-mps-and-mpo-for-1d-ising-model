#include "mps.hpp"
#include "mpo.hpp"
#include "dmrg.hpp"
#include <iostream>

int main(int argc,char*argv[]){
    using namespace dmrg;
    int L=4;
    int d=2;
    int chi_max=1024;
    double h=0.5;
    double J=1.0;
    
    mps::random_mps mps(L,d,chi_max);
    mps.right_normalize();

    mpo::isingmpo mpo(L,d,h,J);
    DMRG engine(mps,mpo,1e-10,chi_max);
    return 0; 
}