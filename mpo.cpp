#include "mpo.hpp"
#include <armadillo>
#include <iostream>
namespace dmrg::mpo{
isingmpo::isingmpo(int L_,int d_,double h_,double J_){
  L=L_;
  d=d_;
  h=h_;
  J=J_;
  init_Ms();
}

void isingmpo::init_Ms(){
   arma::mat sx={{0.0,1.0},{1.0,0.0}};
   arma::mat sz={{1.0,0.0},{0.0,-1.0}};
   arma::mat id=arma::eye(d,d); 
   int bond=3;
   Ms.set_size(L);
   for(int i=1;i<L-1;i++){
    Ms(i).set_size(bond,bond);
    for(int bL=0;bL<bond;bL++){
        for(int bR=0;bR<bond;bR++){
            Ms(i)(bL,bR)=arma::mat(d,d,arma::fill::zeros);
        }
    }
   }
   Ms(0).set_size(1,bond);
   Ms(L-1).set_size(bond,1);
       for(int bL=0;bL<1;bL++){
        for(int bR=0;bR<bond;bR++){
            Ms(0)(bL,bR)=arma::mat(d,d,arma::fill::zeros);
        }
    }
        for(int bL=0;bL<bond;bL++){
        for(int bR=0;bR<1;bR++){
            Ms(L-1)(bL,bR)=arma::mat(d,d,arma::fill::zeros);
        }
    }
   Ms(0)(0,0)=id;
   Ms(0)(0,1)=sx;
   Ms(0)(0,2)=-h*sz;

   for(int i=1;i<L-1;i++){
   Ms(i)(0,0)=id;
   Ms(i)(0,1)=sx;
   Ms(i)(0,2)=-h*sz;
   Ms(i)(1,2)=-J*sx;
   Ms(i)(2,2)=id;
   }

   Ms(L-1)(0,0)=-h*sz;
   Ms(L-1)(1,0)=-J*sx;
   Ms(L-1)(2,0)=id;
}
}