#include "mps.hpp"

#include <armadillo>
#include <cmath>

namespace dmrg::mps{
arma::ivec chi_list(int L, int d, int chi_max){
    arma::ivec a=arma::ones<arma::ivec>(L+1);
    for (int i=0;i<=L/2;i++){
     if(std::pow(d,i)<=chi_max){
        a(i)=std::pow(d,i);
        a(L-i)=std::pow(d,i);  
    }else{
        a(i)=chi_max;
        a(L-i)=chi_max;  
    }
    }
   return a;
}
arma::field<arma::cube> create_random_mps(int L, int d, int chi_max){
   arma::ivec chi=chi_list(L,d,chi_max);
   arma::field<arma::cube>mps(L);
   for(int i=0;i<L;++i){
      int chiL=chi(i);
      int chiR=chi(i+1);
      mps(i) = arma::cube(chiL, d, chiR, arma::fill::randu);
    }
    return mps;
}

random_mps::random_mps(int L_,int d_,int chi_max_){
    L=L_;
    d=d_;
    chi_max=chi_max_;
    chi=chi_list(L,d,chi_max);
    mps=create_random_mps(L,d,chi_max);
    normway="";
}


double random_mps::self_norm() const {
    
    const arma::cube &m0 = mps(0);
    int chi0 = static_cast<int>(m0.n_rows);
    arma::mat F = arma::ones<arma::mat>(chi0, chi0); 

    for (int site = 0; site < L; ++site) {
        const arma::cube &m = mps(site);
        int chiL = static_cast<int>(m.n_rows);
        int dloc = static_cast<int>(m.n_cols);
        int chiR = static_cast<int>(m.n_slices);

        arma::mat newF = arma::zeros<arma::mat>(chiR, chiR);

        for (int s = 0; s < dloc; ++s) {
            arma::mat Xs(chiL, chiR, arma::fill::zeros);
            for (int aR = 0; aR < chiR; ++aR) {
                Xs.col(aR) = m.slice(aR).col(s);
            }
            newF += Xs.t() * F * Xs;
        }

        F = std::move(newF);
    }


    return F(0,0);
}

void random_mps::right_normalize(){
arma::field<arma::cube> Bs(L);
arma::field<arma::cube>mps_local=mps;
for (int i=L-1;i>=0;i--){
    int chil=mps_local(i).n_rows;
    int dloc=mps_local(i).n_cols;
    int chir=mps_local(i).n_slices;
    
    arma::cube ma=arma::reshape(mps_local(i),chil,dloc*chir,1);
    arma::mat m=ma.slice(0);
    arma::mat mT=m.t();

    arma::mat Q,R;
    arma::qr_econ(Q,R,mT);

    int r=Q.n_cols;
    arma::mat QT=Q.t();
    arma::cube QTm(r,dloc*chir,1,arma::fill::zeros);
    QTm.slice(0)=QT;
    arma::cube B=arma::reshape(QTm,r,dloc,chir);
    Bs(i)=B;


    if(i>0){
    arma::mat RT=R.t(); 
    arma::cube mprev=mps_local(i-1);
    int chilp=mprev.n_rows;
    int dprev=mprev.n_cols;
    int chilc=mprev.n_slices;
    arma::cube newM(chilp,dprev,r,arma::fill::zeros);
    for (int a=0;a<chilp;a++){
        for(int s=0;s<dprev;s++){
            for(int k=0;k<r;k++){
               double sum=0;
               for(int b=0;b<chilc;b++){
                sum+=mprev(a,s,b)*RT(b,k);
               } 
               newM(a,s,k)=sum;
            }
        }
    }

    mps_local(i-1)=std::move(newM);

    }
}
mps=std::move(Bs);
normway="right-normalization";
}


void random_mps::left_normalize(){
arma::field<arma::cube>As(L);
arma::field<arma::cube>mps_local=mps;
for(int i=0;i<L;i++){
    int chil=mps_local(i).n_rows;
    int dloc=mps_local(i).n_cols;
    int chir=mps_local(i).n_slices;

    arma::cube ma=arma::reshape(mps_local(i),chil*dloc,chir,1);
    arma::mat m=ma.slice(0);
    arma::mat Q,R;
    qr_econ(Q,R,m);
    
    int r=Q.n_cols;
    arma::cube Qm(r,chil*dloc,1,arma::fill::zeros);
    Qm.slice(0)=Q;
    arma::cube A=arma::reshape(Qm,chil,dloc,r);
    As(i)=A;
    
    if(i<L-1){
       arma::cube mnext = mps_local(i+1); 
       int chil2=mnext.n_rows;
       int d2=mnext.n_cols;
       int chir2=mnext.n_slices;
       arma::cube newM(r,d2,chir2,arma::fill::zeros);
       for(int alpha=0;alpha<r;alpha++){
        for(int s=0;s<d2;s++){
          for(int beta=0;beta<chir2;beta++){
            double sum=0;
            for(int k=0;k<chil2;k++){
                sum+=R(alpha,k)*mnext(k,s,beta);
            }
            newM(alpha,s,beta)=sum;
          }
        }
       }

     mps_local(i+1)=std::move(newM);

    }
}

mps=std::move(As);
normway="left-normalization";
}


}