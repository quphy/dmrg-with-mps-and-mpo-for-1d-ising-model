#include "dmrg.hpp"
#include <armadillo>
#include <iostream>

namespace dmrg{

DMRG::DMRG(const mps::random_mps &mps,const mpo::isingmpo &mpo,double eps_,int chi_max_):
           MPS(mps),MPO(mpo),eps(eps_)
{
    L = MPS.L;
    if(chi_max_<=0) {
        chi_max=1<<L;
    } else {
        chi_max=chi_max_;
    }
    //默认右正则为起点
    if(MPS.normway!="right-normalization"){
        mps::random_mps tmp=MPS;
        tmp.right_normalize();
        MPS=tmp;
    }
    LPs.set_size(L);
    RPs.set_size(L);
    LPs(0)=arma::cube(1,1,1,arma::fill::ones);
    RPs(L-1)=arma::cube(1,1,1,arma::fill::ones);
    for(int i=L-1;i>0;i--){
        RPs(i-1)=next_RP(i);
    }
    energies.reset(); 
}
arma::cube DMRG::next_LP(int i){
   const arma::cube &F=LPs(i);
   const arma::cube &M=MPS.mps(i);
   int chil=M.n_rows;
   int dloc=M.n_cols;
   int chir=M.n_slices;
   const arma::field<arma::mat> &Msite = MPO.Ms(i);
   int bLmax=Msite.n_rows;
   int bRmax=Msite.n_cols;
   arma::cube newF(chir,bRmax,chir,arma::fill::zeros);
   for(int al=0;al<chil;al++){
        for(int bl=0;bl<bLmax;bl++){
            for(int alp=0;alp<chil;alp++){
                double fval=F(al,bl,alp);
                for(int s=0;s<dloc;s++){
                    for(int sp=0;sp<dloc;sp++){
                        for(int br=0;br<bRmax;br++){
                            double m=Msite(bl,br)(s,sp);
                            for(int ar=0;ar<chir;ar++){
                                for(int arp=0;arp<chir;arp++){
                                    newF(ar,br,arp)+=fval*M(al,s,ar)*m*M(alp,sp,arp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return newF;  
}

arma::cube DMRG::next_RP(int i){
  const arma::cube &F=RPs(i);
  const arma::cube &M=MPS.mps(i);
  int chil=M.n_rows;
  int dloc=M.n_cols;
  int chir=M.n_slices;
  const arma::field<arma::mat> &Msite = MPO.Ms(i);
  int bLmax=Msite.n_rows;
  int bRmax=Msite.n_cols;
  arma::cube newF(chil,bLmax,chil,arma::fill::zeros);
  for(int al=0;al<chil;al++){
    for(int bl=0;bl<bLmax;bl++){
        for(int alp=0;alp<chil;alp++){
            for(int ar=0;ar<chir;ar++){
                for(int br=0;br<bRmax;br++){
                    for(int arp=0;arp<chir;arp++){
                        double fval=F(ar,br,arp);
                        for(int s=0;s<dloc;s++){
                          for(int sp=0;sp<dloc;sp++){
                                double m = Msite(bl,br)(s,sp);
                                newF(al,bl,alp)+=M(al,s,ar)*m*fval*M(alp,sp,arp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return newF;
}

arma::cube DMRG::site_update(int i, double &outEnergy){
    const arma::cube &LP = LPs(i);
    const arma::cube &RP = RPs(i);
    const arma::cube &M = MPS.mps(i);
    const arma::field<arma::mat> &Msite = MPO.Ms(i);

    int chiL = M.n_rows;
    int dloc = M.n_cols;
    int chiR = M.n_slices;
    int bLmax = Msite.n_rows;
    int bRmax = Msite.n_cols;

    int N=chiL*dloc*chiR;
    arma::mat H(N,N,arma::fill::zeros);

    auto idx=[&](int aL,int s,int aR){
        return aL+s*chiL+aR*(chiL*dloc);
    };

    for(int al=0; al<chiL; ++al){
        for(int s=0; s<dloc; ++s){
            for(int aR=0; aR<chiR; ++aR){
                int r = idx(al,s,aR);
                for(int alp=0; alp<chiL; ++alp){
                    for(int sp=0; sp<dloc; ++sp){
                        for(int apR=0; apR<chiR; ++apR){
                            int c=idx(alp,sp,apR);
                            double sum = 0.0;
                            for(int bL=0;bL<bLmax;++bL){
                                for(int bR=0;bR<bRmax;++bR){
                                    double m=Msite(bL,bR)(s,sp);
                                    sum += LP(al,bL,alp)*m*RP(aR,bR,apR);
                                }
                            }
                            H(r,c) = sum;
                        }
                    }
                }
            }
        }
    }
    arma::vec eigval;
    arma::mat eigvec;
    bool ok=false;
    arma::sp_mat Hm(H);
    try {
     ok=arma::eigs_sym(eigval, eigvec, Hm, 1, "sa");
    }
    catch(...) 
    { 
    ok = false;
    }
    if(!ok){
        arma::vec D;
        arma::mat V;
        arma::eig_sym(D, V, H);
        eigval = arma::vec(1);
        eigval(0) = D(0);
        eigvec = V.cols(0,0);
    }
    double e = eigval(0);
    outEnergy = e;
    arma::vec v = eigvec.col(0);
    arma::cube newM(chiL, dloc, chiR, arma::fill::zeros);
    for(int al=0; al<chiL; ++al){
        for(int s=0; s<dloc; ++s){
            for(int aR=0; aR<chiR; ++aR){
                int index = idx(al,s,aR);
                newM(al,s,aR) = v(index);
            }
        }
    }
    return newM;
}

void DMRG::left_to_right(){
    if(MPS.normway!= "right-normalization"){
        std::cerr<<"MPS must be right-normalized to run left_to_right"<<std::endl;
        return;
    }
    for(int i=0;i<L;++i){
        double e;
        arma::cube Mnew=site_update(i, e);
        energies.insert_rows(energies.n_rows, arma::vec({e}));

        int chiL=Mnew.n_rows;
        int dloc=Mnew.n_cols;
        int chiR=Mnew.n_slices;

        arma::cube Psi_cu=arma::reshape(Mnew, chiL*dloc,chiR,1);
        arma::mat Psi=Psi_cu.slice(0);
        arma::mat U,V;
        arma::vec S;
        svd_econ(U,S,V,Psi,"right");

        int newchi=std::min((int)S.n_elem, chi_max);
        if(newchi==0) newchi=1;
        arma::mat Ut=U.cols(0,newchi-1);
        arma::vec St=S.rows(0,newchi-1);
        arma::mat Vt=V.cols(0,newchi-1).t();

        St=St/arma::norm(St);
        arma::cube Utm(Ut.n_rows,Ut.n_cols,1,arma::fill::zeros);
        Utm.slice(0)=Ut;
        arma::cube A=arma::reshape(Utm, chiL, dloc, newchi);
        MPS.mps(i)=A;

        if(i < L-1){
            arma::mat SV=arma::diagmat(St) * Vt;
            arma::cube Next=MPS.mps(i+1);
            int oldChiR=Next.n_rows;
            int d2=Next.n_cols;
            int chiRp=Next.n_slices;
            arma::cube newNext(newchi,d2,chiRp,arma::fill::zeros);
            for(int alpha=0;alpha<newchi;++alpha)
                for(int s=0;s<d2;++s)
                    for(int beta=0;beta<chiRp;++beta){
                        double sum=0;
                        for(int k=0;k<oldChiR;++k)
                            sum+=SV(alpha,k)*Next(k,s,beta);
                        newNext(alpha,s,beta)=sum;
                    }
            MPS.mps(i+1)=std::move(newNext);
        }
        if(i<L-1) LPs(i+1) = next_LP(i);
    }
    MPS.normway = "left-normalization";
}

void DMRG::right_to_left(){
    if(MPS.normway != "left-normalization"){
        std::cerr<<"MPS must be left-normalized to run right_to_left"<<std::endl;
        return;
    }
    for(int k=1;k<=L;++k){
        int i=L-k;
        double e;
        arma::cube Mopt=site_update(i, e);
        energies.insert_rows(energies.n_rows, arma::vec({e}));

        int chiL=Mopt.n_rows;
        int dloc=Mopt.n_cols;
        int chiR=Mopt.n_slices;
        arma::cube Psi_cu = arma::reshape(Mopt, chiL, dloc * chiR,1);
        arma::mat Psi=Psi_cu.slice(0);
        arma::mat U,V;
        arma::vec S;
        svd_econ(U,S,V,Psi,"left");

        int newchi=std::min((int)S.n_elem, chi_max);
        if(newchi==0) newchi=1;
        arma::mat Ut=U.cols(0,newchi-1);
        arma::vec St=S.rows(0,newchi-1);
        arma::mat Vt=V.cols(0,newchi-1).t();

        St=St/arma::norm(St);
        arma::cube Vtm(Vt.n_rows,Vt.n_cols,1,arma::fill::zeros);
        Vtm.slice(0)=Vt;
        arma::cube Bfinal=arma::reshape(Vtm, newchi, dloc, chiR);
        MPS.mps(i)=Bfinal;

        if(i>0){
            arma::mat US=Ut*arma::diagmat(St);
            arma::cube Left=MPS.mps(i-1);
            int chiLp=Left.n_rows;
            int d2=Left.n_cols;
            int chiR2=Left.n_slices;
            arma::cube newLeft(chiLp,d2,newchi,arma::fill::zeros);
            for(int a=0;a<chiLp;++a)
                for(int s=0;s<d2;++s)
                    for(int b=0;b<newchi;++b){
                        double sum=0;
                        for(int k=0;k<chiR2;++k)
                            sum += Left(a,s,k) * US(k,b);
                        newLeft(a,s,b) = sum;
                    }
            MPS.mps(i-1) = std::move(newLeft);
        }

        if(i>0) RPs(i-1) = next_RP(i);
    }
    MPS.normway="right-normalization";
}

double DMRG::expect_mpo() const {
    arma::cube F(1,1,1,arma::fill::ones);
    for(int i=0;i<L;++i){
        const arma::cube &M=MPS.mps(i);
        const arma::field<arma::mat> &Msite=MPO.Ms(i);
        int chiL=M.n_rows;
        int dloc=M.n_cols;
        int chiR=M.n_slices;
        int bLmax=Msite.n_rows;
        int bRmax=Msite.n_cols;
        arma::cube newF(chiR, bRmax, chiR,arma::fill::zeros);
        for(int aL=0; aL<chiL; ++aL){
            for(int bL=0; bL<bLmax; ++bL){
                for(int aLp=0; aLp<chiL; ++aLp){
                    double fval = F(aL,bL,aLp);
                    if(std::abs(fval) < 1e-16) continue;
                    for(int s=0;s<dloc;++s){
                        for(int sp=0; sp<dloc; ++sp){
                            for(int bR=0; bR<bRmax; ++bR){
                                double m = Msite(bL,bR)(s,sp);
                                for(int aR=0; aR<chiR; ++aR){
                                    for(int aRp=0; aRp<chiR; ++aRp){
                                        newF(aR,bR,aRp)+=fval*M(aL,s,aR)*m*M(aLp,sp,aRp);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        F = std::move(newF);
    }
    return F(0,0,0);
}
}