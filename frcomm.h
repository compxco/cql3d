c     frcomm.h
c***********************************************************************
c     BEGIN NFREYA (fr..) common blocks
c***********************************************************************


      include 'frname_decl.h'

c................................................................

      common /numbrs/   nion,nneu,nk,nkt,nj,nbctim,njs
     &  ,ialignf1,zero,one,two,half,pi
c     ONETWO DIVERGENCE

c................................................................

      common /nub/ 
     &  bion(ke,kb),bneut(ke,kb)
     &  ,beamof,ebeam(ke,kb)
     &  ,ennub(k_),fap(ke,kb),fwall(ke,kb)
     &  ,hibr(k_,ke,kb),hibrz(kz,ke,kb)
     &  ,hicmz(kz,ke,kb,3)
     &  ,ftrapfi(kz,ke,kb),ftrapfit(k_,ke,kb)
     &  ,angmpf(k_,ke,kb),angmpz(kz,ke,kb)
     &  ,ibeam,ibion(kb),inubplt,mfm1
cBH170808     &  ,psif(kf),rowpsi(k_)  psif(kf) not used in fr routines
cBH170808             routines, and conflicts with real*8 function psif.
cBH170808             NBI test case showed NO CHANGE.
     &  ,rowpsi(k_)
     &  ,pbeam(ke,kb),tenub(k_)
     &  ,sb(k_,ke,kb),sbcx(k_,2),sbion(k_)
     &  ,spb(k_,ke,kb),qb(k_,ke,kb),qbf(k_,ke,kb)
     &  ,zzi(kz,kion),zne(kz),zni(kz,kion)
     &  ,zte(kz),psivol(kz),freyr(kf),
c     ONETWO DIVERGENCE
     1  zeffctv(kz)
c................................................................

      common /nub2/     bencap(k_,ke,kb),fbe(k_,ke,kb),fbi(k_,ke,kb)
     &  ,bke(k_,ke,kb),bki(k_,ke,kb)
     &  ,ecrit(k_),emzrat(k_),enbeam(k_),enbs(k_)
     &  ,enb(k_,ke,kb),enbsav(k_,ke,kb)
     &  ,enbav(k_,ke,kb),enbav0(k_),enbav1(k_)
     &  ,forb(ke,kb),fb11(ke,kb),fb10(ke,kb)
     &  ,fb01(ke,kb),fb00(ke,kb),fber(ke,kb)
     &  ,hdep(k_,ke,kb),hdepz(kz,ke,kb)
     &  ,ppb(k_,ke,kb),ppbsav(k_,ke,kb),ppbav(k_,ke,kb)
     &  ,pinsid(kf),potsid(kf),rinsid(kf),rotsid(kf)
     &  ,qbsav(k_,ke,kb)
     &  ,sbsav(k_,ke,kb),spbsav(k_,ke,kb)
     &  ,taupb(k_,ke,kb),tauppb(k_,ke,kb)
     &  ,taueb(k_,ke,kb),taus(k_),wbeam(k_)
     &  ,wb(k_,ke,kb),wbsav(k_,ke,kb),wbav(k_,ke,kb)
     &  ,wb11(ke,kb),wb10(ke,kb)
cYuP110316     &  ,wb01(ke,kb),wb00(ke,kb),npts,int_space1
     &  ,wb01(ke,kb),wb00(ke,kb)
     &  ,zetaz(kz,ke,kb),zeta(k_,ke,kb)

      pointer xpts,ypts,zpts,rpts  ! (1:npart)
c     ONETWO DIVERGENCE
      pointer vx,vy,vz  ! (1:npart)
      dimension xpts(:),ypts(:),zpts(:),rpts(:),vx(:),vy(:),vz(:)   
      common /xyzpts/ xpts,ypts,zpts,rpts,vx,vy,vz
c................................................................

      common /nub3/
     &  znipm(kprim),atwpm(kprim),iz(kimp)
     &  ,atwim(kimp),zniim(kimp),zti(kz),ncont
c     nub3 is is used for variables related to hexnb routine
c     note that nouthx and ncorin,also required for hexnb,
c     have been added to block io.

c................................................................

      common /io/       ncrt,nin,nout,nqik,neqplt,ntrplt,nitre,ngreen
     &  ,nbplt,neq,nsvsol,nscr,nrguess,nwguess
     &  ,nmix,ntweak,nyok,nupel,nitrex
     &  ,ialign25,eqdskin_fr,guessin,guessout
     &  ,iprt,ialign26,timprt,mprt,ialign27,prtlst(10)
     &  ,jprt,ialign28,timplt,pltlst(30),jflux,jcoef
     &  ,jsourc,jbal,jtfus,ihead,ineu,inub,irfcalc
     &  ,ialign30,ifred,ialign31,banktime,extime,nterow
     &  ,jterow(10),ilastp,itimav,ialign32,vid,ddebug(50)
     &  ,ncorin,iotoray
cBH070410  mplot removed since not used and possible conflict 
cBH070410  with other use of this variable name.

c................................................................

      character*8 namep,namei 

      common /ions/     namep(kprim),namei(kimp),namen(2),atw(kion)
     &  ,dzdtim(k_,kimp)
     &  ,z(k_,kion),zsq(k_,kion),dzdte(k_,kion)
     &  ,zeff_(k_),rfatmwt,nameu(kkq)

c................................................................
c     ONETWO DIVERGENCE

      common/mhd1/ p(ki,kj),xxx(ki),yyy(kj)
      common/beams/ kblspec(kb),nbeams1(kb)
c................................................................




