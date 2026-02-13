c
c
      real*8 function bkef(vpar, tpar, emzpar)
      implicit integer (i-n), real*8 (a-h,o-z)
c
c     This routine evaluates the function Ke, which is related to the trans-
c     fer of parallel momentum from fast ions slowing down on electrons.
c     The function is defined by Callen et al., IAEA Tokyo, Vol. I, 645
c     (1974).
c
      external bkefun
      common /gecom/ vcvo, tstcx, emzrat
      vcvo = vpar
      tstcx = tpar
      emzrat = emzpar
      a = 0.
      b = 1.
      ep = 0.1
      m4 = 3
      bkef = asimp(a,b,ep,m,m4,bkefun)
      return
      end


      real*8 function bkefun(y)
      implicit integer (i-n), real*8 (a-h,o-z)
      common /gecom/ vcvo, tstcx, emzrat
      zero=0.d0
      bkefun = zero
      if(y.eq.zero) return
      v3 = vcvo**3
      arg = (1.+v3)/(y**3+v3)
      pcx = (arg)**(-tstcx/3.)
      b = (y**3*arg)**(emzrat/3.)
      bkefun = y**3*pcx*b/(y**3+v3)
      return
      end

      subroutine bproc(atw,ebkev,ibion,ke,kj,mb,nj,nion,
     &  ene,en,te,z,sb,nprim,sbcx,sbion,ibstart)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c----------------------------------------------------------------
cBH171014:  Doesn't appear to be used anywhere.
cBH171014:  Modifying it with ibstart anyway.
c----------------------------------------------------------------
c----------------------------------------------------------------
c     inputs
c----------------------------------------------------------------
c     atw(i)             : atomic weight of ion species i
c     ebkev(ib)          : maximum particle energy in beam ib
c     ibion              : ion index of beam species
c     ke                 : maximum number of beam components
c     kj                 : maximum number of radial points
c     mb                 : number of beams
c     nj                 : number of radial points
c     nion               : number of ions
c     ene                : electron density (#/cm**3)
c     en(j,k)            : ion density at r(j) for species k (#/cm**3)
c     te(j)              : electron temperature at r(j) (kev)
c     z(j,k)             : z at r(j) for ion species k
c     sb(j,ic,ib)        : hot ion source at r(j) , beam ib, component ic
c     (#/cm**3-sec)
c----------------------------------------------------------------
c     outputs
c----------------------------------------------------------------
c     sbcx(j,i)          : hot ion source at r(j) due to charge exchange
c     with primary ion species i   (#/cm**3-sec)
c     sbion(j)           : hot ion source at r(j) due to ionization
c     by electrons  (#/cm**3-sec)
c----------------------------------------------------------------
c
c     mean free paths for charge exchange, proton ionization and
c     electron ionization are calculated from the fitted results
c     of freeman and jones, clm-r137, culham (1974).
c
c
      dimension atw(*),ebkev(*),ene(*),en(kj,*),te(kj)
      dimension sb(kj,ke,*),sbcx(kj,2),sbion(*)
      dimension z(kj,*)
ccc   
      dimension cfione(7),cfionp(7),rpathcxn(2)
ccc   
      data cfione/-3.173850D+01,1.143818D+01,
     1  -3.833998,7.046692D-01,-7.431486D-02,4.153749D-03,
     2  -9.486967D-05/
      data cfionp/-4.203309D+01,3.557321,
     1  -1.045134,3.139238D-01,-7.454475D-02,8.459113D-03,
     2  -3.495444D-04/
ccc   
      do 5000 j=1,nj
        sbcx(j,1)=0.
        sbcx(j,2)=0.
        sbion(j)=0.
        alogt=0.
        teev = 1.e3*te(j)
c990131        if(teev.gt.1.) alogt=alog(teev)
        if(teev.gt.1.) alogt=log(teev)
        if(teev.gt.1.D+05) alogt=11.51
        expo=cfione(7)
        do 2000 ii=1,6
          if7=7-ii
          expo=expo*alogt+cfione(if7)
 2000   continue
        sgvxne=exp(expo)*ene(j)
ccc   
        do 4000 ib=ibstart,mb
          do 4001 ic=1,3
            ebev=1.e3* ebkev(ib)/(ic*atw(ibion))
            vbeam=1.384e6*sqrt(ebev)
            veli=1./vbeam
            rpathei=sgvxne*veli
            rpath=rpathei
            rpathcxn(1)=0.
            rpathcxn(2)=0.
            do 3200 k=1,nion
              if(atw(k).gt.3.01) go to 2500
              e = 1.e3*ebkev(ib)/(ic*atw(ibion))
c990131              aloge=alog10(e)
              aloge=log10(e)
              sigcx=.6937D-14*(1.-.155*aloge)**2/
     1          (1.+.1112D-14*e**3.3)
              aloge=aloge*2.302585093-6.907755279
              if(aloge.le.-2.30258) go to 2220
              expo=cfionp(7)
              do 2200 ii=1,6
                if7=7-ii
                expo=expo*aloge+cfionp(if7)
 2200         continue
              sigi=exp(expo)
              go to 2240
 2220         continue
              sigi=0.
 2240         continue
ccc   
              rpathcx = sigcx*en(j,k)
              rpathii = sigi*en(j,k)
              go to 3000
c------------------------------------------------------------
c     r.e. olson et al., phys. rev. let. 41, 163 (1978)
c     ionization by impact with impurity
c------------------------------------------------------------
 2500         continue
              ebi=ebkev(ib)/(ic*atw(ibion))
              rpathii=1.0D-17*en(j,k)*46.*z(j,k)*
     .          (32.*z(j,k)/ebi)*(1.-exp(-ebi/(32.*z(j,k))))
              rpathcx=0.
c------------------------------------------------------------
 3000         continue
              rpath=rpath+rpathcx+rpathii
              if(k.le.nprim.and.k.le.2) rpathcxn(k)=rpathcx
 3200       continue
            fractcx1=rpathcxn(1)/rpath
            fractcx2=rpathcxn(2)/rpath
            sbcx(j,1)=sbcx(j,1) + fractcx1*sb(j,ic,ib)
            sbcx(j,2)=sbcx(j,2) + fractcx2*sb(j,ic,ib)
            sbion(j)=sbion(j) + (1.-fractcx1-fractcx2)*sb(j,ic,ib)
 4001     continue
 4000   continue  !On do ib=ibstart,mb
 5000 continue
ccc   
      return
      end
c
c
      real*8 function ceef(i,j,te)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
c--   e-impact excitation rate coefficient from vriens & smeets.
c     used only for excitations to final state with n > 2 (i.e. j > 3).
c
      parameter (ms=21,mc=35)
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
      common/b2/nouthx,istart,ihxbug
      common/b4/en(mc+1),dg(mc+1),ae(ms,mc),be(ms,mc),
     *  de1(ms,mc),de2(ms,mc),ge1(ms,mc),ge2(ms,mc)
      data ryd/13.6/
      ceef1(ni,nj,te)=1.6D-7*sqrt(te)
     *  *exp(-(ryd/te)*(1./dfloat(ni)**2-1./dfloat(nj)**2))
c990131     *  *(ae(ni,nj)*alog(.3*te/ryd+de2(ni,nj))+be(ni,nj))
     *  *(ae(ni,nj)*log(.3*te/ryd+de2(ni,nj))+be(ni,nj))
c990131     *  /(te+ge2(ni,nj)*alog(1.+(dfloat(ni)**3)*te/ryd))
     *  /(te+ge2(ni,nj)*log(1.+(dfloat(ni)**3)*te/ryd))
c
      if((i.ge.j) .or. (j.le.3)) then
        ihxbug = 8
        if(nouthx.gt.0) write(nouthx,3737) i,j
 3737   format(' ??? ceef error: i4= ',i3,'        j= ',i3)
        return
      endif
c
      ni=nfhx(i)
      nj=j-1
      ceef=ceef1(ni,nj,te)
      id=2
      if(ni.eq.1)id=1
      return
      end function ceef
      
      subroutine chek(x,xmin,xmax,imin,imax,ick)
      implicit integer (i-n), real*8 (a-h,o-z)
      ick = 0
      if(imin.eq.0) go to 10
      if(x.ge.xmin) go to 10
      ick = 1
      return
 10   continue
      if(imax.eq.0) return
      if(x.gt.xmax) ick = 1
      return
      end subroutine chek
c
c
      real*8 function cief(i,te)
      implicit integer (i-n), real*8 (a-h,o-z)
c*****************************************************************
c--   e-impact ionization rate coefficient from vriens & smeets,
c     phys. rev. a 22, 940 (1980).
c     used only for 2p (i=3) or for n > 2 (i=n+1).
c
      parameter (ms=21,mc=35)
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
      common/b2/nouthx,istart,ihxbug
      common/b4/en(mc+1),dg(mc+1),ae(ms,mc),be(ms,mc),
     *  de1(ms,mc),de2(ms,mc),ge1(ms,mc),ge2(ms,mc)
      ent(i)=en(i)/te
c
      if(i.le.2) then
        ihxbug = 7
        if(nouthx.gt.0) write(nouthx,3737) i
 3737   format(' ??? cief error: i4= ',i3)
        return
      endif

      cief=9.56D-6*exp(-ent(i))
     *  /(te*sqrt(te)*(ent(i)**2.33+4.38*ent(i)**1.72+1.32*ent(i)))
      return
      end function cief

cKinsey substituted crsecs version below (from onetwo)
c$$$c     ONETWO DIVERGENCE
c$$$      subroutine crsecs(atw, ebkev, ibion, ke, kz, mb, mfm1,
c$$$     *  nion, vbeam, zne, zni, zte, zzi,
c$$$     *  sgvxne, sgxn, sgxnmi,iexcit,
c$$$     *  dtemax,dnemax,dzemax)
c$$$      implicit integer (i-n), real*8 (a-h,o-z)
c$$$      save
c$$$c
c$$$c     ONETWO DIVERGENCE
c$$$
c$$$c     this subroutine calculates charge exchange, proton ionization, and
c$$$c     electron ionization cross sections from the fitted results of
c$$$c     freeman and jones, clm-r137, culham (1974) for iexcit=0.
c$$$c     But for iexcit=-1 W. Stearns simplified formula is used.
c$$$c
c$$$      dimension atw(*), ebkev(*), vbeam(ke,*),
c$$$     *  zne(*), zni(kz,*), zte(*), zzi(kz,*),
c$$$     *  sgvxne(*), sgxn(kz,ke,*), sgxnmi(ke,*)
c$$$ccc   
c$$$      dimension cfione(7),cfionp(7)
c$$$ccc   
c$$$      data cfione/-3.173850e+01,1.143818e+01,
c$$$     1  -3.833998,7.046692e-01,-7.431486e-02,4.153749e-03,
c$$$     2  -9.486967e-05/
c$$$      data cfionp/-4.203309e+01,3.557321,
c$$$     1  -1.045134,3.139238e-01,-7.454475e-02,8.459113e-03,
c$$$     2  -3.495444e-04/
c$$$ccc 
c$$$c      write(*,*) 'cfione(1),cfione(2),cfionp(1),cfionp(1)',
c$$$c     +            cfione(1),cfione(2),cfionp(1),cfionp(1)
c$$$c     ONETWO DIVERGENCE
c$$$      if (iexcit.eq.-1) go to 100
c$$$      do 10 i=1,mfm1
c$$$        alogt=0.
c$$$        teev = 1.e3*zte(i)
c$$$c990131        if(teev.gt.1.) alogt=alog(teev)
c$$$        if(teev.gt.1.) alogt=log(teev)
c$$$        if(teev.gt.1.e+05) alogt=11.51
c$$$        expo=cfione(7)
c$$$        do 11 ii=1,6
c$$$          if7=7-ii
c$$$          expo=expo*alogt+cfione(if7)
c$$$ 11     continue
c$$$        sgvxne(i)=exp(expo)*zne(i)
c$$$ 10   continue
c$$$ccc   
c$$$      do 120 ib=1,mb
c$$$        do 121 j=1,3
c$$$          veli=1./vbeam(j,ib)
c$$$          do 122 i=1,mfm1
c$$$            sgxn(i,j,ib)  = sgvxne(i)*veli
c$$$ 122      continue
c$$$ 121    continue
c$$$ 120  continue
c$$$      do 280 ib=1,mb
c$$$        do 281 ion=1,nion
c$$$          do 27 j=1,3
c$$$            if(atw(ion).gt.3.01) go to 25
c$$$            e = 1.e3*ebkev(ib)/(j*atw(ibion))
c$$$c990131            aloge=alog10(e)
c$$$            aloge=log10(e)
c$$$            sigcx=.6937e-14*(1.-.155*aloge)**2/
c$$$     1        (1.+.1112e-14*e**3.3)
c$$$            aloge=aloge*2.302585093-6.907755279
c$$$            if(aloge.le.-2.30258) go to 21
c$$$            expo=cfionp(7)
c$$$            do 22 ii=1,6
c$$$              if7=7-ii
c$$$              expo=expo*aloge+cfionp(if7)
c$$$ 22         continue
c$$$            sigi=exp(expo)
c$$$            go to 23
c$$$ 21         continue
c$$$            sigi=0.
c$$$ 23         continue
c$$$ccc   
c$$$            do 24 i=1,mfm1
c$$$              sgxncx = sigcx*zni(i,ion)
c$$$              sgxni  = sigi*zni(i,ion)
c$$$              sgxn(i,j,ib)=sgxn(i,j,ib)+sgxncx+sgxni
c$$$ 24         continue
c$$$            go to 27
c$$$c------------------------------------------------------------
c$$$c     r.e. olson et al., phys. rev. lett. 41, 163 (1978)
c$$$c------------------------------------------------------------
c$$$ 25         ebi=ebkev(ib)/(j*atw(ibion))
c$$$            do 26 i=1,mfm1
c$$$              rpath=1.0e-17*zni(i,ion)*46.*zzi(i,ion)*
c$$$     .          (32.*zzi(i,ion)/ebi)*(1.-exp(-ebi/(32.*zzi(i,ion))))
c$$$              sgxn(i,j,ib)=sgxn(i,j,ib)+rpath
c$$$ 26         continue
c$$$c------------------------------------------------------------
c$$$ 27       continue
c$$$ 281    continue
c$$$ 280  continue
c$$$ccc   
c$$$      do 38 ib=1,mb
c$$$        if(ib.gt.1 .and. ebkev(ib).eq.ebkev(1)) go to 36
c$$$        do 35 j=1,3
c$$$          sgxnm=sgxn(1,j,ib)
c$$$          do 31 i=2,mfm1
c$$$            if(sgxnm.lt.sgxn(i,j,ib)) sgxnm=sgxn(i,j,ib)
c$$$ 31       continue
c$$$          sgxnmi(j,ib)=1./sgxnm
c$$$ 35     continue
c$$$        go to 38
c$$$ 36     do 37 j=1,3
c$$$ 37     sgxnmi(j,ib) = sgxnmi(j,1)
c$$$ 38   continue
c$$$      return
c$$$ccc   
c$$$c     ONETWO DIVERGENCE
c$$$
c$$$c..................................................................
c$$$c     Stearns formula
c$$$c..................................................................
c$$$
c$$$ 100  continue
c$$$      do 112 ib=1,mb
c$$$        do 111 j=1,3
c$$$          sigeff1=7.e-19*(dzemax+.2)**.65
c$$$          sigeff2=(3.*dnemax/(2.*dtemax+1.))**.118
c$$$          tem=dfloat(j)*atw(ibion)
c$$$          tem1=-(.7+.02*dzemax)
c$$$          sigeff3=(ebkev(ib)/(1000.*tem))**tem1
c$$$          sigeff=sigeff1*sigeff2*sigeff3
c$$$          do 110 i=1,mfm1
c$$$            sgxn(i,j,ib)=sigeff*zne(i)
c$$$ 110      continue
c$$$          sgxnmi(j,ib)=1./sgxn(1,j,ib)
c$$$ 111    continue
c$$$ 112  continue
c$$$      return
c$$$      end
c
c
      subroutine crsecs(iexcit,atw,ebkev,ibion,mb,mfm1,nion,vbeam,zne,
     .                  zni,zte,zzi,dtemax,dnemax,dzemax,
     .                  sgvxne,sgxn,sgxnmi,ibstart) !for iexcit.le.0 only
      !YuP: subr.crsecs might have bugs in sgxn() array usage.
c
c     This subroutine calculates charge exchange, proton ionization, and
c     electron ionization cross sections from the fitted results of
c     freeman and jones, clm-r137, culham (1974).
c
      implicit  integer (i-n), real*8 (a-h, o-z)
c      include 'params.inc'   BH: The file is in Kinsey's cql3d update.
c                                 Not needed except in standalone freya.
      include 'param.h'
c
      integer mb, nion, ibion
      real*8 atw(kion), ebkev(kb), vbeam(ke,kb), zne(kz), zni(kz,kion),
     .    zte(kz), zzi(kz,kion), sgvxne(kz)
      real*8 sgxn(kcmp1,kz,kbe,ksge), sgxnmi(ke,kb)  ! stand alone code
c
      real*8  cfione(7), cfionp(7)
c
      data cfione / -3.173850D+01, 1.143818D+01, -3.833998D0, 
     .    7.046692D-01, -7.431486D-02, 4.153749D-03, -9.486967D-05/
      data cfionp / -4.203309D+01, 3.557321D0, -1.045134D0,
     .    3.139238D-01, -7.454475D-02, 8.459113D-03, -3.495444D-04/
c
c---:----1----:----2----:----3----:----4----:----5----:----6----:----7-c
c
c... initializations
c     
      WRITE(*,*)'crsecs might have a bug... Use iexcit=5' !YuP[2022-12]
      !It could be ok if only one beam is used. See "BUG?" below.

      if (ibstart.eq.1) then
      do ib=1,kb
         do j=1,ke
            sgxnmi(j,ib)=0.D0
            !do i=1,kz
            !   do k=1,kcmp1 !=1:4
            !      sgxn(k,i,j,ib)=0.D0 !YuP[2022-12-15] BUG?
            !      !See above: real*8 sgxn(kcmp1,kz,kbe,ksge)
            !   enddo
            !enddo
         enddo
      enddo
      sgxn=0.d0 !!YuP[2022-12-15] Replaced the above
      endif  !On ibstart.eq.1
      do i=1,kz
       sgvxne(i)=0.D0
      enddo
c
      if (iexcit.eq.-1) go to 200
c
      do 20 i = 1, mfm1
        alogt = 0.
        teev = 1.D3 * zte(i)
        if (teev.gt.1.D0) alogt = dlog(teev)
        if (teev.gt.1.D5) alogt = 11.51D0
        expo = cfione(7)
        do 10 ii = 1, 6
          if7 = 7 - ii
          expo = expo * alogt + cfione(if7)
   10   continue
        sgvxne(i) = dexp(expo) * zne(i)
   20 continue
c
      do 50 ib=ibstart,mb
        do 40 j=1,ke !was 3
          index=ke*(ib-1)+j !YuP[2023-01-24] combined energy&beam_index
          veli= 1.0 / vbeam(j,ib)
          do 30 i=1,mfm1
            !sgxn(4,i,j,ib) = sgvxne(i) * veli  !YuP[2022-12-15] BUG?
            !See above: real*8 sgxn(kcmp1,kz,kbe,ksge)
            !YuP: Need to change j->ke*(ib-1)+j, ib->1
            sgxn(4,i,index,1) = sgvxne(i) * veli !YuP[2023-01-24]
   30     continue
   40   continue
   50 continue
c
      do 110 ib=ibstart,mb
        do 100 ion=1,nion
          do 90 j=1,ke !was 3
            index=ke*(ib-1)+j !YuP[2023-01-24] combined energy&beam_index
            if (atw(ion).le.3.01) then
              e= 1.D3 * ebkev(ib) / (j*atw(ibion))
              aloge= dlog10(e)
              sigcx= .6937D-14 * (1.D0-.155D0*aloge) ** 2 /
     .            (1.D0+.1112D-14*e**3.3D0)
              aloge= aloge * 2.302585093D0 - 6.907755279D0
              if (aloge.gt.-2.30258) then
                expo= cfionp(7)
                do 60 ii=1, 6
                  if7= 7 - ii
                  expo= expo * aloge + cfionp(if7)
   60           continue
                sigi= dexp(expo)
              else
                sigi = 0.D0  !YuP[2018-01-23] corrected sig to sigi
              endif
c
              do 70 i=1,mfm1
                sgxncx= sigcx * zni(i,ion)
                sgxni= sigi * zni(i,ion)
                !sgxn(4,i,j,ib) = sgxn(4,i,j,ib) + sgxncx + sgxni !YuP[2022-12-15] BUG?
                !      !See above: real*8 sgxn(kcmp1,kz,kbe,ksge)
                !YuP: Need to change j->ke*(ib-1)+j, ib->1
                sgxn(4,i,index,1)=sgxn(4,i,index,1)+sgxncx+sgxni !YuP[2023-01-24]
   70         continue
c
            else
c------------------------------------------------------------
c  r.e. olson et al., phys. rev. lett. 41, 163 (1978)
c------------------------------------------------------------
              ebi= ebkev(ib) / (j*atw(ibion))
              do 80 i=1,mfm1
                rpath = 1.0D-17 * zni(i,ion) * 46.D0 * zzi(i,ion) * 
     .             (32.D0*zzi(i,ion)/ebi) * 
     .             (1.D0-dexp(-ebi/(32.D0*zzi(i,ion))))
                !sgxn(4,i,j,ib) = sgxn(4,i,j,ib) + rpath !YuP[2022-12-15] BUG?
                !      !See above: real*8 sgxn(kcmp1,kz,kbe,ksge)
                !YuP: Need to change j->ke*(ib-1)+j, ib->1
                sgxn(4,i,index,1)= sgxn(4,i,index,1) +rpath !YuP[2023-01-24]
   80         continue
            endif
c------------------------------------------------------------
   90     continue
  100   continue
  110 continue
c
c         do ib=ibstart,mb
c           do j=1,3
c             do i=1,mfm1
c       write(*,'(3i3,2x,1p1e10.4,a14)') i,j,ib,sgxn(4,i,j,ib), ' sgxn'
c             enddo
c           enddo
c         enddo
c
      do 150 ib=ibstart,mb
        if (ib.le.1.or.ebkev(ib).ne.ebkev(1)) then
          do 130 j=1,ke !was 3
            index=ke*(ib-1)+j !YuP[2023-01-24] combined energy&beam_index
            !sgxnm = sgxn(4,1,j,ib) !YuP[2022-12-15] BUG?
            !See above: real*8 sgxn(kcmp1,kz,kbe,ksge)
            sgxnm = sgxn(4,1,index,1) !YuP[2023-01-24]
            do 120 i=2,mfm1
              !if (sgxnm.lt.sgxn(4,i,j,ib)) sgxnm = sgxn(4,i,j,ib) !YuP[2022-12-15] BUG?
              !See above: real*8 sgxn(kcmp1,kz,kbe,ksge)
              !YuP: Need to change j->ke*(ib-1)+j, ib->1
              if (sgxnm.lt.sgxn(4,i,index,1)) sgxnm=sgxn(4,i,index,1) !YuP[2023-01-24]
  120       continue
            sgxnmi(j,ib) = 1.D0 / sgxnm
  130     continue
        else
          do 140 j=1,ke !was 3
            sgxnmi(j,ib) = sgxnmi(j,1)
  140     continue
        endif
  150 continue
c
      return
c
c..................................................................
c     Stearns formula
c..................................................................
c
 200  continue
      do 212 ib=ibstart,mb
        do 211 j=1,ke !was 3
          index=ke*(ib-1)+j !YuP[2023-01-24] combined energy&beam_index
          sigeff1=7.D-19*(dzemax+.2)**.65
          sigeff2=(3.*dnemax/(2.*dtemax+1.))**.118
          tem=dfloat(j)*atw(ibion)
          tem1=-(.7+.02*dzemax)
          sigeff3=(ebkev(ib)/(1000.*tem))**tem1
          sigeff=sigeff1*sigeff2*sigeff3
          do 210 i=1,mfm1
            !sgxn(4,i,j,ib)=sigeff*zne(i) !YuP[2022-12-15] BUG?
            !See above: real*8 sgxn(kcmp1,kz,kbe,ksge)
            !YuP: Need to change j->ke*(ib-1)+j, ib->1
            sgxn(4,i,index,1)=sigeff*zne(i) !YuP[2023-01-24]
 210      continue
          !sgxnmi(j,ib)=1./sgxn(4,1,j,ib) !YuP[2022-12-15] BUG?
          !See above: real*8 sgxn(kcmp1,kz,kbe,ksge)
          !YuP: Need to change j->ke*(ib-1)+j, ib->1
          sgxnmi(j,ib)=1./sgxn(4,1,index,1) !YuP[2023-01-24]
 211    continue
 212  continue
c
c         do ib=ibstart,mb
c           do j=1,3
c             do i=1,mfm1
c       write(*,'(3i3,2x,1p1e10.4,a14)') i,j,ib,sgxn(4,i,j,ib), ' sgxn2'
c             enddo
c           enddo
c         enddo
c
      return
      end subroutine crsecs
c
c
      real*8 function cxr (x)
      implicit integer (i-n), real*8 (a-h,o-z)
c
c ----------------------------------------------------------------------
c this function calculates the charge exchange rate for hydrogen atoms
c interacting with protons in units of cm**3/s.
c x is in units of keV for 1.0D-3 .le. x .le. 100, the
c the formula is taken from the paper by r.l. freeman and e.m. jones
c clm-r 137 culham laboratory 1974.
c for x .lt. 1.0D-3, a rate coefficient derived from an analytic average
c over the approximate cross section  sigma = 0.6937D-14*(1.0-0.155*LOG10
c (e/1ev))**2 is used.  this cross section is an approximation to that
c given by riviere, nuclear fusion 11,363(1971).
c ----------------------------------------------------------------------
c
      if (x .lt.   1.0D-3)  go to 10
      if (x .gt. 100     )  go to 20
      tc  = LOG (x)+6.9077553
      dum = 0.2530205D-4-tc*0.8230751D-6
      tc  = -0.1841757e2+tc*(0.528295-tc*(0.2200477-tc*(0.9750192D-1-tc*
     .      (0.1749183D-1-tc*(0.4954298D-3+tc*(0.2174910D-3-tc*dum))))))
      tc  = EXP (tc)
      cxr = tc
      return
c
   10 tc  = 0.50654 - 6.7316D-2 * LOG  (x)
      cxr =           3.3340D-7 * SQRT (x) * (tc*tc + 7.454D-3)
      return
c
   20 cxr = 0.0
      return
c
      end function cxr
c
c
      BLOCK DATA d01bbf_data
c
c     Took the following data statements out of subroutine
c     d01bbf, since the Lahey compiler objected to their
c     presence in d01bbf.
c     Their presense in d01bbf is not ANSI F77.
c     BobH, 010618.
c
      implicit integer (i-n), real*8 (a-h,o-z)
c
      common / cgh / wgh(70), xgh(70)
c
      equivalence (wgh10, wgh(1)), (xgh10, xgh(1))
      equivalence (wgh16, wgh(11)), (xgh16, xgh(11))
      equivalence (wgh20, wgh(27)), (xgh20, xgh(27))
      equivalence (wgh24, wgh(47)), (xgh24, xgh(47))
c
      dimension wgh10(10), xgh10(10)
      dimension wgh16(16), xgh16(16)
      dimension wgh20(20), xgh20(20)
      dimension wgh24(24), xgh24(24)
c
      common / cgl / wgl(70), xgl(70)
c
      equivalence (wgl10, wgl(1)), (xgl10, xgl(1))
      equivalence (wgl16, wgl(11)), (xgl16, xgl(11))
      equivalence (wgl20, wgl(27)), (xgl20, xgl(27))
      equivalence (wgl24, wgl(47)), (xgl24, xgl(47))

      dimension wgl10(10), xgl10(10)
      dimension wgl16(16), xgl16(16)
      dimension wgl20(20), xgl20(20)
      dimension wgl24(24), xgl24(24)

c
      data wgh10   /
     >  7.6404329D-06,  1.3436457D-03,  3.3874394D-02,  2.4013861D-01,
     >  6.1086263D-01,  6.1086263D-01,  2.4013861D-01,  3.3874394D-02,
     >  1.3436457D-03,  7.6404329D-06/


      data xgh10   /
     >  3.4361591D+00,  2.5327317D+00,  1.7566836D+00,  1.0366108D+00,
     >  3.4290133D-01, -3.4290133D-01, -1.0366108D+00, -1.7566836D+00,
     >  -2.5327317D+00, -3.4361591D+00/


      data wgh16   /
     >  2.6548075D-10,  2.3209808D-07,  2.7118601D-05,  9.3228401D-04,
     >  1.2880312D-02,  8.3810041D-02,  2.8064746D-01,  5.0792948D-01,
     >  5.0792948D-01,  2.8064746D-01,  8.3810041D-02,  1.2880312D-02,
     >  9.3228401D-04,  2.7118601D-05,  2.3209808D-07,  2.6548075D-10/


      data xgh16   /
     >  4.6887389D+00,  3.8694479D+00,  3.1769992D+00,  2.5462022D+00,
     >  1.9517880D+00,  1.3802585D+00,  8.2295145D-01,  2.7348105D-01,
     >  -2.7348105D-01, -8.2295145D-01, -1.3802585D+00, -1.9517880D+00,
     >  -2.5462022D+00, -3.1769992D+00, -3.8694479D+00, -4.6887389D+00/


      data wgh20   /
     >  2.2293936D-13,  4.3993410D-10,  1.0860694D-07,  7.8025565D-06,
     >  2.2833864D-04,  3.2437733D-03,  2.4810521D-02,  1.0901721D-01,
     >  2.8667551D-01,  4.6224367D-01,  4.6224367D-01,  2.8667551D-01,
     >  1.0901721D-01,  2.4810521D-02,  3.2437733D-03,  2.2833864D-04,
     >  7.8025565D-06,  1.0860694D-07,  4.3993410D-10,  2.2293936D-13/


      data xgh20   /
     >  5.3874809D+00,  4.6036824D+00,  3.9447640D+00,  3.3478546D+00,
     >  2.7888061D+00,  2.2549740D+00,  1.7385377D+00,  1.2340762D+00,
     >  7.3747373D-01,  2.4534071D-01, -2.4534071D-01, -7.3747373D-01,
     >  -1.2340762D+00, -1.7385377D+00, -2.2549740D+00, -2.7888061D+00,
     >  -3.3478546D+00, -3.9447640D+00, -4.6036824D+00, -5.3874809D+00/


      data wgh24   /
     >  1.6643685D-16,  6.5846202D-13,  3.0462543D-10,  4.0189712D-08,
     >  2.1582457D-06,  5.6886916D-05,  8.2369248D-04,  7.0483558D-03,
     >  3.7445471D-02,  1.2773962D-01,  2.8617954D-01,  4.2693116D-01,
     >  4.2693116D-01,  2.8617954D-01,  1.2773962D-01,  3.7445471D-02,
     >  7.0483558D-03,  8.2369248D-04,  5.6886916D-05,  2.1582457D-06,
     >  4.0189712D-08,  3.0462543D-10,  6.5846202D-13,  1.6643685D-16/


      data xgh24   /
     >  6.0159256D+00,  5.2593829D+00,  4.6256628D+00,  4.0536644D+00,
     >  3.5200068D+00,  3.0125461D+00,  2.5238810D+00,  2.0490036D+00,
     >  1.5842500D+00,  1.1267608D+00,  6.7417111D-01,  2.2441455D-01,
     >  -2.2441455D-01, -6.7417111D-01, -1.1267608D+00, -1.5842500D+00,
     >  -2.0490036D+00, -2.5238810D+00, -3.0125461D+00, -3.5200068D+00,
     >  -4.0536644D+00, -4.6256628D+00, -5.2593829D+00, -6.0159256D+00/


      data wgl10   /
     >  3.0844112D-01,  4.0111993D-01,  2.1806829D-01,  6.2087456D-02,
     >  9.5015170D-03,  7.5300839D-04,  2.8259233D-05,  4.2493140D-07,
     >  1.8395648D-09,  9.9118272D-13/


      data xgl10   /
     >  1.3779347D-01,  7.2945455D-01,  1.8083429D+00,  3.4014337D+00,
     >  5.5524961D+00,  8.3301527D+00,  1.1843786D+01,  1.6279258D+01,
     >  2.1996586D+01,  2.9920697D+01/




      data wgl16   /
     >  2.0615171D-01,  3.3105785D-01,  2.6579578D-01,  1.3629693D-01,
     >  4.7328929D-02,  1.1299900D-02,  1.8490709D-03,  2.0427192D-04,
     >  1.4844587D-05,  6.8283193D-07,  1.8810248D-08,  2.8623502D-10,
     >  2.1270790D-12,  6.2979670D-15,  5.0504737D-18,  4.1614624D-22/


      data xgl16   /
     >  8.7649410D-02,  4.6269633D-01,  1.1410578D+00,  2.1292836D+00,
     >  3.4370866D+00,  5.0780186D+00,  7.0703385D+00,  9.4383143D+00,
     >  1.2214223D+01,  1.5441527D+01,  1.9180157D+01,  2.3515906D+01,
     >  2.8578730D+01,  3.4583399D+01,  4.1940453D+01,  5.1701160D+01/


      data wgl20   /
     >  1.6874680D-01,  2.9125436D-01,  2.6668610D-01,  1.6600245D-01,
     >  7.4826065D-02,  2.4964417D-02,  6.2025508D-03,  1.1449624D-03,
     >  1.5574177D-04,  1.5401441D-05,  1.0864864D-06,  5.3301209D-08,
     >  1.7579812D-09,  3.7255024D-11,  4.7675293D-13,  3.3728442D-15,
     >  1.1550143D-17,  1.5395221D-20,  5.2864427D-24,  1.6564566D-28/


      data xgl20   /
     >  7.0539890D-02,  3.7212682D-01,  9.1658210D-01,  1.7073065D+00,
     >  2.7491993D+00,  4.0489253D+00,  5.6151750D+00,  7.4590175D+00,
     >  9.5943929D+00,  1.2038803D+01,  1.4814293D+01,  1.7948896D+01,
     >  2.1478788D+01,  2.5451703D+01,  2.9932555D+01,  3.5013434D+01,
     >  4.0833057D+01,  4.7619994D+01,  5.5810796D+01,  6.6524417D+01/


      data wgl24   /
     >  1.4281197D-01,  2.5877411D-01,  2.5880671D-01,  1.8332269D-01,
     >  9.8166273D-02,  4.0732478D-02,  1.3226019D-02,  3.3693491D-03,
     >  6.7216256D-04,  1.0446121D-04,  1.2544722D-05,  1.1513158D-06,
     >  7.9608130D-08,  4.0728590D-09,  1.5070082D-10,  3.9177365D-12,
     >  6.8941811D-14,  7.8198004D-16,  5.3501888D-18,  2.0105175D-20,
     >  3.6057659D-23,  2.4518188D-26,  4.0883016D-30,  5.5753458D-35/


      data xgl24   /
     >  5.9019852D-02,  3.1123915e-01,  7.6609691e-01,  1.4255976D+00,
     >  2.2925621D+00,  3.3707743D+00,  4.6650837D+00,  6.1815351D+00,
     >  7.9275392D+00,  9.9120980D+00,  1.2146103D+01,  1.4642732D+01,
     >  1.7417993D+01,  2.0491460D+01,  2.3887330D+01,  2.7635937D+01,
     >  3.1776041D+01,  3.6358406D+01,  4.1451720D+01,  4.7153106D+01,
     >  5.3608575D+01,  6.1058531D+01,  6.9962240D+01,  8.1498279D+01/


      end
c
c
CMG  changed 11/13/2017    subroutine d01bbf(type,inum,weight,abscis,ier)
      subroutine d01bbf_cql3d(type,inum,weight,abscis,ier)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
      character*2 type
      dimension weight(inum), abscis(inum)
      common/b2/nouthx,istart,ihxbug
      dimension num(4)
      dimension iorg(4)
c
      common / cgh / wgh(70), xgh(70)
c
      equivalence (wgh10, wgh(1)), (xgh10, xgh(1))
      equivalence (wgh16, wgh(11)), (xgh16, xgh(11))
      equivalence (wgh20, wgh(27)), (xgh20, xgh(27))
      equivalence (wgh24, wgh(47)), (xgh24, xgh(47))
c
      dimension wgh10(10), xgh10(10)
      dimension wgh16(16), xgh16(16)
      dimension wgh20(20), xgh20(20)
      dimension wgh24(24), xgh24(24)
c
      common / cgl / wgl(70), xgl(70)
c
      equivalence (wgl10, wgl(1)), (xgl10, xgl(1))
      equivalence (wgl16, wgl(11)), (xgl16, xgl(11))
      equivalence (wgl20, wgl(27)), (xgl20, xgl(27))
      equivalence (wgl24, wgl(47)), (xgl24, xgl(47))

      dimension wgl10(10), xgl10(10)
      dimension wgl16(16), xgl16(16)
      dimension wgl20(20), xgl20(20)
      dimension wgl24(24), xgl24(24)
c
      data num /10,16,20,24/
      data iorg /1,11,27,47/

      do 50 i = 1, 13
        index = i
        if(inum.eq.num(i)) go to 60
 50   continue
      ier = 1
      return

 60   continue
      ier = 0
c
      if (type.ne.'gl') go to 180
      do 100 i = 1, inum
        irel = iorg(index) - 1 + i
        weight(i) = wgl(irel)
        abscis(i) = xgl(irel)
 100  continue
      go to 1000
c
 180  continue
      if(type.ne.'gh') go to 300
      do 200 i = 1, inum
        irel = iorg(index) - 1 + i
        weight(i) = wgh(irel)
        abscis(i) = xgh(irel)
 200  continue
      go to 1000
c
 300  continue
      ier = 1
      return
c
 1000 continue
c
      return
      end
c
c
      subroutine decomp(ndim,n,a,cond,ipvt,work)
      implicit integer (i-n), real*8 (a-h,o-z)
      integer ndim,n
      real*8 a(ndim,n),cond,work(n)
      real*8 zero
      integer ipvt(n)
c
c     decomposes a real matrix by gaussian elimination
c     and estimates the condition of the matrix.
c
c     use solveq to compute solutions to linear systems.
c
c     input..
c
c     ndim = declared row dimension of the array containing  a.
c
c     n = order of the matrix.
c
c     a = matrix to be triangularized.
c
c     output..
c
c     a  contains an upper triangular matrix  u  and a permuted
c     version of a lower triangular matrix  i-l  so that
c     (permutation matrix)*a = l*u .
c
c     cond = an estimate of the condition of  a .
c     for the linear system  a*x = b, changes in  a  and  b
c     may cause changes  cond  times as large in  x .
c     if  cond+1.0 .eq. cond , a is singular to working
c     precision.  cond  is set to  1.0D+32  if exact
c     singularity is detected.
c
c     ipvt = the pivot vector.
c     ipvt(k) = the index of the k-th pivot row
c     ipvt(n) = (-1)**(number of interchanges)
c
c     work space..  the vector  work  must be declared and included
c     in the call.  its input contents are ignored.
c     its output contents are usually unimportant.
c
c     the determinant of a can be obtained on output by
c     det(a) = ipvt(n) * a(1,1) * a(2,2) * ... * a(n,n).
c
      real*8 ek, t, anorm, ynorm, znorm
      integer nm1, i, j, k, kp1, kb, km1, m
c
      zero=0.d0

      ipvt(n) = 1
      if (n .eq. 1) go to 80
      nm1 = n - 1
c
c     compute 1-norm of a
c
      anorm = 0.0
      do 10 j = 1, n
        t = 0.0
        do 5 i = 1, n
          t = t + abs(a(i,j))
 5      continue
        if (t .gt. anorm) anorm = t
 10   continue
c
c     gaussian elimination with partial pivoting
c
      do 35 k = 1,nm1
        kp1= k+1
c
c     find pivot
c
        m = k
        do 15 i = kp1,n
          if (abs(a(i,k)) .gt. abs(a(m,k))) m = i
 15     continue
        ipvt(k) = m
        if (m .ne. k) ipvt(n) = -ipvt(n)
        t = a(m,k)
        a(m,k) = a(k,k)
        a(k,k) = t
c
c     skip step if pivot is zero
c
        if (t .eq. 0.0) go to 35
c
c     compute multipliers
c
        do 20 i = kp1,n
          a(i,k) = -a(i,k)/t
 20     continue
c
c     interchange and eliminate by columns
c
        do 30 j = kp1,n
          t = a(m,j)
          a(m,j) = a(k,j)
          a(k,j) = t
          if (t .eq. 0.0) go to 30
          do 25 i = kp1,n
            a(i,j) = a(i,j) + a(i,k)*t
 25       continue
 30     continue
 35   continue
c
c     cond = (1-norm of a)*(an estimate of 1-norm of a-inverse)
c     estimate obtained by one step of inverse iteration for the
c     small singular vector.  this involves solving two systems
c     of equations, (a-transpose)*y = e  and  a*z = y  where  e
c     is a vector of +1 or -1 chosen to cause growth in y.
c     estimate = (1-norm of z)/(1-norm of y)
c
c     solve (a-transpose)*y = e
c
      do 50 k = 1, n
        t = 0.0
        if (k .eq. 1) go to 45
        km1 = k-1
        do 40 i = 1, km1
          t = t + a(i,k)*work(i)
 40     continue
 45     ek = 1.0
        if (t .lt. 0.0) ek = -1.0
        if (a(k,k) .eq. 0.0) go to 90
        work(k) = -(ek + t)/a(k,k)
 50   continue
      do 60 kb = 1, nm1
        k = n - kb
        t = 0.0
        kp1 = k+1
        do 55 i = kp1, n
          t = t + a(i,k)*work(k)
 55     continue
        work(k) = t
        m = ipvt(k)
        if (m .eq. k) go to 60
        t = work(m)
        work(m) = work(k)
        work(k) = t
 60   continue
c
      ynorm = 0.0
      do 65 i = 1, n
        ynorm = ynorm + abs(work(i))
 65   continue
c
c     solve a*z = y
c
      call solveq(ndim, n, a, work, ipvt)
c
      znorm = zero
      do 70 i = 1, n
        znorm = znorm + abs(work(i))
 70   continue
c
c     estimate condition
c
      cond = anorm*znorm/ynorm
      if (cond .lt. 1.0) cond = 1.0
      return
c
c     1-by-1
c
 80   cond = 1.0
      if (a(1,1) .ne. 0.0) return
c
c     exact singularity
c
 90   cond = 1.0D+32
      return
      end
c
c
      real*8 function dfhx(beta)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
c--   approximation to the function defined by janev & presnyakov,
c     j. phys. b 13, 4233 (1980).
c
      dimension dd(38)
      data dd/
     *  .10450,.121, .138, .157, .175, .200, .229, .260, .300, .339,
     *  .367, .389, .402, .410, .398, .376, .346, .317, .285, .255,
     *  .227, .205, .185, .168, .152, .138, .124, .110, .099, .089,
     *  .079, .070, .062, .054, .047, .041, .035, .02898/
c
      beta1=1./beta
      if(beta1 .le. .2)goto 110
      if(beta1 .ge. 1.e3)goto 120
c990131      a=10.*alog10(beta1)+8.
      a=10.*log10(beta1)+8.
      ia=min0(37,int(a))
      dfhx=dd(ia)+(a-dfloat(ia))*(dd(ia+1)-dd(ia))
      return
 110  dfhx=.5*beta*(1.-1./(8.*beta*sqrt(beta)))*exp(-sqrt(2.*beta))
      return
c990131 120  dfhx=4.*beta*alog(1.4*beta1)
 120  dfhx=4.*beta*log(1.4*beta1)
      return
      end
c
c
      subroutine eigen(xeig)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
      parameter (ms=21,mc=35,mz=1,mi=2 )
      common/cpub/ cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7,cpu8,
     >  cpuii, cpuiz, cpuplf
      common/b0/ns,nc,er0,v0,te,ti,numi,ami(mi),deni(mi),
     *  numz,iz(mz),amz(mz),denz(mz),zcor(mz),zsqcor(mz),
     *  izstrp(mz),dene
      common/b2/nouthx,istart,ihxbug
      common/b7/q(ms+1,ms+1)
      common/b9/eigvr(ms+1,ms+1),eigvl(ms+1),cj(ms+1)
      common/b10/xfrac
      dimension qq(ms+1,ms+1),eigr(ms+1),eigi(ms+1),intger(ms+1)
      dimension evr(ms+1),evi(ms+1),efr(ms+1,ms+1),efi(ms+1,ms+1)
      dimension xn(ms+1),yn(ms+1),work2(3*(ms+1)),coef(2)
      dimension w(2*(ms+1)),z(2*(ms+1),ms+1)
      dimension cjmat(ms+1,ms+1)
      dimension workdc(ms+1),ipvt(ms+1)
      real*8 infac(ms+1,ms+1), lexp(ms+1), in(ms+1), dindx(ms+1)
      real*8 zero
      data xtest/1./
c
c     on return from eigrf, w contains the complex*16 eigenvalues
c     as w= [wr(1),wi(1), ... ,wr(ns),wi(ns)]
c     on return from eigrf, z contains the complex*16 eigenvectors
c     as z(col1)= [zr(1),zi(1), ... ,zr(ns),zi(ns)], etc.,
c     each column representing an eigenvector
c
cBH080118      call second(cpua)
c

      zero=0.d0

      do 10 i=1,ns
        do 101 j=1,ns
          qq(i,j)=q(i,j)
 101    continue
 10   continue
      if(nouthx.gt.0) then
        write(nouthx,11)
 11     format( 'q _ matrix')
        do 13 i=1,ns
          write(nouthx,12) (q(i,j), j=1,ns)
 12       format(1x,1p7e11.3)
 13     continue
      endif


c@    ifail1=0
c@    call f02aff(qq,ms+1,ns,eigr,eigi,intger,ifail1)
c@    if(ifail1.eq.0) go to 15
      ijob = 1
      call eigrf(qq,ns,ms+1,ijob,w,z,ms+1,work2,ifail1)
      if(ifail1.eq.0)goto 15
      ihxbug = 9
      if(nouthx.gt.0) write(nouthx,3939) ifail1
 3939 format(' ??? eigrf routine error: ifail1= ',i4)
      return
c
 15   continue
      do 18 j=1,ns
        do 17 i=1,ns
          eigvr(i,j)=z(2*i-1,j)
 17     continue
        eigvl(j) = w(j*2-1)
 18   continue
      eigmin=1.e30
      do 20 i=1,2*ns,2
c990131        eigmin=amin1(eigmin,abs(w(i)))
        eigmin=min(eigmin,abs(w(i)))
 20   continue
c@    do 20 i=1,ns
c@    eigmin=amin1(eigmin,abs(eigr(i)))
c@    20 continue
      xeig=0.
      if(eigmin.ne.zero)xeig=v0/eigmin
c
c     determine fraction of 3rd excited state (approximate)
      sum=0
      do 25 i=1,ns
        sum=sum+eigvr(i,1)
 25   continue
      xfrac=eigvr(4,1)/sum
      if(nouthx.eq.0) return

      write(nouthx,1000) xeig
 1000 format(' length (eigenvalue):  ',1pe11.4)
      write(nouthx,1010) (i,eigvl(i),i=1,ns)
 1010 format('    i= ',i2,'         eigenvalue= ',1pe11.4)
      do 1030 j = 1, ns
        write(nouthx,1020) j,(eigvr(i,j),i=1,ns)
 1020   format(' eigenvector[',i2,']= ',/,(10x,1p5e11.4))
 1030 continue

c...  apply initial condition constraint
      do 1200 i = 1, ns
        do 1201 j = 1, ns
          cjmat(i,j) = eigvr(i,j)
 1201   continue
 1200 continue
      cj(1) = 1.0
      do 1300 i = 2, ns
        cj(i) = 0.
 1300 continue
      call decomp (ms+1,ns,cjmat,cond,ipvt,workdc)
      call solveq(ms+1,ns,cjmat,cj,ipvt)

      write(nouthx,1320) (cj(i), i=1,ns)
 1320 format(' bc constants: ',/,(10x,1p5e11.4))

      do 1340 j = 1, ns
        do 1350 i = 1, ns
          lexp(i) = eigvl(i) / v0
          infac(j,i) = cj(i) * eigvr(j,i)
 1350   continue
        write (nouthx,1400) j, (infac(j,i), lexp(i), i = 1, ns)
 1400   format( ' *** i(',i2,')=   ',/,
     >    (12x, 1pe15.7,' * exp(',1pe15.7,' * x)') )
 1340 continue
c
c%OS  
      print *,' WARNING: xtest not defined in EIGEN'
c%OS  
      do 1330 j = 1,ns
        in(j) = 0.
        do 1345 i = 1, ns
          in(j) = in(j) + infac(j,i) * exp(lexp(i)*xtest)
 1345   continue
 1330 continue

      write(nouthx,1450)
 1450 format(///' qnj * ij / v0 ...........',/)
      do 1360 j = 1, ns
        sum = 0.
        do 1370 i = 1, ns
          sum = sum + q(j,i) * in(i)
 1370   continue
        write(nouthx,1500) j, sum/v0
 1500   format(15x,i3,5x,1pe15.7)
 1360 continue

cBH080118      call second(cpub)
      cpu7 = cpu7 + cpub - cpua
c
      return
      end
c
c
      real*8 function encapf(vcvo, tstcx)
      implicit integer (i-n), real*8 (a-h,o-z)
c
c     This routine evaluates the function N, which is related to the
c      rate at which fast ions slow down on electrons and thermal ions.
c       This function is defined by Callen et al., IAEA Tokyo, 
c       Vol. I, 645(1974).
c
      v3 = vcvo**3
c990131      tfts = alog(1.+1./v3)/3.
      tfts = log(1.+1./v3)/3.
      if(tstcx.gt.0.01) go to 10
      encapf = tfts
      return
 10   encapf = (1.-exp(-tfts*tstcx))/tstcx
      return
      end
c
c
      subroutine freyorb(atw, codeid, drutpi, drini, droti,
     *  ic, iskip, izi, mfm1, norb, pinsid, potsid, pzone,
     *  rinsid, rotsid, rzone, ri, zetai, v, zi,
     *  ipass, iaxis, ier, izp, wid, wt, zeta,angmtp)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c
c     This subroutine calculates the following orbit parameters for a
c     fast ion:
c     ipass : 1, particle is passing
c     0, particle is trapped
c     iaxis : 1, particle encircles the axis
c     0, particle does not encircle the axis
c     ier   : 1, routine had difficulty determining orbit
c     0, routine obtained reasonable orbit
c     izp   : outermost zone of orbit
c     wid   : weight giving approximate fraction of time spent
c     by fast ion in each zone
c     zeta  : approximate pitch angle cosine in each zone along
c     orbit
c     The fast ion is assumed to be initially traveling in the
c     co-current direction.
c
      dimension pinsid(*), potsid(*), rinsid(*), rotsid(*)
      dimension wt(*), zeta(*), angmtp(*)
      character*8 codeid,flagm,flagp
      data eoverc / 1.6e-20 /
      data xmassp / 1.673e-24 /
c
      zero=0.d0
      one=1.d0
      onem5=1.d-5
      onep5=1.d5
      onem3=1.d-3

c     initialize parameters
c
      flagp = ' '
      flagm = ' '
      ier   = 0
      itm   = 0
      izm   = izi
      psim  = 0.
      rm    = 0.
      wid   = 0.
      do 2 i=1,mfm1
        wt(i) = 0.
        angmtp(i) = 0.
c     angmtp will be used for orbit smearing aof angular momentum
c     this option will be implemented later
    2 zeta(i) = 0.
      wt(izi) = 1.
      zeta(izi) = zetai
c
c     calculate ion mass times c/e; initialize rmajor and psiax
c
      xmass  = atw*xmassp/eoverc
      rmajor = rotsid(1)
      psiax  = potsid(1)
c
c     calculate poloidal magnetic flux at the fast ion birth point
c
      psii = pzone
      if(codeid.ne.'onedee') go to 5
      call inter(droti,0,mfm1,zero,potsid,rzone,psii)
    5 continue
c
c     determine whether fast ion is passing or trapped
c
      pang = xmass*v*zetai*ri - psii
      ipass = 1
      if(abs(zetai).eq.1.) go to 40
      if(zetai.eq.zero) go to 30
      rt = (1.-zetai**2)*ri
      if(rt.gt.rmajor) go to 10
      call inter(drini,0,mfm1,rmajor,pinsid,rt,psit)
      go to 20
 10   call inter(droti,0,mfm1,rmajor,potsid,rt,psit)
 20   pangt = -psit
      if(pang.gt.pangt) go to 40
 30   ipass = 0
 40   continue
c
c     determine whether ion encircles the magnetic axis
c
      iaxis = 1
      zetax2 = 1. - (1.-zetai**2)*ri/rmajor
      if(zetax2.le.0.) go to 50
      zetax = sqrt(zetax2)
      if(ipass.eq.0) zetax = -zetax
      pangx = xmass*v*zetax*rmajor - psiax
      if(ipass.ne.0 .and. pang.lt.pangx) go to 60
      if(ipass.eq.0 .and. pang.gt.pangx) go to 60
 50   iaxis = 0
 60   continue
c
c     calculate poloidal magnetic flux and zone index at outer major
c     radius of orbit
c
      c1 = (1.-zetai**2)*ri
      c2 = 1./(xmass*v)
      c3 = zetai*ri - c2*psii
      if(codeid.ne.'onedee') go to 110
      ra = rmajor + rzone
      go to 115
 110  call inter(drutpi,1,mfm1,psiax,rotsid,psii,ra)
 115  ra = ra + 2.
      
      call scalit(ra, rp, zero, onep5, 1, 1, itp, 10,onem5,onem3,one,0,
     *  c1, c2, c3, droti, mfm1, psip, potsid, rmajor)
      if(rp.lt.ri-0.01 .or. rp.lt.rmajor .or. itp.lt.0) flagp = '*'
      if(flagp.eq.'*') go to 340
      if(codeid.ne.'onedee') go to 150
      xzp = (rp-rmajor)*droti + 1.
      go to 160
 150  xzp = sqrt(psip-psiax)*drutpi + 1.
 160  izp = xzp
      izp = max0(izp,izi)
      izp = min0(izp,mfm1+1)
      zetap = (pang+psip)/(xmass*v*rp)
c
c     determine whether ion is confined or lost; return if it is lost
c
      if(izp.gt.mfm1) go to 350
c
c     calculate poloidal magnetic flux and zone index at inner major
c     radius of orbit
c
      if(iaxis.eq.0) go to 220
      if(codeid.ne.'onedee') go to 210
      ra = rmajor - rzone
      go to 215
 210  call inter(drutpi,1,mfm1,psiax,rinsid,psii,ra)
 215  call scalit(ra, rm, zero, onep5, 1, 1, itm,10,onem5,-onem3,one,0,
     *  c1, c2, c3, drini, mfm1, psim, pinsid, rmajor)
      go to 240
 220  ra = rmajor + 0.25*(rp-rmajor)
      call scalit(ra, rm, zero, onep5, 1, 1, itm,10,onem5,onem3,one,0,
     *  c1, c2, c3, droti, mfm1, psim, potsid, rmajor)
 240  continue
      if(ipass.eq.0 .and. rm.gt.ri+0.01) flagm = '*'
      if(iaxis.ne.0 .and. rm.gt.rmajor) flagm = '*'
      if(itm.lt.0) flagm = '*'
      if(rm.gt.rp-0.01) flagm = '*'
      if(flagm.eq.'*') go to 340
      if(codeid.ne.'onedee') go to 250
      xzm = abs(rmajor-rm)*droti + 1.
      go to 260
 250  xzm = sqrt(psim-psiax)*drutpi + 1.
 260  izm = xzm
      izm = min0(izm,izi)
      zetam = (pang+psim)/(xmass*v*rm)
c
c     calculate orbit width
c
      wid = (rp-rmajor) - abs(rmajor-rm)
c990131      wid = amax1(wid,0.)
      wid = max(wid,zero)
c
c     calculate weights by zone; assume ion spends equal time in each
c     zone it fully traverses and proportionately less time in the
c     innermost and outermost zones of the orbit
c
      if(izp.eq.izm) go to 350
      wtm = 1. - (xzm - izm)
      wtp = xzp - izp
      wtx = 1./(wtm+wtp+izp-izm-1.)
      wt(izm) = wtm*wtx
      wt(izp) = wtp*wtx
      if(izp.eq.izm+1) go to 320
      do 310 i=izm+1,izp-1
 310  wt(i) = wtx
 320  continue
c
c     calculate pitch angle cosines by zone; assume cosine varies
c     linearly with zone index
c
      zdif = zetap - zetam
      zeta(izm) = zetam + 0.5*wt(izm)*zdif
      zeta(izp) = zetap - 0.5*wt(izp)*zdif
      if(izp.eq.izm+1) go to 350
      do 330 i=izm+1,izp-1
 330  zeta(i) = zeta(i-1) + wt(i)*zdif
      go to 350
c
c     set error flag if error was detected
c
 340  izp = izi
      ier = 1
c
c     print out selected results
c
 350  continue
      if(norb.eq.0) return
      nout = iabs(norb)
      if(ic.eq.1) write(nout,1010)
      if(mod(ic-1,iskip).ne.0) return
      write(nout,1020) ic, ipass, iaxis, zetai, ri, zi, psii, rp, flagp,
     *  psip, rm, flagm, psim, itp, itm, izi, izm, izp
      if(norb.lt.0) return
      if(izp.gt.mfm1) return
      write(nout,1030) (wt(i), i=1,mfm1)
      write(nout,1040) (zeta(i), i=1,mfm1)
      return
c
 1010 format(/' orbit parameters'//
     *  4x,'ic ip ix',7x,'zetai',10x,'ri',10x,'zi',8x,'psii',
     *  10x,'rp',8x,'psip',10x,'rm',8x,'psim',
     *  ' itp itm izi izm izp')
 1020 format(i6,2i3,f12.4,2f12.2,e12.3,2(f12.2,a1,e11.3),5i4)
 1030 format(' wt',f9.4,9f12.4/(10f12.4))
 1040 format(' zeta',f7.4,9f12.4/(10f12.4))
      end
c
c
      real*8 function gef(vpar, tpar)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c
c     This routine evaluates the function Ge, which is related to the trans-
c     fer of energy from fast ions slowing down on electrons.  The func-
c     tion is defined by Callen et al., IAEA Tokyo, Vol. I, 645 (1974).
c
      external gefun
      common /gecom/ vcvo, tstcx, emzrat
c     root13 = sqrt(1./3.)
c     fact4  = root13*atan(root13)
      data root13 /0.577350/, fact4 /0.302300/
      if(tpar.gt.0.01) go to 10
      rooty = 1./vpar
      y = rooty**2
c990131      term2 = alog((1.-rooty+y)/(1.+rooty)**2)/(6.*y)
      term2 = log((1.-rooty+y)/(1.+rooty)**2)/(6.*y)
      term3 = root13*atan(root13*(2.*rooty-1.))/y
      term4 = fact4/y
      gef = 2.*(0.5-term2-term3-term4)
      return
 10   vcvo = vpar
      tstcx = tpar
      a = 0.
      b = 1.
      ep = 0.1
      m4 = 3
      gef = asimp(a,b,ep,m,m4,gefun)
      return
      end

      real*8 function gefun(y)
      implicit integer (i-n), real*8 (a-h,o-z)
      common /gecom/ vcvo, tstcx, emzrat
      v3 = vcvo**3
      arg = (1.+v3)/(y**3+v3)
      pcx = (arg)**(-tstcx/3.)
      gefun = 2.*y**4*pcx/(y**3+v3)
      return
      end
c
c
      BLOCK DATA hexnb_data
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c     Data statement of common block variables moved outside
c     subroutine hexnb, and not ANSI F77 (and Lahey compiler
c     complains).  BobH, 010618.
      common/cpub/ cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7,cpu8,
     >  cpuii, cpuiz, cpuplf

      data cpu1/0./, cpu2/0./, cpu3/0./, cpu4/0./, cpu5/0./,
     >  cpu6/0./, cpu7/0./, cpu8/0./

      end
c
c
      subroutine hexnb(istarx,iexcix,ilorenx,mstatx,ncontx,
     >  er0x,tex,tix,numix,amix,denix,
     >  numzx,izx,amzx,izstrx,denzx,bperpx,
     >  ldene,ldeni,ldenz,lsvi,lsvz,lsve,lrad,
     >  lngh,lngl,louthx,lcor,
     >  nsigmav,lambda,hexfrac,ihxerr)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
c     version 21
c--   author:
c     c. d. boley
c     pppl 1984
c--   modified by : (vax / modular)
c     r. m. wieland
c     pppl july,1985
c--   ref.:
c     c. d. boley, r. k. janev, and d. e. post, phys. rev. letters
c     52, 534 (1984).
c--   calling sequence
c     call hexnb(istart,iexcit,ilorent,mstate,ncont,
c     >                   er0,tex,tix,numi,ami,deni,
c     >                   numz,iz,amz,izstrp,denz,bperp,
c     >                   kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,
c     >                   ngh,ngl,nouthx,ncorin,
c     >                   nsigmav,lambda)
c--   input parameters:
c     istart = 1 to initialize rad rates & fine pts       0 otherwise
c     iexcit if =0 then ignore contribution of excited states
c     if =1 then include contributionof excited states.
c     ilorent =0 or 1       whether or not to calculate the maximum
c     principal quantum number ns of the populated
c     states as given by the lorentz ionization
c     limit (1 ==> yes)
c     mstate< parameter ms
c     for ilorent=0, use ns = mstate+1
c     note: the operationalsignificance of ns in the code is to
c     set an upper limit such that any excitations to levels n
c     higher than ns are counted as "ionizations".
c     ncont< parameter mc
c     upper bound to number of continuum states
c     er0    beam energy per amu (ev)
c     te     electron temperature (ev)
c     ti     ion temperature (ev)
c     numi   number of hydrogenic ion species
c     ami    masses of the hydrogenic ions (amu)
c     deni   densities of the hydrogenic ions (cm**-3)
c     numz   number of impurity species
c     if numz gt 0, the coronal data file 'coronb' is read.
c     iz     atomic numbers of the impurities
c     izstrp =0 for coronal equilibrium values of <z> & <zsq>
c     =1 for fully stripped impurities
c     amz    atomic mass number of the impurities [iff izstrp#0]
c     denz   densities of the impurities (cm**-3)
c     bperp  magnetic field perpendicular to beamline (tesla (for ilorent=1))
c     kdene  =0: discard electron reactions
c     =1: include electron reactions (default)
c     kdeni  =0: discard ion (hydrogen) reactions
c     =1: include ion reactions (default)
c     kdenz  =0: discard impurity reactions
c     =1: include impurity reactions (default)
c     ksvi   =0: simple multiplication instead of full sigma-v
c     integral for ions
c     =1: full sigma-v integral for ions
c     ksvz   =0: simple multiplication instead of full sigma-v
c     integral for impurities
c     =1: full sigma-v integral for impurities
c     ksve   =0: sigma-v integrals for electron reactions
c     involving 1s, 2s, and 2p       rates for other
c     electron reactions from vriens & smeets.
c     =1: full sigma-v integrals for electron reactions
c     krad   =0: discard hydrogen line radiation
c     =1: include hydrogen line radiation .
c     ngh        order of gauss-hermite integrations for
c     ion and impurity reactions (default 24)
c     ngl        order of gauss-laguerre integrations for
c     electron reactions
c     permissible values: 10,16,20,24
c     nouthx unit number for messages(to unit nouthx if > 0)
c     (also nouthx>0 gives detailed diagnostic output)
c     ncorin unit number for reading coronb ce data file
c
c--   output:
c     nsigmav n<sigma-v> [sec**-1] : all processes included
c     lambda  mean free path (cm)
c     hexfrac fraction of 3rd excited state
c     ihxerr: error return
c     0     no error
c     1     radiation file coronb not found
c     2     d01bbf error in setting up ngh integration arrays
c     3     d01bbf error in setting up ngl integration arrays
c     4     sezf error: states "i" & "j" are the same!
c     5     sezf error: state "j" is 0
c     6     seef error       states "i" &/or "j" are in error
c     7     cief error
c     8     ceef error
c     9     eigrf error       eigenvalue determination is incorrect
c     10    input parameter error
c--   note:this routine uses imsl libraries.
c
c--   foreign files required:
c     hx2:[wieland.hex]coronb. :  this file contains all the coronal
c     equilibrium data for the various impurities.
c     dsk0:[imsl]imslibs/lib -- the imsl replacements for the nag codes.
c     nag:    f02aff et. al.
c     imsl:   eigrf  et. al.
c     hx2:[wieland.lib]wielib  for diagnostic matrix solvers
c     "decomp" & "solveq" for ax = b .
c     subr second -- a cpu timing routine of your choice
c--   recommended namelist values:
c     kdene = 1
c     kdeni = 1
c     kdenz = 1
c     ksvi  = 0
c     ksvz  = 0
c     ksve  = 0
c     krad  = 1
c     ngh   = 10
c     ngl   = 10
c     iexcit= 1
c     ilorent=0
c     mstate= 4
c     ncont = 30
c.....
c     these result in a cpu time of approx. 7 sec for a 25 point
c     radial profile of n*sigma(rho) with reasonably good accuracy.
c
      parameter (ms=21,mc=35)
      parameter (mz=1,mi=2)

      dimension amix(mi),denix(mi),izx(mz),denzx(mz),izstrx(mz),
     *  amzx(mz)

      common/cpub/ cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7,cpu8,
     >  cpuii, cpuiz, cpuplf

      common/b0/ns,nc,er0,v0,te,ti,numi,ami(mi),deni(mi),
     *  numz,iz(mz),amz(mz),denz(mz),zcor(mz),zsqcor(mz),
     *  izstrp(mz),dene
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
      common/b2/nouthx,istart,ihxbug
      common/b8/iexcit,ilorent,mstate,ncont
      common/b10/xfrac

      real*8 nsigmav, lambda
c
      istart = istarx

c--   move vbls in hexnb commons
      iexcit = iexcix
      ilorent = ilorenx
      mstate = mstatx
      ncont = ncontx
      er0 = er0x
      te = tex
      ti = tix
      numi = numix
      if(numi.gt.mi) go to 9999
      do 2 i = 1, mi
        ami(i) = amix(i)
        deni(i) = denix(i)
    2 continue
      if(numz.gt.mz) go to 9999
      numz = numzx
      do 3 i = 1, mz
        iz(i) = izx(i)
        izstrp(i) = izstrx(i)
        amz(i) = amzx(i)
        denz(i) = denzx(i)
    3 continue
      bperp = bperpx
      kdene = ldene
      kdeni = ldeni
      kdenz = ldenz
      ksvi = lsvi
      ksvz = lsvz
      ksve = lsve
      krad = lrad
      ngh = lngh
      ngl = lngl
      nouthx = louthx
      ncor = lcor
c
      ihxbug = 0
      nsvigma = 0.
      lambda = 0.
      if(istart.eq.0)go to 5

      if(mstate+1.gt.ms .or. ncont+1 .gt. mc) go to 9999
c     ncor=31
      call hradin(ncor,numz,iz,izstrp,amz)
      if(ihxbug.gt.0) go to 20
      call hxinit
c
    5 continue
c
      v0=(1.3841e6)*sqrt(er0)
      call hxradi(te,zcor,zsqcor,numz,iz,izstrp,iwatch)
      dene=0.
      do 17 ki=1,numi
        dene=dene+deni(ki)
 17   continue
      do 18 kz=1,numz
        dene=dene+zcor(kz)*denz(kz)
 18   continue
c
c--   determine calculational mode: include excitations or not?
c
      goto (21,22),iexcit+1
c
c     no excitations included
 21   continue
      nc=1
c     if(nouthx.gt.0) write(nouthx,1000)
 1000 format(/,' no excitations')
      goto 23
c
c     include excitations
 22   continue
      nc=ncont+1
c     if(nouthx.gt.0) write(nouthx,1004) ncont
 1004 format(/,' excitations, with continuum at n=',i3)
c
 23   continue
      call lorent(v0,bperp,nc,ns)
c     if(nouthx.gt.0) write(nouthx,1005)ns
 1005 format(/,' max princ qn= ',i3)

      call hxsvi
      if(ihxbug.gt.0) go to 20

      call hxsve
      if(ihxbug.gt.0) go to 20

      call matri

      call eigen(xeig)
      if(ihxbug.gt.0) go to 20

      lambda = xeig
      nsigmav = v0 / xeig
      hexfrac=xfrac

c
 20   continue
      ihxerr = ihxbug
      istart=0
      return

 9999 continue
      ihxbug = 10
      if(nouthx.gt.0) write(nouthx,3939) numi,numz,mstate,ncont
 3939 format(' ??? inconsistency between input vbls and upper ',
     >  'limits as defined by parameters :',/,
     >  3x,'numi= ',i4,/
     >  3x,'numz= ',i4,/
     >  3x,'mstate= ',i4,/
     >  3x,'ncont= ',i4)
      go to 20

      end subroutine hexnb
c
c
      subroutine hradin(ncor,numz,nz,izstrp,amz)
      implicit integer (i-n), real*8 (a-h,o-z)
c*****************************************************************
c--   ifnumz ne 0, read the post radiation tables (file 'coronb').
      parameter (mz=1,mi=2)
      common/cpub/ cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7,cpu8,
     >  cpuii, cpuiz, cpuplf
      common/b2/nouthx,istart,ihxbug
      common/locrad/arad(8,15,mz)
      character*8 spec
      dimension spec(mz),amz(mz),nz(mz),izstrp(mz)
      character*8 dum,dum2
c
c--   arrangement of arad(j,k,i) for given i:
c
c     1    2    3    4    5    6    7    8    9    10   11   12   13   14  15
c     1  t1   t2   t3   t4   0.   t1   t2   t3   t4   0.   t1   t2   t3   t4  0.
c     2  t2   t3   t4   t5   0.   t2   t3   t4   t5   0.   t2   t3   t4   t5  0.
c     3 a(0) a(0) a(0) a(0)  0.  b(0) b(0) b(0) b(0)  0.  c(0) c(0) c(0) c(0) 0.
c     4  .    .    .    .    .    .    .    .    .    .    .    .    .    .   .
c     5  .    .    .    .    .    .    .    .    .    .    .    .    .    .   .
c     6  .    .    .    .    .    .    .    .    .    .    .    .    .    .   .
c     7  .    .    .    .    .    .    .    .    .    .    .    .    .    .   .
c     8 a(5) a(6) a(5) a(5)  0.  b(5) b(5) b(5) b(5)  0.  c(5) c(5) c(5) c(5) 0.
c
cBH080118      call second(cpua)
      if(numz.eq.0) return
      izsum = 0
      do 5 i = 1,numz
        izsum = izsum + izstrp(i)
    5 continue
      if(izsum.eq.numz) return

      do 60 imp=1,numz
        do 20 k=1,15
          do 10 j=1,8
            arad(j,k,imp)=0.0
 10       continue
 20     continue
c**   find desired element in data table.
c
        open(unit=ncor,file='coronb',status='old')
c     if(length.eq.-1) go to 70
        do 30 j=1,10000
          read(ncor,1010,err=70,end=70)dum,dum2,iz,ia
 69       continue
          if(iz.eq.nz(imp)) go to 40
          read(ncor,1020) dum
          read(ncor,1020) dum
 30     continue
c**   found desired element.
 40     spec(imp)=dum
        amz(imp)=ia
c**   read data for this specie.
        k=1
 50     read(ncor,1020)(arad(j,k,imp),j=1,8),dum
        k=k+1
c990131        if(dum.eq.2hzb.and.k.lt.6) k=6
        if(dum.eq.'zb'.and.k.lt.6) k=6
c990131        if(dum.eq.2hzs.and.k.lt.11) k=11
c990131        if(k.ge.15.and.dum.ne.2hzs) go to 55
        if(dum.eq.'zs'.and.k.lt.11) k=11
        if(k.ge.15.and.dum.ne.'zs') go to 55
        go to 50
c
 55     continue
        close(unit=ncor)
c
 60   continue
cBH080118      call second(cpub)
      cpu8 = cpu8 + cpub - cpua
      return
c**   could not find desired element.
 70   if(nouthx.gt.0) write(nouthx,1030) nz(imp),ncor
      ihxbug = 1
 1010 format(1a2,15x,a4,2i4)
 1020 format(2e10.3,3e15.6,/,3e15.6,/,12x,1a2)
 1030 format(1x,9helement #,i10,19h not found on unit ,i5)
      return
      end
c
c
      real*8 function hxgf(n,y)
      implicit integer (i-n), real*8 (a-h,o-z)
c*****************************************************************
      g0(r)=.9935+.2328/r-.1296/(r*r)
      g1(r)=-(.6282-.5598/r+.5299/(r*r))/r
      g2(r)=(.3887-1.181/r+1.470/(r*r))/(r*r)
c
      rn=dfloat(n)
      if(n.eq.1)goto 11
      if(n.eq.2)goto 12
      hxgf=g0(rn)+g1(rn)/y+g2(rn)/(y*y)
      return
c
 11   hxgf=1.1330-.4059/y+.07014/(y*y)
      return
c
 12   hxgf=1.0785-.2319/y+.02947/(y*y)
      return
      end
c
c
      subroutine hxinit
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
      parameter (ms=21,mc=35)
      common/cpub/ cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7,cpu8,
     >  cpuii, cpuiz, cpuplf
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
      common/b3/f(ms,mc),ar(ms+1,ms+1)
      common/b4/en(mc+1),dg(mc+1),ae(ms,mc),be(ms,mc),
     *  de1(ms,mc),de2(ms,mc),ge1(ms,mc),ge2(ms,mc)
      common/b8/iexcit,ilorent,mstate,ncont
      dimension be1(mc)
      data ryd/13.6/
c
c--   total radiation rate from level n2 to level n1:
      rrate(n2,n1)=(8.0323e9)*(((1./dfloat(n1))**2-(1./dfloat(n2))**2)
     *  *(dfloat(n1)/dfloat(n2)))**2*f(n1,n2)
c
cBH080118      call second(cpua)
c
c--   tabulate oscillator strengths:
      call hxosc
c
      dg(1)=1.
      dg(2)=1.
      dg(3)=3.
      en(1)=ryd
      en(2)=ryd/4.
      en(3)=ryd/4.
      do 7 i=4,mc+1
        dg(i)=(dfloat(i-1))**2
        en(i)=ryd/dg(i)
    7 continue
c
      do 8 n1=1,mc
        an1=dfloat(n1)
c990131        be1(n1)=1.4*alog(an1)/an1-.7/an1-.51/(an1*an1)+1.16/(an1**3)
        be1(n1)=1.4*log(an1)/an1-.7/an1-.51/(an1*an1)+1.16/(an1**3)
     *    -.55/(an1**4)
    8 continue
c
      do 9 n1=1,mstate
        an1=dfloat(n1)
        en1=ryd/(an1*an1)
        do 10 n2=n1+1,mc
          an2=dfloat(n2)
          en12=en1-ryd/(an2*an2)
          ae(n1,n2)=2.*ryd*f(n1,n2)/en12
          be(n1,n2)=4.*ryd*ryd*(1./(en12*en12)+4.*en1/(3.*en12**3)
     *      +be1(n1)*en1*en1/(en12**4))/(an2**3)
          de1(n1,n2)=exp(-be(n1,n2)/ae(n1,n2))-.4*en12/ryd
          s=an2-an1
          de2(n1,n2)=exp(-be(n1,n2)/ae(n1,n2))+.06*s*s/(an1*an1*an2)
          ge1(n1,n2)=ryd*(8.+23.*(s/an1)**2)
     *      /(8.+1.1*an2*s+.8/(s*s)+.4*(s-1.)*sqrt(an2*an2*an2/s))
          ge2(n1,n2)=ryd*(3.+11.*(s/an1)**2)
     *      /(6.+1.6*an2*s+.3/(s*s)+.8*(s-.6)*sqrt(an2*an2*an2/s))
 10     continue
    9 continue
c
c--   radiation rates:
      do 205 i=1,mstate+1
        do 206 j=1,mstate+1
          ar(i,j)=0.
 206    continue
 205  continue
      if(krad.eq.0)go to 1000
c
      do 210 j=4,mstate+1
        nj=j-1
        ar(j,1)=rrate(nj,1)
        do 211 i=4,j-1
          ni=i-1
          ar(j,i)=rrate(nj,ni)
 211    continue
 210  continue
c
c--   2s to 1s:
      ar(2,1)=0.
c
c--   2p to 1s:
      ar(3,1)=(4./3.)*rrate(2,1)
c
c--   2p to 2s:
      ar(3,2)=0.
c
      do 220 j=4,mstate+1
        nj=j-1
        ajsq=(dfloat(nj))**2
c
c--   total radiation to 2s+2p:
        tot=rrate(nj,2)
c
c--   fraction to 2s:
        frac2s=12.*(ajsq-1.)*(ajsq-4.)
        frac2s=frac2s/(frac2s+ajsq*(ajsq-4.)+32.*ajsq*(ajsq-1.))
c
        ar(j,2)=frac2s*tot
        ar(j,3)=(1.-frac2s)*tot
 220  continue
c
 1000 continue
cBH080118      call second(cpub)
      cpu1 = cpub - cpua
c
      return
      end
c
c
      subroutine hxosc
      implicit integer (i-n), real*8 (a-h,o-z)
c*****************************************************************
c--   calculate oscillator strengths.
c
      parameter (ms=21,mc=35)
      common/b3/f(ms,mc),ar(ms+1,ms+1)
      common/b8/iexcit,ilorent,mstate,ncont
c
      zero=0.d0
      one=1.d0
c      pi=acos(-one)
      pi=atan2(zero,-one)
      const=32./(sqrt(27.)*pi)
c
      do 10 i=1,mstate
        ai=dfloat(i)
        do 11 j=i+1,mc
          aj=dfloat(j)
          y=1.-(ai/aj)**2
          f(i,j)=(const*ai/((aj*y)**3))*hxgf(i,y)
 11     continue
 10   continue
      return
      end
c
c
      subroutine hxradi(teev,z,zsq,numimp,iz,izstrp,iwatch)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
c--   evaluate coronal z, zsq, and radiation.
      parameter (mz=1,mi=2)
      common/cpub/ cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7,cpu8,
     >  cpuii, cpuiz, cpuplf
      common/locrad/arad(8,15,mz)
      dimension rad(mz),z(mz),zsq(mz),b(3),iz(mz),izstrp(mz)
      data erad/1./
      zero=0.d0
cBH080118      call second(cpua)
      iwatch=0
      ihigh=0
      tekev=teev*1.0D-3
      do 60 imp=1,numimp
        if(izstrp(imp).eq.1) go to 51
c**   find temperature region. test for less than 5 intervals.
        do 10 j=1,5
          if(arad(2,j,imp).eq.zero) go to 10
          jj=j
          if(tekev.ge.arad(1,j,imp).and.tekev.lt.arad(2,j,imp)) go to 30
 10     continue
c**   temperature out of range.
        iwatch=1
        if(tekev.ge.arad(1,1,imp)) go to 20
c**   temperature too low. return zero power.
c     rad(imp)=0.0
c     z(imp)=0.0
c     zsq(imp)=0.0
c**   interpolation for low te (cdb):
c990131        t1=alog10(arad(1,1,imp))
        t1=log10(arad(1,1,imp))
        do 15 j=1,3
          jp=5*j-4
          b(j)=arad(3,jp,imp)+t1*(arad(4,jp,imp)+t1*(arad(5,jp,imp)
     *      +t1*(arad(6,jp,imp)+t1*(arad(7,jp,imp)
     *      +t1*arad(8,jp,imp)))))
 15     continue
        factor=(tekev/arad(1,1,imp))**erad
        rad(imp)=factor*(10.**b(1))
        z(imp)=factor*b(2)
        zsq(imp)=factor*b(3)
        go to 50
c**   temperature too high. compute value for tmax, then extrapolate.
c990131 20     tl=alog10(arad(2,jj,imp))
 20     tl=log10(arad(2,jj,imp))
        ihigh=1
c990131 30     if(ihigh.eq.0) tl=alog10(tekev)
 30     if(ihigh.eq.0) tl=log10(tekev)
c**   polonomial fits bb=sum(a(ik)*log(te)**k) etc.
        bb=0.0
        cc=0.0
        dd=0.0
        do 40 kk=1,6
          k=7-kk
          bb=bb*tl+arad((k+2),jj,imp)
          cc=cc*tl+arad((k+2),(jj+5),imp)
          dd=dd*tl+arad((k+2),(jj+10),imp)
 40     continue
        rad(imp)=0.0
        z(imp)=cc
        zsq(imp)=dd
        if(bb.lt.-38.0) go to 50
        rad(imp)=10.0**bb
c**   for te>tmax extrapolate as for pure bremsstrahlung.
        if(ihigh.eq.1) rad(imp)=rad(imp)*sqrt(tekev/arad(2,4,imp))
        ihigh=0
 50     continue
        go to 60

c...  fully stripped option [ izstrp(imp)=1 ]
 51     z(imp)   = iz(imp)
        zsq(imp) = iz(imp)**2

 60   continue
c
cBH080118      call second(cpub)
      cpu2 = cpu2 + cpub - cpua
c
      return
      end
c
c
      subroutine hxsve
      implicit integer (i-n), real*8 (a-h,o-z)
c*****************************************************************
c--   calculate the rate coefficients for collisions with electrons.
c
      parameter (ms=21,mc=35,mz=1,mi=2 )
      common/cpub/ cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7,cpu8,
     >  cpuii, cpuiz, cpuplf
      common/b0/ns,nc,er0,v0,te,ti,numi,ami(mi),deni(mi),
     *  numz,iz(mz),amz(mz),denz(mz),zcor(mz),zsqcor(mz),
     *  izstrp(mz),dene
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
      common/b2/nouthx,istart,ihxbug
      common/b4/en(mc+1),dg(mc+1),ae(ms,mc),be(ms,mc),
     *  de1(ms,mc),de2(ms,mc),ge1(ms,mc),ge2(ms,mc)
      common/b6/cii(ms+1,mi),cei(ms+1,ms+2,mi),ciz(ms+1,mz),
     *  cez(ms+1,ms+2,mz),cie(ms+1),cee(ms+1,ms+2),
     *  ccxi(ms+1,mi),ccxz(ms+1,mz)
      dimension et(mc+1)
      dimension xgl(64),wgl(64)
      character*2 typent
c@    external d01bax
c
      sinh1(arg)=sinh(arg)/arg
c
cBH080118      call second(cpua)
c
c--   initialize:
      if(istart.eq.0)goto 5
      zero=0.d0
      one=1.d0
c      pi=acos(-one)
      pi=atan2(zero,-one)
      sqpi=sqrt(pi)
c
c     e-impact thresholds (temp.)
      et(1)=(13.723/13.6)*en(1)
      do 1 i=2,mc+1
        et(i)=en(i)
    1 continue
c
      ifail=0
      typent = 'gl'
CMG  changed 11/13/2017         call d01bbf(typent,ngl,wgl,xgl,ifail) 
      call d01bbf_cql3d(typent,ngl,wgl,xgl,ifail) 
c@    call d01bbf(d01bax,0.,1.,0,ngl,wgl,xgl,ifail)
      if(ifail.eq.0)goto 5
      if(nouthx.gt.0) write(nouthx,3939) ifail
 3939 format(' ??? d01bbf [ngl]  call error: ifail= ',i4)
      ihxbug = 3
      return
c
    5 continue
      nsp1=ns+1
c
      do 205 i=1,ns
        cie(i)=0.
        do 206 j=i+1,nsp1
          cee(i,j)=0.
 206    continue
 205  continue
      ve=5.931e7*sqrt(te)
c
      if(kdene.eq.0)go to 1000
      goto(220,210),ksve+1
c
c--   gauss-laguerre integrations:
 210  continue
      facte=(2./sqpi)*ve*exp(-v0*v0/(ve*ve))
      do 211 kk=1,ngl
        do 212 i=1,ns
          cie(i)=cie(i)
     *      +wgl(kk)*(xgl(kk)+et(i)/te)
     *      *sinh1(2.*v0*sqrt(xgl(kk)+et(i)/te)/ve)
     *      *sief(i,et(i)+te*xgl(kk))
     *      *facte*exp(-et(i)/te)
          do 213 j=i+1,ns
            cee(i,j)=cee(i,j)
     *        +wgl(kk)*(xgl(kk)+(et(i)-et(j))/te)
     *        *sinh1(2.*v0*sqrt(xgl(kk)+(et(i)-et(j))/te)/ve)
     *        *seef(i,j,et(i)-et(j)+te*xgl(kk))
     *        *facte*exp(-(et(i)-et(j))/te)
 213      continue
          do 214 j=nsp1,nc
            cee(i,nsp1)=cee(i,nsp1)
     *        +wgl(kk)*(xgl(kk)+(et(i)-et(j))/te)
     *        *sinh1(2.*v0*sqrt(xgl(kk)+(et(i)-et(j))/te)/ve)
     *        *seef(i,j,et(i)-et(j)+te*xgl(kk))
     *        *facte*exp(-(et(i)-et(j))/te)
 214      continue
 212    continue
 211  continue
      go to 1000
c
 220  continue
c--   gauss-laguerre averages for ionization of 1s, ionization of 2s,
c     and excitations among 1s, 2s, and 2p:
      facte=(2./sqpi)*ve*exp(-v0*v0/(ve*ve))
      do 221 kk=1,ngl
        do 222 i=1,min0(2,ns)
          cie(i)=cie(i)
     *      +wgl(kk)*(xgl(kk)+et(i)/te)
     *      *sinh1(2.*v0*sqrt(xgl(kk)+et(i)/te)/ve)
     *      *sief(i,et(i)+te*xgl(kk))
     *      *facte*exp(-et(i)/te)
          do 2221 j=i+1,min0(3,ns)
            cee(i,j)=cee(i,j)
     *        +wgl(kk)*(xgl(kk)+(et(i)-et(j))/te)
     *        *sinh1(2.*v0*sqrt(xgl(kk)+(et(i)-et(j))/te)/ve)
     *        *seef(i,j,et(i)-et(j)+te*xgl(kk))
     *        *facte*exp(-(et(i)-et(j))/te)
 2221     continue
 222    continue
 221  continue
c
c--   rates from vriens & smeets for other reactions:
c     ionizations--
      do 223 i=3,ns
        cie(i)=cief(i,te)
 223  continue
c
c     excitations--
      do 224 i=1,ns
        do 225 j=i+1,ns
          if(j.le.3)goto 225
          cee(i,j)=ceef(i,j,te)
 225    continue
        do 226 j=nsp1,nc
          cee(i,nsp1)=cee(i,nsp1)+ceef(i,j,te)
 226    continue
 224  continue
c
 1000 continue
cBH080118      call second(cpub)
      cpu5 = cpu5 + cpub - cpua
c
      return
      end subroutine hxsve
c
c
      subroutine hxsvi
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
c--   calculate the rate coefficients for collisions with ions
c     and impurities.
c
      parameter (ms=21,mc=35,mz=1,mi=2)
      common/cpub/ cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7,cpu8,
     >  cpuii, cpuiz, cpuplf
      common/b0/ns,nc,er0,v0,te,ti,numi,ami(mi),deni(mi),
     *  numz,iz(mz),amz(mz),denz(mz),zcor(mz),zsqcor(mz),
     *  izstrp(mz),dene
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
      common/b2/nouthx,istart,ihxbug
      common/b6/cii(ms+1,mi),cei(ms+1,ms+2,mi),ciz(ms+1,mz),
     *  cez(ms+1,ms+2,mz),cie(ms+1),cee(ms+1,ms+2),
     *  ccxi(ms+1,mi),ccxz(ms+1,mz)
      dimension vi(mi),vz(mz)
      dimension xgh(64),wgh(64),w1(32)
      character*2 typent
c@    external d01baw
      data ister0/1/
c
c--   initialize:
cBH080118      call second(cpua)
c
      if(istart.eq.0)goto 5
      zero=0.d0
      one=1.d0
c      pi=acos(-one)
      pi=atan2(zero,-one)
      sqpi=sqrt(pi)
c
      ngh1=(ngh+1)/2
      do 3 i=1,ngh1
        w1(i)=2.
    3 continue
      if(2*ngh1-ngh .eq. 1)w1(ngh1)=1.
      ifail = 0
      typent = 'gh'
CMG  changed 11/13/2017      call d01bbf(typent,ngh,wgh,xgh,ifail)
      call d01bbf_cql3d(typent,ngh,wgh,xgh,ifail)
c@    call d01bbf(d01baw,0.,1.,0,ngh,wgh,xgh,ifail)
      if(ifail.eq.0)goto 5
      if(nouthx.gt.0) write(nouthx,3939) ifail
 3939 format(' ??? d01bbf [ngh]  call error: ifail= ',i4)
      ihxbug = 2
      return
c
    5 continue
      nsp1=ns+1
      print *,'****************************************'
      print *,' warning er0old not defined in zfreya'
      print *,'****************************************'
      if(er0.ne.er0old) ister0=1
c
c--   ion reactions-----------------------------------------------
c
cBH080118      call second(cpua1)
c
c--   first zero out arrays:
      if( (ister0.eq.1 .or. istart.eq.1) .or. ksvi.eq.1) then
        do 10 ki=1,numi
          do 11 i=1,ns
            cii(i,ki)=0.
            ccxi(i,ki)=0.
            do 12 j=i+1,nsp1
              cei(i,j,ki)=0.
 12         continue
 11       continue
 10     continue
      endif
c
      do 13 ki=1,numi
        vi(ki)=1.3841e6*sqrt(ti/ami(ki))
 13   continue

      if(kdeni.eq.0)goto 100
      if(ksvi.eq.0)goto 40
c
c--   gauss-hermite integrations:
      do 20 kk=1,ngh1
        do 21 ki=1,numi
          xp=(xgh(kk)+v0/vi(ki))**2
          xm=(xgh(kk)-v0/vi(ki))**2
          s=1.
          if(xgh(kk)-v0/vi(ki) .lt. 0.)s=-1.
          ei=ti/ami(ki)
          do 22 i=1,ns
            cii(i,ki)=cii(i,ki)+wgh(kk)*w1(kk)
     *        *(xp*siif(i,ei*xp)-s*xm*siif(i,ei*xm))
            ccxi(i,ki)=ccxi(i,ki)+wgh(kk)*w1(kk)
     *        *(xp*scxif(i,ei*xp)-s*xm*scxif(i,ei*xm))
            do 23 j=i+1,ns
              cei(i,j,ki)=cei(i,j,ki)+wgh(kk)*w1(kk)
     *          *(xp*sezf(i,j,one,ei*xp)-s*xm*sezf(i,j,one,ei*xm))
 23         continue
            do 24 j=nsp1,nc
              cei(i,nsp1,ki)=cei(i,nsp1,ki)+wgh(kk)*w1(kk)
     *          *(xp*sezf(i,j,one,ei*xp)-s*xm*sezf(i,j,one,ei*xm))
 24         continue
 22       continue
 21     continue
 20   continue
c
      do 30 ki=1,numi
        facti=vi(ki)*vi(ki)/(2.*sqpi*v0)
        do 31 i=1,ns
          cii(i,ki)=facti*cii(i,ki)
          ccxi(i,ki)=facti*ccxi(i,ki)
          do 32 j=i+1,nsp1
            cei(i,j,ki)=facti*cei(i,j,ki)
 32       continue
 31     continue
 30   continue
      goto 100
c
c--   simple multiplication instead of maxwellian averaging:
c--   cii & cei are independent of plasma parameters in this
c--   approximation       so as long as er0 doesn't change, use
c--   the old values.
c--   ister0=1 ==> restart everything
c--   =0 ==> cruise in no-update mode
 40   continue
c--   redo everything if er0 changes
      if( (ister0.eq.1 .or. istart.eq.1) .or. ksvi.eq.1) then
        do 41 ki=1,numi
          do 411 i=1,ns
            cii(i,ki)=siif(i,er0)*v0
            ccxi(i,ki)=scxif(i,er0)*v0
            do 42 j=i+1,ns
              cei(i,j,ki)=sezf(i,j,one,er0)*v0
 42         continue
            do 43 j=nsp1,nc
              cei(i,nsp1,ki)=cei(i,nsp1,ki)+sezf(i,j,one,er0)*v0
 43         continue
 411      continue
 41     continue
      endif
 100  continue
cBH080118      call second(cpua2)
      cpuii = cpuii + cpua2 - cpua1
c
c--   impurity reactions------------------------------------------
c
c--   zero out arrays:
      do 110 kz=1,numz
        vz(kz)=1.3841e6*sqrt(ti/amz(kz))
        do 111 i=1,ns
          ciz(i,kz)=0.
          ccxz(i,kz)=0.
          do 112 j=i+1,nsp1
            cez(i,j,kz)=0.
 112      continue
 111    continue
 110  continue
c
      if(kdenz.eq.0)go to 1000
      if(ksvz.eq.0)goto 140
c
c--   gauss-hermite integrations:
      do 120 kk=1,ngh1
        do 121 kz=1,numz
          xp=(xgh(kk)+v0/vz(kz))**2
          xm=(xgh(kk)-v0/vz(kz))**2
          s=1.
          if(xgh(kk)-v0/vz(kz) .lt. 0.)s=-1.
          ez=ti/amz(kz)
          zz=zcor(kz)
          do 122 i=1,ns
            ciz(i,kz)=ciz(i,kz)+wgh(kk)*w1(kk)
     *        *(xp*sizf(i,zz,ez*xp)-s*xm*sizf(i,zz,ez*xm))
            ccxz(i,kz)=ccxz(i,kz)+wgh(kk)*w1(kk)
     *        *(xp*scxzf(i,zz,ez*xp)-s*xm*scxzf(i,zz,ez*xm))
            do 123 j=i+1,ns
              cez(i,j,kz)=cez(i,j,kz)+wgh(kk)*w1(kk)
     *          *(xp*sezf(i,j,zz,ez*xp)-s*xm*sezf(i,j,zz,ez*xm))
 123        continue
            do 124 j=nsp1,nc
              cez(i,nsp1,kz)=cez(i,nsp1,kz)+wgh(kk)*w1(kk)
     *          *(xp*sezf(i,j,zz,ez*xp)-s*xm*sezf(i,j,zz,ez*xm))
 124        continue
 122      continue
 121    continue
 120  continue
c
      do 130 kz=1,numz
        factz=vz(kz)**2/(2.*sqpi*v0)
        do 131 i=1,ns
          ciz(i,kz)=factz*ciz(i,kz)
          ccxz(i,kz)=factz*ccxz(i,kz)
          do 132 j=i+1,nsp1
            cez(i,j,kz)=factz*cez(i,j,kz)
 132      continue
 131    continue
 130  continue
      go to 1000
c
c--   simple multiplications:
 140  continue
      do 141 kz=1,numz
        do 142 i=1,ns
          ciz(i,kz)=sizf(i,zcor(kz),er0)*v0
          ccxz(i,kz)=scxzf(i,zcor(kz),er0)*v0
          do 143 j=i+1,ns
            cez(i,j,kz)=sezf(i,j,zcor(kz),er0)*v0
 143      continue
          do 144 j=nsp1,nc
            cez(i,nsp1,kz)=cez(i,nsp1,kz)+sezf(i,j,zcor(kz),er0)*v0
 144      continue
 142    continue
 141  continue
c
 1000 continue
cBH080118      call second(cpub)
      cpu4 = cpu4 + cpub - cpua
      cpuiz = cpuiz + cpub - cpua2
c
      ister0=0
      er0old = er0
      return
      end subroutine hxsvi
c
c
c$$$ Replaced by Kinsey, when adding ADAS cross-sections to old (1990's)
c$$$ CQL3D/NFREYA  (June-July 2013).
c$$$      subroutine inject(atw, codeid, drutpi, droti, dri, dzi,
c$$$     *  elongi, ib, ie, ke, kz, kbe, ki, mfm1, mim1, mjm1,
c$$$     *  newpar, psiax, psi,
c$$$     *  r, rmajor, rin, rmax, sgxn, sgxnmi,
c$$$     *  x0, y0, z0, vx0, vy0, vz0, vbeam, z, zax, zmin, zmax,
c$$$     *  izone, pzone, rzone, rpos, xpos, ypos, zpos)
c$$$      implicit integer (i-n), real*8 (a-h,o-z)
c$$$      save
c$$$c
c$$$c     this subroutine follows the particle from the pivot point
c$$$c     into, through, or around the plasma.
c$$$c
c$$$      dimension psi(ki,*), sgxn(4,kz,kbe,*), sgxnmi(ke,*),
c$$$     *  r(*), z(*), vbeam(ke,*)
c$$$      character codeid*8
c$$$
c$$$      integer iflag
c$$$      data       seed0    /0.0d0/
c$$$
c$$$c      iflag=0
c$$$c
c$$$c%OS  
c$$$cbh   (930920) print *,' WARNING: isol not defined in inject'
c$$$c%OS  
c$$$cbh   (930920) if(newpar.eq.0 .and. isol.eq.0) go to 140
c$$$      if(newpar.eq.0) go to 100
c$$$c
c$$$c     calculate times for particle to enter and exit toroidal box
c$$$c     surrounding plasma
c$$$c
c$$$      call timtor(rin, rmax, x0, y0, z0, vx0, vy0, vz0, zmin, zmax,
c$$$     *  tenter, texit)
c$$$c      write(*,*) 'inject:rin,rmax,x0,y0,z0,vx0,vy0,vz0,zmin,zmax',
c$$$c     +     ',tenter,texit',
c$$$c     +     rin,rmax,x0,y0,z0,vx0,vy0,vz0,zmin,zmax,tenter,texit
c$$$
c$$$c      write(*,*)'inject: tenter',tenter
c$$$      if(tenter.le.-1.e10) go to 140
c$$$c      write(*,*)'inject: tenter',tenter
c$$$c
c$$$c     advance particle to edge of box
c$$$c
c$$$      x0=x0+vx0*tenter
c$$$      y0=y0+vy0*tenter
c$$$      z0=z0+vz0*tenter
c$$$c
c$$$c     set coordinates and time for entering box
c$$$c
c$$$ 100  continue
c$$$      xpos=x0
c$$$      ypos=y0
c$$$      zpos=z0
c$$$      tt=tenter
c$$$c      write(*,*)'inject:xpos,ypos,zpos,tt',xpos,ypos,zpos,tt
c$$$c
c$$$c     follow particle into plasma
c$$$c
c$$$ 110  continue
c$$$c990131      dfac=-alog(ranf())
c$$$c000114*Not portable*      dfac=-log(drand(iflag))
c$$$      dfac=-log(RANDOM_my(seed0))
c$$$      tstep=dfac*sgxnmi(ie,ib)/vbeam(ie,ib)
c$$$      tt=tt+tstep
c$$$c      write(*,*) 'dfac,ie,ib,sgxnmi(ie,ib),vbeam(ie,ib),tstep',
c$$$c     +             dfac,ie,ib,sgxnmi(ie,ib),vbeam(ie,ib),tstep  
c$$$      if(tt.ge.texit) go to 140
c$$$      xpos=xpos+vx0*tstep
c$$$      ypos=ypos+vy0*tstep
c$$$      zpos=zpos+vz0*tstep
c$$$      rpos=sqrt(xpos**2+ypos**2)
c$$$c      write(*,*)'inject:xpos,ypos,zpos,rpos',xpos,ypos,zpos,rpos
c$$$
c$$$c
c$$$c     determine zone in which particle collides for 'onedee' geometry
c$$$c
c$$$      if(codeid.ne.'onedee') go to 120
c$$$      rzone2 = (rpos-rmajor)**2 + (elongi*(zpos-zax))**2
c$$$      rzone = sqrt(rzone2)
c$$$      izone = rzone*droti + 1.
c$$$      go to 130
c$$$c
c$$$c     determine zone in which particle collides for general geometry;
c$$$c     use bilinear interpolation away from magnetic axis and
c$$$c     biquadratic interpolation near the axis
c$$$c
c$$$ 120  i=(rpos-r(1))*dri+1.
c$$$      j=(zpos-z(1))*dzi+1.
c$$$      if(i.gt.mim1) i=mim1
c$$$      if(j.gt.mjm1) j=mjm1
c$$$c990131      psix  = amin1(psi(i,j), psi(i+1,j), psi(i,j+1), psi(i+1,j+1))
c$$$      psix  = min(psi(i,j), psi(i+1,j), psi(i,j+1), psi(i+1,j+1))
c$$$      ptest = (psix-psiax)*(drutpi/mfm1)**2
c$$$      if(ptest.lt.0.02) go to 124
c$$$      area1=(rpos-r(i))*(zpos-z(j))
c$$$      area2=(r(i+1)-rpos)*(zpos-z(j))
c$$$      area3=(r(i+1)-rpos)*(z(j+1)-zpos)
c$$$      area4=(rpos-r(i))*(z(j+1)-zpos)
c$$$      pzone=(area3*psi(i,j)+area4*psi(i+1,j)+area1*psi(i+1,j+1)
c$$$     1  +area2*psi(i,j+1))*dri*dzi
c$$$      go to 126
c$$$ 124  call pfit(psi(i-1,j-1), r(i-1), z(j-1), rpos, zpos, ki, pzone,
c$$$     *  dum, dum)
c$$$c990131 126  pzone = amax1(pzone,psiax)
c$$$ 126  pzone = max(pzone,psiax)
c$$$      izone = sqrt(pzone-psiax)*drutpi + 1.
c$$$c
c$$$c     if particle has psuedo-collision, continue following particle;
c$$$c     if particle has real collision, return
c$$$c
c$$$ 130  if(izone.gt.mfm1) go to 110
c$$$c990131      if(ranf().gt.sgxn(izone,ie,ib)*sgxnmi(ie,ib))
c$$$c000114*Not portable*      if(drand(iflag).gt.sgxn(izone,ie,ib)*sgxnmi(ie,ib))
c$$$      if(RANDOM_my(seed0).gt.sgxn(4,izone,ie,ib)*sgxnmi(ie,ib))
c$$$     1  go to 110
c$$$      return
c$$$c
c$$$c     set flag to indicate that particle hit wall
c$$$c
c$$$ 140  izone = mfm1+1
c$$$      return
c$$$      end
c
c
      subroutine inter(dxi,iroot,nxm1,x0,ytab,x,y)
      implicit integer (i-n), real*8 (a-h,o-z)
c
c     this subroutine performs a fast linear interpolation (or extrapo-
c     lation) for y vs. x for either of two cases:
c     1.  if iroot.eq.0, then x is uniformly spaced with
c     spacing 1/dxi,
c     2.  if iroot.ne.0, then sqrt(x) is uniformly spaced with
c     spacing 1/dxi.
c
      dimension ytab(*)

      xtab = abs(x-x0)
      if(iroot.ne.0) xtab = sqrt(xtab)
      xx = xtab*dxi + 1.
      i  = xx
      i  = min0(i,nxm1)
      wt = xx - i
      y  = (1.-wt)*ytab(i) + wt*ytab(i+1)
      return
      end
c
c
      subroutine itorb(c1,c2,c3,drtabi,mfm1,psival,ptab,rmajor,r,fcap)
      implicit integer (i-n), real*8 (a-h,o-z)
      dimension ptab(*)
c
      call inter(drtabi,0,mfm1,rmajor,ptab,r,psival)
      fcap = r**2 - c1*r - c2**2*psival**2 - 2.*c2*c3*psival - c3**2
      fcap = fcap/r
      return
      end
c
c
      subroutine logint(x,y)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
ccc   
ccc   interpolates y(x) quadratically and logarithmically
ccc   
      dimension xdat(15),ydat(15)
ccc   
      data xdat/4.,6.,8.,10.,20.,30.,40.,60.,80.,100.,
     1  200.,300.,400.,600.,800./
      data ydat/8.95D-01,8.75D-01,8.70D-01,8.65D-01,8.20D-01,
     1  7.25D-01,6.25D-01,4.40D-01,2.90D-01,1.90D-01,2.40D-02,
     2  5.25D-03,1.20D-03,1.60D-04,5.40D-05/
ccc   
      mdat=15
      mdatm=mdat-1
      do 10 i0=2,mdatm
        if(xdat(i0).ge.x) go to 11
 10   continue
 11   if(i0.gt.mdatm) i0=mdatm
      im=i0-1
      ip=i0+1
c990131      ylogp=alog(ydat(ip))
c990131      ylog0=alog(ydat(i0))
c990131      ylogm=alog(ydat(im))
      ylogp=log(ydat(ip))
      ylog0=log(ydat(i0))
      ylogm=log(ydat(im))
      dm=x-xdat(im)
      d0=x-xdat(i0)
      dp=x-xdat(ip)
      d0m=xdat(i0)-xdat(im)
      dp0=xdat(ip)-xdat(i0)
      dpm=xdat(ip)-xdat(im)
      dm0=-d0m
      d0p=-dp0
      dmp=-dpm
      facm=d0*dp/(dm0*dmp)
      fac0=dm*dp/(d0m*d0p)
      facp=dm*d0/(dpm*dp0)
      ylog=facm*ylogm+fac0*ylog0+facp*ylogp
      y=exp(ylog)
ccc   
      return
      end
c
c
      subroutine lorent(v0,bperp,nc,ns)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
      parameter (ms=21)
      common/cpub/ cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7,cpu8,
     >  cpuii, cpuiz, cpuplf
      common/b5/al(ms+1)
      common/b8/iexcit,ilorent,mstate,ncont
      dimension sl(30)
      data w0/4.1341d16/,almin/1.D-10/,almax/1.d15/,expl/4.d0/
c     data sl/
c     *1.00000000e+00,2.50000000e-01,2.77777778e-02,1.73611111e-03,
c     *6.94444444e-05,1.92901235e-06,3.93675989e-08,6.15118733e-10,
c     *7.59405843e-12,7.59405843e-14,6.27608135e-16,4.35838982e-18,
c     *2.57892889e-20,1.31578005e-22,5.84791131e-25,2.28434036e-27,
c     *7.90429189e-30,2.43959626e-32,6.75788439e-35,1.68947110e-37,
c     *3.83100022e-40,7.91528971e-43,1.49627405e-45,2.59769800e-48,
c     *4.15631680e-51,6.14839763e-54,8.43401595e-57,1.07576734e-59,
c     *1.27915260e-62,1.42128067e-65/
      data sl/
     *  1.00000000D+00,2.50000000D-01,2.77777778D-02,1.73611111D-03,
     *  6.94444444D-05,1.92901235D-06,3.93675989D-08,6.15118733D-10,
     *  7.59405843D-12,7.59405843D-14,6.27608135D-16,4.35838982D-18,
     *  2.57892889D-20,1.31578005D-22,5.84791131D-25,2.28434036D-27,
     *  7.90429189D-30,2.43959626D-32,6.75788439D-35,1.68947110D-37,
     *  10*0.0/
c
      fl(n,an3,el)=((4./(an3*el))**(2*n-1))
     *  *exp(-2./(3.*an3*el))/an3
c     input:
c     v0
c     bperp
c     nc
c     output:
c     ns
c     al(i),i=1,ns
c
cBH080118      call second(cpua)
c
      zero=0.d0
      if(nc.ge.2)goto 1
      ns=1
      al(1)=0.
      go to 1000
c
    1 continue
      if(ilorent.eq.0) then
        ns=mstate+1
        do 2 il=1,ns
          al(il)=0.
 2      continue
        go to 1000
      endif
c
      if(bperp.eq.zero)goto 3
c
c--   ep= abs(v x b)/(electric field at 1st bohr radius)
      ep=v0*bperp/5.1417e13
      ncrit=.5/(ep**(1./expl))
      if(ncrit.le.1)ns=1
      if(ncrit.ge.2)ns=ncrit+1
      if(ns.le.(mstate+1))goto 4
c
    3 continue
      ns=mstate+1
c
    4 continue
      do 10 i=1,ns
        if(i.ge.4)goto 13
        goto(11,12,12),i
c990131 11     al(1)=fl(1,1.,ep)
 11     one=1.d0
        al(1)=fl(1,one,ep)
        goto 10
c990131 12     al(i)=fl(2,8.,ep)*.5
 12     eight=8.
        al(i)=fl(2,eight,ep)*.5
        goto 10
 13     ni=i-1
        ani3=(dfloat(ni))**3
        al(i)=fl(ni,ani3,ep)*sl(ni)
 10   continue
c
c--   normalize rates       zero out rates which are too small:
      do 15 i=1,ns
        al(i)=al(i)*w0
        if(al(i) .le. almin)al(i)=0.
 15   continue
c
c--   if some rates are too high, truncate system:
      do 20 i=1,ns
        i1=i
        if(al(i) .gt. almax)goto 21
 20   continue
      go to 1000
 21   continue
      ns=i1-1
 1000 continue
cBH080118      call second(cpub)
      cpu3 = cpu3 + cpub - cpua
c
      return
      end
c
c
      real*8 function  polyf(c,n,x)
      implicit integer (i-n), real*8 (a-h,o-z)
c@    real function polyf(c,n,x)
c*****************************************************************
c--   evaluate the polynomial c(1)+c(2)*x+...+c(n)*x**(n-1).
      real*8 c
c@    real c
      common/cpub/ cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7,cpu8,
     >  cpuii, cpuiz, cpuplf
      dimension c(n)
cBH080118      call second(cpuaa)
      polyf=c(n)
      do 10 j=1,n-1
        polyf=polyf*x+c(n-j)
        polysv = polyf
 10   continue
cBH080118      call second(cpubb)
      cpuplf = cpuplf + cpubb - cpuaa
      return
      end
c
c
      subroutine matri
      implicit integer (i-n), real*8 (a-h,o-z)
c*****************************************************************
c--   form the matrix of rate coefficients which occurs
c     in the system of rate equations.  evaluate deexcitation via
c     detailed balance.
c--   vectorized.
c
      parameter (ms=21,mc=35,mz=1,mi=2 )
      common/cpub/ cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7,cpu8,
     >  cpuii, cpuiz, cpuplf
      common/b0/ns,nc,er0,v0,te,ti,numi,ami(mi),deni(mi),
     *  numz,iz(mz),amz(mz),denz(mz),zcor(mz),zsqcor(mz),
     *  izstrp(mz),dene
      common/b3/f(ms,mc),ar(ms+1,ms+1)
      common/b4/en(mc+1),dg(mc+1),ae(ms,mc),be(ms,mc),
     *  de1(ms,mc),de2(ms,mc),ge1(ms,mc),ge2(ms,mc)
      common/b5/al(ms+1)
      common/b6/cii(ms+1,mi),cei(ms+1,ms+2,mi),ciz(ms+1,mz),
     *  cez(ms+1,ms+2,mz),cie(ms+1),cee(ms+1,ms+2),
     *  ccxi(ms+1,mi),ccxz(ms+1,mz)
      common/b7/q(ms+1,ms+1)
      dimension dge(ms+1),dgi(ms+1)
c
cc*   dump c arrays
c     write(60,3310) cii
c3310 format(/,' cii= ',/,(1x,1p5e11.3))
c     write(60,3315) ccxii
c3315 format(/,' ccxii= ',/,(1x,1p5e11.3))
c     write(60,3320) cei
c3320 format(/,' cei= ',/,(1x,1p5e11.3))
c     write(60,3330) ciz
c3330 format(/,' ciz= ',/,(1x,1p5e11.3))
c     write(60,3335) ccxz
c3335 format(/,' ccxz= ',/,(1x,1p5e11.3))
c     write(60,3340) cez
c3340 format(/,' cez= ',/,(1x,1p5e11.3))
c     write(60,3350) cie
c3350 format(/,' cie= ',/,(1x,1p5e11.3))
c     write(60,3360) cee
c3360 format(/,' cee= ',/,(1x,1p5e11.3))

cBH080118      call second(cpua)
c
      nsp1=ns+1
      do 10 i=1,ns
        do 11 j=1,ns
          q(i,j)=0.
 11     continue
 10   continue
c
c--   detailed-balance factors (note that en's are positive):
      do 15 i=1,ns
        dge(i)=dg(i)*exp(en(i)/te)
        dgi(i)=dg(i)*exp(en(i)/ti)
 15   continue
c
c--   rates due to collisions with electrons
c     radiation rates       lorentz rates:
c
      do 20 j=1,ns
        q(j,j)=
     *    -cie(j)*dene
     *    -al(j)
        do 21 jp=j+1,nsp1
          q(j,j)=q(j,j)
     *      -cee(j,jp)*dene
 21     continue
        do 22 jp=1,j-1
          q(j,j)=q(j,j)
     *      -(dge(jp)/dge(j))*cee(jp,j)*dene
     *      -ar(j,jp)
 22     continue
        do 23 i=j+1,ns
          q(i,j)=q(i,j)
     *      +cee(j,i)*dene
 23     continue
        do 24 i=1,j-1
          q(i,j)=q(i,j)
     *      +(dge(i)/dge(j))*cee(i,j)*dene
     *      +ar(j,i)
 24     continue
 20   continue
c
c--   add rates due to collisions with ions:
      do 30 ki=1,numi
        do 301 j=1,ns
          q(j,j)=q(j,j)
     *      -cii(j,ki)*deni(ki)
          do 31 jp=j+1,nsp1
            q(j,j)=q(j,j)
     *        -cei(j,jp,ki)*deni(ki)
 31       continue
          do 32 jp=1,j-1
            q(j,j)=q(j,j)
     *        -(dgi(jp)/dgi(j))*cei(jp,j,ki)*deni(ki)
 32       continue
          do 33 i=j+1,ns
            q(i,j)=q(i,j)
     *        +cei(j,i,ki)*deni(ki)
 33       continue
          do 34 i=1,j-1
            q(i,j)=q(i,j)
     *        +(dgi(i)/dgi(j))*cei(i,j,ki)*deni(ki)
 34       continue
 301    continue
 30   continue
c
c--   add rates due to collisions with impurities:
      do 40 kz=1,numz
        do 401 j=1,ns
          q(j,j)=q(j,j)
     *      -ciz(j,kz)*denz(kz)
          do 41 jp=j+1,nsp1
            q(j,j)=q(j,j)
     *        -cez(j,jp,kz)*denz(kz)
 41       continue
          do 42 jp=1,j-1
            q(j,j)=q(j,j)
     *        -(dgi(jp)/dgi(j))*cez(jp,j,kz)*denz(kz)
 42       continue
          do 43 i=j+1,ns
            q(i,j)=q(i,j)
     *        +cez(j,i,kz)*denz(kz)
 43       continue
          do 44 i=1,j-1
            q(i,j)=q(i,j)
     *        +(dgi(i)/dgi(j))*cez(i,j,kz)*denz(kz)
 44       continue
 401    continue
 40   continue
c
cBH080118      call second(cpub)
      cpu6 = cpu6 + cpub - cpua
c
      return
      end subroutine matri
c
c
      subroutine newgrid(xold,yold,nold,xnew,ynew,nnew)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c     ---------------------------------------------------------------
c     convert to new grid
c     ---------------------------------------------------------------
      dimension xold(nold),yold(nold),xnew(nnew),ynew(nnew)
      data tol/1.0D-20/
c
      iold=1
      inew=1
      sex=(yold(nold)-yold(nold-1))/(xold(nold)-xold(nold-1))
c     ---------------------------------------------------------------
c     interpolate to new grid that is compute ynew(xnew(inew))
c     ---------------------------------------------------------------
 2000 if(inew.gt.nnew) return
 2100 if(iold.gt.nold) go to 2200
      if(xnew(inew).gt.xold(1)) go to 2105
      ynew(inew)=yold(1)
      inew=inew+1
      go to 2000
 2105 continue
      del=reldif(xold(iold),xnew(inew))
      if(del.gt.tol) go to 2110
      ynew(inew)=yold(iold)
      inew=inew+1
      go to 2000
 2110 if(xold(iold).gt.xnew(inew)) go to 2120
      iold=iold+1
      go to 2100
 2120 ipm1=iold-1
      s=(yold(iold)-yold(ipm1))/(xold(iold)-xold(ipm1))
      ynew(inew)=yold(ipm1)+(xnew(inew)-xold(ipm1))*s
      inew=inew+1
      go to 2000
 2200 ynew(inew)=yold(nold)+sex*(xnew(inew)-xold(nold))
      if(ynew(inew).lt. 0.0) ynew(inew)=0.0
      inew=inew+1
      go to 2000
      end
c jakub urban 110708: changed to integer function
c      real*8 function nfhx(i)
      integer function nfhx(i)
      implicit integer (i-n), real*8 (a-h,o-z)
c*****************************************************************
      nfhx=i-1
      if(i.le.2)nfhx=i
      return
      end
c
c
      subroutine nubplt
      implicit integer (i-n), real*8 (a-h,o-z)
c
c     record debug information
c
c     ONETWO DIVERGENCE
      return
 8100 format(i6)
 8120 format(6e12.5)
 8000 format('****continue****')
 8005 format(5e12.3,i10)
 8010 format(6e12.3)
 8020 format(i6)
      end
c      
c
      subroutine pfit(p,x,y,xv,yv,nx,f,dfdx,dfdy)
      !YuP[2021-01-12] Commented lines related to output dfdx,dfdy.
      ! All calls to this subroutine do not need such output.
      implicit none ! integer (i-n), real*8 (a-h,o-z)
cyup/was      dimension p(nx,*),x(*),y(*)
      real*8 p(4,4),x(4),y(4) !YuP[2018-01-24] Only 4 points are used
      integer nx
      real*8 xv,yv
      real*8 cx(4),cy(4),cxp(4),cyp(4)
      real*8 c2xp(4),c2yp(4)
      real*8 a1,a2,a3,a4, b1,b2,b3,b4
      real*8 f, dfdx, dfdy, d2fdx2, d2fdy2
      integer i,j
c
c-----------------------------------------------------------------------
c
c     bi-quadratic interpolator
c
c     input
c     1.  p     - effectivly a 4x4 matrix of function values
c     but for generality the first dimension of
c     p is nx.
c     2.  x     - associates a grid to the first dimension of p
c     3.  y     - associates a grid to the second dimension of p
c     4.  xv    - location at which bi-quadratic function is evaluated
c     5.  yv    - location at which bi-quadratic function is evaluated
c     6.  nx    - first dimension of p in calling program
c
c     output
c     1.  f     - interpolated value of p at (xv,yv)
c     2.  dfdx  - interpolated value of x-partial of p at (xv,yv)
c     3.  dfdy  - interpolated value of y-partial of p at (xv,yv)
c
c-----------------------------------------------------------------------
c
c
      cx(1:4)=0.d0
      cy(1:4)=0.d0
      cxp(1:4)=0.d0
      cyp(1:4)=0.d0
      c2xp(1:4)=0.d0
      c2yp(1:4)=0.d0
      f=0.d0
      dfdx=0.d0
      dfdy=0.d0
      d2fdx2=0.d0
      d2fdy2=0.d0
      
      a1=(x(1)-x(2))*(x(1)-x(3))*(x(1)-x(4))
      a2=(x(2)-x(1))*(x(2)-x(3))*(x(2)-x(4))
      a3=(x(3)-x(1))*(x(3)-x(2))*(x(3)-x(4))
      a4=(x(4)-x(1))*(x(4)-x(2))*(x(4)-x(3))
      if(a1.eq.0.d0)write(*,*)'pfit: a1=0'
      if(a2.eq.0.d0)write(*,*)'pfit: a2=0'
      if(a3.eq.0.d0)write(*,*)'pfit: a3=0'
      if(a4.eq.0.d0)write(*,*)'pfit: a4=0'
      cx(1)=(xv-x(2))*(xv-x(3))*(xv-x(4))/a1
      cx(2)=(xv-x(1))*(xv-x(3))*(xv-x(4))/a2
      cx(3)=(xv-x(1))*(xv-x(2))*(xv-x(4))/a3
      cx(4)=(xv-x(1))*(xv-x(2))*(xv-x(3))/a4
      !write(*,*)'cx=',cx
      
!      cxp(1)=((xv-x(3))*(xv-x(4))
!     2  +      (xv-x(2))*(xv-x(3))
!     3  +      (xv-x(2))*(xv-x(4)))/a1
!      cxp(2)=((xv-x(3))*(xv-x(4))
!     2  +      (xv-x(1))*(xv-x(3))
!     3  +      (xv-x(1))*(xv-x(4)))/a2
!      cxp(3)=((xv-x(1))*(xv-x(2))
!     2  +      (xv-x(1))*(xv-x(4))
!     3  +      (xv-x(2))*(xv-x(4)))/a3
!      cxp(4)=((xv-x(1))*(xv-x(2))
!     2  +      (xv-x(1))*(xv-x(3))
!     3  +      (xv-x(2))*(xv-x(3)))/a4
!      !write(*,*)'cxp=',cxp
!      c2xp(1)=2.d0*(3.d0*xv-x(2)-x(3)-x(4))/a1
!      c2xp(2)=2.d0*(3.d0*xv-x(1)-x(3)-x(4))/a2
!      c2xp(3)=2.d0*(3.d0*xv-x(1)-x(2)-x(4))/a3
!      c2xp(4)=2.d0*(3.d0*xv-x(1)-x(2)-x(3))/a4
!      !write(*,*)'c2xp=',c2xp

      b1=(y(1)-y(2))*(y(1)-y(3))*(y(1)-y(4))
      b2=(y(2)-y(1))*(y(2)-y(3))*(y(2)-y(4))
      b3=(y(3)-y(1))*(y(3)-y(2))*(y(3)-y(4))
      b4=(y(4)-y(1))*(y(4)-y(2))*(y(4)-y(3))
      if(b1.eq.0.d0)write(*,*)'pfit: b1=0'
      if(b2.eq.0.d0)write(*,*)'pfit: b2=0'
      if(b3.eq.0.d0)write(*,*)'pfit: b3=0'
      if(b4.eq.0.d0)write(*,*)'pfit: b4=0'
      cy(1)=(yv-y(2))*(yv-y(3))*(yv-y(4))/b1
      cy(2)=(yv-y(1))*(yv-y(3))*(yv-y(4))/b2
      cy(3)=(yv-y(1))*(yv-y(2))*(yv-y(4))/b3
      cy(4)=(yv-y(1))*(yv-y(2))*(yv-y(3))/b4
      !write(*,*)'cy=',cy

!      cyp(1)=((yv-y(3))*(yv-y(4))
!     2  +      (yv-y(2))*(yv-y(3))
!     3  +      (yv-y(2))*(yv-y(4)))/b1
!      cyp(2)=((yv-y(3))*(yv-y(4))
!     2  +      (yv-y(1))*(yv-y(3))
!     3  +      (yv-y(1))*(yv-y(4)))/b2
!      cyp(3)=((yv-y(1))*(yv-y(2))
!     2  +      (yv-y(1))*(yv-y(4))
!     3  +      (yv-y(2))*(yv-y(4)))/b3
!      cyp(4)=((yv-y(1))*(yv-y(2))
!     2  +      (yv-y(1))*(yv-y(3))
!     3  +      (yv-y(2))*(yv-y(3)))/b4
!      !write(*,*)'cyp=',cyp
!     
!      c2yp(1)=2.d0*(3.d0*yv-y(2)-y(3)-y(4))/b1
!      c2yp(2)=2.d0*(3.d0*yv-y(1)-y(3)-y(4))/b2
!      c2yp(3)=2.d0*(3.d0*yv-y(1)-y(2)-y(4))/b3
!      c2yp(4)=2.d0*(3.d0*yv-y(1)-y(2)-y(3))/b4
!      !write(*,*)'c2yp=',c2yp

      do 10 i=1,4
        do 11 j=1,4
          f=f+cx(i)*cy(j)*p(i,j)
          !dfdx=dfdx+cxp(i)*cy(j)*p(i,j)
          !dfdy=dfdy+cx(i)*cyp(j)*p(i,j)
          !d2fdx2=d2fdx2+c2xp(i)*cy(j)*p(i,j)
          !d2fdy2=d2fdy2+c2yp(j)*cx(i)*p(i,j)
 11     continue
 10   continue
      !write(*,*)'pfit: exit'
      !pause
      !if (nx.gt.0) return
      !if(abs(d2fdx2).le.1.d-9)write(*,*)'pfit: d2fdx2=0, dfdx=',dfdx
      !if(abs(d2fdy2).le.1.d-9)write(*,*)'pfit: d2fdy2=0, dfdy=',dfdy
      !dfdx=dfdx/d2fdx2
      !dfdy=dfdy/d2fdy2

      return
      end subroutine pfit
c
c
      subroutine polfit(ideg,npnts,x,y,coeff,ier)
      implicit integer (i-n), real*8 (a-h,o-z)
      dimension x(101),y(101),coeff(7),sums(203)
      dimension a(49),b(7),wkarea(7)
      ier=0
      idegp1=ideg+1
      if(ideg.ge.(npnts+1)) ier=1
      if(npnts.gt.101) ier=1
      if(ier.eq.1) return
c---------------------------------------------------------------------
      nsums=2*ideg
      do 2400 i=0,nsums
        s=0.0
        do 2200 j=1,npnts
          s=s+x(j)**i
 2200   continue
        sums(i+1)=s
 2400 continue
      k=0
      do 2800 i=1,idegp1
        jstart=i
        jend=i+idegp1-1
        do 2600 j=jstart,jend
          k=k+1
          a(k)=sums(j)
 2600   continue
 2800 continue
      do 3200 i=0,ideg
        s=0.0
        do 3000 j=1,npnts
          s=s+y(j)*(x(j)**i)
 3000   continue
        b(i+1)=s
 3200 continue
c---------------------------------------------------------------
c     solve set of simultaneous equations
c---------------------------------------------------------------
      call leqt1fl(a,1,idegp1,idegp1,b,0,wkarea,ier)
      do 3400 i=1,idegp1
        coeff(i)=b(i)
 3400 continue
      return
      end

      real*8 function polval(xval,coeff,ideg)
      implicit integer (i-n), real*8 (a-h,o-z)
      dimension coeff(*)
      yval=coeff(ideg+1)
      do 2000 i=ideg+1,2,-1
        yval=yval*xval+coeff(i-1)
 2000 continue
      polval=yval
      return
      end
c
c
      subroutine postnub
      implicit integer (i-n), real*8 (a-h,o-z)
c----------------------------------------------------------------
c     ONETWO DIVERGENCE

      return
      end
c
c
      subroutine prenub
      implicit integer (i-n), real*8 (a-h,o-z)
c
c     this subroutine calculates certain flux surface information needed
c     by freya
c     ONETWO DIVERGENCE

c
      return
      end
c
c
      real*8 function ranorm(iseed)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c     generate one normal (0,1) pseudo random number using a method
c     based on central limit theorem and power residue method
c     output becomes truly normal as k (below) goes to infinity.
c     general formula for the deviate is
c     y=(sum of x(i),i=1,k) -k/2. )/ sqrt(k/12.)
c     where x(i) are are uniformally distributed on 0,1.
c     method is borrowed thru ibm. they use k=12.
c
c     drand(iflag) is [0.,1.] random number; iflag is real*4.
c     iflag=0 gives next number in sequence.  See Absoft f77.
c     For Cray f90: Use ranf
c     Portable: (from GA portlib.f, 000114, bh) Use RANDOM_my(seed0)
      parameter (k=12,rtkd12=1.0000000)

c      integer iflag
c      iflag=0
      data       seed0 /0.0d0/  !I.E. Continue with intialized random_my

c     rtkd12=sqrt(k/12.)
c
      a=0.
      do 10 i=1,k
c     y=ranf(iseed)
c990131        y=ranf()
c000114*Not portable*         y=drand(iflag)
      y=RANDOM_my(seed0)
 10   a=a+y
      ranorm=(a-0.5*k)/rtkd12
      return
      end function ranorm
      
      
      real*8 function reldif(x1,x2)
      implicit integer (i-n), real*8 (a-h,o-z)
      del=abs(x1-x2)
      xabs=abs(x1)
      if(xabs.gt.1.0D-20) del=del/xabs
      reldif=del
      return
      end function reldif
c
c
      subroutine revers(x,work,n)
      implicit integer (i-n), real*8 (a-h,o-z)
      dimension work(n),x(n)
c
      do 2100 i=1,n
        work(i)=x(i)
 2100 continue
      k=0
      do 2200 i=n,1,-1
        k=k+1
        x(k)=work(i)
 2200 continue
      return
      end
c
c
      subroutine rotate(naptr, iatype, aheigh, awidth, alen, bhofset,
     *  bvofset, cangv, cangh, ib, isourc,
     *  costp, sintp, costpp, sintpp, blenp,
     *  nsourc, sangv, sangh, rpivot, zpivot,
     *  mlost, x0, y0, z0, vx0, vy0, vz0)
      implicit integer (i-n), real*8 (a-h,o-z)
ccc   
c     this subroutine advances a particle from source to the pivot point,
c     and transforms coordinates.
c
c     translation(s):  Particle to each aperature aligned along the souce
c     axis.  Return with mlost=1 if particle hits aperature.
c     rotations:       about y-axis so z-axis is aligned vertically;
c     about new z-axis so x-axis parallel to beamline axis.
c     translation:     coordinates to intersection of beamline axis with
c     source exit plane.
c     translation:     particles to each of beamline axis centered aperatures,
c     checking for mlost=1 condition.
c     translation:     particle and then coordinate axis to pivot point.
c     rotation:        x-axis through pivot point and torus center.
c     translation:     origin to torus center.
c
c     ONETWO DIVERGENCE
      parameter(nap=10)
      dimension iatype(nap,*), aheigh(nap,*), awidth(nap,*),
     *  alen(nap,*), bhofset(*), bvofset(*), blenp(*), cangv(*),
     *  cangh(*), sangv(*), sangh(*), rpivot(*), zpivot(*),
     *  costp(*), sintp(*), costpp(*),  sintpp(*)
ccc
      zero=0.d0

cBH110314:  Define iap for nsourc=1 case
      if(nsourc.eq.1)  then
         iap=0
         go to 19
      endif

c     Move particle to each of source-axis centered aperatures, and
c     test for particle passage.
      alen1=0.0
      alen2=0.0
c
      do 10  i=1,naptr
c     iatype.le.4 are source centered aperatures.
        if(iatype(i,ib).gt.4)  go to 11
        alen1=alen(i,ib)-alen2
        alen2=alen(i,ib)
        tpvt=-alen1/vx0
        x0=x0+vx0*tpvt
        y0=y0+vy0*tpvt
        z0=z0+vz0*tpvt
c
        mlost=0
        go to (12,13,14,15),  iatype(i,ib)
 12     if((y0**2+z0**2).le.(0.5*awidth(i,ib))**2) go to 10
        go to 16
 13     if(abs(y0).le.0.5*awidth(i,ib).and.
     1    abs(z0).le.0.5*aheigh(i,ib))  go to 10
        go to 16
 14     if(abs(z0).le.0.5*aheigh(i,ib))  go to 10
        go to 16
 15     if(abs(y0).le.0.5*awidth(i,ib))  go to 10
 16     mlost=1
        return
 10   iap=i
 11   continue
c
c     Source center is taken to lie at vertical distance
c     bvofset and horiz. distance bhofset from beam line axis.
c     Source centerline intesects beam line axis at distance
c     bleni from source.  Coords. are changed from source
c     coordinates to those with x-axis along beamline axis
c     directed in +R direction, z in the vertical direction, and
c     origin in source exit plane.
c
c     Rotate about y-axis of source coordinate system so x-axis
c     is vertical and in exit plane defined by beam axis
c     and source centers.
c      write(*,*) 'isourc = ',isourc
c      write(*,*) 'costp,sintp = ',costp(ib),sintp(ib)
c      write(*,*) 'bhofset,bvofset = ',bhofset(ib),bvofset(ib)
c      write(*,*) 'cangh,cangv = ', cangh(ib),cangv(ib)
c      write(*,*) 'sangh,sangv = ', sangh(ib),sangv(ib)
      temp=x0*costp(ib)-isourc*z0*sintp(ib)
      z0=isourc*x0*sintp(ib)+z0*costp(ib)
      x0=temp
      temp=vx0*costp(ib)-isourc*vz0*sintp(ib)
      vz0=isourc*vx0*sintp(ib)+vz0*costp(ib)
      vx0=temp
c
c     Rotate about z-axis of source system so y-axis lies
c     in source exit plane.
      temp=x0*costpp(ib)-isourc*y0*sintpp(ib)
      y0=isourc*x0*sintpp(ib)+y0*costpp(ib)
      x0=temp
      temp=vx0*costpp(ib)-isourc*vy0*sintpp(ib)
      vy0=isourc*vx0*sintpp(ib)+vy0*costpp(ib)
      vx0=temp
c
c     Translate coordinate axes to beamline axis.
      x0=x0
      y0=y0+isourc*bhofset(ib)
      z0=z0+isourc*bvofset(ib)
c
c
c     Translate particle to beamline centered aperatures and set
c     mlost=1 if particle lost.
 19   continue
      if(iap.eq.naptr)  go to 29
      alen1=0.0
      alen2=abs(x0)
      do 20  i=iap+1,naptr
        alen1=alen(i,ib)-alen2
        if(alen1.lt.0.)  stop 'zfreya: rotate'
        alen2=alen(i,ib)
        tpvt=-alen1/vx0
        x0=x0+vx0*tpvt
        y0=y0+vy0*tpvt
        z0=z0+vz0*tpvt
        mlost=0
        go to (22,23,24,25,26),  iatype(i,ib)-4
 22     if((y0**2+z0**2).le.(0.5*awidth(i,ib))**2) go to 20
        go to 27
 23     if(abs(y0).le.0.5*awidth(i,ib).and.
     1    abs(z0).le.0.5*aheigh(i,ib))  go to 20
        go to 27
 24     if(abs(z0).le.0.5*aheigh(i,ib))  go to 20
        go to 27
 25     if(abs(y0).le.0.5*awidth(i,ib))  go to 20
        go to 27
c     DIII-D  special case:
 26     if(abs(z0).gt.24.75)  go to 27
        if(abs(y0).le.13.45)  go to 20
        if(abs(y0).gt.21.7)  go to 27
        if(abs(z0).gt.24.75*((abs(y0)-13.45)/(-8.25)+1))  go to 27
        go to 20
 27     mlost=1
        return
 20   continue
 29   continue
c
c     translate particle and then coordinate axes to pivot point
c
      tpvt=-(blenp(ib)+x0)/vx0
      x0=0.0
      y0=y0+vy0*tpvt
      z0=z0+vz0*tpvt
ccc   
      if(sangv(ib).eq.zero) go to 30
      zcos=cangv(ib)
      zsin=sangv(ib)
      temp=x0*zcos+z0*zsin
      z0=-x0*zsin+z0*zcos
      x0=temp
      temp=vx0*zcos+vz0*zsin
      vz0=-vx0*zsin+vz0*zcos
      vx0=temp
 30   continue
ccc   
      if(sangh(ib).eq.0.) go to 40
      zcos=cangh(ib)
      zsin=sangh(ib)
      temp=x0*zcos+y0*zsin
      y0=-x0*zsin+y0*zcos
      x0=temp
      temp=vx0*zcos+vy0*zsin
      vy0=-vx0*zsin+vy0*zcos
      vx0=temp
 40   continue
ccc   
      x0 = x0 + rpivot(ib)
      z0 = z0 + zpivot(ib)
ccc   
      return
      end
c
c
      subroutine scalit(xa,xb,xmin,xmax,imin,imax,ifail,maxit,toler
     1  ,pertf,trans,iprint
     2  ,c1,c2,c3,drtabi,mfm1,psival,ptab,rmajor)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c
c
c*****************************************************************
c
c**   subroutine for scalar iteration.
c     scalit is a non-subscripted simplification of vectit, a more
c     general code for iteration with vectors.
c
c**   method - an extension of the secant method.  the procedure
c     resembles newton's method.  after each iteration a linear
c     extrapolation to the root is made.  the slope used for
c     extrapolation is computed by two methods.  the first
c     method is to perturb x a small amount, giving a good
c     approximation to the tangent.  this is accurate, but it
c     does not lead to productive iterations.  the second
c     method is to do one more ordinary iteration and calculate
c     a secant.  this is more productive and sufficiently
c     accurate provided one is already quite close to the
c     solution.  a transition from method 1 to method 2 is made
c     when the test quantity (dx/x) is less than a prescribed
c     value = 'trans'.  the method is quadratically convergent
c     and works even when the uncorrected iteration is
c     divergent.  as with newton's method it is not guaranteed
c     to work unless the first guess is 'sufficiently close' to
c     the desired root.  a false guess correction is provided
c     to force the root (if one exists) to lie within a given
c     interval.
c
c**   summary of arguments
c
c     name      definition
c
c     xa      - first guess - destroyed during calculation.
c     xb      - final guess (converged value if ifail > 0).
c     xmin    - allowable lower limit to x if imin.ne.0
c     xmax    - allowable upper limit to x if imax.ne.0
c     imin    - if = 0, ignore xmin criterion.
c     imax    - if = 0, ignore xmax criterion.
c     ifail   - results indicator -
c     > 0  - converged within ifail cycles,
c     = 0  - abnormal exit,
c     = -1 - failed to converge in maxit cycles.
c     maxit   - upper limit to number of iteration cycles.
c     note - 2 local iterations per cycle.
c     maxit = 10 should be sufficient.
c     toler   - convergence criterion for ratio of  abs(dx/x).
c     pertf   - perturbation factor.  deltax = pertf*abs(x).
c     see above.  typically  0.1
c     trans   - transition value of (dx/x) for solution method.
c     typically .01
c     iprint  - if ne.0, debug print of all x iterations
c     is provided in the printout.
c**   subroutines required by scalit
c
c     fortran name        description
c
c     chek      -  checks to see if a given x is between xmax and xmin.
c
c     itorb     -  a user-supplied external subroutine that defines
c     the basic uncorrected iteration. given x, calculates
c
c**   use of the program scalit
c
c     user supplies calling program.  calling program initializes xa.
c     calling program then calls scalit.  entire iteration up to point
c     of convergence is done in scalit.  scalit returns converged value
c     of x in xb.
c     basic uncorrected iteration must also be defined by user in an
c     external subroutine named 'itorb(x,dx)'.  itorb has arguments
c     x  - the input value of a guess x (must be saved)
c     dx - the output value of a change in x
c
c     p. d. smith   5/10/74
c
c***********************************************************************
c
      dimension ptab(*)
      data icall/0/
      icall = icall + 1
      iter = 0
      test = 1.0e6
      if(iprint.ne.0) write(6,800) icall
      if(iprint.ne.0) write(6,801) iter,xa
 800  format(/' debug print for call',i4,' to scalit'/
     1  ,' x value at end of each iteration printed below',/)
 801  format(' cycle',i3,3x,e15.8)
c
c*    ** now begin iteration/extrapolation cycles.
c
      ifail = 0
 200  continue
c
c*    ** start of an iteration cycle
      iter = iter + 1
c
c*    ** check to see if maximum iteration criterion is exceeded
      if(iter.le.maxit) go to 210
      ifail = -1
      return
 210  continue
c
c*    ** o k to do another cycle.
c
      if(test.lt.trans) go to 150
c
c*    ** use perturbation method to estimate slope.
c*    ** whenever the iteration is still not close to the solution.
c
c*    ** first iteration.
      call itorb(c1,c2,c3,drtabi,mfm1,psival,ptab,rmajor,xa,dxa)
      xb = xa + dxa
      xbmag = abs(xb)
      deltax = pertf*xbmag
c
c*    ** perturb xa
      xa = xa + deltax
c
c*    ** iterate using perturbed xa.
      call itorb(c1,c2,c3,drtabi,mfm1,psival,ptab,rmajor,xa,dxb)
c
c*    ** change xa back to initial value.
      xa = xa - deltax
c
c*    ** calculate slope.
      slope = (dxb - dxa)/deltax
c
c*    ** negative inverse of slope is correction factor.
      fc = -1./slope
 150  continue
c
c*    ** now begin extrapolation/iteration cycle, assuming that the
c*    ** difference method will work.  if it doesn't work, the solution
c*    ** will be cycled back to the perturbation method.
c
c
c*    ** first extrapolate previous xa and dxa to get a value for xb.
      dxb = fc*dxa
      xb = xa + dxb
      if(iprint.ne.0) write(6,801) iter,xb
c
c*    ** check limits of xb
      call chek(xb,xmin,xmax,imin,imax,ick)
      if(ick.eq.0) go to 220
c
c*    ** overshot boundary. increase effectiveness of damping.
      fc = .5*fc
c
c*    ** start iteration over again.
      go to 200
 220  continue
c
c*    ** extrapolation was successful.  continue
c
c*    ** check convergence and return if test is passed.
      xbmag = abs(xb)
      dxmag = abs(dxb)
      if(dxmag.le.0.)    go to 240
      if(xbmag.le.0.) go to 250
      test = dxmag/xbmag
      if(test.lt.trans) go to 230
c
c*    ** solution is not sufficiently close.  must use perturbation.
c*    ** update extrapolation prior to next perturbation.
      xa = xb
 225  continue
      go to 200
c
 230  continue
      if(test.gt.toler)  go to 250
 240  continue
c
c*    ** converged
      ifail = iter
      return
 250  continue
c
c*    ** no convergence - continue with calculations.
c
c*    ** perform an iteration with xb to get dxb.
      call itorb(c1,c2,c3,drtabi,mfm1,psival,ptab,rmajor,xb,dxb)
c
c*    ** calculate new secant slope
      slope = (dxb - dxa)/(xb - xa)
c
c*    ** update xa and dxa prior to next iteration/extrapolation.
      xa = xb
      dxa = dxb
c
c*    ** calculate another correction factor.
      fc = -1./slope
c
c*    ** go back and do another iteration cycle
      go to 200
      end
c
c
      real*8 function scxif(i,er)
      implicit integer (i-n), real*8 (a-h,o-z)
c*****************************************************************
c--   estimated cross sections for chg. exchange .
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
c
      real*8 polyf
c@    real polyf
c
c
      if(i.ge.2)goto 200
c
c--   1s:
c990131      aer=alog(er)
      aer=log(er)
c
c     charge exchange.
c     below 1.e6 ev, use riviere's fit       above 1.e6 ev, use
c     power-law extrapolation.
c
      if(er .gt. 1.e6)goto 105
      siif1=(.6937D-14)*(1.-.06732*aer)**2/(1.+.1112D-14*er**3.3)
      goto 110
 105  siif1=(4.8363D-22)/((1.D-6)*er)**3.3
c
 110  scxif=siif1
      return
c
 200  continue
      ansq=(nfhx(i))**2
      xtilda = ansq*er/9260.
      scxif = 0.
      if(xtilda.lt.1.) scxif=1.465D-11*xtilda*(ansq/er)
      return
      end function scxif
c
c
      real*8 function scxzf(i,z,er)
      implicit integer (i-n), real*8 (a-h,o-z)
c*****************************************************************
c--   cross section for electron loss via collision with impurity.
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
c
      ni=nfhx(i)
      id=2
      if(ni.eq.1)id=1
c
      ansq=(dfloat(ni))**2
      xtilda = ansq*er/(31250.*z)
      scxzf = 0.
      if(xtilda.lt.1.) scxzf=1.465D-11*xtilda*(ansq*z*z/er)
      return
      end
c
c
      real*8 function sdaccf(ni,nj,z,er)
      implicit integer (i-n), real*8 (a-h,o-z)
c*****************************************************************
c--   dacc cross section.
c     janev & presnyakov, j. phys. b 13, 4233 (1980).
c
      parameter (ms=21,mc=35)
      common/b3/f(ms,mc),ar(ms+1,ms+1)
c
      omega=.5*(1./((dfloat(ni))**2)-1./((dfloat(nj))**2))
      alam=sqrt(f(ni,nj)/(2.*omega))
      beta=z*alam*omega/(er/24982.)
      sdaccf=(1.7595D-16)*(z*alam/omega)*dfhx(beta)
      return
      end
c
c
      real*8 function seef(i,j,er)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
c--   cross sections for excitation from i to j due to electron impact.
      parameter (ms=21,mc=35)
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
      common/b2/nouthx,istart,ihxbug
      common/b4/en(mc+1),dg(mc+1),ae(ms,mc),be(ms,mc),
     *  de1(ms,mc),de2(ms,mc),ge1(ms,mc),ge2(ms,mc)
c
      real*8 polyf
c@    real polyf
      real*8 ae12
      dimension ae12(8)
      real*8 ae13
      dimension ae13(9)

      data ryd/13.6/
c
c--   data for 1s to 2s (e impact):
      data emin12/10.2/, emax12/1140.5/
      data ae12/
     *  -1.5138513657d-17, 1.8231019028d-16,-1.9274654811d-16,
     *  8.9291530932d-17,-2.2108041618d-17, 3.0448025264d-18,
     *  -2.2039298214d-19, 6.5491238735d-21/
c@    real ae12
c@    dimension ae12(8)
c@    data ae12/
c@    *-1.5138513657e-17, 1.8231019028e-16,-1.9274654811e-16,
c@    * 8.9291530932e-17,-2.2108041618e-17, 3.0448025264e-18,
c@    *-2.2039298214e-19, 6.5491238735e-21/
c
c--   data for 1s to 2p (e impact)
      data emin13/10.2/, emax13/747.4/
      data ae13/
     *  -2.1078372920d-14, 3.7548968459d-14,-2.8788463637d-14,
     *  1.2394689683d-14,-3.2696005067d-15, 5.4068250737d-16,
     *  -5.4759059881d-17, 3.1084887079d-18,-7.5815695055d-20/
c@    real ae13
c@    dimension ae13(9)
c@    data ae13/
c@    *-2.1078372920e-14, 3.7548968459e-14,-2.8788463637e-14,
c@    * 1.2394689683e-14,-3.2696005067e-15, 5.4068250737e-16,
c@    *-5.4759059881e-17, 3.1084887079e-18,-7.5815695055e-20/
c
      if(i.ge.j)goto 300
      seef=0.
c990131      aer=alog(er)
      aer=log(er)
      if((i.ge.3) .or. (j.ge.4))goto 250
      goto(210,240),i
c
 210  continue
      goto(300,220,230),j
c
c--   1s to 2s:
 220  continue
      if(er .le. emin12)return
      if(er .le. 11.1)  goto 221
      if(er .ge. emax12)goto 222
      seef=polyf(ae12,8,aer)
      goto 223
c--   linear interpolation for region just above threshold:
 221  continue
      seef=((er-10.2)/.9)*1.6297D-17
      goto 223
c--   asymptotic:
 222  continue
      seef=(5.312D-16)/er
 223  continue
      return
c
c--   1s to 2p:
 230  continue
      if(er .le. emin13)return
      if(er .ge. emax13)goto 232
      seef=polyf(ae13,9,aer)
      goto 233
c--   asymptotic:
 232  continue
      seef=(2.65571D-15/er)*(aer-2.120681)
 233  continue
      return
c
c--   2s to 2p:
 240  continue
c990131      seef=(8.617D-15/er)*alog(1.14e4*er)
      seef=(8.617D-15/er)*log(1.14e4*er)
      return
c
 250  continue
      ni=nfhx(i)
      nj=j-1
c--   from vriens & smeets (eq.(14)):
c990131   seef=(1.75947D-16*ryd)*(ae(ni,nj)*alog(.5*er/ryd+de1(ni,nj))
      seef=(1.75947D-16*ryd)*(ae(ni,nj)*log(.5*er/ryd+de1(ni,nj))
     *  +be(ni,nj))/(er+ge1(ni,nj))
c990131      seef=amax1(seef,0.)
      zero=0.d0
      seef=max(seef,zero)
      id=2
      if(ni.eq.1)id=1
      return
c
 300  continue
      ihxbug = 6
      if(nouthx.gt.0) write(nouthx,3939) i,j
 3939 format(' ??? seef error: i4,j are in error       i4=',i3,
     &  /,'   j = ',i3)
      return
      end function seef
c
c
      subroutine setrz(ndim,rmin,rmax,zmin,zmax,dr,dz,nr,nz)
      implicit integer (i-n), real*8 (a-h,o-z)
c-----------------------------------------------------------------------
c     this subroutine sets up (r,z) grid parameters for freya when the
c     optional two-dimensional deposition calculation is used.  default
c     parameters correspond to original eqdisk grid
c-----------------------------------------------------------------------
      dimension ndim(2)
      nr=ndim(1)
      nz=ndim(2)
      dr=(rmax-rmin)/(nr-1)
      dz=(zmax-zmin)/(nz-1)
      return
      end
c
c
      real*8 function sezf(i,j,z,er)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
c--   cross sections for excitation from i to j due to ion impact.
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
      common/b2/nouthx,istart,ihxbug
c
      real*8 polyf
c@    real polyf
      real*8 ai12
      dimension ai12(11)
c@    real ai12
      real*8 ai13
c@    real ai13
      dimension ai13(12)
c
c--   data for 1s to 2s (p impact):
      data emin12/1.e3/, emax12/1.e6/
      data ai12/
     *  -3.3812171941d+05, 3.5976726159d+05,-1.7037301759d+05,
     *  4.7282644608d+04,-8.5167211591d+03, 1.0405267504d+03,
     *  -8.7343818297d+01, 4.9754307346d+00,-1.8412279745d-01,
     *  3.9984716057d-03,-3.8707719188d-05/
c
c--   data for 1s to 2p (p impact):
      data emin13/1.e3/, emax13/1.e6/
      data ai13/
     *  -1.3490069287d+06, 1.4573274888d+06,-7.0815407013d+05,
     *  2.0432442357d+05,-3.8900004212d+04, 5.1319758650d+03,
     *  -4.7884757131d+02, 3.1608287540d+01,-1.4469255104d+00,
     *  4.3759824250d-02,-7.8716881911d-04, 6.3824434435d-06/
c
      if(i.ge.j) then
        ihxbug = 4
        if(nouthx.gt.0) write(nouthx,3939) i,j
 3939   format(' ??? sezf input error: i4=',i3,'        j=',i3)
        return
      endif

      if(z .gt. 1.01)goto 300
c990131      aer=alog(er)
      aer=log(er)
      if((i.ge.3) .or. (j.ge.4))goto 250
      goto(210,240),i
c
 210  continue
      goto(400,220,230),j
c
c--   1s to 2s:
 220  continue
      if(er .ge. emin12)goto 221
      sezf=2.8875D-18*(er/emin12)**.7093
      goto 223
 221  continue
      if(er .ge. emax12)goto 222
      sezf=exp(polyf(ai12,11,aer))
c@    sezf=exp(polyf(ai12,11,aer))
      goto 223
 222  continue
      sezf=1.9564D-12/er
 223  continue
      sezfsv = sezf
      return
c
c--   1s to 2p:
 230  continue
      if(er .ge. emin13)goto 231
      sezf=1.4053D-17*(er/emin13)**.95695
      goto 233
 231  continue
      if(er .ge. emax13)goto 232
c990131      sezf=exp(polyf(ai13,12,aer))
c@    sezf=exp(polyf(ai13,12,aer))
cBH120202      twelve=12
cBH120202      sezf=exp(polyf(ai13,twelve,aer))
      n12=12
      sezf=exp(polyf(ai13,n12,aer))
      goto 233
 232  continue
c990131      sezf=(6.7085D-13/er)*alog(5701.79*er)
      sezf=(6.7085D-13/er)*log(5701.79*er)
 233  continue
      sezfsv = sezf
      return
c
c--   2s to 2p:
c990131 240  sezf=(1.584D-10/er)*alog(.62*er)
 240  sezf=(1.584D-10/er)*log(.62*er)
c990131      sezf=amax1(sezf,0.)
      zero=0.d0
      sezf=max(sezf,zero)
      sezfsv = sezf
      return
c
 250  continue
      ni=nfhx(i)
      id=2
      if(ni.eq.1)id=1
      nj=j-1
      sezf=sdaccf(ni,nj,one,er)
      sezfsv = sezf
      return
c
c--   impurity scattering:
 300  continue
      if((i.eq.1).and.(j.eq.2))goto 312
      if((i.eq.1).and.(j.eq.3))goto 313
      sezf=0.
      sezfsv = sezf
      if((i.eq.2).and.(j.eq.3))return
      ni=nfhx(i)
      nj=j-1
      id=2
      if(ni.eq.1)id=1
      sezf=sdaccf(ni,nj,z,er)
      sezfsv = sezf
      return
 312  continue
      sezf=.25*sdaccf(1,2,z,er)
      sezfsv = sezf
      return
 313  continue
      sezf=.75*sdaccf(1,2,z,er)
      sezfsv = sezf
      return
c
 400  ihxbug = 5
      if(nouthx.gt.0) write(nouthx,3941)
c     ONETWO DIVERGENCE
 3941 format(" ??? sezf error: state j is 0")
      return
      end
c
c
      real*8 function sief(i,er)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
c--   cross sections for ionization due to e impact.
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
      real*8 polyf
c@    real polyf
      real*8 ae1
      dimension ae1(7)
c@    real ae1
c@    dimension ae1(7)
      real*8 ae2
      dimension ae2(9)
c@    real ae2
c@    dimension ae2(9)
c
      data ryd/13.6/
c
c--   data for e-impact ionization of 1s:
      data emin1/13.723/, emax1/620./
      data ae1/
     *  2.3036652148d-15,-3.4298197580d-15, 1.9465132340d-15,
     *  -5.4457519508d-16, 8.0929651995d-17,-6.1527279210d-18,
     *  1.8889193736d-19/
c@    data ae1/
c@    * 2.3036652148e-15,-3.4298197580e-15, 1.9465132340e-15,
c@    *-5.4457519508e-16, 8.0929651995e-17,-6.1527279210e-18,
c@    * 1.8889193736e-19/
c
c--   data for e-impact ionization of 2s:
      data emin2/3.4/, emax2/386./
      data ae2/
     *  8.4162370468d-15,-2.3545910382d-14, 2.4728342689d-14,
     *  -1.2718639429d-14, 3.6382722533d-15,-6.0044378263d-16,
     *  5.5114285405d-17,-2.4253587346d-18, 3.0573876445d-20/
c@    data ae2/
c@    * 8.4162370468e-15,-2.3545910382e-14, 2.4728342689e-14,
c@    *-1.2718639429e-14, 3.6382722533e-15,-6.0044378263e-16,
c@    * 5.5114285405e-17,-2.4253587346e-18, 3.0573876445e-20/
c
c--   be across section (ansq=n**2, ery=e/ryd):
      sbea(ansq,ery)=
     *  (3.519D-16)*(5.*ansq/3.-1./ery-2./(3.*ansq*ery*ery))
     *  /(ery+3.25/ansq)
c
      sief=0.
      if(i.ge.3)goto 130
c990131      aer=alog(er)
      aer=log(er)
      goto(110,120),i
c
c--   1s:
 110  continue
      if(er .le. emin1)return
      if(er .ge. emax1)goto 112
      sief=polyf(ae1,7,aer)
      goto 113
 112  continue
      sief=(1.3563D-15/er)*(aer+1.823647)
 113  continue
      return
c
c--   2s:
 120  continue
      if(er .le. emin2)return
      if(er .ge. emax2)goto 122
      sief=polyf(ae2,9,aer)
      goto 123
 122  continue
      sief=(8.195137D-15/er)*(aer-.9445204)
 123  continue
      return
c
c--   2p and higher:
 130  continue
      ansq=(dfloat(i-1))**2
      if(er .le. ryd/ansq)return
      sief=sbea(ansq,er/ryd)
      return
      end function sief
c
c
      real*8 function siif(i,er)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c*****************************************************************
c--   cross sections for chg. exchange and ion-impact ionization.
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
c
      real*8 polyf
c@    real polyf
c
c--   data for ion-impact ionization of h(1s):
      real*8 ai1
      dimension ai1(6)
      real*8 afj
      dimension afj(7)
c@    real afj
c@    dimension afj(7)
      data ai1/
     *  -4.410635d+02, 1.846170d+02,-3.429509d+01, 3.217263d+00,
     *  -1.512004d-01, 2.825854d-03/
c@    real ai1
c@    dimension ai1(6)
c@    data ai1/
c@    *-4.410635e+02, 1.846170e+02,-3.429509e+01, 3.217263e+00,
c@    *-1.512004e-01, 2.825854e-03/
c
c--   data for ion-impact ionization of h(1s) (freeman & jones):
      data afj/
     *  -.4203309d+2, .3557321d+1,-.1045134d+1, .3139238d+0,
     *  -.7454475d-1,.8459113d-2,-.3495444d-3/
c@    data afj/
c@    *-.4203309e+2, .3557321e+1,-.1045134e+1, .3139238e+0,
c@    * -.7454475e-1,.8459113e-2,-.3495444e-3/
c
      if(i.ge.2)goto 200
c
c--   1s:
c990131      aer=alog(er)
      aer=log(er)
c
c     charge exchange.
c     below 1.e6 ev, use riviere's fit       above 1.e6 ev, use
c     power-law extrapolation.
c
      if(er .gt. 1.e6)goto 105
      siif1=(.6937D-14)*(1.-.06732*aer)**2/(1.+.1112D-14*er**3.3)
      goto 110
 105  siif1=(4.8363D-22)/((1.D-6)*er)**3.3
c
c     ion-impact ionization.
c     at low energies or high energies, use riviere's fits
c     at intermediate energies, use fit to rkj's curve.
c
 110  continue
c
c     freeman and jones option:
c     siif2=exp(polyf(afj,7,alog(.001*er)))
c     goto 120
c     111 continue
c
c990131
      fnum1=807.4
      fnum2=154400.
      if(er .lt. fnum1)goto 113
      if(er .gt. fnum2)goto 114
      siif2=exp(polyf(ai1,6,aer))
c@    siif2=exp(polyf(ai1,6,aer))
      goto 120
 113  continue
      siif2=exp(-80.206+8.156*aer-.3784*aer*aer)
      goto 120
 114  siif2=(1.56346D-12/er)*(aer-1.792160)
 120  continue
      siif=siif1+siif2
      return
c
 200  continue
      ansq=(nfhx(i))**2
      siif=(1.465D-11)*(ansq/er)*(1.-exp(-ansq*er/9260.))
      return
      end function siif
c
c
      real*8 function sizf(i,z,er)
      implicit integer (i-n), real*8 (a-h,o-z)
c*****************************************************************
c--   cross section for electron loss via collision with impurity.
      common/b1/kdene,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl
c
      ni=nfhx(i)
      id=2
      if(ni.eq.1)id=1
c
      ansq=(dfloat(ni))**2
      sizf=(1.465D-11)*(ansq*z*z/er)*(1.-exp(-ansq*er/(31250.*z)))
      return
      end
c
c
      subroutine slow1(atw,atwf,ene,en,enn,ezero,ibcur,ibcx,
     *  ifirst,kj,nj,nion,te,zsq,zsqf,
     *  bke,bki,ecrit,emzrat,encap,ge,gi,taus,fionx)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c     ------------------------------------------------------------------
c     This subroutine evaluates the transfer functions N,Ge,Gi,Ke, and
c     Ki defined by Callen et al., IAEA Tokyo, Vol. I, 645 (1974) to
c     describe fast ion slowing down.  Ge and Gi are the fractions of
c     the fast ion energy transferred to the electrons and thermal ions,
c     respectively.  The remaining energy fraction is assumed lost due
c     to secondary charge exchange, i.e., charge exchange between fast
c     ions and thermal neutrals.  Similarly, Ke and Ki are the fractions
c     of the fast ion parallel momentum transferred to electrons and
c     thermal ions.  Moreover, N, Ge, and Ke are proportional to the
c     slowing-down times for particles, energy, and parallel momentum
c     and are used in subroutine slow2.  The fast ions may be beam ions
c     from neutral beam heating or alpha particles from fusion.
c     --------------------------------------------------------------------
      dimension atw(*), ene(*), en(kj,*), enn(kj,*), te(*), zsq(kj,*)
      dimension bke(*), bki(*), ecrit(*), emzrat(*), encap(*),
     *  ge(*), gi(*), taus(*)
      data pi, rot2pi / 3.14159, 2.50663 /
      data xmasse / 9.110d-28 /
      data xmassp / 1.673d-24 /
      data charg4 / 5.308d-38 /
c
c     calculate constants
c
      econst = 0.5*(4.5*pi*xmassp/xmasse)**0.333333
      tconst = atwf*xmassp/(zsqf*xmasse)
c
c     begin loop over mesh points
c
      do 100 j=1,nj
        if(ifirst.eq.0) go to 20
c
c     calculate ecrit, the "critical energy" at which fast ions transfer
c     equal energy to electrons and thermal ions; also calculate
c     emzrat = mi*<Z>/(mf*[Z])
c
        sum1 = 0.
        sum2 = 0.
        do 10 k=1,nion
          sum1 = sum1 + en(j,k)*zsq(j,k)
 10     sum2 = sum2 + en(j,k)*zsq(j,k)/atw(k)
        ecrit(j) = econst*te(j)*atwf*(sum2/ene(j))**0.666667
        emzrat(j) = sum1/(atwf*sum2)
c
c     calculate taus, the Spitzer momentum exchange time for electron-ion
c     collisions
c
        teerg = 1.6D-9*te(j)
c990131        xlam  = 24. - alog(sqrt(ene(j))/(1.e3*te(j)))
        xlam  = 24. - log(sqrt(ene(j))/(1.e3*te(j)))
        ztaue = sqrt(xmasse*teerg**3)
     *    /(1.33333*rot2pi*ene(j)*charg4*xlam)
        taus(j) = tconst*ztaue
c
c     calculate vrat = sqrt(ecrit/ezero) and taurat = taus/taucx,
c     where taucx is the charge exchange time
c
 20     vrat = sqrt(ecrit(j)/ezero)
        taurat = 0.
        if(ibcx.eq.0) go to 50
        if(atwf.gt.3.) go to 50
        taurat = taus(j) * (enn(j,1)+enn(j,2)) * cxr(ezero/atwf)
c
c     calculate encap, ge, and gi
c
 50     encap(j) = encapf(vrat,taurat)
        ge(j) = gef(vrat,taurat)
        if(ge(j).lt.0.0) ge(j) = 0.0
        gi(j) = 1. - (1.+0.5*taurat)*ge(j)
        if(gi(j).lt.0.0) gi(j) = 0.0
c     fionx allows testing of non-classical slowing down
        if(fionx.eq.0.0) go to 70
        ft=fionx*gi(j)
        if(ft.gt.ge(j)) ft=ge(j)
        gi(j)=gi(j) + ft
        ge(j)=ge(j) - ft
 70     continue
c
c     calculate bke and bki if ibcur.ne.0
c
        if(ibcur.eq.0) go to 100
        bke(j) = bkef(vrat,taurat,emzrat)
        bki(j) = 1. - (1.+taurat)*bke(j)
c
c     end loop over mesh points
c
 100  continue
      return
      end
c
c
      subroutine slow2(bke,dtt,encap,enfsav,ge,ibcur,nj,
     *  ppfsav,qfsav,sfsav,spfsav,taus,wfsav,
     *  enf,ppf,qf,sf,spf,taupf,tauppf,tauef,wf)
      implicit integer (i-n), real*8 (a-h,o-z)
c
c     This subroutine models fast ion slowing down.  The fast ions may
c     be beam ions from neutral beam heating or alpha particles from
c     fusion.
c
      dimension bke(*), encap(*), enfsav(*), ge(*), ppfsav(*), qfsav(*),
     *  sfsav(*), spfsav(*), taus(*), wfsav(*)
      dimension enf(*), ppf(*), qf(*), sf(*), spf(*), taupf(*),
     *  tauppf(*), tauef(*), wf(*)
c
c     begin loop over mesh points
c
      do 100 j=1,nj
c
c     calculate slowing down times
c
        taupf(j) = taus(j)*encap(j)
        tauppf(j) = taus(j)*bke(j)
        tauef(j) = 0.5*taus(j)*ge(j)
c
c     calculate fast ion particle density and delayed particle source
c
        enf(j) = enfsav(j)*exp(-dtt/taupf(j))
     *    + sfsav(j)*taupf(j)*(1.-exp(-dtt/taupf(j)))
        sf(j) = enf(j)/taupf(j)
c
c     calculate fast ion energy density and delayed energy source
c
        wf(j) = wfsav(j)*exp(-dtt/tauef(j))
     *    + qfsav(j)*tauef(j)*(1.-exp(-dtt/tauef(j)))
        qf(j) = wf(j)/tauef(j)
c
c     calculate density and delayed source of fast ion parallel momentum
c
        if(ibcur.eq.0) go to 100
        ppf(j) = ppfsav(j)*exp(-dtt/tauppf(j))
     *    + spfsav(j)*tauppf(j)*(1.-exp(-dtt/tauppf(j)))
        spf(j) = ppf(j)/tauppf(j)
c
c     end loop over mesh points
c
 100  continue
      return
      end
c
c
cBH990903      subroutine smooth(x,y,n,ideg,nuse,nupdate)
      subroutine smooth1(x,y,n,ideg,nuse,nupdate)
      implicit integer (i-n), real*8 (a-h,o-z)
      dimension x(n),y(n),coeff(8)
c
      do 1800 i=1,8
        coeff(i)=0.0
 1800 continue
      big=-1.0
      do 2000 i=1,n
        if(y(i).gt.big) big=y(i)
 2000 continue
      small=big*.001
      i1=0
 2100 i1=i1+1
      i2=i1+nuse-1
      if(i2.le.n) go to 2120
c-----------------------------------------------------
c     use last set of coefficients to finish with
c-----------------------------------------------------
      do 2110 i=i1,n
        y(i)=polval(x(i),coeff,ideg)
 2110 continue
      go to 2150
c-----------------------------------------------------
 2120 if(y(i1).le.small .and. y(i1+1).le.small) go to 2100
      call polfit(ideg,nuse,x(i1),y(i1),coeff,ier)
      do 2140 i=i1,i1+nupdate-1
        y(i)=polval(x(i),coeff,ideg)
 2140 continue
      go to 2100
 2150 do 2160 i=1,n
        if(y(i).lt. 0.0) y(i)=0.0
 2160 continue
      return
      end
c
c
      subroutine solveq(ndim, n, a, b, ipvt)
      implicit integer (i-n), real*8 (a-h,o-z)
c
      integer ndim, n, ipvt(n)
      real*8 a(ndim,n),b(n)
c
c     solution of linear system, a*x = b .
c     do not use if decomp has detected singularity.
c
c     input..
c
c     ndim = declared row dimension of array containing a .
c
c     n = order of matrix.
c
c     a = triangularized matrix obtained from decomp .
c
c     b = right hand side vector.
c
c     ipvt = pivot vector obtained from decomp .
c
c     output..
c
c     b = solution vector, x .
c
      integer kb, km1, nm1, kp1, i, k, m
      real*8 t
c
c     forward elimination
c
      if (n .eq. 1) go to 50
      nm1 = n-1
      do 20 k = 1, nm1
        kp1 = k+1
        m = ipvt(k)
        t = b(m)
        b(m) = b(k)
        b(k) = t
        do 10 i = kp1, n
          b(i) = b(i) + a(i,k)*t
 10     continue
 20   continue
c
c     back substitution
c
      do 40 kb = 1,nm1
        km1 = n-kb
        k = km1+1
        b(k) = b(k)/a(k,k)
        t = -b(k)
        do 30 i = 1, km1
          b(i) = b(i) + a(i,k)*t
 30     continue
 40   continue
 50   b(1) = b(1)/a(1,1)
      return
      end
c
c
cBH130915:  Kinsey replaces sorspt with sorspt1
c
c$$$      subroutine sorspt(bshape, bheigh, bwidth, bhfoc, bvfoc,
c$$$     *  bhdiv, bvdiv, ib, ie, isourc, ke,
c$$$     *  nsourc, sfrac1, vbeam,
c$$$     *  x0, y0, z0, vx0, vy0, vz0)
c$$$      implicit integer (i-n), real*8 (a-h,o-z)
c$$$      save
c$$$ccc   
c$$$ccc   generates a particle at the injector surface with
c$$$ccc   coordinates and velocities x0,y0,z0,vx0,vy0,vz0
c$$$ccc   These coordinates are attached to the source center, with the
c$$$ccc   x-direction perp to the source along the source centerline.
c$$$ccc   
c$$$      character*8 bshape
c$$$      dimension bshape(*), bheigh(*), bwidth(*), bhfoc(*), bvfoc(*),
c$$$     *  bhdiv(*), bvdiv(*), sfrac1(*), vbeam(ke,*)
c$$$      integer iflag
c$$$c     drand(iflag) is real*8 [0.,1.] random number.
c$$$c     iflag=0 gives next number in the sequence.  See Absoft f77.
c$$$c     For Cray F90: use ranf.
c$$$
c$$$      data pio180/0.017453293/
c$$$      data rt2/1.414213562/
c$$$      data       seed0 /0.0d0/
c$$$ccc
c$$$
c$$$  
c$$$      x0=0.
c$$$      isourc=1
c$$$      if(nsourc.eq.1)  go to 10
c$$$c     two sources
c$$$c     sfrac1 is fraction of source current coming from source 1 (upper
c$$$c     or rightmost source, assuming positive offsets).
c$$$c990131      if(ranf().gt.sfrac1(ib))  isourc=-1
c$$$
c$$$c000114*Not portable*      if(RANDOM_my(seed0).gt.sfrac1(ib))  isourc=-1
c$$$      rv=RANDOM_my(seed0)
c$$$      if(rv.gt.sfrac1(ib))  isourc=-1
c$$$ 10   if(bshape(ib).ne."circ") go to 20
c$$$ccc   
c$$$c990131 12   y0=ranf()-0.5
c$$$c990131      z0=ranf()-0.5
c$$$
c$$$c000114*Not portable* 12   y0=drand(iflag)-0.5
c$$$c000114*Not portable*      z0=drand(iflag)-0.5
c$$$ 12   rv=RANDOM_my(seed0)
c$$$      y0=rv-0.5
c$$$      rv=RANDOM_my(seed0)
c$$$      z0=rv-0.5
c$$$
c$$$
c$$$      zsqu=y0**2+z0**2
c$$$      if(zsqu.gt.0.25) go to 12
c$$$      y0=y0*bwidth(ib)
c$$$      z0=z0*bwidth(ib)
c$$$c      write(*,*)'sorspt: rv,ib,bwidth(ib)',rv,ib,bwidth(ib)
c$$$      go to 30
c$$$ccc   
c$$$ 20   continue
c$$$c990131      y0=bwidth(ib)*(ranf()-0.5)
c$$$c990131      z0=bheigh(ib)*(ranf()-0.5)
c$$$c000114*Not portable*      y0=bwidth(ib)*(drand(iflag)-0.5)
c$$$c000114*Not portable*      z0=bheigh(ib)*(drand(iflag)-0.5)
c$$$      rv=RANDOM_my(seed0)
c$$$      y0=bwidth(ib)*(rv-0.5)
c$$$      rv=RANDOM_my(seed0)
c$$$      z0=bheigh(ib)*(rv-0.5)
c$$$
c$$$ccc   
c$$$ccc   
c$$$ccc   
c$$$ccc   
c$$$ 30   continue
c$$$c
c$$$c     Special coding for DIII-D Long Pulse Sources.
c$$$      if(bshape(ib).ne."rect-lps")  go to 32
c$$$      if(abs(z0).gt.12.)  go to 31
c$$$c
c$$$c     particle on central 2 modules
c$$$      vdx=-1.
c$$$      vdy=0.0
c$$$      vdz=0.0
c$$$      go to 33
c$$$c
c$$$c     particle on upper or lower modules
c$$$ 31   vdx=-1.
c$$$      vdy=0.0
c$$$      vdz=-z0/bvfoc(ib)
c$$$      go to 33
c$$$c
c$$$ 32   vdx=-1.
c$$$      vdy=-y0/bhfoc(ib)
c$$$      vdz=-z0/bvfoc(ib)
c$$$ 33   vsqrt=1./sqrt(vdx**2+vdy**2+vdz**2)
c$$$      vx0=vbeam(ie,ib)*vdx*vsqrt
c$$$      vy0=vbeam(ie,ib)*vdy*vsqrt
c$$$      vz0=vbeam(ie,ib)*vdz*vsqrt
c$$$ccc   
c$$$      thz=ranorm(1)*bvdiv(ib)/rt2
c$$$      thy=ranorm(1)*bhdiv(ib)/rt2
c$$$      vz0=vz0+thz*pio180*vx0
c$$$      vy0=vy0+thy*pio180*vx0
c$$$c      write(*,*)'sorspt: ie,ib,bvfoc(ib),y0,z0,vbeam(ie,ib)',
c$$$c     1                   ie,ib,bvfoc(ib),y0,z0,vbeam(ie,ib)
c$$$      return
c$$$      end
c
c
      subroutine sorspt1(nbshape,bheigh,bwidth,bhfoc,bvfoc,
     .                  bhdiv,bvdiv,ib,ie,isourc,nsourc,sfrac1,vbeam,
     .                  x0,y0,z0,vx0,vy0,vz0)

c... generates a particle at the injector surface with
c    coordinates and velocities x0,y0,z0,vx0,vy0,vz0
c    These coordinates are attached to the source center, with the
c    x-direction perp to the source along the source centerline.
c
      implicit  integer (i-n), real*8 (a-h, o-z)
c      include 'params.inc'
      include 'param.h'
c
cBH131015      external RANDOM_my  ! random number generator
      integer nsourc
      real*8  bheigh(kb), bwidth(kb), bhfoc(kb), bvfoc(kb),
     .        bhdiv(kb), bvdiv(kb), sfrac1(kb), vbeam(ke,kb)
      character*8 nbshape(kb)
c
      data pio180 /0.017453293D0/
      data rt2 /1.414213562D0/
      data seed0 /0.d0/
c
      x0 = 0.D0
      isourc=1 !YuP/note: This line changes isourc from 0 to 1 at 1st call
      if (nsourc.eq.1) goto 10
c
c     two sources (DIII-D case)
c     sfrac1 is fraction of source current coming from source 1 (upper
c     or rightmost source, assuming positive offsets).

c        if (RANF().gt.sfrac1(ib)) isourc = -1
      if (RANDOM_my(seed0).gt.sfrac1(ib)) isourc=-1 !Only if nsourc>1
   10 if (nbshape(ib).ne.'circ') go to 20
c
c   12 y0 = RANF() - 0.5D0
   12 y0 = RANDOM_my(seed0) - 0.5D0
c        z0 = RANF() - 0.5D0
      z0 = RANDOM_my(seed0) - 0.5D0
      zsqu = y0 ** 2 + z0 ** 2
      if (zsqu.gt.0.25) go to 12
      y0 = y0 * bwidth(ib)
      z0 = z0 * bwidth(ib)
      go to 30
c
  20  continue
      y0 = bwidth(ib) * (RANDOM_my(seed0)-0.5D0)
      z0 = bheigh(ib) * (RANDOM_my(seed0)-0.5D0)
c
c  Special coding for DIII-D Long Pulse Sources
c
   30 if (nbshape(ib) .ne. 'rect-lps')  go to 32
      if (   ABS (z0) .gt.  12.0     )  go to 31
c
c  particle on central 2 modules
c
      vdx = -1.0
      vdy =  0.0
      vdz =  0.0
      go to 33
c
c  particle on upper or lower modules
c
   31 vdx = -1.D0
      vdy = 0.D0
      vdz = -z0 / bvfoc(ib)
      go to 33
c
   32 vdx = -1.0
      vdy = -y0 / bhfoc(ib)
      vdz = -z0 / bvfoc(ib)
c
  33  vsqrt = 1.D0 / dsqrt(vdx**2+vdy**2+vdz**2)
      vx0 = vbeam(ie,ib) * vdx * vsqrt
      vy0 = vbeam(ie,ib) * vdy * vsqrt
      vz0 = vbeam(ie,ib) * vdz * vsqrt
c
      thz = ranorm(1) * bvdiv(ib) / rt2
      thy = ranorm(1) * bhdiv(ib) / rt2
      vz0 = vz0 + thz * pio180 * vx0
      vy0 = vy0 + thy * pio180 * vx0
c
      return
      end subroutine sorspt1
c
c
      subroutine timtor(rin, rmax, x0, y0, z0, vx0, vy0, vz0,
     *  zmin, zmax, tenter, texit)
      implicit none !integer (i-n), real*8 (a-h,o-z)
c
c     This subroutine calculates the times for a particle to enter and
c     exit a toroidal box surrounding the plasma starting from the
c     point (x0,y0,z0) and moving with velocity (vx0,vy0,vz0).
c
      real*8 rin,rmax,zmin,zmax, x0, y0, z0, vx0, vy0, vz0 !IN
      real*8 tenter, texit !OUT
      real*8 edge(4),timsol(6) !local
      real*8 zero,aa,bb,cc,arg,sqr,xx,yy,zz,rr,tt !local
      integer isol,i,iin !local
      
c
      zero=0.d0

c     specify edges of toroidal box surrounding plasma
c

c      write(*,*) 'x0,y0,z0,vx0,vy0,vz0,tenter,texit',
c     +     x0,y0,z0,vx0,vy0,vz0,tenter,texit

      edge(1)=rin
      edge(2)=rmax
      edge(3)=zmin
      edge(4)=zmax
c
c     find times for particle to intersect inside and outside of box
c
      isol=0
      aa=vx0**2+vy0**2
      if(aa.eq.zero) go to 50
      bb=2.*(vx0*x0+vy0*y0)
      do 40 i=1,2
        cc=x0**2+y0**2-edge(i)**2
        arg=bb**2-4.*aa*cc
        if(arg) 40,30,10
 10     sqr=sqrt(arg)
        tt=(-bb-sqr)/(2.*aa)
        zz=z0+vz0*tt
        if(zz.lt.zmin) go to 20
        if(zz.gt.zmax) go to 20
        isol=isol+1
        timsol(isol)=tt
 20     tt=(-bb+sqr)/(2.*aa)
        zz=z0+vz0*tt
        if(zz.lt.zmin) go to 40
        if(zz.gt.zmax) go to 40
        isol=isol+1
        timsol(isol)=tt
        go to 40
 30     tt=-bb/(2.*aa)
        zz=z0+vz0*tt
        if(zz.lt.zmin) go to 40
        if(zz.gt.zmax) go to 40
        isol=isol+1
        timsol(isol)=tt
 40   continue
c
c     find times for particle to intersect top and bottom of box
c
 50   if(vz0.eq.zero) go to 70
      do 60 i=3,4
        tt=(edge(i)-z0)/vz0
        xx=x0+vx0*tt
        yy=y0+vy0*tt
        rr=sqrt(xx**2+yy**2)
        if(rr.lt.rin) go to 60
        if(rr.gt.rmax) go to 60
        isol=isol+1
        timsol(isol)=tt
 60   continue
 70   continue
c
c     return if particle misses box
c
      tenter = -1.d10 !YuP[2022-12] was -1.e10
      if(isol.eq.0) return
      
      if(isol.gt.6)then
        WRITE(*,*)'subr.timtor: isol>6  BUG?'
        STOP
      endif
c
c     calculate times to enter and exit box
c
      tenter=timsol(1)
      iin=1
      do 80 i=2,isol
        if(timsol(i).ge.tenter) go to 80
        iin=i
        tenter=timsol(i)
 80   continue
      texit=1.d10 !YuP[2022-12] was 1.e10
      do 90 i=1,isol
        if(i.eq.iin) go to 90
        if(timsol(i).lt.texit) texit=timsol(i)
 90   continue
c      write(*,*) 'x0,y0,z0,vx0,vy0,vz0,tenter,texit',
c     +     x0,y0,z0,vx0,vy0,vz0,tenter,texit

      return
      end subroutine timtor
c
c
      subroutine tozone(x,y,n,xz,yz,nz,nout,ncrt)
      implicit integer (i-n), real*8 (a-h,o-z)
      integer iz,ip
      save
c     ---------------------------------------------------------------
c     convert from point values to zone values
c     ---------------------------------------------------------------
      dimension x(*),y(*),xz(*),yz(*)
      data tol/1.0D-20/
c
      ip=1
      iz=1
      yzlast=y(n)
c     ---------------------------------------------------------------
c     interpolate to new grid that is compute yz(xz(iz))
c     ---------------------------------------------------------------
 2000 if(iz.gt.nz) go to 2210
 2100 if(ip.gt.n) go to 9200
      del=reldif(x(ip),xz(iz))
      if(del.gt.tol) go to 2110
      yz(iz)=y(ip)
      iz=iz+1
      go to 2000
 2110 if(x(ip).gt.xz(iz)) go to 2120
      ip=ip+1
      go to 2100
 2120 ipm1=ip-1
      s=(y(ip)-y(ipm1))/(x(ip)-x(ipm1))
      yz(iz)=y(ipm1)+(xz(iz)-x(ipm1))*s
      iz=iz+1
      go to 2000
c     ---------------------------------------------------------------
c     convert to zone
c     ---------------------------------------------------------------
 2210 continue
      do 2300 iz=1,nz-1
        yz(iz)=.5*(yz(iz)+yz(iz+1))
 2300 continue
      yz(nz)=.5*(yz(nz)+yzlast)
      return
c     ---------------------------------------------------------------
c     fatal errors
c     ---------------------------------------------------------------
 9200 continue
      write(nout,8000)
      write(ncrt,8000)
 8000 format(' fatal error in subroutine tozone',//,
     *  ' xz(iz) or xz(izp1) is not contained in interval x(1),x(n)')
      stop
      end

      real*8 function RANDOM_my(seed)
c
c --- basic uniform (0,1) pseudo-random number generator (portable) ----
c     From GA portlib.f (Jan. 14, 2000, bh).
c
c     BH131015 comment:
c     If called with seed.ne.0.d0, it will start a random number
c     sequence with the given seed.  Subsequent calls with seed=0.0d0
c     will get the ensuing RNs in a sequence.
c
      implicit none
c
      save         saved_seed
      real*8 seed, saved_seed,
     .       big,                  big_minus_1
      data   big /2147483648.0d0/, big_minus_1 /2147483647.0d0/
c
      if (seed .ne. 0.0d0)  saved_seed = seed
      saved_seed = MOD (16807.0d0*saved_seed, big_minus_1)
      RANDOM_my  = saved_seed / big
      return
c
      end

c.......................................................................
cBH130915:  Rest of this file down through spline_12 subroutine are
cBH130915:  additions per Kinsey upgrade of freya stopping.
c.......................................................................

      subroutine inject_old (atw,codeid,drutpi,droti,dri,dzi,elongi,
     .    ib,ie,mfm1,mim1,mjm1,newpar,psiax,psi,r,rmajor,rin,rmax,sgxn,
     .    sgxnmi,x0,y0,z0,vx0,vy0,vz0,vbeam,z,zax,zmin,zmax,izone,pzone,
     .    rzone,rpos,xpos,ypos,zpos) !for iexcit.le.0 only
      !YuP: subr.inject_old might have bugs in sgxn() array usage.

c     This subroutine follows the particle from the pivot point
c     into, through, or around the plasma. It is the older version
c     of sub. inject to be used when crsecs (iexcit=0) is used.
c
      implicit  integer (i-n), real*8 (a-h, o-z)
c      include 'params.inc'
      include 'param.h'
c      save
c
cBH131015      external  RANDOM_my             ! random number generator
      real*8 psi(ki,kj), sgxn(kcmp1,kz,kbe,ksge), sgxnmi(ke,kb), 
     .     r(ki), z(kj), vbeam(ke,kb)
      real*8 atw(kion), rin, rmax, zax, zmin, zmax
      character*8 codeid
      data  seed0  /0.0d0/
c
      if (newpar.ne.0.or.isol.ne.0) then
        if (newpar.ne.0) then
c
c... Calculate times for particle to enter and exit toroidal box
c    surrounding plasma:
c
c      write(*,*)
c      write(*,*) 'inside subroutine inject...',newpar
c      write(*,*) 'atw = ',atw
c      write(*,*) 'codeid = ',codeid
c      write(*,*) 'debin = ',debin
c      write(*,*) 'drutpi = ',drutpi
c      write(*,*) 'droti = ',droti
c      write(*,*) 'dri = ',dri
c      write(*,*) 'dzi = ',dzi
c      write(*,*) 'elongi = ',elongi
c      write(*,*) 'rin,rmax = ',rin,rmax
c      write(*,*) 'rmajor = ',rmajor
c      write(*,*) 'psiax = ',psiax
c      write(*,*) 'x0 = ',x0
c      write(*,*) 'y0 = ',y0
c      write(*,*) 'z0 = ',z0
c      write(*,*) 'ib,ie,kbe = ',ib,ie,kbe
c      write(*,*) 'ke,kz,ki = ',ke,kz,ki
c      write(*,*) 'mfm1,mim1,mjm1 = ',mfm1,mim1,mjm1
c      write(*,*) 'nebin = ',nebin
c      write(*,*) 'newpar = ',newpar
c
          call timtor(rin,rmax,x0,y0,z0,vx0,vy0,vz0,zmin,zmax,tenter,
     .        texit)
c      write(*,*)
c      write(*,*) 'after timtor ...'
c      write(*,*) 'rin = ',rin
c      write(*,*) 'rmax = ',rmax
c      write(*,*) 'x0 = ',x0
c      write(*,*) 'y0 = ',y0
c      write(*,*) 'z0 = ',z0
c      write(*,*) 'vx0 = ',vx0
c      write(*,*) 'vy0 = ',vy0
c      write(*,*) 'vz0 = ',vz0
c      write(*,*) 'zmin = ',zmin
c      write(*,*) 'zmax = ',zmax
c      write(*,*) 'tenter = ',tenter
          if (tenter.le.-1.d10) go to 20 
           !YuP[2022-12] was -1.e10
c
c... advance particle to edge of box
c
          x0 = x0 + vx0 * tenter
          y0 = y0 + vy0 * tenter
          z0 = z0 + vz0 * tenter
c
        endif ! newpar.ne.0
c
c... set coordinates and time for entering box
c
        xpos = x0
        ypos = y0
        zpos = z0
        tt = tenter
c
c... follow particle into plasma
c
   10   continue
c        dfac = -dlog(ranf())
        dfac = -LOG(RANDOM_my (seed0))
        tstep = dfac * sgxnmi(ie,ib) / vbeam(ie,ib)
        tt = tt + tstep
c        write(*,*) 'dfac = ',dfac
c        write(*,*) 'vbeam = ',vbeam(ie,ib)
c        write(*,*) 'sgxnmi = ',sgxnmi(ie,ib)
c        write(*,*) 'tstep = ',tstep
c        write(*,*) 'tt,texit = ',tt,texit
        if (tt.lt.texit) then
          xpos = xpos + vx0 * tstep
          ypos = ypos + vy0 * tstep
          zpos = zpos + vz0 * tstep
          rpos = dsqrt(xpos**2+ypos**2)
c
c... determine zone in which particle collides for 'onedee' geometry
c
          if (codeid.eq.'onedee') then
            rzone2 = (rpos-rmajor) ** 2 + (elongi*(zpos-zax)) ** 2
            rzone = SQRT (rzone2)
            izone= INT(rzone * droti + 1.d0)  !YuP[2022-12] INT()
          else
c
c... Determine zone in which particle collides for general geometry;
c    use bilinear interpolation away from magnetic axis and
c    biquadratic interpolation near the axis
c
            i = (rpos-r(1)) * dri + 1.
            j = (zpos-z(1)) * dzi + 1.
            if (i.gt.mim1) i = mim1
            if (j.gt.mjm1) j = mjm1
            psix = dmin1(psi(i,j),psi(i+1,j),psi(i,j+1),psi(i+1,j+1))
            ptest = (psix-psiax) * (drutpi/mfm1)
            if (ptest.ge.0.02) then
              area1 = (rpos-r(i)) * (zpos-z(j))
              area2 = (r(i+1)-rpos) * (zpos-z(j))
              area3 = (r(i+1)-rpos) * (z(j+1)-zpos)
              area4 = (rpos-r(i)) * (z(j+1)-zpos)
              pzone = (area3*psi(i,j)+area4*psi(i+1,j)+area1*
     .            psi(i+1,j+1)+area2*psi(i,j+1)) * dri * dzi
            else
              call pfit(psi(i-1,j-1),r(i-1),z(j-1),rpos,zpos,ki,pzone,
     .            dum,dum)
            endif
            pzone= MAX(pzone,psiax)
            izone= INT(SQRT(pzone-psiax)*drutpi +1.d0) !YuP[2022-12] INT()
c            write(*,*) 'izone-inject = ',izone
          endif ! codeid
c
c... If particle has psuedo-collision, continue following particle;
c    If particle has real collision, return
c
          if (izone.gt.mfm1) go to 10
!YuP/was          if (RANDOM_my(seed0) .gt. 
!YuP/was     .       sgxn(4,izone,ie,ib)*sgxnmi(ie,ib))  go to 10
          !YuP[2022-12-15] BUG in inject_old?
          !See above: real*8 sgxn(kcmp1,kz,kbe,ksge)
          index=ke*(ib-1)+ie !YuP[2023-01-24] combined energy&beam_index
          !YuP[2023-01-24] Changed to:
          if (RANDOM_my(seed0) .gt. 
     .       sgxn(4,izone,index,1)*sgxnmi(ie,ib))  go to 10
          return
c
        endif ! tt.lt.texit
c
      endif  ! newpar.ne.0.or.isol.ne.0
c
c... set flag to indicate that particle hit wall
c
   20 izone= mfm1 + 1
c
      return
      end subroutine inject_old


      subroutine inject1 (atw,codeid,debin,drutpi,droti,dri,ds1,dzi,
     .                   elongi,ib,ie,kb,kbe,ksge,ke,kz,ki,mfm1,mim1,
     .                   mjm1,nebin,newpar,nout,psiax,psi,r,rmajor,rin,
     .                   rmax,sgxn,sgxnloc,sgxnmi,x0,y0,z0,vx0,vy0,vz0,
     .                   vbeam,z,zangrot,zax,zmin,zmax,izone,
     .                   pzone,rzone,rpos,xpos,ypos,zpos,tenter,
     .                   smax,texit)
c
      implicit none ! integer (i-n), real*8 (a-h, o-z)
c
c ----------------------------------------------------------------------
c  This subroutine follows the particle from the pivot point into,
c  through, or around the plasma.  note:  bilinear interpolation is
c  used to obtain psi as a function of track length along a
c  collisionless neutral trajectory (rotating discharges only).
c  bicubic spline interpolation was tested, but found to provide no
c  appreciable increase in accuracy.
c ----------------------------------------------------------------------
c
cBH131015      external  RANDOM_my             ! random number generator
      integer ib,ie,kb,kbe,ksge,ke,kz,ki,mfm1,mim1,mjm1,newpar !IN
      integer nebin !IN  nebin==ne_tk in cqlinput
      integer nout !IN
      integer izone !OUT
      real*8 x0,y0,z0,vx0,vy0,vz0 !IN/OUT
      real*8 dzi,dri,psiax,droti !IN
      real*8 debin !IN debin==de_tk in cqlinput
      real*8 ds1 !IN  ds1==ds_tk in cqlinput
      real*8 pzone,rzone,rpos,xpos,ypos,zpos !IN/OUT
      real*8 atw,zax,zmin,zmax,elongi,rmajor,rin,rmax !IN
      real*8 drutpi !IN
      real*8 smax !OUT
      real*8 texit,tenter !to be found in subr.timtor
      real*8 x1,y1,z1,delt,dt1,area1,area2,area3,area4,p1,r1,tt !local
      real*8 vrel1,ptest,smin_time,rzone2,eova,vrel2,dum,dfac,tstep !local
      real*8 psix,smin_step,xnorm,vdotu,usq !local
      integer i,j,n1,ktk,ir,iz,izone1,ne1,index !local
      real*8 cvec(200) !local
      data      cvec
     .     /  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0,
     .       11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
     .       21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
     .       31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
     .       41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0,
     .       51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
     .       61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0,
     .       71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
     .       81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0,
     .       91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,100.0,
     .      101.0,102.0,103.0,104.0,105.0,106.0,107.0,108.0,109.0,110.0,
     .      111.0,112.0,113.0,114.0,115.0,116.0,117.0,118.0,119.0,120.0,
     .      121.0,122.0,123.0,124.0,125.0,126.0,127.0,128.0,129.0,130.0,
     .      131.0,132.0,133.0,134.0,135.0,136.0,137.0,138.0,139.0,140.0,
     .      141.0,142.0,143.0,144.0,145.0,146.0,147.0,148.0,149.0,150.0,
     .      151.0,152.0,153.0,154.0,155.0,156.0,157.0,158.0,159.0,160.0,
     .      161.0,162.0,163.0,164.0,165.0,166.0,167.0,168.0,169.0,170.0,
     .      171.0,172.0,173.0,174.0,175.0,176.0,177.0,178.0,179.0,180.0,
     .      181.0,182.0,183.0,184.0,185.0,186.0,187.0,188.0,189.0,190.0,
     .      191.0,192.0,193.0,194.0,195.0,196.0,197.0,198.0,199.0,200.0/
c
      parameter (ktk = 100)
      character*8  codeid
      integer i1(ktk) !local
      real*8 sgxn(4,kz,kbe,ksge)
      real*8 sgxnmi(ke,kb), vbeam(ke,kb)
      real*8 sgxnloc(4) !YuP[2022-12-15] Was (kbe)
      real*8 r(*), z(*), psi(ki,*)
      real*8 zangrot(kz) !INPUT
      real*8 e1(ktk), sgxntab(ktk)
      
      real*8 seed0 !local
      data   seed0 /0.d0/
      
      real*8 kevperg, xmassp
      data xmassp / 1.67262310d-24 /    ! proton   mass   (g)
      data kevperg /6.2415064d+08/
c
      real*8 wk_r(4),wk_z(4), wk_p(4,4)
      
      real*8 amaxaf,random_my !external func
      
c     the following assumes that data from previous
c     particle is saved. this ok only if passed through
c     argument list. HSJ
c
      do i=1,ktk
        i1(i)=0
        e1(i)=0.D0
        sgxntab(i)=0.D0
      enddo
      sgxnloc=0.D0 !initialize for all (:)
c
c      write(*,*)
!      if(ib.eq.2)write(*,*) 'inside subroutine inject...',newpar
c      write(*,*) 'atw = ',atw
c      write(*,*) 'codeid = ',codeid
c      write(*,*) 'debin = ',debin
c      write(*,*) 'drutpi = ',drutpi
c      write(*,*) 'droti = ',droti
c      write(*,*) 'dri = ',dri
c      write(*,*) 'ds1 = ',ds1
c      write(*,*) 'dzi = ',dzi
c      write(*,*) 'elongi = ',elongi
c      write(*,*) 'x0 = ',x0
c      write(*,*) 'y0 = ',y0
c      write(*,*) 'z0 = ',z0
c      write(*,*) 'ib,ie,kbe = ',ib,ie,kbe
c      write(*,*) 'ke,kz,ki = ',ke,kz,ki
c      write(*,*) 'mfm1,mim1,mjm1 = ',mfm1,mim1,mjm1
!      if(ib.eq.2)write(*,*) 'nebin = ',nebin
c      write(*,*) 'newpar = ',newpar
      if (newpar .eq. 0)  go to 100
c
c calculate times for particle to enter and exit toroidal box surrounding plasma
c
      call timtor(rin,rmax,x0,y0,z0,vx0,vy0,vz0,zmin,zmax,tenter,texit)
      if(tenter.le.-1.0d10)then !just a printout
        !write(*,*)
        write(*,'(a,1pe11.3,a,1p2e11.3)') 'after timtor: tenter=',
     &           tenter,' x0,vx0= ',x0,vx0
        !write(*,*) 'rin,rmax= ',rin,rmax
        !write(*,*) 'x0,vx0= ',x0,vx0
        !write(*,*) 'y0 = ',y0
        !write(*,*) 'z0 = ',z0
        !write(*,*) 'zmin,zmax= ',zmin,zmax
      endif !printout
      if (tenter .le. -1.0d10)  go to 140 !izone=mfm1+1, exit
c
c advance particle to edge of box
c
      x0 = x0 + vx0*tenter
      y0 = y0 + vy0*tenter
      z0 = z0 + vz0*tenter
c
      if (nebin .ne. 0) then !toroidal rotation (ne_tk>0)
c
c ----------------------------------------------------------------------
c follow collisionless neutral trajectory to obtain minimum mean free path.
c this is required to account for toroidal rotation.
c
c ALSO:
c    Only include those energy groups that fall in a valid flux zone,
c    (izone1 .le. mfm1) for the search in subroutine GETSGXN. The array
c    of energy values e1 is now ne1 elements long rather than n1.
c                                            Daniel Finkenthal    9-6-95
c ----------------------------------------------------------------------
c
      smax = vbeam(ie,ib) * (texit-tenter)
      n1   = 2.0 + smax/ds1 !ds1==ds_tk in cqlinput
      if (n1 .gt. ktk) then
        n1  = ktk
        ds1 = smax / dble(n1-1) !YuP[2022-12-19] was FLOAT
        write (nout, 200) ds1
      endif
      dt1 = smax / (dble(n1-1) * vbeam(ie,ib)) !YuP[2022-12-19] was FLOAT
      ne1 = 0
c
      if (codeid .eq. 'onedee') then
        do i=1,n1
          delt   = (cvec(i) - 1.0) * dt1
          x1     = x0 + delt*vx0
          y1     = y0 + delt*vy0
          z1     = z0 + delt*vz0
          r1     = SQRT (x1**2 + y1**2)
          ir     = (r1-r(1))*dri + 1.0
          iz     = (z1-z(1))*dzi + 1.0
          ir     = MIN0 (ir,mim1)
          iz     = MIN0 (iz,mjm1)
          if (ir .le. 0 .or. iz .le. 0)go to 10  !HSJ 1/28/2000
          p1     = SQRT ((r1-rmajor)**2+(elongi*(z1-zax))**2)
          izone1 = p1*droti + 1.0
          if (izone1 .le. mfm1) then
            ne1     = ne1 + 1
            i1(ne1) = izone1
            usq     = ( x1**2 + y1**2 ) * zangrot(i1(ne1))**2
            vdotu   = (x1*vy0 - y1*vx0) * zangrot(i1(ne1))
            vrel1   = vbeam(ie,ib)**2 + usq - 2.0*vdotu
            e1(ne1) = 0.5 * xmassp * vrel1 * kevperg
            e1(ne1) = ABS (e1(ne1))
          endif
        enddo
      else !(codeid .ne. 'onedee')
        do i=1,n1
          delt  = (cvec(i) - 1.0) * dt1
          x1    = x0 + delt*vx0
          y1    = y0 + delt*vy0
          z1    = z0 + delt*vz0
          r1    = SQRT (x1**2+y1**2)
          ir    = (r1-r(1))*dri + 1.0
          iz    = (z1-z(1))*dzi + 1.0
          ir    = MIN0 (ir,mim1)
          iz    = MIN0 (iz,mjm1)
          if (ir .le. 0 .or. iz .le. 0)go to 10  !HSJ 1/28/2000
          area1 = (r1-r(ir))*(z1-z(iz))
          area2 = (r(ir+1)-r1)*(z1-z(iz))
          area3 = (r(ir+1)-r1)*(z(iz+1)-z1)
          area4 = (r1-r(ir))*(z(iz+1)-z1)
          p1    = (area3*psi(ir,iz)   + area4*psi(ir+1,iz)
     .          +  area2*psi(ir,iz+1) + area1*psi(ir+1,iz+1))*dri*dzi
          p1     =  MAX (p1,psiax)
          izone1 = SQRT (p1-psiax)*drutpi + 1.0
          if (izone1 .le. mfm1) then
            ne1     = ne1 + 1
            i1(ne1) = izone1
            usq     = (x1**2+y1**2)*zangrot(i1(ne1))**2
            vdotu   = (x1*vy0-y1*vx0)*zangrot(i1(ne1))
            vrel1   = vbeam(ie,ib)**2 + usq - 2.0*vdotu
            e1(ne1) = 0.5 * xmassp * vrel1 * kevperg
            e1(ne1) = ABS (e1(ne1))
          endif
        enddo
      endif !(codeid)
c
c      write(*,*)
c       write(*,*) 'calling getsgxn ...'
c       write(*,*) 'ne1 = ',ne1
c      write(*,*) 'e1(ne1) = ',e1(ne1)
c      write(*,*) 'i1(ne1) = ',i1(ne1)
c      write(*,*) 'vrel1 = ',vrel1
c      write(*,*) 'usq = ',usq
c      write(*,*) 'vdotu = ',vdotu
c      write(*,*) 'zangrot  = ',zangrot(i1(ne1))
c      write(*,*) 'ib = ',ib
c      write(*,*) 'ie = ',ie
c      write(*,*) 'nebin = ',nebin
c      write(*,*) 'debin = ',debin,' in inject'
c      write(*,*) 'sgxn(4,22,3,2)-pre = ',sgxn(4,22,3,2)
c
 10   call getsgxn (e1, i1, ne1, ktk, ib, ie, sgxn, nebin, debin, kbe, !Tor.rot.case
     .              ksge, kz, nout, sgxntab)
      !write(*,*) 'after call to getsgxn ...'
      !write(*,*) 'ne1 = ',ne1
c      do i=1,ne1
c       write(*,*) i, sgxntab(i), ' sgxntab'
c      enddo
      xnorm         = amaxaf (sgxntab, 1, ne1)
      sgxnmi(ie,ib) = 1.0 / xnorm
c      write(*,*) 'sgxnmi(ie,ib) = ',sgxnmi(ie,ib)
      end if ! nebin.ne.0
      
      
c      stop
c
c ----------------------------------------------------------------------
c inject neutral into plasma
c ----------------------------------------------------------------------
c
c... set coordinates and time for entering box
c
  100 xpos      = x0
      ypos      = y0
      zpos      = z0
      tt        = tenter
      izone= mfm1 + 1     ! initially neutral is outside the plasma
      smin_step = 0.1                           ! 0.1 cm min step or
      smin_step = MIN (smin_step, smax/1000.0)  ! make scale-independent
      smin_time = smin_step / SQRT (vx0**2 + vy0**2 + vz0**2)
c
c... follow particle into plasma
c
  110 dfac  = -LOG(RANDOM_my(seed0))
c
c     if neutral is not yet in the plasma (izone ge mf) then take steps
c     of minimum size 1 mm until we enter the plasma. we could find the
c     exact plasma boundary but that would be overkill. with 1mm step
c     size we certainly are within any physics scales we could resolve
c     near the plasma edge. this avoids wasting a lot of steps outside
c     the plasma and will be a significant savings if the cross sections
c     are large. dfac is 1/exponentially distributed, so for large steps
c     don't modify tstep. --------------------------- 27 Oct 95 ---- HSJ
c

      if (izone .le. mfm1) then                     !  inside the plasma
        tstep =      dfac * sgxnmi(ie,ib) / vbeam(ie,ib)
      else
        tstep = MAX (dfac * sgxnmi(ie,ib) / vbeam(ie,ib),
     .               smin_time)                     ! outside the plasma
      endif
c
      tt    = tt  + tstep
      if (tt .ge. texit)  then
!       write(*,'(a,i2,i5,1p3e9.2)') 
!     &  'inject1: ie,izone,sgxnmi,vbeam,smin_time=',
!     &  ie,izone,sgxnmi(ie,ib),vbeam(ie,ib),smin_time
       go to 140 !izone=mfm1+1, exit
      endif
      xpos = xpos + vx0*tstep
      ypos = ypos + vy0*tstep
      zpos = zpos + vz0*tstep
      rpos = SQRT (xpos**2 + ypos**2)
c
c  determine zone in which particle collides for 'onedee' geometry
c
      if (codeid .eq. 'onedee') then
        rzone2 = (rpos-rmajor)**2 + (elongi*(zpos-zax))**2
        rzone= SQRT (rzone2)
        izone= INT(rzone*droti + 1.d0)  !YuP[2022-12] INT()
      else
c
c  determine zone in which particle collides for general geometry;
c     use bilinear interpolation away from magnetic axis,
c     and biquadratic interpolation near the axis.
c
        i     = (rpos-r(1))*dri+1.0
        j     = (zpos-z(1))*dzi+1.0
        i     = MIN0 (i,mim1)
        j     = MIN0 (j,mjm1)
        
        psix  = MIN  (psi(i,j),psi(i+1,j),psi(i,j+1),psi(i+1,j+1))
        ptest = (psix-psiax)*(drutpi/mfm1)**2
        
!      if(ib.eq.2)write(*,*)'ptest=',ptest

        if (ptest .ge. 0.02) then
          area1 = (rpos-r(i))*(zpos-z(j))
          area2 = (r(i+1)-rpos)*(zpos-z(j))
          area3 = (r(i+1)-rpos)*(z(j+1)-zpos)
          area4 = (rpos-r(i))*(z(j+1)-zpos)
          pzone = (area3*psi(i,j) + area4*psi(i+1,j)
     .          + area1*psi(i+1,j+1) + area2*psi(i,j+1))*dri*dzi
        else
         !YuP160330: Avoiding out of bounds occuring near R=0 for mirror geometry
         if (i.ne.1) then
           !YuP[2018-01-24] Small mod. of input for pfit:    
           ! Note: subroutine pfit(p,x,y, xv,   yv,  nx,  f,   dfdx,dfdy)
           ! It uses: p(4,4),x(4),y(4)
           dum=0.d0
           pzone=0.d0
           wk_r(1:4)=r(i-1:i+2)
           wk_z(1:4)=z(j-1:j+2)
           wk_p(1:4,1:4)=psi(i-1:i+2,j-1:j+2)
           call pfit(wk_p, wk_r, wk_z, rpos, zpos, ki, pzone, dum, dum)
!           if(ib.eq.2)write(*,*)'pzone,psiax,ki=',pzone,psiax,ki
cyup         call pfit(psi(i-1,j-1), r(i-1), z(j-1), rpos, zpos, ki, pzone,
cyup     *   dum, dum)
         else
           dum=0.d0
           pzone=0.d0
           wk_r(1:4)=r(1:4)
           wk_z(1:4)=z(j-1:j+2)
           wk_p(1:4,1:4)=psi(1:4,j-1:j+2)
           call pfit(wk_p, wk_r, wk_z, rpos, zpos, ki, pzone, dum, dum)
cyup         call pfit(psi(1,j-1), r(1), z(j-1), rpos, zpos, ki, pzone,
cyup     *   dum, dum)
         endif
     
        endif
        pzone=  MAX(pzone,psiax)
        izone= INT(SQRT(pzone-psiax)*drutpi + 1.d0)
c        write(*,*) '** izone = ',izone,psiax
      endif
c
c... if particle has psuedo-collision, continue following particle.
c    if particle has real collision, return.
c

      if (izone .gt. mfm1)  go to 110 ! the particle is inside the box..
c                                     ..but still outside the plasma
      if (nebin .ne. 0) then !toroidal rotation case (ne_tk>0)
        usq   = (rpos*zangrot(izone))**2
        vdotu = (xpos*vy0-ypos*vx0)*zangrot(izone)
        vrel2 = vbeam(ie,ib)**2 + usq - 2.0*vdotu
        eova  = 0.5 * xmassp*vrel2*kevperg
        eova  = ABS (eova)
c        write(*,*) 'eova = ',eova
c        write(*,*) 'vrel2 = ',vrel2
        call getsgxn (eova, izone, 1, ktk, ib, ie, sgxn, nebin, !tor.rot.case
     .                debin, kbe, ksge, kz, nout, sgxnloc)
      else ! nebin==ne_tk=0
        index      = ke*(ib-1) + ie
        sgxnloc(1)= sgxn(1,izone,index,1) !Here: inject1(nebin=0; no rotation)
        sgxnloc(2)= sgxn(2,izone,index,1) !Here: inject1(nebin=0; no rotation)
        sgxnloc(3)= sgxn(3,izone,index,1) !Here: inject1(nebin=0; no rotation)
        !YuP[2023-01] In present logic, we only need sgxnloc(4)= sgxn(4,*)
        !In fact, other sgxnloc(1:3)=sgxn(1:3,*) are computed with old nbsgold
        !and are used for printout/comparison only]
        sgxnloc(4)= sgxn(4,izone,index,1) !Here: inject1(nebin=0; no rotation)
      end if
      if (RANDOM_my(seed0) .gt. sgxnloc(4) * sgxnmi(ie,ib))  go to 110
      return
c
c... set flag to indicate that particle hit wall
c
  140 izone= mfm1 + 1
  200 format (' WARNING from subroutine INJECT1:'                  /
     .        '         maximum number of grid elements exceeded' /
     .        '         increasing ds1 to ', e10.3, ' cm')
      return
c
      end subroutine inject1

c
c
      subroutine nbsgxn (iexcit,namep,namei,mb,mfm1,nebin,nprim,nimp,
     .                   nion,atw_beam,atw,ebkev,ebfac,ibion,vbeam,
     .                   zne,zni,zte,zti,zzi,debin,dtemax,dnemax,dzemax,
     .                   hxfrac,sgxn,sgxnmi,ibstart)
c
c ----------------------------------------------------------------------
c
c  this subroutine calculates the neutral beam attenuation array, sgxn.
c
c     input:
c          iexcit         - atomic excitation model 
c                           0=Freeman-Jones, 5=ADAS, 6=Boley parameterization
c          namei          - name of ith impurity ion species
c          nprim, nimp
c          atw(k)         - atomic mass of ion species k
c          ebkev(mb)      - maximum energy of mb-th neutral beamline
c                           (keV)
c          ebfac          - factor defining upper bound on energy bin
c                           range,  > 1.0
c                           (cqlinput: set fe_tk for ebfac)
c          atw_beam         mass no. of beam
c
c          mb             - number of beamlines modeled
c          mfm1           - number of flux zones minus one
c          nebin          - number of energy bins (rotation case only)
c                           (cqlinput: set ne_tk for nebin)
c          vbeam(ie,mb)   - speed of ie-th energy group of the mb-th
c                           beamline (cm/sec)
c          zne(mfm1)      - local electron density (cm**-3)
c          zni(mfm1,nion) - local density of nion-th ion species
c                           (cm**-3)
c          zzi(mfm1,nion) - local average charge state of nion-th ion
c                           species
c          dtemax,dnemax,dzemax - used in crsecs Stearn formula
c     ONETWO stuff:
c     input from common /io/:
c          ncrt,nout,nouthx,ncorin
c     input from common /ions/:
c          namei, atw
c     input from common /nub3/:
c          iexcit,ilorent,mstate,ncont,kdene,kdeni,kdenz,ksvi,ksvz,ksve,
c          krad,ngh,ngl,znipm,atwpm,iz,zti,izstrp
c     input from common /numbrs/:
c          nprim,nimp,nion
c     END ONETWO stuff
c
c     output:
c          sgxn(i,j,k,l)
c               i - mode index
c                   = 1, fraction of reactions producing electrons;
c                   = 2, fraction of reactions producing species 1 ion;
c                   = 3, fraction of reactions producing species 2 ion;
c                   = 4, total inverse mean free path;
c               j - FREYA zone index
c               k - beam&energy index
c                   k = ke*(ib-1) + ie, where ib is the beam index and ie
c                                      is the beam energy group index
c                                      and ke=3 (three energy groups).
c               l - index for relative energy.  bins are equispaced in
c                   delta(energy) between 0 and max(ebkev(ib))*ebfac,
c                   with bin width given by
c                             delta(energy) = ebmax*ebfac/nebin.
c                   ebfac .ge. 1.0 and nebin are user supplied namelist
c                   variables.
c          sgxnmi(ie,mb)
c                 - minimum inverse mean free path for ie-th energy
c                   group of mb-th beamline.  calculated for stationary
c                   target case only.  otherwise, calculated for each
c                   neutral trajectory in subroutine INJECT.
c          hxfrac(ie,mb)
c                 - fractional density of n = 3 excited state neutral.
c                   calculated in stationary target case only.
c          debin
c                 - width of energy bins (keV/amu)
c ----------------------------------------------------------------------
c
      implicit  integer (i-n), real*8 (a-h, o-z)
c
c      include 'params.inc'  ! both of these .inc's used in stand alone
c      include 'fileiou.inc'
      include 'param.h'
c
      parameter (ksrc = 3)
      character*8 namei(kimp), namep(kprim)   ! added for cql3d
      integer mb, mfm1, nion, nprim
      real*8  ebkev(kb), vbeam(ke,kb)
      real*8  hxfrac(ke,kb),sgxnmi(ke,kb),sgxne(kbe) !YuP[2022-12]BUG? Was sgxne(kz)
      real*8  sgvxne(kz), sgxn(kcmp1,kz,kbe,ksge)  ! stand alone code
c      real*8  sgvxne(kz), sgxn(kcmp1,kz,ke,kb)
c here, atw(kion)=atw(5), atwpm(kprim)=atwpm(3), atwim(kimp)=atwim(2)
c and iz(kimp)=iz(2) in ONETWO
c nimpeqdsk, nprimedsk, and nioneqdsk are read in in prenub.f
c kcm = 3, kcmp1 = kcm + 1 in param.i in Onetwo
c
      integer ibion, iz(kion)
      real*8 ebfac, atw_beam
      real*8 atw(kion), atwpm(kion), atwim(kion)
      real*8 znipm(kion), zniim(kion)
      real*8 zne(kz), zni(kz,kion), zte(kz), zti(kz), zzi(kz,kion)
c
c ----------------------------------------------------------------------
c Original Freeman-Jones routine (iexcit.le.0)
c   -1 for Stearn formula
c    0 for Freeman-Jones
c ----------------------------------------------------------------------
      if (iexcit.le.0) then
        write(*,*) 'Calling Freeman-Jones routine...'
        call crsecs(iexcit,atw,ebkev,ibion,mb,mfm1,nion,vbeam,
     .              zne,zni,zte,zzi,dtemax,dnemax,dzemax,
     .              sgvxne,sgxn,sgxnmi,ibstart)
c
c         do ib=ibstart,mb
c           do j=1,3
c             do i=1,mfm1
c        write(*,'(3i3,2x,1p1e10.4,a12)') i,j,ib,sgxn(4,i,j,ib), ' sgxn'
c             enddo
c           enddo
c         enddo
c        stop
        return
      endif
c ----------------------------------------------------------------------
c Use ADAS routines (iexcit = 5)
c ----------------------------------------------------------------------
      if (iexcit .eq. 5) then
        !write(*,*) 'Calling adassgxn routine...'
        !write(*,'(a,1p3e11.2)') '  bef.adassgxn sgxnmi=',sgxnmi(1:ke,1)
        call adassgxn (namep,namei,mb,mfm1,nprim,nimp,nion,atw,
     .                 ebkev,ebfac,ibion,nebin,vbeam,zne,zni,zte,
     .                 zti,zzi,debin,sgxn,sgxnmi,atw_beam,ibstart)
        !Note the flow: 
        !freya[621]
        !  |-> nbsgxn(case of nebin=0, no rotation) --> 
        !      |->adassgxn->nbsgold() where sgxn(1:4,*) are computed.
        !      |->adassgxn->adasqh6: get sgxeff(1:3)==qrat(1:3), then
        !                   sgxn(4,i,ind,1) = zne(i)*sgxeff(ie)
        !                   sgxnmi(ie,ib)= MAX(sgxnmi(ie,ib),sgxn(4,i,ind,1))
        !                   [adasqh6 only changes sgxn(4,*)]
        !write(*,'(a,1p3e11.2)') '  aft.adassgxn sgxnmi=',sgxnmi(1:ke,1)
        ! do j=1,3
        ! write(*,'(a,4f8.5)')' aft.adassgxn: sgxn(1:4,1,ie)=',sgxn(1:4,1,j,1)
        ! enddo
c
c        do ib=ibstart,mb
c          do j =1,3
c            do i=1,mfm1
c             write(*,'(3i4,2x,1p1e10.4,a16)') 
c     .         i,j,ib,sgxn(4,i,j,ib), ' sgxn-nbsgxn'
c            enddo
c          enddo
c        enddo
c        stop
c
        return !-> exit from iexcit=5 option; No further changes in sgxn()
      endif
c
c ---------------------------------------------------- HSJ-2/5/98 ------
c Boley's parametrization including MSI effects (iexcit=6)
c ----------------------------------------------------------------------
       if (iexcit .eq. 6) then
c         call wrap_xboley (atw,zzi,ebkev,ibion,mb,mfm1,nebin,ebfac,
c     .                     debin,zne,zte,zni,nion,sgxn,sgxnmi,atw_beam)
         return
       endif
c
c ----------------------------------------------------------------------
c HEXNB (iexcit.ne.0)
c ----------------------------------------------------------------------
c
      if (iexcit .ne. 0) then
        if (nouthx .gt. 0)
     .  open (unit = nouthx, file = 'hexnbout', status = 'UNKNOWN')
        istart = 1
c
c       separate primary and impurity ions
c
        do j=1,nion
          if (j .le. nprim) then
            atwpm(j) = atw(j)
          else
            k = j - nprim
            atwim(k) = atw(j)
          endif
        enddo
c
c       get atomic number of impurities
c
        do i=1,nimp
          if(trim(namei(i)).eq.'he' .or. 
     +       trim(namei(i)).eq.'HE' .or. 
     +       trim(namei(i)).eq.'He')  iz(i) =  2
     
          if(trim(namei(i)).eq.'b' .or. 
     +       trim(namei(i)).eq.'B' )  iz(i) =  5  ! YuP added [2015]
          
          if(trim(namei(i)).eq.'c' .or. 
     +       trim(namei(i)).eq.'C' )  iz(i) =  6
          
          if(trim(namei(i)).eq.'o' .or. 
     +       trim(namei(i)).eq.'O' )  iz(i) =  8
          
          if(trim(namei(i)).eq.'si' .or. 
     +       trim(namei(i)).eq.'SI' .or. 
     +       trim(namei(i)).eq.'Si')  iz(i) = 14
          
          if(trim(namei(i)).eq.'ar' .or. 
     +       trim(namei(i)).eq.'AR' .or. 
     +       trim(namei(i)).eq.'Ar')  iz(i) = 18
          
          if(trim(namei(i)).eq.'cr' .or. 
     +       trim(namei(i)).eq.'CR' .or. 
     +       trim(namei(i)).eq.'Cr')  iz(i) = 24
          
          if(trim(namei(i)).eq.'fe' .or. 
     +       trim(namei(i)).eq.'FE' .or. 
     +       trim(namei(i)).eq.'Fe')  iz(i) = 26
          
          if(trim(namei(i)).eq.'ni' .or. 
     +       trim(namei(i)).eq.'NI' .or. 
     +       trim(namei(i)).eq.'Ni')  iz(i) = 28
          
          if(trim(namei(i)).eq.'kr' .or. 
     +       trim(namei(i)).eq.'KR' .or. 
     +       trim(namei(i)).eq.'Kr')  iz(i) = 36
          
          if(trim(namei(i)).eq.'mo' .or. 
     +       trim(namei(i)).eq.'MO' .or. 
     +       trim(namei(i)).eq.'Mo')  iz(i) = 42
          
          if(trim(namei(i)).eq.'w' .or. 
     +       trim(namei(i)).eq.'W' )  iz(i) = 74
        enddo
c
c... no Lorentz ionization limit
c
        bperp = 0.0
      endif
c
c ----------------------------------------------------------------------
c stationary plasma case
c ----------------------------------------------------------------------
c
      if (nebin .eq. 0) then
      
        do ib=ibstart,mb
          do ie=1,ke
            sgxnmi(ie,ib) = 0.D0
            ind  = ke*(ib-1) + ie
            eova = 1.D3*ebkev(ib)/(dFLOAT(ie)*atw_beam) !YuP[2022-12-19] was FLOAT
            vbin = vbeam(ie,ib)
            do i=1,mfm1
              teev = 1.D3*zte(i)
              if (iexcit .ne. 0) then
                tiev = 1.D3*zti(i)
                do j=1,nion
                  if (j .le. nprim) then
                    znipm(j) = zni(i,j)
                  else
                    k = j - nprim
                    zniim(k) = zni(i,j)
                  endif
                enddo
              endif
              sgxne(ind) = fsgxne(vbin,teev,zne(i))
              sgxn(1,i,ind,1) = 0.D0
              do k=1,nion
                sgxn(1,i,ind,1) = sgxn(1,i,ind,1)
     .                          + fsgxni(atw(k),eova,zni(i,k),zzi(i,k))
                if ((k .le. nprim) .and. (k .le. 2.0))
     .            sgxn(k+1,i,ind,1) = fsgxncx(atw(k),eova,zni(i,k))
              enddo
              sgxn(1,i,ind,1) = sgxn(1,i,ind,1) + sgxne(ind)
              sgxn(4,i,ind,1) = sgxn(1,i,ind,1) + sgxn(2,i,ind,1)
     .                        + sgxn(3,i,ind,1)
              sgxn(1,i,ind,1) = sgxn(1,i,ind,1)/sgxn(4,i,ind,1)
              sgxn(2,i,ind,1) = sgxn(2,i,ind,1)/sgxn(4,i,ind,1)
              sgxn(3,i,ind,1) = sgxn(3,i,ind,1)/sgxn(4,i,ind,1)
              if (iexcit .ne. 0) then
c                call hexnb (istart, 1, ilorent, mstate, ncont, eova,
c     .                      teev, tiev, nprim, atwpm, znipm, nimp, iz,
c     .                      atwim, izstrp, zniim, bperp, kdene, kdeni,
c     .                      kdenz, ksvi, ksvz, ksve, krad, ngh, ngl,
c     .                      nouthx, ncorin, rerate, rmfp, hexfrac,
c     .                      ihxerr)
                if (ihxerr .ne. 0) write (ncrt, 1000) ihxerr
                if (ihxerr .eq. 1) then
                  write (ncrt, 1010)
                  write (nout, 1010)
                  call STOP ('subroutine NBSGXN: problem #1', 45)
                end if
                sgxn(4,i,ind,1) = 1.D0/rmfp
                if (i .eq. 1) hxfrac(ie,ib) = hexfrac
              endif
              sgxnmi(ie,ib) = MAX (sgxnmi(ie,ib),sgxn(4,i,ind,1))
            enddo
          enddo
        enddo
        do ib=ibstart,mb
          do ie=1,ke
            sgxnmi(ie,ib) = 1.D0 / sgxnmi(ie,ib)
          enddo
        enddo
        
      else !nebin>0
c
c ----------------------------------------------------------------------
c rotating plasma case
c ----------------------------------------------------------------------
c
        ebmax = 0.D0
        do ib=ibstart,mb
          ebmax = MAX (ebmax,ebkev(ib))
        enddo
        ebmax = ebmax/atw_beam
        debin = ebmax*ebfac/dFLOAT(nebin) !YuP[2022-12-19] was FLOAT
c
        do 340 i=1,mfm1
        teev = 1.D3 * zte(i)
        if (iexcit .ne. 0) then
          tiev = 1.D3 * zti(i)
          do j=1,nion
            if (j .le. nprim) then
              znipm(j) = zni(i,j)
            else
              zniim(j-nprim) = zni(i,j)
            endif
          enddo
        endif !(iexcit .ne. 0)
c
c       electron impact ionization independent of rotation speed
c
        do ib=ibstart,mb
          do ie=1,ke
            ind = ke*(ib-1) + ie
            sgxne(ind) = fsgxne(vbeam(ie,ib),teev,zne(i))
          enddo
        enddo
        !Here ind= ke*(mb-1) + ke
c
        do 340 j=1,nebin
        ebin = dFLOAT(j) * debin !YuP[2022-12-19] was FLOAT
        eova = 1.D3 * ebin
        sgxn(1,i,1,j) = 0.D0
        do kk=1,nion
          sgxn(1,i,1,j) = sgxn(1,i,1,j)
     .                  + fsgxni(atw(kk),eova,zni(i,kk),zzi(i,kk))
          if ((kk .le. nprim) .and. (kk .le. 2.0))
     .      sgxn(kk+1,i,1,j) = fsgxncx(atw(kk),eova,zni(i,kk))
        enddo
        if (iexcit .ne. 0) then
c          call hexnb (istart, 1, ilorent, mstate, ncont, eova,
c     .                teev, tiev, nprim, atwpm, znipm, nimp, iz,
c     .                atwim, izstrp, zniim, bperp, kdene, kdeni,
c     .                kdenz, ksvi, ksvz, ksve, krad, ngh, ngl, nouthx,
c     .                ncorin, rerate, rmfp, hexfrac, ihxerr)
          if (ihxerr .ne. 0) write (ncrt, 1000) ihxerr
          if (ihxerr .eq. 1) then
            write (ncrt, 1010)
            write (nout, 1010)
            call STOP ('subroutine NBSGXN: problem #2', 46)
          endif
        endif

cBH171014: Use of ind below is strange.  It has only been set in 
cBH171014: above do 320 loop, with final value ind=ke*(mb-1)+ke.
cBH171014: It's initial value is ke*(ibstart-1)+1.
cBH171014: OK, I'll take this as final value in following loop.
cBH171014        do 340 k=ind,1,-1
        do 340 k=ind,ke*(ibstart-1)+1,-1  !ind=ke*(ib-1)+ie,beam energy index
          sgxn(1,i,k,j) = sgxn(1,i,1,j) + sgxne(k)
          sgxn(4,i,k,j) = sgxn(1,i,k,j) + sgxn(2,i,1,j) + sgxn(3,i,1,j)
          sgxn(1,i,k,j) = sgxn(1,i,k,j)/sgxn(4,i,k,j)
          sgxn(2,i,k,j) = sgxn(2,i,1,j)/sgxn(4,i,k,j)
          sgxn(3,i,k,j) = sgxn(3,i,1,j)/sgxn(4,i,k,j)
          if (iexcit .ne. 0)  sgxn(4,i,k,j) = 1.D0/rmfp
  340   continue !i=1,mfm1 ; j=1,nebin ; k=ind,ke*(ibstart-1)+1,-1
        !YuP: Same label for different loops - ok?
  
      endif !nebin 
c
 1000 format (' subroutine NBSGXN reports a HEXNB return code of ', i5)
 1010 format (' ERROR: execution terminated - file "coronb" not found')
c
      return
c
      end subroutine NBSGXN
      
c=======================================================================
      real*8 function fsgxncx (atw, e, zni)
c
      implicit integer (i-n), real*8 (a-h, o-z)
c
c ----------------------------------------------------------------------
c this subprogram calculates inverse mean free path due to charge exchange
c from the fitted results of freeman and jones, clm-r137, culham (1974).
c
c     input:
c             atw - atomic weight of target ion
c             e   - relative energy of impinging neutral (ev/amu)
c             zni - density of target ion (cm**-3)
c ----------------------------------------------------------------------
c
      if (atw .gt. 3.01) then
        sigcx = 0.0
      else
        aloge = LOG10 (e)
        sigcx = 0.6937D-14 * (1.0 - 0.155*aloge)**2 /
     .                       (1.0 + 0.1112D-14*e**3.3)
      endif
      fsgxncx = sigcx*zni
c
      return
      end function fsgxncx

      real*8 function fsgxne (vb, te, zne)
c
      implicit  integer (i-n), real*8 (a-h, o-z)
c
c ----------------------------------------------------------------------
c this subprogram evaluates local inverse mean free path for electron impact
c ionization from fitted results of freeman and jones, clm-r137,culham (1974).
c
c     input:
c             vb  - speed of impinging neutral (cm/sec)
c             te  - target electron temperature (ev)
c             zne - target electron density (cm**-3)
c ----------------------------------------------------------------------
c
      dimension cfione(7)
      data      cfione /-3.173850D+01,  1.143818D+01, -3.833998    ,
     .                   7.046692D-01, -7.431486D-02,  4.153749D-03,
     .                  -9.486967D-05/
c
      alogt = 0.0
      if (te .gt. 1.0    )  alogt = LOG (te)
      if (te .gt. 1.0D+05)  alogt = 11.51
      expo = (((((cfione(7) *alogt + cfione(6))*alogt + cfione(5))*alogt
     .          + cfione(4))*alogt + cfione(3))*alogt + cfione(2))*alogt
     .          + cfione(1)
      fsgxne = EXP (expo) * zne / vb
      return
c
      end
c
      real*8 function fsgxni (atw, eova, zni, zzi)
c
      implicit  integer (i-n), real*8 (a-h, o-z)
c
c ----------------------------------------------------------------------
c  this subprogram calculates inverse mean free path due to proton and
c     impurity impact ionization.  proton impact ionization cross
c     sections from fitted results of freeman and jones,clm-r137, culham
c     (1974).  impurity impact ionization cross sections from r.e. olson
c     et al., Phys. Rev. Lett. 41, 163 (1978).
c
c     input:
c             atw  - atomic weight of target ion
c             eova - relative energy of impinging neutral (ev/amu)
c             zni  - density of target ion (cm**-3)
c             zzi  - average charge state of target ion
c ----------------------------------------------------------------------
c
      dimension cfionp(7)
      data      cfionp /-4.203309D+01,  3.557321    , -1.045134,
     .                   3.139238D-01, -7.454475D-02,  8.459113D-03,
     .                  -3.495444D-04/
c
      if (atw .le. 3.01) then
        aloge = LOG10 (eova)
        aloge = aloge * 2.302585093 - 6.907755279
        if (aloge .le. -2.30258) then
          sigi = 0.0
        else
          expo = (((((cfionp(7) *aloge + cfionp(6))*aloge
     .              + cfionp(5))*aloge + cfionp(4))*aloge
     .              + cfionp(3))*aloge + cfionp(2))*aloge
     .              + cfionp(1)
          sigi = EXP (expo)
        endif
        fsgxni = sigi*zni
      else
        ekev   = 1.0D-3*eova
        fsgxni = 1.0D-17*zni*46.0*zzi*(32.0*zzi/ekev)*
     .              (1.0 - EXP (-ekev/(32.0*zzi)))
      endif
      return
c
      end
c
c
      subroutine getsgxn (e, iz, ns, ktk, ib, ie, sgxn, nbins,
     .                    debin, kbe, ksge, kz, nout, sgxnloc)
      !Only for nebin.ne.0 (Tor. rot. case)
c
      implicit integer (i-n), real*8 (a-h, o-z)
c
c ----------------------------------------------------------------------
c this subroutine locates n*sigma in the lookup table, sgxn
c ----------------------------------------------------------------------
c
c --- input through argument list:
c          debin          - energy bin width     (keV/amu)
c          e(ns)          - energy to be located (keV/amu)
c          ib             - beam line index
c          ie             - beam energy group index
c          iz(ns)         - flux zone index
c          nbins          - total number of energy bins
c          ns             - total number of points to be evaluated
c          sgxn(i,iz,j,k) - array of neutral stopping data
c                           i  - data type index
c                                =1, fraction of interactions producing
c                                    electrons;
c                                =2, fraction of interactions producing
c                                    neutral species 1; (via CX)
c                                =3, fraction of interactions producing
c                                    neutral species 2; (via CX)
c                                =4, inverse mean free path (cm**-1)
c                           iz - flux zone index
c                           j  - beam/energy index, j = 3*(ib-1)+ie
c                           k  - energy bin index
c
c --- output through argument list:
c          sgxnloc(i=1:4) - local neutral stopping data (cm**-1)
c                           i - data type index (see above)
c                           note:  if ns>1, only the inverse mean free
c                           path is evaluated (i = 4) and ns data points
c                           are passed.  this facilitates faster
c                           execution when evaluating the max imfp along
c                           the collisionless neutral trajectory (used
c                           in rotating discharges only).  otherwise
c                           more detailed information is passed in the
c                           first four elements only (ns = 1).
c ----------------------------------------------------------------------
c
c   Created:    Glenn Sager?               Date:  ??-???-????
c
c   changes from original version:
c      1) The code will now terminate if the maximum rotational energy
c         bin is exceeded. Detailed Error message provide to both the
c         OUTONE file (nout) and the standard output device (ncrt = 6).
c      2) Fixed some math in array index calculations.
c      3) Today is the 50th Aniversary of Hiroshima. NEVER AGAIN!
c
c                                  D.F. Finkenthal 6-AUG-95
c
c ----------------------------------------------------------------------
c
      real*8 e(ns) !YuP[2019-06-17]: was e(ktk)   BUG?
      real*8 sgxn(4,kz,kbe,ksge)
      real*8 sgxnloc(4) !YuP[2022-12-15] Was (kbe)
      integer iz(ns) !YuP[2019-06-17]: was iz(ktk)   BUG?
      integer imaxa(200)
      integer,save :: imax 
c
      ncrt = 6
c
      ind = 3 * (ib - 1) + ie
      if (ns .gt. 1) then
        !note:  if ns>1, only the inverse mean free
        ! path is evaluated (i = 4) and ns data points
        ! are passed.  this facilitates faster
        ! execution when evaluating the max imfp along
        ! the collisionless neutral trajectory (used
        ! in rotating discharges only).  otherwise
        ! more detailed information is passed in the
        ! first four elements only (ns = 1).
        do i=1,ns
          ibin       = e(i) / debin + 1.0
          imaxa(i)   = ibin
          ibin       = MIN0 (ibin, nbins)
          !YuP sgxnloc(i) = sgxn (4, iz(i), ind, ibin)
          !YuP[2022-12-15] Probably a BUG? Maybe sgxnloc(4)=...
          !YuP[2023-07-11] In other parts of zfreya, the usage is
          !sgxn(1:4,1:kz,1:kbe,1:ksge), where kbe=kb*ke,ksge=20
          if(ibin.gt.ksge) then
            WRITE(*,*)'subr.getsgxn(a): ibin,ksge=',ibin,ksge
            STOP 'subr.getsgxn ibin>ksge, Increase ksge' !YuP[2023-07-11]added
          endif
          sgxnloc(4) = sgxn(4, iz(i), ind, ibin) !YuP[2023-07-11]
        end do
        imax = maxaf (imaxa, 1, ns)
      else    !ns=1    
        ibin       = e(1) / debin + 1.0
        imax       = MAX0 (ibin, imax )
        ibin       = MIN0 (ibin, nbins)
        if(ibin.gt.ksge) then
           WRITE(*,*)'subr.getsgxn(b): ibin,ksge=',ibin,ksge
           STOP 'subr.getsgxn ibin>ksge, Increase ksge' !YuP[2023-07-11]added
        endif
        !YuP[2023-07-11] In other parts of zfreya, the usage is
        !sgxn(1:4,1:kz,1:kbe,1:ksge), where kbe=kb*ke,ksge=20
        sgxnloc(1) = sgxn(1, iz(1), ind, ibin) !here: getsgxn, ns=1
        sgxnloc(2) = sgxn(2, iz(1), ind, ibin) !here: getsgxn, ns=1
        sgxnloc(3) = sgxn(3, iz(1), ind, ibin) !here: getsgxn, ns=1
        sgxnloc(4) = sgxn(4, iz(1), ind, ibin) !here: getsgxn, ns=1
      end if
c      write(*,*) 'sgxn(4,22,3,2) = ',sgxn(4,22,3,2)
c      if(ie.eq.2) stop
c
      if (imax .gt. nbins) then
        emax = amaxaf (e, 1, ns)
        write (nout, 110)  nbins * debin, emax
        write (ncrt, 110)  nbins * debin, emax
        call STOP ('subroutine GETSGXN: max bin energy exceeded', 51)
      end if
c
 110  format (/
     . ' ERROR in GETSGXN: maximum rotational energy bin exceeded'     /
     . ' The highest calculated energy bin is now ', f10.2, ' keV/amu' /
     . ' but GETSGXN tried to look up a value of  ', f10.2, ' keV/amu' /
     . ' The FE_TK input parameter must be increased to overcome',
     . ' this problem!'                                                /
     . ' ONETWO will be terminated to avoid further problems.')
c
      return
c
      end subroutine getsgxn
c
c
      subroutine adassgxn (namep,namei,mb,mfm1,nprim,nimp,nion,atw,
     .                     ebkev,ebfac,ibion,nebin,vbeam,zne,zni,zte,
     .                     zti0,zzi,debin,sgxn,sgxnmi,atw_beam,ibstart)
c
      implicit  integer (i-n), real*8 (a-h, o-z)
CMPIINSERT_INCLUDE
c
c      character what_id*45
c      save      what_id
c      data      what_id/"@(#)cray321.f      12 Feb 98  General Atomics"/
c
c ----------------------------------------------------------------------
c
c  ADASSGXN calculates the effective neutral beam stopping cross sections
c  using the JET Atomic Data and Analysis Structure (ADAS). The cross
c  sections are returned in array sgxn for each beam, beam energy
c  component, and FREYA flux zone.
c
c  The plasma ions are assumed to be fully stripped. Only ion species
c  H, He, B, Be, C, O, N, and Ne are available from ADAS. Any other ion
c  will terminate the code. This can be improved later.
c
c  The routine was made to be as compatible as possible with the existing
c  code. A call is made to the original cross section package in order to
c  determine relative deposition fractions only. The HEXNB package is
c  totally avoided.
c
c  Reference:   Finkenthal, Ph.D. Thesis, UC-Berkeley, 1994
c
c  Created  :   06-jul-1995    by  D. Finkenthal
c ----------------------------------------------------------------------
c
c  Input:
c
c     nprim,nimp,nion
c     atw
c     ebkev(mb)      - full energy of mb-th neutral beamline
c                      (keV)
c     ebfac          - factor defining upper bound on energy bin
c                      range,  .gt. 1.0 (cqlinput: set fe_tk for ebfac)
c     ibion          - index of beam ion species
c                      (e.g. atwb = atw(ibion))
c                      if ibion = -1 beam is dt mixture
c     atw_beam         (use atw_beam for mass in this case)
c     mb             - number of beamlines modeled
c     mfm1           - number of flux zones minus one
c     nebin          - number of energy bins (rotation case only)
c                      (cqlinput: set ne_tk for nebin)
c     vbeam(ie,mb)   - speed of ie-th energy group of the mb-th
c                      beamline (cm/sec)
c     zne(mfm1)      - local electron density (cm**-3)
c     zni(mfm1,nion) - local density of nion-th ion species
c                      (cm**-3)
c     zte(mfm1)      - local electron temperature (KeV)
c     zti0(mfm1)     - local ion temperature (KeV)
c     zzi(mfm1,nion) - local average charge state of nion-th ion
c                           species
c     ONETWO stuff:
c     input from common /io/:
c          ncrt,nout
c     input from common /ions/:
c          namei, atw
c     input from common /numbrs/:
c          nprim,nimp,nion
c     END ONETWO stuff
c
c  Output:
c
c      sgxn(i,j,k,l)
c        i - mode index
c            = 1, fraction of reactions producing electrons;
c            = 2, fraction of reactions producing species 1 ion;
c            = 3, fraction of reactions producing species 2 ion;
c            = 4, total inverse mean free path;
c        j - FREYA zone index
c        k - beam index
c            k = 3*(ib-1) + ie, where ib is the beam index and ie
c                is the beam energy group index
c        l - index for relative energy.  bins are equispaced in
c            delta(energy) between 0 and max(ebkev(ib))*ebfac,
c            with bin width given by
c                   delta(energy) = ebmax*ebfac/nebin.
c            ebfac.ge.1.0 and nebin are user given namelist variables.
c      sgxnmi(ie,mb)
c         - minimum inverse mean free path for ie-th energy
c           group of mb-th beamline.  calculated for stationary
c           target case only.  otherwise, calculated for each
c           neutral trajectory in subroutine INJECT.
c      debin
c         - width of energy bins (keV/amu)
c           Used only for rotatation case.
c ----------------------------------------------------------------------
c
c       include 'params.inc'
      include 'param.h'
c
      character*8 namei(kimp), namep(kprim)   ! added for cql3d
      integer mb, mfm1, nion, nprim
      real*8 ebkev(kb),vbeam(ke,kb)
      real*8 zne(kz),zni(kz,kion),zte(kz),zti0(kz),zzi(kz,kion)
      real*8 sgxn(kcmp1,kz,kbe,ksge),sgxnmi(ke,kb)  ! stand alone code
c      real*8 sgxn(kcmp1,kz,ke,kb),sgxnmi(ke,kb)
c
      real*8 ebfac, debin, atw_beam, atw(kion)
      real*8 sgxeff(3), cnz(kz,20), zeffx(kz), izatom(kion)
      save init, izatom, atwb, izbm
      data init /0/
c
      do i=1, kz
        do k=1,20
          cnz(i,k) = 0.D0
        enddo
      enddo
c
      if (ibstart.eq.1) then 
!      do i=1,kz
!       do j=1,ke
!         do k=1,kb
!          sgxn(1,i,j,k)=0.D0
!          sgxn(2,i,j,k)=0.D0
!          sgxn(3,i,j,k)=0.D0
!          sgxn(4,i,j,k)=0.D0 !YuP: BUG? 
!               !See above declaration: real*8 sgxn(kcmp1,kz,kbe,ksge)
!         enddo
!       enddo
!      enddo
          sgxn=0.d0 !YuP[2022-12-15] replaced the above.
      endif
c
      do i=1,3
       sgxeff(i)=0.D0
      enddo
      
c
      if (init .eq. 0) then
c
c ----------------------------------------------------------------------
c determine atomic number of primary ions
c ----------------------------------------------------------------------
c
      do i=1,nprim
        izatom(i) = 1 ! for H,D,T
        if (trim(namep(i)).eq.'he'   .or. 
     +      trim(namep(i)).eq.'HE'   .or. 
     +      trim(namep(i)).eq.'He') izatom(i) = 2
      enddo
c
c ----------------------------------------------------------------------
c determine atomic number of impurity ions
c ----------------------------------------------------------------------
c
      if (nimp .eq. 0)  go to 3430
      do i=1,nimp
        k = nprim + i
        izatom(k) = 0
        if(trim(namei(i)).eq.'he' .or. 
     +     trim(namei(i)).eq.'HE' .or. 
     +     trim(namei(i)).eq.'He')  izatom(k) =  2
     
        if(trim(namei(i)).eq.'b' .or. 
     +     trim(namei(i)).eq.'B' )  izatom(k) =  5  ! YuP added [2015]
        
        if(trim(namei(i)).eq.'c' .or. 
     +     trim(namei(i)).eq.'C' )  izatom(k) =  6
        
        if(trim(namei(i)).eq.'o' .or. 
     +     trim(namei(i)).eq.'O' )  izatom(k) =  8
        
        if(trim(namei(i)).eq.'si' .or. 
     +     trim(namei(i)).eq.'SI' .or. 
     +     trim(namei(i)).eq.'Si')  izatom(k) = 14
        
        if(trim(namei(i)).eq.'ar' .or. 
     +     trim(namei(i)).eq.'AR' .or. 
     +     trim(namei(i)).eq.'Ar')  izatom(k) = 18
        
        if(trim(namei(i)).eq.'cr' .or. 
     +     trim(namei(i)).eq.'CR' .or. 
     +     trim(namei(i)).eq.'Cr')  izatom(k) = 24
        
        if(trim(namei(i)).eq.'fe' .or. 
     +     trim(namei(i)).eq.'FE' .or. 
     +     trim(namei(i)).eq.'Fe')  izatom(k) = 26
        
        if(trim(namei(i)).eq.'ni' .or. 
     +     trim(namei(i)).eq.'NI' .or. 
     +     trim(namei(i)).eq.'Ni')  izatom(k) = 28
        
        if(trim(namei(i)).eq.'kr' .or. 
     +     trim(namei(i)).eq.'KR' .or. 
     +     trim(namei(i)).eq.'Kr')  izatom(k) = 36
        
        if(trim(namei(i)).eq.'mo' .or. 
     +     trim(namei(i)).eq.'MO' .or. 
     +     trim(namei(i)).eq.'Mo')  izatom(k) = 42
        
        if(trim(namei(i)).eq.'w' .or. 
     +     trim(namei(i)).eq.'W' )  izatom(k) = 74
      enddo
c
c Get beam species
c
 3430 if (ibion .gt. 0) then
        atwb = atw(ibion)
        izbm = izatom(ibion)
      else
        atwb = atw_beam
        izbm = 1
      endif
c
c ----------------------------------------------------------------------
c... Check to make sure that the impurities requested are compatable
c    with ADAS (i.e., H, He, B, Be, C, O, N, or Ne). Terminate with
c    error if an invalid impurity is listed.
c ----------------------------------------------------------------------
c
      do k=1,nion
        if (izatom(k) .gt. 10) then
          write (ncrt, 1010)
          write (nout, 1010)
          call STOP ('subroutine ADASSGXN: unallowed impurity', 181)
        end if
 1010 format
     .   (' *** Execution Terminated:'                                 /
     .    '     The ADAS database only contains atomic cross-sections' /
     .    '     for fully-stripped H, He, B, Be, C, O, N, and Ne ions.'/
     .    '     You must restrict your choice of impurity species to'  /
     .    '     these ions.')
      end do
c
      init = 1
      endif    ! init
c
c ----------------------------------------------------------------------
c... Get original cross sections-Used to determine relative deposition
c    fractions only [YuP: does it refer to sgxn(1:3)?].
c    Total [YuP:sgxn(4)?] cross section is determined using ADAS [subr.adasqh6].
c ----------------------------------------------------------------------
c
      !write(*,'(a,1p3e11.2)') '   bef.nbsgold sgxnmi=',sgxnmi(1:ke,1)
      ![Evaluate old cross sections based on Freeman&Jones,
      ! just for comparison with the newer ADAS]:
      call nbsgold (mb,mfm1,nebin,nprim,nimp,nion,atw,ebkev,
     .              ebfac,ibion,vbeam,zne,zni,zte,zzi,
     .              debin,sgxn,sgxnmi,atw_beam,ibstart)
      !write(*,'(a,1p3e11.2)') '   aft.nbsgold sgxnmi=',sgxnmi(1:ke,1)
      do j=1,3
      write(*,'(a,4f8.5)')' adassgxn/aft.nbsgold: sgxn(1:4,1,ie)=',
     &     sgxn(1:4,1,j,1) !Note: sgxn(4,*) is recomputed by adasqh6().
      enddo
c
c ----------------------------------------------------------------------
c... Set up the cnz arrays (concentrations of plasma ions) and Zeffx
c    zni and zne are the (FREYA zone) densities of electron and ions:
c ----------------------------------------------------------------------
c
      cnz= 0.D0 !YuP[2022-12-20] Initialize ALL values to 0. 
      !(was cnz(i,k)=0 inside the k=1,nion loop below, 
      !which means - not for all k=1:10)  
      do i=1, mfm1
        do k=1, nion
          !cnz(i,k)=0.D0 !YuP[2022-12-20] BUG - not all k. See above.
          if (izatom(k) .eq.  1)  cnz(i, 1) = zni(i,k)/zne(i)
          if (izatom(k) .eq.  2)  cnz(i, 2) = zni(i,k)/zne(i)
          if (izatom(k) .eq.  4)  cnz(i, 4) = zni(i,k)/zne(i)
          if (izatom(k) .eq.  5)  cnz(i, 5) = zni(i,k)/zne(i)
          if (izatom(k) .eq.  6)  cnz(i, 6) = zni(i,k)/zne(i)
          if (izatom(k) .eq.  7)  cnz(i, 7) = zni(i,k)/zne(i)
          if (izatom(k) .eq.  8)  cnz(i, 8) = zni(i,k)/zne(i)
          if (izatom(k) .eq. 10)  cnz(i,10) = zni(i,k)/zne(i)
c          write(*,*) 'cnz = ',i,k,cnz(i,k)
        enddo
c
c The Zeff(i=1,...mfm1) array has been stored in zzi(i,nion+1)
        zeffx(i) = zzi(i,nion+1)
c
      enddo
c
c ----------------------------------------------------------------------
c     stationary plasma case
c ----------------------------------------------------------------------
c
      if (nebin .eq. 0) then
        ierr = 0
        do i=1,mfm1
          do ib=ibstart,mb
            tiev = 1.D3 * zti0(i)
            ecol = 1.D3 * ebkev(ib) / (atwb)
            call adasqh6 (tiev,ecol,izbm,0,zne(i),zeffx(i),
     .                    cnz(i,2),cnz(i,4),cnz(i,5),cnz(i,6),
     .                    cnz(i,7),cnz(i,8),cnz(i,10),
     .                    sgxeff(1:3),ierr) !sgxeff(1:3)==qrat(1:3)
            if (ierr .eq. 1) then
              call STOP ('subroutine ADASSGXN: problem #1', 183)
            end if
            do ie = 1,ke
              ind  = ke*(ib-1) + ie
              sgxn(4,i,ind,1) = zne(i)*sgxeff(ie)
!              write(*,'(a,2i4,1p3e11.2)') 
!     &         '   adassgxn: i,ie,sgxnmi,zne(i),sgxeff(ie)=',
!     &          i,ie,sgxnmi(ie,ib),zne(i),sgxeff(ie)
              sgxnmi(ie,ib)= MAX(sgxnmi(ie,ib),sgxn(4,i,ind,1))
            enddo !ie
          enddo !ib
        enddo ! i
        
        !For printout of cross-sections (fractions of total)
c  sgxn(i,j,k,l)
c  i - mode index
c    = 1, fraction of reactions producing electrons 
!         (impact e+neut -> 2e+i,  impact ion+neut -> ion+ion+e)
c    = 2, fraction of reactions producing species 1 ion
!         (CX   fast_neutral+ion -> FI+slow_neut)
c    = 3, fraction of reactions producing species 2 ion;
!         (CX   fast_neutral+ion -> FI+slow_neut)
c    = 4, total inverse mean free path (normalized above to include ne)
c  j - FREYA zone index (radial)
c  k - beam&energy index = ke*(ib-1) + ie
c  l - related to nebin.ne.0 (normally nebin=0, then l=1 only)
        
CMPIINSERT_IF_RANK_EQ_0     
        !Just a printout of Impact and CX rates, for each ib and ie 
        do ib= ibstart,mb
        do ie= 1,ke
           ind= ke*(ib-1) + ie
          WRITE(*,'(a,2i2,3f8.5)')
     &    ' adassgxn/aft.adasqh6: ib,ie,sgxn(1:3,ir=1,ind,1)=',
     &                            ib,ie,sgxn(1:3,1,ind,1)
        enddo !ie
        enddo !ib
CMPIINSERT_ENDIF_RANK
c
      else !nebin>0
c
c ----------------------------------------------------------------------
c     rotating plasma case
c ----------------------------------------------------------------------
c
        ebmax = 0.D0
        do ib=ibstart,mb
          ebmax = MAX (ebmax,ebkev(ib))
        end do
        ebmax = ebmax/atwb
        debin = ebmax*ebfac/dFLOAT(nebin) !YuP[2022-12-19] was FLOAT
c
        do i=1,mfm1
          tiev  = 1.D3*zti0(i)
          jreff = 0
          do j=1,nebin
            ebin = dFLOAT(j) * debin !YuP[2022-12-19] was FLOAT
            ecol =     1.D3 *  ebin
c
c... Only call ADAS if beam energy (ecol) is greater than 5 keV/amu.
c    Skip over the energy bins that are less than 5 keV/amu. These
c    low energy bins will be calculated using the old cross sections
c    scaled to the adas data at some reference energy.
c
            if (ecol .ge. 5000.0) then
              call adasqh6 (tiev,ecol,izbm,1,zne(i),zeffx(i),
     .                      cnz(i,2),cnz(i,4),cnz(i,5),cnz(i,6),
     .                      cnz(i,7),cnz(i,8),cnz(i,10),
     .                      sgxeff,ierr)
              if (ierr .eq. 1) then
                call STOP ('subroutine ADASSGXN: problem #2', 184)
              endif
c
c... Calculate a scaling factor from the first available (ge. 5.0)
c    ADAS bin energy This will be used to scale the old cross
c    sections to fit in with ADAS for the low energy bins.
c
              if (jreff .eq. 0) then
                jreff     = j
                sgxnscale = zne(i)*sgxeff(1)/sgxn(4,i,1,j)
              endif
              !YuP/commented do k=ke*ibstart,ke*mb
              do k= ke*(ibstart-1)+1, ke*mb
              !YuP[2018-01-24] Originally (one beam species), it was k=1,ke*mb
              !Maybe now it should be ke*(ibstart-1)+1 ?
              !Then when ibstart=1, k starts with 1, as before.
              !Note: mb=(ibstart-1)+nbeams1(kbsp) 
                sgxn(4,i,k,j) = zne(i)*sgxeff(1)
              enddo
c
            endif  ! if ecol
c
          enddo ! j-loop
c
c... Now scale the old cross sections to match the first available ADAS
c    datapoint for the energy bins below 5.0 keV/amu
c
          if (jreff .gt. 1) then
            do  j=1,jreff-1
              !YuP/commented do k=ke*ibstart,ke*mb
              do k= ke*(ibstart-1)+1, ke*mb
              !YuP[2018-01-24] Originally (one beam species), it was k=1,ke*mb
              !Maybe now it should be ke*(ibstart-1)+1 ?
              !Then when ibstart=1, k starts with 1, as before.
              !Note: mb=(ibstart-1)+nbeams1(kbsp) 
               sgold=sgxn(4,i,k,j)
               sgxn(4,i,k,j) = sgxn(4,i,k,j)*sgxnscale
             enddo
            enddo
          endif
c
        enddo  ! i-loop
      endif  ! if nebin
c
c      do ib=1,10
c       do j=1,kbe
c         write(*,'(2i3,2x,1p4e12.4,a12)') ib,j,sgxn(4,ib,j,1),
c     .        sgxn(4,ib,j,2),sgxn(4,ib,j,3),sgxn(4,ib,j,4),' sgxn'
c       enddo
c      enddo
c
      return
      end subroutine adassgxn
c
c
      subroutine adasqh6 (ti,ecol,bmz,iflag,ne,zeff,conche,
     .                    concbe,concb,concc,concn,conco,concne,
     .                    qrat,ierr)
c
c  Calculate and return the rate coefficient for beam stopping
c  This routine is a modified version of L.D. Horton's (JET)
c  QHIOCH6 routine used to access and evaluate the ADAS beam
c  stopping cross sections from ion-specific effective data files.
c
c   changes from QHIOCH6:
c      1) The need for the NAG library has been eliminated using
c         the spline routines SPLINE and SEVAL.
c      2) Only the full-energy component can be calculated by
c         setting iflag=1. This allows lower energies to be
c         read from the ion-specific files.
c
c                                  D.F. Finkenthal 6-JUL-95
c
c   changes from QHIOCH5:
c      1) modified to force rereading of input files when beam
c         species has changed from the last call (using ipass)
c
c                                  L.D. Horton    2 June 92
c
c   changes from QHIOCH4:
c      1) modified to accept beam species as input and to then
c         calculate the beam stopping for either hydrogen or
c         helium beams
c
c                                  L.D. Horton   16 August 91
c
c   changes from QHIOCH3:
c      1) incorporated Fritsch's new calculation for excitation by
c         Be4+ and He2+
c      2) improved calculation of Maxwellian averages in bundled-n
c         code
c      3) proper inclusion of beam energy dependence
c          - stopping rate is now read from a matrix on beam energy
c            and target density; only the temperature dependence is
c            done separately
c      4) included boron, nitrogen and neon as input concentrations
c          - the code skips all zero concentrations
c
c                                  L.D. Horton   31 July 91
c
c   also:  back to Wilhelm's 3 energies-at-a-time so that all spline
c fits can be done at once. Only if the beam energy has changed by
c more than 1% will the density splines be redone.  Since only one
c energy is used per bank, this means that the spline will be done
c only twice per call of ATTS4.
c
c                                  L.D. Horton   14 August 91
c
c ti        : REAL   : ion temperature in eV
c ecol      : REAL   : collision (=beam) energy in eV/amu
c bmz       : INTEGER: beam nuclear charge
c iflag     : INTEGER: flag for full energy calculation only
c ne        : REAL   : electron density in m**-3
c conche    : REAL   : relative Helium concentration   ( 0 < CHE < 1)
c concbe    : REAL   : relative Beryllium concentration
c concb     : REAL   : relative Boron concentration
c concc     : REAL   : relative Carbon concentration
c concn     : REAL   : relative Nitrogen concentration
c conco     : REAL   : relative Oxygen concentration
c concne    : REAL   : relative Neon concentration
c qrat(3)   : REAL   : requested cross section m**2 for full, half,
c                      and third energy beam components
c
c
c      USE param
c      USE io 
c      USE ext_prog_info, only : nchars_12,onetwo_xsct
c
      implicit none ! integer (i-n), real*8 (a-h, o-z)
c      include 'params.inc'
       include 'param.h'
c      include 'io.i'
CMPIINSERT_INCLUDE

c
c ipass : file read switch.  Reread files if beam species has changed
c
      integer    ipass, nchars_12
      character*256 code_xsct
      character  dsn(8,2)*35
      data       ipass/0/
      data       dsn  / 'data/adas/h_h1.dat'  ,
     .                  'data/adas/h_he2.dat' ,
     .                  'data/adas/h_be4.dat' ,
     .                  'data/adas/h_b5.dat'  ,
     .                  'data/adas/h_c6.dat'  ,
     .                  'data/adas/h_n7.dat'  ,
     .                  'data/adas/h_o8.dat'  ,
     .                  'data/adas/h_ne10.dat',
     .                  'data/adas/he_h1.dat' ,
     .                  'data/adas/he_he2.dat',
     .                  'data/adas/he_be4.dat',
     .                  'data/adas/he_b5.dat' ,
     .                  'data/adas/he_c6.dat' ,
     .                  'data/adas/he_n7.dat' ,
     .                  'data/adas/he_o8.dat' ,
     .                  'data/adas/he_ne10.dat'/
c
c Physics Constants
c
      real*8 amu, eV
      parameter (amu = 1.6605D-24)
      parameter (eV  = 1.6022D-12)
c
c Local variables
c
c      integer    bmz, iflag, nebeam
      integer    bmz, iflag
      integer,save :: nebeam
      real*8     ti, ecol, ne, qrat(3)
      real*8 zeff  !IN
      real*8 zeff8 !local
      integer ncrt,nout,ierror,nunadas !local
      integer LENGTH ! integer function
      real*8     conche, concbe, concb, concc, concn, conco, concne
      integer    nsp, maxe, maxn, maxt, ierr
      parameter (nsp = 8) ! 8 different ion species available
!YuP/was      parameter (maxe=15, maxn=10, maxt=10) !Grids: energy,density,temperature
      parameter(maxe=13, maxn=10, maxt=9) !Grids: energy,density,temperature
      !From printout:
      ! ADASQH6: neb(isp)=            13
      ! ADASQH6: ndens(isp)=          10
      ! ADASQH6: ntemp(isp)=           9
      integer    isp, z(nsp), neb(nsp), ndens(nsp), ntemp(nsp), i, j, k
      real*8     eb(maxe,nsp), dens(maxn,nsp), temp(maxt,nsp)
      real*8     tref(nsp), ebref(nsp), denref(nsp), svref(nsp)
      real*8     sven(maxe,maxn,nsp), svt(maxt,nsp), seval
      character  line*80
      data       z /1, 2, 4, 5, 6, 7, 8, 10/
c
c      real*8     ti8, ecol8, ne8, qrat8(3)
      real*8     ti8, ne8, qrat8(3)
      real*8,    save  :: ecol8
      real*8     conc(nsp)
c
      integer ifail !YuP: Not really used
      real*8 be(maxe,maxn,nsp),ce(maxe,maxn,nsp),de(maxe,maxn,nsp)
      real*8 bt(maxt,nsp),ct(maxt,nsp),dt(maxt,nsp)
      real*8 bn(maxn,nsp,3),cn(maxn,nsp,3),dn(maxn,nsp,3)
      real*8 svintn(maxn,nsp,3),svintt,svtot(nsp),svtcor(nsp)
c
      real*8 zeffm1, vbeam
      save neb, ndens, ntemp !(nsp) Grids in ADAS tables
      save z, svref, tref, eb, dens, sven
      save ebref, denref, temp, svt
      save svintn !(maxn,nsp,3)  YuP[2022-12-30] added
      
      save be,ce,de !YuP[2022-12] added: save spline coeffs
      save bt,ct,dt !YuP[2022-12] added: save spline coeffs
      save bn,cn,dn !YuP[2022-12] added: save spline coeffs
c

cBH171118      code_xsct is the name of directory where ADAS directory
cBH171118      data/ is to be found. We first choose to either put this
cBH171118      directory in the local execution directory, or put a
cBH171118      link to it therein. A copy of adas_dir data comes with
cBH171118      the code distribution.
cBH171118      Alternative, cql3d will search /usr/local/ for adas_dir.
      code_xsct="./adas_dir/"
      ierr    = 0
      qrat(1) = 0.d0
      qrat(2) = 0.d0
      qrat(3) = 0.d0
      ti8     = ti
      zeff8   = zeff
      ne8     = ne * 1.0D-13
****  conc(1) = conch
      conc(2) = conche
      conc(3) = concbe
      conc(4) = concb
      conc(5) = concc
      conc(6) = concn
      conc(7) = conco
      conc(8) = concne
      nchars_12 = LENGTH(code_xsct)

      svtcor=0.d0 !YuP[2022-12] added initialization. (local array)
      svtot= 0.d0 !YuP[2022-12] added initialization. (local array)
c
c open and read input file only once
c ** need to save data

c
      ncrt=6 !YuP[2018-01-23] ncrt was not defined here
      nout=6 !YuP[2018-01-23] nout was not defined here
      !-----------
      !ipass=0 !TEST YuP[2022-11-23] - call ADAS anyway (at next freya call)
      !Note: bmz==izbm==izatom(ibion)
      !----------
      if (ipass .ne. bmz) then !YuP: This part is only called once, to read ADAS

      !OUTPUT from spline_12() / INPUT for func.seval() :
      be=0.d0 !YuP[2022-12] added initialization.
      ce=0.d0 !YuP[2022-12] added initialization.
      de=0.d0 !YuP[2022-12] added initialization.
      bt=0.d0 !YuP[2022-12] added initialization.
      ct=0.d0 !YuP[2022-12] added initialization.
      dt=0.d0 !YuP[2022-12] added initialization.
      bn=0.d0 !YuP[2022-12] added initialization.
      cn=0.d0 !YuP[2022-12] added initialization.
      dn=0.d0 !YuP[2022-12] added initialization.     

CMPIINSERT_IF_RANK_EQ_0
        write (ncrt, 1100) 
        write (nout, 1100)
CMPIINSERT_ENDIF_RANK
c
        ecol8 = ecol !saved
c
        nunadas=10001 !YuP/added: any number (except 6) within integer*4 range
        
        do isp=1,nsp
          !call getioun(nunadas,nunadas) !YuP[2022-12]: a dummy subroutine?
          print *,'file =',code_xsct(1:nchars_12)//dsn(isp,bmz) !jmp.den
          open (unit = nunadas,
     .       file = code_xsct(1:nchars_12)//dsn(isp,bmz),status='OLD',
     .       action='read',iostat=ierror)
          write(*,*)' isp=',isp,' ierror=',ierror, ' nunadas=',nunadas
          if (ierror.ne.0) then
             !call getioun(nunadas,nunadas) !YuP: a dummy subroutine?
             print *,'file = ','/usr/local/adas_dir/'//dsn(isp,bmz) !jmp.den
             open (unit = nunadas,
     .       file='/usr/local/adas_dir/'//dsn(isp,bmz),status='OLD',
     .       action='read',iostat=ierror)
             if (ierror.ne.0) then
                WRITE(*,*)
                WRITE(*,*) 'Error finding adas_dir: It is neither in'
                WRITE(*,*) 'the execution directory, or in /usr/local .'
                WRITE(*,*) 'Refer to zfreya.f for more information'
                stop
             endif
          endif
          read (nunadas,  1000) z(isp),svref(isp)
          read (nunadas, '(a)') line
          read (nunadas,  1001) neb(isp),ndens(isp),tref(isp)
          read (nunadas, '(a)') line
          read (nunadas,  1002) (eb(j,isp),j=1,neb(isp))
          read (nunadas,  1002) (dens(k,isp),k=1,ndens(isp))
          read (nunadas, '(a)') line
          do k=1,ndens(isp)
            read (nunadas, 1002) (sven(j,k,isp),j=1,neb(isp))
          end do
          read  (nunadas, '(a)') line
          read  (nunadas,  1003) ntemp(isp),ebref(isp),denref(isp)
          read  (nunadas, '(a)') line
          read  (nunadas,  1002) (temp(j,isp),j=1,ntemp(isp))
          read  (nunadas, '(a)') line
          read  (nunadas,  1002) (svt(j,isp),j=1,ntemp(isp))
          call giveupus_cql3d(nunadas)
          close (nunadas)
c
          write(*,*)'  ADASQH6: neb(isp)=  ',neb(isp) !should not exceed maxe
          write(*,*)'  ADASQH6: ndens(isp)=',ndens(isp) !should not exceed maxn
          write(*,*)'  ADASQH6: ntemp(isp)=',ntemp(isp) !should not exceed maxt
          if(neb(isp).gt.maxe)then !YuP[2022-12]added this check:
            WRITE(*,*)'  ADASQH6: neb(isp)>maxe   Increase maxe'
            STOP
          endif
          if(ndens(isp).gt.maxn)then !YuP[2022-12]added this check:
            WRITE(*,*)'  ADASQH6: ndens(isp)>maxn   Increase maxn'
            STOP
          endif
          if(ntemp(isp).gt.maxt)then !YuP[2022-12]added this check:
            WRITE(*,*)'  ADASQH6: ntemp(isp)>maxt   Increase maxt'
            STOP
          endif
          
          do k = 1, ndens(isp)
            dens(k,isp) = dens(k,isp) * 1.0D-13
          enddo
c
c spline the data on energy and temperature
c
          do k=1,ndens(isp)
            ifail = 0 !YuP: There is no ifail flag in spline_12 ???
            call spline_12 (neb(isp),eb(1,isp),sven(1,k,isp),
     .                   be(1,k,isp),ce(1,k,isp),de(1,k,isp))
            !OUT: be(maxe,maxn,nsp),ce(maxe,maxn,nsp),de(maxe,maxn,nsp)
            ![ b,c,d spline coeffs for energy grid eb(1:neb),
            !  saved for each species 'isp' and each k-grid point ] 
            if (ifail .ne. 0) then
              write (6, '(a)')  ' spline error #1 in ADASQH6'
              ierr = 1
              return
            endif
          enddo
c
          ifail = 0 !YuP: There is no ifail flag in spline_12 ???
          call spline_12 (ntemp(isp),temp(1,isp),svt(1,isp),
     .                 bt(1,isp),ct(1,isp),dt(1,isp))
            !OUT: bt(maxt,nsp),ct(maxt,nsp),dt(maxt,nsp)
            ![ b,c,d spline coeffs for T-grid temp(1:ntemp),
            !  saved for each species 'isp' ] 
          if (ifail .ne. 0) then
            write (6, '(a)')  ' spline error #2 in ADASQH6'
            ierr = 1
            return
          endif
c
c Determine if only the full energy component is requested
c Default is that all three beam energy components are to be determined
c Only the full energy component is interesting for He beams
c
          nebeam = 3
          if (iflag .gt. 0)  nebeam = 1
          if (  bmz .eq. 2)  nebeam = 1
c
c spline on density for each requested energy component (1 or all 3)
c
          do i=1,nebeam
            do k=1,ndens(isp)
              svintn(k,isp,i) = seval(neb(isp),ecol8/i,eb(1,isp),
     .             sven(1,k,isp),be(1,k,isp),ce(1,k,isp),de(1,k,isp))
            enddo
c
            ifail = 0 !YuP: There is no ifail flag in spline_12 ???
            call spline_12(ndens(isp),dens(1,isp),svintn(1,isp,i),
     .                  bn(1,isp,i),cn(1,isp,i),dn(1,isp,i))
            !OUT: bn(maxn,nsp,3),cn(maxn,nsp,3),dn(maxn,nsp,3)
            ![ b,c,d spline coeffs for density-grid dens(1:ndens),
            !  saved for each species 'isp' and each energy point 'i'] 
            if (ifail .ne. 0) then
              write (6, '(a)')  ' spline error #3 in ADASQH6'
              ierr = 1
              return
            endif
          enddo ! i (energy grid)
        enddo !isp
c
 1000   format (i5, 8x, d9.3)
 1001   format (2i5, 7x, d9.3)
 1002   format (8(1x, d9.3) / 8(1x, d9.3))
 1003   format (i5, 7x, d9.3, 7x, d9.3)
 1100   format (/
     . ' Using ADAS, the effective beam stopping cross sections:' /
     . ' ***** Opening and Reading the Atomic Data Tables *****'  /)
c
        ipass = bmz !YuP: at next call to freya, this portion will be skipped.
c
      else ! ipass
c
c  redo density splines only if beam energy has changed by more than 1%
c
        if (ABS (ecol8-ecol) / ecol8 .gt. 0.01) then
          write(*,*)'  ADASQH6: Recalc. density spline coeffs bn,cn,dn'
          do isp=1,nsp
            ecol8 = ecol
            do i=1,nebeam
              do k=1,ndens(isp)
                svintn(k,isp,i) = seval(neb(isp),ecol8/i,eb(1,isp),
     .                sven(1,k,isp),be(1,k,isp),ce(1,k,isp),de(1,k,isp))
              enddo
              ifail = 0 !YuP: There is no ifail flag in spline_12 ???
              call spline_12 (ndens(isp),dens(1,isp),svintn(1,isp,i),
     .                     bn(1,isp,i),cn(1,isp,i),dn(1,isp,i))
              if (ifail .ne. 0) then
                write (6, '(a)')  ' spline error #4 in ADASQH6'
                ierr = 1
                return
              endif
            enddo
          enddo
        endif
        
      endif ! ipass

c !YuP[2022-12] Moved conc()-related part in front of the rest.
c scale the input concentrations to match the required zeff
c
      zeffm1 = 0.d0
      do isp=2,nsp
         zeffm1 = zeffm1 +dfloat(z(isp)*(z(isp)-1))*conc(isp)
      enddo
      
      if (zeffm1 .gt. 1.0D-5) then
        do isp=2,nsp
           conc(isp) = (zeff8-1.0) / zeffm1 * conc(isp)
        enddo
      endif
      
      conc(1) = 1.d0
      do isp=2,nsp
        conc(1) = conc(1) - dfloat(z(isp))*conc(isp)
      enddo

c
c calculate correction to requested temperature
c
      do isp=1,nsp
        if(conc(isp).gt.0.d0)then !YuP[2022-12-20] added
        if (ti8 .le. temp(1,isp)) then
          svtcor(isp) = svt(1,isp)/svref(isp)
!          write(*,'(a,i3,1p2e11.2)')
!     &    '     adasqh6_a: isp,svt(1,isp),svref(isp)=',
!     &                     isp,svt(1,isp),svref(isp)
        else
          svintt      = seval(ntemp(isp),ti8,temp(1,isp),
     .                  svt(1,isp),bt(1,isp),ct(1,isp),dt(1,isp))
          svtcor(isp) = svintt/svref(isp) ! usually ~1.0
!          write(*,'(a,i3,1p2e11.2)')
!     &    '     adasqh6_b: isp,svintt,svref(isp)=',
!     &                     isp,svintt,svref(isp)
        endif
        endif !(conc(isp).gt.0.d0) !YuP[2022-12-20] added
      enddo !isp
!c
!c scale the input concentrations to match the required zeff
!c
!      zeffm1 = 0.d0
!      do isp=2,nsp
!        if(conc(isp).gt.0.d0)then !YuP[2022-12-20] added
!          zeffm1 = zeffm1 +dfloat(z(isp)*(z(isp)-1))*conc(isp)
!        endif !(conc(isp).gt.0.d0) !YuP[2022-12-20] added
!      enddo
!      
!      if (zeffm1 .gt. 1.0D-5) then
!        do isp=2,nsp
!           conc(isp) = (zeff8-1.0) / zeffm1 * conc(isp)
!        enddo
!      endif
!      
!      conc(1) = 1.d0
!      do isp=2,nsp
!        conc(1) = conc(1) - dfloat(z(isp))*conc(isp)
!      enddo
c
c evaluate at three energy components
c
      do i=1,nebeam
c
c interpolate to requested density
c
        do isp=1,nsp
          if(conc(isp).gt.0.d0)then !YuP[2022-12-20] added:
           !skip species that give no contribution.
          if (ne8 .le. dens(1,isp)) then
            svtot(isp) = seval(ndens(isp),dens(1,isp),dens(1,isp),
     .            svintn(1,isp,i),bn(1,isp,i),cn(1,isp,i),dn(1,isp,i))
          else
            svtot(isp) = seval(ndens(isp),ne8,dens(1,isp),
     .            svintn(1,isp,i),bn(1,isp,i),cn(1,isp,i),dn(1,isp,i))
          endif
          endif !(conc(isp).gt.0.d0) !YuP[2022-12-20] added
        enddo
c
c  construct the total stopping cross section
c
        qrat8(i) = 0.d0
        do isp=1,nsp
          if(conc(isp).gt. 0.d0) then !YuP[2022-12-20] added
           !skip species that give no contribution.
           qrat8(i)= qrat8(i) 
     &              +svtot(isp)*svtcor(isp)*dfloat(z(isp))*conc(isp)
!           write(*,'(a,i3,1p2e11.2)')
!     &     '     adasqh6: isp,svtot(isp),svtcor(isp)=',
!     &                    isp,svtot(isp),svtcor(isp)
          endif !(conc(isp).gt.0.d0) !YuP[2022-12-20] added
        enddo
c
c  divide by the beam speed to get a cross section and convert to m**2
c
        vbeam   = SQRT (2.0 * ecol8 / i * ev / amu)
        qrat(i) = qrat8(i) / vbeam
        !write(*,'(a,i3,1p2e11.2)')  '     adasqh6: i,qrat8(i),vbeam=',
        !&                                             i,qrat8(i),vbeam
      enddo !i=1,nebeam
c
      return
      end subroutine adasqh6
c
c
      subroutine nbsgold (mb,mfm1,nebin,nprim,nimp,nion,atw,ebkev,
     .                    ebfac,ibion,vbeam,zne,zni,zte,zzi,debin,
     .                    sgxn,sgxnmi,atw_beam,ibstart)
c
c
c ----------------------------------------------------------------------
c
c     This subroutine calculates the neutral beam attenuation array sgxn
c     using the old cross sections based on Freeman & Jones (1972).
c     No excited state effects are included. These cross sections have
c     been found to be outdated and are not to be used except for purposes
c     of comparison to the newer ADAS data.
c
c     Created:     12-JUL-1995     Daniel Finkenthal
c
c     input:
c
c          ebkev(mb)      - maximum energy of mb-th neutral beamline
c                           (keV)
c          ebfac          - factor defining upper bound on energy bin
c                           range,  > 1.0 (cqlinput: set fe_tk for ebfac)
c          ibion          - index of beam ion species
c                           (e.g., atwb = atw(ibion))
c          mb             - number of beamlines modeled
c          mfm1           - number of flux zones minus one
c          nebin          - number of energy bins (rotation case only)
c                           (cqlinput: set ne_tk for nebin)
c          vbeam(ie,mb)   - speed of ie-th energy group of the mb-th
c                           beamline (cm/sec)
c          zne(mfm1)      - local electron density (cm**-3)
c          zni(mfm1,nion) - local density of nion-th ion species
c                           (cm**-3)
c          zzi(mfm1,nion) - local average charge state of nion-th ion
c                           species
c     input from common /io/:
c          ncrt,nout,nouthx,ncorin
c     input from common /ions/:
c          namei, atw
c     input from common /numbrs/:
c          nprim,nimp,nion
c
c     output:
c          sgxn(i,j,k,l)
c               i - mode index
c                   = 1, fraction of reactions producing electrons;
c                   = 2, fraction of reactions producing species 1 ion;
c                   = 3, fraction of reactions producing species 2 ion;
c                   = 4, total inverse mean free path;
c               j - FREYA zone index
c               k - beam index
c                   k = 3*(ib-1) + ie, where ib is the beam index and ie
c                                      is the beam energy group index
c               l - index for relative energy.  bins are equispaced in
c                   delta(energy) between 0 and max(ebkev(ib))*ebfac,
c                   with bin width given by
c                             delta(energy) = ebmax*ebfac/nebin.
c                   ebfac .ge. 1.0 and nebin are user supplied namelist
c                   variables.
c          sgxnmi(ie,mb)
c                 - minimum inverse mean free path for ie-th energy
c                   group of mb-th beamline.  calculated for stationary
c                   target case only.  otherwise, calculated for each
c                   neutral trajectory in subroutine INJECT.
c          debin
c                 - width of energy bins (keV/amu)
c ----------------------------------------------------------------------
c
c 
      implicit  integer (i-n), real*8 (a-h, o-z)
c      include 'params.inc'   ! for stand alone code
      include 'param.h'
c
      integer mb, mfm1, ibion, nebin, nprim, nimp, nion
      real*8 ebfac,debin,atw(kion),ebkev(kb),vbeam(ke,kb)
      real*8 zne(kz),zni(kz,kion),zte(kz),zzi(kz,kion)
      real*8 sgxn(kcmp1,kz,kbe,ksge),sgxnmi(ke,kb)  ! stand alone code
      real*8 sgxne(kbe)
c      real*8 sgxn(kcmp1,kz,ke,kb),sgxnmi(ke,kb)
c      real*8 sgxne(ke)
c
      if (ibstart.eq.1) then
        do ib=1,kb   !YuP[07-2016] Bug: was 1,ke
        do ie=1,ke !YuP[07-2016] Bug: was 1,kb
          sgxnmi(ie,ib) = 0.D0
        enddo
        enddo
c
!      do i=1,kz
!       do j=1,ke
!         do k=1,kb
!          sgxn(1,i,j,k)=0.D0
!          sgxn(2,i,j,k)=0.D0
!          sgxn(3,i,j,k)=0.D0
!          sgxn(4,i,j,k)=0.D0 !YuP[2022-12-15]: BUG? 
!          !See above declaration: real*8 sgxn(kcmp1,kz,kbe,ksge)
!        enddo
!       enddo
!      enddo
        sgxn=0.d0 !YuP[2022-12-15] replaced the above.
      endif  !On ibstart.eq.1
c
c ----------------------------------------------------------------------
c stationary plasma case
c ----------------------------------------------------------------------
c
      if (nebin .eq. 0) then
        do 220 ib=ibstart,mb
        do 220 ie=1,ke
        sgxnmi(ie,ib) = 0.0
        ind  = ke*(ib-1) + ie
        eova = 1.0e3*ebkev(ib)/(dFLOAT(ie)*atw_beam) !YuP[2022-12-19] was FLOAT
        vbin = vbeam(ie,ib)
        do 220 i=1,mfm1
        teev = 1.0e3 * zte(i)
        sgxne(ind) = fsgxne(vbin,teev,zne(i))
        sgxn (1,i,ind,1) = 0.0
        do k=1,nion
          sgxn(1,i,ind,1) = sgxn(1,i,ind,1)
     .                    + fsgxni(atw(k),eova,zni(i,k),zzi(i,k))
          if ((k .le. nprim) .and. (k .le. 2.0))
     .      sgxn(k+1,i,ind,1) = fsgxncx(atw(k),eova,zni(i,k))
        end do
        sgxn(1,i,ind,1) = sgxn(1,i,ind,1) + sgxne(ind)
        sgxn(4,i,ind,1) = sgxn(1,i,ind,1) + sgxn(2,i,ind,1)
     .                  + sgxn(3,i,ind,1)
        sgxn(1,i,ind,1) = sgxn(1,i,ind,1)/sgxn(4,i,ind,1)
        sgxn(2,i,ind,1) = sgxn(2,i,ind,1)/sgxn(4,i,ind,1)
        sgxn(3,i,ind,1) = sgxn(3,i,ind,1)/sgxn(4,i,ind,1)
        sgxnmi(ie,ib) = MAX (sgxnmi(ie,ib),sgxn(4,i,ind,1))
  220   continue
        do ib=ibstart,mb
          do ie=1,ke
            sgxnmi(ie,ib) = 1.0 / sgxnmi(ie,ib)
          end do
        end do
        
      else !nebin>0
c
c ----------------------------------------------------------------------
c rotating plasma case
c ----------------------------------------------------------------------
c
        ebmax = 0.0
        do ib=ibstart,mb
           ebmax = MAX (ebmax,ebkev(ib))
        end do
        ebmax = ebmax/atw_beam
        debin = ebmax*ebfac/dFLOAT(nebin) !YuP[2022-12-19] was FLOAT
c
        do 340 i=1,mfm1
        teev = 1.0e3*zte(i)
        
c       electron impact ionization independent of rotation speed
        do 320 ib=ibstart,mb
        do 320 ie=1,ke
        ind=ke*(ib-1) + ie
        sgxne(ind) = fsgxne(vbeam(ie,ib),teev,zne(i))
  320   continue
        !Here ind= ke*(mb-1) + ke
  
        do 340 j=1,nebin
        ebin = dFLOAT(j)*debin !YuP[2022-12-19] was FLOAT
        eova = 1.0e3*ebin
        sgxn(1,i,1,j) = 0.0
        do 330 k=1,nion
        sgxn(1,i,1,j) = sgxn(1,i,1,j)
     .                + fsgxni(atw(k),eova,zni(i,k),zzi(i,k))
        if ((k .le. nprim) .and. (k .le. 2.0))
     .    sgxn(k+1,i,1,j) = fsgxncx(atw(k),eova,zni(i,k))
  330   continue

cBH171014: Use of ind below is strange.  It has only been set in 
cBH171014: above do 320 loop, with final value ind=ke*(mb-1)+ke.
cBH171014: It's initial value is ke*(ibstart-1)+1.
cBH171014: OK, I'll take this as final value in following loop.

cBH171014      do 340 k=ind,1,-1
        do 340 k=ind,ke*(ibstart-1)+1,-1  !ind=ke*(ib-1)+ie,beam energy index
          sgxn(1,i,k,j) = sgxn(1,i,1,j) + sgxne(k)
          sgxn(4,i,k,j) = sgxn(1,i,k,j) + sgxn(2,i,1,j) + sgxn(3,i,1,j)
          sgxn(1,i,k,j) = sgxn(1,i,k,j)/sgxn(4,i,k,j)
          sgxn(2,i,k,j) = sgxn(2,i,1,j)/sgxn(4,i,k,j)
          sgxn(3,i,k,j) = sgxn(3,i,1,j)/sgxn(4,i,k,j)
  340   continue !i=1,mfm1 ; j=1,nebin ; k=ind,ke*(ibstart-1)+1,-1
        !YuP: Same label for different loops - ok?
        
      endif !nebin
c
      return
      end subroutine nbsgold
c
c
      real*8  function AMAXAF (array, first, last)
c
c --- replacement for this LIBMATH function in UNICOS ------------------
c
      implicit none

      character what_id*45
      save      what_id
      data      what_id/"@(#)portlib.f      17 Nov 98  General Atomics"/
c
      integer   first, last, index
      real*8    array(last), maximum
c
      maximum = array(first)
c
      do index=first,last
        if (array(index) .gt. maximum)  maximum = array(index)
      end do
c
      AMAXAF = maximum
      return
c
      end
c
c
      integer function MAXAF (array, first, last)
c
c --- replacement for this LIBMATH function in UNICOS ------------------
c
      implicit none
c
      integer   first, last, index
      integer   array(last), maximum
c
      maximum = array(first)
c
      do index=first,last
        if (array(index) .gt. maximum)  maximum = array(index)
      end do
c
      MAXAF = maximum
      return
c
      end
c
c
      integer function LENGTH (string)
c
c --- return trimmed (no trailing spaces or tabs) length of a string ---
c
      implicit none
c
      character  last*1, string*(*)
c
      LENGTH = LEN (string)
      if (LENGTH .ne. 0) then
        last = string(LENGTH:LENGTH)
        do while (last .eq. ' ' .or. last .eq. '        ')
          LENGTH = LENGTH - 1
          if (LENGTH .ne. 0) then
            last = string(LENGTH:LENGTH)
          else
            last = '#'
          end if
        end do
      end if
      return
c
      end
c
c
      subroutine getioun (unitv, unitreq)
c  getioun sets io-unit variable unitv to the value unitret.
c  Depending on freeus_cql3d, unitret either = input variable unitreq or is output.
      integer unitv, unitreq, unitret
      unitret = unitreq
      call freeus_cql3d(unitret)
      unitv = unitret
      return
      end
c
      subroutine freeus_cql3d(unitno) 
      !YuP[2017] renamed to avoid conflicts with TRANSP
c
c     $Id: dummies.f,v 1.4 2000/05/31 21:10:41 freemanj Exp $
c
c dummy replacement for Basis's freeus, which returns a free unit-
c number in its argument.
      integer unitno
      return
      end
c
      subroutine giveupus_cql3d(unitno)
      !YuP[2017] renamed to avoid conflicts with TRANSP
c dummy replacement for Basis's giveupus, which gives up io unit-no. unitno.
      integer unitno
      return
      end
c
c
      subroutine STOP (message, status)
c
c --- print specified message and then exit with the specified status --
c
      implicit none
c
      integer   status
      character message*(*)
c
      if (status .eq. 0) then
        write (*, '(/ '' STOP in '', a /)')
     .  message
      else
        write (*, '(/ '' STOP in '', a / '' EXIT status is'', i4 /)')
     .  message, status
      end if
      call EXIT (status)
c
      end
c
c
      subroutine spline (n, x, y, b, c, d)
c
      implicit  integer (i-n), real*8 (a-h, o-z)
c
      integer n
      real*8  x(n), y(n), b(n), c(n), d(n)
c
c  the coefficients b(i), c(i), and d(i), i = 1,2,...,n are computed
c  for a cubic interpolating spline
c
c    s(x) = y(i) + b(i)*(x-x(i)) + c(i)*(x-x(i))**2 + d(i)*(x-x(i))**3
c
c    for  x(i) .le. x .le. x(i+1)
c
c  input..
c
c    n = the number of data points or knots (n .ge. 2)
c    x = the abscissas of the knots in strictly increasing order
c    y = the ordinates of the knots
c
c  output..
c
c    b, c, d  = arrays of spline coefficients as defined above.
c
c  using  p  to denote differentiation,
c
c    y(i) = s(x(i))
c    b(i) = sp(x(i))
c    c(i) = spp(x(i))/2
c    d(i) = sppp(x(i))/6  (derivative from the right)
c
c  the accompanying function subprogram  seval  can be used
c  to evaluate the spline.
c
      integer nm1, ib, i
      real*8  t
c
      nm1 = n - 1
      if (n .lt. 2)  return
      if (n .lt. 3)  go to 50
c
c  set up tridiagonal system
c
c  b = diagonal, d = offdiagonal, c = right hand side.
c
      d(1) = x(2) - x(1)
      c(2) = (y(2) - y(1))/d(1)
      do i=2, nm1
         d(i)   = x(i+1) - x(i)
         b(i)   = 2.D0 * (d(i-1) + d(i))
         c(i+1) = (y(i+1) - y(i))/d(i)
         c(i)   = c(i+1) - c(i)
      end do
c
c  end conditions.
c  third derivatives at x(1) and x(n) obtained from divided differences
c
      b(1) = -d(1)
      b(n) = -d(n-1)
      c(1) = 0.D0
      c(n) = 0.D0
      if (n .eq. 3)  go to 15
      c(1) = c(3)/(x(4)-x(2)) - c(2)/(x(3)-x(1))
      c(n) = c(n-1)/(x(n)-x(n-2)) - c(n-2)/(x(n-1)-x(n-3))
      c(1) = c(1)*d(1)**2/(x(4)-x(1))
      c(n) = -c(n)*d(n-1)**2/(x(n)-x(n-3))
c
c  forward elimination
c
   15 do i=2, n
        t    = d(i-1)/b(i-1)
        b(i) = b(i) - t*d(i-1)
        c(i) = c(i) - t*c(i-1)
      end do
c
c  back substitution
c
      c(n) = c(n)/b(n)
      do ib=1, nm1
         i = n-ib
         c(i) = (c(i) - d(i)*c(i+1))/b(i)
      end do
c
c  c(i) is now the sigma(i) of the text
c
c  compute polynomial coefficients
c
      b(n) = (y(n) - y(nm1))/d(nm1) + d(nm1)*(c(nm1) + 2.0*c(n))
      do i=1, nm1
         b(i) = (y(i+1) - y(i))/d(i) - d(i)*(c(i+1) + 2.0*c(i))
         d(i) = (c(i+1) - c(i))/d(i)
         c(i) = 3.D0*c(i)
      end do
      c(n) = 3.D0*c(n)
      d(n) = d(n-1)
      return
c
   50 b(1) = (y(2)-y(1))/(x(2)-x(1))
      c(1) = 0.D0
      d(1) = 0.D0
      b(2) = b(1)
      c(2) = 0.D0
      d(2) = 0.D0
      return
c
      end
c======================================================================

      real*8 function seval (n, u, x, y, b, c, d)
      !YuP: Only used by subr. adasqh6().
      !Should be called after spline_12(), 
      !to get the b, c, d coeffs.
c
      implicit  integer (i-n), real*8 (a-h, o-z)
c.......................................................................
c      include 'fileiou.inc'
c      BH: explicitly including:
c-----Include file, FILEIOU.INC, 03/01/92
c-----This file includes the unit numbers of the different I/O files
c-----for the NFREYA project, as well as the title:

      integer ntty, nin, nout, nmhd, ngrf
      common/iou/ntty, nin, nout, nmhd, ngrf
      character*80 title
      common/titl/title
c.......................................................................
c
      integer n
      real*8 u, x(n), y(n), b(n), c(n), d(n)
c
c this subroutine evaluates the cubic spline function
c
c    seval = y(i) + b(i)*(u-x(i)) + c(i)*(u-x(i))**2 + d(i)*(u-x(i))**3
c
c    where  x(i) < u < x(i+1), using Horner's rule
c
c  if  u .lt. x(1)  then  i = 1  is used.
c  if  u .ge. x(n)  then  i = n  is used.

c  ---------------------------HSJ Modification 1/30/98------------------
c  The above two rules are not acceptible in the context in which
c  this routine is used in the ONETWO code. (actually they are not
c  acceptable in any context in my opinion.
c  Interpolating outside the cubic spline fit is totally bogus.)
c  I have changed the rules to read  
c   if u .gt. x(n) then use Y(n) as the most reasonable
c   approximation if you dont actually want to stop the code.
c   similarly if u .lt. x(1) then use y(1) as the returned value.
c  using the first or last point may not be much better in some circumstances
c  but at least we will get physically relevant values !!!!!!!!!!!HSJ
c  Note that the switch extend_seval must be set to 1 to get this
c  behavior. Otherwise an error stop is taken when trying to interpoalte
c  outside the tables.
c -----------------------------------------------------------------------
c
c  input..
c
c    n     = the number of data points
c    u     = the abscissa at which the spline is to be evaluated
c    x,y   = the arrays of data abscissas and ordinates
c    b,c,d = arrays of spline coefficients computed by subr.spline_12()
c
c  if  u  is not in the same interval as the previous call, then a
c  binary search is performed to determine the proper interval.
c
      integer i, j, k
      real*8  dx
      integer extend_seval
      data    i/1/, extend_seval/1/ ! YuP[07-2016] changed extend_seval from 0 to 1
      !YuP: Had problem with extend_seval=0 in mirror machine runs:
      ! Code quits with a message:
      !Subroutine Seval,a spline fit evaluater,
      !has detected that out of bounds interpolation
      !is occuring. The value at which the cubic spline
      !is to be evaluated is     4.16666667E+03
      !the interpolating table extends from     5.00000000E+03 to     1.00000000E+06
      !You can set extend_seval=1 in the first namelist
      !in inone if you want to ignore this problem
      !subroutine SEVAL: Bad Interpolation
c   
c
c  added  1/29/98  HSJ
c
      if (u .lt. x(1)) then
         if(extend_seval .eq. 1)then
           seval=y(1) !It means: if u is to the left of lower boundary,
           return     ! use data from {x(1),y(1)} point in ADAS table.
         else
           write(*,100)u,x(1),x(n)
            write(nout,100)u,x(1),x(n)
c           call STOP('subroutine SEVAL: Bad Interpolation', 1226)
           write(nout,*)'subroutine SEVAL: Bad Interpolation'
           STOP
         endif
      else if(u .gt. x(n)) then
         if(extend_seval .eq. 1)then
           seval=y(n)
           return
         else
           write(*,100)u,x(1),x(n)
            write(nout,100)u,x(1),x(n)
c           call STOP('subroutine SEVAL: Bad Interpolation', 1226)
           write(nout,*)'subroutine SEVAL: Bad Interpolation'
           STOP
         endif
      endif
c
      if (i .ge. n)  i = 1
      if (u .lt. x(i))  go to 10
      if (u .le. x(i+1))  go to 30
c
c  binary search
c

  10  i = 1
      j = n+1
  20  k = (i+j)/2
      if (u .lt. x(k))  j = k
      if (u .ge. x(k))  i = k
      if (j .gt. i+1 )  go to 20
c
c  evaluate spline
c
   30 dx = u - x(i)
      seval = y(i) + dx*(b(i) + dx*(c(i) + dx*d(i)))
      return
c
 100  format('Subroutine Seval,a spline fit evaluater,',/,
     .       ' has detected that out of bounds interpolation',/,
     .       ' is occuring. The value at which the cubic spline',/,
     .       ' is to be evaluated is ',1pe18.8,/,
     .       ' the interpolating table extends from ',1pe18.8,
     .       ' to ',1pe18.8,/,
     .       ' You can set extend_seval=1 in the first namelist',/,
     .       ' in inone if you want to ignore this problem')
c
      end function seval
c
c======================================================================

      subroutine spline_12 (n, x, y, b, c, d)
c
      implicit  integer (i-n), real*8 (a-h, o-z)
c
      integer n
      real*8  x(n), y(n), b(n), c(n), d(n)
c
c  the coefficients b(i), c(i), and d(i), i = 1,2,...,n are computed
c  for a cubic interpolating spline
c
c    s(x) = y(i) + b(i)*(x-x(i)) + c(i)*(x-x(i))**2 + d(i)*(x-x(i))**3
c
c    for  x(i) .le. x .le. x(i+1)
c
c  input..
c
c    n = the number of data points or knots (n .ge. 2)
c    x = the abscissas of the knots in strictly increasing order
c    y = the ordinates of the knots
c
c  output..
c
c    b, c, d  = arrays of spline coefficients as defined above.
c
c  using  p  to denote differentiation,
c
c    y(i) = s(x(i))
c    b(i) = sp(x(i))
c    c(i) = spp(x(i))/2
c    d(i) = sppp(x(i))/6  (derivative from the right)
c
c  the accompanying function subprogram  seval()  can be used
c  to evaluate the spline.
c
      integer nm1, ib, i
      real*8  t
c
      nm1 = n - 1
      if (n .lt. 2)  return
      if (n .lt. 3)  go to 50
c
c  set up tridiagonal system
c
c  b = diagonal, d = offdiagonal, c = right hand side.
c
      d(1) = x(2) - x(1)
      c(2) = (y(2) - y(1))/d(1)
      do i=2, nm1
         d(i)   = x(i+1) - x(i)
         b(i)   = 2.0 * (d(i-1) + d(i))
         c(i+1) = (y(i+1) - y(i))/d(i)
         c(i)   = c(i+1) - c(i)
      end do
c
c  end conditions.
c  third derivatives at x(1) and x(n) obtained from divided differences
c
      b(1) = -d(1)
      b(n) = -d(n-1)
      c(1) = 0.0
      c(n) = 0.0
      if (n .eq. 3)  go to 15
      c(1) = c(3)/(x(4)-x(2)) - c(2)/(x(3)-x(1))
      c(n) = c(n-1)/(x(n)-x(n-2)) - c(n-2)/(x(n-1)-x(n-3))
      c(1) = c(1)*d(1)**2/(x(4)-x(1))
      c(n) = -c(n)*d(n-1)**2/(x(n)-x(n-3))
c
c  forward elimination
c
   15 do i=2, n
        t    = d(i-1)/b(i-1)
        b(i) = b(i) - t*d(i-1)
        c(i) = c(i) - t*c(i-1)
      end do
c
c  back substitution
c
      c(n) = c(n)/b(n)
      do ib=1, nm1
         i = n-ib
         c(i) = (c(i) - d(i)*c(i+1))/b(i)
      end do
c
c  c(i) is now the sigma(i) of the text
c
c  compute polynomial coefficients
c
      b(n) = (y(n) - y(nm1))/d(nm1) + d(nm1)*(c(nm1) + 2.0*c(n))
      do i=1, nm1
         b(i) = (y(i+1) - y(i))/d(i) - d(i)*(c(i+1) + 2.0*c(i))
         d(i) = (c(i+1) - c(i))/d(i)
         c(i) = 3.0*c(i)
      end do
      c(n) = 3.0*c(n)
      d(n) = d(n-1)
      return
c
   50 b(1) = (y(2)-y(1))/(x(2)-x(1))
      c(1) = 0.0
      d(1) = 0.0
      b(2) = b(1)
      c(2) = 0.0
      d(2) = 0.0
      return
c
      end subroutine spline_12
c
c


