c
c
      subroutine micxinil
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c..................................................................
c     This routine computes various constants and constant arrays.
c     These include arrays employed for integration in velocity
c     space both at the midplane and at various points along particle
c     orbits. Many are used in the routines used to compute the
c     collisional coefficients, cfp*****
c..................................................................

      include 'param.h'
      include 'comm.h'
ccc      real*8,dimension(iy):: prnt1,prnt2,prnt3
ccc      real*8,dimension(iy):: prnt4,prnt5,prnt6

c..................................................................
c     Define the relativistic correction gamma and related functions
c..................................................................
      do 35 j=1,jx
        gamma(j)=sqrt(1.+xsq(j)*cnorm2i)
        gamsqr(j)=gamma(j)**2
        gamcub(j)=gamma(j)**3
        gammi(j)=1./gamma(j)
        gamm2i(j)=gammi(j)**2
        if(xsq(j)*cnorm2i.lt.1.e-8 .or. relativ .eq. "disabled") then
          gamm1(j)=.5*xsq(j)/cnorm2
        else
          gamm1(j)=gamma(j)-1.
        endif
        tcsgm1(j)=2.*cnorm2*gamm1(j)
        uoc(j)=x(j)/cnorm
        ! For plots only:
        !if (kelecg.eq.1) then
        !   enerkev(j)=gamm1(j)*restmkev
        !else
        !   enerkev(j)=gamm1(j)*fmass(niong)*clite2/ergtkev
        !endif
        !YuP[2018-01-08] added 2nd index (k).
        !Can be several general ion species.
        do k=1,ngen
           enerkev(j,k)=gamm1(j)*fmass(k)*clite2/ergtkev
        enddo
 35   continue
c
      if(relativ .eq. "fully") then
        write(*,*)'cnorm=',cnorm 
        one6 = 1.d0/6.d0
        do 36 j=1,jx
          alphan(j,-1)=x(j)/cnorm
          gamman(j,1)=gamma(j)
          gamman(j,-1)=gammi(j) !-YuP-> added: needed at line 374 cfpcoefr
          if(xsq(j)*cnorm2i.lt.1.e-8) then
            asnha(j)= alphan(j,-1)*(1.d0 - one6*alphan(j,-1)**2) !-YuP-> higher accuracy
          else
c990131            asnha(j)=alog(alphan(j,-1)+sqrt(alphan(j,-1)**2+1.))
            asnha(j)=log(alphan(j,-1)+sqrt(alphan(j,-1)**2+1.))
          endif
 36     continue
        do 37 i=-2,mx+2
          do 38 j=2,jx
            gamman(j,i)=gamman(j,1)**i
            alphan(j,i)=alphan(j,-1)**(-i)  !BH080327, Needs checking?
 38       continue
        if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
        write(*,*)'i,alphan(1:4,i)',i,(alphan(j,i),j=1,4)
        endif
 37     continue
      endif
c      write(*,*)'((gamman(j,i),j=1,4),i=-2,3)'
c$$$      write(*,99)((gamman(j,i),j=1,4),i=-2,3)
c      write(*,*)'(alphan(j,i),j=1,4),i=-2,3)'
c$$$      write(*,99)((alphan(j,i),j=1,4),i=-2,3)
c 99   format(4(1pd14.7))

c..................................................................
c     compute various coefficients used in subroutine cfpcoefc
c..................................................................
      do 100 m=0,mx
        fac=(4.*pi)/(2*m+1)
        cog(m,1)=fac/(2*m+3)
        cog(m,2)=fac/(2*m-1)
        cog(m,3)=(m+2)*cog(m,1)
        cog(m,4)=(m+1)*cog(m,1)
        cog(m,5)=m*cog(m,2)
        cog(m,6)=(m-1)*cog(m,2)
        cog(m,7)=(m+1)*cog(m,3)
        cog(m,8)=(m-1)*cog(m,5)
        cog(m,9)=m*cog(m,7)
        cog(m,10)=(m+3)*cog(m,7)
        cog(m,11)=(m-2)*cog(m,8)
        cog(m,12)=(m+1)*cog(m,8)
        cog(m,13)=fac*m
        cog(m,14)=fac*(m+1)
        cog(m,15)=fac
 100  continue
      do 110 m=0,mx
        ix1(m)=1-m
        ix2(m)=2+m
        ix3(m)=3-m
        ix4(m)=4+m
        ix5(m)=2+m
        ix6(m)=1-m
        ix7(m)=m
        ix8(m)=-1-m
 110  continue
      do 130 iq=-5-mx,4+mx
        do 120 j=2,jx
          xm(j,iq)=(x(j)*gammi(j))**iq
 120    continue
 130  continue
c..................................................................
c     compute various coefficients used in subroutine cfpcoefc
c..................................................................
      fctrl(0)=1.
cBH_YP090809   do 131 l=1,mx+2
      do 131 l=1,2*mx+2
        fctrl(l)=fctrl(l-1)*dfloat(l)
 131  continue
cBH_YP090809      do 132 l=0,mx+2
cBH_YP090809        do 133 m=0,l
cBH_YP090809          choose(l,m)=fctrl(l)/(fctrl(m)*fctrl(l-m))
cBH_YP090809 133    continue
cBH_YP090809 132  continue
      do l=0,2*mx+2
         do m=0,mx+2
            choose(l,m)=zero
         enddo
      enddo
      do 132 l=0,2*mx+2
           do 133 m=0,l
              if (m.le.mx+2) choose(l,m)=fctrl(l)/(fctrl(m)*fctrl(l-m))
 133       continue
 132  continue

c......................................................................
c     compute cosines and related quantities off the midplane
c     recall iyh index of the theta mesh point closest to pi/2 and
c     less than pi/2. There is no mesh point at pi/2.
c
c     CQL3D:
c     cosz(i,l,lr_) is the cosine of the pitch angle that a particle
c     will have at orbit point z(l,lr_) if its midplane pitch angle is
c     y(i,l_)
c     CQLP: (define orbit dependent arrays, only once l_=1)
c     cosz(i,l,lr_) is the cosine at orbit point z(l,lr_)
c
c     z(l,lr) covers half-pol turn, eqsym.ne."none"; else, full-turn.
c......................................................................

c     (assumes lrz=1 when cqlpmod=enabled)
      if (cqlpmod.eq."enabled" .and. l_.ne.lrors) go to 999

c.......................................................................
c     loop over lz
c.......................................................................

      do 210 l=1,lz
c     test if l in lsindx(i) => ilcqlp=1
        ilcqlp=0
        if (cqlpmod.eq."enabled" .and. ls.eq.lz) ilcqlp=1
        if (cqlpmod.eq."enabled" .and. ls.lt.lz) then
          do 211 ill=1,ls
            if (lsindx(ill) .eq. l) then
              ilcqlp=1
              go to 212
            endif
 211      continue
 212      continue
          if (ilcqlp.eq.1 .and. ill.ne.indxls(l)) call wpwrng(6)
        endif

        iiy=iy
        iiyh=iyh
        iitl=itl
        iitu=itu
        if (cqlpmod.eq."enabled") then
          if (ilcqlp .eq. 1) then
            iiyh=iyh_(indxls(l))
            iiy=iy_(indxls(l))
            iitl=itl_(indxls(l))
            iitu=itu_(indxls(l))
          else
            iiyh=iyh_(1)
            iiy=iy_(1)
            iitl=itl_(1)
            iitu=itu_(1)
          endif
        endif

c       Following, down to imax, also OK for eqsym.eq."none"
        if (ilcqlp .eq. 0) then
          do 140 i=1,iiyh
            cosz(i,l,lr_)= 1. - bbpsi(l,lr_)*(sinn(i,lmdpln_)**2)
            if(cosz(i,l,lr_) .le. 1.e-11)  cosz(i,l,lr_)=em90
            cosz(i,l,lr_)=sqrt(cosz(i,l,lr_))
 140      continue
        else  ! On ilcqlp
          do 141 i=1,iiyh
            cosz(i,l,lr_)=coss(i,indxls(l))
 141      continue
        endif
        do 150 i=1,iiyh
          iii=iiy+1-i
          cosz(iii,l,lr_)=-cosz(i,l,lr_)
 150    continue
c......................................................................
c     imax(l,lr_) is the highest i such that a particle with a midplane
c     pitch angle y(i,lmdpln_) attains and passes z(l,lr_) before turning.
c     Exception: imax(lz,lr_)=itl, i.e., particle only attains z(lz,lr_).
c                Reset below, after end of l-loop.
c......................................................................

        imax(l,lr_)=1
        if (ilcqlp .eq. 0) then
          do 160 i=2,iyh_(lmdpln_)
            if (cosz(i,l,lr_) .le. 1.e-10) go to 170
 160      continue
          imax(l,lr_)=iyh_(lmdpln_)
          go to 180
 170      imax(l,lr_)=i-1
 180      continue
        else
          imax(l,lr_)=iiyh
        endif

cBH091031        write(*,*)'micxinil:lr_,itl,l,imax(l,lr_)',lr_,itl,l,imax(l,lr_)

c..................................................................
c     sinz and tanz are defined analogously to cosz (above)
c..................................................................
        if (ilcqlp .eq. 0) then
          do 190 i=1,iiy
            if(i.le.imax(l,lr_)) then
              sinz(i,l,lr_)=sqrt(1.-cosz(i,l,lr_)**2)
            else
              sinz(i,l,lr_)=one
            endif
            tanz(i,l,lr_)=sinz(i,l,lr_)/cosz(i,l,lr_)
            yz(i,l,lr_)=acos(cosz(i,l,lr_))
 190      continue
        else
          do 191 i=1,iiy
            sinz(i,l,lr_)=sinn(i,indxls(l))
            tanz(i,l,lr_)=sinz(i,l,lr_)/cosz(i,l,lr_)
            yz(i,l,lr_)=acos(cosz(i,l,lr_))
 191      continue
        endif
c..................................................................
c     tot is a tangent ratio (related to batot(i,lr_) computed
c     in subroutine bavgmax)
c..................................................................
        su=1./sqrt(bbpsi(l,lr_))
        do 600 i=1,iiy
           iif=0
           if (eqsym.eq."none") then
              if ((l.eq.1).or.(l.eq.lz).or.(i.eq.1).or.(i.eq.iiy)) iif=1
           else
              if ((l.eq.1).or.(i.eq.1).or.(i.eq.iiy)) iif=1
           endif
           if (iif.eq.1) then
              tot(i,l,lr_)=su
           else
              tot(i,l,lr_)=tann(i,lmdpln_)/tanz(i,l,lr_)
           endif
 600    continue
        sinz(1,l,lr_)=0.
        tanz(1,l,lr_)=0.
        yz(1,l,lr_)=0.
        do 200 i=1,iiyh
          iii=iiy+1-i
          sinz(iii,l,lr_)=sinz(i,l,lr_)
          tanz(iii,l,lr_)=-tanz(i,l,lr_)
          yz(iii,l,lr_)=pi-yz(i,l,lr_)
          tot(iii,l,lr_)=tot(i,l,lr_)
 200    continue

ccc      do i=1,iy
ccc         prnt1(i)=sinz(i,l,lr_)
ccc         prnt2(i)=cosz(i,l,lr_)
ccc         prnt3(i)=tanz(i,l,lr_)
ccc         prnt4(i)=tot(i,l,lr_)
ccc      enddo

      lll=l  !break point for gdb

c     end of loop over lz
 210  continue

      if (cqlpmod .ne. "enabled") imax(lz_bmax(lr_),lr_)=iitl

c..................................................................
c     compute legendre polynomials and their derivatives along
c     orbits which pierce the midplane at y(i,lmdpln_).
c..................................................................

c     Need ss and pm for dpcosz, which is needed by dcofleg for
c     soft xray and fusion rate modules.
      if (softxry.ne.'disabled') then
         mmx= max(msxr, mmx)
      elseif (sigmamod.ne.'disabled') then
         mmx= max(mmsv, mmx)
      else
         mmx=mx
      endif
      mmxp1=mmx+1

      call bcast(ss(1,1,0,lr_),zero,iymax*lz*(mmxp1+1))
      call bcast(ssy(1,1,0,lr_),zero,iymax*lz*mxp1)
      call bcast(ssyi(1,1,0,lr_),zero,iymax*lz*mxp1)
      call bcast(ssyy(1,1,0,lr_),zero,iymax*lz*mxp1)
      call bcast(ssyyy(1,1,0,lr_),zero,iymax*lz*mxp1)   !Not used?
      call bcast(dpcosz(1,1,0,lr_),zero,iymax*lz*mmxp1)
      !YuP[2021-03-11] iy->iymax in the above few lines

      do 270 l=1,lz
c     test if l in lsindx(i) => ilcqlp=1
        ilcqlp=0
        if (cqlpmod.eq."enabled" .and. ls.eq.lz) ilcqlp=1
        if (cqlpmod.eq."enabled" .and. ls.lt.lz) then
          do 271 ill=1,ls
            if (lsindx(ill) .eq. l) then
              ilcqlp=1
              go to 272
            endif
 271      continue
 272      continue
        endif

        iiy=iy
        iiyh=iyh
        iitl=itl
        iitu=itu
        if (cqlpmod.eq."enabled") then
          if (ilcqlp .eq. 1) then
            iiyh=iyh_(indxls(l))
            iiy=iy_(indxls(l)) !CQLP: iiy can be different at different l
            iitl=itl_(indxls(l))
            iitu=itu_(indxls(l))
          else
            iiyh=iyh_(1)
            iiy=iy_(1)
            iitl=itl_(1)
            iitu=itu_(1)
          endif
        endif
        !write(*,*)'micxinil: l,iiy,imax(l,lr_)=',l,iiy,imax(l,lr_)
        do 260 ii=0,1
          do 250 iii=1,imax(l,lr_)
            i=ii*iii-(iiy+1-iii)*(ii-1) ! i=iiy+1-iii or i=iii
            tom1(0)=1.
            tom2(0)=0.
            tom3(0)=0.
            tom4(0)=0.
            tom4(1)=0.
            pm(0,l_)=1.
            tom1(1)=cosz(i,l,lr_)  !Evidently P_1(cos_theta)
            tom2(1)=1.
            tom3(1)=0.
            pm(1,l_)=0.
            if (mmx .eq. 0) go to 230
            do 220 m=2,mmxp1
              tom1(m)=((2*m-1)*cosz(i,l,lr_)*tom1(m-1)-(m-1)*tom1(m-2))
     +          /m  ! P_m(cos_theta), using recursion formula
              tom2(m)=(2*m-1)*tom1(m-1)+tom2(m-2)
              tom3(m)=(2*m-1)*tom2(m-1)+tom3(m-2)
              tom4(m)=(2*m-1)*tom3(m-1)+tom4(m-2)
              pm(m,l_)=-(m-1)*pm(m-2,l_)/m
 220        continue
 230        continue
c..................................................................
c     ss(i,l,m) is the m'th order Legendre Polynomial evaluated
c     at z(l,lr_) for a particle with pitch angle y(i,l_) at the midplane.
c     ssy, etc are derivatives of the above.
c..................................................................
            do 240 m=0,mmxp1
              ss(i,l,m,lr_)=tom1(m)
 240        continue
            do 2401 m=0,mx
              ssy(i,l,m,lr_)=-tom2(m)*sinz(i,l,lr_)
              ssyi(i,l,m,lr_)=-tom2(m)
              ssyy(i,l,m,lr_)=-tom2(m)*cosz(i,l,lr_)+sinz(i,l,lr_)**2
     +          *tom3(m)
              ssyyy(i,l,m,lr_)=sinz(i,l,lr_)*(tom2(m)+3.*cosz(i,l,lr_)
     +          *tom3(m)
     1          -sinz(i,l,lr_)**2*tom4(m))
 2401       continue
 250      continue
 260    continue
 270  continue

c.......................................................................
c     Compute weights used in cfpleg to compute the Legendre coefficients
c.......................................................................

      do 360 l=1,lz
c     test if l in lsindx(i) => ilcqlp=1
        ilcqlp=0
        if (cqlpmod.eq."enabled" .and. ls.eq.lz) ilcqlp=1
        if (cqlpmod.eq."enabled" .and. ls.lt.lz) then
          do 3601 ill=1,ls
            if (lsindx(ill) .eq. l) then
              ilcqlp=1
              go to 3602
            endif
 3601     continue
 3602     continue
          if (ilcqlp.eq.1 .and. ill.ne.indxls(l)) call wpwrng(6)
        endif

        iiy=iy
        iiyh=iyh
        iitl=itl
        iitu=itu
        if (cqlpmod.eq."enabled") then
          if (ilcqlp .eq. 1) then
            iiyh=iyh_(indxls(l))
            iiy=iy_(indxls(l))
            iitl=itl_(indxls(l))
            iitu=itu_(indxls(l))
          else
            iiyh=iyh_(1)
            iiy=iy_(1)
            iitl=itl_(1)
            iitu=itu_(1)
          endif
        endif

c..................................................................
c     dpcosz(i,l,m,lr_) is a theta integration coefficient for summing
c     Legendre integrals at z(l,lr_). It contains the contribution
c     between i and i+1. For i=imax, it integrates between imax and 
c     the turning-point (at theta_pol=pi/2).
c     Note that dcofleg will then defines the weights between i-1/2
c     and i+1/2. Thus dcofleg(imax) includes [imax-1/2,turning-point]
c..................................................................
c
        do 290 i=1,imax(l,lr_)
          dpcosz(i,l,0,lr_)=cosz(i+1,l,lr_)-cosz(i,l,lr_)
          if (mmx .eq. 0) go to 290
          do 280 m=1,mmx
            dpcosz(i,l,m,lr_)=ss(i+1,l,m+1,lr_)-ss(i,l,m+1,lr_)
     1        +ss(i,l,m-1,lr_)-ss(i+1,l,m-1,lr_)
            dpcosz(i,l,m,lr_)=dpcosz(i,l,m,lr_)/(2.*m+1)
 280      continue
 290    continue

c     imax:
        dpcosz(imax(l,lr_),l,0,lr_)=-cosz(imax(l,lr_),l,lr_)
        if (mmx .eq. 0) go to 291
        do 281 m=1,mmx
          dpcosz(imax(l,lr_),l,m,lr_)=
     1      ( pm(m+1,l_)-ss(imax(l,lr_),l,m+1,lr_)+
     +      ss(imax(l,lr_),l,m-1,lr_)-pm(m-1,l_) ) / (2.*m+1)
 281    continue
 291    continue

        do 350 m=0,mmx
          zt=-1.
          if (m/2*2 .eq. m) zt=1.
          do 340 i=1,imax(l,lr_)
c     !!! changed to ii=iiy-i+1 instead to iiy-i, thus dpcosz(ii), ii>iyh
c     has the contribution between ii-1 and ii. In this way, dpcosz(iyh+1)
c     is also correctly defined.
            ii=iiy-i+1
 340      dpcosz(ii,l,m,lr_)=zt*dpcosz(i,l,m,lr_)
 350    continue

c.......................................................................
c     Constructs weights needed for a Gauss-Lagrangian type integration
c     for computing the Legendre coefficients
c.......................................................................

        if (ilcqlp .eq. 1) then
          ilundsc=indxls(l)
          call tdinlegw(l,lrindx(ilundsc),indxlr(lrindx(ilundsc)),
     +      ilundsc)
        else if (cqlpmod .ne. "enabled") then
          call tdinlegw(l,lr_,indxlr_,l_)
        else
          ilundsc=indxls(l)
          call tdinlegw(l,lrindx(ilundsc),indxlr(lrindx(ilundsc)),1)
        endif

 360  continue

      call dscal(iymax*lz*mmxp1,-one,dpcosz(1,1,0,lr_),1) !YuP[2021-03] iy->iymax
cBH110330:  Doesn't look like waa and wbb are used for anything?
      call dscal(lz*mxp1,-one,waa(1,0,lr_),1)
      call dscal(lz*mxp1,-one,wbb(1,0,lr_),1)



c......................................................................
c     lmax(i,l_) is the highest l such a particle with midplane
c     pitch angle y(i,l_) attains and passes z(l,l_). 
c     Exception: lmax(itl,l_)=lz_bmax(lr_), although particle attains
c                but does not pass z(lz,l_).
c                taunew.ne."enabled", lmax(itl,l_) reset in 
c                micgnbnd, called from baviorbto: elsewise, 
c                lmax(itl,l_) is set here to lz_bmax(lr_).
c     NOTE for eqsym.eq."none": Don't symmetrize lmax about pi/2,
c                               instead use lmax(i>iyh) for lower
c                               half of flux surface.
c......................................................................

      if (eqsym.ne."none") then
         ilzhfs=lz
      else  !  i.e., non-up-down symmetric case
         ilzhfs=lz_bmax(lr_)
      endif

      do 400 i=1,iyh_(lmdpln_)
        if (cqlpmod .ne. "enabled") then

          do 370 l=1,ilzhfs
            if (abs(cosz(i,l,lr_)) .le. 1.e-10) go to 380
 370      continue
          lmax(i,lr_)=ilzhfs
          go to 390
c         trapped particle at z < z(l):
 380      lmax(i,lr_)=l-1
 390      continue
          if (eqsym.eq."none") then  ! Non-symm: Do lower half
             do l=lz,ilzhfs+1,-1
                if (abs(cosz(i,l,lr_)) .le. 1.e-10) go to 381
             enddo
             go to 391
 381         lmax(iy+1-i,lr_)=l+1    ! Note: l+1 at lower bpsi
 391         continue
          else
             lmax(iy+1-i,lr_)=lmax(i,lr_)
          endif

        else ! (cqlpmod.eq."enabled")
          do 392 l=2,lz
            if (i .gt. imax(l,lr_)) go to 393
 392     continue
 393     lmax(i,lr_)=l-1
         lmax(iy_(lmdpln_)+1-i,lr_)=lmax(i,lr_)
        endif
 400  continue

      if (taunew.eq."enabled") then
         lmax(itl,lr_)=lz_bmax(lr_)
         lmax(itu,lr_)=lz_bmax(lr_)
      endif


c.......................................................................
c     zboun(i,lr_) is the pt along an orbit at which a trapped particle
c     bounces. Passing particles are assigned z_bmax(lr_)+1.e-10 
c     (z_bmax(lr_)-1.e-10 if eqsym.eq."none" and i.gt.iyh).
c
c     taunew.eq."disabled:
c     Particles at the pass-trap boundary are temporarily
c     assigned zstar. See subroutine baviorbt.  zstar presently
c     only used in baviorbto, and micgmbnd which is called
c     from baviorbto.  zboun(itl,lr_) is only used in
c     baviorbto, then reset to zmax(lr_)+1.e-10.
c
c     For eqsym.eq."none": Don't symmetrize zboun about pi/2, rather,
c                          use zboun(i>iyh) for lower FS region.
c                          The bounce pt is in terms of the z-mesh, and
c                          has value > z_bmax(lr_) in the lower region.
c.......................................................................

      do 420 i=1,iyh_(lmdpln_)

c       default: transiting  and tp=bndry particles
        zboun(i,lr_)=z_bmax(lr_)+1.e-10
        if (eqsym.ne."none") then
           zboun(iy_(lmdpln_)+1-i,lr_)=zboun(i,lr_)
        else
           zboun(iy_(lmdpln_)+1-i,lr_)=z_bmax(lr_)-1.e-10
        endif

        if (i .le. itl_(lmdpln_)) go to 420
        iupdown=+1  ! Indicates upper half-FS
        zboun(i,lr_)=psiinv(1./sinn(i,lmdpln_)**2,iupdown)
        if (eqsym.ne."none") then
           zboun(iy_(lmdpln_)+1-i,lr_)=zboun(i,lr_)
        else
           iupdown=-1
           zboun(iy_(lmdpln_)+1-i,lr_)=
     +                           psiinv(1./sinn(i,lmdpln_)**2,iupdown)
        endif
 420  continue

      if (taunew.ne."enabled") then
      zstar=zboun(itl_(lmdpln_)+2,lr_)+.7*(zboun(itl_(lmdpln_)+1,lr_)
     +     -zboun(itl_(lmdpln_)+2,lr_))
      zboun(itl_(lmdpln_),lr_)=zstar
      zboun(itu_(lmdpln_),lr_)=zstar
      do 696 l=1,lz
        if (zboun(itl_(lmdpln_),lr_) .le. z(l,lr_)) go to 697
 696  continue
 697  continue
c%OS  if (cqlpmod .ne. "enabled") then
      lmax(itl_(lmdpln_),lr_)=l-1
      lmax(itu_(lmdpln_),lr_)=l-1
c%OS  endif
      endif

c......................................................................
c     thtab(l,lr_) is the pitch angle at the midplane that a particle
c     must have to bounce at z(l,lr_).
c     Allow for a "little" waver of bbpsi to values less than 1.
c......................................................................
      do 430 l=1,lz
         if (bbpsi(l,lr_) .lt. 1.) then
            thtab(l,lr_)=pi/2.
         else
            thtab(l,lr_)=acos(sqrt(1.-1./bbpsi(l,lr_)))
         endif
 430  continue
c      write(*,*) 'micxinil: lz,lr_,bbpsi ',
c     +           (bbpsi(l,lr_),l=1,lz)
c      write(*,*) 'micxinil: lz,lr_,thtab ',
c     +           (thtab(l,lr_),l=1,lz)

      do 440 k=1,ntotal
        do 441 i=1,ntotal
          satioz2(i,k)=(bnumb(i)/bnumb(k))**2
          satiom(i,k)=fmass(i)/fmass(k)
 441    continue
 440  continue
c......................................................................
c     zmaxpsi(lr_) is a normalization coefficient used to determine
c     the average density/cc from the orbit integrated density (/cm**2).
c     psidz(lr_) is another orbit integral required to determine some
c     types of averages.
c     r0drdz(lr_) is used with elecflag='parallel'.
c     Note: zmaxpsi, psidz, r0drdz  only computed on lrindx() mesh pnts.
c
c     Note: These quantities will be ~2x as big, for eqsym.eq."none".
c......................................................................
      zmaxpsi(lr_)=0.
      psidz(lr_)=0.
      r0drdz(lr_)=0.
      do 902 l=1,lz
        zmaxpsi(lr_)=zmaxpsi(lr_)+dz(l,lr_)/bbpsi(l,lr_)
        psidz(lr_)=psidz(lr_)+dz(l,lr_)*bbpsi(l,lr_)
        r0drdz(lr_)=r0drdz(lr_)+dz(l,lr_)*rmag/solrz(l,lr_)
 902  continue
      zmaxpsii(lr_)=1.0/zmaxpsi(lr_)
      do 910 j=1,jx-1
        ifp(j)=j+1
 910  continue
      ifp(jx)=jx
      do 920 i=1,iymax
        iota(i)=i
        ident(i)=1 ! integer
 920  continue

 999  continue

      return
      end
