c
c
      subroutine micxinit
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     Called repeatedly from tdinitl with ll=lrors,1,-1; tdnflxs(ll)
c
c     This routine determines certain constants and the fundamental
c     meshes. These meshes are x (momentum/rest mass or speed), 
c     and y(pitch angle,l_).
c     These meshes are tailored to suit
c     requirements specified by the user in the namelist input.
c
c     It also defines related mesh quantities such as sinn, coss, dy,..
c..................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE
c.......................................................................


      iotalim=jx
      if(iy.gt.jx) iotalim=iy
      if(iotalim.lt.200) iotalim=200
      if(mx.gt.iotalim) iotalim=mx
      iotalim=iotalim+2

c..................................................................
c     IOTA and REALIOTA are only dimensioned (0:750)xxx   (0:6400)
c..................................................................

cBH111124      if(iotalim.ge.750) then
      if(iotalim.ge.6400) then
        WRITE(*,*)'micxinit: Check iotalim' 
        call exit(1)
      endif
      do 1 i=0,iotalim
        iota(i)=i
 1    continue
      do 2 i=0,iotalim
        realiota(i)=dfloat(i)
 2    continue

c..................................................................
c     Define the pitch angle mesh, y(i,l_).
c
c     itl and itu will be the indices of the lower (upper) pass/trap
c     boundary. The theta (y) mesh will have the following
c     characteristics: y(1,l_)=0. y(itl,l_)=thb(l_) (p/t bndry).
c     y(iyh,l_) last mesh pt before pi/2. No mesh pt at pi/2.
c     IF tfac.ge.0.:
c     y(itl-1,l_) and y(itl+1,l_) positioned by input tbnd(1) to be
c     very close to y(itl,l_) to create a boundary sliver region.
c     tfac is a geometric mesh factors like tfacz.
c     IF tfac.lt.0., then use |tfac| for mesh factor, and
c       DO NOT create boundary sliver region [for transp="enabled"
c                                    using soln_method.ne."direct"].
c     The y mesh can be specified in a number of ways:
c     If meshy="free" the various theta meshes on the flux surfaces
c     are independent. A number of options are available on how
c     to construct the mesh (see tfac, yreset, etc below).
c     If meshy="fixed_mu" or "fixed_y", then in the former case
c     y(i,l_) for all i have the same adiabatic invariant mu (except
c     for the complication caused by the p/t interface); in the latter
c     case y(i,l_) indep of l_  (except for the complication arising
c     from the p/t interface. See array idx(i,l_)).
c..................................................................


      if (eqsym.eq."none") then  !Only if eqmod.eq."enabled",
                                 !        eqsource="eqdsk"
         ilzhfs=lz_bmax(lr_)
      else
         ilzhfs=lz
      endif
      if (cqlpmod.eq."enabled" .and. numclas .eq. 1) ilzhfs=lz/2+1
      !YuP[2025-12-12] added cqlpmod.eq."enabled" in the above line     
      tbnd(l_)=tbnd(1)

      if (meshy.ne."free") then

        call tdtry !meshy "fixed_y" or "fixed_mu" (cqlpmod=enabled or disabled)
        !OUT: iyh_(l_)  iy_(l_)  itl_(l_)  itu_(l_)  iyjx_(l_)   y(i,l_)

      else if (cqlpmod .ne. "enabled") then  !meshy.eq."free"

c..................................................................
c     meshy=free and cqlpmod="disabled".
c     Pack in theta mesh points if yreset="enabled".
c**    [Not working at present, and if "enabled" will be reset to
c**     "disabled" in ainsetva.f, for lrz>1.] 
c..................................................................

        iyh1=iyh
        if (yreset .eq. "enabled") iyh1=iyh-numby
        iy1=2*iyh1
        xpi=half*pi
        thb(l_)=asin(sqrt(1./bbpsi(ilzhfs,lr_)))
        pctg=2.*thb(l_)/pi
        itl=(.85*pctg+.15)*iyh1
        inew=iy/2+itl-1
        if (inew.eq.(inew/2)*2) itl=itl-1
        if (itl .ge. iyh1-3) itl=iyh1-3
        if (itl .lt. 4) itl=4
        itx=itl-1
        itu=iy1+1-itl
        y(1,l_)=0.
        htfg=abs(tfac)*(thb(l_)/(itx-1))   ! ?? or abs(tfac) ??
        y(itx,l_)=thb(l_)
        y(itx-1,l_)=thb(l_)-htfg
        call micgetr(itx,thb(l_),htfg,ram,ksingul)
        if (ksingul .eq. 1) call diagwrng(8)
        do 60 i=itx-2,2,-1
          y(i,l_)=y(i+1,l_)+ram*(y(i+1,l_)-y(i+2,l_))
 60     continue
        dypi=(xpi-thb(l_))*half/(iyh1-itl)
        y(itl+1,l_)=thb(l_)
        htfg=abs(tfac)*((xpi-dypi-thb(l_))/(iyh1-itl-1))   ! or abs(tfac) ??
        y(itl+2,l_)=htfg+thb(l_)
        call micgetr(iyh1-itl,xpi-dypi-thb(l_),htfg,ram,ksingul)
        if (ksingul .eq. 1) call diagwrng(8)
        do 70 i=itl+3,iyh1
          y(i,l_)=y(i-1,l_)+ram*(y(i-1,l_)-y(i-2,l_))
 70     continue
        sa=y(itl+2,l_)-thb(l_)
        sb=thb(l_)-y(itx-1,l_)
        zt=sa
        if (sb .lt. sa) zt=sb
        if (zt .lt. tbnd(l_)) tbnd(l_)=zt*.5
        y(itl,l_)=thb(l_)
        y(itl-1,l_)=thb(l_)-tbnd(l_)
        y(itl+1,l_)=thb(l_)+tbnd(l_)

CDIR$ IVDEP
        do 100 i=1,iyh1
          y(iy1+1-i,l_)=pi-y(i,l_)
 100    continue
        if (yreset .eq. "enabled") then
          do 600 i=1,iyh1
            if (y(i,l_) .gt. ylower) go to 601
 600      continue
 601      if(i .gt. iyh1) call diagwrng(13)
          ilower=i
          do 602 i=ilower,iyh
            if (y(i,l_) .ge. yupper) go to 603
 602      continue
 603      continue
          if (y(i,l_) .eq. yupper) yupper=yupper-1.e-6
          iupper=i
          if (ilower .lt.itl-1 .and. iupper .lt. itl-1) go to 620
          if (ilower .gt. itl+1 .and. iupper .gt. itl+1) go to 620
          if (yupper .lt. y(iyh1,l_)) go to 620
          call diagwrng(13)
 620      continue
          ntot=numby+(iupper-ilower)
          dyy=(yupper-ylower)/(ntot-1)
          if(iyh1.le.iupper) then
            y(iyh,l_)=y(iyh1,l_)
          else
            do 608 i=0,iyh1-iupper
              y(iyh-i,l_)=y(iyh1-i,l_)
 608        continue
          endif
          if(ilower.ne.(numby+iupper)) then
            y(ilower,l_)=ylower
            do 605 i=ilower+1,ilower+ntot-1
              y(i,l_)=y(i-1,l_)+dyy
 605        continue
          endif
        endif  !On yreset.eq."enabled"

CDIR$ IVDEP
        do 610 i=1,iyh
          y(iy+1-i,l_)=pi-y(i,l_)
 610    continue
        do 621 i=1,iyh
          if (abs (y(i,l_)-thb(l_)) .lt. 1.e-10) itl=iota(i)
 621    continue
        itu=iy+1-itl

      else  !on meshy.ne."free",etc ! (cqlpmod.eq."enabled")
c.......................................................................
c     y mesh for CQLP case, for meshy="free"
c.......................................................................

        zdy=pi/(iy-1)
        iyh=iy/2
        do 650 i=1,iy
          y(i,l_)=dfloat(i-1)*zdy !equidistant grid that includes 0 & pi
 650    continue
        thb(l_)=asin(sqrt(psis(l_)/bbpsi(ilzhfs,lr_)))
        do 651 i=1,iyh
          if (y(i,l_) .gt. thb(l_)) go to 652
 651    continue
 652    itl=i-1
        itu=iy+1-itl
c     tbnd set to offset from thb
        tbnd(l_)=thb(l_)-y(itl,l_) 
        !default: tbnd(1)=.002, tbnd(2:lrorsa)=0.

      endif  !On meshy.ne."free",etc
      !write(*,*)'micxinit: l_,itl,itu=',l_,itl,itu

      if (meshy.eq."free") then !YuP[2021-03-04] Added if().
       ! For meshy="fixed_mu" or "fixed_y",
       ! the following values are set inside subr. tdtry,
       ! and they can be quite different from these values:
       ! (example: iy_() can be smaller than iy for "fixed_mu")
       iy_(l_)=iy
       iyh_(l_)=iyh
       iyjx_(l_)=iy*jx
       itl_(l_)=itl
       itu_(l_)=itu
       y(1,l_)=0.d0
       y(iy,l_)=pi
      endif
    
      if(symtrap.eq."enabled") then
         !Number of indep theta pts, for symmetric trapped region
         inew_(l_)=iy_(l_)/2+itl_(l_)-1   
      else
         inew_(l_)=iy_(l_) !general case (not symmetric trapped reg.)
      endif
      inewjx_(l_)=inew_(l_)*jx         !Number eqns for l_

      if(cqlpmod.ne."enabled")then !YuP[2021-03-04] Added if(). 
      !No problem for CQLP, so - do not print this warning.
      if (itl_(l_).eq.iyh_(l_)) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'***************************************************'
         WRITE(*,*)'WARNING/micxinit: itl.eq.iyh.   There are problems'
         WRITE(*,*)' in the code, e.g. in impavnc0, for this condition.'
         WRITE(*,*)' It can be avoided by larger rya(1) or iy.'
         WRITE(*,*)'***************************************************'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK   
      endif
      endif


c..................................................................
c     Create the momentum/rest mass or speed mesh x.
c..................................................................


      if (xfac .ge. 0.) then

        x(1)=0.
        x(jx)=xmax
        hx=xmax/(jx-1)
        hxfg=xfac*hx
        x(2)=hxfg   !Spacing between first mesh points is
                    !xfac*hx, where hx is uniform spacing increment.
        call micgetr(jx,xmax,hxfg,ram,ksingul)
        if (ksingul .eq. 1) call diagwrng(8)
        do 120 j=3,jx-1
          x(j)=x(j-1)+ram*(x(j-1)-x(j-2))
 120    continue
      else
        jlwr=xpctlwr*jx
        if(jlwr .lt. 3) jlwr=3
        hx=xlwr/(jlwr-1)
        x(1)=0.
        do 200 j=2,jlwr
          x(j)=hx*(j-1)
 200    continue
        jmdl1=xpctmdl*jx
        jmdl=jmdl1+jlwr
        if (tandem.eq."enabled") then
          jmdl1=jx-jlwr
          jmdl=jx
        endif
        hm=(xmdl-xlwr)/jmdl1
        do 201 j=jlwr+1,jmdl
          x(j)=x(j-1)+hm
 201    continue
        if (tandem.eq."enabled") then
          x(jx)=xmax
          goto 203
        endif
        hu=(xmax-xmdl)/(jx-jmdl)
        do 202 j=jmdl+1,jx
          x(j)=x(j-1)+hu
 202    continue
 203    continue
      endif
      
      
![2022-03-19] Set up mesh for parallel momentum/mass.
!     This is only needed for plots of F_par (reduced distr func)
!     as a function of x_par; see subr.pltprppr.
!     This setup of xl() and xlm() arrays is done in fle_*** subroutines,
!     it is reproduced below.
!     Why we need it here - because not always subr.pltprppr is called at n=0,
!     but the 'xl'=xlm array is saved at n=0 only. 
      jflh=(jfl+1)/2 !jfl is set in namelist, then adjusted to odd value
      if(xlfac.ge.0.)then !xlfac can be 1(uniform), <1, >1, etc
        xl(1)=-xmax !xmax=1. by default
        xl(jflh)=0.
        xl(jfl)=xmax
        hx=xmax/(jfl-jflh)
        hxfg=xlfac*hx
        xl(jflh+1)=hxfg
        xl(jflh-1)=-hxfg
        call micgetr(jflh,xmax,hxfg,ram,ksingul)
        if (ksingul .eq. 1) call diagwrng(8)
        do j=jflh+2,jfl-1
          xl(j)=xl(j-1)+ram*(xl(j-1)-xl(j-2))
          jj=jfl+1-j
          xl(jj)=-xl(j)
        enddo
      else !xlfac
        jflwr=xlpctlwr*jflh !xlpctlwr=.1 by default
        if(jflwr .lt. 3) jflwr=3
        hx=xlwr/(jflwr-1) !xlwr=1./43. by default
        xl(jflh)=0.
        do j=jflh+1,jflh-1+jflwr
          xl(j)=hx*(j-jflh)
        enddo
        jmdl1=xlpctmdl*jflh !xlpctmdl=.4 by default
        jmdl=jmdl1+jflwr
        hm=(xlmdl-xllwr)/jmdl1 ! xlmdl=.25,xllwr=1./43. by default
        do j=jflh+jflwr,jflh-1+jmdl
          xl(j)=xl(j-1)+hm
        enddo
        hu=(xmax-xlmdl)/(jflh-jmdl)
        do j=jflh+jmdl,jfl
          xl(j)=xl(j-1)+hu
        enddo
        do j=jflh+1,jfl
           jj=jfl+1-j
           xl(jj)=-xl(j)
        enddo
      endif ! xlfac
      do j=1,jfl-1
         xlm(j)=0.5*(xl(j)+xl(j+1)) !This is used for plots Fpar(xlm)
         dxl(j)=xl(j+1)-xl(j)
      enddo
      xlm(0)=xl(1)-0.5*dxl(1)
      xlm(jfl)=xl(jfl)+0.5*dxl(jfl-1)
      dxl(0)=dxl(1)
      dxl(jfl)=dxl(jfl-1)
![2022-03-19] Done Set up mesh for parallel momentum/mass.
      
            
      !YuP [02-25-2016]
      !Check that there are sufficient number of v-grid points  
      !over thermal part of the distribution (at least 3 points).
      ! If not, print warning.
      do k=1,ngen
       do j=jx,1,-1 ! scan backward
          if(cqlpmod .ne. "enabled")then
           vth_l= vth(k,lr_)
           ll=lr_
          else !(cqlpmod.eq."enabled") ! YuP[2021-02-26]
           vth_l= vthpar(k,ls_)
           ll=ls_
          endif
          if(vth_l .le. x(j)*vnorm)then
            j_thermal=j !it is such j that v(j_thermal) is just below vth()
          endif
       enddo
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,'(a)')"=================================================="
      WRITE(*,'(a,3i4,2f13.8)') 
     + "micxinit: k,ll, j_thermal, x(j_thermal), vth/vnorm =",
     +  k, ll, j_thermal, x(j_thermal), vth_l/vnorm
      if(j_thermal.lt.3)then
      WRITE(*,'(a)')"  WARNING(micxinit): V-grid is too coarse."
      WRITE(*,'(a)')"  Thermal part of distribution is covered by only"
      WRITE(*,'(a,i5,a)')"    j=", j_thermal," points."
      WRITE(*,'(a)')"  The solution may become unstable."
      WRITE(*,'(a)')"  Consider increasing jx or setting xfac<1."
      endif
CMPIINSERT_ENDIF_RANK   
      enddo ! k=1,ngen

c.......................................................................
cl    3. The fundamental arrays are determined above. Derivative
c     and ancillary mesh arrays are determined below.
c     Compute arrays used in differencing.
c.......................................................................

c.......................................................................
cl    3.1 Define the mesh point centered integration coefficients
c     employed for orbit field line integrals, z mesh defined in micxiniz
c.......................................................................

      do 310 l=2,lz-1
        dz(l,lr_)=(z(l+1,lr_)-z(l-1,lr_))*.5
 310  continue
      dz(1,lr_)=(z(2,lr_)-z(1,lr_))*.5
      dz(lz,lr_)=(z(lz,lr_)-z(lz-1,lr_))*.5
      if(cqlpmod.eq."enabled")then ! For CQLP only
      if (sbdry.eq."periodic")then !YuP/was: .and. transp.eq."enabled") then
        !YuP: For sbdry="periodic", we could always set this
        !(i.e., transp or no transp, should not matter):
        dz(1,lr_)=2.*dz(1,lr_)
        dz(lz,lr_)=dz(lz,lr_)+0.5*(z(2,lr_)-z(1,lr_))
      endif
      endif

c.......................................................................
cl    3.2 Define speed integration meshes dx** and their inverses ex**
c.......................................................................
c$$$      do 320 j=1,jxm1
c$$$         dxp5(j)=x(j+1)-x(j) 
c$$$c     BH080425:   Have made storage of dxp5,dxm5 and exp5,exm5 separate.
c$$$         dxm5(j+1)=dxp5(j)
c$$$         xmidpt(j)=x(j)+dxp5(j)*.5
c$$$ 320  continue
c$$$c      xmidpt(1)=x(1)
c$$$      xmidpt(jx)=x(jx)
c$$$      dxp5(jx)=0.
c$$$      dxm5(1)=0.
c$$$      dxm5(jx+1)=dxp5(jx)
c$$$      dxp5(0)=dxm5(1)
c$$$      do 321 j=1,jxm1
c$$$        exp5(j)=1./dxp5(j)
c$$$        exm5(j+1)=exp5(j)
c$$$ 321  continue
c$$$      exp5(jx)=0.
c$$$      exm5(1)=0.
c$$$      exm5(jx+1)=exp5(jx)
c$$$      exp5(0)=exm5(1)
cBH081018:  Using new coding from cql3d_bipointered_ngena2_multiUR, 
cBH081018:  since it is a bit cleaner.
      do 320 j=1,jxm1
        dxp5(j)=x(j+1)-x(j)
        xmidpt(j)=x(j)+dxp5(j)*.5
 320  continue
c      xmidpt(1)=x(1)
      xmidpt(jx)=x(jx)
      dxp5(jx)=0.
c      dxm5(1)=0.
cBH080910:  Have removed equivalence of dxp5/dxm5
      dxp5(0)=0.
      do j=1,jx+1
         dxm5(j)=dxp5(j-1)
      enddo

cBH080910:  Have removed equivalence of exp5/exm5
      do 321 j=1,jxm1
        exp5(j)=1./dxp5(j)
 321  continue
      exp5(jx)=0.
c      exm5(1)=0.
      exp5(0)=0.
      do j=1,jx+1
         exm5(j)=exp5(j-1)
      enddo

      do 322 j=1,jx
        dx(j)=.5*(dxp5(j)+dxm5(j))
        dxi(j)=one/dx(j)
        xsq(j)=x(j)**2
        xcu(j)=x(j)**3
c     The following looks a little fishy to me, but check it
c     out later. Why skip j=1 case?  (BobH: 970720)
        if (j .gt. 1) then
          xi(j)=one/x(j)
          x2i(j)=xi(j)**2
          x3i(j)=xi(j)*x2i(j)
        endif
        if (j.lt.jx) then ! YuP-101227: modified to include j=1
          xcenter(j)=(x(j)+x(j+1))*.5
          xcensq(j)=xcenter(j)**2
          xcent3(j)=xcenter(j)**3 ! YuP-101227: added
        endif
 322  continue
c%OS  dx(jx) should be equal to 0.5*dxm5(jx) ? => next line commented
c%OS  dx(jx)=dxm5(jx)

c..................................................................
c     determine additional arrays used for integration over velocity.
c..................................................................
      do 323 j=1,jx
        cint2(j)=dx(j)*xsq(j)
 323  continue
      cint2(1)=x(2)**3/24.

c.......................................................................
cl    3.3 Define theta integration coefficients dy** and their inverses.
c.......................................................................
      iyh=iyh_(l_) !YuP[2021-03-04] iyh-->iyh_(l_)
      !For meshy="fixed_mu", iy_(l_) can be smaller than iy at some l_.
      !But don't use iy=iy_(l_) here! It will overwrite iy,
      !which is the namelist value, and iy is used to set dimensions
      !of many arrays.
      do 330 i=1,iyh
        dyp5(i,l_)=y(i+1,l_)-y(i,l_)
        ii=i+1
        dym5(ii,l_)=y(ii,l_)-y(ii-1,l_)
 330  continue
      dym5(iyh,l_)=y(iyh,l_)-y(iyh-1,l_)
      dym5(1,l_)=0.
      dyp5(0,l_)=0.
      dyp5(iy_(l_),l_)=0.
      do 331 i=1,iyh
        eyp5(i,l_)=1./dyp5(i,l_)
        ii=i+1
        eym5(ii,l_)=1./dym5(ii,l_)
 331  continue
      do 332 i=iyh+1,iy_(l_)-1  !YuP[2021-03-04] iy-->iy_(l_)
        dyp5(i,l_)=dyp5(iy_(l_)-i,l_)
        eyp5(i,l_)=eyp5(iy_(l_)-i,l_)
        dym5(i,l_)=dym5(iy_(l_)-i+2,l_)
        eym5(i,l_)=eym5(iy_(l_)-i+2,l_)
 332  continue
      dym5(iy_(l_),l_)=y(iy_(l_),l_)-y(iy_(l_)-1,l_)
      eym5(iy_(l_),l_)=1./dym5(iy_(l_),l_)
      eyp5(0,l_)=0.
      eyp5(iy_(l_),l_)=0.
      eym5(1,l_)=0.
      do 333 i=1,iy_(l_)
        dy(i,l_)=.5*(dym5(i,l_)+dyp5(i,l_)) !end points are 1/2 of other dy
        dyi(i,l_)=one/dy(i,l_)
c..................................................................
c     determine cosine and sine arrays.
c..................................................................
        sinn(i,l_)=sin(y(i,l_))
        coss(i,l_)=cos(y(i,l_))
        tann(i,l_)=sinn(i,l_)/coss(i,l_)
 333  continue
      coss(1,l_) = 1.d0
      coss(iy_(l_),l_)=-1.d0  ! YuP: Why needed?

      do 334 i=1,iy_(l_)-1
        ymid(i,l_)=y(i,l_)+dyp5(i,l_)*.5
 334  continue
cYuP      ymid(1,l_)=y(1,l_)    ! YuP 120627 commented out
c ymid is used in luf (or luf_bin) searches only.
c The value of ymid(1,#) should be  0 + dyp5(1,#)*.5,
c similar to internal points.
c With ymid(1,l_)=y(1,l_)=0  as was defined above,
c the luf function can never return the value of i=1;
c it can only be 2 or larger.
      ymid(iy_(l_),l_)=y(iy_(l_),l_) !could be y(iy,#)+dym5(i,#)/2 (not important)
      
c..................................................................
c     determine additional arrays used for integration over theta.
c..................................................................
      do 337 i=1,iy_(l_)-1
        cynt2(i,l_)=dy(i,l_)*twopi*sinn(i,l_)
 337  continue
      cynt2(1,l_)=0.25*pi*y(2,l_)**2
      cynt2(iy_(l_),l_)=cynt2(1,l_)
      twoint(l_)=0.
      do 338 i=1,iy_(l_)
        twoint(l_)=twoint(l_)+cynt2(i,l_)
 338  continue

c..................................................................
c     truncd(j) array: 
c     To taper RF diffusion over last 10 points of velocity, 
c     if ineg="trunc_d"
c..................................................................
      if (ineg.eq."trunc_d") then
         if (jx.le.11) stop 'micxinit/trunc_d: jx.le.11'
         do j=jx-11,jx
            truncd(j)=taper(x(j),zero,2.*x(jx-11),2.*(x(jx-1)-x(jx-11)))
         enddo
      endif



      return
      end
