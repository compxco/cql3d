c
c
      subroutine restcon
      implicit integer (i-n), real*8 (a-h,o-z)
      save
      include 'param.h'
      include 'comm.h'

c..................................................................
c     Connor formula for toroidal resitivity,
c     J.W. Connor, R.C. Grimm, R.J. Hastie, P.M. Keeping,
c     Nucl. Fus. 13, 211 (1973).
c     The following calculation of the transiting particle fraction,
c     xconn (=I in Connor et al.),
c     is not entirely clear to me, but for a multi-flux-surface
c     test case, 1.-xconn agrees to almost
c     3 significant figures with trapfrac(lr_) from eqfndpsi.f.
c     (BobH, 010803).
c..................................................................

      s=0.
      ssg=0.
      u=0.
      uu=0.
      ilzhfs=lz
      if (cqlpmod.eq."enabled" .and. numclas .eq. 1) ilzhfs=lz/2+1
      !YuP[2025-12-12] added cqlpmod.eq."enabled" in the above line
      do 10 l=1,ilzhfs
        uu=uu+dz(l,lr_)*bbpsi(l,lr_)**2
 10   u=u+dz(l,lr_)*bbpsi(l,lr_)
      qz=.75*u/pi
      qqg=.75*uu/pi
      do 11 i=1,itl-2
!      if(l_.eq.15)then
!      write(*,*)'i, tau,psiiv=',
!     & i,tau(i,lr_), psiiv(i,lr_)
!      endif
        qx=coss(i,l_)*sinn(i,l_)**2*cynt2(i,l_)
        s=s+qx/(tau(i,lr_)*(psiiv(i,lr_)-sinn(i,l_)**2))
        ssg=ssg+qx/(tau(i,lr_)*(1.-psiba(i,lr_)*sinn(i,l_)**2))
 11   continue
      qx=cynt2(itl-1,l_)+pi*sinn(itl,l_)*(y(itl,l_)-y(itl-1,l_))
      qx=coss(itl-1,l_)*sinn(itl-1,l_)**2*qx
      s=s+qx/(tau(itl-1,lr_)*(psiiv(itl-1,lr_)-sinn(itl-1,l_)**2))
      ssg=ssg+qx/(tau(itl-1,lr_)*(1.-psiba(itl-1,lr_)*
     *  sinn(itl-1,l_)**2))
      xconn=s*qz
      rovsc(lr_)=(1.+.471*(1.-xconn))/xconn/(1.+.039*(1.-xconn))
      fc=ssg*qqg
      cr=.56/bnumb(kionn)*(3.-bnumb(kionn))/(3.+bnumb(kionn))
      eovsz=1./((1.-cr*(1.-fc))*fc)
      return
      end
