c
c
      subroutine wploweq
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     Determine equilibrium quatities in the lower half cross-section
c     assuming an up/down symmetric flux surface. The total mesh
c     points is lsmax, thus the mesh point at theta_pol=pi is lsmax/2+1
c.......................................................................

      include 'param.h'
      include 'comm.h'

c.......................................................................
cl    1. Copy values of top half flux-surface onto lower half
c     Assumes there is only one flux surface lr_ when cqlpmod=enabled
c.......................................................................

      lr_=lrindx(1)
      zmaxtot=2.*zmax(lr_)

c     lz meshes
      do 100 l=lz/2+2,lz
        z(l,lr_)=zmaxtot-z(lz-l+2,lr_)
        pol(l,lr_)=2.*pi-pol(lz-l+2,lr_)
c     symmetric quantities
        bbpsi(l,lr_)=bbpsi(lz-l+2,lr_)
 100  continue

c     ls meshes
      do 110 l=ls/2+2,ls
        sz(l)=zmaxtot-sz(ls-l+2)
c     symmetric quantities
        psis(l)=psis(ls-l+2)
        psipols(l)=psipols(ls-l+2)
        solrs(l)=solrs(ls-l+2)
        dsz(l)=dsz(ls-l+2)
        dszp5(l)=dszm5(ls-l+2)
        dszm5(l)=dszp5(ls-l+2)
        eszp5(l)=eszm5(ls-l+2)
        eszm5(l)=eszp5(ls-l+2)
c     anti-symmetric quantities
        psisp(l)=-psisp(ls-l+2)
        solzs(l)=-solzs(ls-l+2)
 110  continue
c     fix some quantities (periodic conditions assumed)
      z(lz+1,lr_)=zmaxtot !YuP[2021-04-09] lz+1 can get to lz+1
      pol(lz+1,lr_)=2.*pi !YuP[2021-04-09] lz+1 can get to lz+1
      bbpsi(lz+1,lr_)=bbpsi(1,lr_) !YuP[2021-04-09] lz+1 can get to lz+1
      sz(ls+1)=zmaxtot
      psis(0)=psis(ls)
      psis(ls+1)=psis(1)
      psisp(0)=psisp(ls)
      psisp(ls+1)=psisp(1)
      psipols(0)=psipols(ls)
      psipols(ls+1)=psipols(1)
      solrs(0)=solrs(ls)
      solrs(ls+1)=solrs(1)
      solzs(0)=solzs(ls)
      solzs(ls+1)=solzs(1)
      dszp5(0)=dszp5(ls)
      eszp5(0)=eszp5(ls)
      dszp5(ls/2+1)=sz(ls/2+2)-sz(ls/2+1)
      eszp5(ls/2+1)=1./dszp5(ls/2+1)
      dszm5(1)=dszp5(0)
      eszm5(1)=1./dszm5(1)
      dszm5(ls+1)=dszm5(1)
      dszm5(0)=dszm5(ls)
      eszm5(ls+1)=eszm5(1)
      eszm5(0)=eszm5(ls)
      dsz(0)=0.5*(dszm5(0)+dszp5(0))
      dsz(1)=0.5*(dszm5(1)+dszp5(1))
      dsz(ls)=dsz(0)
      dsz(ls+1)=dsz(1)
      dsz(ls/2+1)=0.5*(dszm5(ls/2+1)+dszp5(ls/2+1))

      do 111 k=1,ntotal
        do 112 l=ls/2+2,ls
          denpar(k,l)=denpar(k,ls-l+2)
          temppar(k,l)=temppar(k,ls-l+2)
 112    continue
 111  continue

      return
      end
