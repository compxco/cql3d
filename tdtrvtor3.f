c
c
      subroutine tdtrvtor3(f1,f2,vp,vp_,kopt,k)
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     This routine takes the function f1 defined on the full
c     velocity mesh and interpolates it onto the velocity mesh
c     on which the transport is done. The new function is f2
c     and it has the same integral as f1 (particle conserving),
c     with vp and vp_ being the weights for f1 and f2 respectively.
c     That is: sum(f1(i)*vp(i)) = sum(f2(i)*vp_(i)), i=1,iy_(l)
c
c     This new version of tdtrvtor is such that two successive calls
c     to tdtrvtor2(f1,f2,.,.,iopt) and then tdtrrtov2(f2,f1b,.,.,iopt)
c     gives f1b = f1 at each point i=1,iy_(l)
c
c     kopt = 1: transformation of distribution function
c     2:       "        of quasilinear term (velsou) (ADI)
c     3:       "        of transport term (spasou)   (ADI)
c
c     Assumes that vp is vp(lrindx(l)) if kopt = 1 and vp(l) otherwise,
c     where l=1,lrors
c..............................................................

      include 'param.h'
      include 'comm.h'

      dimension f1(0:iyp1,0:jxp1,ngen,*)
      dimension f2(0:iyp1,0:jxp1,ngen,*)
      dimension vp(iy,lrza), vp_(iy,lrza)
      dimension vpeff(iy,1), vpeff_(iy,1)
c.......................................................................
c     copy weight according to option

      if (kopt .eq. 1) then
        do 6 i=1,iy_(l_)
          vpeff(i,1)=vp(i,lr_)
          vpeff_(i,1)=vp_(i,lr_)
 6      continue
      else
        do 7 i=1,iy_(l_)
          vpeff(i,1)=vp(i,l_)
          vpeff_(i,1)=vp_(i,l_)
 7      continue
      endif

c.......................................................................
c     kopt=3: loop over lrz instead of lrors
c.......................................................................
      leff=lrors
      if (kopt .eq. 3) leff=lrz

      call dcopy (iyjx2*ngen*leff,f1,1,f2,1)

      ill=l_
      if (kopt .eq. 3) ill=indxlr_
      do 30 j=1,jx
        f2(itl-2,j,k,ill)=(f1(itl-2,j,k,ill)*vpeff(itl-2,1)+
     1    f1(itl-1,j,k,ill)*vpeff(itl-1,1)+
     1    f1(itl,j,k,ill)*vpeff(itl,1)*.5)/vpeff_(itl-2,1)
        f2(itl+2,j,k,ill)=(f1(itl+2,j,k,ill)*vpeff(itl+2,1)+
     1    f1(itl+1,j,k,ill)*vpeff(itl+1,1)+
     1    f1(itl,j,k,ill)*vpeff(itl,1)*.5)/vpeff_(itl+2,1)
        f2(itu-2,j,k,ill)=f2(itl+2,j,k,ill)
        f2(itu+2,j,k,ill)=(f1(itu+2,j,k,ill)*vpeff(itu+2,1)+
     1    f1(itu+1,j,k,ill)*vpeff(itu+1,1)+
     1    f1(itu,j,k,ill)*vpeff(itu,1)*.5)/vpeff_(itu+2,1)

        if (kopt .eq. 1) then
          f_vtor(j,k,(kopt-1)*6+1,ill)=f1(itl-2,j,k,ill) /
     /      f2(itl-2,j,k,ill)
          f_vtor(j,k,(kopt-1)*6+2,ill)=f1(itl+2,j,k,ill) /
     /      f2(itl+2,j,k,ill)
          f_vtor(j,k,(kopt-1)*6+3,ill)=f1(itu+2,j,k,ill) / 
     /      f2(itu+2,j,k,ill)
          f_vtor(j,k,(kopt-1)*6+4,ill)=f1(itl,j,k,ill) /
     /      f1(itl-2,j,k,ill)
          f_vtor(j,k,(kopt-1)*6+5,ill)=f1(itl,j,k,ill) /
     /      f1(itl+2,j,k,ill)
          f_vtor(j,k,(kopt-1)*6+6,ill)=f1(itl,j,k,ill) /
     /      f1(itu+2,j,k,ill)
        endif
 30   continue

      do 40 i=itl-1,itl+1
        do 41 j=1,jx
          i_=iy_(l_)+1-i  !YuP[2021-03-11] iy-->iy_(l_)
          f2(i,j,k,l_)=0.
          f2(i_,j,k,l_)=0.
 41     continue
 40   continue

      return
      end
