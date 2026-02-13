c
c
      subroutine wpvptb
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     Reset some bounce average quantities for CQLP run
c..................................................................

      include 'param.h'
      include 'comm.h'

c.......................................................................
c     1. Set lambda=vptb=cos(theta_0)*tau to 1.0
c.......................................................................

!      do 100 i=1,iymax !YuP[2021-03-11] iy-->iymax
!        vptb(i,lr_)=1.d0
! 100  continue
        vptb(:,:)=1.d0 !YuP: all indexes, to be sure

      return
      end
