c
c
      subroutine exsweepx(k)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     This routine checks the solution obtained after the theta split
c     by differentiation of the solution in time and comparison with
c     the right hand side. temp4(i,j) will contain the l.h.s. and will b
c     multiplied by cint2(j)*cynt2(i,l_) so that it will contain the
c     the change in the local particle number density. temp4(i,j) will
c     be the r.h.s.  and the two arrays should (ideally) be identical.
c     Each array is summed for purposes of comparison.
c..................................................................

      include 'param.h'
      include 'comm.h'

      include 'advnce.h'
      sumleft=0.
      sumright=0.
      do 90 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
        do 100 j=1,jx

c..................................................................
c     Differentiate the flux gfu
c..................................................................

          temp5(i,j)=gfu(i,j,k)
          temp3(i,j)=(gfu(i,j,k)-gfu(i,j-1,k))*cynt2(i,l_)

c..................................................................
c     Krook operator..
c..................................................................

     1      +cah(i,j)*temp2(i,j)*vptb(i,lr_)*cint2(j)*cynt2(i,l_)

c..................................................................
c     Particle source...
c..................................................................

     1      +.5*so(i,j)*vptb(i,lr_)*cint2(j)*cynt2(i,l_)
          temp3(i,j)=temp3(i,j)*dtr*one_

c..................................................................
c     Left hand side..
c..................................................................

          temp4(i,j)=(temp2(i,j)-temp1(i,j))
     1      *vptb(i,lr_)*cint2(j)*cynt2(i,l_)
     1      *one_
          sumleft=sumleft+temp4(i,j)
          sumright=sumright+temp3(i,j)
 100    continue
 90   continue
      error=(sumleft-sumright)/xlndn(k,lr_)
      if (iactst.eq."abort") then
        if (error.gt.1.e-8) call diagwrng(15)
      endif
      return
      end
