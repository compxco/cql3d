c
c
      subroutine exsweept(k)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     This routine checks the solution obtained after the velocity split
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

c..................................................................
c     Pass/trapped boundary condition (itl and itu)
c..................................................................

        if (i .eq. itl .or. i .eq. itu) go to 90
        do 100 j=1,jx

c..................................................................
c     differentiate the flux hfu
c..................................................................

          temp6(i,j)=hfu(i,j)
          temp3(i,j)=(hfu(i,j)-hfu(i-1,j))*dx(j)*twopi

c..................................................................
c     Particle source
c..................................................................

     1      +.5*so(i,j)*vptb(i,lr_)*cint2(j)*cynt2(i,l_)
          temp3(i,j)=temp3(i,j)*one_*dtr

c..................................................................
c     differentiate l.h.s.
c..................................................................

          temp4(i,j)=(temp2(i,j)-temp1(i,j))
     1      *vptb(i,lr_)*cint2(j)*cynt2(i,l_)
     1      *one_
          sumleft=sumleft+temp4(i,j)
          sumright=sumright+temp3(i,j)
 100    continue
 90   continue

c..................................................................
c     Pass/trapped contribution...
c..................................................................

cdir$ ivdep
      do 200 j=1,jx
        temp6(itl,j)=hfu(itl,j)
        temp6(itu,j)=hfu(itu,j)
        temp3(itl,j)=-(hfu(itl-1,j)-2.*hfu(itl,j)-hfu(itu,j))*pi*dx(j)
     1    +.5*cah(itl,j)*temp2(itl,j)*vptb(itl,lr_)
     1    *cynt2(itl,l_)*cint2(j)
     1    +.5*so(itl,j)*vptb(itl,lr_)*cint2(j)*cynt2(itl,l_)
        temp3(itl,j)=temp3(itl,j)*one_*dtr
        temp3(itu,j)=temp3(itl,j)
        temp4(itl,j)=(temp2(itl,j)-temp1(itl,j))*vptb(itl,lr_)*cint2(j)
     1    *cynt2(itl,l_)*one_
        temp4(itu,j)=temp4(itl,j)
        sumleft=sumleft+2.*temp4(itl,j)
        sumright=sumright+2.*temp3(itl,j)
 200  continue
      error=(sumleft-sumright)/xlndn(k,lr_)
      if (iactst.eq."abort") then
        if (error.gt.1.e-8) call diagwrng(15)
      endif
      return
      end
