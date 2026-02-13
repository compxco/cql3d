

c
c
      subroutine coefegad(k)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     Adds in coefficient of energy loss term to fp coefficients.
c..................................................................

      include 'param.h'
      include 'comm.h'

      if(k.gt.ngen)  return
      do 10  j=1,jx
        do 11  i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
          da(i,j)=da(i,j)+egylosa(i,j,k,indxlr_)
 11     continue
 10   continue
      return
      end
