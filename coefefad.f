c
c
      subroutine coefefad(k)
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     This routine adds in the contribution of the D.C. electric
c     field to the coefficients used for time advancement.
c..................................................................

      include 'param.h'
      include 'comm.h'

c.......................................................................
cl    Include electrostatic field
c.......................................................................

      do 20 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
        if(cqlpmod.eq."enabled") then !YuP[2021-03-18] Added
         ztrda=-elparnw(l_)*charge/vnorm*coss(i,l_) !elparnw(): only for CQLP
         ztrdd= elparnw(l_)*charge/vnorm*sinn(i,l_)**2
        else
         ztrda=0.d0
         ztrdd=0.d0
        endif
        do 21 j=1,jx
          da(i,j)=da(i,j)+bnumb(k)/fmass(k)*(elecfld(lr_)*cex(i,j,l_)+
     +      ztrda*xsq(j))
          dd(i,j)=dd(i,j)+bnumb(k)/fmass(k)*(elecfld(lr_)*cet(i,j,l_)+
     +      ztrdd*x(j))
 21     continue
 20   continue

      return
      end
