c
c
c
c
      subroutine coefefld
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     compute coefficients required to represent parallel
c     electric field (flux surface averaged) due to ohmic current
c..................................................................

      include 'param.h'
      include 'comm.h'

      call bcast(cex(1,1,l_),zero,iymax*jx) !YuP[2021-03-11] iy-->iymax
      call bcast(cet(1,1,l_),zero,iymax*jx)
c     Division by 300. below is conversion to cgs: 300 volts/statvolt.
C%OS  coefld=-radmaj*fpsi(lr_)*onovrp(2,lr_)*flxavgd(lr_)
      if (cqlpmod .ne. "enabled") then
         if (efflag .eq. "toroidal") then
            coefld=-rmag*fpsi(lr_)*onovrp(2,lr_)*flxavgd(lr_)
     +           *charge/300./vnorm
cBH000926 Adding option for specification electric field parallel to B.
         elseif (efflag .eq. "parallel") then
            coefld=-r0drdz(lr_)*charge/300./vnorm  
         endif
      endif
C%OS  if (cqlpmod .eq. "enabled") coefld=-radmaj*fpsi(lr_)/solrs(l_)**2
      if (cqlpmod.eq."enabled") then
        if (mod(nummods,10).le.4 .or. transp.ne."enabled" .or. 
     +    lmidvel.eq.0) then
          coefld=-rmag*fpsi(lr_)/solrs(l_)**2
     +      /psis(l_)/bmidplne(lr_)*charge/300./vnorm
        else
          coefld=-rmag*fpsi(lr_)/0.125/(solrs(l_)+solrs(l_+1))**2
     +      /(psis(l_)+psis(l_+1))/bmidplne(lr_)*charge/300./vnorm
        endif
      endif

c%OS  
c%OS  coefld=-rmag*fpsi(lr_)/solrs(1)**2
c%OS  +                         /psis(1)/bmidplne(lr_)*charge/300./vnorm
c%OS
  
      iend=itl-1
      if (cqlpmod.eq."enabled" .and. symtrap.ne."enabled") iend=iyh

      do 40 i=1,iend
        ii=iy_(l_)+1-i !YuP[2021-03-11] iy-->iy_(l_)
        do 50 j=1,jx
          cex(i,j,l_)=coefld*coss(i,l_)*xsq(j)
          cex(ii,j,l_)=-cex(i,j,l_)
          cet(i,j,l_)=-coefld*sinn(i,l_)**2*x(j)
          cet(ii,j,l_)=cet(i,j,l_)
 50     continue
 40   continue
      return
      end
