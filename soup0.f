c
c
      subroutine soup0(l)
      implicit integer (i-n), real*8 (a-h,o-z)
      integer l !INPUT
         !soup0(l) defines sovt(j,kk,m,lr_) array,
         !based on xem1,2(*,lr) input for src localization in energy
         !Presently xem1, xem2 do not depend on l index along field line,
         !but maybe in future - they will, so added l index to soup0(l)
      save
      include 'param.h'
      include 'comm.h'

cAYP201124 change the calculation of tam1 since xem1 and xem2 can be zero
c I am not sure what physics meaning is for tam1 and sovt
c However it should work for small xem2 and sovt->0
      if (soucoord.eq."cart") return
      do 10 kk=1,ngen
        do 11 m=1,nso
          do 12 j=1,jx
cAYP201124 Replacing	    tam1(j)=-(x(j)-xem1(kk,m,lr_))**2/xem2(kk,m,lr_)
            st1_=x(j)-xem1(kk,m,lr_)
            st2_=-(st1_*st1_)
            st1_=xem2(kk,m,lr_)
            if(dabs(st1_).lt.1.d-177) then
             tam1(j)=st2_/1.d-177
cBH201217             sovt(j,kk,m,lr)=dexp(tam1(j))
             sovt(j,kk,m,lr_)=dexp(tam1(j))
            else
             tam1(j)=st2_/st1_
             sovt(j,kk,m,lr_)=dexp(tam1(j))
            endif
 12       continue
 11     continue
 10   continue
      return
      end
