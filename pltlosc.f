c
c
      subroutine pltlosc
      implicit integer (i-n), real*8 (a-h,o-z)
c
c     Plot contours of the loss region..
c
c     Modified from Graflib to pgplot calls by Yuri Petrov, 090727,
c     using PGPLOT + GRAFLIBtoPGPLOT.f routines (put in pltmain.f).
c
      include 'param.h'
      include 'comm.h'

      if (noplots.eq."enabled1") return

      do 100 k=1,ngen
        suu=0.
        do 92001 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
          do 92002 j=1,jx
            if(gone(i,j,k,indxlr_).lt.-.9) then
              temp1(i,j)=vnorm*x(j)*f(i,j,k,l_)/tau(i,lr_)
            elseif (gone(i,j,k,indxlr_).gt.0.) then
              temp1(i,j)=0.
            elseif (gone(i,j,k,indxlr_) .le. 0.) then
              temp1(i,j)=vnorm*x(j)*f(i,j,k,l_)
     1          *(-gone(i,j,k,indxlr_))/tau(i,lr_)
            else
              temp1(i,j)=gone(i,j,k,indxlr_)*f(i,j,k,l_)
            endif
            if (temp1(i,j).ne.zero) suu=temp1(i,j)
92002     continue
92001   continue
        if (suu.eq.0.) go to 92003

        write(t_,588) k
 588    format("Loss due to lossmode(k) and torloss(k), k=",i5)
        CALL PGPAGE
        call pltcont(k,1,t_,8) ! itype=8 for pltlosc
92003   continue
 100  continue
      return
      end
