c
c
      subroutine coefload(k)
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     This routine adds in the krook operator contribution to the
c     coefficients employed in time advancement..
c..................................................................

      include 'param.h'
      include 'comm.h'


c..................................................................
c     toroidal loss term
c..................................................................

      call losstor(k)

c..................................................................
c     orbit loss term - see subroutine losscone.
c..................................................................

      do 30 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
        do 31 j=1,jx
          if(gone(i,j,k,indxlr_).lt.zero) then ! lost
            gon(i,j)=vnorm*x(j)*gone(i,j,k,indxlr_)/tau(i,lr_)
          elseif(gone(i,j,k,indxlr_).eq.zero)then ! confined
            gon(i,j)=zero
          else
            gon(i,j)=(1.-exp(dtreff*gone(i,j,k,indxlr_)))/dtreff
          endif

c..................................................................
c     Add in the contributions from subroutine losstor (taulos)
c     cah when multiplied by vptb(i,lr_) and by F becomes the Krook opera
c..................................................................

          cah(i,j)=gon(i,j)-1./taulos(i,j,indxlr_)
 31     continue
 30   continue
      
!      if(k.eq.2 .and. sum(gon).ne.zero)then
!      write(*,*)'coefload: k,lr_,sum(gon)',k,lr_,sum(gon)
!      endif 
      
      return
      end
