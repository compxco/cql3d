c
c
      subroutine wpcthta
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     This routine calculates the coefficient cthta(i,j) encounting for
c     the mu*grad_parallel B force, when cqlpmod=enabled
c..................................................................

      include 'param.h'
      include 'comm.h'
c.......................................................................

      do 100 j=1,jx
        if (mod(nummods,10).le.4 .or. lmidvel.eq.0) then
          ztra=-0.5*x(j)*vnorm*psisp(l_)/psis(l_)
        else
          ztra=-0.5*x(j)*vnorm*(psisp(l_)+psisp(l_+1))
     /      /(psis(l_)+psis(l_+1))
        endif
        do 110 i=1,iy_(l_)  !YuP[2021-03-12] iy-->iy_(l_)
          cthta(i,j)=ztra*sinn(i,l_)*dyi(i,l_)
 110    continue
 100  continue

      return
      end
