c
c
      subroutine micxinim
      implicit integer (i-n), real*8 (a-h,o-z)

c......................................................................
c     Creates a single integration coefficient. (if l_.eq.lmdpln_):
c     2*pi*sin(theta0)*|cos(theta0)|*tau*d(theta0), tau with no v.
c......................................................................

      include 'param.h'
      include 'comm.h'

            !This subr. is only called when if l_=lmdpln_
            !For CQLP, l_=lmdpln_=1

      do 10 i=1,iy_(l_)
        vpint(i,lr_)=cynt2(i,l_)*abs(coss(i,l_))*tau(i,lr_)
 10   continue

      return
      end
