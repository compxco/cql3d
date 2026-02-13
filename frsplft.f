c
c
      subroutine frsplft(lrz,oldx,oldf,npts,ynewx,ynewf)
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     Interpolates with splines between cql3d radial mesh and the
c     (finer) NFREYA radial mesh.
c..................................................................


      include 'param.h'
      parameter (nwka=3*lrza+1)
      dimension work(nwka),oldx(*),oldf(*),ynewx(*),ynewf(*),i1p(2),
     1  secondd(lrza),itab(3),tab(3)

      i1p(1)=4 !=4 means cubic spline; requires at least 4 points
      i1p(2)=4
        if(lrz.lt.4)then !YuP[2021-11-17] added check/stop for coeff1
          STOP ' frsplft: subr.coeff1(lrz,...) requires lrz>3'
          !Actually, it could work for lrz=3; 
          !Use i1p()=2 (first derivative is given),
          !Need values of 1st derivatives at rho=0 and rho=1.
        endif
      call coeff1(lrz,oldx,oldf,secondd,i1p,1,work)
      itab(1)=1
      itab(2)=0
      itab(3)=0
      do 10 l=1,npts
        call terp1(lrz,oldx,oldf,secondd,ynewx(l),1,tab,itab)
        ynewf(l)=tab(1)
 10   continue
      return
      end
