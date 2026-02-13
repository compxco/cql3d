c
c
      subroutine wpsavf
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     This routine saves various versions of the distribution
c     function.
c..............................................................

      include 'param.h'
      include 'comm.h'
c.......................................................................

      if (relaxtsp.ne."enabled" .or. n.lt.nonavgf .or.
     +  n.gt.nofavgf) go to 200
      !Note: Default values:  relaxtsp="disabled", nonavgf=5, nofavgf=10


c.......................................................................
cl    1. relaxtsp=enabled: mix f_n and f_n+1
c.......................................................................

      do 100 k=1,ngen
        do 110 j=0,jx+1
          do 120 l=0,ls+1
            do 130 i=0,iymax+1  !YuP[2021-03-08] was i=0,iy_(l)+1
              !YuP: iy_() is decl. as iy_(lrors), i.e. No value at l=0.
              !For l=0, we could use iy.
              !Simply change iy_(l)-->iymax at any l point.
              !Some of i points may not be needed (when meshy="fixed_mu")
              !but it does not matter here. Copying the whole range in i.
              fnp0(i,j,k,l)=0.5*(fnp0(i,j,k,l)+fnp1(i,j,k,l))
              fnp1(i,j,k,l)=fnp0(i,j,k,l)
 130        continue
 120      continue
 110    continue
 100  continue
      call dcopy(iyjx2*ngen*ls,fnp1(0,0,1,1),1,f(0,0,1,1),1)
      !call dcopy(iyjx2*ngen*ls,fnp1(0,0,1,1),1,f_(0,0,1,1),1)
      !Why fnp1-->f_ ?  Commented [it messes up with diagnostics, 
      !particularly with "9= Power computed from df/dt"]

      return

c.......................................................................
cl    2.  f=f_n+1
c.......................................................................

 200  continue

      call dcopy(iyjx2*ngen*(ls+2),fnp1(0,0,1,0),1,fnp0(0,0,1,0),1)
      call dcopy(iyjx2*ngen*ls,fnp1(0,0,1,1),1,f(0,0,1,1),1)
      !call dcopy(iyjx2*ngen*ls,fnp1(0,0,1,1),1,f_(0,0,1,1),1)
      !Why fnp1-->f_ ?  Commented [it messes up with diagnostics, 
      !particularly with "9= Power computed from df/dt"]

      return
      end
