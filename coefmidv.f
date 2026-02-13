c
c
      subroutine coefmidv(c,nn)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
      include 'param.h'
      include 'comm.h'

c..................................................................
c     This routine redefines the velocity mesh points at the
c     half mesh points (j+1/2) so that the fluxes are defined
c     conveniently for differencing. (nn=1 means da; nn=2 means db
c     nn=3 means dc)
c..................................................................

      dimension c(iymax,0:jx) !YuP[2021-03-11] iy-->iymax

c..................................................................
c     special averaging for da and db; accounts for the
c     fact that near v=0, da is proportional to x**3 and
c     db is proportional to x**2. also see loops 210, 220 at
c     the end of this routine.
c..................................................................

      if (nn .eq. 1) then
c     note: x(jx)=1.
        do 110 j=2,jx-1
          call dscal(iymax,x3i(j),c(1,j),1) !Note: x3i(j)=one/x(j)**3
 110    continue
        call dcopy(iymax,c(1,2),1,c(1,1),1)
      elseif (nn .eq. 2) then
        do 120 j=2,jx-1
          call dscal(iymax,x2i(j),c(1,j),1) !Note: x2i(j)=one/x(j)**2
 120    continue
        call dcopy(iymax,c(1,2),1,c(1,1),1)
      endif
      do 2 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_) here and below
        do 21 j=1,jx-1
          temp1(i,j)=(c(i,j)+c(i,j+1))*.5
 21     continue
 2    continue

c..................................................................
c     redefine da at jx+1/2..
c..................................................................

      if (nn .eq. 1) then
        do 1 i=1,iy_(l_)
          if(c(i,jx) .lt. 0.) then
            temp1(i,jx)=c(i,jx)
          else
            temp1(i,jx)=zero
          endif
 1      continue

c..................................................................
c     redefine db and dc to zero near j=jx
c..................................................................

      else
        do 3 i=1,iy_(l_)
          temp1(i,jx)=zero !YuP[] was 0.
 3      continue
      endif

c..................................................................
c     Redefine da and db at the pass-trapped boundaries
c..................................................................

      if (nn.ne.3 .and. cqlpmod.ne."enabled") then
        do 5 j=1,jx
          temp1(itl,j)=.25*( temp1(itl-1,j)/vptb(itl-1,lr_)+
     +                    2.*temp1(itl+1,j)/vptb(itl+1,lr_)+
     +                       temp1(itu+1,j)/vptb(itu+1,lr_) 
     *                                          )*vptb(itl,lr_)
          temp1(itu,j)=temp1(itl,j)
 5      continue

c..................................................................
c     Set dc=0 at pi and 0
c     Effectively implements bc df/d(theta) = 0 at pi and 0
c..................................................................

      else if (nn .eq. 3) then
        do 20 j=1,jx
          temp1(1,j)=zero !YuP[] was 0.
          temp1(iy_(l_),j)=zero !YuP[] was 0.
 20     continue
      endif

c..................................................................
c     set value of coefficients near zero to force zero flux at v=0
c..................................................................

      do 6 i=1,iy_(l_)
        c(i,0)=0.
        if (nn.le.2 .or. lbdry0.ne."enabled" .or. advectr.gt.10.) then
          c(i,1)=temp1(i,1)
        else
          c(i,1)=zero !YuP[] was 0.
        endif
        do 7 j=2,jx
          c(i,j)=temp1(i,j)
 7      continue
 6    continue
      if (nn .ne. 2) goto 9
!YuP[2019-07-08] Why do we need to impose a lower limit on abs(b(i,j))?
! In case of no-RF run, all db,dc coeffs are zero initially, 
! but this resetting below, c(i,j)=em40, makes db(i,j) a non-zero value,
! and it results in non-zero values of RF power in diagentr(*,lefct=3)
!      do 80 j=1,jx
!        do 81 i=1,iy_(l_)
!          if(abs(c(i,j)).le. em40) then
!            c(i,j)=em40 ! for nn=2. i.e. for b(i,j) coeff.
!          endif
! 81     continue
! 80   continue
 9    continue

c..................................................................
c     special averaging for da,db
c..................................................................

      if (nn .eq. 1) then
        do 210 j=1,jx-1
          call dscal(iymax,xcent3(j),c(1,j),1) ! xcent3(j)=xcenter(j)**3
 210    continue
      elseif (nn .eq. 2) then
        do 220 j=1,jx-1
          call dscal(iymax,xcensq(j),c(1,j),1) ! xcensq(j)=xcenter(j)**2
 220    continue
      endif
      
        call bcast(temp1(0,0),zero,iyjx2)
      
      return
      end
