c
c
      subroutine wptrmuy
!Called from tdinitl when  cqlpmod.eq."enabled"
!  .and. meshy.eq."fixed_mu" .and. tfac.lt.0.0
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'

      dimension zmunod(iymax+lsa),zmidth0(lsa+2),zyfix(lsa+1)
      !YuP[2021-03-11] Changed iy-->iymax in declarations
      !(just in case if iy is changed by iy=iy_(l_) somewhere)

c..............................................................
c     Special mu mesh such that at each position l, y(iyh_(l),l) is
c     close to pi/2.
c.......................................................................

      ztfac=abs(tfac)
      iyhtr=iymax/2
      ilshalf=ls
      !YuP[2021]: Probably need to enforce iy/2 > ls-1, otherwise 
      !  iypass=iyhtr-ilshalf+1 == iy/2 -ls+1 may become 0 or negative
      if (sbdry .eq. "periodic") ilshalf=ls/2+1
      write(*,*)'wptrmuy0 sbdry,ilshalf,ls ',sbdry,ilshalf,ls
      itrap=ilshalf-1
      if (ztfac .le. 1.0) then
        do 110 l=2,ilshalf
          zmunod(ilshalf+1-l)=psis(1)/psis(l)*cos(epsthet)**2
 110    continue
      else if (ztfac .le. 2.0) then
        do 111 l=2,itrap
          zmunod(ilshalf+1-l)=psis(1)/(0.5*(psis(l)+psis(l+1)))
 111    continue
        zmunod(1)=psis(1)/psis(ilshalf)*cos(epsthet)**2
      else
c     angle at midplane ends at pi/2 at surface s, will be mid-angles at theta0
        do 112 l=2,ilshalf
          zmidth0(ilshalf+1-l)=asin(sqrt(psis(1)/psis(l)))
 112    continue
        zmidth0(ilshalf)=pi/2.
        lmindy=1
        do 113 l=2,ilshalf-2
          if (zmidth0(l+1)-zmidth0(l) .le. 
     !      zmidth0(lmindy+1)-zmidth0(lmindy)) lmindy=l
 113    continue
c     smallest interval at lmindy, put y point in middle
        zyfix(lmindy+1)=0.5*(zmidth0(lmindy)+zmidth0(lmindy+1))
        do 114 l=lmindy+1,ilshalf-2
          zyfix(l+1)=2.*zmidth0(l)-zyfix(l)
 114    continue
        do 115 l=lmindy,1,-1
          zyfix(l)=2.*zmidth0(l)-zyfix(l+1)
 115    continue
c     check mesh
        write(6,'(/," y mesh in trap region at s=1:",/,(1p10e13.4))') 
     !    (zyfix(l),l=1,ilshalf-1)
        write(6,'(/," (y(i)+y(i+1))/2 at s=1:",/,(1p10e13.4))') 
     !    (0.5*(zyfix(l)+zyfix(l+1)),l=1,ilshalf-1)
        write(6,'(/," y-mid mesh in trap region :",/,(1p10e13.4))') 
     !    (zmidth0(l),l=1,ilshalf)
        do 116 l=1,ilshalf-2
          if (zyfix(l+1) .le. zyfix(l)) stop 'bad yfix'
 116    continue
c     insert in sin**2 array
        do 117 l=1,itrap
          zmunod(l)=sin(zyfix(l))**2
 117    continue
      endif

      zrat=iyhtr*(1.-zmunod(1))/dfloat(itrap)
      if (zrat .le. 1.5) then
        iypass=iyhtr-ilshalf+1
        zdmupas=zmunod(1)/dfloat(iypass)
       write(*,*)'wptrmuy1: iyhtr,ilshalf,itrap',iyhtr,ilshalf,itrap 
       !40,80,79; or     75,81,80 then iypass= 75-81+1=-5
        do 120 i=1,iypass
          mun(i)=dfloat(i-1)*zdmupas
 120    continue
        do 121 l=1,itrap
          mun(iypass+l)=zmunod(l)
 121    continue
      else
        iypass=int(iyhtr*zmunod(1))
        zdmupas=zmunod(1)/dfloat(iypass)
       write(*,*)'wptrmuy2: iypass',iypass
        do 122 i=1,iypass+1
          mun(i)=dfloat(i-1)*zdmupas
 122    continue
        iextra=iyhtr-iypass-itrap
       write(*,*)'wptrmuy3: iextra',iextra
        do 123 ii=1,iextra
          zdmumax=0.0
          do 124 ll=1,itrap-1
            if (zmunod(ll+1)-zmunod(ll) .gt. zdmumax) then
              ilmax=ll
              zdmumax=zmunod(ll+1)-zmunod(ll)
            endif
 124      continue
          do 125 ll=itrap+1,ilmax+2,-1
            zmunod(ll)=zmunod(ll-1)
 125      continue
          zmunod(ilmax+1)=0.5*(zmunod(ilmax)+zmunod(ilmax+2))
          itrap=itrap+1
 123    continue
c
       write(*,*)'wptrmuy4: itrap',itrap
        do 126 l=2,itrap
          mun(iypass+l)=zmunod(l)
 126    continue
      endif

      return
      end
