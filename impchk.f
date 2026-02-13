c
c
      subroutine impchk(k)
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     This routine computes the density gain (from dF/dt) and compares i
c     locally to the density gain from the r.h.s. of the F.P. equation.
c     sumleft and sumright are the integrated totals. error is the
c     normalized difference and should be as close to roundoff as
c     possible.
c..................................................................

      include 'param.h'
      include 'comm.h'
      include 'advnce.h'
C%OS  
      dimension zermx(51),imx(51),jmx(51),zsumj3(iymax),zsumj4(iymax)
      !YuP[2021-03-11] Renamed iy-->iymax in declarations, just in case.
      dimension zdiffj(iymax),zdiffi(jx)
      dimension zdns(lrors) !local (not really needed)
C%OS  
      fpithta(i,j)=f(i+1,j,k,l_)*(1.-dithta(i,j,l_)) + 
     +  f(i  ,j,k,l_)*dithta(i,j,l_)

c.......................................................................

      sumleft=0.
      sumright=0.
      call bcast(tam1,zero,jx)
      zdns(l_)=0.0

c..................................................................
c     temp3 contains the r.h.s. and temp4 the l.h.s.
c..................................................................

      do 10 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
        if ((i.eq.itl .or. i.eq.itu) .and. cqlpmod.ne."enabled")go to 10
c%OS  do 2 j=1,jx
        do 2 j=2,jx-1
          temp4(i,j)=(f(i,j,k,l_)-f_(i,j,k,l_))
     1      *vptb(i,lr_)*cynt2(i,l_)*cint2(j)
     1      *one_
          temp3(i,j)=gfi(i,j,k)*qz(j)
          temp3(i,j)=temp3(i,j)-gfi(i,j-1,k)*qz(j)
          temp3(i,j)=temp3(i,j)+(hfi(i,j)-hfi(i-1,j))*ry(i,j)
          temp3(i,j)=(temp3(i,j)+vptb(i,lr_)
     1      *(cah(i,j)*f(i,j,k,l_)+so(i,j))+spasou(i,j,k,l_)+
     1      cthta(i,j)*(fpithta(i,j)-fpithta(i-1,j)) )
     1      *cynt2(i,l_)*cint2(j)*one_
          temp3(i,j)=temp3(i,j)*dtreff
          sumleft=sumleft+temp4(i,j)
          sumright=sumright+temp3(i,j)
          temp5(i,j)=gfi(i,j,k)
          temp6(i,j)=hfi(i,j)
          temp1(i,j)=qz(j)*(gfi(i,j,k)-gfi(i,j-1,k))
          temp2(i,j)=ry(i,j)*(hfi(i,j)-hfi(i-1,j))
          zdns(l_)=zdns(l_)+temp3(i,j)/dtreff-
     +      vptb(i,lr_)*spasou(i,j,k,l_)*cynt2(i,l_)*cint2(j)
 2      continue
 10   continue

c..................................................................
c     Pass/trapped boundary
c..................................................................

      if (cqlpmod.eq."enabled") go to 50

c%OS  should be j=1,jx? may depend on lbdry
      do 3 j=2,jx-1
        temp4(itl,j)=(f(itl,j,k,l_)-f_(itl,j,k,l_))
     1    *vptb(itl,lr_)*cynt2(itl,l_)
     1    *cint2(j)*one_
        temp4(itu,j)=temp4(itl,j)
        temp3(itl,j)=gfi(itl,j,k)*qz(j)
        temp3(itl,j)=temp3(itl,j)-gfi(itl,j-1,k)*qz(j)
        temp3(itl,j)=
     1    temp3(itl,j)-r2y(j)*(hfi(itl-1,j)-2.*hfi(itl,j)-hfi(itu,j))
        temp3(itl,j)=temp3(itl,j)+(cah(itl,j)*f(itl,j,k,l_)
     1    +so(itl,j))*vptb(itl,lr_) + spasou(itl,j,k,l_)
        temp3(itl,j)=temp3(itl,j)*dtreff*cint2(j)*cynt2(itl,l_)
     1    *one_
        temp3(itu,j)=temp3(itl,j)
        sumleft=sumleft+2.*temp4(itl,j)
        sumright=sumright+temp3(itl,j)*2.
        temp5(itl,j)=gfi(itl,j,k)
        temp5(itu,j)=gfi(itu,j,k)
        temp6(itl,j)=hfi(itl,j)
        temp6(itu,j)=hfi(itu,j)
 3    continue

 50   continue

C%OS  
c     compute highest errors
      do 99 ii=1,10
        zermx(ii) = 0.0
 99   continue
c
      do 100 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
        zsumj3(i) = 0.0
        zsumj4(i) = 0.0
        do 110 j=1,jx
          zsumj3(i) = zsumj3(i) + temp3(i,j)
          zsumj4(i) = zsumj4(i) + temp4(i,j)
          zerabs = abs(temp3(i,j)-temp4(i,j))
          if (zerabs .le. zermx(10)) go to 110
          if (zerabs .gt. zermx(1)) then
            do 111 ii=9,1,-1
              zermx(ii+1) = zermx(ii)
              imx(ii+1) = imx(ii)
              jmx(ii+1) = jmx(ii)
 111        continue
            zermx(1) = zerabs
            imx(1) = i
            jmx(1) = j
            go to 110
          endif
          do 112 int=9,1,-1
            if (zerabs.le.zermx(int) .and. zerabs.gt.zermx(int+1)) 
     +        then
              do 113 ii=9,int+1,-1
                zermx(ii+1) = zermx(ii)
                imx(ii+1) = imx(ii)
                jmx(ii+1) = jmx(ii)
 113          continue
              zermx(int+1) = zerabs
              imx(int+1) = i
              jmx(int+1) = j
              go to 110
            endif
 112      continue
 110    continue
        zdiffj(i) = zsumj4(i)-zsumj3(i)
 100  continue
      do 120 j=1,jx
        zsumi3 = 0.0
        zsumi4 = 0.0
        do 121 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
          zsumi3 = zsumi3 + temp3(i,j)
          zsumi4 = zsumi4 + temp4(i,j)
 121    continue
        zdiffi(j) = zsumi4-zsumi3
 120  continue
c
C%OS  if (l_.eq.1 .and. (n/2)*2..eq.n)
C%OS  + write(6,'(1pe10.2,2i5)') (zermx(ii),imx(ii),jmx(ii),ii=1,10)

c..................................................................
c     Symmetry about pi/2. in trapped region.
c..................................................................

      if (symtrap .eq. "enabled") then
        do 4 i=iyh+1,itu
          ii=iy_(l_)+1-i  !iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
          do 6 j=1,jx
            temp3(i,j)=temp3(ii,j)
 6        continue
 4      continue
      endif

c..................................................................
c     Compute the normalized error.
c..................................................................


      error=(sumleft-sumright)/xlndn(k,lr_)
      if (iactst.eq."abort") then
        if (error.gt.1.e-8) call diagwrng(15)
      endif

c%OS  
      if (l_ .eq. lrors) then
        idumy=0
 999    continue
      endif
c%OS  
      return
      end
