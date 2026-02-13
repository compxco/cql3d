c
c
      subroutine sourcef
      implicit integer (i-n), real*8 (a-h,o-z)
      save
      include 'param.h'
      include 'comm.h'


c..................................................................
c     If soucoord.ne."disabled",
c     computes the source m of species k - simple Gaussian profiles.
c     If knockon.ne."disabled",
c     also calc. knock-on source of electrons (Besedin and Pankratov).
c..................................................................

      !YuP[2022-02-10] Added logics for CQLP runs.
      !The original design below finds a BA source
      !by summing up local sources along field line.
      !In CQLP, we should not do it.
      !However, all relevant arrays are independent on l index.
      !To make this work for CQLP, assume that the source is set
      !at l=1 only. Use szm1z(1,1,1)=0. in cqlinput, 
      !otherwise it will not be detected.
      if(cqlpmod.ne."enabled")then !YuP[2022-02-10] added:
      l_start=1
      l_end=lz  !CQL3D case
      else ! cqlpmod.eq."enabled"  CQLP case
      l_start=1
      l_end=1   !CQLP case
      endif

      do 900 k=1,ngen
      if (soucoord .ne. "disabled") then
            
        do 800 m=1,nso
          if (n .lt. nonso(k,m) .or. n .gt. noffso(k,m)) go to 800
          call bcast(temp2(0,0),zero,iyjx2)
          do 700 l=l_start,l_end  !was 1,lz

            if (cqlpmod.eq."enabled") then !YuP[2025-09-01] added
              iiy=iy_(l) !CQLP: iy_ at given poloidal index;
            else !CQL3D
              iiy=iy_(l_) !CQL3D: at given radial index
            endif
          
            if(cqlpmod.ne."enabled")then !YuP[2022-02-10] CQL3D/CQLP distinction
              !--CQL3D
              !Determine the local source at z(l,lr_) in preparation for bounce-av
              do i=1,imax(l,lr_)
              ii=iiy+1-i !YuP[2021-03-11]CQL3D iy-->iy_(l_)
              call soup(cosz(i,l,lr_),l,k,m)  !theta_loc<pi/2
              do j=1,jx
                temp1(i,j)=soupp(j,lr_)
              enddo ! j
              call soup(cosz(ii,l,lr_),l,k,m)  !theta_loc>pi/2
              do j=1,jx
                temp1(ii,j)=soupp(j,lr_)
              enddo ! j
              enddo ! i 
              !Do the bounce average of the source..
              do i=1,imax(l,lr_)
              ii=iiy+1-i ! !YuP[2021-03-11] iy-->iy_(l_) [Here: in l-loop]
              ax=dtau(i,l,lr_)/tau(i,lr_)
              if ((l.ne.lz .and. lmax(i,lr_).eq.l)) then
                ax=ax+dtau(i,l+1,lr_)/tau(i,lr_)
              endif
              do j=1,jx
                temp2(i,j)=temp2(i,j)+ax*temp1(i,j)
                temp2(ii,j)=temp2(ii,j)+ax*temp1(ii,j)
              enddo ! j
              enddo ! i
            
            else !CQLP:  no bounce-averaging
            
              do i=1,iiy
                call soup(cosz(i,l,lr_),l,k,m)  !all theta_loc =[0:pi]
                do j=1,jx
                  temp2(i,j)=soupp(j,lr_)
                enddo ! j
              enddo ! i
              
            endif ! cqlpmod
             
 700      continue  ! On l

c..................................................................
c     Zero out source where less than small fraction of the
c     peak.  This facilitates use of ineg='enabled' option
c     zeroing out f where f<0 occurs, for all j above max of
c     the source.   BH120317.
c..................................................................
          temp2frac=1.d-8
          temp2max=zero
          do j=1,jx
             do i=1,iymax !!YuP[2021-03-11] iy-->iy_(l_)
                temp2max=max(temp2max,temp2(i,j))
             enddo
          enddo
          temp2frac=temp2frac*temp2max
          do j=1,jx
             do i=1,iymax !!YuP[2021-03-11] iy-->iy_(l_)
                if (temp2(i,j).lt.temp2frac) temp2(i,j)=zero
             enddo
          enddo
          

c..................................................................
c     Compute the density in preparation for scaling to achieve
c     the desired current.
c..................................................................
          s1=0.d0
          do 300 i=1,iymax !!YuP[2021-03-11] iy-->iymax or iy_(l_)
            do 250 j=1,jx
              s1=s1+temp2(i,j)*cynt2(i,l_)*cint2(j)*vptb(i,lr_)
              !Note: vptb(:,:)=1.d0 for CQLP runs
 250        continue
 300      continue

c..................................................................
c     Determine q1 so that the flux surface averaged current
c     will be asor (particles/sec/cc) after scaling.
c..................................................................

          if (s1.ne.zero) q1=asor(k,m,lr_)/s1*zmaxpsi(lr_)/one_

c..................................................................
c     scale the current to be equal to asor
c..................................................................

          call dscal(iyjx2,q1,temp2(0,0),1)
          do 200 j=1,jx
          do 150 i=1,iymax !!YuP[2021-03-11] iy-->iymax or iy_(l_)
           !source(i,j,k,indxlr_)=source(i,j,k,indxlr_)+temp2(i,j) !before[2022-02-11]
            source(i,j,k,l_)=source(i,j,k,l_)+temp2(i,j) !after[2022-02-11]
 150      continue
 200      continue

c..................................................................
c     xlncur is in units of particles/cm**2 (field line density)
c..................................................................

          xlncur(k,l_)=xlncur(k,l_)+asor(k,m,lr_)*zmaxpsi(lr_)
          xlncurt(l_)=xlncurt(l_)+xlncur(k,l_)
          !YuP[2022-02-11] now l_ (was lr_)
 800    continue !  m=1,nso
 
      endif !(soucoord .ne. "disabled")
 900  continue ! k=1,ngen
 
      return
      end subroutine sourcef
