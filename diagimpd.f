c
c
      subroutine diagimpd(k)
      implicit integer (i-n), real*8 (a-h,o-z)
c..................................................................

c     This routine computes the density gain (loss) due to particle
c     sources (losses) from call to time advancement routine impavnc.
c     Gains are positive, losses negative.
c     The indices refer to the following processes:
c     1- density gained due to setting negative values of
c     distribution function to 0 during velocity sweep.
c     Will be zero for implicit time advancement.
c     Indices 1 and 2 used only with splitting scheme.
c     2- density gained due to setting negative values of
c     distribution function to 0 during theta sweep. (sub diagxswt)
c     Will be zero for implicit time advancement.
c     3- particle source contribution.
c     4- particle loss due to flux at high velocity end.
c     5- loss orbits (part of krook operator)
c     6- toroidal losses(part of krook operator)
c     7- fusion losses (part of krook operator)
c     8- density gained due to setting negative values of
c     distribution to 0 after call to impadv (implicit).
cBH180809: Should add calc of density change due to rescaling?
cBH180809: Done elsewhere? Yuri added plot of added particles at .ps end 


c
c     Further the normalized velocity flux at mesh point x(j+.5) is put
c     into vflux(j,k,l_). In the absence of Krook or source terms
c     vflux(jx,k,l_)=1/n * dn/dt *tauee(lr_)  (n is line density).
c

c     Last the routine sets f(i,j,k,l_)=0 if f(i,j,k,l_) < 0
c     if ineg .eq. "enabled".or."trunc_d"
c..................................................................


      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      include 'advnce.h'

      dimension lefcti(12) ! local working array-index
c..................................................................
c     Call routine to scale distribution if desired.
c..................................................................
ccc      call diagscal(k)  ! YuP: moved outside

c..................................................................
c     Initialize..
c..................................................................


      !call dcopy(iyjx2,f(0,0,k,l_),1,temp2(0,0),1)
      temp2(0:iymax+1,0:jx+1)= f(0:iymax+1,0:jx+1,k,l_)
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)
      sgain(1,k)=0.
      sgain(2,k)=0.
      sgain(3,k)=0.
      sgain(4,k)=0.
      sgain(5,k)=0.
      sgain(6,k)=0.
      sgain(7,k)=0.
      sgain(8,k)=0.
      lefcti(1)=-1
      lefcti(2)= 0
      lefcti(3)= 1
      lefcti(4)= 2
      lefcti(5)= 3
      lefcti(6)= 5
      lefcti(7)= 6
      lefcti(8)= 7
      lefcti(9)= 8
      lefcti(10)= 11
      lefcti(11)= 12
      call bcast(tam4,zero,jx)
      call bcast(tam5,zero,jx)
      call bcast(tam6,zero,jx)
cBH081125:  Added zeroing of vflux.  Otherwise, compiler dep results?
      call bcast(vflux(1,k,l_),zero,jx)

c..................................................................
c     Krook operator terms are integrated over velocity space.
c     Simultaneously, compute new density.
c..................................................................

      do 10 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
        !Note that for meshy="fixed_mu" iy_(l_) can be less than iy
        do 2 j=1,jx
          tam4(j)=tam4(j)+vptb(i,lr_)*temp2(i,j)*cynt2(i,l_)
c
c     Bad orbits..
          tam5(j)=tam5(j)+vptb(i,lr_)*dtreff*gon(i,j)*temp2(i,j)*
     *      cynt2(i,l_)
c
c     Toroidal losses
          tam6(j)=tam6(j)-vptb(i,lr_)*dtreff/taulos(i,j,indxlr_)
     1      *temp2(i,j)*cynt2(i,l_)
 2      continue
 10   continue

c..................................................................
c     Integrate over x (speed) to obtain relevant diagnostics
c     and the density.
c..................................................................

      sden=0.
      do 3 j=1,jx
        sgain(5,k)=sgain(5,k)+tam5(j)*cint2(j)*one_ ! Bad orbits
        sgain(6,k)=sgain(6,k)+tam6(j)*cint2(j)*one_ ! Tor. losses
        sden=sden+tam4(j)*cint2(j)*one_
c     For diagnostic purposes to compare with tem5 at end of diaggnde:
c     Summing tam7 over j will give flux surface averaged density.
        tam7(j)=tam4(j)*cint2(j)/zmaxpsi(lr_)
 3    continue

c..................................................................
c     Ion source (freyasou) or electron_KO (sourceko) gain..
c..................................................................

      sgain(3,k)=xlncur(k,l_)*dtreff !YuP[2022-02-11] now l_ (was lr_)

c..................................................................
c     Compute:
c     (1) losses at high velocity terminator
c     (2) normalized flux as a function of velocity
c..................................................................
      do 6 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
        do 5 j=1,jx
          vflux(j,k,l_)=vflux(j,k,l_)+one_*gfi(i,j,k)*cynt2(i,l_)
 5      continue
 6    continue

c..................................................................
c     Compute density lost at upper terminator (j=jx)
c..................................................................

      sgain(4,k)=vflux(jx,k,l_)*dtreff ! Note: j=jx here
ccc       write(*,'(a,i5,e13.5)')'diagimpd sgain(4,k)=', l_,sgain(4,k)

c..................................................................
c     Compute normalized flux
c..................................................................

      if(cqlpmod.ne."enabled")then !YuP[2021-03-30] CQL3D/CQLP treatment
        do j=1,jx
         vflux(j,k,l_)=vflux(j,k,l_)*tauee(lr_)/xlndn00(k,lr_) !CQL3D
        enddo
      else !cqlpmod.eq."enabled"
        do j=1,jx
         vflux(j,k,l_)=vflux(j,k,l_)*tauee(ls_)/xlndn00(k,lr_) !CQLP
        enddo
      endif

      if (n .eq. 0 .or. n/nchec*nchec .ne. n)  go to 90

c..................................................................
c     determine energy transfer diagnostics
c..................................................................

      entr(k,4,l_)=0.
      
      do lfct=1,11 ! cpu-intensive calc. for lefct= -1:3; 11 and 12
        lefct=lefcti(lfct) ! == -1:3; 5:8; 11,12; 
        !(4 is skipped because it's a sum;  9,10 calc-ed below)
CMPIINSERT_MPIWORKER_LFCT
CMPIINSERT_IF_RANK_EQ_MPIWORKER
        call diagentr(lefct,k)
CMPIINSERT_ENDIF_RANK
CMPIINSERT_SEND_RECV_ENTR
      enddo ! lfct 
      
CMPIINSERT_BARRIER
CMPIINSERT_BCAST_ENTR

      if (k.eq.kelecg) sorpw_rf(kelecg,lr_)=entr(k,3,l_) !-YuP
      if (k.eq.kiong(k)) sorpw_rf(k,lr_)=entr(k,3,l_) !-YuP RF source
      
 90   continue
      s=0.

c..................................................................
c     if ineg="enabled" or "enabled1" or "trunc_d", set f.le.0 to zero.
c..................................................................

      !call dcopy(iyjx2,temp2(0,0),1,temp1(0,0),1)
      temp1=temp2 ! for all (0:iy+1,0:jx+1)
      call bcast(temp4,zero,iyjx2)

      if (ineg .eq. "disabled") go to 141

      if (ineg.eq."enabled" .or. ineg.eq."enabled1" 
     1     .or.ineg.eq."trunc_d") then
      ! [YuP-101228: Was there an error? (ineg.ne."enabled")]
      !------------------------------------------------------------
      ! Before 03-2011: Set f to 0 for points where f(i,j)<0.
      ! After  03-2011: If a point f(i,j).le.0 is found, 
      ! f(i,jj) is set to 0 for ALL jj.ge.jmin_nosource
      ! where jmin_nosource is the smallest j above which 
      ! there is no source (for any i).
      ! For points below jmin_nosource, if f<0 is found,
      ! set f to zero for this point (as done before).
      ! As of 20110408, ineg.eq."enabled1" sets to zero
      ! as for "enabled", except don't check for source points.
      !------------------------------------------------------------
      jmin_nosource=1 
      if (ineg.ne."enabled1") then
                !If source is present, jmin_nosource.gt.1 will be found.
      do j=jx,1,-1 ! Scan from largest v towards zero
        !source_j=sum(abs(source(:,j,k,lr_))) ! Sum over i (all i-range)!before[2022-02-11]
        source_j=sum(abs(source(:,j,k,l_))) !Sum over i ![2022-02-11]
        if (source_j .ne.zero) then !source is present for this j-level
          jmin_nosource= j+1 ! Smallest j above which there is no source
          goto 409
        endif
      enddo
 409  continue
      
      do 411 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
      do 410 j=1,jx
         if ( temp2(i,j) .le. zero) then ! Found f(i,j)<=0.
            ! Set f to zero at this local point.
            temp1(i,j)=zero ! Set this f(i,j) to zero.
            temp4(i,j)=-temp2(i,j) ! For diagnostics.
            j0=max(j,jmin_nosource)
            do jj= j0,jx ! Scan all jj>=jmin_nosource
               temp1(i,jj)=zero ! Set f(i,jj) to zero.
               temp4(i,jj)=-temp2(i,jj) ! For diagnostics.
            enddo
         endif
 410  continue
 411  continue

      else ! on ineg

      f011=temp2(1,1)
      fmin=1.d-20*f011
      do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
      do j=1,jx
         if ( temp2(i,j) .le. fmin) then ! Found f(i,j)<=fmin
            ! Set f to zero at this local point.
            temp1(i,j)=zero ! Set this f(i,j) to zero.
            temp4(i,j)=-temp2(i,j) ! For diagnostics.
            j0=max(j,jmin_nosource)
            do jj= j0,jx ! Scan all jj>=jmin_nosource
               temp1(i,jj)=zero ! Set f(i,jj) to zero.
               temp4(i,jj)=-temp2(i,jj) ! For diagnostics.
            enddo
         endif
      enddo ! On i
      enddo ! On j

      endif  ! On ineg.ne."enabled1"
c
c     routine diagdens will compute density gained by setting negative
c     values of distribution to 0.
c
      call diagdens(xline,xmidp,eline)
      sgain(8,k)=xline*one_ ! from f<0 being set to 0.
      engain(k)=eline*one_

      elseif(ineg.eq."renorm") then

c     find min and max of f
        fmin=ep100
        fmax=-ep100
        do 500 j=1,jx
          call aminmx(temp1(1,j),1,jx,1,ffmin,ffmax,jmin,jmax)
c990131          fmin=amin1(ffmin,fmin)
c990131          fmax=amax1(ffmax,fmax)
          fmin=min(ffmin,fmin)
          fmax=max(ffmax,fmax)
 500    continue

        if(fmin.lt.0.) then
c     calculate line average density of f
          call dcopy(iyjx2,temp1(0,0),1,temp4(0,0),1)
          call diagdens(xline0,xmidp0,eline0)
c     add 1.1*fmin to f
          fadd=1.1*abs(fmin)
          do 520 j=1,jx
            do 521 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
              temp1(i,j)=fadd+temp1(i,j)
 521        continue
 520      continue
c     recalculate density
          call dcopy(iyjx2,temp1(0,0),1,temp4(0,0),1)
          call diagdens(xline1,xmidp1,eline1)
c         renormalize f to old density
          ffac=xline0/xline1
          do 530 j=1,jx
            do 531 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
              temp1(i,j)=ffac*temp1(i,j)
 531        continue
 530      continue
c     change of line density (/2) and line energy density 
          sgain(8,k)=(xline1-xline0)*one_
          engain(k)=(eline1-eline0)*one_
        endif ! on fmin
        
      endif  ! on ineg


 141  continue
 
CMPIINSERT_IF_RANK_EQ_0      
      !WRITE(*,*)'diagimpd fmin,jmin_nosource=',fmin,jmin_nosource
!      WRITE(*,'(a,2i4,2e15.7)')
!     + 'diagimpd BEFORE f update. n,l_,MIN(temp2),MAX(temp2)',
!     +   n,l_,MINVAL(temp2),MAXVAL(temp2)
!      WRITE(*,'(a,2i4,2e15.7)')
!     + 'diagimpd neg.val.adjusted n,l_,MIN(temp1),MAX(temp1)',
!     +   n,l_,MINVAL(temp1),MAXVAL(temp1)
CMPIINSERT_ENDIF_RANK
      call dcopy(iyjx2,temp1(0,0),1,f(0,0,k,l_),1)
CMPIINSERT_IF_RANK_EQ_0      
!      WRITE(*,*)
!     + 'diagimpd AFTER  f update. MIN(f),MAX(f),SUM(f)=', 
!     +          MINVAL(f),MAXVAL(f),SUM(f)
CMPIINSERT_ENDIF_RANK
      !!!if(l_.eq.lrz)  pause
      
      if (n .gt. 0 .and. n/nchec*nchec .eq. n) then
        call diagentr(9,k)
        call diagentr(10,k)
      endif

      return
      end
      
CMPIINSERT_SUB_SEND_ENTR      
