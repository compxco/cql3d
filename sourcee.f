c
c
      subroutine sourcee
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     Subroutine sourcee is the controlling routine for the
c     analytic  SOURCE routines, all of which begin with "sou".
c     This is a facile model, it simply computes a source
c     profile (usually Gaussian in nature) with a specified
c     current. A more sophisticated model for ions utilizes NFREYA, the
c     Monte Carlo Beam deposition code. These are the "fr" routines with
c     frmod="enabled"
c     The "sou" routines are independent of the "fr" routines and vice-
c     versa.
c     Also, calls Knock On source modules, if specified.
c..................................................................

      save
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE     

c..................................................................
c     return if source is not to be recomputed this time step
c..................................................................

      if (nsou.eq.1 .or. n.eq.0) then
         continue
      elseif (mod(n,nsou).eq.1 .and. n.ne.1) then
         continue
      else
         return
      endif

c..................................................................
c     Return if the number of sources per species (nso) = 0
c..................................................................

      if (nso.eq.0) then
        return
      endif

c..................................................................
c     Initialization...
c..................................................................

      if (n.eq.0 .and. l_.eq.lmdpln_) then !---------------------------
         !YuP[2022-02-11] added l_.eq.lmdpln_ 
         !because now sourcee is called for every l_,
         !which in case of CQLP is l_=1:lz,
         !and the initialization should only be done
         !once, i.e., at l_=lmdpln_=1 point.
         !Note that subr.sounorm has internal l=1,lz loop.

c..................................................................
c     Compute constants used to determine Gaussian source profiles.
c     If soucoord="cart" specify cartesian species parameters,
c     and if soucoord="polar" specify polar parameters.
c     If soucoord="disabled", no gaussian sources.
c..................................................................

      if (soucoord .ne. "disabled") then
        !YuP[2022] This part should revised if we add l=1:lz index
        !into input arrays.
        do 20 k=1,ngen
          do 10 m=1,nso
            if (soucoord.eq."cart") then
              sxllm1(k,m,lr_)=sign(one,sellm1(k,m))*
     1          sqrt(abs(sellm1(k,m))/fions(k))
              sxllm2(k,m,lr_)=sellm2(k,m)/fions(k)
              sxppm1(k,m,lr_)=sqrt(seppm1(k,m)/fions(k))
              sxppm2(k,m,lr_)=seppm2(k,m)/fions(k)
            else
              xem1(k,m,lr_)=sqrt(sem1(k,m)/fions(k))
              xem2(k,m,lr_)=sem2(k,m)/fions(k)
              cosm1(k,m,lr_)=cos(sthm1(k,m)*pi/180.)
              cosm2(k,m,lr_)=scm2(k,m)
              !Note: fions(k) is enorm, when k=kiong
              write(*,*)'sourcee: sem1',sem1(k,m)
              write(*,*)'sourcee: xem1,xem2,cosm1,cosm2',
     &         xem1(k,m,lr_),xem2(k,m,lr_),cosm1(k,m,lr_),cosm2(k,m,lr_)
            endif
            zm1(k,m,lr_)=zmax(lr_)*szm1(k,m)
            zm2(k,m,lr_)=(zmax(lr_)*szm2(k,m))**2
            write(*,'(a,1p4e12.5)')'sourcee: zm1,zm2,z(lz),zmax=',
     &        zm1(k,m,lr_),zm2(k,m,lr_),z(lz,lr_),zmax(lr_)
 10       continue
 20     continue
      endif  ! On soucoord

c..................................................................
c     sounor(k,m,l,lr_) will contain normalization constants 
c     after the call to sounorm
c..................................................................

        call bcast(sounor(1,1,1,lr_),one,lz*ngen*nso) !YuP[2021-04] nsoa-->nso
        if (soucoord .ne. "disabled") then
        !------------
        call sounorm ! at n=0 only. Get sounorm(k,m,l,lr_) for every l=1:lz
        !------------
        do 40 k=1,ngen
          do 30 m=1,nso
            if (nonso(k,m) .eq. 1) nonso(k,m)=0
 30       continue
 40     continue
        endif  ! On soucoord
      endif  ! On n.eq.0 -------------------------------------------

      if ((ampfmod.eq."disabled") .or.
     &    (ampfmod.eq."enabled" .and. it_ampf.eq.1) ) then !YuP[2020-01]
        !YuP[2020-01] That is, when ampfmod.eq."enabled", 
        !initialize the two arrays below only at 1st iteration, 
        !and then reuse them at higher iterations, i.e., it_ampf>1.
        !Note: there is if(ampfmod.eq."enabled" .and. it_ampf.gt.1)return
        !clause in sourceko.f. 
        !So, if we don't use a corresponding clause here,
        !then at it_ampf>1 the arrays source(), xlncur(1,l_)  
        !will be set to 0.0 but not computed in sourceko.f.
c..................................................................
c     Initialize the source profile to zero.
c..................................................................
        !call bcast(source(0,0,1,indxlr_),zero,iyjx2*ngen) !before[2022-02-11]
        call bcast(source(0,0,1,l_),zero,iyjx2*ngen) ![2022-02-11]
c..................................................................
c     xlncur will contain the source current (/cm**2/sec).
c     In general asor*zmaxpsi(lr_)=xlncur  (asor in units particles/cc)
c..................................................................
        call bcast(xlncur(1,l_),zero,ngen) !YuP[2022-02-11] now l_ (was lr_)
      endif  !YuP[2020-01]

c..................................................................
c     Determine Gaussian+knock-on  source(i,j,k,l_) array
c..................................................................

      call sourcef

      call sourceko
     
c..................................................................
c     define source profile uniquely at x=0.
c..................................................................

      call sourc0

c..................................................................
c     Compute the source power.
c..................................................................
      do k=1,ngen
        write(*,*)'sourcee: l_,max(source)=',l_,maxval(source(:,:,k,l_))
        call sourcpwr(k) ! From source() array [may include NBI or KO]
      enddo
      !write(*,*)'sourcee: sorpw_nbi', sorpw_nbi(1,1)
      return
      end subroutine sourcee
