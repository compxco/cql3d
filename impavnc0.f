c
c
      subroutine impavnc0(kopt)
      implicit integer (i-n), real*8 (a-h,o-z)

c.................................................................
c     Impose unicity point at u=0 as boundary condition
c
c     kopt=0:
c     This routine advances the Fokker-Planck equation one time step
c     for the implicit case (implct=enabled)
c     (IMPAVNC0 calls IMPCHK which differentiates the solution and
c     compares it with the r.h.s of the advanced F.P. equation to 
c     check for accuracy)
c     kopt=2: Coding still needed.  See achiefn.f comments.
c     kopt=3:
c     Calc h and g functions for iterative Amp-Far eqn solve
c.................................................................

      include 'param.h'
      include 'comm.h'
      include 'advnce.h'

CMPIINSERT_INCLUDE     

cBH080303    real*4 etime,tm1,tm(2) !Dclrtns for lf95 ETIME(). See man etime.
cBH080303    Using f95 intrinsic subroutine, cpu_time
      real*4 tm1,tm(2)    !Dclrtns for lf95 ETIME(). See man etime.

      character*8 alloc, factor, dalloc
      dimension zavarj1(iymax+4),ijaj1(iymax+4)
      !YuP[2021-03-11] Changed iy-->iymax in declarations
      !(just in case if iy is changed by iy=iy_(l_) somewhere)
      dimension  zmatcont(12), janelt(12)
      character*1 transpose

c.................................................................
c     matrix declared as pointer
c     need inewjmax in common to use with pointer
c.................................................................
      integer ipivot(iymax*jx) ! output of subr.dgbtrf(). No need to save.
      real*8,dimension(:,:),allocatable :: abd !(md1abd,inewjx) md1abd=3*(iymax+3)+1 
!      pointer abd 
!      integer ipivot(:) 
!      pointer ipivot
!      common/gauss_/ipivot
!      common/gauss_abd/abd
!      dimension ampfda(:,:),ampfdd(:,:)
!      pointer ampfda,ampfdd
!      common /ampf/ ampfda,ampfdd !BH: Saves this data
      real*8 ampfda(0:iymax,jx),ampfdd(0:iymax,jx)
!      save abd !YuP[2019-09] cannot use save: memory alloc problems on NERSC

      integer icount_imp   !Counter for calls to impavnc0
      data icount_imp/0/
      save inewmax, mlmax, mumax, icount_imp, icoeff, ieq, 
     1     ieq_tot, icoeff_est, icoeff_est_tot

      twopi2=twopi*twopi
c.......................................................................
c     Indicate progress of code
c.......................................................................

c     icount_imp counts total calls to impavnc0 (from achiefn).
c     At each time step: first call to impavnc0 is with lr_=1.
c     ifirst_lr=1 indicates lr_=1 call (otherwise ifirst_lr=0).
c     ilast_lr=1 indicates lr_=lrz call (otherwise ilast_lr=0).

c Not used anymore      icount_imp=icount_imp+1
      ifirst_lr=0
      ilast_lr=0
      if (l_.eq.1 .or. lrz.eq.1) ifirst_lr=1
      if (l_.eq.lrz) ilast_lr=1
c      write(*,'(a,4i5)')'impavnc0: n,icount_imp,ifirst_lr,ilast_lr',
c     +                     n,icount_imp,ifirst_lr,ilast_lr

c.......................................................................
c     nadjoint is set = 0, to remove effect of Olivier Sauter's adjoint
c        operator changes for this subroutine.   This is not checked
c        out for iterative solve.  Sauter has used it with the
c        soln_methed="direct" solve, in bootstrap current calculations
c        (O.Sauter, C. Angioni, and Y.R. Lin-Liu, Pop 66, 2834 (1999).
c.......................................................................

       nadjoint=0

c.......................................................................
c     icount_ese/icount_ampf are counters for eseswtch or ampfmod cases
c.......................................................................

       icount_ese=1
       icount_ampf=1

c.......................................................................
c     define effective number of unknowns in the theta direction and
c     resulting band widths
c     see below for comments on inew, etc.
c.......................................................................

      itl=itl_(l_)
      itu=itu_(l_)

      iyjx=iyjx_(l_) !YuP[2021-03-04]added. 
      ![Actually, these two lines are set in tdnflx]
      iyh=iyh_(l_) !YuP[2021-03-04]added. iyh is used in many places below
      !Note that for meshy="fixed_mu" iyh_(l_) can be less than iy/2 
      !But don't use iy=iy_(l_) here! It will overwrite iy,
      !which is the namelist value, and iy is used to set dimensions
      !of many arrays.
       
cBH071029:  During work on 3d fully-implicit solve, found a problem
cBH071029:  with iyh=itl cases.  Fixing this case, and moving ipassbnd
cBH071029:  setting to here [See further notes below for ipassbnd]:
cBH071029:  [But subsequently found had problems with iyh=itl elsewhere
cBH071029:  in the code, so suggest bigger rya(1) or iy for time being.]
      if (itl.lt.iyh) then
         inew= iy_(l_)/2 + itl-1 !YuP[2021-03-04] changed iy-->iy_(l_)
         ipassbnd=iyh-itl+1 
      else  !i.e., itl=iyh
         inew= iy_(l_)  !YuP[2021-03-04] changed iy-->iy_(l_) 
         ![2021-03-04] for meshy="fixed_mu" iy_(l_) can be less than iy 
         ipassbnd=0  !i.e., no trapped region for iyh=itl, 
                     !as in following case.
      endif
      if (symtrap .ne. "enabled") then !For CQLP, symtrap has to be "disabled"
         inew=iy_(l_)  !YuP[2021-03-04] changed iy-->iy_(l_) 
         !Note that for meshy="fixed_mu" iy_(l_) can be less than iy 
         ipassbnd=0 ! for symtrap='disabled'
      endif
      if (nadjoint.eq.1 .and. symtrap.eq."enabled") inew=2*itl-1
      if (symtrap.ne."enabled" .or. nadjoint.eq.1) ipassbnd=0

c     On first call to impanvc0, set up some max dimensions
ccc      if(icount_imp.eq.1) then 
        inewmax=inew
        if (meshy.eq."fixed_mu".or. manymat.eq."enabled") then
          do 15 ll=2,lrors
            inew1=iy_(ll)/2+itl_(ll)-1
            ![2021-03-04] for meshy="fixed_mu" iy_(l_) can be less than iy 
            if (symtrap .ne. "enabled") inew1=iy_(ll)
            if (nadjoint.eq.1 .and. symtrap.eq."enabled")
     +        inew1=2*itl_(ll)-1
            if (inew1.gt.inewmax) inewmax=inew1
 15       continue
        endif
        mlmax=inewmax+2 ! YuP: not used?
        mumax=inewmax+2 ! YuP: not used?
        inewjmax=inewmax*jx
ccc      endif

      ml=inew+2
      mu=inew+2
      if (symtrap.ne."enabled" .and. cqlpmod.ne."enabled") then
c     itu+1 contributes to the equation for itl
        ml=inew+3
        mu=inew+3
        mlmax=inewmax+3 ! YuP: not used?
        mumax=inewmax+3 ! YuP: not used?
      endif

      ibandto=ml+mu+1
      inewjx=inew*jx
      navnc=navnc+1
      md1abd= 3*(iymax+3)+1 ! Should be at least 2*ml+mu+1 (see below)
      !Note that for meshy="fixed_mu", iy_(l_) can be less than iy,
      !but here md1abd is used for dimensioning of abd() array   
      !write(*,*)'impavnc0: l_,inew,inew_(l_)=', l_,inew, inew_(l_) 

      !YuP[07-2017] Moved this part from micxinit and tdinitl, 
      ! as it may depend on k in a multi-species run
      ieq_tot=0
      do ll=1,lrors
         if(symtrap.eq."enabled") then
           !Number of indep theta pts, for symmetric trapped region
           inew_(ll)=iy_(ll)/2+itl_(ll)-1   
         else
           inew_(ll)=iy_(ll) !general case (not symmetric trapped reg.)
         endif
         inewjx_(ll)=inew_(ll)*jx   !Number eqns for ll
         ieq_tot=ieq_tot+inew_(ll)*jx 
         if (ll.eq.1) then
            ieq_(ll)=1
         else  ! Eqn no. at beginning of each flux surface:
            ieq_(ll)=ieq_(ll-1)+inew_(ll-1)*jx  
         endif
      enddo
      ieq_(lrors+1)=ieq_tot
      ieq_tot=ieq_(lrors+1)

cBH070419:   ITSOL Add, down to ....
c.................................................................
c     Set size of LAPACK and SPARSKIT2 matrices, and allocate space
c     Probably should move these calcs to it3dalloc.f
c.................................................................

c      write(*,*)'impavnc0: iy,iyh,itl,itu,ml,mu,inew,jx,inewjx',
c     +     iy,iyh,itl,itu,ml,mu,inew,jx,inewjx

         iwk_ilu=25000000  !sub ilut will give error if not suff lrg
c                          !NEED bigger than single flux surface solve
c                          !values, for given iy,jx.
         krylov=50  !Krylov subspace size, see below
         
      if ( soln_method.eq.'itsol' ) then
         icoeff_est=(iymax+3)*3+(jx-1)*9*inew  !Number of coeffs + abit
         icsrij=icoeff_est
         icsrip=inewjx+1
         icsri2=2*inewjx
         icsrikry=(krylov+1)*inewjx
      endif  !  on soln_method.eq.'itsol'

      if ( soln_method.eq.'itsol1' ) then
         lapacki=ml+mu+1
         lapackj=inewjx
         icsrij=lapacki*lapackj
         icsrip=lapackj+1
         icsri2=2*lapackj
         icsrikry=(krylov+1)*lapackj
      endif  !  on soln_method.eq.'itsol1'

      if ( soln_method.eq.'it3dv' .and. ifirst_lr.eq.1 ) then
         ! no transport here
         icoeff_est_tot=0
         do ll=1,lrors
            icoeff_est=(iy_(ll)+2)*3+(jx-1)*9*inew_(ll) !Number coeffs + abit
            icoeff_est_tot=icoeff_est_tot+icoeff_est
         enddo
         icsrij=icoeff_est_tot
         icsrip=ieq_tot+1
         icsri2=2*ieq_tot
         icsrikry=(krylov+1)*ieq_tot
      endif

      if ( soln_method.eq.'it3drv' .and. ifirst_lr.eq.1 ) then
         icoeff_est_tot=0
           do ll=1,lrors
              icoeff_est=(iy_(ll)+2)*3+(jx-1)*9*inew_(ll) !Number coeffs + abit
     +                  + 2*inew_(ll)*jx  !  for additional Drr coeffs
              icoeff_est_tot=icoeff_est_tot+icoeff_est
           enddo
           icsrij=icoeff_est_tot
           icsrip=ieq_tot+1
           icsri2=2*ieq_tot ! for jw_ilu(icsri2) can be just ieq_tot ?
           icsrikry=(krylov+1)*ieq_tot
           ! Above gives storage for the velocity space coeffs.
           ! Additional storage is needed for the radial coeff matrix
           ! This data will be stored in ar_csr, jar_csr, iar_csr
           icsrijr=ieq_tot*3   ! roughly .le.3 coeffs per equation
           ! Also require larger matrix to contain the CSR addition of
           ! the velocity and radial space coeff matrices.
           icsrijc=icsrij+2*ieq_tot !sum of a_csr and off-diagon. ar_csr
      endif

      iwk_ilu=max(icsrijc*5,50000000)
      !iwk_ilu=25000000 !sub ilut will give error (-2,-3) if not suff lrg
                        !NEED bigger than single flux surface solve
                        !values, for given iy,jx.

c     Allocate it3d related storage [No-op for soln_method='direct']

c-YuP      if (ifirst_lr.eq.1 .and. n.eq.1 .and. nefiter.eq.1) call it3dalloc
      if(soln_method.ne.'direct') then 
        if(ASSOCIATED(rhs0)) then  !-YuP  New logic for allocation
          ! rhs0 and other arrays are already allocated => do nothing
        else ! Not allocated yet
          call it3dalloc
        endif
      endif

c     Just to check:
c      write(*,*)'impavnc0:size(abd_lapack,1),size(abd_lapack,2)=',
c     +     size(abd_lapack,1),size(abd_lapack,2)
c      write(*,*)'impavnc0:size of a_csr,ja_csr,ia_csr,alu,jlu,ju',
c     +     size(a_csr),size(ja_csr),size(ia_csr),size(alu),
c     +     size(jlu),size(ju)
c      write(*,*)'impavnc0:size of jw_ilu,w_ilu_rhs0,sol,vv',
c     +     size(jw_ilu),size(w_ilu),size(rhs0),size(sol),size(vv)

 

c     So, no confusion:
      if (soln_method.eq.'itsol1') 
     +                 call bcast(abd_lapack(1,1),zero,icsrij)

c      write(*,*)'impavnc0: size(ja_csr),icsrip=',size(ja_csr),icsrij
c      write(*,*)'impavnc0: size(ia_csr),icsrip=',size(ia_csr),icsrip

      if (soln_method.eq.'itsol' .or. soln_method.eq.'itsol1') then
         call bcast(a_csr(1),zero,icsrij)
         call ibcast(ja_csr(1),0,icsrij)
         call ibcast(ia_csr(1),0,icsrip)
      elseif (soln_method.eq.'it3dv') then
         if (ifirst_lr.eq.1) then
            call bcast(a_csr(1),zero,icsrij)
            call ibcast(ja_csr(1),0,icsrij)
            call ibcast(ia_csr(1),0,icsrip)
         endif
      elseif (soln_method.eq.'it3drv') then
         if (ifirst_lr.eq.1) then
cBH080429   Problem:  ipofi only calculated once in code, in tdtranspn.
cBH080429            call ibcast(ipofi(1,1),0,iy*lrz)
            call bcast(a_csr(1),zero,icsrij)
            call ibcast(ja_csr(1),0,icsrij)
            call ibcast(ia_csr(1),0,icsrip)
            call bcast(ar_csr(1),zero,icsrijr)
            call ibcast(jar_csr(1),0,icsrijr)
            call ibcast(iar_csr(1),0,icsrip)
            call bcast(ac_csr(1),zero,icsrijc)
            call ibcast(jac_csr(1),0,icsrijc)
            call ibcast(iac_csr(1),0,icsrip)
         endif
      endif

c     Copy current distribution into f_..   YuP: moved to tdchief:
ccc      call dcopy(iyjx2*ngen,f(0,0,1,l_),1,f_(0,0,1,l_),1) 

c      write(*,*) 'impavnc0, l_,elecfld:', l_ ,elecfld(l_)  
c..................................................................
c     Logic for allocation/deallocation
c..................................................................

c-YuP      alloc="disabled"
c-YuP      dalloc="disabled"  !-YuP: not used
c     allocate once for all time-step
c-YuP      if ( soln_method.eq.'direct' .or. soln_method.eq.'itsol1') then
      
c-YuP  C--MPIREPLACE_ALLOC  
c-YuP: Now allocation is based on first call to impavnc0
c-YuP      if (l_.eq.1  .and. n.eq.1 .and. nefiter.eq.1)  then 
c-YuP         alloc="enabled"
c-YuP         ialloc=1
c-YuP      endif
c     Special case for multiple calls at n=nstop
c-YuP      if (n.eq.nstop.and.ialloc.eq.0) then
c-YuP          alloc="enabled"
         !-YuP  Problem: called during first iteration of eflditer;
         ! Should be after last nefiter iteration (which is unknown).
         ! Not really necessary to allocate at last time step?         
c-YuP      endif

c     deallocate once at last time-step
!-YuP C--MPIREPLACE_DEALLOC !-YuP: no deallocation anymore
c-YuP      if (l_.eq.lrors .and. n.ge.nstop) then
c-YuP         dalloc="enabled" !-YuP: not used
c-YuP         ialloc=0 !-YuP: not used
c-YuP      endif

c-YuP      endif  ! On soln_method

c..................................................................
c     Allocate space for abd and ipivot. 
c YuP-101231  New logic for allocation: 
c simply check whether the array is already allocated or not.
c..................................................................
      if (soln_method.eq.'direct'.or.soln_method.eq.'itsol1') then      
        lenabd=md1abd*inewjmax
        if(md1abd.lt.(2*ml+mu+1))then
          STOP 'impavnc0/direct: md1abd.lt.(2*ml+mu+1) detected. stop'
        endif
        if(iyjx.lt.inewjx)then
          write(*,*)'iyjx,inew*jx,inewjx_(l_)=',iyjx,inewjx,inewjx_(l_)
          STOP 'impavnc0/direct: iyjx.lt.inewjx detected. stop'
        endif
        if(size(rhs).lt.inewjx)then
          STOP 'impavnc0/direct: size(rhs).lt.inewjx detected. stop'
        endif
c-YuP      if (alloc.eq."enabled") then
        !if(ASSOCIATED(abd)) then  
        if(ALLOCATED(abd))then
         ! abd and ipivot are already allocated => do nothing 
          !deallocate(abd)
          !allocate(abd(md1abd,inewjx),STAT=istat) ! With new value of inewjx
          abd=zero !YuP[2019-04-24] was absent
          ipivot=0 !YuP[2019-04-24] was absent
        else ! Not allocated yet
          !write(*,*)'impavnc0 Before allocate; Size req. for abd:',lenabd
          !YuP[2019-04-24] allocate(abd(md1abd,inewjmax),STAT=istat) !Big size ~3*jx*iy^2
          ![2019-04-24] abd() is used in subr.dgbtrf and dgbtrs with dims abd(md1abd,inewjx)
          !So changing allocation to
          allocate(abd(md1abd,inewjx),STAT=istat) !YuP[2019-04-24] !Big size ~3*jx*iy^2
          ![2019-04-24] The problem is - inewjx may change from one flux surface
          ! to another, and that is what inewjmax was trying to account for
          !(it is the maximum of inew*jx = (iyh+itl-1)*jx over all flux surfaces).
          ! However, notice that impavnc0 is called sequentially 
          ! starting from ll=1 (smallest rho) to larger rho, and the region
          ! with passing particles is gradually reduced, 
          !  i.e., inew=(iyh+itl-1) is reduced.
          ! Then, we can use the abd() storage defined during very 1st call,
          ! i.e., during ll=1 call, when inew*jx is largest.
          !write(*,*)'impavnc0 abd: Allocate/istat=',istat
          abd=zero 
          ipivot=0 !call ibcast(ipivot,0,iyjx)
         ! will preset the values to zero in loop k=1,ngen
        endif
      endif ! on soln_method= direct or itsol1

c.................................................................
c     loop over all time advanced (general) species..
c.................................................................

      do 600 k=1,ngen

c..................................................................
c     if (colmodl.eq.4) f(i,j,ngen,l_) contains a drifting Maxwellian
c     whose momentum corresponds to that of the general electron
c     species contained in f(i,j,kelecg,l_);
c     the purpose of this species is to cool the general electrons
c     without absorbing net momentum;
c     the species is treated as a background or fixed field and is
c     updated only with respect to its momentum if the general
c     electron species momentum changes during the time step.
c..................................................................

        if (colmodl.eq.4 .and. k.eq.ngen) goto 600

c.................................................................
c     normalized time step..
c.................................................................

        rbgn=1./dtreff

c.................................................................
c     The routine coefstup(k) does the following:
c     Preset all coefficients to zero. da, db, and dc are the three
c     coefficients associated with the velocity (v) flux while
c     dd, de, df are associated with the theta flux. The contribution
c     from the Krook operator (cah(i,j)*f(i,j,l_)) is stored in cah.
c     Note cah does not contribute to da through df. These six
c     coefficients incorporate only wave-particle, collisional and
c     d.c. electric field effects. The ion source is represented
c     through so(i,j).
c     The routine coefstup(k) adds into the above arrays all the 
c     relevant physics.
c.................................................................

        call coefstup(k)
        
c.................................................................
c     Determine whether matrix factorization and related space
c     allocation and de-allocation are to be done.
c     To save memory, if ngen (the number of general species) is
c     greater than 1, factorization is done every time step.
c.................................................................

        impchg=impcoef+imprf+impelec+impcah+irstart

c..................................................................
c     Logic for LU factorization..
c..................................................................

        factor="disabled" !Using prior factorization
        if (ngen.gt.1 ) then
          factor="enabled" ! Factorize
        endif
        if (impchg.gt.0) factor="enabled" ! Factorize

c.................................................................
c     The coefficients of the equation are currently defined on the
c     same mesh as the distribution function f. The fluxes are best
c     defined (from the point of view of differencing and particle
c     conservation) on mid meshpoints. We undertake here to
c     interpolate the coefficients as needed to either (i,j+1/2)
c     (velocity flux) or to (i+1/2,j) (theta flux).
c     Finally to enforce boundary conditions (zero flux in general
c     except at the pass/trapped boundary) certain coefficients
c     are zeroed out or suitably averaged at specific mesh points.
c     The numbers 1,2,3 appearing in the calls below signify
c     which coefficient is being treated.

c     first the velocity flux..
c.................................................................

        call coefmidv(da,1)
        call coefmidv(db,2)
        call coefmidv(dc,3)

c.................................................................
c     the theta flux..
c.................................................................

        call coefmidt(dd,1)
        call coefmidt(de,2)
        call coefmidt(df,3)

c.................................................................
c     The differencing is done ala Chang and Cooper as generalized
c     by Karney to 2-D. This requires defining a couple of arrays
c     di(i,j,k,l_) and dj(i,j,k,l_) which are used to define f 
c     at mid mesh points: 
c     to wit f(i,j+1/2,k,l_)=f(i,j+1,k,l_)*(1.-dj(i,j,k,l_))+ 
c     f(i,j,k,l_)*dj(i,j,k,l_). Similarly for theta.
c     The routines coefwtj and coefwti are used to
c     determine the dj and di arrays.
c.................................................................

        call coefwtj(k)
        call coefwti(k)
        
c.................................................................
c     Loop-back point for eseswtch or ampfmod calculations
c.................................................................

 10     continue

c.................................................................
c     Proceed to setup coeffs and call the Gaussian elimination routine.
c     i indexes theta, j indexes speed and ieq keeps a running total. It
c     is necessary to separate out the various boundary equations
c     from the the inner 9-point equation. Boundary equations can
c     involve from 1 to 12 points (or unknowns) depending on location
c     in velocity space.  In addition, lbdry(k)='conserv'.or.'consscal'
c     is a special case at j=1 involving inew unknowns.

c     ieq will keep track of which equation is being considered
c     below. This defines the row of the matrix being inverted. The
c     theta mesh points (index i) are incremented in the inner
c     loop.
c.................................................................

        if (soln_method.eq."itsol" .or. soln_method.eq."itsol1"
     +      .or. soln_method.eq."direct") then
           ieq=0
           icoeff=0         !  CSR coefficient storage counter (ITSOL)
        elseif(soln_method.eq.'it3dv' .or. soln_method.eq.'it3drv') then
           ! reset counters only at first flux surface on new time-step
           if ( ifirst_lr.eq.1 ) then
              ieq=0
              icoeff=0      !  CSR coefficient storage counter 
           endif
        endif
!YuP        if (itl.lt.iyh) then
!YuP           inew=iy/2 + itl - 1
!YuP           ipassbnd=iyh-itl+1
!YuP        else                    !i.e., itl=iyh
!YuP           inew=iy
!YuP           ipassbnd=0           !i.e., no trapped region for iyh=itl, 
!YuP                                !      as in following case.
!YuP        endif
!        !YuP[2021-03-01] Why the above def. for inew and ipassbnd
!        !repeated here? In any case, it should be followed by 
!        if (symtrap .ne. "enabled") then !For CQLP, symtrap has to be "disabled"
!         inew=iy_(l_)
!         ipassbnd=0 ! for symtrap='disabled'
!        endif !YuP[2021-03-01] added
 
        
        
ccc        if(ieq. eq. ieq_(l_+1)-1)  ieq=ieq_(l_)-1     
c.................................................................
c     abd is the band matrix in a format expected by the sgbtrf routine.
c     That is, if a(i,j) is the matrix to invert, we have:
c     abd(iband+i-j,j) = a(i,j), with iband = ml+mu+1=total bandwidth
c     As if each column is moved up/down such that each diagonal element
c     is at abd(iband,j).  NOTE:  i,j in this comment designate the row
c     and column, respectively, of the general matrix a(,) to be
c     inverted, not the velocity space locations.
c
c     if factor .eq. "enabled" then we have to recompute the entire
c     matrix in preparation for factorization. Otherwise all
c     that has to be done is to recompute the r.h.s.  - The
c     code has to be called with factor .eq. "enabled" before it
c     can be called with factor="disabled".
c     
c     See below for comments on unknowns numbering
c.................................................................

        if (factor.eq."disabled") then   !Using prior factorization...
                                         !endif on factor at line 1834
          if (.not.(ampfmod.eq."enabled". and. kopt.eq.3
     +         .and.cqlpmod.ne."enabled" .and. k.eq.kelec)) then  

          WRITE(*,*)'impavnc0:  factor.eq."disabled", OK?'
          ipassbnd=iyh+1-itl
          if (symtrap.ne."enabled" .or. nadjoint.eq.1) ipassbnd=0
          do 2002 j=1,jx
            ihalf=0
            icntsww=1
            if (iyh.eq.itl) icntsww=0
            if (symtrap .ne. "enabled") icntsww=itl-iyh
            if (nadjoint.eq.1 .and. symtrap.eq."enabled") then
              icntsww=0
              ihalf=1
            endif
            do 2003 indexsww=1,inew
              ieq=ieq+1
              if (j .eq. 1 .and. (lbdry(k).ne."conserv" .and. 
     +          lbdry(k).ne."consscal")) then
                rhs(ieq)=f_(iyh,1,k,l_) ! YuP: Why iyh?  BH: all same
              else ! includes j=1 when lbdry(k).eq."conserv"
                if(indexsww.le.ipassbnd) then
                  i=iyh+1-indexsww
                elseif(ihalf.eq.0) then
                  i = itl - icntsww
                  ihalf = 1
                else
                  i = itu + icntsww
                  ihalf = 0
                  icntsww = icntsww + 1
                endif
                !For scal(neq,l_) :
                neq=ieq+(k-1)*iyjx   ! for multiple species (k.gt.1)

                alambda= vptb(i,lr_) ! ZOW or Hybrid-FOW

                rhs(ieq)=z00(i,j,k)
c.......................................................................
c               Modification for eseswtch.eq."enabled" to obtain Kupfer
c               g function.
c               This code section only to be used for factor.eq.disabled
c.......................................................................

                rhs(ieq)=z00(i,j,k)
                if (cqlpmod.eq."enabled".and.eseswtch.eq."enabled") then
                   cnst=-bnumb(k)/fmass(k)*charge/vnorm
                   if (j.ne.1.or.j.ne.jx) then
                      dfdvz=half*(f(i,j+1,k,l_)-f(i,j-1,k,l_))
     +                         *coss(i,l_)*dxi(j)
                      if (i.ne.1 .or. i.ne.iy_(l_) )then 
                         !YuP[2021-03-04] changed iy-->iy_(l_)
                         !for meshy="fixed_mu" iy_(l_) can be less than iy
                         dfdvz=dfdvz 
     +                         -half*(f(i+1,j,k,l_)-f(i-1,j,k,l_))
     +                         *sinn(i,l_)*xi(j)*dyi(i,l_)
                      endif
                   elseif (j.eq.1) then
                      dfdvz=zero
                   elseif (j.eq.jx) then
                      dfdvz=(f(i,jx,k,l_)-f(i,jx-1,k,l_))
     +                      *coss(i,l_)*dxi(jx)
                      dfdvz=dfdvz 
     +                      -half*(f(i+1,j,k,l_)-f(i-1,j,k,l_))
     +                      *sinn(i,l_)*xi(j)*dyi(i,l_)
                   endif
                   rhs(ieq)=cnst*dfdvz
                endif  !  On cqlpmod/eseswtch

                rhs(ieq)=rhs(ieq)*scal(neq,l_)

                if (nadjoint.eq.1 .and. symtrap.eq."enabled"
     +            .and. i.eq.itu) rhs(ieq)=zero !YuP[2019-04-24] was 0.
                
              endif  ! On if j.eq.1.and.lbdtry(k).ne.conserv;  else

 2003       continue ! On indexsww
 2002     continue  ! On j
          endif  ! On .not.(ampfmod....

                
c.......................................................................
c               Modification for ampfmod.eq."enabled" to obtain Kupfer
c               g funtion.
c               This code section only to be used for factor.eq.disabled
c.......................................................................

          if (ampfmod.eq."enabled". and. kopt.eq.3 .and.icount_ampf.eq.2
     +         .and.cqlpmod.ne."enabled" .and. k.eq.kelec) then  
c$$$                   cnst=-bnumb(k)/fmass(k)*charge/vnorm
c$$$                   if (j.ne.1.or.j.ne.jx) then
c$$$                      dfdvz=half*(f(i,j+1,k,l_)-f(i,j-1,k,l_))
c$$$     +                         *coss(i,l_)*dxi(j)
c$$$                      if (i.ne.1 .or. i.ne.iy) then
c$$$                         dfdvz=dfdvz 
c$$$     +                         -half*(f(i+1,j,k,l_)-f(i-1,j,k,l_))
c$$$     +                         *sinn(i,l_)*xi(j)*dyi(i,l_)
c$$$                      endif
c$$$                   elseif (j.eq.1) then
c$$$                      dfdvz=zero
c$$$                   elseif (j.eq.jx) then
c$$$                      dfdvz=(f(i,jx,k,l_)-f(i,jx-1,k,l_))
c$$$     +                      *coss(i,l_)*dxi(jx)
c$$$                      dfdvz=dfdvz 
c$$$     +                      -half*(f(i+1,j,k,l_)-f(i-1,j,k,l_))
c$$$     +                      *sinn(i,l_)*xi(j)*dyi(i,l_)
c$$$                   endif
c$$$                   rhs(ieq)=cnst*dfdvz
c
cBH140101: RHS from BA coeffs for tor E field
c
!             if(ASSOCIATED(ampfda)) then
!                continue
!             else               ! Not allocated yet
!                allocate(ampfda(iy,0:jx),STAT=istat)
!                if (istat.ne.0) WRITE(*,*)'impanvc0: ampfda alloc prblm'
!                allocate(ampfdd(0:iy,jx),STAT=istat)
!                if (istat.ne.0) WRITE(*,*)'impanvc0: ampfdd alloc prblm'
!             endif
             call bcast(ampfda,zero,SIZE(ampfda))
             call bcast(ampfdd,zero,SIZE(ampfdd))

c   Set up the ampfda/ampfdd coeffs, and shift to bin bndries
c   (Following coefedad, and above.  See, also, 131231 notes.)  
c   (Could dimension by 1:lrz also, and set up once for all l_.)
             do j=1,jx
                do i=1,iy_(l_) !YuP[2021-03-04]
                   ![2021-03-04] for meshy="fixed_mu" iy_(l_) can be less than iy 
cYuP170227                   ampfda(i,j)=bnumb(k)/fmass(k)*cex(i,j,l_)
cYuP170227                   ampfdd(i,j)=bnumb(k)/fmass(k)*cet(i,j,l_)
cBH170228                   ampfda(i,j)=bnumb(k)/fmass(k)*cex(i,j,l_)*300.d0
cBH170228                   ampfdd(i,j)=bnumb(k)/fmass(k)*cet(i,j,l_)*300.d0
cYuP170228                   ampfda(i,j)=bnumb(k)/fmass(k)*cex(i,j,l_)
cYuP170228                   ampfdd(i,j)=bnumb(k)/fmass(k)*cet(i,j,l_)
                   ampfda(i,j)=bnumb(k)/fmass(k)*cex(i,j,l_)*300.d0
                   ampfdd(i,j)=bnumb(k)/fmass(k)*cet(i,j,l_)*300.d0
                   ! Note: cex and cet coeffs are 0 in the trapped cone.
                enddo
             enddo
             call coefmidv(ampfda,1)
             call coefmidt(ampfdd,1)
                   

             ipassbnd=iyh+1-itl ! here: ampfmod case
             if (symtrap.ne."enabled") ipassbnd=0  ! here: ampfmod case
             do j=1,jx
                ihalf=0
                icntsww=1
                if (iyh.eq.itl) icntsww=0
                if (symtrap .ne. "enabled") icntsww=itl-iyh
                do indexsww=1,inew
                   ieq=ieq+1
cyup                   if (j .eq. 1 .and. (lbdry(k).ne."conserv" 
cyup     +                  .and. lbdry(k).ne."consscal")) then
                   if(j.eq.1)then !YuP: for 'g' function, should be true for any lbdry
cBH170228                      rhs(ieq)=f_(iyh,1,k,l_) ! YuP: Why iyh?  BH: all same
                        !rhs_old=rhs(ieq)
                      rhs(ieq)=0.d0 !for 'g' function, it should be true for any lbdry
                        !write(*,*)'l_,j,i,rhs=',l_,j,indexsww,rhs_old
                   else ! YuP: j>1, any lbdry
                      if(indexsww.le.ipassbnd) then
                         i=iyh+1-indexsww
                      elseif(ihalf.eq.0) then
                         i = itl - icntsww
                         ihalf = 1
                      else
                         i = itu + icntsww
                         ihalf = 0
                         icntsww = icntsww + 1
                      endif

                      !For scal(neq,l_) :
                      neq=ieq+(k-1)*iyjx   ! for multiple species (k.gt.1)

                      alambda= vptb(i,lr_) ! ZOW or Hybrid-FOW
                      
                      !rhs_old=rhs(ieq)

                      fh0p=fh(i,j+1,1,l_)+fh(i,j,1,l_)
                      fh0m=fh(i,j,1,l_)+fh(i,j-1,1,l_)
                      fhp0=fh(i+1,j,1,l_)+fh(i,j,1,l_)
                      fhm0=fh(i,j,1,l_)+fh(i-1,j,1,l_)
cBH170407  Following had very small effect on a test (see a_change.h),
cBH170407  and have concern it could affect stability of the solution.
cBH170407                      !BH170406
cBH170407                      !Adding nonlinear (NL) term using previous iteration fg:
cBH170407                      if (it_ampf.gt.1) then
cBH170407                         fh0p=fh0p+delecfld0n(l_,n,it_ampf-1)*
cBH170407    1                        (fg(i,j+1,1,l_)+fg(i,j,1,l_))
cBH170407                         fh0m=fh0m+delecfld0n(l_,n,it_ampf-1)*
cBH170407     1                        (fg(i,j,1,l_)+fg(i,j-1,1,l_))
cBH170407                         fhp0=fhp0+delecfld0n(l_,n,it_ampf-1)*
cBH170407     1                        (fg(i+1,j,1,l_)+fg(i,j,1,l_))
cBH170407                         fhm0=fh0m+delecfld0n(l_,n,it_ampf-1)*
cBH170407     1                        (fg(i,j,1,l_)+fg(i-1,j,1,l_))
cBH170407                      endif

                      rhs(ieq)=half*(
     1                      (ampfda(i,j)*fh0p
     2                     -ampfda(i,j-1)*fh0m)
     +                     /cint2(j)
     3                     +(ampfdd(i,j)*fhp0
     4                     -ampfdd(i-1,j)*fhm0)
cYuP170227     +                     /(xsq(j)*sinn(i,l_)*dy(i,l_)))
     +                     /(xsq(j)*cynt2(i,l_)/twopi))
     
                      rhs(ieq)=rhs(ieq)*scal(neq,l_)
c                      write(*,'(a,3i4,4e12.3)')'l_,j,i,rhs=',
c     +                l_,j,i,rhs_old,rhs(ieq),ampfda(i,j),ampfdd(i,j)

                   endif        ! On if j.eq.1
                enddo           ! On indexsww
             enddo              ! On j
          endif                 ! On ampfmod/kopt

        else  ! factor.ne."disabled", i.e. FACTORIZE

c.......................................................................
c
c     reinitialize to zero matrix and rhs
c
c          write(*,*)'impavnc0:size(rhs) ',size(rhs)
          if (soln_method.eq.'direct' .or. soln_method.eq.'itsol1') then
              abd=zero !YuP[2019-04-24] was 0.
              do i=1,inewjx_(l_)
                rhs(i)=zero !YuP[2019-04-24] was 0.
                 ipivot(i)=0
              enddo
          elseif (soln_method.eq.'itsol') then
              do i=1,inewjx_(l_)
                rhs(i)=zero !YuP[2019-04-24] was 0.
              enddo
          elseif (ifirst_lr.eq.1) then  ! for it3dv and it3drv
              do i=1,ieq_tot
                rhs(i)=zero !YuP[2019-04-24] was 0.
              enddo
          endif

c..................................................................
c     The next set of variables are defined for use in code for
c     performing the second compression.  The value I is defined as
c     in the past and IEQ is incremented by unity each iteration.
c     However, due to the remapping of equation (node) number to
c     (i,j) pair of coordinates, the value of I "jumps" around.
c
c     As the index IEQ increases (for a given value of J), we start
c     at I=IYH (iy/2) and move clockwise to I=itl-1. Then we alternate
c     between the left and right sides, moving clockwise on the right
c     side and CCW on the left side.  As an example, for the case
c     of iy=16, itl=5, a sequence of inew=12 (12=iyh + itl-1) terms is
c     repeated for each value of J (n=12*(j-1))
c
c     ieq=n+1, n+2,  n+3,  n+4, n+5, n+6 , n+7 , n+8 , n+9 , n+10, n+11, n+12
c     i =iyh,iyh-1,itl+1,itl,itl-1,itu+1,itl-2,itu+2,itl-3,itu+3,itl-4,itu+4
c     (in this case, iyh-2=6=itl+1, itu=iy+1-itl=12)
c
c     ipassbnd is the number of pitch-angle-points from iyh to itl (i.e.,
c     number of points to trapped-passing boundary at itl, including iyh).
c
c     idistl - the distance, from point (i,j) to point (i+1,j)
c
c     idistr - the distance, from point (i,j) to point (i-1,j)
c
c     IHALF is used during the mapping of ieq (node number) to
c     values of I (mesh point I,J). IHALF is 0 if I<=iyh, otherwise
c     IHALF=1.
c     ICNTSWW - keeps track of the amount to add to ITL (IHALF=0)
c     or ITU (IHALF=1) to obtain current I value.
c
c     if (symtrap.ne."enabled") => no symmetry in trapping region =>
c     compute all the iy mesh points:
c     iyh,iyh+1,iyh-1,iyh+2,...,1,iy (=> assumes iyh=iy/2)
c     
c     nadjoint=1: Solve the adjoint equation. As f_adj=0 in trapped region,
c     solve in passing region only (i<itl, i>itu) and  at i=itu, setting
c     f_adj(itu)=0, and f(itl)=f(itu) as boundary cond.
c     The first point is i=itu, then itl-1,itu+1,... =>inew=2*itl-1
c..................................................................

cBH071029           ipassbnd=iyh+1-itl
cBH071029           if (symtrap.ne."enabled" .or. nadjoint.eq.1) ipassbnd=0
cBH070527          ieq=0
c          write(*,*)'impavnc0, before j and i loops: ieq,icoeff=',
c     +                                               ieq,icoeff

c%OS  
c          zzzto=0.0
c          zzz12=second()
c%OS  

c..................................................................
c     do loop 12 is the major loop over j (momentum).
c..................................................................

          zrhsj1=zero !YuP[2019-04-24] was 0.
cBH040719          call bcast(zavarj1,zero,mu+1)
cBH040719          call ibcast(ijaj1,0,mu+1)
          call bcast(zavarj1,zero,iymax+4)
          zmatcont=zero !YuP[2019-04-24] added
          janelt=0  !YuP[2019-07-12] added
          call ibcast(ijaj1,0,iymax+4)

          do 12 j=1,jx
          
            icntsww=1
            if (itl.eq.iyh) icntsww=0
            if (symtrap .ne. "enabled") icntsww=itl-iyh
            ihalf=0
            if (nadjoint.eq.1 .and. symtrap.eq."enabled") then
              icntsww = 0
              ihalf = 1
            endif
            idistl = -1
            idistr = 1
            if (symtrap .eq. "enabled") then
               idist_itl= 2
               idist_itu= 0 ! not used in this case         
            elseif (symtrap .eq. "disabled") then
               idist_itl= 3
               idist_itu= 1
            else ! "fow" (not happening here)
               idist_itl= itu-itl+1
               idist_itu=-idist_itl
            endif

c..................................................................
c     indexsww is the major loop over I (theta).
c..................................................................

            do 11 indexsww=1,inew
            
              if(indexsww.le.ipassbnd) then
                i=iyh+1-indexsww
              elseif(ihalf.eq.0) then
                i = itl - icntsww
                ihalf = 1
                idistr = 2
                idistl = -2
                if (symtrap .ne. "enabled") then
                  if (i .eq. iyh) idistl=+1
                else
                  if (i .eq. itl-1) idistl=-1
                endif
              else
                i = itu + icntsww
                ihalf = 0
                icntsww = icntsww + 1
                idistr = -2
                idistl = 2
                if(symtrap.ne."enabled" .and. i.eq.iyh+1) idistr=-1
              endif

c.................................................................
c     Separate out various boundary equations:
c     itl (itu) is the lower (upper) p/t boundary.
c     iyh is the mesh point index just below pi/2.
c     iy is the mesh point index of pi and
c     jx is the maximum velocity mesh point index.
c     For the current equation being considered (equation number
c     ieq) define the columns of the sparse matrix of order
c     iyjx where non-zero elements are to be found. This number
c     will vary from 1 to 12 elements in a given row depending
c     on whether the equation is an interior equation (9) or
c     a pass/trapped equation (12 in general) or another
c     boundary equation (1 or 2 or 4 or 6 or 8).
c     For lbdry(k).eq. ("conserv" or "consscal"),
c     the j=1 conservation equations give larger iy bandwidth,
c     linking j=1,i=inew to all j=2 distribution values (i=1,inew).
c..................................................................
              alambda= vptb(i,lr_) ! ZOW

              ieq=ieq+1
              ibandpieq=ibandto + ieq
              
              if(lbdry(k).eq."fixed" .or. 
     +           lbdry(k).eq."scale" .or. lbdry(k).eq."scale_n0")then
                ! YuP[07-24-2014] Put j=1 case BEFORE itl/itu (goto31)
                ! There is no trap-pass bndry for j=1
                if (j .eq. 1) go to 2
              endif
              
              if (i.eq.itu .and. cqlpmod.ne."enabled") go to 31
              if (i.eq.itu .and. nadjoint.eq.1.and.symtrap.eq."enabled")
     +          go to 31
              if (j .eq. jx) go to 1
              if (j .eq. 1) go to 2
              if (i .eq. 1) go to 3
              if (i .eq. iy_(l_)) go to 4 !YuP[2021-03] iy-->iy_(l_)
              ![2021-03-04] for meshy="fixed_mu" iy_(l_) can be less than iy 
              if (i.eq.itl .and. cqlpmod.ne."enabled") go to 5
              if (i.eq.iyh .and. symtrap.eq."enabled") goto 998

c.................................................................
c     Define the r.h.s.
c.................................................................

              rhs(ieq)=z00(i,j,k)

c.................................................................
c     Non-boundary region, nine point scheme.
c     Most of (i,j) points (internal)
c..................................................................

              janelt(1)=ieq-inew+idistr
              janelt(2)=ieq-inew
              janelt(3)=ieq-inew+idistl
              janelt(4)=ieq+idistr
              janelt(5)=ieq
              janelt(6)=ieq+idistl
              janelt(7)=ieq+inew+idistr
              janelt(8)=ieq+inew
              janelt(9)=ieq+inew+idistl
              nvar=9

c.................................................................
c     Define the 9 coefficients of the left hand side
c     for example xmm(i,j)*f(i-1,j-1,l_)+ x0m(i,j-1)*f(i,j-1,l_) +..
c     + seven other terms on left side = z00(i,j,k)
c     xmm, xpm, t0m, z00 etc are statement functions defined in
c     advnce.h
c
c     abd stores the transposed of the band matrix in the LAPACK sgbtrf format
c.................................................................

              zmatcont(1)=xmm(i,j)
              zmatcont(2)=x0m(i,j)
              zmatcont(3)=xpm(i,j)
              zmatcont(4)=xm0(i,j)
              zmatcont(5)=x00(i,j)
              zmatcont(6)=xp0(i,j)
              zmatcont(7)=xmp(i,j)
              zmatcont(8)=x0p(i,j)
              zmatcont(9)=xpp(i,j)                          

c     Normalize equation
              call impnorm(xnorm,zmatcont,rhs(ieq),nvar)

              go to 5000

 998          continue

c.................................................................
c     Added for the case i=iyh - using symmetry the "+1" points
c     are the same as the ith point - therefore, the coeffients are
c     added and one element removed.
c     (if symtrap=enabled)
c.................................................................

              rhs(ieq)=z00(i,j,k)

              janelt(1)=ieq-inew+idistr
              janelt(2)=ieq-inew
              janelt(3)=ieq+idistr
              janelt(4)=ieq
              janelt(5)=ieq+inew+idistr
              janelt(6)=ieq+inew
              nvar=6

              zmatcont(1)=xmm(i,j)
              zmatcont(2)=x0m(i,j)+xpm(i,j)
              zmatcont(3)=xm0(i,j)
              zmatcont(4)=x00(i,j)+xp0(i,j)
              zmatcont(5)=xmp(i,j)
              zmatcont(6)=x0p(i,j)+xpp(i,j)

              call impnorm(xnorm,zmatcont,rhs(ieq),nvar)

              go to 5000

c.................................................................
c     j=1 (means velocity=0 boundary condition)
c     Impose all points equal to f(iy_(l_),1,k,l_) and add equations
c     at all i onto eq. for i=iy_(l_).
c     First consider case where f is fixed at v=0 for
c     (lbdry.ne.("conserv" or "consscal")):
c     Do not fill in matrix, will be done after "5000 continue"
c..................................................................

 2            continue ! j=1 (v=0) point (totally iy_(l_), or "inew" points)
              if (lbdry(k).eq."fixed") then
                janelt(1)=ieq
                nvar=1
                   zmatcont(1)=one !YuP[2019-04-24] was 1.
                rhs(ieq)=f_(iyh,1,k,l_) ! f_ at previous t-step
                   xnorm=one !YuP[2019-04-24] was 1.
                go to 5000
              endif
              
              if(lbdry(k).eq."scale")then
                ! Set f(j=1) to mean of f_(j=2,i) (mean over i).
                ! Notice: variation of f() along pitch-angle
                ! can be factor of 2.
                ! This is consistent with assumption that the new f() 
                ! should be nearly Maxwellian at v~0, so that df/dv=0 at v=0.
                ! The overall magnitude of distribution will be set
                ! by rescaling to a target density (see diagscal)
                !-> First, find the mean value of f_(i) at j=2:
                !f_max=0.d0
                !f_min=1.d100
                !f_mean=0.d0
                !i_mean=0
                !do im=1,iy_(l_)
                !   f_max=  max(f_max,f_(im,2,k,l_))
                !   f_min=  min(f_min,f_(im,2,k,l_))
                !   f_mean= f_mean + f_(im,2,k,l_)
                !   i_mean= i_mean+1
                !enddo
                !f_mean=f_mean/float(i_mean)
                ! Add to the matrix: f(j=1)=rhs=f_mean(j=2) :
                janelt(1)=ieq
                nvar=1
                   zmatcont(1)=one !YuP[2019-04-24] was 1.
                   xnorm=one !YuP[2019-04-24] was 1.
                rhs(ieq)=f_(iyh,1,k,l_) !original (same as mean(f_(:,j=1,k,l_))
                !rhs(ieq)=f_(iyh,2,k,l_) ! j=2 at prev.step ! gives neg.dens.
                !rhs(ieq)= f_max !unstable
                !rhs(ieq)= f_min ! results in neg.dens.
                !rhs(ieq)= f_mean !unstable: jiggles in n(rho), enrgy(rho)

                ! Another version: impose df/dv =0 (for each i).
                ! Differencing corresponds to f(j=1,i)-f(j=2,i) =0.
!                rhs(ieq)=0.d0
!                janelt(1)=ieq      !corresponding to f(j=1) (for given i)
!                janelt(2)=ieq+inew !corresponding to f(j=2) (and same i)
!                nvar=2
!                zmatcont(1)=+0.5
!                zmatcont(2)=-0.5
!                call impnorm(xnorm,zmatcont,rhs(ieq),nvar)
                ! Tests: not stable in runs with lossmode+bootstrap.
                ! In runs with only lossmode (no bootstrap) - ok.
                
                go to 5000
              endif ! "scale"

c..................................................................
c "conserv"   Now the conservative boundary condition. i=1 first...
c.................................................................
              if (i .eq. 1) then
                rhs(ieq)=z00(i,j,k)

                janelt(1)=ieq
                janelt(2)=ieq+idistl
                janelt(3)=ieq+inew
                janelt(4)=ieq+inew+idistl
                nvar=4
c
                zmatcont(1)=x00(i,j)
                zmatcont(2)=xp0(i,j)
                zmatcont(3)=x0p(i,j)
                zmatcont(4)=xpp(i,j)

                call impnorm(xnorm,zmatcont,rhs(ieq),nvar)
c                if (ieq.le.840) write(*,*)'ieq,rhs(ieq)',ieq,rhs(ieq)
c                write(*,*)'HERE0.1: i,j,icntsww,xnorm',i,j,icntsww,xnorm
                go to 5000

c.................................................................
c     Now for the case j=1 and i=iy_(l_) (theta=pi).
c.................................................................

              elseif (i .eq. iy_(l_)) then !YuP[2021-03-04] iy-->iy_(l_)
                ![2021-03-04] for meshy="fixed_mu" iy_(l_) can be less than iy 
                rhs(ieq)=z00(i,j,k)

                janelt(1)=ieq+idistr
                janelt(2)=ieq
                janelt(3)=ieq+inew+idistr
                janelt(4)=ieq+inew
                nvar=4
c
                zmatcont(1)=xm0(i,j)
                zmatcont(2)=x00(i,j)
                zmatcont(3)=xmp(i,j)
                zmatcont(4)=x0p(i,j)

                call impnorm(xnorm,zmatcont,rhs(ieq),nvar)
c                if (ieq.le.840) write(*,*)'ieq,rhs(ieq)',ieq,rhs(ieq)
c                write(*,*)'HERE0.2:i,j,icntsww,xnorm',i,j,icntsww,xnorm
                go to 5000

c.................................................................
c     j=1 ,i=itl
c.................................................................

              elseif (i.eq.itl .and. cqlpmod.ne."enabled") then
                rhs(ieq)=z00(i,j,k)

                janelt(1)=ieq+idistr
                janelt(2)=ieq+idistr+inew
                janelt(3)=ieq
                janelt(4)=ieq+inew
                janelt(5)=ieq+idistl
                janelt(6)=ieq+idistl+inew
c     itu+1
                if (symtrap .eq. "enabled") then
                  janelt(7)=ieq+2
                  janelt(8)=ieq+inew+2
                else
                  janelt(7)=ieq+3
                  janelt(8)=ieq+inew+3
                endif

                nvar=8
c
                zmatcont(1)=tm0(j)
                zmatcont(2)=tmp(j)
                zmatcont(3)=t00(j)
                zmatcont(4)=t0p(j)
                zmatcont(5)=tp0(j)
                zmatcont(6)=tpp(j)
                zmatcont(7)=tu0(j)
                zmatcont(8)=tup(j)

                call impnorm(xnorm,zmatcont,rhs(ieq),nvar)
c                if (ieq.le.840) write(*,*)'ieq,rhs(ieq)',ieq,rhs(ieq)
c                write(*,*)'HERE0.3: i,j,icntsww,xnorm',i,j,icntsww,xnorm
                go to 5000

c.................................................................
c     j=1, i=iyh
c.................................................................

              elseif(i.eq.iyh .and. symtrap.eq."enabled") then
                rhs(ieq)=z00(i,j,k)

                janelt(1)=ieq+idistr
                janelt(2)=ieq
                janelt(3)=ieq+inew+idistr
                janelt(4)=ieq+inew
                nvar=4
c
                zmatcont(1)=xm0(i,j)
                zmatcont(2)=x00(i,j)+xp0(i,j)
                zmatcont(3)=xmp(i,j)
                zmatcont(4)=x0p(i,j)+xpp(i,j)

                call impnorm(xnorm,zmatcont,rhs(ieq),nvar)
c                if (ieq.le.840) write(*,*)'ieq,rhs(ieq)',ieq,rhs(ieq)
c                write(*,*)'HERE0.4: i,j,icntsww,xnorm',i,j,icntsww,xnorm
                go to 5000

c.................................................................
c     case j=1, i not equal 1 or iy_(l_) or ...
c.................................................................

              else
                rhs(ieq)=z00(i,j,k)

                janelt(1)=ieq+idistr
                janelt(2)=ieq
                janelt(3)=ieq+idistl
                janelt(4)=ieq+inew+idistr
                janelt(5)=ieq+inew
                janelt(6)=ieq+inew+idistl
                nvar=6
c
                zmatcont(1)=xm0(i,j)
                zmatcont(2)=x00(i,j)
                zmatcont(3)=xp0(i,j)
                zmatcont(4)=xmp(i,j)
                zmatcont(5)=x0p(i,j)
                zmatcont(6)=xpp(i,j)
                call impnorm(xnorm,zmatcont,rhs(ieq),nvar)
c                if (ieq.le.840) write(*,*)'ieq,rhs(ieq)',ieq,rhs(ieq)
c                write(*,*)'HERE0.5: i,j,icntsww,xnorm',i,j,icntsww,xnorm
                go to 5000
              endif

c.................................................................
c     i=1 case (theta=0) : j=2,3,4.....,jx-1
c.................................................................

 3            continue
              rhs(ieq)=z00(i,j,k)

              janelt(1)=ieq-inew
              janelt(2)=ieq-inew+idistl
              janelt(3)=ieq
              janelt(4)=ieq+idistl
              janelt(5)=ieq+inew
              janelt(6)=ieq+inew+idistl
              nvar=6
c
              zmatcont(1)=x0m(i,j)
              zmatcont(2)=xpm(i,j)
              zmatcont(3)=x00(i,j)
              zmatcont(4)=xp0(i,j)
              zmatcont(5)=x0p(i,j)
              zmatcont(6)=xpp(i,j)

              call impnorm(xnorm,zmatcont,rhs(ieq),nvar)

              go to 5000
 4            continue

c.................................................................
c     i=iy case (theta=pi) : j=2,3,4,......,jx-1
c.................................................................

              rhs(ieq)=z00(i,j,k)

              janelt(1)=ieq-inew+idistr
              janelt(2)=ieq-inew
              janelt(3)=ieq+idistr
              janelt(4)=ieq
              janelt(5)=ieq+inew+idistr
              janelt(6)=ieq+inew
              nvar=6
c
              zmatcont(1)=xmm(i,j)
              zmatcont(2)=x0m(i,j)
              zmatcont(3)=xm0(i,j)
              zmatcont(4)=x00(i,j)
              zmatcont(5)=xmp(i,j)
              zmatcont(6)=x0p(i,j)

              call impnorm(xnorm,zmatcont,rhs(ieq),nvar)

              go to 5000

 5            continue

c.................................................................
c     Pass-trapped boundary (lower): all velocities except v=0.
c     (if symtrap=enabled)
c.................................................................

              rhs(ieq)=z00(i,j,k)

              janelt(1)=ieq-inew+idistr
              janelt(2)=ieq+idistr
              janelt(3)=ieq+inew+idistr
              janelt(4)=ieq-inew
              janelt(5)=ieq
              janelt(6)=ieq+inew
              janelt(7)=ieq-inew+idistl
              janelt(8)=ieq+idistl
              janelt(9)=ieq+inew+idistl
c     itu+1
              if (symtrap .eq. "enabled") then
                janelt(10)=ieq+2-inew
                janelt(11)=ieq+2
                janelt(12)=ieq+inew+2
              else
                janelt(10)=ieq+3-inew
                janelt(11)=ieq+3
                janelt(12)=ieq+inew+3
              endif
              nvar=12
c
              zmatcont(1)=tmm(j)
              zmatcont(2)=tm0(j)
              zmatcont(3)=tmp(j)
              zmatcont(4)=t0m(j)
              zmatcont(5)=t00(j)
              zmatcont(6)=t0p(j)
              zmatcont(7)=tpm(j)
              zmatcont(8)=tp0(j)
              zmatcont(9)=tpp(j)
              zmatcont(10)=tum(j)
              zmatcont(11)=tu0(j)
              zmatcont(12)=tup(j)

              call impnorm(xnorm,zmatcont,rhs(ieq),nvar)

              go to 5000

 1            continue

c.................................................................
c     Upper boundary j=jx...
c     if pass/trapped boundary (and j=jx) then go to 7
c.................................................................

              if (i.eq.itl .and. cqlpmod.ne."enabled") go to 7

c.................................................................
c     if theta=pi go to 14; if theta=0 go to 13
c.................................................................

              if (i .eq. iy_(l_)) go to 14 !YuP[2021-03-04] iy-->iy_(l_)
              ![2021-03-04] for meshy="fixed_mu" iy_(l_) can be less than iy 
              if (i .eq. 1) go to 13
              if(i.eq.iyh .and. symtrap.eq."enabled") goto 991

c.................................................................
c     inner theta points at v=vmax (hyperbolic)
c.................................................................

              rhs(ieq)=z00(i,j,k)

              janelt(1)=ieq-inew+idistr
              janelt(2)=ieq+idistr
              janelt(3)=ieq-inew
              janelt(4)=ieq
              janelt(5)=ieq+idistl-inew
              janelt(6)=ieq+idistl
              nvar=6
c
              zmatcont(1)=xmm(i,j)
              zmatcont(2)=xm0(i,j)
              zmatcont(3)=x0m(i,j)
              zmatcont(4)=x00(i,j)
              zmatcont(5)=xpm(i,j)
              zmatcont(6)=xp0(i,j)

              call impnorm(xnorm,zmatcont,rhs(ieq),nvar)

              go to 5000

 991          continue

c.......................................................................
c     i=iyh and symtrap=enabled
c.......................................................................
              rhs(ieq)=z00(i,j,k)

              janelt(1)=ieq-inew+idistr
              janelt(2)=ieq+idistr
              janelt(3)=ieq-inew
              janelt(4)=ieq
              nvar=4
c
              zmatcont(1)=xmm(i,j)
              zmatcont(2)=xm0(i,j)
              zmatcont(3)=x0m(i,j)+xpm(i,j)
              zmatcont(4)=x00(i,j)+xp0(i,j)

              call impnorm(xnorm,zmatcont,rhs(ieq),nvar)

              go to 5000

 7            continue

c.................................................................
c     v=vmax ; theta=pass/trapped bndry
c     (if symtrap=enabled)
c.................................................................

              rhs(ieq)=z00(i,j,k)

              janelt(1)=ieq-inew+idistr
              janelt(2)=ieq+idistr
              janelt(3)=ieq-inew
              janelt(4)=ieq
              janelt(5)=ieq+idistl-inew
              janelt(6)=ieq+idistl
c     itu+1
              if (symtrap .eq. "enabled") then
                janelt(7)=ieq+2-inew
                janelt(8)=ieq+2
              else
                janelt(7)=ieq+3-inew
                janelt(8)=ieq+3
              endif

              nvar=8
c
              zmatcont(1)=tmm(jx)
              zmatcont(2)=tm0(jx)
              zmatcont(3)=t0m(jx)
              zmatcont(4)=t00(jx)
              zmatcont(5)=tpm(jx)
              zmatcont(6)=tp0(jx)
              zmatcont(7)=tum(jx)
              zmatcont(8)=tu0(jx)

              call impnorm(xnorm,zmatcont,rhs(ieq),nvar)

              go to 5000

 14           continue

c..................................................................
c     v=vmax , theta=pi
c.................................................................

              rhs(ieq)=z00(i,j,k)

              janelt(1)=ieq
              janelt(2)=ieq+idistr
              janelt(3)=ieq-inew
              janelt(4)=ieq-inew+idistr
              nvar=4
c
              zmatcont(1)=x00(i,j)
              zmatcont(2)=xm0(i,j)
              zmatcont(3)=x0m(i,j)
              zmatcont(4)=xmm(i,j)

              call impnorm(xnorm,zmatcont,rhs(ieq),nvar)

              go to 5000
 13           continue

c.................................................................
c     v=vmax, theta=0
c.................................................................

              rhs(ieq)=z00(i,j,k)

              janelt(1)=ieq
              janelt(2)=ieq+idistl
              janelt(3)=ieq+idistl-inew
              janelt(4)=ieq-inew

              nvar=4
c
              zmatcont(1)=x00(i,j)
              zmatcont(2)=xp0(i,j)
              zmatcont(3)=xpm(i,j)
              zmatcont(4)=x0m(i,j)

              call impnorm(xnorm,zmatcont,rhs(ieq),nvar)

              go to 5000

 31           continue

c.................................................................
c     i=itu
c     if symtrap.ne.enabled .or. nadjoint.eq.1
c.................................................................

              rhs(ieq)=z00(i,j,k)

              if (nadjoint.eq.1 .and. symtrap.eq."enabled") then
c     set f(itu)=0
                janelt(1)=ieq
                nvar=1
                   zmatcont(1)=one !YuP[2019-04-24] was 1.0
                   rhs(ieq)=zero !YuP[2019-04-24] was 0.

              else if (j .eq. 1) then

                janelt(1)=ieq+idistr
                janelt(2)=ieq+inew+idistr
                janelt(3)=ieq
                janelt(4)=ieq+inew
                janelt(5)=ieq+idistl
                janelt(6)=ieq+inew+idistl
                janelt(7)=ieq+1
                janelt(8)=ieq+inew+1
                nvar=8
c
                zmatcont(1)=tp0(j)
                zmatcont(2)=tpp(j)
                zmatcont(3)=t00(j)
                zmatcont(4)=t0p(j)
                zmatcont(5)=tu0(j)
                zmatcont(6)=tup(j)
                zmatcont(7)=tm0(j)
                zmatcont(8)=tmp(j)
              else if (j .eq. jx) then

                janelt(1)=ieq-inew+idistr
                janelt(2)=ieq+idistr
                janelt(3)=ieq-inew
                janelt(4)=ieq
                janelt(5)=ieq-inew+idistl
                janelt(6)=ieq+idistl
                janelt(7)=ieq+1-inew
                janelt(8)=ieq+1
                nvar=8
c
                zmatcont(1)=tpm(jx)
                zmatcont(2)=tp0(jx)
                zmatcont(3)=t0m(jx)
                zmatcont(4)=t00(jx)
                zmatcont(5)=tum(jx)
                zmatcont(6)=tu0(jx)
                zmatcont(7)=tmm(jx)
                zmatcont(8)=tm0(jx)

              else

                janelt(1)=ieq-inew+idistr
                janelt(2)=ieq+idistr
                janelt(3)=ieq+inew+idistr
                janelt(4)=ieq-inew
                janelt(5)=ieq
                janelt(6)=ieq+inew
                janelt(7)=ieq-inew+idistl
                janelt(8)=ieq+idistl
                janelt(9)=ieq+inew+idistl
                janelt(10)=ieq+1-inew
                janelt(11)=ieq+1
                janelt(12)=ieq+inew+1
                nvar=12
c
                zmatcont(1)=tpm(j)
                zmatcont(2)=tp0(j)
                zmatcont(3)=tpp(j)
                zmatcont(4)=t0m(j)
                zmatcont(5)=t00(j)
                zmatcont(6)=t0p(j)
                zmatcont(7)=tum(j)
                zmatcont(8)=tu0(j)
                zmatcont(9)=tup(j)
                zmatcont(10)=tmm(j)
                zmatcont(11)=tm0(j)
                zmatcont(12)=tmp(j)

              endif

              call impnorm(xnorm,zmatcont,rhs(ieq),nvar)

              go to 5000

c..................................................................
c     After an equation has been set, code jumps to 5000
c..................................................................

 5000         continue

c.......................................................................
c     SOLN_METHOD='direct'/'itsol1': put normalized contribs into matrix
c.......................................................................

            if (soln_method.eq.'direct'.or.soln_method.eq.'itsol1') then

cBH050804 bugfix, adding following if clause:
              if(lbdry(k).ne."conserv" .and. lbdry(k).ne."consscal")then

                do jcont=1,nvar
!              if(l_.eq.8)then
              !if(j.le.2)then
!               write(*,*)'1. j,i,jcont,janelt(jcont)=',
!     &                       j,i,jcont,janelt(jcont)
              !elseif(j.eq.3)then
              ! stop
              !endif
!              endif
                  abd(ibandpieq-janelt(jcont),janelt(jcont))=
     +              zmatcont(jcont)
                end do

              else  !i.e.,lbdry(k).eq. ("conserv" or "consscal")


              if (j .ne. 1) then

                do jcont=1,nvar
                  !write(*,*)'2.  janelt(jcont)=',janelt(jcont)
                  abd(ibandpieq-janelt(jcont),janelt(jcont))=
     +              zmatcont(jcont)
                end do

cBH050804 bugfix:              else
cBH070419:             elseif (lbdry(k).eq."conserv") then
cBH070419: Wonder what I was thinking?

              else  !i.e., j=1

c.......................................................................
c     for j=1 impose f(i)=cst condition
c.......................................................................

                do 5100 icon=1,nvar
                  if (janelt(icon) .le. inew) then
c     (i,j=1) contributions -> (iy,1) point
                    zavarj1(1)=zavarj1(1)+zmatcont(icon)*xnorm !here direct
                  else
c     (i,j=2) contributions -> (i,2) point
                    zavarj1(janelt(icon)-inew+1)=
     +                zavarj1(janelt(icon)-inew+1)+zmatcont(icon)*xnorm
                    ijaj1(janelt(icon)-inew+1)=janelt(icon)
                  endif
 5100           continue
                zrhsj1=zrhsj1+rhs(ieq)*xnorm

c     modify matrix and right-hand-side
c     assumes point iy is the last of the j=1 sequence
c
                if (ieq .lt. inew) then
c     i<iy,j=1: f(i)-f(iy)=0.
                  nvar=2
                  zmatcont(1)=0.5
                  zmatcont(2)=-0.5
                  janelt(1)=ieq
                  janelt(2)=inew
                            rhs(ieq)=zero !YuP[2019-04-24] was 0.
                            xnorm=2.d0 !YuP[2019-04-24] was 2.0
                  do jcont=1,nvar
                  !write(*,*)'3.  janelt(jcont)=',janelt(jcont)
                    abd(ibandpieq-janelt(jcont),janelt(jcont))=
     +                zmatcont(jcont)
                  end do
                else
c     i=iy
c   Replace equation by sum of equations for i=1,iy, 
c   using f(i)=f(iy), i=1,iy-1.
c   Thus, number of contributions should be at most mu to the  
c   right of (i=iy,j=1) point => inew+1.
c   Bug fix, Olivier Sauter, 030509: right of (i=iy,j=1) point => mu+1.
                  ijaj1(1)=ieq
c   Bug fix, Olivier Sauter, 030509:     nvar=mu + 1
                  nvar=inew + 1
                  rhs(ieq)=zrhsj1
                  call impnorm(xnorm,zavarj1,rhs(ieq),nvar)
                  do jcont=1,nvar
                  !write(*,*)'4.  ijaj1(jcont)=',ijaj1(jcont)
                 abd(ibandpieq-ijaj1(jcont),ijaj1(jcont))=zavarj1(jcont) !here direct
                  end do
                endif  ! on ieq.lt.new/else

              endif  ! on j.ne.1/else
              
              endif  ! on lbdry(k)

              endif  ! on soln_method.eq.'direct'
                     !   .or.soln_method.eq.'itsol1'

c.......................................................................

c.......................................................................
c      SOLN_METHOD='itsol',  put normalized contribs into matrix
c.......................................................................

              if (soln_method.eq.'itsol') then

              ia_csr(ieq)=icoeff+1   ! Coeff counter at beginning of row

cBH050804 bugfix, adding following if clause:
              if(lbdry(k).ne."conserv" .and. lbdry(k).ne."consscal")then

                do jcont=1,nvar
                   icoeff=icoeff+1
                   a_csr(icoeff)=zmatcont(jcont)
                   ja_csr(icoeff)=janelt(jcont)
                end do

              else   !i.e., lbdry(k).eq.('conserv' or 'consscal')


              if (j .ne. 1) then

                do jcont=1,nvar
                   icoeff=icoeff+1
                   a_csr(icoeff)=zmatcont(jcont)
                   ja_csr(icoeff)=janelt(jcont)
                end do

cBH050804 bugfix:              else
cBH070419:             elseif (lbdry(k).eq."conserv") then
cBH070419: Wonder what I was thinking?

              else   !  i.e., j=1
c.......................................................................
c     for j=1 impose f(i)=cst condition
c.......................................................................

                do 5200 icon=1,nvar
                  if (janelt(icon) .le. inew) then
c     (i,j=1) contributions -> (iy,1) point
                    zavarj1(1)=zavarj1(1)+zmatcont(icon)*xnorm !here itsol
                  else
c     (i,j=2) contributions -> (i,2) point
                    zavarj1(janelt(icon)-inew+1)=
     +                zavarj1(janelt(icon)-inew+1)+zmatcont(icon)*xnorm !here itsol
                    ijaj1(janelt(icon)-inew+1)=janelt(icon)
                  endif
 5200           continue
                zrhsj1=zrhsj1+rhs(ieq)*xnorm

c     modify matrix and right-hand-side
c     assumes point iy is the last of the j=1 sequence
c
                if (ieq .lt. inew) then
c     i<iy,j=1: f(i)-f(iy)=0.
                  nvar=2
                  zmatcont(1)=0.5
                  zmatcont(2)=-0.5
                  janelt(1)=ieq
                  janelt(2)=inew
                            rhs(ieq)=zero !YuP[2019-04-24] was 0.
                            xnorm=2.d0 !YuP[2019-04-24] was 2.0
                  do jcont=1,nvar
                    icoeff=icoeff+1
                    a_csr(icoeff)=zmatcont(jcont)
                    ja_csr(icoeff)=janelt(jcont)
                  end do
                else
c     i=iy
c   Replace equation by sum of equations for i=1,iy, 
c   using f(i)=f(iy), i=1,iy-1.
c   Thus, number of contributions should be at most mu to the  
c   right of (i=iy,j=1) point => inew+1.
c   Bug fix, Olivier Sauter, 030509: right of (i=iy,j=1) point => mu+1.
                  ijaj1(1)=ieq
c   Bug fix, Olivier Sauter, 030509:     nvar=mu + 1
                  nvar=inew + 1
                  rhs(ieq)=zrhsj1
                  call impnorm(xnorm,zavarj1,rhs(ieq),nvar) !here itsol
                  do jcont=1,nvar
                    icoeff=icoeff+1
                    a_csr(icoeff)=zavarj1(jcont) !here itsol
                    ja_csr(icoeff)=ijaj1(jcont)
                  end do
                endif  ! on ieq

              endif  ! on j
              
              endif  ! on lbdry(k)

              endif  ! on soln_method.eq.'itsol'

c.......................................................................
c      SOLN_METHOD='it3dv' or 'it3drv', put normed contribs into matrix
c.......................................................................

              if ( soln_method.eq.'it3dv' .or. 
     +             soln_method.eq.'it3drv') then

              ia_csr(ieq)=icoeff+1   ! Coeff counter at beginning of row

cBH050804 bugfix, adding following if clause:
              if(lbdry(k).ne."conserv" .and. lbdry(k).ne."consscal")then

                do jcont=1,nvar
                   icoeff=icoeff+1
                   a_csr(icoeff)=zmatcont(jcont)
                   ja_csr(icoeff)=janelt(jcont)
                end do
                
              else   !i.e., lbdry(k).eq.("conserv".or."consscal")


              if (j .ne. 1) then

                do jcont=1,nvar
                   icoeff=icoeff+1
                   a_csr(icoeff)=zmatcont(jcont)
                   ja_csr(icoeff)=janelt(jcont)
                end do

cBH050804 bugfix:              else
cBH070419:             elseif (lbdry(k).eq."conserv") then
cBH070419: Wonder what I was thinking?

              else   !  i.e., j=1
c.......................................................................
c     for j=1 impose f(i)=cst condition
c.......................................................................

                do 5300 icon=1,nvar
                  if ((janelt(icon)-(ieq_(l_)-1)) .le. inew) then
c     (i,j=1) contributions -> (iy,1) point
                    zavarj1(1)=zavarj1(1)+zmatcont(icon)*xnorm !here it3drv
                  else
c     (i,j=2) contributions -> (i,2) point
                    zavarj1((janelt(icon)-(ieq_(l_)-1))-inew+1)=
     +                zavarj1((janelt(icon)-(ieq_(l_)-1))-inew+1)+
     +                zmatcont(icon)*xnorm
                    ijaj1((janelt(icon)-(ieq_(l_)-1))-inew+1)=
     +                janelt(icon)-(ieq_(l_)-1)
                  endif
 5300           continue
                zrhsj1=zrhsj1+rhs(ieq)*xnorm

c     modify matrix and right-hand-side
c     assumes point iy is the last of the j=1 sequence
c
                if ((ieq-(ieq_(l_)-1)).lt. inew) then
c     i<iy,j=1: f(i)-f(iy)=0.
                  nvar=2
                  zmatcont(1)=0.5
                  zmatcont(2)=-0.5
                  janelt(1)=ieq
                  janelt(2)=inew+(ieq_(l_)-1)
                            rhs(ieq)=zero !YuP[2019-04-24] was 0.
                            xnorm=2.d0 !YuP[2019-04-24] was 2.0
                  do jcont=1,nvar
                    icoeff=icoeff+1
                    a_csr(icoeff)=zmatcont(jcont)
                    ja_csr(icoeff)=janelt(jcont)
                  end do
                else
c     i=iy
c   Replace equation by sum of equations for i=1,iy, 
c   using f(i)=f(iy), i=1,iy-1.
c   Thus, number of contributions should be at most mu to the  
c   right of (i=iy,j=1) point => inew+1.
c   Bug fix, Olivier Sauter, 030509: right of (i=iy,j=1) point => mu+1.
                  ijaj1(1)=ieq - (ieq_(l_)-1)
c   Bug fix, Olivier Sauter, 030509:     nvar=mu + 1
                  nvar=inew + 1
                  rhs(ieq)=zrhsj1
                  call impnorm(xnorm,zavarj1,rhs(ieq),nvar)
                  do jcont=1,nvar
                    icoeff=icoeff+1
                    a_csr(icoeff)=zavarj1(jcont) !here it3drv
                    ja_csr(icoeff)=ijaj1(jcont) + (ieq_(l_)-1)
                  end do
                endif  ! on ieq

              endif  ! on j
              
              endif  ! on lbdry(k)

              endif  ! on soln_method.eq.'it3dv' or 'it3drv'

c.......................................................................
c     Save scaling for reuse by no-factorize solution or radial transp
c.......................................................................
              if (soln_method.eq."itsol" .or. soln_method.eq."itsol1"
     +           .or. soln_method.eq."direct") then
     
                 kku=ieq+(k-1)*iyjx  !-YuP-> version from impavnc
              else ! for soln_method it3dv or it3drv
                 kku=ieq-(ieq_(l_)-1)+(k-1)*iyjx 
              endif
              scal(kku,l_)=one/xnorm !!YuP[2019-04-24] was 1./xnorm
c              if(kku.le.0 .or. kku.gt.iyjx*ngen) then
c                write(*,*)'impavnc0: k,l_,ieq,ieq_(l_),kku,ieq_tot=',
c     ~                          k,l_,ieq,ieq_(l_),kku,ieq_tot
c                pause
c              endif

c..................................................................
c     End of major j and I loops..
c..................................................................
c              write(*,*)'ij-loop-end: ieq,rhs(ieq) ',ieq,rhs(ieq)
 11         continue
 12       continue
          
c..................................................................
c     Final element for CSR storage
c..................................................................

          if ( soln_method.ne.'direct' ) then
          ia_csr(ieq+1)=icoeff+1  !CSR storage includes coeff count+1
                                  !recorded at last eqn+1.

c         Estimating number of coeffs per flux surface
c         to compare with actual number:
c          write(*,*)'impanvc0: Number of CSR coeffs: icoeff=',icoeff
c         icoeff_est=(iy+2)*3+(jx-1)*9*((iyh-itl)+2*itl)  Calc'd above
c          write(*,*)'impavnc0: Estimate of of icoeff=', icoeff_est
c         Check coeff storage:
          if (icoeff.gt.size(a_csr)) then
             WRITE(*,*)'impavnc0:icoeff.gt.size(a_csr)'
             STOP
          endif
          endif  ! on soln_method.ne.'direct'

c     Check number of equations:
c        if (soln_method.eq.'itsol' .or. soln_method.eq.'itsol1') then
c           write(*,*)'impavnc0: ieq should equal inewjx: ',ieq,inewjx
c        elseif(soln_method.eq.'it3dv' .or. soln_method.eq.'it3drv') then
c           write(*,*)'impavnc0: ieq should equal ieq_tot: ',ieq,ieq_tot
c           pause
c        endif

        
      if (soln_method.eq.'itsol') then
      iunit=39
      i1=1
      i2=icoeff
cyup      if (n.eq.1 .and. ifirst_lr.eq.1) open(iunit)
cyup      write(iunit,*)'a_csr(i1:i2), l_,i1,i2 ',l_,i1,i2
cyup      write(iunit,110) a_csr(i1:i2)
cyup      write(iunit,*)'ja_csr(i1:i2), l_,i1,i2 ',l_,i1,i2
cyup      write(iunit,111) ja_csr(i1:i2)
cyup      write(iunit,*)'ia_csr(1:ieq), l_,ieq ',l_,ieq
cyup      write(iunit,111) ia_csr(1:ieq)
cyup      write(iunit,*)'rhs(1:ieq), l_,ieq ',l_,ieq 
cyup      write(iunit,110) rhs(1:ieq)
cyup      if (n.eq.1. and. ilast_lr.eq.1) close(iunit)
      endif  ! on solution_meth=itsol

c..................................................................
c     Next endif is the end for "if(factor.eq."disabled") then".
c     Thus matrix and rhs are defined
c..................................................................

        endif     ! on factor

c        write(*,*)'impavnc0:  factor =',factor



        if (soln_method.eq.'itsol' .or. soln_method.eq.'itsol1') then
           n_rows_A=inewjx
        elseif (soln_method.eq.'it3dv'.or. soln_method.eq.'it3drv') then
           n_rows_A=ieq_tot
        endif


c..................................................................
c     SOLN_METHOD = 'it3drv':
c     Calculate the CSR matrix of coeffs
c     Add the coefficients into the above velocity space coeffs
c..................................................................

        if (soln_method.eq.'it3drv' .and. ilast_lr.eq.1) then
        call tdtranspn

c Use SPARSKIT (in BLASSM/blassm.f) CSR addition routine aplb,
c      subroutine aplb (nrow,ncol,job,a,ja,ia,b,jb,ib,
c     *     c,jc,ic,nzmax,iw,ierr)
c Computes matrix sum:    C = A+B, in CSR format
c nzmax	= integer. The  length of the arrays c and jc.
c         aplb will stop if the result matrix C  has a number 
c         of elements that exceeds nzmax. See ierr.
c ierr	= integer. serving as error message. 
c         ierr = 0 means normal return,
c         ierr .gt. 0 means that aplb stopped while computing the
c         i-th row  of C with i=ierr, because the number 
c         of elements in C exceeds nzmax.
c work arrays:
c iw	= integer work array of length equal to the number of
c         columns in A.

      ieqp=54
      i1=1
      i2=iar_csr(ieqp)

        call aplb(ieq_tot,ieq_tot,1,a_csr,ja_csr,ia_csr,
     +            ar_csr,jar_csr,iar_csr,ac_csr,jac_csr,iac_csr,
     +            icsrijc,jw_ilu,ierr)

CMPIINSERT_IF_RANK_EQ_0
       WRITE(*,*)'impavnc0 aft.aplb: ieq_tot, iyjx*lrz, ierr',
     +             ieq_tot,iyjx*lrz,ierr
CMPIINSERT_ENDIF_RANK

      ieqp=54
      i1=1
      i2=iac_csr(ieqp)

cBH070419:   Add for iterative sparse matrix solve, down to
cBH070419:   Call direct solve Gaussian..... 
c..................................................................
c     This is for soln_method='itsol1'.
c     [This method uses abd_lapack which is ml+mu+1 wide, and
c     uses much more storage than necessary.   The preferred
c     method is soln_method='itsol' which directly populates 
c     the CSR coeff matrix.]
c
c     Put matrix into SPARSKIT format, i.e., compressed sparse row
c     storage, CSR.  This is done with SPARSKIT2 routine bndscr,
c     which converts from LAPACK storage format.
c     In the present case, their are 1+ml leading rows in abd(,)
c     which have no input values.   LAPACK storage format
c     does not include these dummy rows.
c     Create Harwell/Boeing formatted matrix for plotting, and
c     call SPARSKIT2 routine which plots outline of matrix entries.
c..................................................................

cNOTE:  There are a lot of 0.0d0 entries in abd_lapack
cNOTE:  which we can eliminate.   Best would be to set up
cNOTE:  a_csr,ja_csr,ia_csr directly from the above
cNOTE:  coefficient calculation.   Then would have a much
cNOTE:  smaller sparse coeff set, and consequently much
cNOTE:  faster execution.
c       [Now done with soln_method='itsol']

c       i_orig,j_orig refer to original A(,) coeff array
c       before adjustment to lapack order.

        endif  ! on soln_method.eq.'it3drv' and ilast_lr


        if (soln_method.eq.'itsol1') then

        icount=0
        do j=1,inewjx
           do i=1,ml+mu+1
           abd_lapack(i,j)=abd(ml+i,j)
           enddo
        enddo

        nabd=ml+mu+1
        lowd=ml+mu+1
        ml_a=ml
        mu_a=mu
        call bndcsr(n_rows_A,abd_lapack,nabd,lowd,ml_a,mu_a,
     1              a_csr,ja_csr,ia_csr,icsrij,ierr_csr)
        if (ierr_csr.ne.0) then
           WRITE(*,*)'impavnc0/bndcsr: STOP ierr_csr=',ierr_csr
           stop
        endif

        endif  !  on soln_method.eq.'itsol1'


      if (soln_method.eq.'itsol' .or. soln_method.eq.'itsol1') then 
 110  format(10(1pe10.2))
 111  format(10i10)
      endif  ! on soln_method itsol or itsol1


      if ( (soln_method.eq.'it3dv')  .and. 
     +     ilast_lr.eq.1 .and. n.eq.1  ) then
c        write out coeffs in a format which can be compared directly 
c        with soln_method=itsol output.
         iunit=39
         i1=0  !beginning of coeff print for given surface
         i2=0  !end of coeff print for given surface
         l1=1  !eqn number at beginning of given flux surface
         l2=1  !eqn number at beginning of next flux surface
         do ll=1,lrz
            i1=i2+1
            l1=l2
            if (ll.lt.lrz) then
               l2=ieq_(ll+1)
               i2=ia_csr(l2)-1
            else
               l2=ieq_tot+1
               i2=ia_csr(l2)-1
            endif
         enddo
      endif  ! on soln_method .eq. it3dv

      if (soln_method.eq.'itsol' .or. soln_method.eq.'itsol1' .or.
     +    soln_method.eq.'it3dv'.or.soln_method.eq.'it3drv') then 

c     Call ilut preconditioner, following SPARSKIT:rilut.f test
c      subroutine ilut(n,a,ja,ia,lfil,droptol,alu,jlu,ju,iwk,w,jw,ierr)

        ierr=0
        lfil0=min(lfil,n_rows_A)

c-YuP        write (*,*) ' ++ ILUT Preconditioner ++++ '
c-YuP        write (*,*) ' ++ droptol =, lfil.le.n_rows_A  ++ ',droptol,lfil0

        call cpu_time(tm1)
cBH070523:  Change array input to be first element,
cBH070523:  per compiler results for lf95 by Ershov:       
c        call ilut (n_rows_A,a_csr,ja_csr,ia_csr,lfil0,droptol,
c     1       alu,jlu,ju,iwk_ilu,w_ilu,jw_ilu,ierr)
c On return:
c alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing
c           the L and U factors together.
c ju      = integer array of length n containing the pointers to
c           the beginning of each row of U in the matrix alu,jlu.
c ierr    = integer. Error messages.  0 is succesful return.
c
c
c  For soln_method .eq. 'it3dv' or 'it3drv', only call at end
c  of coeff setup over ll=1,lrz flux surfaces.
        ! Additional arguments used in ilutp:
        permtol= 0.5d0 ! 0.d0 -> means never permute
        mbloc=   n_rows_A
        !iperm_ilu(2*n) is output.
        ! If permtol=0. and mbloc=n_rows_A, 
        ! the result from ilutp() is identical to result from ilut()
        ! 

        if ( soln_method.eq."itsol" .or. soln_method.eq."itsol1" ) then
           call ilut (n_rows_A,a_csr,ja_csr,ia_csr,lfil0,
     +          droptol,alu,jlu,ju,iwk_ilu,w_ilu,
     +          jw_ilu,ierr)
                             !Setup up above, since vv size dependency
           do i=1,n_rows_A
              rhs0(i)=rhs(i)    !Copy rhs, for input to pgmres since
                                !pgmres modifies this input.
           enddo
        elseif(soln_method.eq.'it3dv') then
           ! perform soln only at last flux surface (l_=lrz)
           if ( ilast_lr.eq.1 ) then
              call ilut (n_rows_A,a_csr,ja_csr,ia_csr,lfil0,
     +             droptol,alu,jlu,ju,iwk_ilu,w_ilu,
     +             jw_ilu,ierr)
c              call ilutp(n_rows_A,a_csr(1),ja_csr(1),ia_csr(1),
c     +             lfil0,droptol,permtol,mbloc,
c     +             alu(1),jlu(1),ju(1),iwk_ilu,w_ilu(1),jw_ilu(1),
c     +             iperm_ilu(1),ierr)
CMPIINSERT_IF_RANK_EQ_0
c       WRITE(*,*)'impavnc0 aft.ilut:  l_,ierr',l_,ierr,soln_method
CMPIINSERT_ENDIF_RANK
              do i=1,n_rows_A
                 rhs0(i)=rhs(i) !Copy rhs, for input to pgmres since
                                !this input is modified.
              enddo
           endif
        elseif(soln_method.eq.'it3drv') then
           ! perform soln only at last flux surface (l_=lrz)
           if ( ilast_lr.eq.1 ) then
              call ilut (n_rows_A,ac_csr,jac_csr,iac_csr,lfil0,
     +             droptol,alu,jlu,ju,iwk_ilu,w_ilu,
     +             jw_ilu,ierr)
c              call ilutp(n_rows_A,ac_csr(1),jac_csr(1),iac_csr(1),
c     +             lfil0,droptol,permtol,mbloc,
c     +             alu(1),jlu(1),ju(1),iwk_ilu,w_ilu(1),jw_ilu(1),
c     +             iperm_ilu(1),ierr)
CMPIINSERT_IF_RANK_EQ_0
c       WRITE(*,*)'impavnc0 aft.ilut:  l_,ierr',l_,ierr,soln_method
CMPIINSERT_ENDIF_RANK
              do i=1,n_rows_A
                 rhs0(i)=rhs(i) !Copy rhs, for input to pgmres since
                                !this input is modified.
              enddo
           endif
        endif
        
        call cpu_time(tm(1))
        tm1= tm(1) - tm1
c        write(*,*)'impavnc0: icount_imp, time for ilut tm1=',
c     +                        icount_imp,tm1
        if (ierr.ne.0) then
             WRITE(*,*)'impavnc0 after ilut: ierr=',ierr
             STOP 'if ierr=-2 or -3, reduce lfil or increase iwk_ilu'
             ! Try to reduce lfil (say, to 10) or increase iwk_ilu
        endif


c      Iterative solve using:
c       subroutine pgmres(n, im, rhs, sol, vv, eps, maxits, iout,
c     *                    aa, ja, ia, alu, jlu, ju, ierr)

       krylov1=krylov        !Size of kylov subspace, whatever that is?
       
c.......................................................................
c     Initial soln sol for the iterative method is 
c     soln at the start of the time step.  At the  first time
c     step it will be either a Maxwellian or a restart distribution.
c     
c     For each j (velocity) value, take the first ipassbnd # of elements
c     from f and place them in the right half of trapped
c     region starting at iyh.
c.......................................................................

       if (soln_method.eq."itsol" .or. soln_method.eq."itsol1") then

          l1=l_
          l2=l_
          
       elseif(soln_method.eq.'it3dv' .or. soln_method.eq.'it3drv') then
                                ! perform soln
                                ! only at last flux surface (l_=lrz).
          if ( ilast_lr.eq.1 ) then
             l1=1
             l2=lrz
          else
             l1=0
             l2=-1              !i.e., following ll do-loop is no-op
          endif
       endif
       
       iii=0                   !index to point to end of eqn/soln 
                               !number for ll-flux surface.
       do ll=l1,l2
          do j=1,jx
             i1=(j-1)*inew_(ll) + iii  !eqn/soln index at end of
                                       !previous j,ll
             ihalf=0
             
             if (symtrap.eq."enabled".and. nadjoint.ne.1) then
                if (itl_(ll).lt.iyh_(ll)) then
                   inew=iyh_(ll) + itl_(ll) - 1
                   ipassbnd=iyh_(ll)-itl_(ll)+1
                else            !ie., itl_(ll)=iyh_(ll)
                   inew=iy_(ll)
                   ipassbnd=0   !i.e., no trapped particles
                                !as in following case.
                endif
                if (symtrap .ne. "enabled") then
                   inew=iy_(ll)
                   ipassbnd=0 ! for symtrap='disabled'
                endif
                
                do  ii=1,ipassbnd
                   sol(i1+ii)=f(iyh_(ll)+1-ii,j,k,ll)
                enddo
                icntsww=1
                if (itl_(ll).eq.iyh_(ll)) icntsww=0
             else if (nadjoint.eq.1 .and. symtrap.eq."enabled") then
                ipassbnd=0
                icntsww=0
                ihalf=1
             else
                ipassbnd=0
                icntsww=itl_(ll)-iyh_(ll)
             endif
             
c.......................................................................
c     Alternating between the left and right sides, place the remaining
c     values (for this value of J) below ITL and above ITU.
c.......................................................................
             
             do ii=ipassbnd+1,inew_(ll)
                if(ihalf.eq.0) then
                   sol(i1+ii)=f(itl_(ll)-icntsww,j,k,ll)
                   ihalf=1
                else
                   sol(i1+ii)=f(itu_(ll)+icntsww,j,k,ll)
                   ihalf=0
                   icntsww=icntsww+1
                endif
             enddo              !  On ii
             
          enddo                 !  On j
          
          iii=iii+inewjx_(ll)  !update index to point to end of eqn/soln 
                               !number for ll-flux surface.
          
       enddo                    !  On ll       

c     Set prmres error tolerance and max iterations internally:
       epsilon=1.d-5
       maxits=400 !200
       iout=0 !38 ! 0 means no writing to fort.iout file.
                ! Writing into this file causes problems for MPI (I/O).
                ! Keep it 0, to avoid slow runs ! YuP[04-2017]
                ! Alternatively, set in your batch script:
                ! export FORT_BUFFERED=1
                !(or the csh equivalent: setenv FORT_BUFFERED 1)
                ! At NERSC, it was the default in before 2017
                ! but it caused problems with some
                ! newer compilers.
                ! It seems its removal causes the I/O to take longer.
                ! Yuri, the underscores gave bcompare a rought time. Remove this line.


       call cpu_time(tm1)

c--------------------------------------------------------------
c     call PGMRES
c--------------------------------------------------------------
c     Put sol into rhs vector (i.e., as in soln from direct solve)
       if ( soln_method.eq."itsol" .or. soln_method.eq."itsol1" ) then
          call pgmres(n_rows_A,krylov1,rhs0,sol,vv,epsilon,
     +         maxits,iout,a_csr,ja_csr,ia_csr,alu,
     +         jlu,ju,ierr)
          rhs(1:n_rows_A)=sol(1:n_rows_A)
       elseif(soln_method.eq.'it3dv') then
                                ! perform soln 
                                ! only at last flux surface (l_=lrz).
          if ( ilast_lr.eq.1 ) then
CMPIINSERT_IF_RANK_EQ_0
c       WRITE(*,*)'impavnc0 before pgmres:  ierr',ierr,soln_method
CMPIINSERT_ENDIF_RANK
             call pgmres(n_rows_A,krylov1,rhs0,sol,vv,epsilon,
     +            maxits,iout,a_csr,ja_csr,ia_csr,alu,
     +            jlu,ju,ierr)
CMPIINSERT_IF_RANK_EQ_0
c       WRITE(*,*)'impavnc0  after pgmres:  ierr',ierr
CMPIINSERT_ENDIF_RANK
             do inr = 1,n_rows_A
               rhs(inr)=sol(inr) 
               !-> rhs(1:n_rows_A)=sol(1:n_rows_A) results in stack overflow
             enddo
          endif
       elseif(soln_method.eq.'it3drv') then
                                ! perform soln 
                                ! only at last flux surface (l_=lrz).
          if ( ilast_lr.eq.1 ) then
          
CMPIINSERT_IF_RANK_EQ_0
c       WRITE(*,*)'impavnc0 before pgmres:  ierr',ierr,soln_method
CMPIINSERT_ENDIF_RANK
             call pgmres(n_rows_A,krylov1,rhs0,sol,vv,epsilon,
     +            maxits,iout,ac_csr,jac_csr,iac_csr,alu,
     +            jlu,ju,ierr)
CMPIINSERT_IF_RANK_EQ_0
c       WRITE(*,*)'impavnc0  after pgmres:  ierr',ierr
CMPIINSERT_ENDIF_RANK
     
             do inr = 1,n_rows_A
               rhs(inr)=sol(inr) 
               !-> rhs(1:n_rows_A)=sol(1:n_rows_A) results in stack overflow
             enddo
          endif
       endif       
       
       call cpu_time(tm(1))
       tm1= tm(1) - tm1
c       write(*,*)'impavnc0: time for pgmres tm1=',tm1
       if (ierr.ne.0) then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'impavnc0 after pgmres, ierr=',ierr
          WRITE(*,*)'ierr=1 converg. not achieved in itmax iterations'
          WRITE(*,*)' -1 Initial guess seems to be the exact solution'
CMPIINSERT_ENDIF_RANK
          STOP 'ierr.ne.0, from pgmres'
       endif

      endif  ! on   soln_method.eq.'itsol' .or. soln_method.eq.'itsol1'
             ! .or. soln_method.eq.'it3dv' .or. soln_method.eq.'it3drv'

c..................................................................
c     Call direct solve Gaussian elimination algorithm.
c..................................................................

      if (soln_method.eq.'direct') then
c
         call cpu_time(tm1)
         
c     factorize matrix
c_cray  64-bit-compiler uses sgbtrf (from lapack library).
c_pc    32-bit-compiler uses dgbtrf (from lapack library).
        if (factor .ne. "disabled") then
c          call sgbtrf(inewjx,inewjx,ml,mu,abd,md1abd,ipivot,info)

!          if(n_(l_).eq.1 .or. n_(l_).eq.14)then
!            val_min=ep90
!            val_max=-ep90
!            val_sum=0.d0
!            do i=1,iy
!            do j=1,jx
!              val= di(i,j,k,l_)
!              val_sum=val_sum+val
!              val_min=MIN(val_min,val)
!              val_max=MAX(val_max,val)
!            enddo
!            enddo
!            write(*,*)'impavnc/consscal: MIN,MAX,SUM val',
!     &                             val_min,val_max,val_sum
!          endif
          !write(*,*)'before dgbtrf: ml,mu,sum(abd)',ml,mu,sum(abd)
          call dgbtrf(inewjx,inewjx,ml,mu,abd(1:md1abd,1:inewjx),md1abd,
     &                ipivot(1:inewjx),info)
!          if(sum(ipivot).lt.0)then
!           WRITE(*,*)' sum(abd),sum(ipivot)=',inewjx,md1abd,
!     &     sum(abd),sum(ipivot), size(abd),size(ipivot) 
!            pause  
!          endif 
          if (info .ne. 0) then
           WRITE(*,*)' sum(abd),sum(ipivot)=',inewjx,md1abd,
     &     sum(abd),sum(ipivot), size(abd),size(ipivot)    
            PRINT *,
     &'ERR after sgbtrf in impavnc0: n,l_,kopt,icount_ampf,info,factor',
     &       n,l_,kopt,icount_ampf,info,factor
            stop 'impavnc0 1'
          endif
        endif ! factor .ne. "disabled"
c       tm1 = etime(tm) - tm1
c       write(*,*)'impavnc0: time for dgbtrf tm1=',tm1

c     solve system
c_cray  64-bit-compiler uses sgbtrs (from lapack library).
c_pc    32-bit-compiler uses dgbtrs (from lapack library).
  
cBH080303        tm1 = etime(tm)
       call cpu_time(tm1)
       
        inbrhs = 1
        transpose = 'n'
c        call sgbtrs(transpose,inewjx,ml,mu,inbrhs,abd,md1abd,ipivot
        call dgbtrs(transpose,inewjx,ml,mu,inbrhs,
     &       abd(1:md1abd,1:inewjx),md1abd,
     &       ipivot(1:inewjx),rhs(1:inewjx),inewjx,info)
     
        if (info .ne. 0) then
          PRINT *,' warning after sgbtrs in impavnc0: info = ',info
          stop 'impavnc0 2'
        endif
c       tm1 = etime(tm) - tm1
c       write(*,*)'impavnc0: time for dgbtrs tm1=',tm1

       endif  !on soln_method.eq.'direct'
     

c     Deallocate it3d related storage
c-YuP       if (n.eq.nstop .and. ilast_lr.eq.1) then
c-YuP           call it3ddalloc 
         !-YuP  Problem: called during first iteration of eflditer;
         ! Should be after last nefiter iteration (which is unknown).
         ! Not really necessary to deallocate?
c-YuP       endif

c.................................................................
c     The next small block expands the compressed solution in rhs()
c     and replicates the mirror-imaged values in the trapping region 
c     (i.e. copy solution rhs to f)
c..................................................................

       if (soln_method.eq."itsol" .or. soln_method.eq."itsol1"
     +      .or. soln_method.eq."direct") then
          l1=l_
          l2=l_
          
       elseif(soln_method.eq.'it3dv' .or. soln_method.eq.'it3drv') then
                             ! soln performed only at last flux surface
c          write(*,*)'impavnc0: n,ilast_lr or rhs=>f',n,ilast_lr
          if ( ilast_lr.eq.1 ) then
             l1=1
             l2=lrz
          else
             l1=0
             l2=-1           !i.e., following ll do-loop is no-op
          endif
       endif  ! On soln_method
       
       iii=0                   !index to point to end of eqn/soln 
                               !number for ll-flux surface.


c Now update f() (solution is obtained => ilast_lr=1)
       
c       jstart = 1 ! For lbdry="conserv" or lbdry="consscal"
       jstart = 1 ! For lbdry .ne. "fixed"
       if(lbdry(k).eq."fixed") jstart=2 
       !For "fixed", f(j=1) is same as at the previous time step.
       
       do ll=l1,l2  
        do 966 j=jstart,jx  ! YuP: jstart=2 for lbdry="fixed"
           i1=(j-1)*inew_(ll) + iii  !eqn/soln index at end of
                                     !previous j,ll

c..................................................................
c     For each j (velocity) value, take the first ipassbnd # of elements
c     from rhs and place them in the right half of the pass-trapped
c     region starting at iyh.
c..................................................................

          ihalf=0

          if (symtrap.eq."enabled".and. nadjoint.ne.1) then
             if (itl_(ll).lt.iyh_(ll)) then
                inew=iyh_(ll) + itl_(ll) - 1
                ipassbnd=iyh_(ll)-itl_(ll)+1
             else               !ie., itl_(ll)=iyh_(ll)
                inew=iy_(ll)
                ipassbnd=0      !i.e., no trapped particles
                                !as in following case.
             endif
             if (symtrap .ne. "enabled") then
                inew=iy_(ll)
                ipassbnd=0
             endif
             
             do ii=1,ipassbnd
                f(iyh_(ll)+1-ii,j,k,ll)=rhs(i1+ii)
             enddo
        
             icntsww=1
             if (itl_(ll).eq.iyh_(ll)) icntsww=0
          else if (nadjoint.eq.1 .and. symtrap.eq."enabled") then
             ipassbnd=0
             icntsww=0
             ihalf=1
          else
             ipassbnd=0
             icntsww=itl_(ll)-iyh_(ll)
          endif

c..................................................................
c     Alternating between the left and right sides, place the remaining
c     values (for this value of J) below ITL and above ITU.
c..................................................................

          do 959 ii=ipassbnd+1,inew_(ll)
            if(ihalf.eq.0) then
              f(itl_(ll)-icntsww,j,k,ll)=rhs(i1+ii)
              ihalf=1
            else
              f(itu_(ll)+icntsww,j,k,ll)=rhs(i1+ii)
              ihalf=0
              icntsww=icntsww+1
            endif
 959      continue

c..................................................................
c     Mirror the right half of the pass-trapped region into the left hal
c..................................................................

          if (symtrap .eq. "enabled") then
            if (nadjoint .eq. 1) f(itl_(ll),j,k,ll)=f(itu_(ll),j,k,ll)
            do 957 i=iyh_(ll)+1,itu_(ll)
              ii=iy_(ll)+1-i
              f(i,j,k,ll)=f(ii,j,k,ll)
 957        continue
          endif
          
 966    continue   !   On j
 
c..................................................................
c     Following if-endif fixes a nipple which grows on f at v=0
c     in lbdry()="scale" conditions.  If namelist lbdry(k)="scale"
c     we reset it to "consscal'.  "scale" is no longer operable.
c..................................................................

cBH-YuP: 170623
cBH-YuP: Following Syun'ichi Shiraiwa, 170623, this adds to f(v=0),
cBH-YuP: and gives growing distribution function and current.
cBH-YuP: The problem evidently arose when jstart
cBH-YuP: Syun'inchi: Probably avg is better.
cBH-YuP: We will use pitch angle phase space avg
cBH-YuP:         if(lbdry(k).eq."scale")then
cBH-YuP:            ! copy MAX[f(j=2)] to f(j=1), YuP150410
cBH-YuP:            f_j2_max= MAXVAL(f(1:iy,2,k,ll)) 
cBH-YuP:            f(1:iy,1,k,ll)= f_j2_max
cBH-YuP:         endif ! "scale"

        if (  ampfmod.eq."enabled" .and. kopt.eq.3
     +        .and. cqlpmod.ne."enabled") then
          if(icount_ampf.eq.2)then ! this is fg=='g' function
            f(1:iymax,1,k,ll)=0.d0 !this is 'g' function: should be 0 at j=1
            f(1:iymax,0,k,ll)=0.d0 !this is 'g' function: should be 0 at j=0
          endif
        endif


        iii=iii+inewjx_(ll)  !update index to point to end of eqn/soln 
	                       !number for ll-flux surface.	
						                 
       enddo  !  On ll ; finished with updating f
       
!       WRITE(*,'(a,4i6,a)')
!     +  'impavnc:sol.found,f updtd. Mpirank,l_,kopt,icount_ampf,factor',
!     +                              mpirank,l_,kopt,icount_ampf,factor
       ! Note, with MPI, and soln_method.eq.'direct',
       ! parallelization is done over ll (l_) index.
       ! mpirank=0 is not doing calculations, only receiving data.

c.......................................................................
c     eseswtch="enabled" section, to control and calculate
c     fluxes for quasi-neutral electrostatic field.
c.......................................................................

        if (cqlpmod.eq."enabled" .and. eseswtch.eq."enabled") then

           if (icount_ese.eq.1) then
              
                   !             Store new f into fh
             call dcopy(iyjx2,f(0:iymax+1,0:jx+1,kelec,l_),1, 
     +                       fh(0:iymax+1,0:jx+1,1,l_),1) ! we are in k-loop here
                   !YuP[2019-05-30] Note that in case of cqlpmod.eq."enabled",
                   ! fh() is allocated as fh(0:iy+1,0:jx+1,1:ngen,0:ls+1) in wpalloc
              factor="disabled"
              icount_ese=2
              go to 10

           elseif (icount_ese.eq.2) then

             ! g function is in f.
             ! Calculate fluxes, restore f with f_, and return.
              flux1(l_)=fluxpar(1,x,coss(1:iy_(l_),l_),
     +                  cynt2(1:iy_(l_),l_),cint2,temp1,iy_(l_),jx)
              !YuP[2021-03-04] changed iy-->iy_(l_)
              ![2021-03-04] for meshy="fixed_mu" iy_(l_) can be less than iy 
              flux2(l_)=fluxpar(1,x,coss(1:iy_(l_),l_),
     +                  cynt2(1:iy_(l_),l_),cint2,temp2,iy_(l_),jx)
              f(0:iymax+1,0:jx+1,k,l_)=f_(0:iymax+1,0:jx+1,k,l_) !We are in k-loop here
              go to 999
           endif
           
        endif ! cqlpmod.eq."enabled"
       

c.......................................................................
c     ampfmod="enabled"/kopt=3 section, to calculate time-advance
c     toroidal electric field according to Ampere_Faraday equations.
c.......................................................................

        if (  ampfmod.eq."enabled" .and. kopt.eq.3
     +        .and. cqlpmod.ne."enabled") then

           if (icount_ampf.eq.1) then
              
                   !             Store new f into fh
c              call dcopy(iyjx2,f(0:iy+1,0:jx+1,kelec,l_),1, 
c     +                        fh(0:iy+1,0:jx+1,1,l_),1)
              fh(0:iymax+1,0:jx+1,1,l_)=f(0:iymax+1,0:jx+1,kelec,l_)
              !YuP[05-2017] corrected dcopy in the above line:
              !was iyjx2*ngen, but we only copy one species (kelec)
              factor="disabled"
              icount_ampf=2 !next, re-define rhs() for the 'g' function equations
              go to 10 ! return, for finding fg next

           elseif (icount_ampf.eq.2) then

           !             g function is in f, store into fg
c           call dcopy(iyjx2,f(0:iy+1,0:jx+1,kelec,l_),1, 
c     +                     fg(0:iy+1,0:jx+1,1,l_),1)
              fg(0:iymax+1,0:jx+1,1,l_)=f(0:iymax+1,0:jx+1,kelec,l_)
              !YuP[05-2017] corrected dcopy in the above line:
              !was iyjx2*ngen, but we only copy one species (kelec)
cBH170312:  Checking effect of zeroing correction part of distn:
cBH170312:              call bcast(fg(0,0,1,l_),zero,iyjx2*ngen)
c             Restore f with f_, and return.
c              call dcopy(iyjx2,f_(0:iy+1,0:jx+1,kelec,l_),1,
c     +                          f(0:iy+1,0:jx+1,kelec,l_),1)
              f(0:iymax+1,0:jx+1,kelec,l_)=f_(0:iymax+1,0:jx+1,kelec,l_)
              !YuP[05-2017] corrected dcopy in the above line:
              !was iyjx2*ngen, but we only copy one species (kelec)
              go to 999
           endif
           
        endif

c.................................................................
c     Differentiate the solution and compare with r.h.s.
c.................................................................
        if (soln_method.ne.'it3dv' .or. soln_method.ne.'it3drv') then
           if (iactst.ne."disabled") call impchk(k)
        endif

c.......................................................................
c     compute vel source term needed in transport eq. for ADI method
c.......................................................................
        if (adimeth.eq."enabled" .and. transp.eq."enabled"
     +    .and. n.ge.nonadi) call tdtrvsou(k)

c YuP:        call diagimpd(k) !---> moved to tdchief.f
        
c..................................................................
c     End of "k" (species) loop.
c..................................................................
 600  continue ! k



c-YuP      if(dalloc.eq."enabled") then
         !-YuP  deallocate(abd,STAT=istat)
         !-YuP  deallocate(ipivot,STAT=istat)
         !-YuP  Problem: called during first iteration of eflditer;
         ! Should be after last nefiter iteration (which is unknown).
         ! Not really necessary to deallocate?
c-YuP      endif
      irstart=0
 999  continue



      return
      end subroutine impavnc0




C=======================================================================
C=======================================================================      
         real*8 function z00(i,j,k)
         implicit integer (i-n), real*8 (a-h,o-z)
ccc         save  ! YuP: Not really needed
         include 'param.h'
         include 'comm.h'
         include 'advnce.h'

c      CONTAINS     PG90 at GA couldn't accept this construct,
c                   complaining that functions in advnce.h were 
c                   being redefined.

c     removed this construct for franklin.nerc.gov: pg compiler

c.......................................................................
c     z00 is the right hand side of the equation, and holds the 
c     explicit-in-time rhs of the FP difference equations.
c
c     The terms involving the factors bsl, bsu , x**_ and t0**_ 
c     are related to calculation of the bootstrap effect.
c     We assume virtually that the distribution is skewed asymetrically
c     in the trapped region...that is we assume (virtually) that 
c     f(itl).ne.f(itu) and that the difference is driven by
c     a df/dr term through bsl and bsu. Since this term involves f at
c     different radial positions, it cannot figure into the solution 
c     implicitly, that is, it is differenced explicitly. The resulting 
c     contributions appear below. There will be contributions from 
c     i=itl-1, itu+1, itl and itu only.
c     All contributions are zero elsewhere, and are zero everywhere 
c     if bootcalc= "disabled".   (Refer to Harvey et al, 1993 Sherwood
c     Theory Mtg; E. Westerhof and A.G. Peters, Computer Physics Comm.,
c     Vol. 95, p. 131-138 (1996).)
c.......................................................................


c     statement functions for itl or itu [depracated in f95] 

      t0ml_(j)=qz(j)*(
     1 cl(itl-1,j-1)*dj(itl,j-1,k,l_)*eym5(itl,l_))
     1 +r2y(j)*(-de(itl-1,j)*(1.-di(itl-1,j-1,k,l_)))
     1 /(2.*dx(j))

      t00l_(j)=
     1 +qz(j)*(
     1 -cl(itl-1,j)*dj(itl,j,k,l_)*eym5(itl,l_)
     1 +cl(itl-1,j-1)*(1.-dj(itl,j-1,k,l_))*eym5(itl,l_))
     1 +r2y(j)*(dd(itl-1,j)*(1.-di(itl-1,j,k,l_))
     1 +df(itl-1,j)*eym5(itl,l_))

      t0pl_(j)=qz(j)*(
     1 -cl(itl-1,j)*eym5(itl,l_)*(1.-dj(itl,j,k,l_)))
     1 +r2y(j)*de(itl-1,j)/(2.*dx(j))*(1.-di(itl-1,j+1,k,l_))


      t0mu_(j)=qz(j)*(
     1 -cl(itu+1,j-1)*dj(itu,j-1,k,l_)*eyp5(itu,l_))
     1 +r2y(j)*(
     1 +de(itu,j)*di(itu,j-1,k,l_))/(2.*dx(j))

      t00u_(j)=
     1 +qz(j)*(
     1 +cl(itu+1,j)*dj(itu,j,k,l_)*eyp5(itu,l_)
     1 -cl(itu+1,j-1)*(1.-dj(itu,j-1,k,l_))*eyp5(itu,l_))
     1 +r2y(j)*(
     1 -dd(itu,j)
     1 *di(itu,j,k,l_)
     1 +df(itu,j)*eyp5(itu,l_))

      t0pu_(j)=qz(j)*(
     1 +cl(itu+1,j)*(1.-dj(itu,j,k,l_))*eyp5(itu,l_))
     1 +r2y(j)*(
     1 -de(itu,j)*di(itu,j+1,k,l_)/
     1 (2.*dx(j)))


         
c  Test if bootstrap calc is needed (giving iboot=1)
         iboot=0
         if (bootcalc.ne."disabled" .and.
     1        (i.eq.(itl-1).or.i.eq.itl.or.
     2         i.eq.itu.or.i.eq.(itu+1))) iboot=1

         z00f=vptb(i,lr_)*(f_(i,j,k,l_)/dtreff+so(i,j)) +
     +        spasou(i,j,k,l_)

         if (iboot.eq.1) then
            if (i.eq.(itl-1)) then
c              itl-1 case:
               z00itl1=z00f
     1              -xpm(i,j)*bsl(j-1,k,l_)-xp0(i,j)*bsl(j,k,l_)
     2              -xpp(i,j)*bsl(j+1,k,l_)
               z00t=z00itl1
            else
c              itu+1 case:
               if (i.eq.(itu+1))then
                  z00itu1=z00f
     1                 -xmm(i,j)*bsu(j-1,k,l_)-xm0(i,j)*bsu(j,k,l_)
     2                 -xmp(i,j)*bsu(j+1,k,l_)
                  z00t=z00itu1
c              itl (or itu)case:
               else
                  z00itl=z00f
     1                 -(t0ml_(j)*bsl(j-1,k,l_)+t00l_(j)*bsl(j,k,l_)
     2                 +t0pl_(j)*bsl(j+1,k,l_)+t0mu_(j)*bsu(j-1,k,l_)
     3                 +t00u_(j)*bsu(j,k,l_)+t0pu_(j)*bsu(j+1,k,l_))
                  z00t=z00itl
               endif
            endif
            z00=z00t

         else ! iboot=0
            z00=z00f
         endif

         end function z00


c      end of file impavnc0.f
