c
c
      subroutine tdchief
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     This routine directs the calculation of CQL3d; it controls
c     input, output, calls to CQL, calls to WD*, and holds the
c     main loops over radius of the toroidal device.
c..................................................................

      include 'param.h'
      include 'comm.h'
      include 'name.h'
CMPIINSERT_INCLUDE

      character*8 icall,iplotsxr
      data iflag1/0/

c.......................................................................
c     Open cqlinput NL file and adjust to new setup0/setup structure,
c     if old two setup namelist sections are present [maintaining
c     backwards compatibility].  BH070414.
c.......................................................................
      call ainadjnl(0) ! at mpirank=0 only (but it can change cqlinput)
CMPIINSERT_BARRIER

      !Look for the presence of &FSETUP namelist,
      !adjust cqlinput: rename &FSETUP to &SETUP0 (later, restore cqlinput)
      call ainadjnl_fsetup_setup0(0) !at mpirank=0 (but it can change cqlinput)
      !pause
CMPIINSERT_BARRIER
c.......................................................................
c     Set default values for setup0 namelist 
c.......................................................................
      call aindfpa
      
c..................................................................
c     Read in first namelist, setup0: determines type of run, etc.
c..................................................................
      open(unit=2,file='cqlinput',status='old')
      read(2,setup0)
CMPIINSERT_BARRIER

c..................................................................
c     Set some major input parameters according to 
c     first namelist setup
c..................................................................
      call ainsetpa

c.......................................................................
c     Zero/set some arrays
c.......................................................................
      call aclear

      read(2,setup)  ! Gets pltinput variable, for ainplt routine.
      close(2)

      sumdtr=zero

c.................................................................
c     If noplots.ne."enabled1", initialize PGPLOT and plot 
c     namelist input and parameters which are set in code before
c     compilation.
c.................................................................
      if (noplots.ne."enabled1") then
        call pltinit   ! Initiates PGPLOT
        call ainplt
        open(unit=2,file='cqlinput',status='old')
        read(2,setup0)         !  re-read 1st (i.e., setup0) namelist
        close(2)
        call ainsetpa          !  re-set according to setup0 nml
        call ainpltpa ! plots out the parameters
      endif


c..................................................................
c     If namelist variable lrzmax.eq.1, then run the 2D code and
c     dispense with the "td" module. (Runs thru achief1 and achiefn
c     to a normal exit in ntloop).
c..................................................................
c-YuP: Not used anymore. Use general call to achiefn below, in ll-loop
c-BH:  Except that some lrzmax=1 test cases are still used, and 
c-BH:  errors can arise when only only one value of a mesh
c-BH:  is called  for.  For example, tdrmshst fails.
c-BH:  However, additional functionality, such as sigmamod plotting
c-BH:  depdends on lrzmax.ge.4 (BH120308: check this).

      if (lrzmax.eq.1 .and.(cqlpmod.ne."enabled") ) then
         !YuP[2021-02-26] added (cqlpmod.ne."enabled")
         nefiter=1              ! counts iterations; elecfld iterations for
                                ! one flux surface are not functional; 
                            ! set nefiter to 1 for logic control in impavnc0
         call achief1           ! YuP: only called during n=0; Why needed?
                                ! BH:  In the past, at least, this call with
                                !      lrzmax=1, time-stepped the soln
                                !      to n=nstop in achiefn.
      endif

c..................................................................
c     call routine which controls array initialization
c     (and sets n=0, n_(1:lrors)=0 though call aindflt1).
c..................................................................

      call tdinitl !-> call ainitial !-> finit,losscone,sourcee,frnfreya,...
         ! tdinitl-> eqcoord-> eqfndpsi-> eqorbit-> trace flux surf.
         ! solr(l,lr_), solz(l,lr_) are R,Z coords. of flux surface      
      dtr0=dtr

c..................................................................
c     Initialize main netCDF write, if netcdfnm.ne."disabled".
c     If netcdfshort.eq.'lngshrtf' determine number of distn saves.
c..................................................................

CMPIINSERT_IF_RANK_EQ_0
      nsavet=0 !YuP[2019-06-07] Should be defined in any case of netcdfshort
      isave=0  !YuP[2019-06-08] Initialize here: Index for nonzero values of increasing nsave(1:nsavea)
      ! And for 'lngshrtf', it is set below:
      if (netcdfshort.eq.'lngshrtf') then
         do i=1,nsavea
            if (nsave(i).ge.0 .and. nsave(i).le.nstop) then
              isave=i ! Indicator that there are time frames to be saved
              nsavet=nsavet+1
            endif 
         enddo
      endif
      
      if (netcdfnm.ne."disabled") then
         call netcdfrw2(0)
      endif
CMPIINSERT_ENDIF_RANK

c.......................................................................
c     LOOP OVER TIME-STEP n
c.......................................................................
 10   continue  !Loop back point, from end of subroutine

CMPIINSERT_BARRIER
CMPIINSERT_STARTTIMESTEP
      call cpu_time(t_n_start) !-.-.-.-.-.-.-.-.-.-.-.-.-.

c.......................................................................
c     Reset dtr time step if (n+1).eq.nondtr1(i)
c     (achiefn, below, advances n to n+1 near its beginning). 
c.......................................................................

      do 15 i=1,ndtr1a
         if ((n+1).eq.nondtr1(i)) then
            dtr=dtr1(i)
            dtreff=dtr
            dttr=dtr*nrstrt
         endif
 15   continue

c.......................................................................
c     Recalculate neutral beam source, if (n+1).eq.nonvphi or noffvphi,
c     or, 
c     if at beginning of a beam pulse, and have time-dep background.
c     Set ibeampon= beam pulse on-indicator (used in coefstup)
c         ibeamponp= indicates freya calc carried out for given pulse.
c.......................................................................
      if (frmodp.eq."enabled") then !YuP[2025-12-12] added if
!YuP[2022-11-17]      if ((n+1).eq.nonvphi .or. (n+1).eq.noffvphi) then
      if ((n+1).ge.nonvphi .and. (n+1).lt.noffvphi) then
        !Update NBI deposition (==> update ion source)
CMPIINSERT_IF_RANK_EQ_0      
        WRITE(*,*)'tdchief/nonvphi: call frnfreya, n=',n,'time=',timet
CMPIINSERT_ENDIF_RANK
        call frnfreya(frmodp,fr_gyrop,beamplsep,beamponp,beampoffp,
     .                nbeamsp,hibrzp,mfm1p,noplots,kfrsou,src_nbi_ep)
      endif
      endif !(frmodp.eq."enabled") !YuP[2025-12-12] added if
      

cBH171014: Could make this function of beam number.
      k=kfrsou(1)  !Only set up for one modulated beam species.
      if(k.gt.0)then
      do is=1,nso
        if ((n+1).ge.nonso(k,is) .and. (n+1).lt.noffso(k,is)) then
c          Check and apply criteria for square wave pulsed beam:
           if (beamplsep.ne.'disabled' .and. k.eq.kfrsou(1)) then
c             Set flag and time indicating time beam first starts
              if (iflag1.eq.0) then
                 iflag1=1
                 timestrt=timet+dtreff !Need to check at time of next
                                       !velocity step.
                 timespn=beamponp+beampoffp
                 timeon=beamponp
              endif
c           Determine if in an on-period of the beam pulse cycle
              iperiod=(timet+dtreff-timestrt)/timespn + em12 !allowing for
                                                    !roundoff near zero.
              time1=iperiod*timespn !Time at beginning of a pulse
              timedf=timet+dtreff-time1
              if (timedf.le.timeon) then
                 ibeampon=1
                 if (ibeamponp.eq.0.and.nbctime.ne.0) then
                 write(*,*)'tdchief/nbctime:frnfreya,n=',n,'time=',timet
                   call frnfreya(frmodp,fr_gyrop,beamplsep,beamponp,
     +                 beampoffp,nbeamsp,hibrzp,mfm1p,noplots,kfrsou(1)
     &                 ,src_nbi_ep)
                   ibeamponp=1
                 endif
              else
                 ibeampon=0
                 ibeamponp=0
              endif
           endif                !On beamplsep and kfrsou
        endif                   !On (n+1)
      enddo                     !On is
      endif ! k>0

c$$$c    Alternative to above do loop, test coding:
c$$$      ibeampon=1
c$$$      ibeamponp=1


c.......................................................................
c     Obtain transport time-dependent scale factor, for required cases
c.......................................................................
          if (difus_io(1).eq."drrin") then
             if (n_d_rr.ne.0) then
                !Getting time-dep scale factors
                do k=1,n_d_rr   !n_d_rr obtained in diffus_io
                   drrt(k)=difus_io_scale(k,1)
                enddo
             endif
          elseif(difus_io(1).eq."drrdrin") then
             if (n_d_rr.ne.0) then
                !Getting time-dep scale factors
                do k=1,n_d_rr   !n_d_rr obtained in diffus_io
                   drrt(k)=difus_io_scale(k,1)
                   drt(k)=difus_io_scale(k,2)
                enddo
             endif
          endif
      
c.......................................................................
c     compute source term due to spatial transport operator (ADI method)
c.......................................................................

      if (transp.eq."enabled" .and. adimeth.eq."enabled" .and. 
     +  (n+1).ge.nonadi) then
        dtreff=dtr / 2.0
        dttr=dtreff
c     radial transport
        if (cqlpmod .ne. "enabled") call tdtrrsou
c     parallel transport
        if (cqlpmod.eq."enabled" .and. n.ge.nontran) call wparsou
      endif

c..................................................................
c     Call urf module..
c..................................................................
      t_urf1=0.
      t_urf2=0.
      t_urf3=0.      
      if (urfmod.ne."disabled") then
        call cpu_time(t_urf1) !-.-.-.-.-.-.-.-.-.-.-.-.-.
        call urfchief
        call cpu_time(t_urf2) !-.-.-.-.-.-.-.-.-.-.-.-.-.
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*) 'tdchief after urfchief'
CMPIINSERT_ENDIF_RANK
c       Set up netcdf store of rf data.
CMPIINSERT_IF_RANK_EQ_0
        if (netcdfnm.ne."disabled" .and. n.eq.0) then
           do krf=1,mrf !YuP:04-2010: Separate data file for each wave type krf
              call netcdfrf(0,krf) !kopt=0: initialize and write grids,...
              WRITE(*,*) 'after netcdfrf(0,krf)  krf=', krf
           enddo
        endif
CMPIINSERT_ENDIF_RANK
        call cpu_time(t_urf3) !-.-.-.-.-.-.-.-.-.-.-.-.-.
        !YuP[03-2016] Repeat plotting surfaces, but now - with rays
        if (noplots.ne."enabled1" .and. eqmod.eq."enabled"
     +      .and. n.eq.0) then
           if(mrf.gt.0)then ! just in case !YuP[2020-10-19] was mrfn
              !(mrf is supposed to be >0 when urfmod.ne."disabled")
              do krf=1,mrf !YuP[2020-10-19] was krf=1,mrfn
               ! We only need to plot for each wave type, not each wave mode
               call tdplteq(krf) ! separate page with rays for each krf type
              enddo
           endif
        endif
      endif ! urfmod.ne."disabled"

c.......................................................................
c     cqlpmod.eq.enabled:
c     Compute parallel electric field from Poisson equation
c.......................................................................

      if (transp.eq."enabled" .and. cqlpmod.eq."enabled")
     +  call wpelecf(11)  ! eleven, not ll

c.......................................................................
c     cqlpmod.eq.enabled:
c     Compute electrostatic parallel electric field from constant flux
c     condition (for CQLP + current drive localized along field line).
c     Uses Kupfer (PoP, 1995) f and g function decompostion in achiefn.
c     This is a nonlocal Efield calc, and therefore all space 
c     points must be cycled through.
c     BH131103 Comment added:  This option is not operational.
c              Compare with the stella code, where it is implemented.
c              Also compare with ampfmod coding.
c.......................................................................

      !.................................................................
      if(cfp_integrals.eq.'enabled')then
      !YuP[08-2017][2020-07-02] Added:
        if (ntotal.gt.ngen) call cfp_integrals_maxw
      !Calculate certain integrals 
      ! needed for subr. cfpcoefn.
      ! These integrals describe a contribution to BA coll.coefs
      ! from local collisions of general species with the background 
      ! Maxwellian species (search "kbm=ngen+1,ntotal").
      ! These integrals only depend on mass (fmass)
      ! and local temperature of these (Maxwellian) species.
      ! So, instead of calculating them over and over again
      ! at each point along particle orbit (of the general species),
      ! calculate them once as a table over temperature grid 
      ! (search "T-grid"), and then reuse them by matching a local T
      ! along orbit with the nearest value in the T-grid.
      ! These integrals may need to be updated at each step if the temp() of 
      ! the Maxwellian species is varied in time (as a prescribed form).
      endif ! (cfp_integrals.eq.'enabled')
      !.................................................................

      if (cqlpmod.eq."enabled" .and. eseswtch.eq."enabled" .and.
     +    sbdry.eq."enabled") then
         kopt=2
         ilend=lrors
         do ll=1,ilend
            call tdnflxs(ll) 
            call achiefn(kopt) ! achiefn(2) here (cqlpmod="enabled")
         enddo
         call efld_cd(dsz(1:ls),ls,vnorm, 
     &       flux1(0:ls+1),flux2(0:ls+1),elparnw(0:ls+1),flux0)
         !YuP[2021-02-26] Changed lrors-->ls, for appropriate meaning
         ! as ls= FPE grid in case of CQLP 
         !In sub.efld_cd: dz, flux1, flux2, elparnw are dimensioned as (0:ls+1).
         !Here, dz(lz,lrzmax) starts with index (1,1),
         !but flux1(0:ls+1),flux2(0:ls+1),elparnw(0:ls+1) start with index 0.
         !YuP[2019-05-30] corrected sub.efld_cd
         ! so that dz argument starts with index 1, i.e. dz(1:ls) [dz(1:lrors)]
         ! Note that in sub.efld_cd the loop is over kk=1,ls [kk=1,lrors]
         ! so that indexes 0 and ls+1 are not used.
         !However, the correct way is to use sz(l)=z(lsindx(l),lr_)
         !and dsz(l)=0.5*(sz(l+1)-sz(l-1)) which are over FPE grid,
         !so, changed dz()-->dsz(1:ls) ![2021-03-18]
      endif

c.......................................................................
c     MAIN VEL TIME STEP: loop over each radial position sequentially
c.......................................................................

      ilend=lrors
      
      if(cqlpmod.eq."enabled")then ! CQLP only
      if (transp.eq."enabled" .and.
     +  mod(nummods,10).ge.5 .and. sbdry.ne."periodic"
     +  .and. lmidvel.ne.0) then
        ilend=lrors-1
      endif
      endif

      if (nstop.eq.0) go to 2

cBH131230:  Why won't ampfmod work before MAIN VEL TIME STEP?? nstop=0 issue?
c.......................................................................
c     Compute the time-advanced toroidal electric field from Ampere-
c     Faraday equations using Kupfer h and g functions.
c     
c     cqlpmod.ne.enabled has been checked.
c     Initialize electric fields at turnon of ampfmod.
c.......................................................................


      if (ampfmod.eq.'enabled' .and. n.eq.0) then
         !YuP[21-08-2017] added: no time adv. yet, no ampf iterations yet,
         !simply fill-in and save the values at n=0 for plotting.
         !!! Besides, if nonampf=1, we need the zero iteration, 
         !!! which is taken from the previous time step,
         !!! and so we need values of elecfldn at n=0 in such case.
         !!! See a note just after call ampfefldb few lines below.
         do niter=0,nampfmax ! or up to nefitera
           do ll=1,lrz
           elecfldn(ll,n,niter)=elecfld(ll)/300.d0 !here: n=0
           enddo
           ! Also need the end (bndry) point: elecfldb not set for iproelec='prbola-t'
           elecfldn(lrz+1,n,niter)=elecfldb/300.d0 !here: n=0
           ! Also save ll=0 point (magn.axis point): Better to project to 0?
           elecfldn(0,n,niter)=elecfld(0)/300.d0   !here: n=0
         enddo
      endif

      if (ampfmod.eq."enabled" .and. n+1.ge.nonampf) then
         ! n is not updated yet. Here n=0,1,2,...,nstop-1.
         ! For example, if nstop=5 and nonampf=5 
         ! (meaning: apply ampf calc. at the last step),
         ! we need to start using this part when n+1=nstop=nonampf
         
         if (n+1.eq.nonampf) then
           call ampfinit ! n is not updated yet, Here n=0,1,...,nstop-1
         endif
         
         if (kelecg.eq.0) then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)
            WRITE(*,*)'Amp-Faraday eqns require electron gen species'
            WRITE(*,*)
CMPIINSERT_ENDIF_RANK  
            stop 'Amp-Far: kelecg should not be 0'
         endif
c        Find dtr for this time-step
         do i=1,ndtr1a
            if ((n+1).eq.nondtr1(i)) then
               dtr=dtr1(i)
               dtreff=dtr
               dttr=dtr*nrstrt
            endif
         enddo
         !Copy current distribution f into f_ (later f is restored from f_)
         ! f_(0:iy+1,0:jx+1,kelec,:)=f(0:iy+1,0:jx+1,kelec,:) !amp-far
         call dcopy(iyjx2*lrors,f(0:iymax+1,0:jx+1,1:1,1:lrors),1,
     &                         f_(0:iymax+1,0:jx+1,1:1,1:lrors),1) !ampfmod=enabled
         ! For amp-far - one species (kelecg)
         ! f() will store fg,fh, then f() is restored from f_() [impavnc0,Line~2590]
c        Bring background profiles up to time step n
         ! No effect if bctime=0 (time-indep. profiles).
cBH191002 Added a call profiles immediately after calc of updated f, l 607.
cBH191002 Tried removing following line, along with adjustment of n in
cBH191002 profiles.f, but the resulting test runs were too different.
cBH191002 This could be investigated, to clarify why.  Seems like a repeat
cBH191002 call to me.    
         call profiles ! if(ampfmod.eq.'enabled' .and. n+1.ge.nonampf)
                       ! skip elecfld background profiles
         !YuP[2019-12-18] Save reden(kelecg,ll) and energy(kelecg,ll)
         !They could be modified around call_tdboothi 
         !during ampfsoln iterations.
         !These saved values will be restored after all iterations are done.
         reden_n(1:lrz)= reden(kelecg,1:lrz)  !YuP[2019-12-18] saved
         energy_n(1:lrz)=energy(kelecg,1:lrz) !YuP[2019-12-18] saved
         
c        Get given boundary condition (i.e., edge) elec field, at time
c        advanced position (i.e., elecfldn(ll,n+1,0) at ll=lrz+1 point).
c        If time-dependent (nbctime>0),
c        then it is given by time advanced elecb() or elecin_t(njene,).
cBH191012         call ampfefldb(n+1,time+dtr)  BH: time=0. here.
         call ampfefldb(n+1,timet+dtr)
c        Besides, subr.ampfefldb Sets zero iteration of elecfldn 
c        equal to previous time step radial profile, like this:
         ! elecfldn(ll,nn,0)=elecfldn(ll,nn-1,it_prev) ! where nn==n+1 (and ll=0:lrz)
         ! So, if nonampf=1, then we have here n=0, or nn=1,
         ! which means that we need the values of elecfldn at nn-1=0.
         

c        elecfld(1:lrz) has been set.  If n=1=nonampf, then is given;
c                       If n.ge.nonampf, then set near end of ampfsoln.

c        Determine the time-advanced tor e-field
         kopt=3  !Amp-Faraday option
         ilend=lrz

c        Initialize elecfldn(ll,n+1,it=0) for upcoming ampfsoln
         do ll=1,ilend
            elecfldn(ll,n+1,0)=elecfld(ll)/300.d0
         enddo
         !YuP[21-08-2017] added but then commented out: 
         !elecfldn(lrz+1,n+1,0)=elecfldb/300.d0 !Not needed, done in ampfefldb

c        Obtain the h and g functions over all radii and iteratively
c        solve Ampere-Faraday eqns for the tor electric field.
         it_ampf=0
         do it=1,nampfmax
            it_ampf=it
CMPIINSERT_BARRIER

cBH131109: Maybe don't need from bndry to center for fh/fg?   do ll=ilend,1,-1
cBH131109: Backwards is giving bounds problem with abd in impavnc0.
cBH131109: Present system is determining all fh/fg first as function
cBH131109: of radius, then solve AF eqn.  Future solves may require
cBH131109: fix the counter resetting for ll=ilend,1,-1.

            do ll=1,ilend
               call tdnflxs(ll)  !get l_,lr_,..
CMPIINSERT_MPIWORKER
              ! It will insert :
              ! if(soln_method.eq.'direct' .and. lrzmax.gt.1) then
              !    ! Parallelization for the impavnc0 solver is limited 
              !    ! to soln_method='direct' (for now)
              !    mpiworker= MOD(ll-1,mpisize-1)+1  !1...(mpisize-1)
              ! else
              !    ! In all other cases, perform calculations 
              !    ! for all flux surfaces on mpirank=0, then broadcast results
              !    mpiworker=0
              ! endif
               call achiefn(kopt) !kopt=3 here !Increments n by 1, for each ll
                                  !Gives fh,fg(,,1,ll) for each ll.
                                  !it_ampf is passed in common block
               ! YuP test/printout (comment when not needed):
               !amp_f_=ampfarl(f_(0:iy+1,0:jx+1,kelec,ll),ll)*dtr !ampfarl has 1/dtr factor
               !amp_f= ampfarl( f(0:iy+1,0:jx+1,kelec,ll),ll)*dtr !ampfarl has 1/dtr factor
               !amp_h= ampfarl(fh(0:iy+1,0:jx+1,kelec,ll),ll)*dtr
               !amp_g= ampfarl(fg(0:iy+1,0:jx+1,kelec,ll),ll)*dtr
               !YuP[2019-05-30] Note that fh() and fg() are allocated as
               ! fh(0:iy+1,0:jx+1,1,1:lrz), i.e. k=1 index (for now).
               ! If kelec>1, we can get a problem here.
!               WRITE(*,'(a,4i4,3e12.4)')
!     +         'after achiefn(3): mpirank,n,it,ll,integrals f, f-h, g:',
!     +            mpirank,n,it,ll, amp_f, amp_f-amp_h, amp_g  
               ! YuP test/printout: 
               !Confirmed that f() remains unchanged during
               !iterations in this it loop, and it remains equal to f_()
CMPIINSERT_SEND_RECV_AMPF
               !It will send or recv data (each ll), but only in case of
               !soln_method='direct' (for now)
            enddo ! ll
CMPIINSERT_BARRIER
            !Restore distribution f from f_ (done on all cores):
            !f(0:iy+1,0:jx+1,kelec,:)=f_(0:iy+1,0:jx+1,kelec,:) !amp-far: restore
            call dcopy(iyjx2*lrors,f_(0:iymax+1,0:jx+1,1:1,1:lrors),1,
     &                              f(0:iymax+1,0:jx+1,1:1,1:lrors),1) !ampfmod=enabled
            
! broadcast fh and fg for iyjx2*1*lrz index range:
CMPIINSERT_BCAST_DISTRIBUTION_AMPF
CMPIINSERT_BCAST_COLL_COEFFS
CMPIINSERT_BCAST_SCAL
CMPIINSERT_BCAST_VELSOU
CMPIINSERT_BCAST_SOURCE
CMPIINSERT_BCAST_XLNCUR
CMPIINSERT_BCAST_SORPW_NBI
CMPIINSERT_BARRIER
            ! Solve for iteratively updated tor electric field
            ! n is time advanced value
            !YuP: just in case, for proper vaue of n:
            n=n_(1) ! not advanced yet.
            nn=n+1 ! advanced by +1, for saving solution [elecfldn(ll,nn,it)]
            call ampfsoln(it,nn) !here nn=nonampf,...,nstop
            !ampfsoln uses fg,fh,f_() functions at ALL ll (radial indexes)
            ! Presently, ampfdiff does nothing, i.e. gives iflag=1
            ! In future, refine this, making it=it(1:lrz).
            call ampfdiff(iflag) !iflag=0, only if error criteria met.
                                 !Presently a dummy call giving iflag=1
            if (iflag.eq.0) go to 16            
         enddo !On it (Note: it is incremented by 1 at the exit of loop)
 16      continue
         !YuP[2019-12-18] Restore reden(kelecg,ll) and energy(kelecg,ll)
         !They could be modified around call_tdboothi 
         !during ampfsoln iterations.
         !These values were saved and now restored :
         reden(kelecg,1:lrz)=  reden_n(1:lrz) !YuP[2019-12-18] restored
         energy(kelecg,1:lrz)=energy_n(1:lrz) !YuP[2019-12-18] restored
 
c        Now, setup with time-advanced electric field in elecfld(),
c        then at end of ampfmod interations, call achiefn(0) below.
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,'(a,i4,3e14.7)')
     +    'tdchief/aft.ampfsoln: n,sum(elecfld),sum(f),sum(f_)',
     +    n,sum(elecfld),sum(f(:,:,1,:)),sum(f_(:,:,1,:))
          ! The print above is always same in MPI runs, from run to run.
CMPIINSERT_ENDIF_RANK

      endif  !On ampfmod

c...........................................................
c     Loop back point for electric field iteration
c       (if efiter.eq."enabled")
c...........................................................
      nefiter=1 ! counts iterations
      do ll=1,ilend
        call tdnflxs(ll) ! determine l_,lr_, etc.
        nefiter_(l_)=1 ! counts iterations for each flux surface
      enddo
      
 20   continue  !Loop back point for electric field iteration

      call cpu_time(t_before_soln) !-.-.-.-.-.-.-.-.-.-.-.-.-.
CMPIINSERT_BARRIER

      ! Special test: Make a non-restarting "run3" (with nstop=15),
      ! but in the middle of the run read distrfunc.nc, 
      ! which is a copy of mnemonic.nc 
      ! produced during similar but shorter "run1" with nstop=13.
      ! So, in such a run3, the distr.func. is over-written
      ! at step n=13 (going into n=14) by a saved distr.func. from run1.
      ! Result in such run3 is exactly same as in a normal run3, 
      ! when these 6 lines are commented (so f is NOT overwritten).
      ! It proves that the data is recordered and read correctly.
!c      if((n.eq.0 .and. nstop.eq.2).or.(n.eq.13))then !FOR TEST ONLY: 0 for run2; 13 for run3
!c        nlrestrt='ncdfdist'  ! FOR TEST ONLY: temporary enable
!c        l_=lrors !Because the data is only read when l_=lrors in tdreadf
!c        call tdreadf(2)      ! FOR TEST nstop02_test14 ONLY
!c        nlrestrt='disabled'  ! FOR TEST ONLY: restore 
!c      endif ! FOR TEST ONLY
      !For run2(modified) - This placement is important, BEFORE f->f_ copying
      !Then run2 gives same as test 9--12 (same as usual finit->tdreadf)
      
c     Copy current distribution f(old,prev.step) into f_
      !YuP[2019-09] Strange. This method of copying gives overflow [on PC/intel]:
      ! f_(0:iy+1,0:jx+1,1:ngen,1:lrors)=f(0:iy+1,0:jx+1,1:ngen,1:lrors)
      !This also gives overflow: f_=f 
      ! But using dcopy() is ok (with range specified, or not specified - same)
      !write(*,*)'tdchief-597 sum(f_),sum(f)=',sum(f_),sum(f)
      call dcopy(iyjx2*ngen*lrors,f(0:iymax+1,0:jx+1,1:ngen,1:lrors),1,
     &                           f_(0:iymax+1,0:jx+1,1:ngen,1:lrors),1)
            !write(*,*)'tdchief-522 after f_ to f', sum(f_),sum(f)

      if (transp.eq."enabled" .and. n.ne.0 .and. adimeth.ne."enabled"
     +      .and. soln_method.ne."it3drv" .and. nefiter.eq.1)  then
c.......................................................................
c     With adimeth, one assumes f (in tdtrrsou or wparsou) 
c     as the starting point of the new iteration.
c     Thus should not need to redefine here.
c     CQLP case: f defined in wpsavf is assumed to be the good one.
c     soln_method=it3drv:  soln is in f(,,,) on last call on the set of
c     flux surfaces (soln at start of step is in f_(,,,)).
c.......................................................................
        if(cqlpmod.ne."enabled" .and. n.ge.nontran .and. n.lt.nofftran)
     +      call dcopy(iyjx2*ngen*lrors,frn_2(0,0,1,1),1,f(0,0,1,1),1)
      endif
      
c..................................................
c     Bring background profiles up to time step n.
c     Exclude updating of electric field elecfld(1:lrz) for 
c     ampfmod.eq.enabled .and. n.ge.nonampf [Check this for
c     the eseswtch case also.]
c..................................................

      if(nefiter.eq.1) then 
         call profiles
             !write(*,*)'tdchief[587]/profiles: n,n_(1)',n,n_(1)
         if(cqlpmod.ne."enabled")  call impurity_update         
      endif ! (nefiter.eq.1)
                 
c.......................................................................
cBH131103      do 1 ll=1,ilend   !ilend=lrz for cqlpmod.ne.'enabled'
c
c     Can reverse order of do 1 ll=1,ilend, if needed for purposes
c     of solving Ampere-Faraday eqns starting at given edge voltage.
c     Results are unchanged for simple efield case, BUT additional
c     debugging required for TCV ECCD/radial diffusion test case (CD
c     profiles are similar, but total driven current changes from 
c     108 kA to 138 kA.)
cBH131107 Moved the ll=1,ilend reversal to above if(ampfmod.eq.enabled..
c.......................................................................
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,'(a,i4,3e14.7)')
     +    'tdchief/bef.achiefn(0): n,sum(elecfld),sum(f),sum(f_)',
     +    n,sum(elecfld),sum(f(:,:,1,:)),sum(f_(:,:,1,:))
          ! The print above is always same in MPI runs, from run to run.
CMPIINSERT_ENDIF_RANK
c      Checked: sum(da),etc, are zeroes here.

      !not really needed: call starnue_sptz !Get starnue(),tauee(),taueeh(),sptzr(); all lr

      do 1 ll=1,ilend   !ilend=lrz for cqlpmod.ne.'enabled'
        !determine local variables depending on flux surface (l_,iy,..)
        call tdnflxs(ll) !-> get l_,lr_,...
        ! Reset time step if (n+1).eq.nondtr1(i). .AND. LRZMAX=1
        do i=1,ndtr1a
           if ((n+1).eq.nondtr1(i)) then
              dtr=dtr1(i)
              dtreff=dtr
              dttr=dtr*nrstrt
           endif
        enddo

c......................................................................
c     If ampfmod.eq.enabled, but n.lt.nonampf, then insert values of
c     elecfldn(1,,) from elecfld [for plotting with .nc].
c......................................................................

      if (ampfmod.eq.'enabled' .and. n+1.lt.nonampf) then
         ! Here, n is NOT advanced yet for the next step. 
         ! Here n=0,1,2,...,nstop-1.
         !    n+1=1,2,3,...,nstop
         ! For example, if nstop=5 and nonampf=5 
         ! (meaning: apply ampf calc. at the last step),
         ! we need to skip this part
         ! when n+1=nonampf,
         ! but use it when n+1<nonampf
         ! Save into the future-updated n+1= 1,2,3,nonampf-1 
cBH170329
         !YuP[21-08-2017] Corrected, for the fields storage BEFORE the ampf is applied:
         do niter=0,nampfmax ! or up to nefitera
           elecfldn(ll,n+1,niter)=elecfld(ll)/300.d0
           ! here: ll=1:lrz. But we also need the end (bndry) point:
           elecfldn(lrz+1,n+1,niter)=elecfldb/300.d0 !here: n+1.lt.nonampf
           ! Can also save ll=0 point (magn.axis point):
           elecfldn(0,n+1,niter)=elecfld(0)/300.d0
         enddo
      endif


c......................................................................
c     TIME ADVANCE ON FLUX SURFACE rz(ll).
c     If cqlpmod.ne.'enabled', transp='enabled', soln_method='it3drv',
c       then following call only adds ll flux surface contributions to
c       the coefficient matrix, and then solves for time advanced distn
c       at last flux surface, ll=lrz.
c......................................................................

CMPIINSERT_MPIWORKER
c    It will insert :
c      if(soln_method.eq.'direct' .and. lrzmax.gt.1) then
c         ! Parallelization for the impavnc0 solver is limited 
c         ! to soln_method='direct' (for now)
c         mpiworker= MOD(ll-1,mpisize-1)+1  !1...(mpisize-1)
c      else
c         ! In all other cases, perform calculations 
c         ! for all flux surfaces on mpirank=0, then broadcast results
c         mpiworker=0
c      endif

        call achiefn(0)  !--> if implct='enabled', calls impavnc0

CMPIINSERT_SEND_RECV
c     It will send or recv data, but only in case of
c     soln_method='direct' (for now)

 1    continue ! End loop over radius (or point along fld line, for CQLP):
               !  New f is obtained for each l_
               
!      if(n.eq.1)then
!         open(unit=17,file="pitch_i.txt",status="replace")
!            write(17,'(10ES12.5E2)') (y(i,1),i=1,iymax) !for l_=1 only
!         close(17)
!         open(unit=18,file="u_j.txt",status="replace")
!            write(18,'(10ES12.5E2)') (vnorm*x(j),j=1,jx)
!         close(18)
!         open(unit=19,file="f_ij.txt",status="replace")
!         do j=1,jx
!            write(19,'(10ES12.5E2)') (f(i,j,1,1)/vnorm**3, i=1,iy)
!         enddo
!         close(19)
!
!         open(unit=17,file="pitch_long_i.txt",status="replace")
!            write(17,'(10ES16.9E2)') (y(i,1),i=1,iymax) !for l_=1 only
!         close(17)
!         open(unit=17,file="dpitch_long_i.txt",status="replace")
!            write(17,'(10ES16.9E2)') (dy(i,1),i=1,iymax)
!         close(17)
!
!         m=0
!         write(*,*)' m, 0.5*(2*m+1)*sum(dcofleg(:,m))=',
!     &               m,0.5*(2*m+1)*sum(dcofleg(:,1,m,1))
!         m=1
!         write(*,*)' m, 0.5*(2*m+1)*sum(dcofleg(:,m))=',
!     &               m,0.5*(2*m+1)*sum(dcofleg(:,1,m,1))
!         open(unit=17,file="P1_i.txt",status="replace")
!            write(17,'(10ES16.9E2)') (dcofleg(i,1,m,1),i=1,iymax) !for l_=1 only
!         close(17)
!         m=2
!         write(*,*)' m, 0.5*(2*m+1)*sum(dcofleg(:,m))=',
!     &               m,0.5*(2*m+1)*sum(dcofleg(:,1,m,1))
!         open(unit=17,file="P2_i.txt",status="replace")
!            write(17,'(10ES16.9E2)') (dcofleg(i,1,m,1),i=1,iymax) !for l_=1 only
!         close(17)
!         m=3
!         write(*,*)' m, 0.5*(2*m+1)*sum(dcofleg(:,m))=',
!     &               m,0.5*(2*m+1)*sum(dcofleg(:,1,m,1))
!         open(unit=17,file="P3_i.txt",status="replace")
!            write(17,'(10ES16.9E2)') (dcofleg(i,1,m,1),i=1,iymax) !for l_=1 only
!         close(17)
!         m=4
!         write(*,*)' m, 0.5*(2*m+1)*sum(dcofleg(:,m))=',
!     &               m,0.5*(2*m+1)*sum(dcofleg(:,1,m,1))
!         open(unit=17,file="P4_i.txt",status="replace")
!            write(17,'(10ES16.9E2)') (dcofleg(i,1,m,1),i=1,iymax) !for l_=1 only
!         close(17)
!         pause
!      endif
!
               
!      write(*,*)'tdchief aft.achiefn: MIN,MAX,SUM of f:',
!     & n,MINVAL(f),MAXVAL(f),SUM(f) 
!      Checked: sum(da),etc, are same in run2(restart) and run3(no restart)
CMPIINSERT_IF_RANK_EQ_0
      if(sum(srckotot).gt.0.d0)then !YuP[2020-10-20] print only if >0
      do ll=1,ilend
         WRITE(*,'(a,2i4,1e14.7)')'sourceko: n,ll,srckotot',
     +   n,ll,srckotot(ll) 
      enddo
      endif
CMPIINSERT_ENDIF_RANK

      !YuP[2020-02-11] call profiles  !YuP: moved this further, after call_diaggnde
      
      write(*,*)'=====> tdchief[828] after FPE soln, aft.profiles. n=',n
      write(*,*)' '
CMPIINSERT_BARRIER
CMPIINSERT_BCAST_DISTRIBUTION
CMPIINSERT_BCAST_COLL_COEFFS
CMPIINSERT_BCAST_SCAL
CMPIINSERT_BCAST_VELSOU
CMPIINSERT_BCAST_SOURCE
CMPIINSERT_BCAST_XLNCUR
CMPIINSERT_BCAST_SORPW_NBI
      call cpu_time(t_after_soln) !-.-.-.-.-.-.-.-.-.-.-.-.-.

      do ll=1,ilend  ! Re-scale f if needed; 
        call tdnflxs(ll) ! determine l_,lr_, etc.
        do k=1,ngen  ! Compute density gains and losses, and powers.
           ! For lbdry0='disabled',  Redefine f at v=0 so it is unique:
           ! (For lbdry0='enabled', coeff matrix is set up 
           !   to automatically maintain unicity.)
           if (lbdry0.ne."enabled") then !-YuP: moved here from impavnc0
             call dcopy(iyjx2,f(0,0,k,l_),1,fxsp(0,0,k,l_),1)
             s=0.
             t=0.
             do 2100 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
               s=s+vptb(i,lr_)*cynt2(i,l_)
               t=t+vptb(i,lr_)*cynt2(i,l_)*f(i,1,k,l_)
 2100        continue
             do 2200 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
               f(i,1,k,l_)=t/s
 2200        continue
           endif
           !YuP-101227: Skip diagnostics here; It is done after 21_continue.
           !(to skip, set lbdry(1)='fixed'): 
           call diagscal(k) !YuP! renorm f if lbdry(k)=scale/consscal
        enddo ! k
      enddo ! ll

      call cpu_time(t_after_diag1) !-.-.-.-.-.-.-.-.-.-.-.-.-.

c..................................................................
c     Check/iterate on electric field to obtain a target current
c..................................................................
      if (efiter.eq."enabled") then
         ! Stop iterations, if beyond iteration limit:
         if (nefiter .gt. nefitera) go to 21
         ! Stop iterations if n<(control turn on step):
         if (n. le. noncntrl) go to 21
         write(*,*)'TDCHIEF/EFLDITER: Time step=',n,
     ~             '    Starting iteration #',nefiter
         nefiter_all=0 ! Summ-up nefiter_out (output of eflditer)
         do ll=1,ilend ! Check current for each flux surface
            ! determine local variables depending on flux surface (l_, iy,..)
            call tdnflxs(ll)
            if(nefiter_(l_).ge.1) then
              nefiter_in=nefiter_(l_) ! remember the input value of nefiter.
              ! re-adjust elecfld(l_) if target current is not achieved:
              call eflditer ! nefiter_(l_) is both in and out; now in comm.h !
              ! If target current is achieved, nefiter_(l_)->0 for this flux surface.
              nefiter_out=nefiter_(l_)
              nefiter_all=nefiter_all+nefiter_out 
              ! nefiter_all will remain 0 if nefiter_out=0 for each flux surface.
              !!!nefiter_(l_)=nefiter_(l_)+1 ! counts iterations for each flux surf.
              ! Comment the line above to skip iterations for surfaces for which 
              ! target current is achieved (otherwise - check them during next iteration)
            endif
         enddo ! ll
         if (nefiter_all.eq.0) go to 21 ! Finished with iterations
         ! Otherwise: Restore distribution function back to value at beginning of
         ! of the time step  (saved near beginning of impanvc/impavnc0).
         do ll=1,ilend 
            ! determine local variables depending on flux surface (l_, iy,..)
            call tdnflxs(ll)
c            call dcopy(iyjx2,f_(0:iy+1,0:jx+1,kelec,l_),1, 
c     &                        f(0:iy+1,0:jx+1,kelec,l_),1)
            f(0:iymax+1,0:jx+1,kelec,l_)=f_(0:iymax+1,0:jx+1,kelec,l_)
            !YuP[2019-05-30] Corrected species index '1'-->'kelec'
         enddo ! ll
         nefiter=nefiter+1 ! counts iterations
         go to 20 ! Another iteration, using old f() but new elecfld()
      endif
      
 21   continue

CMPIINSERT_BARRIER

c     Accumulate favg time average, if tavg.ne."disabled"
      if (tavg.ne."disabled") then
         do ii=1,ntavga
         tavg12=tavg2(ii)-tavg1(ii)
         if (tavg12 .gt. 0.d0) then
         if (timet.gt.tavg1(ii) .and. timet.le.tavg2(ii)) then
           !Note: if at the end of run
           ! timet is still less than the very 1st time point  
           ! for averaging t=tavg1(1) 
           ! (may happen if nstop and/or dtr are too small)
           ! then the averaging process never happens, 
           ! and so sumdtr remains zero.
           sumdtr=sumdtr+dtr   !Accumulate dtr, for denom in time-avg
           do ll=1,ilend
           do k=1,ngen   
               do i=0,iy_(ll) ! !YuP[2021-03-11] iy-->iy_(ll)
                  do j=0,jx
                    favg(i,j,k,ll)=favg(i,j,k,ll)+dtr*f(i,j,k,ll)
                  enddo
               enddo
           enddo  ! On k
           enddo  ! On ll
         endif  ! On timet
         endif  ! On tavg12>0 
         enddo  ! On ii
         ! At the end of run form the distribution averaged over all
         ! "ii-blips" [tavg1(ii);tavg2(ii)]
         if (n .eq. nstop) then
           if(sumdtr.lt.em90)then  
             ! YuP[03-01-2016] fix added for sumdtr=0 case
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)'tdchief WARNING: No time averaging of f was done'
           WRITE(*,*)'tdchief WARNING: Probably t=nstop*dtr is less'
           WRITE(*,*)'tdchief WARNING: than tavg1(1).'
           WRITE(*,*)'tdchief WARNING: Instead of favg(), '
           WRITE(*,*)'tdchief WARNING: the last available f() will be'
           WRITE(*,*)'tdchief WARNING: saved into mnemonic.nc file. '
CMPIINSERT_ENDIF_RANK
             do ll=1,ilend
             do k=1,ngen   
               do i=0,iy_(ll) ! !YuP[2021-03-11] iy-->iy_(ll)
               do j=0,jx
                  favg(i,j,k,ll)=f(i,j,k,ll)
               enddo
               enddo
             enddo  ! On k
             enddo  ! On ll
           else  ! sumdtr>0
             do ll=1,ilend
             do k=1,ngen   
               do i=0,iy_(ll) ! !YuP[2021-03-11] iy-->iy_(ll)
               do j=0,jx
                  favg(i,j,k,ll)=favg(i,j,k,ll)/sumdtr
               enddo
               enddo
             enddo  ! On k
             enddo  ! On ll
           endif ! sumdtr =0 or >0
         endif  ! On n.eq.nstop
      endif  ! On tavg
      
      
      do ll=1,ilend  ! perform diagnostics [Power transfer diagnostics].
        call tdnflxs(ll) ! determine l_,lr_, etc.
        call cfpgamma ! Re-calc. Coul.Log for the new distr.func.
        do k=1,ngen  ! Compute density gains and losses, and powers.
           call coefstup(k) ! To define da...df coeffs, gon(i,j), etc
           call coefmidv(da,1)
           call coefmidv(db,2)
           call coefmidv(dc,3)
           call coefmidt(dd,1)
           call coefmidt(de,2)
           call coefmidt(df,3)
           call coefwtj(k)
           call coefwti(k)
           call diagimpd(k) !->diagentr , get sgain(1:8,k)
        enddo ! k
        call achiefn(1) !-> call diaggnde: <energy>, density, energy transfer
        !  Write out distributions and meshes (controlled by namelist):
        if (n .ge. nstop) call dskout(ll)
        if (n .ge. nstop) call dsk_gr
      enddo ! ll
      !write(*,*)'tdchief-935 sum(f_),sum(f)=',sum(f_),sum(f)
      !YuP[2020-02-11] Moved the updating of profiles here,  
      ! after rescaling of f(), and after call_diaggnde :
      ! Need to know the values of currtp() [calc-ed in diaggnde]
      ! for some options in subr.profiles [curr_fit]
      !call profiles !YuP[2021-08-04] Moved even more further down,
      !after "do 30" loop.
      !Reason: When transp='enabled', the call to diaggnde is done further,
      !around line 1045. The update of time-dependent profiles should be done
      !after diaggnde. Simple test: one run with transp=disabled,
      !another - with transp=enabled, but setting d_rr-->0.
      !The results should be same (at least to a good accuracy). 
      
      
CMPIINSERT_BARRIER
      
c......................................................................
c     Integrate power densities over plasma volume
c......................................................................

      call diagentr_vol ! For CQL3D only (not for CQLP)
        
      call cpu_time(t_after_diag2) !-.-.-.-.-.-.-.-.-.-.-.-.-.

 2    continue ! if nstop.eq.0, from l 328

      if (ilend .eq. lrors-1) n_(lrors)=n_(lrors-1)


!YuP[2021-03-22] Not certain if the radial (or parallel) transport
!(about 50 lines below) should be here or few lines higher, i.e.
!in front of diagnostics part [in front of diagimpd->diagentr].
!Probably better here, otherwise f(l_) at given l_ is modified,
!and there is no corresponding power transfer diagnostics  
!occuring from spatial transport. 
c.......................................................................
c     Time advance spatial transport equation [except if transport
c     with soln_method=it3drv, not applicable since done in impavnc0].
c.......................................................................
      if (soln_method.ne.'it3drv') then
      if(transp.eq."enabled" .and. n.ge.nontran .and. n.le.nofftran)then
        if (cqlpmod .ne. "enabled") then ! CQL3D:
          ! The radial transport (tdtr...) routines follow. Store the results
          ! of the velocity space split in fvn. Then call the transport
          ! subroutine, tdtransp.
          call tdtransp
          call tdtrsavf
        else  ! (cqlpmod.eq."enabled")  CQLP:
          ! Parallel transport.
          !First compute E_n+1/2 from Poisson and modify velsou accordingly
          call wpelecf(2)
          !BH,YuP[2021-03-08] Presumably the usage of noffso(1,1) here
          ! is accidental.  Renaming it into n_sub_wp,
          ! meaning: number of smaller sub-steps withing dtreff,
          ! for shortening of parallel-transport time step:
          n_sub_wp=1 !Can make it into namelist if needed.
          dtreff=dtreff/n_sub_wp !YuP/was: dtreff/noffso(1,1)
          do in=1,n_sub_wp !YuP/was: 1,noffso(1,1)
            if (meshy.eq."fixed_mu") call wptramu
            if (meshy.ne."fixed_mu") call wptrafx
            call wpsavf
          enddo
          !write(*,*)'tdchief-986 sum(f_),sum(f)=',sum(f_),sum(f)
          !Restore the time step:
          dtreff=dtreff*n_sub_wp !YuP/was: noffso(1,1)*dtreff
        endif ! on cqlpmod
      endif  ! on transp, etc.
      endif  ! on soln_method.ne.it3drv


c.......................................................................
c     Plot flag and plot time set:
c.......................................................................

      iplt3d=0
      if (noplots.ne."enabled1") then
         do i=1,nplota
            if(n.eq.nplt3d(i)) then
               iplt3d=i
               tplt3d(i)=timet
            endif
         enddo
      endif

c.......................................................................
c     Diagnostics (if transp=enabled)
c.......................................................................
      do 30 ll=1,lrors
c......................................................................
c     determine local variables depending on flux surface (l_, iy,..)
c......................................................................
        call tdnflxs(ll)
                
        if (transp.eq."enabled") then
          call diaggnde
c..................................................
c     Compute plasma resistivity for electron runs.
c..................................................
          call restvty
c..................................................
c     Compute various diagnostics...
c..................................................
          call diag
c..................................................
c     Obtain data for time dependent plots
c..................................................
          call ntdstore
c..................................................
c     plotting logic for 3-D code...
c       (iplot set in subroutine achiefn)
c..................................................
          if ((n.ge.nstop .or. iplot.ne.0 .or. iplt3d.ne.0).and.
     +        n.ge.nontran .and. n.le.nofftran) then
     
            if(soln_method.ne.'it3drv') then !YuP[2024-07-25] added if()
            call tdtrfcop(1) !copy frn-->f, and call diaggnde
            call diaggnde
            !If soln_method='it3drv', frn (and fvn) is not used,
            !and then gn becomes 0.0 in diaggnde.
            !Once gn=0.0, the "stop" is triggered at this mpi core
            if (l_ .eq. lmdpln_) then
              !if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
              !write(*,*)'tdchief before xlndnr: n,l_',n,l_
              !endif
              do 25 k=1,ngen
                xlndnr(k,lr_)=xlndn(k,lr_)
                energyr(k,lr_)=energy(k,lr_)
                currr(k,lr_)=curr(k,lr_)/3.e9
 25           continue
            endif
            if ((adimeth.eq."enabled" .or. cqlpmod.ne."enabled") .and.
     +        n.ge.nontran .and. n.le.nofftran) then
              call tdtrfcop(2) !copy fvn-->f, and call diaggnde
              call diaggnde
              if (l_ .eq. lmdpln_) then
              !if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
              !write(*,*)'tdchief before xlndnv: n,l_',n,l_
              !endif
                do 26 k=1,ngen
                  xlndnv(k,lr_)=xlndn(k,lr_)
                  energyv(k,lr_)=energy(k,lr_)
                  currv_(k,lr_)=curr(k,lr_)/3.e9
 26             continue
              endif
            endif
            call tdtrfcop(3)  !Restore f() [Copies all ngen species]
            call diaggnde     !Loops over all species
            endif !(soln_method.ne.'it3drv') !YuP[2024-07-25] added if()
c$$$  Commenting following, in favor of dimensioning pwrrfs with l_,
c$$$    to save results calculated in impavnc0 ==> diagimpd ==> diagentr.
c$$$    BH090806
c$$$            do k=1,ngen
c$$$               call diagentr(3,k) !To get pwrrfs at current l_ surface
c$$$            enddo
            if (noplots.ne."enabled1" .and. 
     +          (iplot.ne.0 .or. n.ge.nstop) ) then
                !YuP[2017-11-27] Added n.ge.nstop, similar to achiefn
c$$$            if ( iplot.ne.0) then
              call pltmain ! and only when transp.eq."enabled" here
            endif
          endif ! n.ge.nstop ...
c.......................................................................
c     write some radially dependent netcdf output, similar to pltmain.
c.......................................................................
          call netcdfmain
        endif !End of transp="enabled" if-block
c......................................................................
c     Store updated information.
c......................................................................
        call tdtoaray
 30   continue !  ll=1,lrors
CMPIINSERT_BARRIER
 
 
!YuP[2023-05-17]commented  call profiles  !YuP[2021-08-04] Moved here from line 954.
                     !Brings namelist specified profiles up to time n
!YuP[2023-05-17] why commented:                     
! This call redefines reden() for k=kelecg and kelecm 
! [in case of imp_ne_method.eq.'ni_list']  as
!   reden= dens_ion_main*Zmain + SUM(dens_imp(kstate)*Zimp(kstate))
! So, this reden now has no relationship 
! to the distribution function f().
! By design, it could be used further for rescaling f() 
! but only if lbdry()= scale, consscal or conscalm
! (through calling subr.diagscal).
! In case of lbdry=conserv, the distribution is NOT rescaled,
! and plots show the redefined reden() which is not based on f().

c.......................................................................
cl    Compute flux surface average current and resistivity for CQLP case
c.......................................................................
      if (transp.eq."enabled" .and. cqlpmod.eq."enabled") call wpavg

c......................................................................
c     Compute conservation constant for transp="enabled"
c......................................................................

      if (transp.eq."enabled" .and. cqlpmod.ne."enabled") then
        call tdtrcon
      endif

c.......................................................................
c     compute some radial diagnostics.
c.......................................................................

c%OS  still to verify and change if cqlpmod=enabled
      call tddiag

c.......................................................................
c     Calc fusion rates, if sigmamod=enabled
c.......................................................................

        if (sigmamod.eq."enabled" .and. msig.gt.0) then
          icall="notfrst"
          call sigv(icall)
        endif ! sigmamod

c.......................................................................
c     Calc xray spectra, if softxry.ne."disabled".and.kelecg.gt.0
c     Spectra is stored in the netCDF output file at each time step.
c     Also, set plot flag  [iplt3d is set above, if plotting this step].
c.......................................................................

      if (lrzmax.lt.3 .and. softxry.ne."disabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'*******************************************'
         WRITE(*,*)'WARNING/tdchief:  SXR not computed for lrzmax.lt.3'
         WRITE(*,*)'*******************************************'
CMPIINSERT_ENDIF_RANK  
      else
         if (softxry.ne."disabled".and.kelecg.gt.0) then
            icall="notfrst"
            ilold=l_
            iplotsxr='no'
            if (iplt3d.ne.0 .or. n.eq.nstop .and. noplots.ne.'enabled1')
     +           iplotsxr='yes'
            call tdsxray(icall,iplotsxr) !Soft Xray diagnostic
            !if(n.eq.nstop)WRITE(*,*)'tdchief/aft. tdsxray rank=',mpirank
            call tdnflxs(ilold)
         endif
      endif


c.......................................................................
c     NPA synthetic diagnostic
c.......................................................................

        if (npa_diag.ne."disabled".and.niong.ne.0) then
           if (n.eq.nstop .or. npa_diag.eq.'ncdf_all') then 
              call tdnpadiag(icall)
           endif   
        endif

cBH090513:  BOB, check why no call tdnflxs after tdnpadiag??

c.......................................................................
c     plot radial information
c.......................................................................

      if ((iplt3d.ne.0 .or. n.eq.nstop).and.lrzmax .ge. 3) then
        icall="notfrst"
        if (noplots.ne."enabled1") then
          call tdpltmne
c         if equilibrium changed during run:
C%OS      if (eqmod.eq."enabled".and.n.ge.nstop) call tdplteq
            ! Plot the change in Total N of ptcles as a func of rho, for different time.
            call plt_fow_cons ! Can be used for ZOW
        endif
      endif

      if (n.eq.nstop.and.pltra.eq."enabled") then
        call pltrun
      endif

c.......................................................................
c     write radial information
c.......................................................................
      write(*,*)'tdchief/before tdoutput: iplt3d,n=',iplt3d,n
      if (iplt3d.ne.0 .or. n.eq.nstop) then
        call tdoutput(2)
      endif

      ! f(R,Z,u,theta):  ![2023-09-26] Copied from CQL3D-FOW version
      if (f4d_out.ne."disabled" .and. n.eq.nstop) call f4dwrite ![2023-09-26]

      ! f(pol.angle, vpar, mu) at fixed rho==f3d_rho :
      if (f3d_out.ne."disabled" .and. n.eq.nstop) call f3dwrite ![2023-09-26]

c.......................................................................
c     write netCDF output file, if netcdfnm.ne."disabled"
c.......................................................................
CMPIINSERT_IF_RANK_EQ_0
      if (netcdfnm.ne."disabled" .and. nstop.ne.0) then
         if (netcdfshort.eq.'lngshrtf') then
            isave=0
            do i=1,nsavet
               if(n.eq.nsave(i)) then
                  isave=i
                  tsave(i)=timet
               endif
            enddo
         endif
         call netcdfrw2(1)
         if (urfmod.eq."enabled") then
            do krf=1,mrf !YuP:04-2010: Separate data file for each wave type krf
               call netcdfrf(1,krf)
            enddo
         endif
      endif
CMPIINSERT_ENDIF_RANK
c.......................................................................
c     re-write ray data at last step in text file, 
c     if urfmod & urfwrray="enabled"
c.......................................................................

      if (urfmod.ne."disabled" .and. urfwrray.eq."enabled" 
     +  .and. n.eq.nstop .and. nstop.ne.0) call urfwrite

c.......................................................................
c     Write tavg12, sumdtr, for tavg case
c.......................................................................

      if (tavg.ne."disabled") then
         tavg12=0.d0
         do ii=1,ntavga
            tavg12=tavg12+(tavg2(ii)-tavg1(ii))
         enddo
CMPIINSERT_IF_RANK_EQ_0
         if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
         WRITE(*,*)
         WRITE(*,*)'tavg.ne.disabled: tavg12 and sumdtr should be close'
         WRITE(*,*)'at end of run: tavg12,sumdtr = ',tavg12,sumdtr
         endif
CMPIINSERT_ENDIF_RANK
      endif

c.......................................................................
c     check for call exit
c.......................................................................
      call cpu_time(t_end) !-.-.-.-.-.-.-.-.-.-.-.-.-.
CMPIINSERT_BARRIER
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,'(a,f10.3,a,f10.3,a,f10.3)')
     +                'TDCHIEF: tcpu_urfchief=',t_urf2-t_urf1,
     +                '    tcpu_netcdf=',   t_urf3-t_urf2,
     +                '    tcpu_impavnc=', t_after_soln-t_before_soln
     
         WRITE(*,'(a,2f10.3)') ' tcpu_diagnostics 1 and 2:',  
     +                          t_after_diag1-t_after_soln,
     +                          t_after_diag2-t_after_diag1
     
         WRITE(*,'(a,i5,a,f10.3)') ' Finished Time Step ======>',  n,
     +                          '     tcpu(sec.)==',  t_end-t_n_start
         WRITE(*,'(/)')
CMPIINSERT_ENDIF_RANK

CMPIINSERT_BARRIER
      call tdtloop

      if (n .lt. nstop) go to 10 !-> next time step

c.......................................................................
c     write out radial diffusion coeffs at end of program, if enabled.
c.......................................................................
      if (difus_io(1).eq."drrout") then
         call diffus_io(0)
         call diffus_io(1)
      elseif (difus_io(1).eq."drrdrout") then
         call diffus_io(0)
         call diffus_io(2)
      endif
      
      return
      end

CMPIINSERT_SUB_SEND_DATA

CMPIINSERT_SUB_SEND_DATA_AMPF
