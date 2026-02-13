c
c
      subroutine tdinitl ! called only at n=0
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     This routine initializes the arrays required by CQL3D.
c     It also controls the initialization of distribution
c     functions and quasilinear coefficients.
c..................................................................

      include 'param.h'
      include 'comm.h'

      include 'name.h'
CMPIINSERT_INCLUDE     

      character*8 icall,iplotsxr
      character*128 filenm ! template for file name with data 
      !real*8 a_new(njene)  ! working array !Not used
      real*8 ZDISTR(100) ! local, for subr. ADCDO()

      if (n.gt.0) return

c..................................................................
c     set defaults for namelisted quantities and other variables
c..................................................................

      call aindflt
      call eqindflt
      call urfindfl
      call aindflt1

c.......................................................................
c     Read namelist from previous run if rerun case. Thus namelist from
c     this run should only reset some diagnostic or numerical variables.
c     This call will position pointer in file distrfunc to next read f.
c.......................................................................

cBH110201      if (nlrestrt.ne."disabled") call tdreadf(1)
      if (nlrestrt.eq."enabled") call tdreadf(1)

c..................................................................
c     read in namelist input for CQL3D
c..................................................................
      open(unit=2,file='cqlinput',status='old')
        read(2,setup)
        read(2,trsetup)
        read(2,sousetup)
        read(2,eqsetup)
        read(2,rfsetup)
      close(2)

      if (partner.eq."selene") then
        open(unit=18,file='kcqlsel',status='old')
        read(18,2) ncount,noplots
        eqdskalt="enabled"
        close(unit=18)
      endif
 2    format(i5,a8)

c..................................................................
c     Call routine which finds electron and ion species indices.
c..................................................................

      call ainspec

c.......................................................................
c     set variables dependent on input variables
c     (May add an impurity ion, increasing species indices, if
c      iprozeff.ne.'disabled').
c.......................................................................

      call ainsetva
      
      !------- General data on impurity ions; All ionization states ----
      !YuP[2020-07-02] Earlier, this subr. was part of set_gscreen_hesslow.
      ! Now separated, for better logic flow.
      nstates=0 !Initialize; To be found in subr.set_impurity_data below.
      ! If no suitable impurity is present 
      ! (example: imp_type is set to 0 in cqlinput),
      ! then nstates remains 0 ==> Effectively, no impurities.
      if( (imp_depos_method.ne.'disabled').or.
     &    (read_data.eq.'nimrod')              )then  
          !BH,YuP[2021-01-21] added option when data is obtained from files.
          !Value of imp_type should match NIMROD data files. Select from:
          !imp_type: 1->He,2->Be,3->C,4->N,5->Ne,6->Ar,7->Xe,8->W,9->B
         call set_impurity_data !-> bnumb_imp(0:nstates),excit_enrgy(0:nstates),etc
         !Some arrays that depend on nstates 
         ! are allocated in subr.set_impurity_data.
         !Other could be allocated in ainalloc, e.g. those that depend on 
         ! combination of nmax+nstates.
      endif

c.......................................................................
c     Allocate arrays and radial transport arrays, if required
c.......................................................................

      call ainalloc
      if (ampfmod.eq."enabled" .and.cqlpmod.ne."enabled") call ampfalloc
      
      if (transp.eq."enabled" .and. cqlpmod.ne."enabled") call tdtraloc
      if (transp.eq."enabled" .and. cqlpmod.eq."enabled") call wpalloc

c.......................................................................
c     print namelists
c.......................................................................

      if (nmlstout.eq."enabled") then
         open(unit=2,file='cqlinput',delim='apostrophe',status='old')
         write(6,*)'  In tdinitl: '
         write(6,setup0)
         write(6,setup)
         write(6,trsetup)
         write(6,sousetup)
         write(6,eqsetup)
         write(6,rfsetup)
         close(2)
      elseif (nmlstout.eq."trnscrib") then
         write(6,*)'  In tdinitl: '
         call ain_transcribe("cqlinput")
      else
         write(6,*)
         write(6,*) 'mnemonic = ',mnemonic
         write(6,*)
      endif

c..................................................................
c     Call the initialization routines for the appended modules..
c..................................................................
      call eqinitl
      call urfinitl ! mrfn= is set here also
      call frinitl
      open(unit=2,file='cqlinput',delim='apostrophe',status='old')
      call frset(lrz,noplots,nmlstout,ngen,kfrsou(1:nbeams))   ! Uses unit 2
      close(2)

      if (machine .ne. "toroidal") call tdwrng(1)
      if (lrzdiff.eq."enabled" .and. frmodp.eq."enabled") 
     +  call diagwrng(18)

c.....................................................................
c     This routine initializes the normalized theta mesh that is used
c     if the meshes on the various flux surfaces are linked.
c.....................................................................

      call tdtrmuy
      write(*,*)'aft. tdtrmuy: iymax=',iymax

c.....................................................................
c     Call routine which initializes certain arrays using input
c     data. Also initialize the radial (rz) mesh and some plasma
c     parameters arrays.
c.....................................................................

      if(lrzmax.gt.1) call tdxinitl

c.....................................................................
c     Call routines to initialize any time-dependent (parabolic,...)   
c     profiles, and to set them up in place of profiles 
c     otherwise set up in tdxinitl.
c.....................................................................

      if(nbctime.gt.0 .and. iprozeff.eq."curr_fit")then
           !If nbctime.gt.0, and this is a restart case, then for
           !iprozeff.eq."curr_fit, need to restore the zeff(ll)
           !from end of the previous run (from distrfunc.nc).
           if(nlrestrt.eq."ncdfdist")then
              call tdreadf(3)  !kopt=3 restores zeff profile
           endif  !On nlrestrt.eq."ncdfdist"
           !YuP[2020-02-11] Moved this part from profiles to tdinitl
      endif

c.....................................................................
c     Determine mesh normalization constant vnorm.
c.....................................................................

      call ainvnorm

c.......................................................................
cl    1.1 Loop over all radial mesh points lr=1,..,lrzmax to determine
c     the plasma and equilibrium parameters on the full radial mesh
c     Also determine the mesh parallel to the magnetic field line
c.......................................................................

      if (cqlpmod.eq."enabled" .and. numclas.eq.1 .and. ls.eq.lsmax)then
        lz=lz/2+1
        lsmax=lsmax/2+1
        ls=ls/2+1
      endif

      do 110 ll=lrzmax,1,-1

c.......................................................................
c     Sets up the parameters on each lrzmax flux surface. Thus does not
c     define lr_=lrindx(l_), but lr_=l_, l_=1,lrzmax
c     (Not like in tdnflxs)
c......................................................................

        l_=ll
        lr_=ll
c     these two indices should not be used in loop 110 => set to -1, to detect
c     out of bound errors
        lmdpln_=-1
        indxlr_=-1

c..................................................................
c     Call an initialization routine which determines flux surface
c     geometry and magnetic field structure.
c..................................................................

        call aingeom !-> eqcoord-> eqfndpsi-> eqorbit-> trace flux surf.
             ! solr(l,lr_), solz(l,lr_) are R,Z coords. of flux surface
c.......................................................................
c     Initialize mesh along the magnetic field line, as well as
c     density and temperature profiles if cqlpmod=enabled
c.......................................................................

        call micxiniz

c..................................................................
c     Copy some radial and non-time dependent diagnostic quantities
c..................................................................

        call tdtoarad

 110  continue ! ll=lrzmax,1,-1

CMPIINSERT_IF_RANK_EQ_0      
      do ir=1,lrz
         WRITE(*,'(a,i3,4e13.5)')'ir,rya,rpcon,rmcon,equilpsi=',
     +                ir,rya(ir),rpcon(ir),rmcon(ir),equilpsi(ir)
      enddo
       !pause
CMPIINSERT_ENDIF_RANK
             
 
      !-----------------------------------------------------------------   
      !BH,YuP[2021-01-21] Option to read data files.
      !(Initial purpose - coupling with NIMROD. 
      ! Can be extended to coupling with other codes.)
      if(read_data.eq.'nimrod')then
        kopt=1 ! Setup radial grid ryain() based on R-grid in data files,
        !also fill-up spline-t arrays like enein_t(), tein_t(), etc.
        !This part should be called AFTER set_impurity_data, because
        !we need to know the number nstates for a given impurity type.
        jtmm=0 ! to be incremented
        do jtm=1,nbctime !nbctime was set to number of data files in ainsetva,
          !which is the number of time slices in case of read_data='nimrod'
          call read_data_files(read_data_filenames(jtm),jtm,
     &                            ipresent,t_data,kopt)
          if(ipresent.eq.1)then ! File is found
            jtmm=jtmm+1
            bctime(jtmm)=t_data ![sec], from reading 1st line in data file
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'Done jtm,jtmm,read_data_filenames=',
     &      jtm,jtmm,
     &      TRIM(read_data_filenames(jtmm)),
     &      '  bctime[s]=',bctime(jtmm), '  E(r=0)=',elecin_t(1,jtm)
            WRITE(*,*)'  '
CMPIINSERT_ENDIF_RANK    
          endif     
        enddo ! jtm
      endif ! read_data
      !-----------------------------------------------------------------   

      !YuP[2021-02-08] Moved calls to read_data_files and to profiles
      !here, after call to aingeom->eqcoord related subroutines.
      !For subr.read_data_files, need to have arrays related to geometry
      !of flux surfaces, because it evaluates resistivities.
      !And subr.profiles should be called after read_data_files.
      !if(nbctime.gt.0)then !YuP[2019-09-18] This logic is inside subr.
         call profiles
      !endif


c.......................................................................
c     Determine equilibrium parameters at lower half of cross-section
c     on lrindx(1) flux surface
c.......................................................................

      if (cqlpmod.eq."enabled" .and. numclas.eq.1 .and. ls.eq.lsmax)then
        lz=2*(lz-1)
        lsmax=2*(lsmax-1)
        ls=2*(ls-1)
        write(*,*)'tdinitl before wploweq: lz,ls,lsmax=', lz,ls,lsmax
        call wploweq
      endif

c.....................................................................
c     Redefines mu-mesh at midplane if needed
c.....................................................................

      if (cqlpmod.eq."enabled" .and. meshy.eq."fixed_mu" .and.
     .  tfac.lt.0.0) call wptrmuy
      !write(*,*)'tdinitl aft. wptrmuy: iy,mun=', iy,mun(:)

c.......................................................................
cl    1.1.2 Initialize some plasma parameters on entire lrzmax mesh
c.......................................................................

      call ainpla

c.....................................................................
cl    1.1.3 First Loop over spatial variable index for which the equations
c     will be solved to compute y-mesh on all l_ indices
c     (this takes micxinit out of ainitial subroutine)
c.....................................................................
      do 113 ll=lrors,1,-1

c......................................................................
c     determine local variables depending on variable index l_
c......................................................................

        call tdnflxs(ll)

c.......................................................................
c     call a routine to determine meshes y, x and related quantities
c.......................................................................

        call micxinit

 113  continue

      ieq_tot=0
      do ll=1,lrors
         call tdnflxs(ll) !YuP[2021-03-04] Added (just in case)
         ieq_tot=ieq_tot+inewjx_(ll) ! inewjx_() is defined in micxinit
         if (ll.eq.1) then
            ieq_(ll)=1
         else  ! Eqn no. at beginning of each flux surface:
            ieq_(ll)=ieq_(ll-1)+inewjx_(ll-1)  
         endif
      enddo
      ieq_(lrors+1)=ieq_tot



c.......................................................................
c     adapt dyp5(iyh) and dy(iyh) such that sum(cynt2) over i is constant
c     on all l_ surfaces
c.......................................................................

      if (nchgdy .eq. 1) then
        do 114 ll=1,lrors
          call tdnflxs(ll)
          call wpchgdy ! called for each l_
 114    continue
      endif
      write(*,*)'iymax=',iymax
      ! YuP[Sept-2014] moved this part outside of ainitial: 
      ! needed for proper work of FOW-logic.
      do ll=lrors,1,-1
        call tdnflxs(ll)
        call micxinil ! integration coefficients, mostly for cfpcoefn:
        ! gamma,alphan,cog,cosz,zmaxpsi (uses bbpsi,solrz from micxiniz)
        !if(cqlpmod.eq."enabled")then !YuP[2021-04-13] test:
          !set tau-->1.0 and dtau-->1.0 (instead of getting them from baviorbto)
          !call bcast(tau(1,lr_),1.d0,iymax)        !YuP[2021-04-13] ok?
          !call bcast(dtau(1,1,lr_),1.d0,iymax*lz)  !YuP[2021-04-13] ok?
          !For CQLP, tau and dtau should not be needed.
          !From tests, resetting them to 1.0 does not affect the solution,
          !but it does change some diagnostics: 
          !- parallel distribution (see subr. fle_pol),
          !- printout "Connor", "One2/spt(Z)", etc, probably through <1/R>
          !  and similar BA values from subr.bavgmax. 
          !  (Not "Hinton" resistivity)
        !else
        if (l_ .eq. lmdpln_) then
           ! calculate tau()==v*tauB_ZOW [also dtau(), vptb()]
           !write(*,*)' tdinitl--> baviorbto   ll,l_=',ll,l_
           !In CQLP, baviorbto is called with l_=1 only
           if(taunew.eq."enabled") then
              call baviorbt ! must be called after micxiniz
           else
              call baviorbto
           endif
        endif
        !endif
      enddo

c....................................................................
c     Set some more radial mesh quantities.
      call tdrmshst !must be called AFTER: aingeom,tdtoarad,micxinit,micxinil
      write(*,*)'aft tdrmshst 1'
c....................................................................

cBH160530
c     Must be called after micxiniz and baviorbt/baviorbto:
cYuP      if (lossmode(1).eq.'simplban' .or. lossmode(1).eq.'simplbn1'
cYuP      +     .or.ndeltarho.ne.'disabled') 
cYuP      +     call deltar
cYuP[07-2017] exclude simplban  - deltar is not needed.
      if(ndeltarho.ne."disabled")then 
         !YuP[2020-10-20] Removed lossmode from if() below. 
         !deltarho-related arrays are not used in sub.losscone, for now.
!YuP      if (lossmode(1).eq.'simplbn1'.or.ndeltarho.ne.'disabled') 
          call deltar
      endif


      !------------------------- contribution from partially ionized ions -----
      !YuP[2019-07-29] Set gscreen(p) function (of normalized momentum p) 
      !that describes the effect of partially screened ions 
      !on enhanced scattering of electrons (fast electrons can "probe" 
      !the inner structure of a partially ionized ion).
      !Also, set hbethe(p) function that describes the slowing down
      !of free electron on bound electrons in partially ionized ion
      !or neutral atom.
      !See Hesslow et al, JPP-2018,vol.84, Eq.(2.25) and (2.31).
      if(gamafac.eq."hesslow" .and. kelecg.eq.1)then
         call set_gscreen_hesslow !(imp_type is in namelist)
         !This subroutine requires x(j) mesh, gamma(j) values,
         !so it should be called AFTER micxinil.
         !It uses data for different chemical elements,
         !with different ionization states.
      endif !(gamafac.eq."hesslow" .and. kelecg.eq.1)


c.....................................................................
cl    1.2 Loop over spatial variable index for which the equations will be
c     solved. Compute initial distributions and sources etc.
c.....................................................................

      do 120 ll=lrors,1,-1

c......................................................................
c     determine local variables depending on variable index l_
c......................................................................

        call tdnflxs(ll)

c.....................................................................
c     transfer input data from CQL3d to CQL
c.....................................................................

        call tdstin

c.....................................................................
c     Call spatial variable "surface" initialization routine
c..................................................................

        call ainitial ! mrfn= is set here also (in vlf or vlh)

c..................................................................
c     Call a routine to initialize some urf module routines.
c..................................................................

c     Tried following, but gives problem of code stopping?? [BH041111].
c        if (cqlpmod.ne."enabled".and.urfmod.ne."disabled") call urfsetup
        if (cqlpmod.ne."enabled") call urfsetup ! mrfn= is set here also

c..................................................................
c     Copy some diagnostic quantities
c..................................................................

        call tdtoaray

 120  continue
c----------------------------------------------------------------------

c..................................................................
c     If running with the Brambilla code, create and dump and eqdsk
c     file for Brambilla ray tracing code to read in.
c..................................................................

      if (partner.eq."bramb") then
        call tdeqdsk
      endif

c....................................................................
c     Set some more radial mesh quantities.
c....................................................................

      call tdrmshst

c..................................................................
c     Call first-order radial orbit-shift module
c..................................................................

cBH160530      if (ndeltarho.ne."disabled") call deltar
         
c....................................................................
c     Call routines to set up radial integration coefficients, radial
c     diffusion and advection coefficients, certain flags and the radial
c     Chang-Cooper weights.
c....................................................................

      if (transp.eq."enabled") then
        if (cqlpmod .ne. "enabled") then
c     warning: these routines use lrors, not lrzmax
          call tdtrvint ! CQL3D only (not CQLP)

c         Obtain radial diffusion d_rr, and optionally
c         pinch velocity d_r, from file.  In this case, the
c         d_rr and d_r can be multiplied by a time-dependent
c         scale factor at point of use, in trans.h and tdtravct.
c         The read in d_rr/d_r won't be changed.
          if (difus_io(1).eq."drrin") then
             call diffus_io(3) !Reads d_rr
             if (n_d_rr.ne.0) then
                !Getting time-dep scale factor at t=0
                do k=1,n_d_rr   !n_d_rr obtained in diffus_io
                   drrt(k)=difus_io_scale(k,1)
                enddo
             endif
          elseif(difus_io(1).eq."drrdrin") then
             pinch="disabled"   !Since getting d_r from file
             call diffus_io(4) !Reads d_rr and d_r
             if (n_d_rr.ne.0) then
                !Getting time-dep scale factor at t=0
                do k=1,n_d_rr   !n_d_rr obtained in diffus_io
                   drrt(k)=difus_io_scale(k,1)
                   drt(k)=difus_io_scale(k,2)
                enddo
             endif
          else
             call tdtrdfus ! CQL3D only (not CQLP)
          endif

          call tdtrflg ! CQL3D only (not CQLP)
          call tdtrwtl ! Both CQL3D and CQLP (Sets all dl(,,,)=0.5)
        else ! (cqlpmod.eq."enabled")
          call wpinitl
        endif
      endif
      call tddiag

c%OS  
c     check some mesh values
      call wpmshchk  !disabled by somebody
c%OS  


c.....................................................................
c     Set up sigma-v calculation
c.....................................................................

      if (sigmamod.eq."enabled")  then
        icall="first"
        call sigv(icall)
      endif

c.....................................................................
c     Set up soft X-ray and NPA diagnostic.
c.....................................................................

      if (lrzmax .ge. 3) then
c**bh if (softxry.eq."enabled".and.kelecg.gt.0.and.eqmod.ne."enabled")
        if (softxry.ne."disabled".and.kelecg.gt.0)
     1    then
          icall="first"
          if (noplots.ne."enabled1") then
             iplotsxr='yes'
          else
             iplotsxr='no'
          endif
          call tdsxray(icall,iplotsxr)
        endif
        if (npa_diag.ne."disabled".and.niong.ne.0) then
           icall="first"
cBH_to skip first call, per Vincent Tang:     call tdnpadiag(icall)
cBH100816:  No.      Also, actually for npa, icall has no effect,
cBH100816:  as calculation as setup repeated each call to tdnpadiag.
           call tdnpadiag(icall)
        endif
      endif

c..................................................................
c     Call the neutral beam source module
c..................................................................

cBH171014:  Enable calls for multiple beam species.
      if (frmodp.eq."enabled") then !YuP[2025-12-12] added if
CMPIINSERT_IF_RANK_EQ_0      
      WRITE(*,*)'tdinitl:call frnfreya, n=',n,'time=',timet
CMPIINSERT_ENDIF_RANK
      ! nbeams,nbeamsp= 0,0   here
      call frnfreya(frmodp,fr_gyrop,beamplsep,beamponp,beampoffp,
     &              nbeamsp,hibrzp,mfm1p,noplots,kfrsou,src_nbi_ep)
      !nbeamsp is one of outputs: got value from nbeams (cqlinput variable)
      !Now nbeamsp is saved into comm.h
CMPIINSERT_IF_RANK_EQ_0      
      WRITE(*,*)'tdinitl/aft.frnfreya: mfm1, nbeamsp,src_nbi_ep=',
     &  mfm1p,nbeamsp,src_nbi_ep
      WRITE(*,*)'tdinitl/aft.frnfreya: kfrsou(1:nbeamsp)=',
     & kfrsou(1:nbeamsp)
CMPIINSERT_ENDIF_RANK

      endif !(frmodp.eq."enabled") !YuP[2025-12-12] added if
c
c     Initialize ibeampon/ibeamponp (present and previous time step)
      if (frmodp.eq."enabled") then
         ibeampon=0   !Reset in tdchief, for pulse beam
         ibeamponp=1  !Indicates beam source calculated
      else
         ibeampon=0
         ibeamponp=0
      endif

c..................................................................
c     Plot out initial radial profiles
c..................................................................
      if (noplots.ne."enabled1") call tdpltmne

c..................................................................
c     Read RF diffusion coefficient data for multiple flux
c     surfaces if rdcmod="format1" or "aorsa" or "format2"
c..................................................................

      if (rdcmod.eq."format1" .or. rdcmod.eq."aorsa" 
     +                        .or. rdcmod.eq."format2") then
        call rdc_multi
      endif

c..................................................................
c     print initial output and plot equilibrium surfaces
c..................................................................
      call tdoutput(1)
      if (noplots.ne."enabled1") then
      if((eqmod.eq."enabled") .or.
     &   (eqmod.eq."disabled" .and. psimodel.eq."spline") )then
      !In case of eqmod='disabled', solr() and solz() can be empty,
      ! so tdplteq would fail.
      !Actually, with psimodel='spline', they are set now !YuP[2020-01-29]      
         call tdplteq(0) !'0' means no rays (Here, no data on rays yet)
      endif
      endif
c$$$      if (eqmod .eq. "enabled") call tdplteq

!------------------------- contribution from partially ionized ions -----
      if((imp_depos_method.ne.'disabled').and.(kelecg.eq.1))then
         !YuP[2020-06-24] Changed (gamafac.eq."hesslow") to (imp_depos_method.ne.'disabled')
         !   [a more general logic]
         !YuP[2020-06-22]call set_gscreen_hesslow !(imp_type is in namelist)
         !YuP[2020-06-22] Moved this call to set_gscreen_hesslow,
         !about 300 lines up, BEFORE call ainitial(which calls diaggnde),
         !because diaggnde calculates zeff() 
         !and zeff4() (at each n, including n=0).
         !-------------------------------------------------------
         !--- Setup arrays/table for calculation of <Z> and <Z^2> 
         !    for a given impurity (imp_type)
         kopt=0 !Read data file, find value of nstates, setup arrays/table
         ! Note: INPUT argument imp_type is obtained from namelist.
         temp_Te=0.d0 ! Not needed during setup
         dens_nD0=0.d0 ! Not needed during setup
         dens_ne=0.d0 ! Not needed during setup
         tau_r=0.d0 ! Not needed during setup
         if(adpak.eq.'enabled')then
            call set_get_ADPAK(kopt,imp_type, 
     &                         temp_Te,dens_nD0,dens_ne,tau_r,
     &                         z1av,z2av) 
           !(INPUT: kopt,imp_type,temp_Te,dens_nD0,dens_ne,tau_r)
         endif ! if adpak.ne.enabled, subr.ADCDO will be used.
         
         !------------- FOR TESTS ONLY (example of usage):
!         kopt=1 ! To get <Z> and <Z^2> 
!         dens_ne=1.d13 !Should get from reden(kelecg,lr_)
!         dens_nD0= 7.d-3*dens_ne !Should get dens_nD0 from namelist var
!         !Tested: 2.0*dens_ne ! above 1.0 is treated as 1.0 case.
!         !1.d-8*dens_ne , 1.d-7*dens_ne , 7.d-3*dens_ne All tested ok
!         !(run, copy/paste the printout of 'temp_Te,z1av,z2av' below
!         ! into Matlab script test_Z_vs_Te.m, make plots)
!         tau_r=2.154d12/dens_ne !Should get tau_r(rho) from namelist.
!         do itemp=1,100 ! 100 points in log10 scale for Te values
!           temp_Te= log10(0.4d-3) 
!     &             +(itemp-1)*(log10(10.0)-log10(0.4d-3))/(100-1) ! kev
!           temp_Te= 10.0**temp_Te 
!           call set_get_ADPAK(kopt,imp_type, 
!     &                         temp_Te,dens_nD0,dens_ne,tau_r,
!     &                         z1av,z2av) 
!           !(INPUT: kopt,imp_type,temp_Te,dens_nD0,dens_ne,tau_r)
!           !WRITE(*,*) temp_Te,z1av,z2av
!           iZatom=INT(bnumb_imp(nstates)) !should be integer as an argument to this subr.
!           !Could simply use iZatom=nstates
!           call get_distr_charge_states(iZatom,z1av,z2av, fz) 
!           !          INPUT: iZatom,z1av,z2av   OUTPUT: fz(0:Zatom)
!           ! Alternatively: 
!           INUCZ=INT(bnumb_imp(nstates)) !atom nuclear charge
!           ZTE= temp_Te*1d3 ! ZTE is Te in eV
!           ZNE= dens_ne  ! electron density in cm-3
!           ZNA= dens_nD0 ! density of hydrogen atoms in cm-3
!           ZTA= temp_Te*1d3 !T= Timp/Mimp +Ta/Ma ~Ta/Ma. For D: Ti/2 (Ma=2)
!           !ZTA is relative temperature for charge exchange of ions with
!           !hydrogen atoms which is Tz/Mz+Th/Mh
!           CALL ADCDO(INUCZ,ZTE,ZNE,ZTA,ZNA,
!     &           ZS, ZS2, EZRAD, EZLOSS, ZDISTR)
!           !output:
!           ! ZS is <z>
!           ! ZS2 is <z^2>
!           ! EZRAD is the radiation power W*cm3
!           ! EZLOSS is the ionization energy losses, W*cm3
!           ! ZDISTR is array for distribution function of excited states with
!           ! sum(ZDISTR)=1
!           ! For printout/verification:
!           sum_fz_Z1=0.d0
!           sum_fz_Z2=0.d0
!           sum_ZDISTR_Z1=0.d0
!           sum_ZDISTR_Z2=0.d0
!           do iz=0,iZatom
!             !write(*,*) iz,fz(iz),ZDISTR(iz+1)
!             sum_fz_Z1= sum_fz_Z1 +fz(iz)*iz
!             sum_fz_Z2= sum_fz_Z2 +fz(iz)*iz*iz
!             sum_ZDISTR_Z1= sum_ZDISTR_Z1 +ZDISTR(iz+1)*iz
!             sum_ZDISTR_Z2= sum_ZDISTR_Z2 +ZDISTR(iz+1)*iz*iz
!           enddo
!           !write(*,*)'Above: z,fz(z),ZDISTR(z) over all charge states z'
!           write(*,*)'Te_keV, ZTE,ZNE,ZTA,ZNA=',temp_Te,ZTE,ZNE,ZTA,ZNA
!           write(*,*)'z1av, ZS =',z1av,ZS
!           !write(*,*)'z2av, ZS2=',z2av,ZS2
!           write(*,*)'sum_fz_Z1, sum_ZDISTR_Z1=',sum_fz_Z1,sum_ZDISTR_Z1
!           !write(*,*)'sum_fz_Z2, sum_ZDISTR_Z2=',sum_fz_Z2,sum_ZDISTR_Z2
!         enddo ! itemp
!         !WRITE(*,*)'Above: Te[keV], z1av, z2av'
!         pause !----------- TESTS ONLY (done)
         
         !---> For pellet propagation/ablation model:
         if(imp_depos_method.eq.'pellet') then
           write(*,*)'--- CALLING set_get_pellet(kopt=0...) ---'
           kopt=0 ! to setup some arrays and calc. pellet_Cablation
           pellet_rho=2.0 !For plots, will be found at t>pellet_tstart.
           pellet_Mrem=pellet_M0 !For plots, remaining mass; here: t=0
           temp_t0(kelecg,1:lrz)=temp(kelecg,1:lrz) ! before pellet starts
           reden_t0(kelecg,1:lrz)=reden(kelecg,1:lrz) ! before pellet starts
           temp_wk(1:lrz)=temp(kelecg,1:lrz)
           reden_wk(1:lrz)=reden(kelecg,1:lrz)
           call set_get_pellet(kopt, timet, lrz, rmag, 
     &           rpcon(1:lrz), rmcon(1:lrz), dvol(1:lrz), 
     &           temp_wk(1:lrz), reden_wk(1:lrz),
     &           pellet_V, pellet_M0, pellet_Rstart, pellet_tstart,
     &           pellet_rcloud1, pellet_rcloud2, pellet_rp0,
     &           pellet_pn, pellet_pt, pellet_pm,
     &           ipellet_method, pellet_fract_rem, 
     &           ipellet_iter_max, pellet_iter_accur, pellet_Cablation,
     &              Gablation,dMpellet_sum, dMpellet_dvol_sum(1:lrz) )
           ! To calculate particle density, use  convert= Avogadro/atw 
           ! where atw is the atomic weight (40 g/mol for Argon), or simply
           ! (Avogadro*proton)/(atw*proton) = 1.0/fmass_imp
           ! where fmass_imp=atw*proton (in comm.h)
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)'--- just after set_get_pellet(kopt=0...) ---'
           WRITE(*,*)'timet=',timet, ' Raxis,rmag=',Raxis,rmag
           WRITE(*,*)'pellet_Cablation=',pellet_Cablation
CMPIINSERT_ENDIF_RANK
           ! Note: At given instant t=timet, pellet is at 
           ! Rpellet(t)= pellet_Rstart -pellet_V*(timet-pellet_tstart)
           ! It means that a given flux surface at R=rpcon(lr) 
           ! is reached by pellet at instant
           ! t= pellet_tstart +(pellet_Rstart-rpcon(lr))/pellet_V
           ! We use this t as the onset time for the 
           ! exp(-(t-tstart)/tau) temperature drop, 
           ! i.e. as the value for 
           ! temp_expt_tstart(lr) in case of iprote='prb-expt' option.
           dtdelay= 0.d0 !1.0*((rpcon(lrz)-rmag)/lrz)/pellet_V
           do ll=1,lrz
              if(temp_expt_tstart(ll).eq.0.d0)then
               temp_expt_tstart(ll)= pellet_tstart +dtdelay+
     &                    (pellet_Rstart-rpcon(ll))/pellet_V
               temp_expt_T1(:,ll)=temp(:,ll) ! save the initial T 
               !for the start of fast drop. It will be reset in profiles.f
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'ll,temp_expt_tstart(ll)',
     &         ll,temp_expt_tstart(ll)      
CMPIINSERT_ENDIF_RANK
              endif   
           enddo
           ! Note that we DO NOT consider the region of R 
           ! inboard of magnetic axis (rmcon(lr) values).
           ! We assume that as soon as the pellet reaches 
           ! R=rpcon(lr) radius, the temperature decay happens
           ! at entire surface at once, i.e. at rpcon(lr) 
           ! and rmcon(lr) points the values of temp_expt_tstart(lr)
           ! are same (same flux surface).
         endif ! imp_depos_method.eq.'pellet'
        
        !--- FOR TEST ONLY ------------------------
        ! Checked: With a very small time step 
        !(230 time steps for pellet to cross full R range),
        ! and with a very large time step (one step to cross plasma)
        ! the results are same - dMpellet_dvol_sum() as a func. of rho
        ! is same.
!         kopt=1 !To propagate pellet, and calc. dMpellet_dvol_sum(1:lrz)
!         tpellet= pellet_Rstart/pellet_V !estimate t of flying through plasma
!         write(*,*)'tpellet=',tpellet, ' dtr=',dtr
!         npell= int(tpellet/(dtr*100))
!         npell=max(npell,1)
!         !do lr=1,lrz
!         !  write(*,*) rpcon(lr)
!         !enddo
!         !write(*,*)'npell=', npell
!         !pause
!         do it=1,npell ! test: step in time, find dMpellet_dvol_sum()
!            t_test= it*dtr*100
!            call set_get_pellet(kopt, t_test, lrz, rmag, 
!     &           rpcon(1:lrz), rmcon(1:lrz), dvol(1:lrz), 
!     &           temp(kelecg,1:lrz), reden(kelecg,1:lrz),
!     &           pellet_V, pellet_M0, pellet_Rstart, pellet_tstart,
!     &           pellet_rcloud1, pellet_rcloud2, pellet_rp0,
!     &           pellet_pn, pellet_pt, pellet_pm,
!     &           ipellet_method, pellet_fract_rem, 
!     &           ipellet_iter_max, pellet_iter_accur, pellet_Cablation,
!     &              Gablation,dMpellet_sum, dMpellet_dvol_sum(1:lrz) ) 
!           dens_imp_allstates(1:lrz)= dMpellet_dvol_sum(1:lrz)/fmass_imp
!           do lr=1,lrz
!             write(*,*) dens_imp_allstates(lr)
!           enddo
!           write(*,*)'ABOVE:  dens_imp_allstates(lr) at it=',it
!         enddo ! it 
!         pause
        !----END OF TEST --------------------------
      endif ! (imp_depos_method.ne.'disabled') .and. kelecg.eq.1

!------------------------- contribution from partially ionized ions -----

      return
      end
