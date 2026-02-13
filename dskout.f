c
c
      subroutine dskout(ll)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     At the end of the run, this routine at the option of the user 
c     writes out to disk file 'idskf' the namelist input deck 
c     and various computed quantities, including the distn functions.
c     Also, write out to disk file 'idskrf' rf diffusion coefficients,
c     and related quantities.
c..................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      character*80 line
      
CMPIINSERT_IF_RANK_NE_0_RETURN

      if (lrzmax.le.1) then
        if((idskf.eq. "disabled".and.idskrf.eq."disabled")
     1    . or. n.ne.nstop+1)  return
      else
        if((idskf.eq. "disabled".and.idskrf.eq."disabled")
     1    . or. n.ne.nstop)  return
      endif
      if(ll.gt.1)  go to 4
      if(idskf.ne."disabled")
     1  open(unit=4,file=idskf,delim='apostrophe',status='unknown')
      if(idskrf.ne."disabled")
     1  open(unit=5,file=idskrf,delim='apostrophe',status='unknown')
ccc      close(unit=2) ! YuP: Why here?
      ilen=0

c..................................................................
c     The input namelist file is transcribed onto the beginning
c     of file idskf and/or idskrf
c..................................................................

      open(unit=2,file='cqlinput',delim='apostrophe',status='old')
 1    read(2,1003) line
      if (line(1:3).eq."end") go to 3
      if(idskf.ne."disabled") write(4,1003) line
      if(idskrf.ne."disabled") write(5,1003) line
      go to 1
 3    close(unit=2)
      if(idskf.ne."disabled") 
     +     write(4,1006) '***Begin computed output***'
      if(idskrf.ne."disabled") 
     +     write(5,1006) '***Begin computed output***'
 4    continue 

c..................................................................
c     In the following disk write to file named idskf:
c     This subroutine is called to write data for each FP'd
c          flux surface.
c     ll=  FP flux surface number 
c          (ll=1:lrors, lrors.le.lrzmax, see cqlinput_help))
c          (lrindx(ll) gives flux surface number on the full
c                     radial mesh.(lrindx(ll)=ll if lrors=lrzmax,
c                     and using cql3d mode(cqlpmod="disabled"))).
c     iy,jx= dimensions in theta and u(momentum/mass)
c           (In the case where iy varies with ll, iy(1) will be greatest.)
c     lrors= number of flux surfaces FP'd.
c     lrzmax= number of flux surfaces, including any not FP'd.
c     x = momentum-per-mass(nomalized to maximum 1.)
c         at  each flux surface lrindx(ll)
c     y = theta(radians) mesh at  each flux surface lrindx(ll)
c     rovera= normalized radius (ll)
c             (rho, see Hinton and Haseltine for non-circ).
c             (generally ~sqrt(tor. flux), other coords available.)
c     elecfld = toroidal electric field (volts/cm)
c     bthr(lr_) - the poloidal magnetic field at theta-poloidal = pi/2.
c     btoru(lr_) - the toroidal magnetic field at the same position. 
c     bthr0(lr_), btor0(lr_), are the poloidal and  toroidal
c         magnetic fields at the outer midplane of the flux surface. 
c     reden= electron density at minimum B point on flux surface.
c     temp= initial electron temperature (keV)
c     radmin= plasma minor radius (cms).
c     vnorm= normalization momentum-per-mass (maximum on grid) (cm/sec)
c     vmaxdvt= vnorm/(temp/mass)**0.5
c     eovedd= electric  field, normalized to Driecer field 
c             (calc'd in sub restvty).
c
c     distribution function normalised so that
c         integral( (dx)**3 f) = density at minumum B point.
c
c..................................................................

      if(idskf.ne."disabled") then
        write(4,1004)  ll, iy,jx,lrors,lrzmax,ngen
        write(4,1004)  itl,itu
        write(4,1005)  (x(j),j=1,jx)
        write(4,1005)  (y(i,ll),i=1,iy)
        do 1000 k=1,ngen
          write(4,1005)  bnumb(k),fmass(k)
          write(4,1005)  rovera(lrindx(ll)),elecfld(lrindx(ll)),
     +                   bthr(lrindx(ll)),btoru(lrindx(ll))
          write(4,1005)  bthr0(lrindx(ll)),btor0(lrindx(ll)),
     +                   reden(k,lrindx(ll)),temp(k,lrindx(ll))
          vmaxdvt=vnorm/(4.19e7*sqrt(2.*1000.*temp(k,lrindx(ll))))
          write(4,1005)  radmin,vnorm,vmaxdvt,eovedd
          write(4,1005)  ((f(i,j,k,ll),i=1,iy),j=1,jx)
 1000   continue
      endif


c..................................................................
c     In the following disk write:
c     vptb is lambda=u_parallel_o * tau_bounce.
c     (tau_bounce= portion of bounce time from the outer equatorial
c     plane to the bounce point or the inner equatorial plane).
c     temp1 is the bounce averaged uu-diffusion coefficient (cgs units).
c..................................................................
      if(idskrf.ne."disabled") then
        write(5,1004)  ll,iy,jx,lrors,lrzmax
        write(5,1005)  (y(i,ll),i=1,iy)
        write(5,1005)  (x(j),j=1,jx)
        write(5,1005)  rovera(lrindx(ll)),vnorm
        write(5,1005)  (vptb(i,lrindx(ll)),i=1,iy)
        vn2=vnorm*vnorm
        do 10 k=1,mrfn
          vmaxdvt=vnorm/(4.19e7*sqrt(2.*1000.*temp(k,lrindx(ll))))
          write(5,1005) reden(k,lrindx(ll)),temp(k,lrindx(ll)),vmaxdvt
          do 11 i=1,iy
 11       temp1(i,1)=0.0
          do 12 j=2,jx
            x2=x(j)**2
            do 13 i=1,iy
 13         temp1(i,j)=urfb(i,j,indxlr(lrindx(ll)),k)*vn2/
     /          (vptb(i,lrindx(ll))*x2)
 12       continue
          write(5,1005) ((temp1(i,j),i=1,iy),j=1,jx)
 10     continue
      endif
 1003 format(a80)
 1004 format(16i5)
 1005 format(5e16.8)
 1006 format(a27)
      if(idskf.ne."disabled".and.ll.eq.lrors)  close(unit=4)
      if(idskrf.ne."disabled".and.ll.eq.lrors)  close(unit=5)
      return
      end subroutine dskout

!=======================================================================
!=======================================================================


      subroutine read_data_files(filenm,jtm,ipresent,t_data,kopt)
      !YuP[2021-01-21] read data files.  Revised [2021-02-08].
      !(Initial purpose - coupling with NIMROD. 
      ! Can be extended to coupling with other codes.)
      ! For coupling with NIMROD, 
      ! each file contains data at one time slice.
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h' !contains enein_t(njenea,ntotala,nbctimea),
               !tein_t(njenea,nbctimea), etc.
               !Also added dbb2in_t(:,:) !(njene,nbctime)
               !Also added ryain_t(:,:)  !(njene,nbctime) Time-dep. ryain array
CMPIINSERT_INCLUDE

      !INPUT: filenm, jtm, kopt;  Also nstates (from comm.h)
      !OUTPUT: ipresent (=1 if the filenm is present, =0 otherwise)
      !t_data= t[sec], obtained from reading 1st line in given data file.
      !Other output: the data that is read from file is copied into
      ! CQL3D arrays like enein_t(), tein_t(), etc.
  
      character*128,INTENT(IN) :: filenm
      integer,INTENT(IN) :: jtm !Index for data file
      integer,INTENT(IN) :: kopt
      integer,INTENT(OUT):: ipresent
      real*8,INTENT(OUT) :: t_data
      integer kode,istat,iunit ! local
      integer idum !local
      real*8 dum1,dum2,dum3, pi ! local
      character*8 cdum !local
      integer njene_data !local
      integer irow !local
      character*64 format_data !local, To form a format line
      ! For reading columns :
      integer ix_data
      real*8 r_data,psi_data,b_data,db_data,e_data,curr_data,elecd_data
      real*8 dene_data,denD0_data,deni_data,denz_data
      real*8 ti_data,te_data
      
      real*8 zeff_in,rbb,eps_in,trapfrac_in
      real*8 sptzr_out,pressau1,pressau2
      
      real*8,dimension(:),allocatable :: denzz_data ![0:nstates]
      save denzz_data
      
      ipresent=0  !To be changed to 1 if filenm is found
      t_data=0.d0 !To be found from reading data: time slice [sec]
      iunit=14
      
      pi=atan2(zero,-one)
      
      if(.NOT.ASSOCIATED(d_rr))then !temporary; normally is done in tdtraloc
       allocate(d_rr(0:iyp1,0:jxp1,ngen,0:lrz),STAT=istat)
       call bcast(d_rr,zero,SIZE(d_rr))
      endif
      
      open(unit=iunit,file=filenm,status='old',iostat=kode)
      if (kode.ne.0) then
!CMPIINSERT_IF_RANK_EQ_0
!         WRITE(*,*)'subr.read_data_files:',filenm,' cannot be opened'
!CMPIINSERT_ENDIF_RANK         
         ipresent=0
         goto 200 !-> return/exit
      else
         ipresent=1
      endif

 10   format(i6) ! First number in first line: Radial grid size.
 11   format(i6,4x,a6,i6,3x,a6,d12.5) !The whole first line.
      
      if(kopt.eq.0  .and. ipresent.eq.1)then
        !Only check the presence of file; 
        !Also check the radial grid.  Do not read data yet.
        read(iunit,10) njene_data
        njene=njene_data !To be saved into name_decl.h
        !This value is checked in ainsetva against njenea
        goto 200 !-> return/exit
      endif
      
      if(kopt.eq.1 .and. ipresent.eq.1)then
        idum=0
        cdum='none'
        dum1=0.d0

        read(iunit,11,iostat=kode) idum,cdum,idum,cdum,dum1
        if (kode.ne.0) then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'subr.read_data_files:',TRIM(filenm)
          WRITE(*,*)'subr.read_data_files: Corrupted 1st line'
          WRITE(*,*)'time[sec]==dum1=',dum1
CMPIINSERT_ENDIF_RANK         
          STOP 'In read_data_files: cannot read 1st line'
        endif
        t_data=dum1 !OUTPUT: to be used to fill bctime() array
        
        !allocate storage for array to read data
        if (.NOT. ALLOCATED(denzz_data)) then
          allocate(denzz_data(0:nstates),STAT=istat)
          !To read data for impurity, all ionization states,
          !starting from 0 (neutral)
        endif
 
        !Allocate time-dependent variable 
        !It is set as pointer in comm.h, 
        !and stored in  common/impur_read_data/dens_imp_t
        !Only needed when read_data='nimrod'.
        !Note that dens_imp_t() is not a namelist var, it is just a storage
        !for density of ionization states at each time slice.
        if(.NOT. ASSOCIATED(dens_imp_t)) then
          allocate(dens_imp_t(0:nstates,1:njene,1:nbctime),STAT=istat)
        endif

        !if(transp.eq."enabled")then !could add later
        if(.NOT. ASSOCIATED(dbb2in_t)) then
          allocate(dbb2in_t(1:njene,1:nbctime),STAT=istat)
          allocate(dbb2(1:lrz),STAT=istat)
        endif
        !endif !transp

        if(.NOT. ASSOCIATED(ryain_t))then !YuP[2021-08-20] added: time-dependent
          allocate(ryain_t(1:njene,1:nbctime),STAT=istat)
        endif

    
        !Form format specs.  ONLY FOR NIMROD data files: 
        write(cdum,'(i4)') (nstates+1) ! columns corr to 0:nstates
        format_data='(1x,i4,1x,13d16.9,'//TRIM(cdum)//'(d16.9))' 
        !write(*,*) 'format_data=',format_data
        !write(*,*) 't_data[sec]=',t_data
        
        ! Read row by row.
        ! After 2021-02-29, additional column is added 
        !   for |dB| = (B-<B>)/B where <..> flux average, or (B-B0)/B
! ix,R,psi,|B|,|dB|,E.B/B,J.B/B,elecd,ne,nD,ni,nz,Ti,Te,Nz0,Nz1,Nz2,Nz3,Nz4,Nz5,Nz6,Nz7,Nz8,Nz9,Nz10
! [SI units]
! Note, from printout: E_data/J_data  matches mu0*elecd, up to 2nd-3rd digit.
! In NIMROD, the "spitzer model" is a simple scaling eta_0*T^(-1.5)
! with floor and ceiling (to prevent the current density being too high or too low)
        read(iunit,*) !Skip one row (with headings)
        do irow=1,njene       
           read(iunit,format_data,iostat=kode) 
     &         ix_data,
     &         r_data, psi_data, b_data, db_data,
     &         e_data, curr_data, elecd_data,
     &         dene_data,denD0_data,deni_data,denz_data,
     &         ti_data,te_data,
     &         (denzz_data(idum),idum=0,nstates) 
           if (kode.ne.0) then
CMPIINSERT_IF_RANK_EQ_0
             WRITE(*,*)'subr.read_data_files: cannot read line=',irow+2
CMPIINSERT_ENDIF_RANK         
             STOP 'In read_data_files: cannot read line'
           endif
     
           !1.Verified:  denz_data ~ sum(denzz_data), up to 7-8 digits.
           ![ denz_data is from the column designated as "nz" ] 
           !We don't need denz_data by itself, so - no adjustment here.
           !Also noticed that sometimes denzz_data(idum) have 
           !small negative values (at initial time slices).
           !---> Resetting to be not lower than 0.0 :
           do kstate=0,nstates
              denzz_data(kstate)= max(denzz_data(kstate),0.d0)
           enddo
           !Also, just in case:
           deni_data= max(deni_data,0.d0) ! For main ion "ni"
           
           !2.TEST for quasineutrality.
           sum_ni_Zi= deni_data*bnumb(kionm(1)) !from column "ni", 
                      !which is for the main ion (D+)
           sum_ni_Zi2=deni_data*bnumb(kionm(1))**2 !For Zeff computation.
           sum_nimp_zstate=0.d0 !To count electrons from impurity, all Zstates
           do kstate=1,nstates !These are additional ions from impur.source.
             sum_nimp_zstate= sum_nimp_zstate
     &                       +denzz_data(kstate)*bnumb_imp(kstate)
             sum_ni_Zi2= sum_ni_Zi2
     &                  +denzz_data(kstate)*bnumb_imp(kstate)**2 !For Zeff
           enddo ! kstate
           sum_ni_Zi= sum_ni_Zi +sum_nimp_zstate !total number of free electrons
           err1= abs(dene_data-sum_ni_Zi)/dene_data !Supposed to be ~0
           if(err1.gt.1.d-2)then
             write(*,'(a,2e16.9)')
     &       '  dene_data,sum_ni_Zi=',dene_data,sum_ni_Zi
           endif
           !RESULTS: At t_data>0, dene_data = sum_ni_Zi up to 7 digit,
           ! but at t_data=0 [the very 1st data file]
           ! the value of dene_data [column "ne"] is corrupted - 
           ! it is set to 0.4e20 at all radial points 
           ! (supposed to be the edge value only)
           !---> Resetting ne to have quasineutrality:
           dene_data= sum_ni_Zi 
           zeff_in= sum_ni_Zi2/sum_ni_Zi !Additionally, Zeff is computed
           !END TEST          
           
           !3. LOWER LIMIT for Te and Ti
           !Noticed that Te and Ti may go down to ~0.3 eV in data tables.
           !---> Resetting to be not lower than temp_min_data :
           te_data= max(te_data,temp_min_data*1d3) 
           ti_data= max(ti_data,temp_min_data*1d3) 
           ! ti_data is in eV, while temp_min_data is in keV in cqlinput namelist
           
           !--------- Now populate CQL3D arrays
      
           !YuP[2024-07-25] Added radcoord="polflx" 
           !and "sqpolflx" (through psi_data)
           if(radcoord.eq.'rminmax')then
             !if(t_data.eq.0.d0)then !YuP[2021-08-13] added
              ! with this condition, the definition of rho is based 
              ! on the very first data file. Then ignore other files for ryain. 
              ryain_t(irow,jtm)=r_data !Here in [m]
              !The above definition will be normalized by (r_data_max-r_data_min)
              ! after all rows are read.
             !endif
           else !((radcoord.eq.'polflx').or.(radcoord.eq.'sqpolflx'))
              ryain_t(irow,jtm)=psi_data !Units not important - will be norm-ed
              !This is only the first step in definition, 
              !later - psilim will be determined,
              !and ryain_t will be redefined to be norm-ed to [0;1]
           endif
           
           !YuP[2024-07-25] Added - detection of irow_lcfs,
           !which we find as the last irow where psi_data>0
           !(psi_data<0 is outside of LCFS) 
           if(psi_data.ge.0.d0)then
             irow_lcfs=irow !YuP[2024-07-25] Detection of irow_lcfs
             !This value is updated until psi_data >= 0, 
             !then it will stop updating
           endif
           
           tein_t(irow,jtm)= te_data*1.d-3 !converted to keV
           tiin_t(irow,jtm)= ti_data*1.d-3 !converted to keV
           !Main Maxwellian species (only one, for now, D+):
           enein_t(irow,kionm(1),jtm)= deni_data*1.d-6 !converted to cm^-3
           !Electron species:
           if(kelecg.ne.0)then !General e species
           enein_t(irow,kelecg,jtm)= dene_data*1.d-6 !converted to cm^-3
           endif
           if(kelecm.ne.0)then !Maxwellian e
           enein_t(irow,kelecm,jtm)= dene_data*1.d-6 !converted to cm^-3
           endif
           
           !Electric field [V/cm]
           !Note:  checked that E_data/j_data  
           ! matches elecd*mu0 (==elecd_data*12.6e-7),
           ! up to 2nd-3rd digit. 
           if(elecfld_data.eq.'readdata')then ![2022-07-13] New option
             !Get E directly from data, from column "E.B/B"
             elecin_t(irow,jtm)= e_data*1.d-2 !converted V/m to V/cm
           endif !(elecfld_data.eq.'readdata')then 
           
           
           !The following evaluation of eta 
           !is only needed for elecfld_data.eq.'jspitzer'
           !but we evaluate it in any case, for check/printout
           r_data_cm= r_data*100.d0 ! Given major radius for this row in data
           if(r_data_cm.lt.rpcon(1))then ! Smaller than rya(1) point
             lr=1
             rbb= rgeom(lr)*bmod0(lr)/bthr0(lr)
             eps_in= eps(lr)
             trapfrac_in= trapfrac(lr)
             bpsi_max_in= bpsi_max(lr) ! Bmax/Bmin
           elseif(r_data_cm.ge.rpcon(lrz))then !Larger than rya(lrz)
             lr=lrz
             rbb= rgeom(lr)*bmod0(lr)/bthr0(lr)
             eps_in= eps(lr)
             trapfrac_in= trapfrac(lr)
             bpsi_max_in= bpsi_max(lr) ! Bmax/Bmin
           else ! scan 1:lrz range in Rpcon, find the nearest lr-nodes
             !Note: rpcon(lr)== Routboard[cm] corresponds to rya(lr)
             do ll=1,lrz-1
               if( r_data_cm.ge.rpcon(ll) .and. 
     &             r_data_cm.lt.rpcon(ll+1)     )then
               lr=ll
               rbb= rgeom(lr)*bmod0(lr)/bthr0(lr)
               eps_in= eps(lr)
               trapfrac_in= trapfrac(lr)
               bpsi_max_in= bpsi_max(lr) ! Bmax/Bmin
               endif
             enddo
           endif
           !YuP: From printout for DIII-D equilibrium:
           !eps, sqrt(1.-Bmin/Bmax), trapfrac = 8.078E-03 1.266E-01 1.302E-01
           !eps, sqrt(1.-Bmin/Bmax), trapfrac = 1.053E-01 4.351E-01 4.512E-01
           !eps, sqrt(1.-Bmin/Bmax), trapfrac = 2.042E-01 5.785E-01 5.931E-01
           !eps, sqrt(1.-Bmin/Bmax), trapfrac = 3.599E-01 7.239E-01 6.733E-01
           !The 1st line is for plasma center, the last line is for plasma edge
!             write(*,'(a,1p3e11.3)')
!     &        'eps, sqrt(1.-1/bpsi_max), trapfrac =', 
!     &         eps_in, sqrt(1.d0-1.d0/bpsi_max_in), trapfrac_in
           !  Find Spitzer resistivity [IN/OUT: cgs units, and keV]
!      write(*,*)'===========  r_data [m]=',r_data
!      write(*,'(a,1p9e11.4)')
!     &     ' Te,Ti[ev],ne,ni[m3],mi[g],Zeff,rbb[cm],eps,trapfrac=',
!     &     te_data, ti_data,dene_data, deni_data, fmass(kionm(1)), 
!     &     zeff_in,  rbb,eps_in,trapfrac_in
           !trapfrac_in=0.d0 !TEST ONLY!
           call sptz(te_data*1.d-3, ti_data*1.d-3,
     &          dene_data*1.d-6, deni_data*1.d-6, fmass(kionm(1)), 
     &          zeff_in,  rbb,eps_in,trapfrac_in,  
     &          sptzr_out,pressau1,pressau2)
           !Based on estimate of Spitzer resistivity 
           ! with neoclassical trapping correction pressau1 (~ 1.2x -- 3.6x):
           eta_sptzr= sptzr_out*9.e9   ! Resistivity [SI]
!           e_sptzr= curr_data*(eta_sptzr*pressau1) ! V/m, A/m^2
           !Note: the above uses eta_sptzr*pressau1, 
           !which is collision-less limit. 
           !----------------------------------------------------------
           ![2022-01-06]Better use collisional limit (as in NIMROD):
           e_sptzr= curr_data*(eta_sptzr*pressau2) ! V/m, A/m^2
           !e_sptzr= curr_data*eta_sptzr !TEST ONLY! V/m, A/m^2
           !In this case - in profiles.f, 
           ! change curra(1,ll)/sig_starnue0(ll)
           !     to curra(1,ll)/sig_starnue(ll)
           !----------------------------------------------------------
           !Just to save data, for printout: (xjin_t is not used during run)
           xjin_t(irow,jtm)=curr_data*1.d-4 ! A/cm^2 
!      write(*,'(a,1p3e11.4)')'OUT: eta_sptzr[SI],pressau1,pressau2=',
!     &            sptzr_out,pressau1,pressau2
!      write(*,*)' '

           if(elecfld_data.eq.'jspitzer')then ![2022-07-13] Original option
           !by default, evaluate E as j_data/resistivity.
           elecin_t(irow,jtm)= e_sptzr*1.d-2 !converted V/m to V/cm
           !Note: in NIMROD file, e_data is labeled as "E.B/B",
           !so - is it parallel? Should we reset efflag="parallel" ?
           !Also the proper sign is of some concern.
           !Maybe need to use bsign here? 
           !write(*,'(a,e12.6)')'elecin_t()=',elecin_t(irow,jtm)
           endif !elecfld_data.eq.'jspitzer'
           
CMPIINSERT_IF_RANK_EQ_0
        !Check the difference in two methods for E, print out if too large
        e_data_abs= abs(e_data)
        e_sptzr_abs=abs(e_sptzr)
        if( e_data_abs > 1d-2 )then !Only check when E > 0.01 V/m
        if(abs(e_data_abs-e_sptzr_abs)>0.2*(e_data_abs+e_sptzr_abs))then
             !eta_data= e_data/curr_data  ! Resistivity [SI]
         write(*,'(a,i5,1p4e11.3)')'  ix,t_data, r_data,e_data,e_sptzr',
     &              irow-1, t_data, r_data, e_data, e_sptzr ![sec; m; V/m]
        endif
        endif
CMPIINSERT_ENDIF_RANK         
           
           !Impurity - densities of each ionization state
           do kstate=0,nstates
              dens_imp_t(kstate,irow,jtm)=denzz_data(kstate)*1.d-6 !to cm^-3
           enddo
           
           !if(transp.eq."enabled")then !could add later
             ![2021-08] Added, For NIMROD coupling 
             !(reading deltaB/B data)
             dbb2in_t(irow,jtm)= db_data**2 !== (dB/B)^2
             !Drr is formed at each time step in subr.profiles, based on
             !See Rechester, Rosenbluth, Phys.Rev.Lett.40,40(1978).
             !A pitch-angle dependence is added as in 
             !Harvey,McCoy,Hsu,Mirin, PRL 47, p.102 (1981)
           !endif !transp
           
        enddo ! irow=1,njene 

        !YuP[2024-07-25] Done above - detection of irow_lcfs,
        !which we find as the last irow where psi_data>0
        !(psi_data<0 is outside of LCFS).
        !Redefine njene now, to keep data within rho<1 only:
        njene=irow_lcfs !VERY IMPORTANT
        !(Typical example: njene_data=144, irow_lcfs=120)
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'Based on psi_data, njene is reset to njene=',njene
CMPIINSERT_ENDIF_RANK         

        !YuP[2024-07-25] Added radcoord="polflx" 
        !and "sqpolflx" (through psi_data)
        !if(t_data.eq.0.d0)then
        if(radcoord.eq.'rminmax')then
          ! with this condition, the definition of rho is based 
          ! on the very first data file. Then ignore other files for ryain. 
          !Above, in the irow=1,njene loop, this array was saved:
          !ryain_t(irow,jtm)= r_data 
          r0_data=     ryain_t(1,jtm)         !R at m.axis
          r_data_lcfs= ryain_t(irow_lcfs,jtm) !R at LCFS
          ryain_t(:,jtm)=(ryain_t(:,jtm)-r0_data)/(r_data_lcfs-r0_data)
        else !((radcoord.eq.'polflx').or.(radcoord.eq.'sqpolflx'))
          !Above, in the irow=1,njene loop, this array was saved:
          !ryain_t(irow,jtm)=psi_data 
          psi0_data=     ryain_t(1,jtm)         !psi at m.axis
          psi_data_lcfs= ryain_t(irow_lcfs,jtm) !psi at LCFS
          
          if(radcoord.eq.'polflx')then
            ryain_t(:,jtm)= (ryain_t(:,jtm)-psi0_data)/
     &                       (psi_data_lcfs-psi0_data)
          endif
          if(radcoord.eq.'sqpolflx')then
            ryain_t(:,jtm)= sqrt( (ryain_t(:,jtm)-psi0_data)/
     &                            (psi_data_lcfs-psi0_data)   )
          endif
        endif !radcoord
        !endif !(t_data.eq.0.d0)
        
        !write(*,*)'ryain=',ryain_t(1:njene,jtm) ! checked: [0;1] range
        !write(*,*)kelecg,kelecm
        
      endif ! kopt.eq.1 .and. ipresent.eq.1
      
      
 200  continue
      close(iunit)

      return
      end subroutine read_data_files
