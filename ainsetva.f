c
      subroutine ainsetva
      implicit integer (i-n), real*8 (a-h,o-z)
      character*8 eqmirror

      save

c.......................................................................
c     This routine sets auxiliary values which depend on
c     input variables.
c     Should be called after reading the namelist setup 
c     (not after setup0).
c     Also, it checks some namelist variables for consistency, 
c       and resets some if necessary.
c.......................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE
c
      !real*8:: tmpt(njene)  !Temporary array, local. YuP[2019-10-29]
      character*6 cdum !local
      real*8:: nbeams1p(kb),kblspecp(kb)
      
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)''
      WRITE(*,*)'In ainsetva'
CMPIINSERT_ENDIF_RANK
c
c.......................................................................
c     0. Check for consistency of some parameter settings
c        (in param.h).
c......................................................................

      if (lrorsa.ne.max(lrza,lsa)) 
     +     stop 'check consistency of lrorsa in param.h'

cBH081106:Found following condition leads to overwrite.
cBH081106:For cqlpmod.ne.enabled, could check code to see how
cBH081106:essential the condition is.

!YuP[2021-03-30] No longer needed:  if (lsa.lt.lrza) stop 'check that lsa.ge.lrza?'

!      if (cqlpmod.eq."enabled" .and. lza.lt.lsa) !YuP[2021-04-14] lza no longer used
!     +     stop 'check consistency of lza,lsa for cqlpmod=enabled'

cBH060314      if (nmodsa.ne.3)
cBH060314     +     stop 'Better check out code for nmodsa.ne.3'

      if (nefitera.lt.2) stop 'Coding assumes nefitera.ge.2'

c.......................................................................
cl    1. check and define some mesh values
c.......................................................................

c     cannot run CQL3D with lrzmax>lrz (only CQL or CQLP)
      if (lrzdiff.eq."enabled" .and. transp.eq."enabled" .and.
     +  cqlpmod.ne."enabled") call diagwrng(17)
      if (lsdiff.eq."enabled" .and.  transp.eq."enabled" .and.
     +  cqlpmod.eq."enabled") call wpwrng(4)
      if (lrzdiff.eq."enabled" .and. partner.eq."selene") WRITE(*,
     +  '("WARNING: partner=selene and lrzdiff=enabled not checked")')

      if (urfmod.ne."disabled" .and. meshy.eq."fixed_mu")
     +  call diagwrng(19)
     
      !if (urfmod.eq."enabled" .and. iy.gt.255) call diagwrng(22)
        !The above limitation - is it because of call pack() limitation?
        !subr. pack() is not used anymore.
        !YuP[2024-04-24] Removing this limitation - 
        !now CQL3D is using pack16() only,
        !which has a much higher limit (32768). 
	

c     storage problem for calc of parallel distn function, if
c     lrz.gt.lz
      if ((pltprpp.eq."enabled" .or.  knockon.ne."disabled") .and.
     +     lrz.gt.lz) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'ainsetva/pltprpp:  lrz=', lrz, ' lz=', lz
         WRITE(*,*)'Need lrz.le.lz for pltprpp or knockon enabled'
CMPIINSERT_ENDIF_RANK
         lz=lrz
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'WARNING: lz is reset to lrz=', lrz
CMPIINSERT_ENDIF_RANK
         !STOP
      endif


      if ((iy/2)*2 .ne. iy) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,220)
CMPIINSERT_ENDIF_RANK
 220     format(//,'WARNING:  Making iy even, increasing by 1',//)
         iy=iy+1
      endif

c%OS  if (iy .lt. 16) iy=16
c%OS  if (jx .lt. 8) jx=8
      iyh=iy/2
      jxm1=jx-1
      jxp1=jx+1
      iyp1=iy+1
      iyjx=iy*jx
      iyjxp1=iy*(jx+1)
      iyp1jx=(iy+1)*jx
      iyjx2=(iy+2)*(jx+2)


      if (ndeltarho.ne."disabled". and. (nt_delta/2)*2.ne.nt_delta) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,223)
CMPIINSERT_ENDIF_RANK
 223     format(//,'WARNING:  Making nt_delta even, increasing by 1',//)
         nt_delta=nt_delta+1
      endif

c     fr_gyrop='disabled' is default, for backwards compatability.
c     Warn on use of fr_gyro for ZOW particles.
      if (frmodp.eq.'enabled') then
         if (fr_gyrop.ne.'disabled') then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)
            WRITE(*,*)'WARNING: ZOW NB ions, but using fr_gyro=enabled'
            WRITE(*,*)
CMPIINSERT_ENDIF_RANK
         endif
      endif

      if (ndeltarho.ne."disabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,224)
CMPIINSERT_ENDIF_RANK
 224     format(//,'WARNING: ndeltarho.ne."disabled"',/
     +             '         Remember that baviorbt (taunew=enabled)',/
     +             '         is setup with nii=1. Consider nii.gt.1',//)
      endif

      if (ndeltarho.ne."disabled". and. taunew.ne."enabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,225)
CMPIINSERT_ENDIF_RANK
 225     format(//,'ERROR:  Presently ndeltarho calls are only setup',/
     +             '        for taunew=enabled',//)
         STOP
      endif

      if (ndeltarho.ne."disabled". and. lrzdiff.ne."disabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,226)
CMPIINSERT_ENDIF_RANK
 226     format(//,'ERROR:  ndeltarho feature not setup for setup',/
     +             '        for lrzdiff.ne."disabled"',//)
         STOP
      endif

      if (ndeltarho.ne."disabled". and. urfdmp.eq."secondd") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,227)
 227     format(//,'ERROR:  Presently ndeltarho calls are only setup',/
     +             '        for urfdmp.ne.disabled',//)
         WRITE(*,228)
CMPIINSERT_ENDIF_RANK
 228     format(//,'ERROR:  Resetting urfdmp="firstd"',/ 
     +             '         See cqlinput_help',//)
         urfdmp="firstd"
      endif



      if (ngen.gt.ngena .or. nmax.gt.nmaxa) call diagwrng(12)
      
!      if (lz.gt.lza) then !YuP[2021-04-14] lza is not used anymore
!         lz=lza
!         WRITE(*,*)'WARNING: lz reset to lza =', lza
!      endif
      
cBH110330      if (msxr.gt.mx) then
cBH110330        msxr=mx
cBH110330CMPIINSERT_IF_RANK_EQ_0
cBH110330        WRITE(*,201) mx 
cBH110330CMPIINSERT_ENDIF_RANK
cBH110330 201    format('WARNING: msxr reset to mx =', i5)
cBH110330      endif
cBH110331      if (mmsv.gt.mx) then
cBH110331        mmsv=mx
cBH110331CMPIINSERT_IF_RANK_EQ_0
cBH110331        WRITE(*,202) mx
cBH110331CMPIINSERT_ENDIF_RANK
cBH110331 202    format('WARNING: mmsv reset to mx =', i5)
cBH110331      endif
      mxp1=mx+1
      if (njene.gt.njenea) call diagwrng(16)

      do 100 ll=1,lrors
        iy_(ll)=iy
        iyh_(ll)=iyh
        iyjx_(ll)=iyjx
 100  continue
      iymax=iy

      !ipxy=min(51,iy) !YuP[2020-10-26] Why the lower limit is needed?
      !jpxy=min(101,jx+1) !YuP[2020-10-26] Why the lower limit is needed?
      if (mod(jpxy,2).eq.0) jpxy=jpxy-1

c.......................................................................
c     1.1 Check mx so no overwrite because of
c        tamt1ptr=temp1ptr, tamt2ptr=temp4ptr.
c        (Overwrite has already occured, but better to warn late
c        than not at all).
c        Similarly, will have overwrite if
c        jpxy*ipxy.gt.(iyp1+1)*(jxp1+1).
c.......................................................................

      !-YuP: Why needed ???      COMMENTING it out:
ccc      if(2*jx*(mx+5)*(mx+5).gt.(iy+2)*(jx+2)) then
ccc      write(*,*)'ainsetva: Possible tamt1 and tamt2 overwrite',jx,iy,mx
ccc         stop 'problem with tamt1 and tamt2 overwrite'
ccc      endif

      if (jpxy*ipxy .gt.  iy*(jx+1)  .or.
     +    jpxy*ipxy .gt. (iy+1)*jx   ) then
         stop 'problem with jpxy*ipxy dimensioning'
      endif
c


c.......................................................................
cl    2. define effective time-step according to model
c.......................................................................

      do 50 i=1,ndtr1a
      if (nondtr1(i).ge.0) then
        if (mod(nondtr1(i),nrstrt).ne.0) then
          nondtr1(i)=(mod(nondtr1(i),nrstrt)+1)*nrstrt
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,101) nondtr1(i),i
CMPIINSERT_ENDIF_RANK
 101      format('WARNING: nondtr1(i) reset to ',i5,'  i=',i2)
        endif
      endif
      if (nondtr1(i).eq.n) dtr=dtr1(i)
 50   continue

      dtreff=dtr
      if (transp.eq."enabled") then
        dttr=dtr*nrstrt
        if (adimeth .eq. "enabled") then
          dtreff=dtr
          nrstrt=1
          dttr=dtreff
        endif
        do k=1,ngen
           if (difus_type(k).eq."neo_smpl" .and. k.eq.kelecg) then
CMPIINSERT_IF_RANK_EQ_0
              WRITE(*,*)"STOP: Not sensible to use ion neocl for e's"
CMPIINSERT_ENDIF_RANK
              STOP
           endif
        enddo
      endif

c     Time-dep radial transport multipliers
      do k=1,ngena
         drrt(k)=one
         drt(k)=one
      enddo

      if (transp.ne."enabled") nontran=0
      if (transp.eq."enabled" .and. adimeth.eq."enabled") then
        if (nonadi .lt. nontran) nonadi=nontran
      endif

      if (ndifus_io_t.gt.nbctimea) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)"STOP: ndifus_io_t.gt.nbctimea"
CMPIINSERT_ENDIF_RANK
         STOP
      endif

      !--------------------------------------------------------------
      ! For Method of deposition of impurity:
      if(pellet.eq.'enabled')then !for backward compatibility
        imp_depos_method='pellet' !YuP[2019-12-05]
        tstart_imp= pellet_tstart
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)"WARNING: Instead of pellet='enabled'/'disabled' use"
         WRITE(*,*)" imp_depos_method='pellet'/'instant'/'disabled'. "
         WRITE(*,*)" Setting imp_depos_method=",imp_depos_method
         WRITE(*,*)"WARNING: Instead of setting pellet_tstart value, "
         WRITE(*,*)" set tstart_imp value (in cqlinput)."
         WRITE(*,*)" Setting tstart_imp to pellet_tstart=",tstart_imp
         WRITE(*,*)"WARNING: imp_ne_method is set to ",imp_ne_method
CMPIINSERT_ENDIF_RANK
      endif  
      !--------------------------------------------------------------

c.......................................................................
cl    3. Check and adjust species if iprozeff.ne."disabled"
c.......................................................................

      if (iprozeff.ne."disabled") then
        !  iprozeff = 
        != "parabola","prbola-t", "spline"  or "spline-t", then this 
        ! option gives ion densities so as to achieve a given zeff
        ! radial profile, consistent with charge neutrality and the 
        ! given electron density profile.
        if((imp_depos_method.ne.'disabled').and.(kelec.ne.0))then
        !YuP[2020-06-24] Changed (gamafac.eq."hesslow") to (imp_depos_method.ne.'disabled')
        !   [a more general logic]
          iprozeff='disabled'
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'ainsetva: For (imp_depos_method.ne."disabled") '
          WRITE(*,*)'   and (kelec.ne.0), iprozeff is reset to',iprozeff
CMPIINSERT_ENDIF_RANK
            !YuP[2019-09-18] Density of electrons and Zeff
            ! will be calculated consistently:
            ! ne will be found from reden() of ions(kion) and from 
            ! dens_imp(kstate,lr) of additional ions (from pellet, etc.)
        endif ! (imp_depos_method.ne.'disabled')
      endif

      if (iprozeff.ne."disabled") then  !endif at line 285
        !  iprozeff = 
        != "parabola","prbola-t", "spline"  or "spline-t", then this 
        ! option gives ion densities so as to achieve a given zeff
        ! radial profile, consistent with charge neutrality and the 
        ! given electron density profile.

c       Check for electrons
        if (kelec.eq.0) stop 'ainsetva: need electron species'
c       Check for at least one maxwellian ion.  For the
c       case of only general species ions, could adjust code slightly.
        if (nionm.eq.0) 
     +     stop 'ainsetva: need at least one maxwellian ion species' 

c
c     Check number of ion Maxwl species with different bnumb.
        if (nionm.lt.1) stop 'ainsetva: ion species problem: nionm.lt.1'
        ndif_bnumb=1
        do k=2,nionm
           if (abs(bnumb(kionm(k))/bnumb(kionm(1))-1.).gt.0.01) 
     +          ndif_bnumb=ndif_bnumb+1
        enddo
          


        if (ndif_bnumb.eq.1) then
c         add a species
          nionm=nionm+1
          if (nionm.gt.nmaxa) stop 'ainsetva:  check nmaxa'
          kionm(nionm)=kionm(nionm-1)+1
          ntotal=ntotal+1
          nmax=nmax+1
          if (kelecm.gt.kionm(1)) then   ! Move up electron data
            kelecm=kelecm+1
            fmass(kelecm)=fmass(kelecm-1)
            bnumb(kelecm)=bnumb(kelecm-1)
            kspeci(1,kelecm)=kspeci(1,kelecm-1)
            kspeci(2,kelecm)=kspeci(2,kelecm-1)
            reden(kelecm,0)=reden(kelecm-1,0)
            reden(kelecm,1)=reden(kelecm-1,1)
            do 60  ll=1,njene
              enein(ll,kelecm)=enein(ll,kelecm-1)
 60         continue
            temp(kelecm,0)=temp(kelecm-1,0)
            temp(kelecm,1)=temp(kelecm-1,1)
            !YuP[2025-12-12] Also need to move up powers npwr and mpwr:
            mpwr(kelecm)=mpwr(kelecm-1) !For T_e_maxw
            npwr(kelecm)=npwr(kelecm-1) !For T_e_maxw
            if (nbctime.ne.0) then
               do jtm=1,nbctime
                  redenc(jtm,kelecm)=redenc(jtm,kelecm-1)
                  redenb(jtm,kelecm)=redenb(jtm,kelecm-1)
                  tempc(jtm,kelecm)=tempc(jtm,kelecm-1)
                  tempb(jtm,kelecm)=tempb(jtm,kelecm-1)
               enddo
            endif
          endif
          if (kelec.gt.kionm(1))  kelec=kelec+1
c201030          fmass(kionm(nionm))=2.*50.*1.676e-24 !YuP:suggested to use "100*proton" here
c201030          bnumb(kionm(nionm))=50.
          fmass(kionm(nionm))=2.*100.*1.676e-24 !YuP:suggested to use "200*proton" here
          bnumb(kionm(nionm))=100.
          kspeci(1,kionm(nionm))="impurity"
          kspeci(2,kionm(nionm))="maxwell"
          if (nbctime.ne.0) then
             do jtm=1,nbctime
                redenc(jtm,kionm(nionm))=1.    !Nominal value, to
                redenb(jtm,kionm(nionm))=1.    !be reset with zeff.
                tempc(jtm,kionm(nionm))=tempc(jtm,kionm(nionm)-1)
                tempb(jtm,kionm(nionm))=tempb(jtm,kionm(nionm)-1)
             enddo
          endif
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'ainsetva: Impurity species added, k=',k
          WRITE(*,*)'ainsetva: bnumb=',bnumb(kionm(nionm))
          WRITE(*,*)'ainsetva: fmass/proton_mass=',
     +                         fmass(kionm(nionm))/proton
CMPIINSERT_ENDIF_RANK
        elseif (ndif_bnumb.eq.2) then
          continue  
        else ! ndif_bnumb ne.1, ne.2
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'ainsetva: iprozeff=', iprozeff
          WRITE(*,*)'ainsetva: ndif_bnumb=',ndif_bnumb
CMPIINSERT_ENDIF_RANK
          stop 'ainsetva: ion species problem. ndif_bnumb'
        endif

      endif  !On iprozeff.ne."disabled"

c.......................................................................
c     Reset parabolic profile specifiers for initial profile calc
c     if time-dependent profiles specified
c     BH171124: Actually, should now be using iprone="prbola-t" in
c               nbctime.ne.0 cases where time dependent profiles are
c               desired.  Will ask that cqlinput be checked.
c.......................................................................
      if (nbctime.ne.0) then
      
         !YuP[2019-09-18] Moved this check (of T<0) from profiles.f.
         !Need to check just once, before start of simulation
         if(iprote.eq.'prbola-t' .or. iprote.eq.'spline-t')then
         do it=1,nbctime      
          if (tempc(it,kelec).le.zero .and. tein_t(1,it).le.zero) then
          WRITE(*,*) "Time-dependent Te profile input problem at it=",it
             STOP 'Te<0'
          endif
         enddo
         endif
         !Assume any time dep in tempc is in first ion species.    
         if(iproti.eq.'prbola-t' .or. iproti.eq.'spline-t')then
         do it=1,nbctime
          if (tempc(it,kionn).le.zero .and. tiin_t(1,it).le.zero) then
          WRITE(*,*) "Time-dependent Ti profile input problem at it=",it
             STOP 'Ti<0'
          endif
         enddo ! [2019-09-18]
         endif
         
         if (iprone.eq."parabola" .or. iprote.eq."parabola"
     1      .or.iproti.eq."parabola" .or. iprozeff.eq."parabola"
     1      .or.iproelec.eq."parabola" .or. iprovphi.eq."parabola"
     1      .or. iprocur.eq."parabola"   ) then
            !YuP[2018-01-02] Added iprocur in the above if()then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'------------------------------------------------'
            WRITE(*,*) 'ainsetva: nbctime=',nbctime
            WRITE(*,*) 'If nbctime.ne.0: CHECK ipro*** settings.'
            WRITE(*,*) 'Some of them (or all) are set to "parabola" .'
            WRITE(*,*) 'Resetting those to time-dependent "prbola-t" '
            WRITE(*,*)
CMPIINSERT_ENDIF_RANK
            ! BH,YuP[2018-01-02] Added resetting of ipro*** values 
            ! to "prbola-t", in case when the run 
            ! is done with nbctime>0 (time-dependent profiles),
            ! but ipro*** values are set to "parabola". 
!            if(iprone.eq."parabola") then
!               iprone="prbola-t"
CMPIINSERT_IF_RANK_EQ_0
!               WRITE(*,*)'iprone is reset to "prbola-t" '
CMPIINSERT_ENDIF_RANK
!            endif
            if(iprote.eq."parabola") then
               iprote="prbola-t"
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'iprote is reset to "prbola-t" '
CMPIINSERT_ENDIF_RANK
            endif
            if(iproti.eq."parabola") then
               iproti="prbola-t"
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'iproti is reset to "prbola-t" '
CMPIINSERT_ENDIF_RANK
            endif
            if(iprozeff.eq."parabola") then
               iprozeff="prbola-t"
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'iprozeff is reset to "prbola-t" '
CMPIINSERT_ENDIF_RANK
            endif
            if(iproelec.eq."parabola") then
               !YuP[2020-04-13] No need to reset:   iproelec="prbola-t"
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'WARNING: nbctime>0 but iproelec.eq."parabola"'
               !YuP[2020-04-13] No need to reset: WRITE(*,*)'iproelec is reset to "prbola-t" '
CMPIINSERT_ENDIF_RANK
            endif
            if(iprocur.eq."parabola") then
               iprocur="prbola-t"
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'iprocur is reset to "prbola-t" '
CMPIINSERT_ENDIF_RANK
            endif
            if(iprovphi.eq."parabola") then
               iprovphi="prbola-t"
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'iprovphi is reset to "prbola-t" '
CMPIINSERT_ENDIF_RANK
            endif
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'------------------------------------------------'
CMPIINSERT_ENDIF_RANK
         endif
         ! Note that if(iprote.eq.'prb-expt' or 'spl-expt'), the above logic
         ! will NOT reset iprote to "prbola-t", and so 
         ! the code will go into subr. profiles()
         ! no matter what the value of nbstime (0 or not).
         ! This is what we want - this case is treated in subr.profiles.
      endif ! (nbctime.ne.0) 
      
      if (nbctime.ne.0) then ! YuP[2018-01-02] Since ipro** are reset
        ! to "prbola-t" (above), this part is not needed anymore:
        if (iprone.eq.'parabola') then
           do k=1,ntotal
              reden(k,0)=redenc(1,k)
              reden(k,1)=redenb(1,k)
CMPIINSERT_IF_RANK_EQ_0
              WRITE(*,*)'ainsetva WARN: nbctime>0, but iprone=parabola.'
              WRITE(*,*)'Setting reden(k,0) profiles from redenc(1,k)'
              WRITE(*,*)'Setting reden(k,1) profiles from redenb(1,k)'
              WRITE(*,*)'ainsetva: k,reden(k,0:1)=',
     &                             k,reden(k,0),reden(k,1)
CMPIINSERT_ENDIF_RANK
           enddo
        endif
        if (iprote.eq.'parabola') then
           do k=1,ntotal
              if (k.eq.kelecg .or. k.eq.kelecm) then
                 temp(k,0)=tempc(1,k) ! Note: tempc(1:nbctime,k)
                 temp(k,1)=tempb(1,k)
              endif
           enddo
        endif
        if (iproti.eq.'parabola') then
           do k=1,ntotal
              if (k.ne.kelecg .and. k.ne.kelecm) then
                 temp(k,0)=tempc(1,k)
                 temp(k,1)=tempb(1,k)
              endif
           enddo
        endif
        if (iprozeff.eq.'parabola') then
           zeffin(0)=zeffc(1)
           zeffin(1)=zeffb(1)
        endif
        if (iprovphi.eq.'parabola') then
           vphiplin(0)=vphic(1)
           vphiplin(1)=vphib(1)
        endif
        if (iproelec.eq.'parabola') then
           !YuP[2020-04-13] No need to reset: elecfld(0)=elecc(1)
           !YuP[2020-04-13] No need to reset: elecfld(1)=elecb(1)
        endif
      endif !  on nbctime.ne.0
      
      
      !-----------------------------------------------------------------   
      !BH,YuP[2021-01-21] Option to read data files.
      !(Initial purpose - coupling with NIMROD. 
      ! Can be extended to coupling with other codes.)
      if(read_data.eq.'disabled')then
         continue ! Nothing to check here.
      elseif(read_data.eq.'nimrod')then !check that files are present
        ! Note: data files are declared as
        !character*128, dimension(1000) :: read_data_filenames
        ! Max number of files is 1000, for now. 
        ! For coupling with NIMROD, each file contains data at one time slice.
        ! Therefore, it is recommended to match the max number of files
        ! with value of nbctimea [set in param.h]
        kopt=0 !Below: Only check the presence of files 
        !and read the size of radial grid [top line in data files]; 
        n_files=0 ! Counter of files that are found, see below.
        njene=0 ! To be changed if at least one data file is found.
        do i=1,size(read_data_filenames)
          !Form a template for file name. !YuP[2021-04-06]
          ! Example: 'rycq019800.dat', so it is made of prefix 'rycq0',
          ! followed by 3-digit number and suffix '00.dat'
          ! Example: Use these variables in namelist:
          ! read_data_filenames_prefix='rycq0'
          ! read_data_filenames_suffix='00.dat' 
          ! Max number of files is set by this range: 000:999,
          ! i.e. max 1000 files
          write(cdum,'(i0.3)') i-1 !Write the sequential number into a string
          !Note that we start with i-1=0, 
          !because normally there is a file like rycq000000.dat
          read_data_filenames(i)=
     &      TRIM(read_data_filenames_prefix) //TRIM(cdum)//
     &      TRIM(read_data_filenames_suffix)  ! Filename is formed
          !write(*,*) read_data_filenames(i)
          !Now try to read it:
          if(read_data_filenames(i).ne."notset")then
            !Try to read this file:
            call read_data_files(read_data_filenames(i),i,
     &                              ipresent,t_data,kopt)
            !Note that the numbers in sequence of data files 
            !may have gaps, like
            !  rycq020000.dat, rycq020500.dat, rycq021500.dat, ...
            !so that not all rycq0***00.dat will be present.
            if(ipresent.eq.0)then !could not find the file
!CMPIINSERT_IF_RANK_EQ_0
!              WRITE(*,*)
!     &        'read_data_filenames(i)=',TRIM(read_data_filenames(i))
!              WRITE(*,*) 'FILE NOT FOUND'
!CMPIINSERT_ENDIF_RANK        
              continue 
            else ! ipresent=1, which means: file is found
              n_files= n_files+1 ! Counter of data files.
              !Re-numerate this file, to have the list of only those 
              !files that are present on disk :
              read_data_filenames(n_files)=read_data_filenames(i)
            endif
          endif
        enddo ! i
        if(n_files.eq.0)then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'read_data=',read_data,'  N. of data files=0'
CMPIINSERT_ENDIF_RANK         
          STOP 'Set read_data_filenames(i) in cqlinput'
        else ! at least one file is found; Print the list:
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)' --- For read_data= ', TRIM(read_data) 
          do i=1,n_files
            WRITE(*,*)
     &      'i, read_data_filenames(i)=',i,TRIM(read_data_filenames(i))
          enddo
CMPIINSERT_ENDIF_RANK         
        endif
        if(n_files.gt.nbctimea)then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'read_data=',read_data,'  N. of data files=',n_files
          WRITE(*,*)'STOP: n_files > nbctimea. '
CMPIINSERT_ENDIF_RANK         
          STOP 'Increase nbctimea in param.h'
        endif
        if(njene.le.1)then ! either no data found, or found data 
          !with only one radial point (should have at least two?)
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'read_data=',read_data,
     &    ' Found size of radial grid from data: njene=',njene
          WRITE(*,*)'STOP: njene is too small'
CMPIINSERT_ENDIF_RANK         
          STOP 'Check data in read_data_filenames(i)'
        endif
        if(njene.gt.njenea)then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'read_data=',read_data,
     &    ' Found size of radial grid from data: njene=',njene
          WRITE(*,*)'STOP: njene > njenea. '
CMPIINSERT_ENDIF_RANK         
          STOP 'Increase njenea in param.h'
        endif
        !Check that electrons is the general species
        if (kelecg.eq.0) then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'For read_data=',read_data,
     &    ' need general species = electrons'
CMPIINSERT_ENDIF_RANK         
          STOP 'Need electron general species'
        endif
        !Check that there is one maxwellian ion, and it is D+.
        !NIMROD data has only one "main" ion species, D+.
        if (nionm.eq.0 .or. nionm.ge.2) then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'For read_data=',read_data,
     &    ' need one ionic Maxwellian species = D+'
CMPIINSERT_ENDIF_RANK         
          STOP 'ainsetva: need one maxwellian ion species' 
        endif
        if ((kspeci(1,kionm(1)).ne.'d').and.
     &      (kspeci(1,kionm(1)).ne.'D')      ) then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'For read_data=',read_data,
     &    ' need one ionic Maxwellian species = D+'
CMPIINSERT_ENDIF_RANK         
          STOP 'ainsetva: need D+ maxwellian species' 
        endif
        !If basic tests are passed, reset some of variables:
        nbctime=n_files ! Reset nbctime to number of data files.
        !This is only applicable for read_data.eq.'nimrod' case!
        !The data will be placed into corresponding arrays over ryain grid.
        !Reset options for profiles:
        iprone=  "spline-t" !To use enein_t(njenea,ntotala,nbctimea) array
        iprote=  "spline-t" !To use tein_t(njenea,nbctimea)
        iproti=  "spline-t" !To use tiin_t(njenea,nbctimea)
        iproelec="spline-t" !To use elecin_t(njenea,nbctimea)
        !The arrays are filled-in in subr.tdinitl.
        !On E field: in NIMROD file, E_data is labeled as "E.B/B",
        !so it is Eparallel. Reset efflag="parallel" :
        efflag="parallel"
        ! zeffin_t() is Not provided by NIMROD, use iprozeff='disabled'. 
        iprozeff='disabled'
        !iprocur="spline-t"  !xjin_t() is Not provided by NIMROD
        !iprovphi="spline-t" !vphiplin_t() is Not provided by NIMROD
        ! Also reset radcoord:
        !YuP[2024-07-25] Added radcoord="polflx" 
        !and "sqpolflx" (through psi_data)
        if((radcoord.eq.'rminmax').or.
     &     (radcoord.eq.'polflx').or.
     &     (radcoord.eq.'sqpolflx')    )then
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)'For read_data: radcoord=',radcoord
CMPIINSERT_ENDIF_RANK         
        else      
          stop 'read_data: use radcoord=rminmax or polflx or sqpolflx'
          !Before [2024-07-25] The only option: radcoord='rminmax'
          !In NIMROD data files, the major R grid is given, see r_data
        endif !radcoord
        imp_depos_method='disabled' !Just in case. Should not be used.
        if(transp.eq.'enabled')then
          !The values of Drr==d_rr() are set in
          !subr.profiles, at every time step, 
          !using time-dependent data on dB/B
          !from NIMROD (other codes for coupling could be added).
          !Settings for namelist variable difusr,difus_rshape,difus_vshape,
          !difus_type are not relevant (not used).
          !For setting of pinch, we should avoid "simple" and "case1","case1n"
          if(pinch.eq."simple" .or. pinch.eq."case1" 
     &                         .or. pinch.eq."case1n" )then
            WRITE(*,*)'WARNING: For read_data=',read_data
            WRITE(*,*)
     &        '  pinch can only be disabled,case2,case2n,case3,case3n'
            !YuP[2021-08] Still needs testing
            STOP 'ainsetva/wrong pinch setting'
          endif
        endif !transp
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)' ----------------------------------------- '
        WRITE(*,*)'WARNING: For read_data=',read_data
        WRITE(*,*)
     &  ' values of iprone,iprote,iproti,iproelec are reset to spline-t'
        WRITE(*,*)'  efflag is set to   ',efflag
        WRITE(*,*)'  iprozeff is set to ',iprozeff
        WRITE(*,*)'  radcoord is set to ',radcoord
        WRITE(*,*)
     &  'WARNING: Be sure imp_type is set to match NIMROD data',imp_type
        WRITE(*,*)
     &  'Based on the list of data files [read_data_filenames]:'
        WRITE(*,*)'     nbctime is set to ',nbctime
        WRITE(*,*)'     njene is set to   ',njene
        WRITE(*,*)' ----------------------------------------- '
        WRITE(*,*)' '
CMPIINSERT_ENDIF_RANK         
        !Value of imp_type should match NIMROD data files. Select from:
        !imp_type: 1->He,2->Be,3->C,4->N,5->Ne,6->Ar,7->Xe,8->W,9->B
      else !No other options, for now, but can be added later.
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'WARNING:  read_data=',read_data
        WRITE(*,*)'STOP: read_data can only be "disabled" or "nimrod" '
CMPIINSERT_ENDIF_RANK
        STOP 'namelist var. read_data  has wrong value'
      endif  ! read_data.eq.***
      !-----------------------------------------------------------------    

      

c     Check only tmdmeth='method1', in case of "spline-t" profiles
      if (nbctime.gt.0) then
         if (      iprone.eq."spline-t"
     +      .or. iprote.eq."spline-t"
     +      .or. iproti.eq."spline-t"
     +      .or. iprozeff.eq."spline-t"
     +      .or. iproelec.eq."spline-t"
     +      .or. iprocur.eq."spline-t"
     +      .or. iprovphi.eq."spline-t") then
            if (tmdmeth.ne."method1") then
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'Warning:  Re-setting tmdmeth=method1'
CMPIINSERT_ENDIF_RANK
               tmdmeth="method1"
            endif
         endif
c     Shift bctime() by bctshift (useful for restart runs with time-dep
c     namelist input data. bctime() can be shifted to zero time for 
c     the restart.
      if (bctshift.ne.0.d0) then
         do i=1,nbctimea
            bctime(i)=bctime(i)-bctshift
         enddo
      endif
      endif

c     Rescale bctime, if bctimescal.ne. 1.
      if (bctimescal.ne.1.d0) then
         do i=1,nbctimea
            bctime(i)=bctimescal*bctime(i)
         enddo
      endif




      
c.......................................................................
cl    4. Numerical parameters
c
cl    4.1 Set non-valid flags and parameters related to CQLP
c.......................................................................

      if (cqlpmod.ne."enabled" .or. transp.ne."enabled") then
        !YuP commented:  sbdry="disabled" !YuP[2021-03-03] Why reset?
        !It is getting reset for CQLP, when transp="disabled".
        !WRITE(*,*)'ainsetva/WARNING: sbdry is reset to ',sbdry
        !YuP: Because of this reset, the problems happen in 
        !subroutine wptrmuy (which is called for meshy="fixed_mu").
        !Example: iy=80, ls=80, then in wptrmuy we get
        ! iyhtr=iymax/2 = 40, 
        ! ilshalf=ls = 80,   and then (line 67)
        ! iypass=iyhtr-ilshalf+1 = 40-80+1 = -39 ,
        ! a negative index going into mun(i).
        ! For (sbdry.eq."periodic") we have 
        ! ilshalf=ls/2+1 = 41, then iypass=0
        !Probably CQLP settings should be adjusted,
        !to have iy/2 > ls-1 ?  Not sure...
      endif

      numclas=0 !For CQL3D, just in case (numclas should not be used)
      if (cqlpmod.eq."enabled") then
        if (lsmax .ge. 5) lz=lsmax ! Here: CQLP
        numclas=nummods/10 ! Here: CQLP
        numindx=mod(mod(nummods,10),5)
        if (numclas.eq.1 .and. transp.eq."enabled") then ! Here: CQLP
          if (updown .ne. "symmetry") call wpwrng(7)
          if (sbdry .ne. "periodic") call wpwrng(8)
          !For numclas.eq.1  ==> Be sure that ls is even:
          if ((ls/2)*2 .ne. ls) call wpwrng(9) ! "ls odd with 9<nummods<20"
          if ((lz/2)*2 .ne. lz) call wpwrng(10)
          !YuP: it seems that when numclas.eq.0, we still need ls to be even;
          !     the code crashes otherwise.
          !YuP[2021-03-12] commented if (jx.lt.iy) stop 'check rhspar/bndmats storage in comm.h'
          ! See wpalloc:
          !rhspar(0:2*ls+nbanda+1,jx,2)
          !bndmats(jx,0:ls+1,nbanda,2)
          !YuP[2021-03-12] Changed jx--> max(jx,iymax)
        endif
c     So far assumes lmidpln=1, otherwise, z, sz,psi, bmidpln, etc 
c     should be redefined
c     
        if (lmidpln .ne. 1) call wpwrng(3)
        lmdpln(:)=lmidpln !Here: CQLP case,  lmidpln=1
c     parallel transport related flags
        if (transp.eq."enabled") then
          if (nonelpr .lt. nontran) nonelpr=nontran
          if (noffelpr .gt. nofftran) noffelpr=nofftran
          if (numindx.eq.2 .and. numixts.ne.-1 .and. numixts.ne.1) 
     1      call wpwrng(16)
          if (numindx.eq.4 .and. lmidvel.ne.0) call wpwrng(17)
          if (meshy.eq."fixed_mu" .and. tfac.lt.0.0) then
            !YuP added warning message.
            !Problem noticed in subr. wptrmuy 
            !(which is called for meshy="fixed_mu").
            !Example: iy=80, ls=80, then in wptrmuy we get
            ! iyhtr=iymax/2 = 40, 
            ! ilshalf=ls = 80,   and then (line 67)
            ! iypass=iyhtr-ilshalf+1 = 40-80+1 = -39 ,
            ! a negative index going into mun(i).
            ! For (sbdry.eq."periodic") we have 
            ! ilshalf=ls/2+1 = 41, then iypass=0
            !Probably CQLP settings should be adjusted,
            !to have iy/2 > ls-1 ?  
            !For now, just print a warning
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'WARNING: For CQLP, advised to have iy/2 > ls-1'
CMPIINSERT_ENDIF_RANK
          endif
        endif ! transp in CQLP

c     flags for options not yet valid with CQLP
        if (colmodl .eq. 4) call wpwrng(1)
        if (eoved .ne. 0.0) then !(default is -0.01)
CMPIINSERT_IF_RANK_EQ_0
          PRINT *,'CQLP: eoved changed to 0.0 (disabled)'
CMPIINSERT_ENDIF_RANK
          eoved=0.d0 !CQLP case
        endif
        implct="enabled"
        machine="toroidal"
        npa_diag="disabled"
        softxry="disabled"
        symtrap="disabled" !Here: CQLP
        urfmod="disabled"
c$$$        vlfmod="disabled"
        vlhmod="disabled"
        lh="disabled"
        ech="disabled"
        fw="disabled"
        syncrad="disabled"
        bremsrad="disabled"
        qsineut="disabled"
        pltlos="disabled"
        !nso=0 ! Why was it set to 0 ?
        !pltso="disabled" !Why?
        do 315 k=1,ngen
          lossmode(k)="disabled"
          torloss(k)="disabled"
          do 316 l=0,lrz
            tauegy(k,l)=0.0
 316      continue
 315    continue

        if(taunew.eq.'enabled')then !YuP[2021-03-10] added
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'WARNING: For CQLP, only taunew="disabled" is ready'
CMPIINSERT_ENDIF_RANK
         STOP 'CQLP: Use taunew="disabled" '
         !It will use subr.baviorbto 
         ![to compute tau(i,lr_) and dtau(i,l,lr_) ]
         !YuP: Not sure it is really needed for CQLP; 
         !It is used for plots of parallel distr.func.
        endif
        
      endif ! cqlpmod (CQLP part)

c.......................................................................
cl    4.2 Miscellaneous, Checks for Consistency...
c.......................................................................

      if (nstop.gt.99999) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*) 'If nstop greater than i5 format, then can'
         WRITE(*,*) 'obtain incorrect results from the code.'
         WRITE(*,*) 'Recode output of time step n for greater values.'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
         stop
      endif
         

c     This is a zero-orbit-width version of cql3d. Thus, no 
c     finite-orbit-width FOW effects included.  We keep fow
c     related namelist names for backward compatibility.
      if (fow.ne.'disabled') then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*) "This is a ZOW version of cql3d."
         WRITE(*,*) "fow reset to 'disabled'"
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
      endif
      fow='disabled' ! just in case, reset anyway


      do k=1,ngen
          if(lossmode(k).eq.'simplban' .or. 
     +       lossmode(k).eq.'simplbn1' .or. 
     +       lossmode(k).eq.'simplbn2'     ) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'For lossmode(k) banana losses, with NB it is '// 
     +   'ADVISEABLE to consistently use ndeltarho = enabled or freya'
         WRITE(*,*)'Check cqlinput_help for updates on this'
CMPIINSERT_ENDIF_RANK
          endif
          !YuP[01-2017] Adjusted/disabled simplbn1 and simplbn2 
          !These new options do not work properly yet.
          !One of problems: Failure in ngen=2 multiURF tests
          !because subr. deltar is setup/called for one general species only,
          !and so all deltar* arrays are saved for k=1 gen.species only.
          !As a temporary measure, reset to simplban:
          if(lossmode(k).eq.'simplbn1' .or. 
     +       lossmode(k).eq.'simplbn2' ) then
CMPIINSERT_IF_RANK_EQ_0
             WRITE(*,*)'lossmode(k)=simplbn1 or simplbn2 are not ready'
             WRITE(*,*)'--------- Resetting to simplban --------------'
CMPIINSERT_ENDIF_RANK
             lossmode(k)='simplban'
          endif ! YuP
      enddo ! k

c     Need to set tavg if f4d_out="tavg"
      if (f4d_out.eq."tavg" .and. tavg.eq."disabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'f4d_out=tavg, but tavg=disabled'
CMPIINSERT_ENDIF_RANK
         stop 'Need to set tavg values'
      endif

c     Expand eegy, if lrzmax.gt.1, and values at lr_=2 are zero.
c     (This is for convenience of input).

      do ny=1,negyrg-1
         do ii=1,2
            do kk=1,ngen
               if (eegy(ny,ii,kk,2).eq.zero) then
                  do l=2,lrzmax
                     eegy(ny,ii,kk,l)=eegy(ny,ii,kk,1)
                  enddo
               endif
            enddo
         enddo
      enddo


      ntotal=ngen+nmax
      if (eqmod.ne."disabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,399)
CMPIINSERT_ENDIF_RANK
 399     format(' WARNING: psimodel set =spline, for eqmod.ne.disabled')
         psimodel="spline"
      endif
      !YuP[2020-01-29] Added , for eqmod="disabled"
      if (eqmod.eq."disabled" .and. (psimodel.ne."spline")) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'For eqmod=', eqmod
         WRITE(*,399)
CMPIINSERT_ENDIF_RANK
 3999    format(' WARNING: Strongly recommended to use psimodel=spline')
         !psimodel="spline" ! YuP: Or should we enforce it?
         !For example, Some arrays that are used in ampfmod calculations
         !in case of eqmod="disabled" are only setup when psimodel="spline"
      endif !YuP[2020-01-29]


      if (yreset.eq."enabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,400) 
CMPIINSERT_ENDIF_RANK
 400     format(' WARNING: yreset set disabled, Needs recommissioning')
         yreset="disabled"
      endif
      if (transp.eq."enabled" .and. soln_method.ne."direct" .and.
     +    tfac.ge.0.) then 
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'WARNING:  Resetting tfac=-1. for transp=enabled/'
     +             //'soln_method not =direct'
CMPIINSERT_ENDIF_RANK
         tfac=-1.
      endif
      if (qsineut .eq. "enabled") locquas="disabled"
cBH100517:  Not needed.      if (lrzmax.eq.1) nrstrt=0

      !YuP[2021-02-26] was:  if (lrzmax.eq.1 .and. meshy.ne."free") then
      if (lrzmax.eq.1 .and. (meshy.ne."free") 
     &     .and.(cqlpmod.ne."enabled") ) then
         !YuP[2021-02-26] added (cqlpmod.ne."enabled"): 
         ! We need meshy=fixed_mu in case of CQLP run,
         ! and during such run we have lrzmax=1
        meshy="free"
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,'(/"  WARNING: meshy has been changed to ""free"" ",
     +    "as lrzmax=1 (does not work otherwise)"/)')
CMPIINSERT_ENDIF_RANK
      endif
      
      if (soln_method.eq."it3drv".and. meshy.ne."fixed_y") then
        meshy="fixed_y"
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,'(/"  WARNING: meshy has been changed to ""fixed_y"" ",
     +    "as soln_method=it3drv (does not work otherwise)"/)')
CMPIINSERT_ENDIF_RANK
      endif
      if (zero.lt.rovera(1) .and. rovera(1).lt.1.e-8) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,398)
CMPIINSERT_ENDIF_RANK
 398     format(' WARNING: Resetting rovera(1) to minumum 1.e-8')
         rovera(1)=1.e-8
      endif
      if (ngauss .le. 0) then
        analegco="enabled"
      else
        analegco="disabled"
      endif

      if (eqmod.eq."enabled") then
CMPIINSERT_IF_RANK_EQ_0
        if (psimodel.eq."axitorus") WRITE(*,403)
CMPIINSERT_ENDIF_RANK
 403    format('WARNING: psimodel.eq.""axitorus"", see cqlinput_help')
        if (psimodel.ne."spline") then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,401)
CMPIINSERT_ENDIF_RANK
 401      format(' WARNING: psimodel set =""spline"" '//
     +                      ' for eqmod=""enabled"" ')
          psimodel="spline"
        endif
      endif

      if (niong.eq.0 .and. sigmamod.eq."enabled") 
     +        stop 'set sigmamod=disabled'

      if (sigmamod.eq."enabled" .and. tandem.eq."enabled") then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,402)
CMPIINSERT_ENDIF_RANK
 402      format('WARNING:sigmamod set disabled,Commission tandem case')
          sigmamod="disabled"
      endif

      if(efswtch.eq."method2" .or. efswtch.eq."method3" .or.
     +   efswtch.eq."method4") then
        if(iprocur.eq."prbola-t")then !YuP[2019-10-29]added if(..."prbola-t")
        if (totcrt(1).ne.zero .and. xjc(1).eq.zero)
     +   stop 'iprocur.eq."prbola-t": inconsistent totcrt(1),xjc(1)'
        endif
        if(iprocur.eq."spline-t")then !YuP[2019-10-29]Similar for "spline-t"
        if (totcrt(1).ne.zero .and. xjin_t(1,1).eq.zero)
     +  stop 'iprocur.eq."spline-t": inconsistent totcrt(1),xjin_t(1,1)'
        endif
      endif ! efswtch.eq."method2","method3","method4"

      if(iprozeff.eq."curr_fit")then !YuP[2019-10-29]added check
      if(totcrt(1).eq.zero)then
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'For iprozeff=curr_fit, should have totcrt(1).ne.0.'
CMPIINSERT_ENDIF_RANK
        stop 'iprozeff=curr_fit but totcrt(1).eq.zero'
      endif
      endif
      
cBH170630: Changing lbdry="scale" to "consscal", following problem with
cBH170630: recent "scale" modification giving non-SS results, as pointed
cBH170630: out by Syun'ichi Shiraiwa (2017-06-23).
cBH170630: "consscal" conserves density at the v=0 boundary, and then
cBH170630: linearly rescales the distribution to obtain specified density.
         ! Note: it is not always a good idea to rescale the distr.func.
         ! Example: When there is a large-power NBI source, 
         ! comparing to the initial background 
         ! set in cqlinput. The value of xlndn00 
         ! (in ratio(k,lr_)=xlndn00(k,lr_)/runden, 
         ! which is the rescaling factor) is based 
         ! on the initial density, so it does not include particles from NBI.
         ! And the value of runden (or sden) does include all sources.
         ! In such a case, it is better to use lbdry(k)="conserv"
	 ! (Then the density of species k will increase.)
      do k=1,ngen
         if (lbdry(k).eq."scale")  then
            lbdry(k)="consscal"
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,404) !YuP[2020-12-22] print only when resetting
CMPIINSERT_ENDIF_RANK
         endif
      enddo
 404  format('WARNING:reset (as of 170630) lbdry="scale" to "consscal"')
      if (kelecg.ne.0) then
         if (redenc(1,kelecg).ne.zero .and. 
     +    lbdry(kelecg).ne."scale") then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,203)
CMPIINSERT_ENDIF_RANK
         endif
 203     format('WARNING:lbdry(kelecg).ne.scale,'//
     +        ' redenc(1,kelecg).ne.0.?')
      endif

      if (zeffc(1).ne.zero .and. 
     +   (iprozeff.ne."parabola" .and. iprozeff.ne."prbola-t" 
     &                           .and. iprozeff.ne."curr_fit") )
     +         stop 'inconsistent zeffc(1) and iprozeff'
               !BH,YuP[2019-10-31],[2020-11-01] Added iprozeff.ne."curr_fit"

      if (gamaset.eq.zero .and. (ngen.gt.1.or.kelecg.ne.1)) 
     +         stop 'inconsistent gamaset'

      if (nso.gt.nsoa) 
     +         stop 'Check nso and nsoa'

      if (efiter.eq."enabled" .and. efswtch.eq."method1") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,204)
CMPIINSERT_ENDIF_RANK
         efiter="disabled" !The only option for efswtch.eq."method1"
      endif
 204  format('WARNING: reset efiter=disabled, consistent with efswtch')

      if (efiter.eq."enabled" .and. efswtch.eq."method4") then 
         !YuP[2019-10-29] Added, for method4, setting efiter to disabled. ok???
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,204)
CMPIINSERT_ENDIF_RANK
         efiter="disabled" !The only option for efswtch.eq."method4"
      endif
      
      if (efswtch.eq."method6") then 
        if (efiter.eq."enabled" ) then 
          !YuP[2022-03-10] Added, for method6, no need in iterations
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,204)
CMPIINSERT_ENDIF_RANK
          efiter="disabled" !The only option for efswtch.eq."method6"
        endif
        if (iproelec.eq.'spline-t' .or. iproelec.eq.'prbola-t')then
        !YuP[2022-03-10] Don't use time-dependent profiles for method6
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'efswtch.eq."method6" use iproelec=parabola or spline'
CMPIINSERT_ENDIF_RANK
        STOP
        endif
      endif ! efswtch.eq."method6"

      if (noncntrl.ne.0 .and. (efswtch.ne."method2" .and.
     +     efswtch.ne."method3")) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,205)
CMPIINSERT_ENDIF_RANK
         noncntrl=0
      endif
 205  format('WARNING: resetting noncntrl=0, for efswtch as specified')

      if (negyrg.gt.negyrga) stop 'check negyrg'

      if (negyrga.gt.jx) stop 'check negyrga.le.jx, to use tam1,tam2'

      if (scatmod.eq."disabled".and.scatfrac.ne.1.) then
         scatfrac=1.
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,206)
CMPIINSERT_ENDIF_RANK
      endif
 206  format('WARNING: scatfrac set =1., for scatmod.eq.disabled')

      if (jfl.gt.jx) then
         jfl=jx ! if jfl>jx, reset
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,207)
CMPIINSERT_ENDIF_RANK
      endif
 207  format('WARNING: check jfl too big. Had to reset to jx (or jx-1)')

cBH180717:  noticed possible problem in fle, for knockon.eq.enabled:
      if ((knockon.ne."disabled").and.(nonko.le.nstop)) then
         !YuP[2020-12-22] Changed knockon.eq."enabled" to knockon.ne."disabled"
         !YuP[2020-12-22] Added (nonko.le.nstop)
         !In cqlinput, knockon could be set to 'enabled' (or other),
         !but effectively it is disabled because of large 
         !value of nonko=10000 (example).
         if (jfl.ne.jx) then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,230)
 230        format(//,'WARNING for knockon calc: Need jfl=jx in fle',//)
CMPIINSERT_ENDIF_RANK
            jfl=jx
         endif
      endif
      !YuP[2020-12-22] Moved adjustment of jfl to odd value here:
      if (mod(jfl,2).eq.0) jfl=jfl-1  
      ! jfl needed to be odd because of jpxyh=(jfl+1)/2 in pltprppr.f
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'ainsetva: jfl=', jfl
CMPIINSERT_ENDIF_RANK
            
         

      if (pltra.ne."disabled" .and. knockon.ne."enabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,208)
CMPIINSERT_ENDIF_RANK
      endif
 208  format('WARNING: Could have pltra problem, knockon.ne."enabled"')

      if (tandem.eq."enabled" .and. pltlim.ne.'x') then
         pltlim='x'
         pltlimm=1.0
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,209)
CMPIINSERT_ENDIF_RANK
      endif
 209  format('WARNING: Resetting pltlim=x, for tandem=enabled')

      if (trapmod.eq."enabled" .and. trapredc.gt.0.95) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,210)
CMPIINSERT_ENDIF_RANK
      endif
 210  format('WARNING:  Have had problems for trapredc.lt.0.99')


cBH131104:  ADD BUNCH more qualifications to use of ampfmod,
cBH131104:  ensuring not using conflicting code capabilites,
cBH131104:  for example, using with iterative electric field control.
      if (ampfmod.eq."enabled") then
      
         if(ampfadd.eq."add_bscd" .or. ampfadd.eq."neo+bscd")then
          !YuP[2019-12-26] Added ampfadd: make sure other settings 
          !are consistent with certain settings of ampfadd.
          if(bootst.ne."enabled")then
            bootst="enabled" 
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'WARNING: For ampfadd=',ampfadd
            WRITE(*,*)'  Need bootst="enabled" and jhirsh=99; Resetting'
            WRITE(*,*)'  bootst=',bootst
CMPIINSERT_ENDIF_RANK
          endif
          if(jhirsh.eq.0)then ! Maybe better check that jhirsh.ne.99 ?
            jhirsh=99 !Note: tdboothi works for jhirsh=99 or 88
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'WARNING: For ampfadd=',ampfadd
            WRITE(*,*)'  Need bootst="enabled" and jhirsh=99; Resetting'
            WRITE(*,*)'  jhirsh=',jhirsh
CMPIINSERT_ENDIF_RANK
          endif
         endif !YuP[2019-12-26] done checking ampfadd

         !do k=1,ngen
         if(soln_method.eq.'it3drv' .or. 
     +      soln_method.eq.'it3dv'      ) then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'Warning: Setting soln_method=direct: for ampfmod'
            WRITE(*,*)'      Code not presently set up for it3dv/it3drv'
CMPIINSERT_ENDIF_RANK
            soln_method='direct'
         endif
         !enddo ! k=1,ngen
         if (lrz.ne.lrzmax) then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)
            WRITE(*,*)'STOP/ampfmod: Not good to use lrz.ne.lrzmax'
            WRITE(*,*)
CMPIINSERT_ENDIF_RANK
            stop
         endif
cBH131104:  NEED to increase boundary options/include t-dep Vphib.
         if ((iproelec.eq.'parabola' .or. iproelec.eq.'prbola-t').or.
     +        (iproelec.eq.'spline')) then
            !YuP[2019-10-31] iproelec.eq.'spline-t' is not setup for ampfmod
            if (efswtch.eq."method1".and.tmdmeth.eq."method1") then
               continue
            else
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'ampfmod case only setup for efswtch=method1'
               WRITE(*,*)'STOP1: Incorrectly specd boundary elec fld'
CMPIINSERT_ENDIF_RANK
               stop 'STOP1/ampfmod'
            endif
         else
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'ampfmod setup for iproelec=parabola,spline,prbola-t'
            WRITE(*,*)'STOP2: Incorrectly specd boundary elec fld'
CMPIINSERT_ENDIF_RANK
            stop 'STOP2/ampfmod'
         endif

         if (iproelec.eq.'spline' .or.iproelec.eq.'spline-t') then
            if (abs(ryain(1)).gt.em10.or.
     +          abs(ryain(njene)-1.d0).gt.em10) then
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'STOP Incorrectly specd ryain for ampfar case'
CMPIINSERT_ENDIF_RANK
               stop
            endif
         endif

         if (efflag.ne."toroidal") then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'STOP: AmpFar assumes toroidal elec fld calc'
CMPIINSERT_ENDIF_RANK
            stop
         endif

         if (nlrestrt.eq."ncdfdist" .and. elecscal.ne.one) then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'STOP: AmpFar restart assumes elecscal=1.'
CMPIINSERT_ENDIF_RANK
            stop
         endif

         if (nlrestrt.eq."ncdfdist" .and. nonampf.ne.0) then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'WARNING: AmpFar restart assumes nonampf=0'
CMPIINSERT_ENDIF_RANK
            nonampf=0
         endif
     
      endif  !On ampfmod
      
      if (ampfmod.eq."enabled".and.cqlpmod.eq."enabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'STOP: ampfmod.ne.enabled with cqlpmod.eq.enabled'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
         stop
      endif

      if (nlrestrt.eq."ncdfdist" .or. nlrestrt.eq."ncregrid") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'WARNING: Check that did not have netcdfshort='
         WRITE(*,*)'         longer_f or lngshrtf'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
      endif

      if(nlrestrt.ne."disabled")then 
         !YuP[2019-07-15] Added check/reset of chang and urfdmp settings.
         !With present definition of jchang() in subr. coefwti,
         !in case of chang.eq."noneg", the values of jchang may not be always
         !updated for each distr.function at a given time step.
         !Instead, they can be taken from previous time steps.
         !So, for a restart run, these values of jchang should be saved 
         !together with values of distr.function, for the subsequent restart.
         !This is not done presently.
         !Until the values of jchang are saved for a restart, 
         !or until the subr.coefwti is changed (for example, by enforcing
         ! updating jchang values at every time step),
         !here we simply reset the chang option to "enabled".
         !For chang="enabled", the values of jchang are all =jx.
         if(chang.eq."noneg")then 
            chang="enabled" ! Reset, for restart runs
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'chang=noneg is not compatible with restart option'
         WRITE(*,*)'Resetting to enabled'
         WRITE(*,*)'Make sure that all initial runs also had same chang'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
         endif
         !----------
         !YuP[2019-07-15] Check/reset urfdmp suitable for a restart run.
         ! Subr.urfavg (called for urfdmp="secondd") is not suitable
         ! for a restart run, because it requires averaging of f()
         ! over 3 time steps. So, if we want a restart run, 
         ! we need to save data from those 3 time steps,
         ! or at least save the g_() function (set in urfavg) into *.nc file.
         ! So, in case of restart it is better to use urfdmp='firstd' ;
         ! then this subr. is not called.
         if(urfdmp.eq."secondd")then 
            urfdmp="firstd" ! Reset, for restart runs
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'urfdmp=secondd is incompatible with restart option'
         WRITE(*,*)'Resetting to firstd'
         WRITE(*,*)'Make sure all initial runs also had urfdmp=firstd '
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
         endif
      endif

      if (nlwritf.ne."disabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'WARNING: If this run to be restarted, ensure that'
         WRITE(*,*)'         netcdfshort .ne. longer_f or lngshrtf.'
         WRITE(*,*)'         Only save f at last time step'
         WRITE(*,*)'WARNING: Also, chang=noneg is incompatible '
         WRITE(*,*)'         with restart option.  Use chang=enabled '
         WRITE(*,*)'WARNING: Also, urfdmp=secondd is incompatible '
         WRITE(*,*)'         with restart option.  Use urfdmp=firstd '
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
         if (netcdfshort.eq."longer_f") netcdfshort="disabled"
         if (netcdfshort.eq."lngshrtf") netcdfshort="disabled"
      endif

      if (eseswtch.eq."enabled") then
         if (.not.(cqlpmod.eq."enabled".and. sbdry.eq."periodic")) then
            stop 'Check use of eseswtch'
         elseif (efswtch.ne."disabled") then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,211) 
CMPIINSERT_ENDIF_RANK
 211        format('WARNING:  Re-setting efswtch=disabled, for now')
            efswtch="disabled"
         endif
      endif

      if (bootcalc.ne."disabled".and.lrzdiff.ne."disabled") then
         stop 'bootcalc.ne.disabled .and. lrzdiff.ne.disabled'
      endif

      if (efswtch.eq."method5") then
         if (eqmod.ne."enabled") stop 'Efswtch=method5,eqmod.ne.enabled'
         if (eqsource.ne."eqdsk") then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,216)
CMPIINSERT_ENDIF_RANK
 216        format('WARNING: efswtch=method5 with eqsource.ne.eqdsk',/,
     +           'Not checked out')
         endif
         if (efflag.ne."parallel") then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,219)
CMPIINSERT_ENDIF_RANK
 219        format('WARNING: efswtch=method5 with efflag.ne.parallel',/,
     +           'Could cause problems with RFP.  See cqlinput_help.')
         endif
      endif

      if (.not.(eqmod.eq."enabled" .and. eqsource.eq."eqdsk"))  then
         if (eqsym.eq."none") then
CMPIINSERT_IF_RANK_EQ_0
         !WRITE(*,*)
         !WRITE(*,*)'WARNING:  Re-setting eqsym="average":'  
         !WRITE(*,*)'          It should not ="none" for eqmod/eqsource'
         !WRITE(*,*)
         !YuP[2020] the above message is obsolete; Now can handle any eqsym
CMPIINSERT_ENDIF_RANK
         endif
      endif
      

      !----- For Miller Equilibrium:   ---------------------------------
c**   REF: R.L. Miller et al., "Noncircular, finite aspect ratio, local
c**   equilibrium model", Phys. Plasmas, Vol. 5, No. 4, April 1998.
c**   Setup is done similar to COGENT version (MillerBlockCoordSysF.ChF)  
c**   The difference is in units: COGENT uses [Tesla, meters],
c**   while CQL3D uses [Gauss, cm]. 
      if (eqsource.eq."miller")  then
      
         if(eqmod.ne."enabled")then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)
            WRITE(*,*)'For Miller equilibr.: Set eqmod="enabled" '
CMPIINSERT_ENDIF_RANK
            stop
         endif
      
         if(fpsimodl.ne."constant") then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)
            WRITE(*,*)'For Miller equilibr.: Set fpsimodl="constant" '
CMPIINSERT_ENDIF_RANK
            stop
         endif

         if(eqsym.ne."none") then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)
            WRITE(*,*)'For Miller equilibr.: Set eqsym="none" '
            ! YuP: Need to check, maybe eqsym='avg_zmag' is also ok?
CMPIINSERT_ENDIF_RANK
            stop
         endif

CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'eq_miller_rmag=',eq_miller_rmag ! Magnetic axis: major radius coord [cm]
         WRITE(*,*)'eq_miller_zmag=',eq_miller_zmag ! Magnetic axis: vertical coord [cm]
         WRITE(*,*)'eq_miller_btor=',eq_miller_btor ! Tor field at Geom. center of LCFS [Gauss]
         WRITE(*,*)'eq_miller_radmin=',eq_miller_radmin   ! Plasma minor radius [cm]
         WRITE(*,*)'eq_miller_cursign=',eq_miller_cursign  ! Sign of Plasma Current [+1. or -1.]
         WRITE(*,*)'eq_miller_psimag=',eq_miller_psimag ! Pol.flux at magn.axis [cgs] Set as positive
         WRITE(*,*)'eq_miller_psilim=',eq_miller_psilim ! Pol.flux at LCFS
         WRITE(*,*)'eq_miller_psi_n=',eq_miller_psi_n ! n and m powers for PSI(r) profile as in 
         WRITE(*,*)'eq_miller_psi_m=',eq_miller_psi_m ! PSI(r)= psilim + (psimag-psilim)*(1-(r/a)^n)^m
         WRITE(*,*)'eq_miller_deltaedge=',eq_miller_deltaedge ! Triangularity of LCFS (at r=radmin)
         WRITE(*,*)'eq_miller_kappa=',eq_miller_kappa  ! Vertical elongation (const for all surfaces)
         WRITE(*,*)'eq_miller_drr0=',eq_miller_drr0  ! dR0/dr  we assume Shafr.shift=-drr0*r
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK

         if(abs(eq_miller_rmag).ge. 1.d99) then
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)'Set eq_miller_rmag in cqlinput' 
CMPIINSERT_ENDIF_RANK
           stop ! stop at all cores
         endif

         if(eq_miller_psimag.le.eq_miller_psilim)then
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)'Set eq_miller_psimag > eq_miller_psilim' 
CMPIINSERT_ENDIF_RANK
           stop ! stop at all cores
         endif
         
         if(abs(eq_miller_btor).ge. 1.d99) then
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)'Set eq_miller_btor in cqlinput' 
CMPIINSERT_ENDIF_RANK
           stop ! stop at all cores
         endif

         if(abs(eq_miller_radmin).ge. 1.d99) then
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)'Set eq_miller_radmin in cqlinput' 
CMPIINSERT_ENDIF_RANK
           stop ! stop at all cores
         endif

         if(abs(eq_miller_psimag).ge. 1.d99) then
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)'Set eq_miller_psimag in cqlinput' 
CMPIINSERT_ENDIF_RANK
           stop ! stop at all cores
         endif

         if(abs(eq_miller_psilim).ge. 1.d99) then
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)'Set eq_miller_psilim in cqlinput' 
CMPIINSERT_ENDIF_RANK
           stop ! stop at all cores
         endif

         if(eq_miller_drr0.gt. 0.d0) then
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)'Set eq_miller_drr0 as a negative value' 
CMPIINSERT_ENDIF_RANK
           stop ! stop at all cores
         endif

      endif
      ! See subr. eq_miller() for definition of surfaces and fields.
      !------------------------------------------------------------------
      
      

      eqmirror="disabled"
      if( eqsource.eq."eqdsk".and.machine.eq."mirror") then
         eqmirror="enabled"
      else
         eqmirror="disabled"
      endif
      if (eqsource.eq."mirror1" .or. eqmirror.eq."enabled") then ! YuP[03-2016]
       STOP 'eqsource=mirror1 is not available in this CQL3D version'
      endif ! (eqsource.eq."mirror1" .or. eqmirror.eq."enabled")


      if (nv.gt.nva) then
         stop 'SXR: nv .gt. nva.  Also check length of XR input arrays.'
      endif

      if (transp.eq."enabled" .and. relaxden.ne.1.) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,217)
CMPIINSERT_ENDIF_RANK
 217     format('WARNING: transp.eq.enabled .and. relaxden.ne.1.',/,
     +      '     relaxden defn changed from divisor to multiplier,',/,
     +      '     BH: April 23, 2001')
      endif

c     Checking rya input (rzset.ne."disabled"):
      if (rzset.ne."disabled") then
      if (rya(0).ne.zero) then
         stop 'rya(0).ne.0. ==>  error in namelist input: rya(1)=?'
      endif
      if (lrz.gt.1) then
         do ll=2,lrz
            if ( (rya(ll)-rya(ll-1)).le.0. ) stop 'Check rya() input'
         enddo
      endif
      endif

CMPIINSERT_IF_RANK_EQ_0
      if (transp.eq."enabled" .and. 
     +   (pinch.ne."simple"  .or. pinch.ne."simplen" .or. 
     +   pinch.ne."case1"  .or. pinch.ne."case1n" .or. 
     +   pinch.ne."case2"  .or. pinch.ne."case2n" .or. 
     +   pinch.ne."case3"  .or. pinch.ne."case3n" )) WRITE(*,218)
CMPIINSERT_ENDIF_RANK
 218  format('WARNING: transp=enabled, Check settings of pinch')

      if (transp.eq."enabled" .and. radcoord.ne."sqtorflx") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'WARNING: transp=enabled not setup for '//
     +                   'radcoord.ne.sqtorflx'
CMPIINSERT_ENDIF_RANK
c         stop
      endif
      

      if (soln_method.eq.'itsol'.or. soln_method.eq.'itsol1' .or.
     +    soln_method.eq.'it3dv'.or. soln_method.eq.'it3drv') then
         if (cqlpmod.eq.'enabled' .or.
     +       symtrap.ne.'enabled' ) then
CMPIINSERT_IF_RANK_EQ_0
                WRITE(*,*)'ainsetva: Check/re-configure itsol storage'//
     +                    ' for cqlpmod/symtrap/nadjoint.ne.0 cases'
CMPIINSERT_ENDIF_RANK
                STOP
         endif
      endif

c     Necessary for some code logic:
      if (transp.eq."disabled" .and. soln_method.eq.'it3drv') then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'ainsetva: If transp.eq.disabled, then soln_method'
         WRITE(*,*)'ainsetva: should eq direct,itsol,itsol1, or it3dv'
         WRITE(*,*)'ainsetva: Resetting soln_method it3drv ==> it3dv'
CMPIINSERT_ENDIF_RANK
         soln_method='it3dv'
      endif

c     Check on consistency of urfmod
      ikrf=0
      if (urfmod.ne."disabled" .and. lh.eq."disabled" .and.
     +    ech.eq."disabled" .and. fw.eq."disabled") then
         ikrf=0
      else
         ikrf=1
      endif
      if (urfmod.ne."disabled") then
         do krf=1,nmodsa
cBH100829            if (rftype(krf).eq."notset") ikrf=ikrf+1
            if (rftype(krf).ne."notset") ikrf=ikrf+1
         enddo
      endif
      if (ikrf.eq.0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'Warning: setting urfmod=disabled, since'
         WRITE(*,*)'         lh, ech, and fw, =disabled, '
         WRITE(*,*)' and all rftype(1:nmodsa)= notset'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
         urfmod="disabled"
      endif
      do krf=1,nmodsa
         if (nrfspecies(krf).gt.ngen) then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*) 'Check nrfspecies.lt.ngen'
CMPIINSERT_ENDIF_RANK
            stop
         endif
      enddo
         

c     Checking consistency of netcdfvecXX settings:
      if (netcdfvecc.ne."disabled") then
         if (netcdfvece.ne."disabled" .and. netcdfvece.ne.netcdfvecc)
     +                     stop 'STOP:check netcdfvecXX for consistency'
         if (netcdfvecrf.ne."disabled" .and. netcdfvecrf.ne.netcdfvecc)
     +                     stop 'STOP:check netcdfvecXX for consistency'
         if (netcdfvecal.ne."disabled" .and. netcdfvecal.ne.netcdfvecc)
     +                     stop 'STOP:check netcdfvecXX for consistency'
      endif
      if (netcdfvece.ne."disabled") then
         if (netcdfvecrf.ne."disabled" .and. netcdfvecrf.ne.netcdfvece)
     +                     stop 'STOP:check netcdfvecXX for consistency'
         if (netcdfvecal.ne."disabled" .and. netcdfvecal.ne.netcdfvece)
     +                     stop 'STOP:check netcdfvecXX for consistency'
      endif

      if (netcdfvecrf.ne."disabled") then
         if (netcdfvecal.ne."disabled" .and. netcdfvecal.ne.netcdfvecrf)
     +                     stop 'STOP:check netcdfvecXX for consistency'
      endif

         
      if (frmodp.ne."disabled") then
         if (nso.ne.1) stop 'STOP: frmodp.ne.disabled, NEED nso=1'
         if (nbeamsp.gt.kb) stop 'STOP:frmod case, NEED nbeamsp.le.kb' 
         do ib=1,nbeamsp
            if (kfrsou(ib).eq.0 .or. kfrsou(ib).gt.ngen) 
     +          stop 'STOP: frmod case, NEED to reset kfrsou(1:nbeamsp)'
         enddo
c        Check kfrsou() for the beam lines properly grouped
         kbsp=1   !Number of different beamline species
         nbeams1p(1)=1 !Number of beamlines for each species
         kblspecp(1)=kfrsou(1) !spec index for each nbeams1p beamline set
         do ib=2,nbeamsp
            if (kfrsou(ib).eq.kfrsou(ib-1)) then
               nbeams1p(kbsp)=nbeams1p(kbsp)+1
            elseif(kfrsou(ib).ne.0) then
               kbsp=kbsp+1      !increment number of beamline species
               kblspecp(kbsp)=kfrsou(ib)
               nbeams1p(kbsp)=nbeams1p(kbsp)+1
            else                !kfrsou(ib).eq.0
               go to 1
            endif
         enddo
 1       continue
         
         if (kbsp.gt.1) then
            do ks=1,kbsp
               do kss=ks+1,kbsp
                  if(kblspecp(kss).eq.kblspecp(ks)) then
                     STOP 'kfrsou is improperly ordered'
                  endif
               enddo
            enddo
         endif

      endif  !On frmodp

c     In view of checking settings in sub frset, this condition
c     should not (cannot) occur.
      if (beamplsep.ne."disabled") then
         if (frmodp.eq."disabled")
     +        stop 'STOP:frmod.eq.disabled but beamplse.ne.disabled'
      endif
         
c     Checking consistency of rdcmod and lrzdiff
      if (rdcmod.ne."disabled" .and. lrzdiff.ne."disabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'ainsetva:  STOP, rdcmod requires lrzdiff=disabled'
CMPIINSERT_ENDIF_RANK
         stop
      endif

c     Make sure nrdc is not too large.
      if (nrdc.gt.nrdca) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*) "STOP: Need to increase nrdca in param.h"
CMPIINSERT_ENDIF_RANK
         stop
      endif

c     Make sure nrdc=1, for rdcmod="format2"
      if (rdcmod.eq."format2" .and. nrdc.ne.1) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*) "STOP: Need nrdc=1 for rdcmod=format2"
         WRITE(*,*) "      Else, need to recode"
CMPIINSERT_ENDIF_RANK
         stop
      endif

c     Adjusting rdcscale(1) for backwards compatibility
      if (rdcscale(1).eq.1.d0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'**** ainsetva: resetting rdcscale(1)=pwrscale(1) **'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
         rdcscale(1)=pwrscale(1)
      endif

c     Soft X-ray
      if (softxry.ne."disabled" .and. kelecg.eq.0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'**** ainsetva: Inappropriate calc of SXR ******'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
      endif

c     Check rd(),x_sxr(),z_sxr() for previous input method, in
c     which they were scalars, i.e., only one initial position was
c     specified.  Fill in 2:nv, if necessary.
      if (nv.gt.1) then

         if (x_sxr(1).eq.zero .and. z_sxr(1).eq.zero) then  !rd input
            if (rd(2).eq.zero) then
               do i=2,nv
                  rd(i)=rd(1)
                  thetd(i)=thetd(1)
               enddo
            endif
         else  !x_sxr,z_sxr input
            if (x_sxr(2).eq.zero .and. z_sxr(2).eq.zero) then
               do i=2,nv
                  x_sxr(i)=x_sxr(1)
                  z_sxr(i)=z_sxr(1)
               enddo
            endif
         endif

      endif  ! on nv



c     NPA
      if (npa_diag.ne."disabled" .and. niong.eq.0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'**** ainsetva: Inappropriate calc of NPA ******'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
      endif
      if (npa_diag.ne."disabled" .and. softxry.ne."disabled") then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'ainsetva: STOP'
         WRITE(*,*)'ainsetva: Code not complete set up for simultaneous'
         WRITE(*,*)'ainsetva: calc of SXR and NPA.  Need separate'
         WRITE(*,*)'ainsetva: storage for mx related variables, unless'
         WRITE(*,*)'ainsetva: the related data is recalculated for each'
         WRITE(*,*)'ainsetva: use of the diagnostics.'
CMPIINSERT_ENDIF_RANK
         STOP
      endif

c     Check rd_npa(),x_npa(),z_npa() for previous input method, in
c     which they were scalars, i.e., only one initial position was
c     specified.  Fill in 2:nv_npa, if necessary.
      if (nv_npa.gt.1) then

         if (x_npa(1).eq.zero .and. z_npa(1).eq.zero) then  !rd input
            if (rd_npa(2).eq.zero) then
               do i=2,nv_npa
                  rd_npa(i)=rd_npa(1)
                  thetd_npa(i)=thetd_npa(1)
               enddo
            endif
         else  !x_npa,z_npa input
            if (x_npa(2).eq.zero .and. z_npa(2).eq.zero) then
               do i=2,nv
                  x_npa(i)=x_npa(1)
                  z_npa(i)=z_npa(1)
               enddo
            endif
         endif

      endif  ! on nv_npa



c.......................................................................
c     Check on structure of integer*1, if (urfmod.ne."disabled").
c     The following has been set up for the Absoft fortran
c     compiler.  Other compilers may be different.
c.......................................................................
!      if (urfmod.ne."disabled") then
!         ip0=-128
!         ip255=127
!         iu0=ip0+128
!         iu255=ip255+128
!         if (iu0.ne.0 .or. iu255.ne.255) then
!            stop 'check pack unpack in urfpkun.f'
!         endif
!      endif
! BH,YuP[2020-12-18] The above few lines are no longer needed: 
! switched to pack16/unpack16, which uses unteger*2
	

c.......................................................................
cl    4.3 Warnings about Changed Use of Variables
c         (For example, previously used as numeric and character
c          input.
c.......................................................................

CMPIINSERT_IF_RANK_EQ_0
      do k=1,ngen
         if (fpld(1,k).ne.zero) WRITE(*,212)
      enddo
 212  format(//,'WARNING: Have changed namelist input options',/,
     +           8x,    ' fpld_dsk ==> -1.0',/,
     +           8x,    ' fpld_ds1 ==> -2.0',//)

      if (izeff.ne."backgrnd") WRITE(*,213)
 213  format(//,'WARNING:  Check use of izeff. See cqlinput_help',//)
CMPIINSERT_ENDIF_RANK

      if (nconteq.ne."psigrid".and.nconteqn.eq.0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,214)
CMPIINSERT_ENDIF_RANK
 214     format(//,'Must set nconteqn, if nconteq.ne.psigrid',//)
         stop 'Must set nconteqn; See cqlinput_help'
      endif

      if (nconteq.eq."psigrid".and.nconteqn.ne.0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,215)
CMPIINSERT_ENDIF_RANK
 215     format(//,'Must reset nconteq, if nconteqn.ne.0',//)
         stop 'Must reset nconteq: See cqlinput_help'
      endif

      if (nconteq.eq."psigrid".and.nconteqn.ne.0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,221)
CMPIINSERT_ENDIF_RANK
 221     format(//,'NO LONGER RECOMMENDED: See cqlinput_help (BH171211)'
     1        ,//)
      endif


c.......................................................................
cl    5. Plot flags
c.......................................................................

c     could be problems if nrskip.eq.0 and the irzplt() are not
c     all distinct (or zero), or are .gt. lrz.
      if (nrskip.eq.0) then
         do i=1,lrors
            if (irzplt(i).gt.lrors) then
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'WARNING: irzplt(i).gt.lrors are set =0, i=',i
CMPIINSERT_ENDIF_RANK
               irzplt(i)=0
            endif
         enddo
         do i=1,lrors-1
            do j=i+1,lrors
               if (irzplt(j).eq.irzplt(i) .and.
     +             irzplt(j).ne.0 ) then
CMPIINSERT_IF_RANK_EQ_0
                  WRITE(*,*)
     +                   'WARNING: Non-distinct irzplt(j) set =0, j=',j
CMPIINSERT_ENDIF_RANK
                  irzplt(j)=0
               endif
            enddo
         enddo
      endif

      if (eqmod.ne."enabled") pltvs="rho"
      if (pltvs.ne."rho".and. pltvs.ne."psi") pltvs="rho"
      nirzplt=0  ! keeps count of number of radial plot positions

c Having an mplot (compiler?) problem on viz machine:
c      write(*,*)'ainsetva before nrskip setting mplot: ll=1:lrors,mplot'
c      do ll=1,lrors
c         write(*,*) ll,mplot(ll)
c      enddo

      do 500 ll=1,lrors
        if (nrskip .ne. 0) then
          if ((ll-1) .eq. (ll-1)/nrskip*nrskip) then
             mplot(ll)="enabled"
             nirzplt=nirzplt+1
          endif
        else
          do 510 lu=1,lrors
            if (ll .eq. irzplt(lu)) then
               mplot(ll)="enabled"
               nirzplt=nirzplt+1
            endif
 510      continue
        endif
 500  continue
      if (lrors .eq. 1) then
         mplot(1)="enabled"
         nirzplt=1
      endif


      do k=1,ngen
        if (   lbdry(k).ne. "conserv".and. lbdry(k).ne."consscal"
     +   .and. lbdry(k).ne. "fixed"  .and. lbdry(k).ne."scale" 
     +   .and. lbdry(k).ne. "conscalm") then
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)"lbdry(k)=",lbdry(k)
        WRITE(*,*)"lbdry(k) can only be 'conserv', 'consscal'," // 
     +  "'fixed', 'scale' or 'conscalm' "
CMPIINSERT_ENDIF_RANK
           STOP
        endif
      enddo

c Part of printed output re viz problem
c      write(*,*)'ainsetva after nrskip: nirzplt= ',nirzplt
c      write(*,*)'ainsetva after nrskip:   ll=1:lrors,mplot '
c      do ll=1,lrors
c         write(*,*) ll,mplot(ll)
c      enddo

      return
      end
