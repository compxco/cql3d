c
c
      subroutine ainspec
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     Determine 
c     kelecg = the index of the general species electrons (0,if none). 
c     kelecm = the background Maxwellian species electrons (0,if none).
c     kionn =  1st ionic species (be it general or background species).
c     kelec = kelecg (if kelecg>0) or  kelec = kelecm (if kelecg=0)
c     niong = number of general species ions
c     kiong(1:niong) = index of general ion specie (otherwise 0 for 1:ngen)
c     kiongg(k=1:ngen) = k if an ions species, otherwise 0.
c     nionm = number of ion background Maxwellian species.
c     kionm(1:nionm) = Maxwellian ion specie indices (otherwise 0).
c
c     There must be at least one electron and  
c     one ionic species present.
c
c    From cqlinput_help:
c    In the following, the index (k) refers to the species.
c    If 1 .le. k .le. ngen, then k is a general (time-advanced) species.
c    k.gt.ngen means species k is background Maxwellian.
c    In k order, the general (non-Maxwellian time advanced species) 
c    come before the background species. Electrons and
c    ions can be mixed, and any species can be represented
c    simultaneously as a general and a background species.
c    If there is more than one general species ions, they must 
c    not be separated in k by the electron species 
c    (i.e., put general electrons at front or back of 
c    the general ion species).
c    If there is more than one Maxwellian ion, then they must
c    not be separated in k by the electron species.   If more than
c    one of the ion species has the same charge number bnumb(), (e.g.,
c    H+, D+, and/or T+) then they should be placed successively at the
c    beginning of the Maxwellian ion species.

c.......................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      ntotal=ngen+nmax

      kelecg=0
CDIR$ NEXTSCALAR
      do 1000 k=1,ngen
        if (fmass(k) .lt. 1.d-27) then
          kelecg=k
          goto 1001
        endif
 1000 continue
 1001 continue
      kelecm=0
CDIR$ NEXTSCALAR
      do 1002 k=ngen+1,ntotal
        if (fmass(k) .lt. 1.d-27) then
          kelecm=k
          goto 1003
        endif
 1002 continue
 1003 continue
      if (kelecg.gt.0) then
        kelec=kelecg
      else
        kelec=kelecm
      endif
cBH180908      if (kelec.eq.0) call diagwrng(9)
      if (kelec.eq.0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*) 'WARNING: Unphysical plasma, only one species.'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
      endif

         
      kionn=0
CDIR$ NEXTSCALAR
      do 1005 k=1,ntotal
        if (fmass(k).gt.1.d-27) then
          kionn=k
          goto 1006
        endif
 1005 continue
 1006 continue
      do 1007 k=1,ngen
        kspeci(2,k)="general"
 1007 continue
      do 1008 k=ngen+1,ntotal
        kspeci(2,k)="maxwell"
 1008 continue

      niong=0
      do 1009  k=1,ngena
        kiong(k)=0
        kiongg(k)=0
 1009 continue
c     Electron mass is 9.1e-28 grams. Use this to distinguish ions.
      do 1010  k=1,ngen
        if (fmass(k).gt.1.d-27 .and. kspeci(2,k).eq."general") then
          niong=niong+1
          kiong(niong)=k
          kiongg(k)=k
        endif
 1010 continue
      nionm=0
      do 1011 k=1,nmaxa
        kionm(k)=0
 1011 continue
      do 1012 k=ngen+1,ntotal
        if (fmass(k).gt.1.d-27 .and. kspeci(2,k).eq."maxwell") then
          nionm=nionm+1
          kionm(nionm)=k
        endif
        if(fmass(k).eq.1.e-29)then !YuP[2022-05-31] added warning
          !Note that fmass(k)=1.e-29 is set in aindflt
          !It is possible that user sets nmax=2 (and ngen=1, for example),
          !while the intention is to use only one Maxw.species,
          !and then the values are not set for fmass(k=3)
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)
          WRITE(*,*)'WARNING: fmass/bnumb not set in cqlinput for k=',k
CMPIINSERT_ENDIF_RANK
          !Maybe better stop the run? 
          STOP 'fmass/bnumb not set in cqlinput for this k; check nmax'
        endif
 1012 continue

cBHTemp      if (kionn.eq.0) call diagwrng(9)
      if (kionn.eq.0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*) 'WARNING: Unphysical plasma, only one species.'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
      endif

CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)
      WRITE(*,*)'ainspec: ngen,nmax,ntotal',
     +                    ngen,nmax,ntotal
      WRITE(*,*)'ainspec: kelecg,kelecm,kelec',
     +                    kelecg,kelecm,kelec
      WRITE(*,*)'ainspec: niong,kiong(1:niong)',niong,kiong(1:niong)
      if(nionm.gt.0)then ![2022-05-26] added nionm condition
        WRITE(*,*)'ainspec: nionm,kionm(1:nionm)',nionm,kionm(1:nionm)
      else
        WRITE(*,*)'ainspec: nionm=',nionm !Can be no Maxw.ions (only general)
      endif
      WRITE(*,*)'ainspec: nionm/kionm may be adjusted upwards when'
      WRITE(*,*)'ainspec:   iprozeff.ne."disabled"; see cqlinput_help'
      WRITE(*,*)
CMPIINSERT_ENDIF_RANK


      !YuP[2022-05-31] Added more verification for species.
      !In the list of kg (general sp.) and list of km (Maxw.sp.)
      !identify same species (in km list, the species 
      !can be repeated in case of colmodl=3 runs)
      if(.NOT.ASSOCIATED(km_same)) then
         allocate(km_same(1:ntotal),STAT=istat) ![2022-05]
         km_same=0
         allocate(kg_same(1:ntotal),STAT=istat) ![2022-05]
         kg_same=0
      endif
      do kg=1,ngen ! General species list
      do km=ngen+1,ntotal ! Maxwellian list
         if(bnumb(kg).eq.bnumb(km))then
         if(abs(fmass(kg)-fmass(km)).lt.1d-3*(fmass(kg)+fmass(km)))then
              ! 'kg' and 'km' are same species.
              km_same(kg)=km ! pointing from kg to km
              kg_same(km)=kg ! pointing from km to kg
         endif
         endif
      enddo
      enddo
      !If none of same-species are found, km_same and kg_same remain =0
      
      if(colmodl.eq.0)then !Fully non-linear: there should be no duplicates
        ! of general species in km list (Maxwellian list).
        if( sum(km_same).ne.0 ) then ! or sum(kg_same).ne.0
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'For colmodl=0, no duplicate species are allowed.'
          WRITE(*,*)'Remove duplicate species from Maxwellian list,'
          WRITE(*,*)' or set kfield(k) to "disabled" '
CMPIINSERT_ENDIF_RANK
          !stop 'ainspec: colmodl=0'
          do kg=1,ngen ! General species list
             km=km_same(kg) !Duplicate species in Maxwellian list
             kfield(km)='disabled'
CMPIINSERT_IF_RANK_EQ_0
             WRITE(*,*)'For km=',km,' resetting kfield(km) to disabled'
CMPIINSERT_ENDIF_RANK
          enddo
        endif 
      endif

      if(colmodl.eq.3)then !Using P0 from Maxwellian duplicate.
        !Must have EACH general species duplicated in Maxw.list
        ![presently, colmodl is NOT a func. of species number].
        do kg=1,ngen
          if (km_same(kg).eq.0) then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'For colmodl=3 '
            WRITE(*,*)'add Maxwellian duplicate for general species=',kg
CMPIINSERT_ENDIF_RANK
            stop 'ainspec: Add duplicate species to Maxwellian list'
          endif
        enddo
      endif
      !YuP[2022-01-12] Done/Added more verification for species.

      return
      end
