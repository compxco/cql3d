c
c
      subroutine frinitz(
     +        nprim,nimp,nion,ibion,namep,namei,atw,fd,smth,nbeams)
      implicit integer (i-n), real*8 (a-h,o-z)
c
      include 'param.h'
      include 'comm.h'    ! <<<<<==========
CMPIINSERT_INCLUDE

c     nprim,nimp,fd,smth,nbeams are in variables.
c     nion,ibion,namep,namei,atw are out variables.

c..................................................................
c     This routine is called by frinitl which includes frcomm,
c     after reading the frsetup namelist.
c     It communicates between common blocks of CQL3D and the
c     segregated CRAY32 (NFREYA ONETWO version) frcomm.h common blocks,
c     by passing values through the subroutine arguments.
c
c     smooth_ in comm.h is set from smooth in frcomm.h, passed
c       through argument smth.
c     fd specified in frinitl passed here to give a beam dt mixture ratio.
c     nprim,nimp,nion,ibion,namep,namei,atw are in frcomm.h, not comm.h.
c     These variables are determined from comm.h data, and passed to 
c     frcomm.h through the arguments of frinitz.
c       
c..................................................................

      character*8 namep,namei
      dimension namep(*),namei(*),atw(*),ibion(*)

      smooth_=smth

c.....................................................................
c     Determine the index of the general species electrons, the
c     background Maxwellian species electrons, and the first ionic
c     species. There must be at least one electron species and at least
c     one ionic species because the code forces quasineutrality.
c   
c     NOTE: Check for overlap with ainspec.f  (BobH, 990704).
c.....................................................................

      kelecg=0
CDIR$ NEXTSCALAR
      do 1000 k=1,ngen
        if (fmass(k) .lt. 1.e-27) then
          kelecg=k
          goto 1001
        endif
 1000 continue
 1001 continue
CDIR$ NEXTSCALAR
      do 1002 k=ngen+1,ntotal
        if (fmass(k) .lt. 1.e-27) then
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
         WRITE(*,*) 'WARNING: Unphysical plasma: no electrons?'
         WRITE(*,*) '         Have not checked with NBI turned on.'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
      endif


CDIR$ NEXTSCALAR
c     Find first ion species (general or maxwellian).
      do 1005 k=1,ntotal
        if (fmass(k).gt.1.e-27) then
          kionn=k
          goto 1006
        endif
 1005 continue
 1006 continue

c..................................................................
c     Define nprim, nimp and nion as in ONETWO. This involves getting
c     rid of possible electron species which may reside in CQL3D
c     Recall nimp is the number of impurity species input in
c     namelist frstup. ntotal is the total number of species.
c     nprim is the number of primary species, and this number is
c     also input in frstup.
c..................................................................
      nion=nprim+nimp
c..................................................................
c     Now define the namep and namei array as used in ONETWO
c..................................................................

CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*) 'frinitz: nprim,nimp = ',nprim,nimp
CMPIINSERT_ENDIF_RANK

      do ib=1,nbeams  ![2025-12-16] fixing an issue related to kfrsou
       if( kfrsou(ib).eq.0 )then
         !For each beam 'ib' there should be an associated species 'k'.
         !In old version of CQL3D the namelist variable kfrsou was a scalar,
         !and its value was applied for all beams.
         !Now kfrsou() is a vector, 
         !and default values are set to kfrsou(:)=0.
         !Therefore, old cqlinput where several beams are used, 
         !and kfrsou (as a scalar) is set to 2 
         !(which means - 1st and 2nd beams are pointing to species k=2)
         !would fail now.
         !Check and adjust kfrsou(ib) for such cases.
         if(ib>1)then
         if(kfrsou(ib-1)>0)then
            !copy the species value from the previous beam "ib-1"
            !to the present:
            kfrsou(ib)=kfrsou(ib-1) ![2025-12-16]
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*) 'frinitz/WARNING: for beam ib=',ib,
     &        ' resetting kfrsou(ib)=',kfrsou(ib)
CMPIINSERT_ENDIF_RANK
         endif
         endif
       endif !kfrsou(ib).eq.0
      enddo !ib ![2025-12-16]

      do ib=1,kb
         ibion(ib)=0 !initialize; find in the scan below
      enddo
      
      do ib=1,nbeams  !BH171014: Enabling beams with different species.
      kk=0
      do 500 k=1,ntotal
        ! exclude e for the stopping cross-section:
        if (k.eq.kelecg .or. k.eq.kelecm) go to 500 
        kk=kk+1   !Counts non-electron, i.e.,ion species in cql3d list
        if (kk.gt.nprim) go to 500
!        if (k.eq.kfrsou) ibion=kk  Modifying for more than one NB species
        if (k.eq.kfrsou(ib)) ibion(ib)=kk
        namep(kk)=kspeci(1,k)
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'frinitz: k,namep(primary_ion), ib,ibion(ib)=',
     &   k,trim(namep(kk)), ib,ibion(ib)
CMPIINSERT_ENDIF_RANK
 500  continue
      if(ibion(ib).le.0)then
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'frinitz:ibion()=0, prob kfrsou()=0; Set to proper k'
CMPIINSERT_ENDIF_RANK
        STOP
      endif
      enddo  !On ib

cBH081022      kk=0
cBH081022      do 510 k=ntotal+1-nimp,ntotal
cBH081022        kk=kk+1
cBH081022        namei(kk)=kspeci(1,k)
cBH081022 510  continue

cBH081022:  Find impurity species name (not previously working)
cBH081022:  Presently set up for only one species. (Else, more coding).
cBH081022:  Take impurities to be distinct species with atw.gt.4.

      if (nimp.gt.1) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'STOP:  NBI (Freya setup) only for nimp.le.1.'
CMPIINSERT_ENDIF_RANK
         STOP 'frinitz: nimp>1'
      endif
      do kk=1,nimp
         do k=1,ntotal
            if (fmass(k)/proton .gt. 4.1) then
c BH081022:  Add code here, if need nimp.gt.1.
               namei(kk)=kspeci(1,k)
               ksave=k  !for use below, for atw of impurity
            endif
         enddo
      enddo
               

c----------------------------------------------------------------
c     determine atomic weights of primary ions
c----------------------------------------------------------------
      do 3410 i=1,nprim
        atw(i) = 0.
        if(trim(namep(i)).eq.'h'  .or. 
     +     trim(namep(i)).eq.'H') atw(i)=1.
     
        if(trim(namep(i)).eq.'d'  .or. 
     +     trim(namep(i)).eq.'D') atw(i)=2.
     
        if(trim(namep(i)).eq.'t'  .or. 
     +     trim(namep(i)).eq.'T') atw(i)=3.
     
        if(trim(namep(i)).eq.'dt' .or. 
     +     trim(namep(i)).eq.'DT' .or. 
     +     trim(namep(i)).eq.'Dt' .or. 
     +     trim(namep(i)).eq.'dT') atw(i) = fd*2. + (1.-fd)*3.
     
        if(trim(namep(i)).eq.'he' .or. 
     +     trim(namep(i)).eq.'HE' .or. 
     +     trim(namep(i)).eq.'He') atw(i)=4.
     
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'frinitz/primary_ion list: i, trim(namep(i)), atw(i)',
     +                       i, trim(namep(i)), atw(i)
CMPIINSERT_ENDIF_RANK
        if(atw(i).eq.zero) call frwrong(1)
 3410 continue !i=1,nprim


c----------------------------------------------------------------
c     determine atomic weights of impurity ions
c----------------------------------------------------------------
      if(nimp.eq.0) go to 3430
      do 3420 i=1,nimp
        k = nprim + i
        atw(k) = 0.
        if(trim(namei(i)).eq.'he' .or. 
     +     trim(namei(i)).eq.'HE' .or. 
     +     trim(namei(i)).eq.'He') atw(k)= 4.
     
        if(trim(namei(i)).eq.'b' .or. 
     +     trim(namei(i)).eq.'B' ) atw(k)= 11.  ! YuP added [2015]

        if(trim(namei(i)).eq.'c' .or. 
     +     trim(namei(i)).eq.'C' ) atw(k)= 12.
     
        if(trim(namei(i)).eq.'o' .or. 
     +     trim(namei(i)).eq.'O' ) atw(k)= 16.
        
        if(trim(namei(i)).eq.'si' .or. 
     +     trim(namei(i)).eq.'SI' .or. 
     +     trim(namei(i)).eq.'Si') atw(k)= 28.
     
        if(trim(namei(i)).eq.'ar' .or. 
     +     trim(namei(i)).eq.'AR' .or. 
     +     trim(namei(i)).eq.'Ar') atw(k)= 40.
     
        if(trim(namei(i)).eq.'cr' .or. 
     +     trim(namei(i)).eq.'CR' .or. 
     +     trim(namei(i)).eq.'Cr') atw(k)= 52.
     
        if(trim(namei(i)).eq.'fe' .or. 
     +     trim(namei(i)).eq.'FE' .or. 
     +     trim(namei(i)).eq.'Fe') atw(k)= 56.
        
        if(trim(namei(i)).eq.'ni' .or. 
     +     trim(namei(i)).eq.'NI' .or. 
     +     trim(namei(i)).eq.'Ni') atw(k)= 59.
        
        if(trim(namei(i)).eq.'kr' .or. 
     +     trim(namei(i)).eq.'KR' .or. 
     +     trim(namei(i)).eq.'Kr') atw(k)= 84.
        
        if(trim(namei(i)).eq.'mo' .or. 
     +     trim(namei(i)).eq.'MO' .or. 
     +     trim(namei(i)).eq.'Mo') atw(k)= 96.
        
        if(trim(namei(i)).eq.'w' .or. 
     +     trim(namei(i)).eq.'W' ) atw(k)= 184.
        
        if(trim(namei(i)).eq.'mixt') atw(k)= 20.
        
        if(trim(namei(i)).eq.'a' .or. 
     +     trim(namei(i)).eq.'A' ) atw(k)=int(fmass(ksave)/proton +0.1)
                                            !Compatible with cql3d added
                                            !impurity, for given zeff.
        if(atw(k).eq.zero) call frwrong(2)
 3420 continue
 3430 continue

      if(nimp.gt.0) then
CMPIINSERT_IF_RANK_EQ_0
         do i=1,nimp
         WRITE(*,*)'frinitz/impurity list: trim(namei(i)),atw(nprim+i)',
     +                          trim(namei(i)),atw(nprim+i)
         enddo
CMPIINSERT_ENDIF_RANK
      endif

      return
      end
