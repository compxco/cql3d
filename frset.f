c
c
      subroutine frset(lrz,noplots,nmlstout,ngen,kfrsou)
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'frcomm.h'
      include 'frname.h'
CMPIINSERT_INCLUDE

      character*1024 t_
      character*8 noplots,nmlstout
      REAL*4 :: R40=0.,R45=5.

      REAL*4 RILIN
      
      integer kfrsou(kb)

c..................................................................
c     frset initializes non-namelist variables.
cBH081020c     Also, resets namelist variable nprim=1, if not =1.
cBH170807:  nprim can be greater than 1, for beams with
cBH170807:  different species.
c..................................................................



c..................................................................
c     Jump out if lrz=1
c..................................................................

      if (lrz.eq.1) then
         ranseed=dble(7**7) !YuP[2021-01-10] Added dble
         read(2,frsetup)
         if (nmlstout.eq."enabled") write(6,frsetup)
         dummy=RANDOM_my(ranseed)
         goto 300 !-> allocate vars. and return
      endif

      ncorin=75
      do 1710 j=1,kprim
        znipm(j)=0.0d0
 1710 atwpm(j)=0.0d0
      do 1720 j=1,kimp
        iz(j)=0
        atwim(j)=0.0d0
 1720 zniim(j)=0.0d0

      ncont=30

c..................................................................
c     Set defaults for beam smoothing (none)...
c..................................................................

      smooth=.05

c..................................................................


c..................................................................
c     Read in data for NFREYA
c..................................................................

      read(2,frsetup)
      if (nmlstout.eq."enabled") write(6,frsetup)

      if (frmod.eq."disabled") return


cBH081020      if (nprim.ne.1) then
cBH081020         nprim=1
cBH081020         write(*,210)
cBH081020      endif
cBH081020 210  format('WARNING: Resetting nprim=1. Check FREYA if want nprim>1')

cBH171014  Check nprim=ngen.  ALSO, should have nimp.ne.0 if impurities
cBH171014  (non gen species ions contributing to charage balance)
cBH171014  are in the plasma.  ADD LATER.
!YuP[2023] Actually, it is not always nprim=ngen.
!Example:   { D_gen(k=1),       T_gen(k=2),       e_gen(k=3);  some Maxw species}
!Set kfrsou={ kfrsou(beam#1)=1, kfrsou(beam#2)=2, kfrsou(3:kb)=0 }
!In the above, nbeams=2 (D+T beams), (nbeams must be set in cqlinput).
!The scan below will give nprim=2 ("Two primary ionic species"),
!even when nprim is set to 1 in cqlinput (by mistake).

      if (nprim.gt.ngen) then !YuP[2023-01-09] added this basic check
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'frset: Check nprim in cqlinput. Should not exceed ngen'
CMPIINSERT_ENDIF_RANK
      STOP 'nprim>ngen  Check cqlinput'
      endif

!YuP[2023]      if (nprim.ne.ngen) then
!         write(*,*)
!         write(*,*) 'Resetting nprim=ngen.  Should correct cqlinput.'
!         write(*,*)
         !YuP[2022-06-30] Added a scan over gen.species, to determine nprim
         do k=1,ngen
         do ib=1,nbeams !nbeams is from cqlinput
           if(kfrsou(ib).eq.k)then
              nprim=max(nprim,k)
           endif
         enddo
         enddo
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'frset: Number of primary ionic species nprim=',nprim
CMPIINSERT_ENDIF_RANK
!YuP[2023]      endif

c     Check nbeams not too large.
      if (nbeams .gt. kb) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*)'Namelist nbeams.gt.kb:  increase parameter kb'
CMPIINSERT_ENDIF_RANK
         stop 'frset: increase kb in param.h'
      endif

      if (multiply.ne."disabled".and.multiplyn.eq.0) then
         WRITE(*,214)
 214     format(//,'Must set multiplyn, if multiply.ne.disabled',//)
         stop 'frset: Must set multiplyn'
      endif
      
      if((ne_tk.ne.0) .and. (ksge.le.2))then !YuP[2023-01-24] added check/stop
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)
        WRITE(*,*)
     &  'frset: For ne_tk>0 (tor.rotation) increase ksge to 20 or more'
CMPIINSERT_ENDIF_RANK
        stop 'frset: ksge is too low'
      endif
      
c
c     ION PARAMETERS
c--------------------------------------------------------------------
c     nprim       Number of primary ion species
c     nimp        Number of impurity ion species
c     namep(i)    Name of ith primary ion species
c     'h' , protons
c     'd' , deuterons
c     't' , tritons
c     'dt', mixture of d and t
c     'he', thermal alphas
c     namei(i)    Name of ith impurity ion species
c     'he', helium
c     'c' , carbon
c     'o' , oxygen
c     'si', silicon
c     'ar', argon
c     'ti', titanium
c     'cr', chromium
c     'fe', iron
c     'ni', nickel
c     'kr', krypton
c     'mo', molybdenum
c     'w' , tungsten
c     fd          Number fraction of deuterons in d-t mixture
c--------------------------------------------------------------------
      fd=.5d0   !No longer namelist input

c..................................................................
c     Call subroutine linked to CQL3D common blocks for more initializat
c..................................................................

      call frinitz(nprim,nimp,nion,ibion,namep,namei,atw,fd,smooth,
     1             nbeams)


c..................................................................
c     Initiate random number generator for NFREYA
c..................................................................
c     See comments in freyasou.f/subroutine frnnoa.
c990131      call ranset(ranseed)

cBH131015: ranseed is initialized in frinitl to 7**7
cBH131015: This call to RANDOM_my initialized a RN sequence.      
      dummy=RANDOM_my(ranseed)

c     Reset npart, for nubeam list case
      if (read_birth_pts.eq."enabled") npart=nbirth_pts


CMPIINSERT_IF_RANK_EQ_0
      ! make plots on mpirank.eq.0 only
      if (noplots .ne. "enabled1") then
      !YuP[2021-01-10] No need to call PGPAGE. 
      !Continue on same page as "PARAMETER VALUES"
      write(t_,1000)
 1000 format("FR (freya beam deposition) model parameters:")
      RILIN=11.
      CALL PGMTXT('T',-RILIN,R40,R40,t_)

      write(t_,1001) 
 1001 format("npart is the number of ions launched")
      RILIN=12.
      CALL PGMTXT('T',-RILIN,R40,R40,t_)

      write(t_,1002) npart
 1002 format("====>npart = ",i7)
      RILIN=13.
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      endif
CMPIINSERT_ENDIF_RANK


c..................................................................
c     Set constants (for real*8)
c..................................................................
 300  continue

      zero=0.d0
      one=1.d0
      two=2.d0
      half=0.5d0
c      pi=4.d0*atan2(one,one)
      pi=atan2(zero,-one)

      allocate(rpts(npart),STAT=istat)
      write(*,*)'frset  rpts: istat=',istat
      if(istat.eq.0) call bcast(rpts,zero,npart)
      
      allocate(xpts(npart),STAT=istat)
      write(*,*)'frset  xpts: istat=',istat
      if(istat.eq.0) call bcast(xpts,zero,npart)
      
      allocate(ypts(npart),STAT=istat)
      write(*,*)'frset  ypts: istat=',istat
      if(istat.eq.0) call bcast(ypts,zero,npart)
      
      allocate(zpts(npart),STAT=istat)
      write(*,*)'frset  zpts: istat=',istat
      if(istat.eq.0) call bcast(zpts,zero,npart)
      
      allocate(vx(npart),STAT=istat)
      write(*,*)'frset  vx: istat=',istat
      if(istat.eq.0) call bcast(vx,zero,npart)
      
      allocate(vy(npart),STAT=istat)
      write(*,*)'frset  vy: istat=',istat
      if(istat.eq.0) call bcast(vy,zero,npart)

      allocate(vz(npart),STAT=istat)
      write(*,*)'frset  vz: istat=',istat
      if(istat.eq.0) call bcast(vz,zero,npart)

      return
      end

