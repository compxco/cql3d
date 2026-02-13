c
c
      subroutine frnfreya(frmod_,fr_gyro_,beamplse_,beampon_,beampoff_,
     1     nbeams_,hibrz_,mfm1_,noplots,kfrsou,src_nbi_e_)
      implicit integer (i-n), real*8 (a-h,o-z)

cBH171014: kfrsou added.  In comm.h it is an array indicating the
cBH171014: species for each beamline.

c.................................................................
      include 'param.h'
      include 'frcomm.h'
CMPIINSERT_INCLUDE

c     ONETWO DIVERGENCE: SEE COMMENTS AT BEGINNING OF FREYA
      character*8 frmod_,fr_gyro_,beamplse_,noplots,codeid,src_nbi_e_
      real*8, intent(out) :: hibrz_(kz,ke,kb)
      integer, intent(out) :: mfm1_,nbeams_
      integer, intent(in) :: kfrsou(*)

      character*8 ifirst
      save ifirst
      data ifirst/"first"/

c     To pass these freya namelist in frcomm.h to comm.h
      frmod_=frmod
      fr_gyro_=fr_gyro
      beamplse_=beamplse
      beampon_=beampon
      beampoff_=beampoff
      nbeams_=nbeams
      src_nbi_e_=src_nbi_e !->src_nbi_ep  [2022-06-30] Added variable [not ready]
      

c..................................................................
c     This routine controls all of the NFREYA routines.
c     Called from subroutine tdinitl and tdchief
c..................................................................

c..................................................................
c     Return if input variable frmod is "disabled"
c..................................................................

      if (frmod.ne."enabled") return
      elong=0
      
      !YuP[2022-11-28]Added resetting (similar to frset):
      call bcast(rpts,zero,npart)
      call bcast(xpts,zero,npart)
      call bcast(ypts,zero,npart)
      call bcast(zpts,zero,npart)
      call bcast(vx,zero,npart)
      call bcast(vy,zero,npart)
      call bcast(vz,zero,npart)

c..................................................................
c     First initialize some (iteration or time dependent) data.
c     [Possible dependence is on equilibrium, average temp, 
c      density, etc.   Check in frstup.]
c     Independent of beam number of beam species (kfrsou()).
c..................................................................

      call frstup(mf,mfm1,mi,mj,nion,potsid,codeid,rin,rmax,
     1  zax,zmin,zmax,zni,zne,zte,zti,zzi,p,psivol,xxx,yyy,
     1  nprim,nimp,zeffctv,zshift)

c.......................................................................
c     zshift if frstup above is passed back from comm.h data to 
c     adjust beam pivot height according to any shift in the equilibrim.
c.......................................................................

      if (ifirst.eq."first") then
         do ib=1,nbeams
            zpivot(ib)=zpivot(ib)-zshift
         enddo
         ifirst="notfirst"
      endif

c.......................................................................
c     Up until now (171014), the neutral beam module has allowed for
c     been restricted multiple neutral beams, but all beam lines must
c     be of the same plasma species.  Thus, for ngen.ge.2 cases, e.g.,
c     D and T general species, only one species can be injected.  
c     This restriction is removed, with the introduction of dimensioned
c     kfrsou(ib=1:nbeams).  nbeams remains equal to the total number of
c     beamlines, but each kfrsou(ib) can point to a different general
c     species.   For convenience, beamlines involving the same species
c     are assumed grouped together in kfrsou().
c     Determine nbeams1(ibs=1:number_of_separate_species). Subroutine 
c     freya will be called for each separate species, with 
c     nbeams1(ibs) beam lines.
c.......................................................................

      do kk=1,kb
         nbeams1(kk)=0
      enddo
      kbs=1   !Number of different beamline species
      nbeams1(1)=1 !Number of beamlines for each species
      kblspec(1)=kfrsou(1) !spec index for each nbeams1 beamline set
      
      do ib=2,kb
         if (kfrsou(ib).eq.kfrsou(ib-1)) then
            nbeams1(kbs)=nbeams1(kbs)+1
         elseif(kfrsou(ib).ne.0) then
            kbs=kbs+1           !increment number of beamline species
            kblspec(kbs)=kfrsou(ib)
            nbeams1(kbs)=nbeams1(kbs)+1
         else                   !kfrsou(ib).eq.0
            go to 1
         endif
      enddo  !On ib
 1    continue

CMPIINSERT_IF_RANK_EQ_0      
      WRITE(*,*)'frnfreya: kbs=',kbs
CMPIINSERT_ENDIF_RANK

c..................................................................
c     BH171014:
c     Putting loop around NFREYA call, to apply it to possibly more
c     than one beam line set, i.e., ngen.ge.2 plasma general species.
c..................................................................

      ibstart=1  !Starting beamline for each species

      do kbsp=1,kbs   !Down to line 184

      if (kbsp.ge.2) ibstart=ibstart+nbeams1(kbsp-1)
      
      k=kfrsou(ibstart) !YuP[2018-01-24] added species identification,
                       ! for plots
c..................................................................
c     Call NFREYA (cray32 from ONETWO)
c..................................................................

CMPIINSERT_IF_RANK_EQ_0      
      WRITE(*,*)'frnfreya:mi,mj,codeid', mi,mj,codeid
      WRITE(*,*)'frnfreya:rin,rmax', rin,rmax
      WRITE(*,*)'frnfreya:zax,zmin,zmax',zax,zmin,zmax
      WRITE(*,*)'frnfreya/bef.freya: kbsp,ibstart=',kbsp,ibstart
CMPIINSERT_ENDIF_RANK

      call freya(ipts,mi,mj,codeid,rin,rmax,zax,zmin,zmax,kbsp,ibstart)

CMPIINSERT_IF_RANK_EQ_0      
      WRITE(*,*) 'frnfreya: ------> Done calling freya. k=',k
      if (ipts.eq.0) 
     1       WRITE(*,*)'frnfreya: WARNING, ipts=0, NB missed plasma?'
CMPIINSERT_ENDIF_RANK

c..................................................................
c     Compute the total number of particles/sec actually deposited in
c     the plasma. This requires subtracting off from the neutral current
c     the fraction that are lost at the aperture and the fraction
c     lost at the walls. ORBIT EFFECTS ARE NOT CURRENTLY CONSIDERED.
c     [2014-5: cql3d-fow now accounts for gc orbits and gyro-radius 
c      offsets.]
c     Note that in the event that Freya input variable bptor is
c     used, we are specifying the total power injected into the tokamak
c     disregarding any losses at the apertures. The expression for
c     the deposited current below takes this into account. While fap may
c     not be zero, in routine FREYA the total number of neutralized
c     particles has been normalized up to account for losses at
c     the apertures. In the event that bptor is not used, bneut is
c     not renormalized, and the expression below makes sense as it
c     stands.
c..................................................................

      curdep=0.d0 !YuP[2022-12]added d0
cBH171014      do 10 ib=1,nbeams
      do 10 ib=ibstart,(ibstart-1)+nbeams1(kbsp)
      do 20 ie=1,3
        curdep=curdep+bneut(ie,ib)*(one-fap(ie,ib))*(one-fwall(ie,ib))
 20   continue
 10   continue

c..................................................................
c     Now determine the source arrays used in CQL3D
c..................................................................

      call freyasou(xpts,ypts,zpts,rpts,vx,vy,vz,ipts,curdep,
     1  bmsprd,multiply,multiplyn,ibstart)
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*) 'frnfreya: Done freyasou. k=',k
CMPIINSERT_ENDIF_RANK


c..................................................................
c     Pass additional data to cql3d through subroutine arguments
c..................................................................
      mfm1_=mfm1
      do ib=1,kb
      do ie=1,ke
      do i=1,kz !YuP: first dimension - inner-most loop
         hibrz_(i,ie,ib)=hibrz(i,ie,ib)
      enddo
      enddo
      enddo

c..................................................................
c     Plot the FREYA birth points.
c..................................................................
        call frplteq(xpts,ypts,zpts,rpts,vx,vy,vz,ipts,curdep,
     1    nfrplt,frplt,k)
        !YuP[2018-01-24] added species identification k

      enddo  !On kbsp

      return
      end
