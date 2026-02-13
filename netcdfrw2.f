
c
c
      subroutine netcdfrw2(kopt)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

c     This subroutine is only called from MPI rank=0.      

c     This subroutine uses netCDF-3 subroutines.

c     This subroutine creates/writes a netCDF file
c       of standard data from cql3d.  (Additional data is
c       written for RF into a seperate file, if urfmod.ne."disabled").
c     netCDF file ==> "mnemonic".nc
c     netCDF file ID ==> ncid
c     time-step counter ==> numrec1
c
c     While moments of distn f and the electric field are saved after
c     each time-step, the distribution function itself is saved 
c     only after the last time-step [netcdfshort='disabled'],
c     for n.eq.nsave() [if netcdfshort='lngshrtf'], or
c     after every time step [netcdfshort='longer_f']
c
c     Action of the subroutine is controlled by kopt and n:
c       kopt=0, n=0, initialize "mnemonic".nc and write data
c       kopt=1, n.gt.0, write data
c

c --- include file for netCDF declarations 
c --- (obtained from NetCDF distribution)
      include 'netcdf.inc'

c     Will crop distribution functions to used size,
c     that is, take off the extra row and column of
c     points associated with f(0,0,k,l).

c --- some stuff for netCDF file ---
      integer ncid,vid,istatus
      integer xdim,ydim,rdim,r0dim,r00dim,kdim,zdim,tdim,gdim !-YuP: gdim added
      integer xdimf,ydimf ! for 'f', corresp. to (1:iy),(1:jx)
      integer tsavedim
      integer twodim,fourdim,fivedim,thirteendim,chardim,char64dim
      integer nendim,nvdim,nmodsdim,ntavgdim,mrfnp1_dim,mrfnp3_dim
      integer nt_deltadim,nr_deltadim,nz_deltadim
      integer start2(2), start3(3), start4(4)
      
      !!!!For fusion neutrons diagnostics:!!!
      integer nv_fusdim, nrho_fusdim !!!
      integer start_fusn(3), count_fusn(3), fusn_dims(3) !!!
      ! other fusion diagn: for sigftt and other diagn.  !!!
      integer start_fus(3),  count_fus(3),  fus_dims(3)  !!!
      !------
      
      integer nen_npadim,nv_npadim,npaproc_dim
      integer nena_dim, nva_dim, nrho_npadim
      integer npaproc_dims(2)
      integer count_npaproc(2),start_npaproc(2)
      integer npaenn_dims(2)
      integer count_npaenn(2),start_npaenn(2)
c
      integer count_hibr(3),start_hibr(3)
      integer count_sorpw(2), start_sorpw(2)
c
      integer dims(4),count(4),start(4)
      integer dimsg(5),countg(5),startg(5)  !To accnt for ngen.gt.1 cases
      integer dimssave(4),countsave(4),startsave(4)
      integer dimsgsave(5),countgsave(5),startgsave(5)  !For ngen.gt.1 cases
      integer dimsf(4),countf(4),startf(4) ! YuP[2018-09-28]
      integer start1(4)
      integer count_xr(3),start_xr(3)
      integer count_npa(3),start_npa(3)
      integer count_powurf(2),start_powurf(2)
      integer count_rfpwr(3),start_rfpwr(3)
      integer count_powrf(3),start_powrf(3)
      integer count_powrft(2),start_powrft(2)
      integer count_powers(4),start_powers(4)
      integer r_dims(2),r_count(2)
      integer rk_dims(3),start_rk(3),count_rk(3)
      integer r0k_dims(3),start_r0k(3),count_r0k(3)
      integer r0_dims(2),r0_count(2)
      integer r00_dims(2),r00_count(2)
      integer y_dims(2),y_count(2)
      integer species_dims(3),species_count(3)
      integer kspeci_dims(3),kspeci_count(3)
      integer tau_dims(2),tau_count(2)
      integer z_dims(3),z_count(3),z_count1(3)
      integer delta_dims(3),delta_count(3),delta_start(3)
Cdeltarhop      Remove the Cdeltarhop to include deltarhop in .nc file
Cdeltarhop      integer deltap_dims(3),deltap_count(3),deltap_start(3)
      integer xray_dims(3)
      integer npa_dims(3)
      integer rfpwr_dims(3)
      integer dims_powurf(2)
      integer dims_powrft(2),dims_powrf(3)
      integer powers_dims(4)
      integer currv_dims(3)
      integer currv_dimsg(4)
c
      integer hibr_dims(3)  !  JK
      integer sorpw_dims(2)
c
      integer xlm_dim     ![2022-03-19] for saving xlm() and fl()
      integer dims_fl(3)  ![2022-03-19] for saving xlm and fl
      integer count_fl(3) ![2022-03-19] for saving xlm and fl

      integer start_elecfldn(3),count_elecfldn(3),elecfldn_dims(3)
      
      !YuP[2019-09] Added, to save bnumb_imp(), dens_imp(), etc.
      integer impur_nstates_dim, impur_dims(3), impur_count(3)

      character*100 ltitle

      character*8 ndelta_op
           
      real*8, allocatable :: wkpack(:) ! local working array for pack21
      integer, allocatable :: iwkpack(:) ! local working array for ipack21
      
c      character(len=8), dimension(npaproc) :: npa_proc  !automatic var

      data start/1,1,1,1/, start1/1,1,1,1/
      data startsave/1,1,1,1/
      data start_rk/1,1,1/, start_r0k/1,1,1/
      data startgsave/1,1,1,1,1/
      data start_xr/1,1,1/
      data start_npa/1,1,1/
      data start_npaproc/1,1/
      data start_npaenn/1,1/
      !  For fusion neutrons and other diagnostics:
      data start_fusn/1,1,1/, start_fus/1,1,1/
      !--
      data start_powurf/1,1/
      data start_rfpwr/1,1,1/,start_powers/1,1,1,1/
      data start_powrf/1,1,1/
      data start_powrft/1,1/
      data delta_start/1,1,1/
      data start_hibr/1,1,1/
      data start_sorpw/1,1/
      data start2/1,1/
      data start3/1,1,1/
      data start4/1,1,1,1/
   
Cdeltarhop      data deltap_start/1,1,1/
c
CMPIINSERT_IF_RANK_NE_0_RETURN
c This subroutine is only called from MPI rank=0.

       !WRITE(*,*) 'inside netcdfrw2.f...'
c       WRITE(*,*) 'frmodp=',frmodp
c       WRITE(*,*) 'hibrzp(i,1,1),hibrzp(i,2,1),hibrzp(i,3,1)'
c       do i=1,22
c       do i=1,nconteqn
c         WRITE(*,'(i4,2x,0p9f9.4)') i, hibrzp(i,1,1),
c     >        hibrzp(i,2,1),hibrzp(i,3,1),
c     >        hibrzp(i,1,2),hibrzp(i,2,2),hibrzp(i,3,2),
c     >        hibrzp(i,1,3),hibrzp(i,2,3),hibrzp(i,3,3)
c       enddo
c

c     Maximum of iy_() as function of radius:
      iymx=0  !-YuP-101215: Don't use iy=; it's in common /params/
             ! Don't let overwrite the cqlinput value!
      do l=1,lrors !YuP[2021-03-11] was lrz  (in CQLP runs: 1:ls, CQL3D 1:lrz)
         iymx=max(iymx,iy_(l))
      enddo
      if(iymx.gt.iymax) stop 'netcdfrw2: iy_(l) should not exceed iy' 
      !Now All arrays are saved over iymax, not iy_(ll)

      if (.NOT.ALLOCATED(wkpack)) then ! allocate working array for pack21
         nwkpack=max(iyjx2, iymax*lrors, ntotala*(lrza+1), 
     +               lz*lrzmax, iymax*lz, nena*nva, 
     +               iymax*lrzmax, lrza*npaproca) +10 ! +10 just in case
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)
         allocate(wkpack(nwkpack),STAT=istat)
         call bcast(wkpack,zero,SIZE(wkpack))
      endif

      if (.NOT.ALLOCATED(iwkpack)) then ! allocate working array for ipack21
         nwkpack=max(iyjx2, iymax*lrors, ntotala*(lrza+1), 
     &               lz*lrzmax, iymax*lz, nena*nva, 
     &               iymax*lrzmax, lrza*npaproca) +10 ! +10 just in case
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)
         allocate(iwkpack(nwkpack),STAT=istat)
         call ibcast(iwkpack,0,SIZE(iwkpack))
      endif
c     Ensure that tem1-2 has sufficient dim for (pack21) equivalences:
c      if (iy*lrors.gt.iyjx2) 
c     +    stop '1: in netcdfrw2: Need to set jx>lrza'
c      if (ntotala*(lrza+1).gt.iyjx2) 
c     +    stop '2: in netcdfrw2: Need ntotala*(lrza+1)<(iy+2)*(jx+2)'
c      if (lz*lrzmax.gt.iyjx2) 
c     +    stop '3: in netcdfrw2: Need lz*lrza<(iy+2)*(jx+2)'
c      if (iy*lz.gt.iyjx2)
c     +    stop '4: in netcdfrw2: Need iy*lz<(iy+2)*(jx+2)'
c      if (nena*nva.gt.iyjx2)
c     +    stop '5: in netcdfrw2: Need nena*nva<(iy+2)*(jx+2)'
c
c      if (iy*lrzmax.gt.iyjx2) 
c     +    stop '6: in netcdfrw2: Need iy*lrza<(iy+2)*(jx+2)'
c      if (lrza*npaproca.gt.iyjx2) 
c     +    stop '7: in netcdfrw2: Need lrza*npaproca<(iy+2)*(jx+2)'


cBH110320C-----------------------------------------------------------------------
cBH110320C     Only set up for cqlpmod.ne."enabled",ngen=1, for the time being.
cBH110320C-----------------------------------------------------------------------

cBH110320      if (cqlpmod.eq."enabled" .or. ngen.ne.1) then
cBH110320         WRITE(*,*) 'WARNING: netcdfrw2 subroutine not implemented'
cBH110320         return
cBH110320      endif

cdir$ master

      count(1)=iymax
      count(2)=jx
      count(3)=lrz
      count(4)=1
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)

      
      !For recording of 'f', when ngen=1   
      startf(1)=1 
      startf(2)=1 
      startf(3)=1 ! will be changed
      startf(4)=1 ! will be changed
      countf(1)=iymax
      countf(2)=jx 
      countf(3)=1  !radial index  ! will be changed
      countf(4)=1  !record number ! will be changed

      !For recording of 'f', when ngen>1   
      startg(1)=1 
      startg(2)=1 
      startg(3)=1 ! will be changed
      startg(4)=1 ! will be changed
      startg(5)=1 ! will be changed
      countg(1)=iymax
      countg(2)=jx 
      countg(3)=1  !radial index  ! will be changed
      countg(4)=1  !species index ! will be changed
      countg(5)=1  !record number ! will be changed

      count_xr(1)=nen
      count_xr(2)=nv
      count_xr(3)=1

      count_npa(1)=nen_npa
      count_npa(2)=nv_npa
      count_npa(3)=1

      count_npaenn(1)=lrzmax
      count_npaenn(2)=npaproc

      count_npaproc(1)=8
      count_npaproc(2)=npaproc

      count_fus(1)=lrzmax
      count_fus(2)=4
      count_fus(3)=1

      count_powurf(1)=mrfn+1
      count_powurf(2)=1

      count_rfpwr(1)=lrz
      count_rfpwr(2)=mrfn+3
      count_rfpwr(3)=1

      count_powrf(1)=lrz
      count_powrf(2)=nmodsa
      count_powrf(3)=1

      count_powrft(1)=lrz
      count_powrft(2)=1

      count_powers(1)=lrz
      count_powers(2)=13
      count_powers(3)=ngen
      count_powers(4)=1

      r_count(1)=lrz
      r_count(2)=1

      count_rk(1)=lrz
      count_rk(2)=ngen
      count_rk(3)=1

      count_r0k(1)=lrzmax
      count_r0k(2)=ngen
      count_r0k(3)=1

      r0_count(1)=lrzmax
      r0_count(2)=1

      r00_count(1)=lrzmax+1
      r00_count(2)=1

      if(cqlpmod.ne."enabled")then ! CQL3D
      y_count(1)=iymax 
      y_count(2)=lrz ! for CQL3D
      else ! CQLP  !YuP[2022-02-09] added
      y_count(1)=iymax 
      y_count(2)=lz ! for CQLP
      endif

      species_count(1)=ntotal
      species_count(2)=lrzmax
      species_count(3)=1

      tau_count(1)=iymax
      tau_count(2)=lrzmax

      z_count(1)=iymax
      z_count(2)=lz
      z_count(3)=lrzmax

      z_count1(1)=iymax
      z_count1(2)=lz
      z_count1(3)=1
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)

      delta_count(1)=nr_delta
      delta_count(2)=nz_delta
      delta_count(3)=nt_delta

Cdeltarhop      deltap_count(1)=nt_delta
Cdeltarhop      deltap_count(2)=lz
Cdeltarhop      deltap_count(3)=lrzmax

      kspeci_count(1)=8
      kspeci_count(2)=2
      kspeci_count(3)=ntotal
c
      count_hibr(1)=kz
      count_hibr(2)=ke
      count_hibr(3)=kb
      count_sorpw(1)=ngena
      count_sorpw(2)=lrza

      !YuP[2019-09] Added, to save bnumb_imp(), dens_imp(), etc.
      impur_count(1)=nstates+1 ! +1 because of kstate=0
      impur_count(2)=lrz
      impur_count(3)=1  ! for each time step (recording data)
      !Note: If no impurities are present, nstates is 0

c
c.......................................................................
cl    1. Initialize part, creating new netcdf file
c

c --- begin if ---
      WRITE(*,*)'netcdfrw2: kopt=',kopt,'  n=',n
      if ((kopt.eq.0) .and. (n.eq.0)) then !endif at l 3282

C-----------------------------------------------------------------------
c
cl     1.1 create netCDF file and define dimensions,variables 
c          and attributes
c

c.......................................................................
cl    1.1.1 create netCDF filename
c     CLOBber old file, if it exists.
c     istatus is 0, if no errors.
      write(t_,1000) mnemonic(1:length_char(mnemonic))
 1000 format(a,".nc")
c-YuP:      ncid=nccre(t_,NCCLOB,istatus)
      istatus = NF_CREATE(t_, NF_CLOBBER, ncid) !-YuP: NetCDF-f77
      call check_err(istatus)

c.......................................................................
cl    1.1.2 define dimensions
c     p. 43 of netcdf-3 manual
      
      istatus= NF_DEF_DIM(ncid, 'xdim',     jx,       xdim)
      istatus= NF_DEF_DIM(ncid, 'ydim',     iymax,    ydim)
      
      ! for 'f', corresp. to (1:iy),(1:jx):
      istatus= NF_DEF_DIM(ncid, 'xdimf',    jx,     xdimf)
      istatus= NF_DEF_DIM(ncid, 'ydimf',    iymax,  ydimf)

      istatus= NF_DEF_DIM(ncid, 'rdim',     lrz,      rdim)
      istatus= NF_DEF_DIM(ncid, 'r0dim',    lrzmax,   r0dim)
      istatus= NF_DEF_DIM(ncid, 'kzdim',     kz,      kzdim) ! JK: freya
      istatus= NF_DEF_DIM(ncid, 'kedim',     ke,      kedim)
      istatus= NF_DEF_DIM(ncid, 'kbdim',     kb,      kbdim)
      istatus= NF_DEF_DIM(ncid, 'ngenadim',  ngena,   ngenadim)
      istatus= NF_DEF_DIM(ncid, 'lrzadim',   lrza,    lrzadim)
c      WRITE(*,*) 'kzdim,kedim,kbdim = ',kzdim,kedim,kbdim
c      stop
      istatus= NF_DEF_DIM(ncid, 'twodim',       2,    twodim)
      istatus= NF_DEF_DIM(ncid, 'fourdim',      4,    fourdim)
      istatus= NF_DEF_DIM(ncid, 'fivedim',      5,    fivedim)
      istatus= NF_DEF_DIM(ncid, 'thirteendim', 13,    thirteendim)
      istatus= NF_DEF_DIM(ncid, 'r00dim',   lrzmax+1, r00dim)
      istatus= NF_DEF_DIM(ncid, 'zdim',     lz,       zdim)
      istatus= NF_DEF_DIM(ncid, 'nendim',   nen,      nendim)
      istatus= NF_DEF_DIM(ncid, 'nvdim',    nv,       nvdim)      
      istatus= NF_DEF_DIM(ncid, 'nen_npadim',   nen_npa,  nen_npadim)
      istatus= NF_DEF_DIM(ncid, 'nv_npadim',    nv_npa,    nv_npadim)
      istatus= NF_DEF_DIM(ncid, 'npaproc_dim',  npaproc, npaproc_dim)
      istatus= NF_DEF_DIM(ncid, 'mrfnp1_dim', mrfn+1,   mrfnp1_dim)
      istatus= NF_DEF_DIM(ncid, 'mrfnp3_dim', mrfn+3,   mrfnp3_dim)
      istatus= NF_DEF_DIM(ncid, 'nmodsdim', nmodsa,    nmodsdim)
      istatus= NF_DEF_DIM(ncid, 'ntavgdim', ntavga,    ntavgdim)
      
      istatus= NF_DEF_DIM(ncid, 'gen_species_dim', ngen,   gdim)
      istatus= NF_DEF_DIM(ncid, 'species_dim',     ntotal, kdim)
      
      istatus= NF_DEF_DIM(ncid,'nt_deltadim', nt_delta, nt_deltadim)
      istatus= NF_DEF_DIM(ncid,'nr_deltadim', nr_delta, nr_deltadim)
      istatus= NF_DEF_DIM(ncid,'nz_deltadim', nz_delta, nz_deltadim)

      istatus= NF_DEF_DIM(ncid, 'chardim',      8,    chardim)
      istatus= NF_DEF_DIM(ncid, 'char64dim',   64,    char64dim)
      
      !YuP[2019-09] Added, to save bnumb_imp(), dens_imp(), etc.
      istatus= NF_DEF_DIM(ncid,'impur_dim', nstates+1,impur_nstates_dim)
      !Note: If no impurities are present, nstates is 0

      if(ampfmod.eq.'enabled')then
      start_elecfldn(1)=1 !0
      start_elecfldn(2)=1 !0
      start_elecfldn(3)=1
      count_elecfldn(1)=lrz+2 !Starts at l=0. Put boundary value as last
                              !radial entry.
      count_elecfldn(2)=nstop+1
      count_elecfldn(3)=nampfmax+1
      print*,'count_elecfldn(1:3)=',count_elecfldn
      istatus= NF_DEF_DIM(ncid,'nampfmax_dim',nampfmax+1,nampfmax_dim)
      istatus= NF_DEF_DIM(ncid,'nonch_dim',   nstop+1,   nonch_dim) 
      istatus= NF_DEF_DIM(ncid,'nrho_dim',    lrz+1,     nrho_dim) 
      istatus= NF_DEF_DIM(ncid,'nrho_dim1',   lrz+2,     nrho_dim1) 
      endif

      istatus= NF_DEF_DIM(ncid,'xlm_dim',jfl,xlm_dim) ![2022-03-19] For fl(jfl,lrors)

c     unlimited dimension for time, dimension name= 'tdim'
      !istatus= NF_DEF_DIM(ncid, 'tdim',NF_UNLIMITED,tdim) 
      !YuP: why NF_UNLIMITED needed?
      istatus= NF_DEF_DIM(ncid, 'tdim',nstop+1,tdim) !YuP[2018-09-28]
      !changed to nstop+1, to accomodate dimsf() logic, see below.
      
      !YuP[2018-09-28] Added 'tsavedim' to save timet 
      ! at selected t steps (corr. to nsave(i)=n steps)
      ! Can have only one NF_UNLIMITED ?
      !istatus=NF_DEF_DIM(ncid,'tsavedim',NF_UNLIMITED,tsavedim) ! failed
      ! nsavet is counted in tdchief
      istatus= NF_DEF_DIM(ncid,'tsavedim',nsavet+1,tsavedim)

c     define vector of dimensions, unlimited last
      dims(1)=ydim
      dims(2)=xdim
      dims(3)=rdim
      dims(4)=tdim

      !dimsg() are used for saving of 'f'==f(), in case of ngen>1
      dimsg(1)=ydimf
      dimsg(2)=xdimf
      if(cqlpmod.ne."enabled")then ! CQL3D
      dimsg(3)=rdim !CQL3D
      else ! CQLP  !YuP[2022-02-09] added
      dimsg(3)=zdim !CQLP
      endif
      dimsg(4)=gdim
      dimsg(5)=tdim ! can be re-defined, see below

      !dimsf() are used for saving of 'f'==f(), in case of ngen=1
      dimsf(1)=ydimf
      dimsf(2)=xdimf
      if(cqlpmod.ne."enabled")then ! CQL3D
      dimsf(3)=rdim !CQL3D
      else ! CQLP  !YuP[2022-02-09] added
      dimsf(3)=zdim !CQLP
      endif
      ! Now define the remaining dimsf(4:5)
      if (netcdfshort.eq.'enabled') then
         !Do nothing: no storage defined.
      elseif ( (netcdfshort.eq.'longer_f').or.
     +         (netcdfshort.eq.'lngshrtf')     ) then ! define storage
         if (ngen.eq.1) then    !maintaining backwards compatability
            !to be set below: vid=ncvdef2(ncid,'f',NCDOUBLE,4,dimsf,istatus)
            dimsf(4)=tdim
            if(netcdfshort.eq.'lngshrtf') dimsf(4)=tsavedim
         else  !ngen.ge.2
            !to be set below: vid=ncvdef2(ncid,'f',NCDOUBLE,5,dimsg,istatus) 
            !Additional dim included for ngen.gt.1 cases
            dimsg(4)=gdim ! to contain 'k' index
            dimsg(5)=tdim !
            if(netcdfshort.eq.'lngshrtf') dimsg(5)=tsavedim 
         endif  !on ngen
      else     !disabled, Standard o/p: f at last time step
         if (ngen.eq.1) then    !maintaining backwards compatability
            !to be set below: vid=ncvdef2(ncid,'f',NCDOUBLE,3,dimsf,istatus)
         else  !ngen.ge.2
            !to be set below: vid=ncvdef2(ncid,'f',NCDOUBLE,4,dimsg,istatus)
            !Additional dim included for ngen.gt.1 cases
            dimsg(4)=gdim !! to contain 'k' index
         endif  !on ngen
      endif  !on netcdfshort : storage defined

      dims_fl(1)=xlm_dim  !=jfl [2022-03-19] for saving xlm and fl()
      count_fl(1)=jfl
      if(cqlpmod.ne."enabled")then ! CQL3D
      dims_fl(2)=rdim !CQL3D
      count_fl(2)=lrz
      else ! CQLP 
      dims_fl(2)=zdim !CQLP
      count_fl(2)=lz
      endif
      dims_fl(3)=tdim
      count_fl(3)=1 ! time dimension
      ![2022-03-19] saving fl() - for one species only, for now.

      r_dims(1)=rdim
      r_dims(2)=tdim

      rk_dims(1)=rdim
      rk_dims(2)=gdim
      rk_dims(3)=tdim

      r0_dims(1)=r0dim
      r0_dims(2)=tdim

      r0k_dims(1)=r0dim
      r0k_dims(2)=gdim
      r0k_dims(3)=tdim

      r00_dims(1)=r00dim
      r00_dims(2)=tdim
      
      if(cqlpmod.ne."enabled")then !CQL3D
      y_dims(1)=ydim
      y_dims(2)=rdim ! for CQL3D
      else ! cqlpmod.eq."enabled" CQLP  !YuP[2022-02-09]added
      y_dims(1)=ydim
      y_dims(2)=zdim ! for CQLP
      endif

      species_dims(1)=kdim
      species_dims(2)=r0dim
      species_dims(3)=tdim

      tau_dims(1)=ydim
      tau_dims(2)=r0dim

      z_dims(1)=ydim
      z_dims(2)=zdim
      z_dims(3)=r0dim

      if (softxry.ne."disabled" .and. softxry.ne."ncdf_all") then
         xray_dims(1)=nendim
         xray_dims(2)=nvdim
         xray_dims(3)=twodim    !First and last sets of spectra
      elseif ( softxry.eq."ncdf_all") then
         xray_dims(1)=nendim
         xray_dims(2)=nvdim
         xray_dims(3)=tdim      !All sets of spectra
      endif

      if (npa_diag.ne."disabled" .and. npa_diag.ne."ncdf_all") then
         npa_dims(1)=nen_npadim
         npa_dims(2)=nv_npadim
         npa_dims(3)=twodim    !First and last sets of spectra
      elseif ( npa_diag.eq."ncdf_all") then
         npa_dims(1)=nen_npadim
         npa_dims(2)=nv_npadim
         npa_dims(3)=tdim      !All sets of spectra
      endif

      if (npa_diag.ne."disabled") then
         npaproc_dims(1)=chardim
         npaproc_dims(2)=npaproc_dim
         npaenn_dims(1)=r0dim
         npaenn_dims(2)=npaproc_dim
      endif
c
      hibr_dims(2)=kedim
      hibr_dims(3)=kbdim
      sorpw_dims(1)=ngenadim
      sorpw_dims(2)=lrzadim
c
      fus_dims(1)=r0dim       !Radial profiles at last step
      fus_dims(2)=fourdim
      fus_dims(3)=tdim

      dims_powurf(1)=mrfnp1_dim
      dims_powurf(2)=tdim

      rfpwr_dims(1)=rdim
      rfpwr_dims(2)=mrfnp3_dim
      rfpwr_dims(3)=tdim

      dims_powrf(1)=rdim
      dims_powrf(2)=nmodsdim
      dims_powrf(3)=tdim

      dims_powrft(1)=rdim
      dims_powrft(2)=tdim

      powers_dims(1)=rdim
      powers_dims(2)=thirteendim
      powers_dims(3)=gdim
      powers_dims(4)=tdim

      currv_dims(1)=xdim
      currv_dims(2)=rdim
      currv_dims(3)=tdim

      currv_dimsg(1)=xdim
      currv_dimsg(2)=rdim
      currv_dimsg(3)=gdim
      currv_dimsg(4)=tdim

      delta_dims(1)=nr_deltadim
      delta_dims(2)=nz_deltadim
      delta_dims(3)=nt_deltadim

Cdeltarhop      deltap_dims(1)=nt_deltadim
Cdeltarhop      deltap_dims(2)=zdim
Cdeltarhop      deltap_dims(3)=r0dim

      kspeci_dims(1)=chardim
      kspeci_dims(2)=twodim
      kspeci_dims(3)=kdim

      if(ampfmod.eq.'enabled')then
      elecfldn_dims(1)= nrho_dim1 ! correspond to lrz+2 (0:lrz+1)
      elecfldn_dims(2)= nonch_dim ! corr. to nstop+1   (0:nstop)
      elecfldn_dims(3)= nampfmax_dim                 ! (0:nampfmax)
      endif

      !YuP[2019-09] Added, to save bnumb_imp(), dens_imp(), etc.
      impur_dims(1)=impur_nstates_dim  ! nstates+1
      impur_dims(2)=rdim ! lrz
      impur_dims(3)=tdim
      
      
c.......................................................................
cl    1.1.3 define variables

c     Note, the variable IDs (denoted below as vid) are
c     not saved here in this subroutine; rather, the IDs
c     are retrieved from the netCDF data file, as needed,
c     by calling the netCDF routine ncvid.

c     netcdf variable_define:
c     nf_def_var(netcdf_id,variable_name,variable_type,
c                number_of_dimensions,
c                vector_for_length_per_dimension,varid)
c     Note: Unlimited dimension must be last
c     Refer to p. 50, netcdf-3 manual

c     NF_DOUBLE for REAL*8:


c--------------------------
c     Time-independent data
c--------------------------
      
      ltitle='Main netcdf output from CQL3D version: '//version
      if( length_char(ltitle).gt.100 ) stop 'Adjust ltitle in netcdfrw2'
      call ncaptc2(ncid,NCGLOBAL,'title',NCCHAR,length_char(ltitle),
     +     ltitle,istatus)

      vid=ncvdef2(ncid,'version',NCCHAR,1,char64dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +          'CQL3D version number',istatus)

      vid=ncvdef2(ncid,'mnemonic',NCCHAR,1,char64dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +          'Mnemonic run identifier',istatus)

      vid=ncvdef2(ncid,'ampfmod',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +          'Ampere-Faraday module switch',istatus)

      vid=ncvdef2(ncid,'urfmod',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,17,
     +          'URF module switch',istatus)

      vid=ncvdef2(ncid,'rdcmod',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,45,
     +          'RF Heating and CD from input diffusion coeffs',istatus)

      vid=ncvdef2(ncid,'frmod',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,17,
     +          'NBI module switch',istatus)

      vid=ncvdef2(ncid,'beamplse',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,17,
     +          'Pulsed NBI switch',istatus)

      vid=ncvdef2(ncid,'transp',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +          'Radial transport module switch',istatus)

      vid=ncvdef2(ncid,'tavg',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +          'Indicates calc of time-avg distn favg',istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +          'in which case favg is o/p in place of f,',istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,35,
     +          'except in netcdfshort=longer_f case',istatus)
      
c.......................................................................

      vid=ncvdef2(ncid,'f4d_out',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,44,
     +          'Indicates output for 4D distn/ separate file',istatus)

cBH181025:  Not used !YuP[2023-09-26] revived
      vid=ncvdef2(ncid,'f3d_out',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,44,
     +          'Indicates output for 3D distn/ separate file',istatus)

      vid=ncvdef2(ncid,'netcdfshort',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,46,
     +        'Indicates level of output of data to .nc files',istatus)

      vid=ncvdef2(ncid,'eqdskin',NCCHAR,1,char64dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +          'Name of input eqdsk, for eqsource=eqdsk',istatus)

      vid=ncvdef2(ncid,'ngen',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +           'Number of general species',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'ntotal',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,26,
     +           'Number of species: gen+max',istatus)
      call check_err(istatus)
      
      vid=ncvdef2(ncid,'kspeci',NCCHAR,3,kspeci_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,44,
     +          'Name of species and spec as general or maxwl',istatus)
      
      vid=ncvdef2(ncid,'bnumb',NCDOUBLE,1,kdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +           'atomic number or each species',istatus)
      call check_err(istatus)
      
      vid=ncvdef2(ncid,'fmass',NCDOUBLE,1,kdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +           'mass of each species',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +                     'grams',istatus)
      call check_err(istatus)
      
      vid=ncvdef2(ncid,'lrzmax',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,38,
     +          'Number of radial bins(=r0dim, .ge.lrz)',istatus)
      call check_err(istatus)
      
      vid=ncvdef2(ncid,'radcoord',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,45,
     +          'Radial coordinate is proportional to radcoord',istatus)

      vid=ncvdef2(ncid,'rya',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +           'Normalized radial mesh at bin centers',istatus)
     
      vid=ncvdef2(ncid,'Rp',NCDOUBLE,1,r0dim,istatus)   ! rpcon(lrz) array
      call check_err(istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +           'Major radius of bin centers at outerboard',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'Rm',NCDOUBLE,1,r0dim,istatus)   ! rmcon(lrz) array
      call check_err(istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +           'Major radius of bin centers at innerboard',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'rhomax',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
     +           'Generalized plasma minor radius',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'radmaj',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +           'Nominal major radius',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'rpmconz',NCDOUBLE,1,r00dim,istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,43,
     +           'Major radius of bin boundaries on the outer',istatus)
      call ncaptc2(ncid,vid,'long_name2',NCCHAR,36,
     +           'equatorial plane, starting at radmaj',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'btor',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +           'Nominal tor mag fld at radmaj',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +           'Gauss',istatus)

      vid=ncvdef2(ncid,'toteqd',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +           'Tor equilibrium plasma current',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,4,
     +           'Amps',istatus)
      
      vid=ncvdef2(ncid,'rgeomp',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +           '0.5*(max-min) of major radius',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'r0geomp',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +           '0.5*(max+min) of major radius',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'rmag',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,26,
     +           'Magnetic axis major radius',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'zmag',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
     +           'Magnetic axis vertical position',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'eqsym',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +          'Indicator for symmetrization',istatus)

c.......................................................................
      vid=ncvdef2(ncid,'zshift',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +           'Vertical shift of equilibrium per eqsym',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'eps0',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +           'Inv aspect ratio = rgeomp/r0geomp',istatus)

      vid=ncvdef2(ncid,'elong',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,26,
     +           'Elongation = zgeomp/rgeomp',istatus)

      vid=ncvdef2(ncid,'zgeomp',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,26,
     +           'Approx half-height to LCFS',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'rgeom1',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +           'Approx inner major radius to LCFS',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'rgeom2',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +           'Approx outer major radius to LCFS',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'area',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +           'Cumulative area to bin centers',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +           'cms^2',istatus)

      vid=ncvdef2(ncid,'darea',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,35,
     +           'Incremental tor area of radial bins',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +           'cms^2',istatus)

      vid=ncvdef2(ncid,'vol',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +           'Cumulative volume to bin centers',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +           'cms^3',istatus)

      vid=ncvdef2(ncid,'dvol',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +           'Incremental volume to bin centers',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +           'cms^3',istatus)

      vid=ncvdef2(ncid,'equilpsi',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +           'Poloidal flux function at bin centers',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cgs',istatus)

      vid=ncvdef2(ncid,'psivalm',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,46,
     +           'Pol flux fctn at radial outer edge of each bin',
     +           istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cgs',istatus)

      vid=ncvdef2(ncid,'dpsi',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,44,
     +           'Incremental pol flux function at bin centers',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cgs',istatus)

      vid=ncvdef2(ncid,'psimag',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,34,
     +           'Pol flux function at magnetic axis',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cgs',istatus)

      vid=ncvdef2(ncid,'psilim',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,36,
     +           'Pol flux function at plasma boundary',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cgs',istatus)

      vid=ncvdef2(ncid,'h_r',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +           'H*rho ONETWO geometry factor',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'qsafety',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +           'Safety factor at bin centers',istatus)

c     Output current calculated from eqdsk:
      if (eqmod.eq."enabled") then
      vid=ncvdef2(ncid,'curreq',NCDOUBLE,1,r0dim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,35,
     +        'Toroidal current density from EQDSK',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,9,
     +        'Amps/cm**2',istatus)
      endif

      vid=ncvdef2(ncid,'lrz',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +            'Number of FPd radial surface bins (=rdim)',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'lrindx',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +           'Radial indices of FPd surfaces',istatus)

      vid=ncvdef2(ncid,'jx',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,35,
     +            'momentum-per-mass dimension (=xdim)',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'x',NCDOUBLE,1,xdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +           'normalized momentum-per-mass',istatus)
     
      vid=ncvdef2(ncid,'xl',NCDOUBLE,1,xlm_dim,istatus) ![2022-03-19] for saving fl()
      call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +           'normalized parallel momentum-per-mass',istatus)

      vid=ncvdef2(ncid,'enerkev',NCDOUBLE,1,xdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
     +           'energy=restmkev*(gamma(x(:))-1)',istatus)

      vid=ncvdef2(ncid,'uoc',NCDOUBLE,1,xdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,34,
     +           'uoc=x(:)/cnorm=mom-per-mass/clight',istatus)

      vid=ncvdef2(ncid,'dx',NCDOUBLE,1,xdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +           'dx centered on x-mesh points',istatus)

      vid=ncvdef2(ncid,'cint2',NCDOUBLE,1,xdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,7,
     +           'x**2*dx',istatus)

      vid=ncvdef2(ncid,'vnorm',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +           'velocity (momentum-per-mass) norm',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +                     'cms/sec',istatus)

      vid=ncvdef2(ncid,'enorm',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +                     'Energy normalization',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'keV',istatus)

      vid=ncvdef2(ncid,'iy',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +            'max Pitch angle dimension (=ydim)',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'y',NCDOUBLE,2,y_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,11,
     +           'pitch angle',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +           'radians',istatus)

      vid=ncvdef2(ncid,'dy',NCDOUBLE,2,y_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +           'dy centered on y-mesh points',istatus)

      vid=ncvdef2(ncid,'cynt2',NCDOUBLE,2,y_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,14,
     +           '2*pi*sin(y)*dy',istatus)

      vid=ncvdef2(ncid,'iy_',NCLONG,1,y_dims(2),istatus) 
      ![2022-02-09] changed to y_dims(2) which can be either lrzmax or lz
      !depending on cqlpmod
      call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +            'Pitch angle dimension at each r (le ydim)',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'itl',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +           'lower trapped-passing bndy index',istatus)

      vid=ncvdef2(ncid,'itu',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +           'upper trapped-passing bndy index',istatus)

      vid=ncvdef2(ncid,'lz',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +            'dimension of z-grid along B',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'z',NCDOUBLE,2,z_dims(2),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,16,
     +           'Distance along B',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'cms',istatus)

      vid=ncvdef2(ncid,'dz',NCDOUBLE,2,z_dims(2),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +           'dz centered on z-points',istatus)

      vid=ncvdef2(ncid,'solrz',NCDOUBLE,2,z_dims(2),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,21,
     +           'Major radius of z pts',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'cms',istatus)

      vid=ncvdef2(ncid,'solzz',NCDOUBLE,2,z_dims(2),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,15,
     +           'Height of z pts',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'cms',istatus)

      vid=ncvdef2(ncid,'pol',NCDOUBLE,2,z_dims(2),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +           'Poloidal angle, measured about mag axis',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +                     'radians',istatus)

      vid=ncvdef2(ncid,'bbpsi',NCDOUBLE,2,z_dims(2),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,12,
     +           'B(z)/B(z=0)',istatus)

      vid=ncvdef2(ncid,'imax',NCLONG,2,z_dims(2),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +           'Max i s.t. part passes z(l)',istatus)

      vid=ncvdef2(ncid,'lmax',NCLONG,2,tau_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,24,
     +           'Max l s.t. i passes z(l)',istatus)

      vid=ncvdef2(ncid,'zboun',NCDOUBLE,2,tau_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +           'Bounce point z-value',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'cms',istatus)

      vid=ncvdef2(ncid,'zmaxpsi',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,15,
     +           'Integral dz/bbpsi',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'cms',istatus)

      vid=ncvdef2(ncid,'tau',NCDOUBLE,2,tau_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +           'tau_bounce * abs(speed)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'cms',istatus)

      vid=ncvdef2(ncid,'dtau',NCDOUBLE,3,z_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,24,
     +          'abs(speed)*dtau in dz(l)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'cms',istatus)

      vid=ncvdef2(ncid,'beampon',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +          'On time, per cylce, of beam',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,4,
     +                     'secs',istatus)

      vid=ncvdef2(ncid,'beampoff',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +          'Off time, per cylce, of beam',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,4,
     +                     'secs',istatus)

      vid=ncvdef2(ncid,'tavg1',NCDOUBLE,1,ntavgdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
     +          'tavg case:  interval start time',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,4,
     +                     'secs',istatus)

      vid=ncvdef2(ncid,'tavg2',NCDOUBLE,1,ntavgdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +          'tavg case:  interval stop time',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,4,
     +                     'secs',istatus)

      vid=ncvdef2(ncid,'ndeltarho',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +          'Indicates first order-orbit shift options',istatus)
      
c.......................................................................

      ndelta_op="enabled"
      if( (ndeltarho.ne."disabled").and.(ndelta_op.eq."enabled"))then 
         !YuP[2020-10-20] Removed lossmode from if() below. 
         !deltarho-related arrays are not used in sub.losscone, for now.
!YuP      if ((ndeltarho.ne.'disabled'.or.lossmode(1).eq.'simplban').and.
!YuP     +     ndelta_op.eq."enabled") then
         vid=ncvdef2(ncid,'deltarho',NCDOUBLE,3,z_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,43,
     +        'Orbit shift FROM BA FS vs theta0,z,rya,/|v|',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,26,
     +        "Norm'd to rhomax*|x*vnorm|",istatus)

Cdeltarhop         vid=ncvdef(ncid,'deltarhop',NCDOUBLE,3,deltap_dims,istatus)
Cdeltarhop         call ncaptc2(ncid,vid,'long_name',NCCHAR,48,
Cdeltarhop     +       'Orbit shift FROM BA FS vs local theta,z,rya,/|v|',istatus)
Cdeltarhop         call ncaptc2(ncid,vid,'units',NCCHAR,26,
Cdeltarhop     +        "Norm'd to rhomax*|x*vnorm|",istatus)

         vid=ncvdef2(ncid,'r_delta',NCDOUBLE,1,nr_deltadim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +        'deltarz major radius mesh',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'cms',istatus)

         vid=ncvdef2(ncid,'z_delta',NCDOUBLE,1,nz_deltadim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +        'deltarz vertical Z-mesh',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'cms',istatus)

         vid=ncvdef2(ncid,'t_delta',NCDOUBLE,1,nt_deltadim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +        'deltarz local pitch angle mesh',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +        'radians',istatus)

         vid=ncvdef2(ncid,'deltarz',NCDOUBLE,3,delta_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,46,
     +        'Orbit shift TO BA FS vs local theta,R,Z,/|v|',istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,49,
     +        'Thus, co-current ions at any R,Z have neg deltarz',
     +        istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,54,
     +        "Norm'd to max of coord specified by radcoord*|x*vnorm|",
     +        istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,55,
     +        'I.e., normed to 1.*velocity (cms/sec) for most radcoord',
     +        istatus)

         vid=ncvdef2(ncid,'delta_bdb0',NCDOUBLE,2,delta_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,54,
     +        'Ratio of mag fld B at deltarz R,Z over B0 at midplane',
     +        istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,8,
     +        "unitless",istatus)

      endif  ! On deltarho
      if(ampfmod.eq.'enabled')then
         vid=ncvdef2(ncid,'elecfldn',NCDOUBLE,3,elecfldn_dims,istatus)
c         WRITE(*,*)'elecfldn_dims,ncid,vid,istatus=',
c     +              elecfldn_dims,ncid,vid,istatus
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,53,
     +   'Tor. Electric Field as func of iteration,time,radius,',
     +   istatus)
         call ncaptc2(ncid,vid,'long_name3',NCCHAR,48,
     +   'Bin centered, except last value the bndry value.',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,12,
     +                         'statVolts/cm',istatus)
         ! elecfldn,delecfld0 are in cgs !
      endif


      vid=ncvdef2(ncid,'bthr',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,35,
     +           'Equil Pol B field at theta_pol=pi/2',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +                     'Gauss',istatus)

      vid=ncvdef2(ncid,'btoru',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,35,
     +           'Equil Tor B field at theta_pol=pi/2',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +                     'Gauss',istatus)

      vid=ncvdef2(ncid,'btor0',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,48,
     +       'Tor mag fld strength at min |B| on flux surfaces',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +                     'Gauss',istatus)

      vid=ncvdef2(ncid,'bmidplne',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,38,
     +           'Min mag fld |B| on a rad flux surfaces',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +                     'Gauss',istatus)

      vid=ncvdef2(ncid,'efflag',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +          'Indicates elecfld is toroidal or parallel',istatus)

c     X-ray data:

      vid=ncvdef2(ncid,'softxry',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,55,
     +     'X-ray diagnostic is disabled,enabled, ncdf_all, or e-ion' ,
     +     istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,36,
     +      'Data for 1st and last step is output' ,istatus)
      call ncaptc2(ncid,vid,'long_name2',NCCHAR,35,
     +      'or, data for all steps for ncdf_all' ,istatus)
      call check_err(istatus)

      if (softxry .ne. "disabled") then
 
         if (x_sxr(1).ne.zero  .or. z_sxr(1).ne.zero) then

      vid=ncvdef2(ncid,'x_sxr',NCDOUBLE,1,nvdim,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,42,
     +           'X-ray detector major radius (tor angle=0.)',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'z_sxr',NCDOUBLE,1,nvdim,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,21,
     +           'X-ray detector height',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

         else

      vid=ncvdef2(ncid,'rd',NCDOUBLE,1,nvdim,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +           'X-ray detector minor radius ',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'thetd',NCDOUBLE,1,nvdim,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +           'X-ray detector poloidal angle',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +           'degrees',istatus)

         endif

      vid=ncvdef2(ncid,'nv',NCLONG,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +        'Number of X-ray viewing cords',istatus)
         call check_err(istatus)
 
      vid=ncvdef2(ncid,'nen',NCLONG,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +        'Number of equispaced energies in spectra',istatus)
         call check_err(istatus)
 
      vid=ncvdef2(ncid,'msxr',NCLONG,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,47,
     +        'Highest order of Legendre poly in f expr for XR',istatus)
         call check_err(istatus)

      vid=ncvdef2(ncid,'enmin',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
     +        'X-ray minimun energy in spectra',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'keV',istatus)

      vid=ncvdef2(ncid,'enmax',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
     +        'X-ray maximum energy in spectra',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'keV',istatus)
         
      vid=ncvdef2(ncid,'en_',NCDOUBLE,1,nendim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,13,
     +        'Photon Energy',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'keV',istatus)
         
      vid=ncvdef2(ncid,'eflux',NCDOUBLE,3,xray_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,17,
     +        'X-ray energy flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,22,
     +        'ergs/cm**2/sec/ster/eV',istatus)
         
      vid=ncvdef2(ncid,'efluxt',NCDOUBLE,2,xray_dims(2),istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +        'Integrated X-ray energy flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,19,
     +        'ergs/cm**2/sec/ster',istatus)

      endif ! On softxry .ne. "disabled"

      
      !YuP[2019-09] Moved this section outside of 
      ! if (npa_diag .ne. "disabled") section.
      ! Let the *.nc file contain this information
      ! regardless of npa_dia, 
      ! in case we want to use neutral density for other purpose.
      vid=ncvdef2(ncid,'ipronn',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,39,
     +      'disabled, default, zero neutral density' ,istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,51,
     +   'exp, ennl exponential falloff with dist from radmin' ,istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,46,
     +   'spline, give neutral density profiles vs ryain' ,istatus)
      call check_err(istatus)
      
      !YuP[2019-09] added [[
      vid=ncvdef2(ncid,'gamafac',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,48,
     &  'energy dependent Coulomb log (default: disabled)' ,istatus)
     
      vid=ncvdef2(ncid,'pellet',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,36,
     &  'Pellet injection (default: disabled)' ,istatus)
     
      vid=ncvdef2(ncid,'imp_type',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     &  'Impurity type for gamafac=hesslow',istatus)
     
      vid=ncvdef2(ncid,'nstates',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,36,
     &  'Number of charge states for impurity',istatus)
     
      vid=ncvdef2(ncid,'fmass_imp',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,21,
     &  'mass of impurity atom',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,4,
     &  'gram',istatus)

      vid=ncvdef2(ncid,'pellet_M0',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,22,
     &  'Initial mass of pellet',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,4,
     &  'gram',istatus)

      vid=ncvdef2(ncid,'pellet_Cablation',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     &  'Coefficient for ablation rate of pellet',istatus)
     
      if(nstates.gt.0)then ! If no impurities, nstates=0. 
        vid=ncvdef2(ncid,'bnumb_imp',NCDOUBLE,1,impur_nstates_dim,
     &   istatus)
     
        call ncaptc2(ncid,vid,'long_name',NCCHAR,62,
     &  'Charge number for each ionization state (incl.Z=0 for neutral)'
     &   ,istatus)
     
        vid=ncvdef2(ncid,'dens_imp_allstates',NCDOUBLE,2,impur_dims(2:3)
     &    ,istatus)
        call ncaptc2(ncid,vid,'long_name',NCCHAR,47,
     &  'Density of impurity, all charge states together',istatus)
        call ncaptc2(ncid,vid,'units',NCCHAR,6,
     +  '/cm**3',istatus)

        vid=ncvdef2(ncid,'dens_imp',NCDOUBLE,3,impur_dims(1:3),istatus)
        call ncaptc2(ncid,vid,'long_name',NCCHAR,53,
     +  'Density of impurity, for each charge state (incl.Z=0)',istatus)
        call ncaptc2(ncid,vid,'units',NCCHAR,6,
     +  '/cm**3',istatus)
     
        vid=ncvdef2(ncid,'pellet_rho',NCDOUBLE,1,impur_dims(3),istatus)
        call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     &  'rho at pellet position at each time step',istatus)
     
        vid=ncvdef2(ncid,'Gablation',NCDOUBLE,1,impur_dims(3),istatus)
        call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     &  'Ablation rate of pellet at each time step',istatus)
        call ncaptc2(ncid,vid,'units',NCCHAR,8,
     +  'gram/sec',istatus)

        vid=ncvdef2(ncid,'pellet_Mrem',NCDOUBLE,1,impur_dims(3),istatus)
        call ncaptc2(ncid,vid,'long_name',NCCHAR,48,
     &  'Remaining mass[gram] of pellet at each time step',istatus)
      endif ! nstates.gt.0
      !YuP[2019-09] added ]]
      

c     NPA data:

      vid=ncvdef2(ncid,'npa_diag',NCCHAR,1,chardim,istatus)
      call check_err(istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,47,
     +      'NPA diagnostic is disabled,enabled, or ncdf_all' ,istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,45,
     +      'enabled: Data for 1st and last step is output' ,istatus)
      call ncaptc2(ncid,vid,'long_name2',NCCHAR,45,
     +      'or, ncdf_all: data for all steps for ncdf_all' ,istatus)

      if (npa_diag .ne. "disabled") then
 
         if (x_npa(1).ne.zero  .or. z_npa(1).ne.zero) then

            vid=ncvdef2(ncid,'x_npa',NCDOUBLE,1,nv_npadim,istatus)
            call check_err(istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +           'NPA detector major radius (tor angle=0.)',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

            vid=ncvdef2(ncid,'z_npa',NCDOUBLE,1,nv_npadim,istatus)
            call check_err(istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,19,
     +           'NPA detector height',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

         else ! rd_npa/thetd_npa input

            vid=ncvdef2(ncid,'rd_npa',NCDOUBLE,1,nv_npadim,istatus)
            call check_err(istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +           'NPA detector minor radius ',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

            vid=ncvdef2(ncid,'thetd_npa',NCDOUBLE,1,nv_npadim,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +           'NPA detector poloidal angle',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +           'degrees',istatus)
         endif

            vid=ncvdef2(ncid,'nv_npa',NCLONG,0,0,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +        'Number of NPA viewing cords',istatus)
            call check_err(istatus)
 
            vid=ncvdef2(ncid,'nen_npa',NCLONG,0,0,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +        'Number of equispaced energies in spectra',istatus)
            call check_err(istatus)
 
            vid=ncvdef2(ncid,'npaproc',NCLONG,0,0,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +        'Maximum number of CX processes considered',istatus)
            call check_err(istatus)

            vid=ncvdef2(ncid,'enmin_npa',NCDOUBLE,0,0,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +        'NPA minimun energy in spectra',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'keV',istatus)

            vid=ncvdef2(ncid,'enmax_npa',NCDOUBLE,0,0,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +        'NPA maximum energy in spectra',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'keV',istatus)

         vid=ncvdef2(ncid,'npa_process',NCCHAR,2,npaproc_dims,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,54,
     +      '(1)cxh,(2)cxb4,(3)cxhe,(4)cxc,(5)radrecom, or notset()',
     +       istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,40,
     +      'Indicates which CX process(es) are included:' ,istatus)
         call check_err(istatus)

         vid=ncvdef2(ncid,'atten_npa',NCCHAR,1,chardim,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,42,
     +      'enabled, normal calculation of attenuation' ,istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,40,
     +      'disabled, for numerical testing purposes' ,istatus)
         call check_err(istatus)

         vid=ncvdef2(ncid,'en_',NCDOUBLE,1,nendim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,17,
     +        'CX Neutral Energy',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'keV',istatus)
         
         vid=ncvdef2(ncid,'ennscal',NCDOUBLE,1,npaproc_dim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,51,
     +     'Scale factor for density assoc with each CX process'
     +     ,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,49,
     +     'ennscal has been used in calc of the enn profiles'
     +     ,istatus)
         
      vid=ncvdef2(ncid,'enn',NCDOUBLE,2,npaenn_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +        'Density associated with each npa_process',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,6,
     +        '/cm**3',istatus)
         
      vid=ncvdef2(ncid,'eflux_npa',NCDOUBLE,3,npa_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,12,
     +        'Neutral flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,19,
     +        '#/cm**2/sec/ster/eV',istatus)
         
         vid=ncvdef2(ncid,'efluxt',NCDOUBLE,2,npa_dims(2),istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +        'Integrated neutral flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,16,
     +        '#/cm**2/sec/ster',istatus)

      endif  ! On npa_diag .ne. "disabled"


c     Fusion reaction rate data
      vid=ncvdef2(ncid,'sigmamod',NCCHAR,1,chardim,istatus)
      call check_err(istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,46,
     +      'fusion rates diagnostic is disabled or enabled',
     +      istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,45,
     +      'Data for Rx 1:4 is output (see cqlinput_help)' ,istatus)
      call check_err(istatus)

      if (sigmamod .ne. "disabled") then ! define netcdf names

         vid=ncvdef2(ncid,'isigmas',NCLONG,1,fourdim,istatus)
         call check_err(istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +        'Indicators of the particular rates calcd',istatus)

         vid=ncvdef2(ncid,'mmsv',NCLONG,0,0,istatus)
         call check_err(istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,44,
     +        'Max order Legendre expansion of Rx cross-sec',istatus)

         vid=ncvdef2(ncid,'isigsgv1',NCLONG,0,0,istatus)
         call check_err(istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,47,
     +        'Indicator of Rx due to gen distn with self, =0 ',istatus)

         vid=ncvdef2(ncid,'isigsgv2',NCLONG,0,0,istatus)
         call check_err(istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +        'Indicator of Bgrnd Max not included, =0 ',istatus)
         call check_err(istatus)

      endif  !  On sigmamod
         
c--------------------------
c     Time-dependent data
c--------------------------

      vid=ncvdef2(ncid,'time',NCDOUBLE,1,tdim,istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +                     'seconds',istatus)

      vid=ncvdef2(ncid,'bctshift',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,36,
     +        'Time shift of bctime(), for restarts',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +        'seconds',istatus)

      vid=ncvdef2(ncid,'bctimescal',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,38,
     +        'Scale factor of bctime(), stretching t',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +        'seconds',istatus)

      !YuP[2018-09-28], BH181112 added for 'lngshrtf' option,
      !for saving f() distr.func. at selected (nsave()) t steps only.
      vid=ncvdef2(ncid,'nsave',NCLONG,1,tsavedim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +           'Selected time steps, n.eq.nsave(1:nsavet)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,4,'none',istatus)

      if((netcdfshort.eq.'lngshrtf').and.isave.ne.0) then  !isave/nsavet set in tdchief
        !YuP[2019-06-08] Added  if()
        ! Define 'tsave' only for this case, otherwise 'tsave' will be empty
      vid=ncvdef2(ncid,'tsave',NCDOUBLE,1,tsavedim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +           'Times selected using n.eq.nsave(1:nsavet)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,'seconds',istatus)
      endif

      vid=ncvdef2(ncid,'den_e',NCDOUBLE,2,r00_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,16,
     +           'Electron density',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,6,
     +           '/cm**3',istatus)

      vid=ncvdef2(ncid,'density',NCDOUBLE,3,species_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,34,
     +           'Densities, general and Maxwellians',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,6,
     +           '/cm**3',istatus)

      vid=ncvdef2(ncid,'zeff',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,4,
     +           'Zeff',istatus)

      !YuP[07-31-2014] Added:
      vid=ncvdef2(ncid,'consn',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,59,
     +  'Particle Conservation (ptcln(lr)-ptcln_t0(lr))/ptcln_t0(lr)',
     +  istatus)
      !YuP[07-31-2014] Added:
      vid=ncvdef2(ncid,'bscurr_e_gen',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,58,
     +  'Bootstrap current: Hirshman-Sauter Model. e_general (FPed)',
     +  istatus)
      !YuP[07-31-2014] Added:
      vid=ncvdef2(ncid,'bscurr_e_maxw',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,54,
     +  'Bootstrap current: Hirshman-Sauter Model. e_maxwellian',
     +  istatus)
      !YuP[07-31-2014] Added:
      vid=ncvdef2(ncid,'bscurr_i_gen',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,58,
     +  'Bootstrap current: Hirshman-Sauter Model. i_general (FPed)',
     +  istatus)
      !YuP[07-31-2014] Added:
      vid=ncvdef2(ncid,'bscurr_i_maxw',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,54,
     +  'Bootstrap current: Hirshman-Sauter Model. i_maxwellian',
     +  istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,'A/cm**2',istatus)
      !YuP[2019-12-23] Added:
      vid=ncvdef2(ncid,'currpar_starnue0',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     &  'Current based on sigma_banana (starnue=0)',
     &  istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,'A/cm**2',istatus)
      !YuP[2019-12-23] Added:
      vid=ncvdef2(ncid,'currpar_starnue',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,43,
     &  'Current based on sigma_coll-neo (starnue>0)',
     &  istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,'A/cm**2',istatus)


      vid=ncvdef2(ncid,'temp',NCDOUBLE,3,species_dims,istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'keV',istatus)

      vid=ncvdef2(ncid,'energy',NCDOUBLE,3,species_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +           'FSA Energy per particle',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'keV',istatus)

      if (ngen.eq.1) then
         vid=ncvdef2(ncid,'wpar',NCDOUBLE,2,r_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +        'FSA Parallel Energy per particle',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'keV',istatus)
         
         vid=ncvdef2(ncid,'wperp',NCDOUBLE,2,r_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +        'FSA Perpendicular Energy per particle',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'keV',istatus)
      else  !  ngen.ge.2
         vid=ncvdef2(ncid,'wpar',NCDOUBLE,3,rk_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +        'FSA Parallel Energy per particle',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'keV',istatus)
         
         vid=ncvdef2(ncid,'wperp',NCDOUBLE,3,rk_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +        'FSA Perpendicular Energy per particle',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'keV',istatus)
      endif  !  on ngen

      vid=ncvdef2(ncid,'elecfld',NCDOUBLE,2,r00_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +           'Parallel Electric Field',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,8,
     +           'Volts/cm',istatus)

      vid=ncvdef2(ncid,'edreicer',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,45,
     +          'E_D Dreicer elec fld, e.g., Kulsrud PRL(1973)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,8,
     +           'Volts/cm',istatus)

      vid=ncvdef2(ncid,'runaway_rate',NCDOUBLE,2,r_dims,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,45,
     +          'Runaway rate, determined from e flux off grid',istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,39,
     +          'Runaway rate = 1/n * dn/dt / nu_Kulsrud',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,8,
     +           'Unitless',istatus)

      vid=ncvdef2(ncid,'denra',NCDOUBLE,2,r_dims,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,31,
     +          'Runaway FSA density above ucrit',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,6,
     +           '/cm**3',istatus)

      vid=ncvdef2(ncid,'curra',NCDOUBLE,2,r_dims,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,44,
     +          'Runaway FSA parallel cur density above ucrit',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,10,
     +           'Amps/cm**2',istatus)

      vid=ncvdef2(ncid,'ucrit',NCDOUBLE,2,r_dims,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,38,
     +          'Critical momentum per mass for runaway',istatus)
cBH071013      call ncaptc2(ncid,vid,'units',NCCHAR,6,
cBH071013     +           'cm/sec',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,19,
     +           'Normalized to vnorm',istatus)

c     Knockon electron data
      vid=ncvdef2(ncid,'knockon',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,47,
     +      'Knockon src of high energy elec is en-/disabled' ,istatus)

      if (knockon.ne."disabled") then
         vid=ncvdef2(ncid,'eoe0',NCDOUBLE,2,r_dims,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,39,
     +        'Elecfld/Critical knockon electric field',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,8,
     +        'Unitless',istatus)
         
         vid=ncvdef2(ncid,'srckotot',NCDOUBLE,2,r_dims,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,31,
     +        'FSA Knockon source density rate',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,11,
     +        '#/cm**3*sec',istatus)
         
         vid=ncvdef2(ncid,'denfl',NCDOUBLE,2,r_dims,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,38,
     +        'FSA Elec Density from KO Reduced Distn',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +        '#/cm**3',istatus)
      endif  ! on knockon
      
      !if(transp.eq."enabled")then !could add this condition later.
      if(read_data.eq.'nimrod')then
         vid=ncvdef2(ncid,'dbb2',NCDOUBLE,2,dims(3:4),istatus) !(lrz,time)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,28,
     +        '(deltaB/B)^2 from data files',istatus)
      endif
      !endif !transp
      
      
c     In this section: we assume not both rdcmod and urfmod.ne.disabled
c     mrfn=number of modes =1, in subroutine rdc_multi.
      if (rdcmod.ne."disabled") then
         if (rdcmod.eq."enabled" .and. urfmod.ne."disabled") then
            WRITE(*,*)
            WRITE(*,*)'Warning: netcdfrw2 not set up for both'
            WRITE(*,*)'Warning: rdcmod and urfmod.ne.disabled.'
            WRITE(*,*)'Warning: urfmod output will overwrite rdcmod o/p'
            WRITE(*,*)
         endif
         vid=ncvdef2(ncid,'rfpwr',NCDOUBLE,3,rfpwr_dims,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,40,
     +        'RF power densities (sorpw_rf(*,1:mrfn)):',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,41,
     +        'Radially integrated: sorpw_rfi(*,mrfn+3)=',istatus)
         call ncaptc2(ncid,vid,'long_name2',NCCHAR,40,
     +        'Summed rf+nbi pwr den: sorpwt(*,mrfn+2)=',istatus)
         call ncaptc2(ncid,vid,'long_name3',NCCHAR,39,
     +        'Radially integrated: sorpwti(*,mrfn+3)=',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,37,
     +        'Watts/cm**3, except Watts for sorpwti',istatus)
         
      endif  !On rdcmod

c
c... new Freya stuff
c
	WRITE(*,*) frmodp
      if (frmodp.eq."enabled") then

           vid=ncvdef2(ncid,'hibrz',NCDOUBLE,3,hibr_dims,istatus)
           call ncaptc2(ncid,vid,'long_name0',NCCHAR,29,
     +         'Normalized hot ion birth rate',istatus)
     
           vid=ncvdef2(ncid,'sorpw_nbi',NCDOUBLE,2,sorpw_dims,istatus)
           call ncaptc2(ncid,vid,'long_name0',NCCHAR,28,
     +         'NBI+FUS Source Power Density',istatus)
           call ncaptc2(ncid,vid,'units',NCCHAR,10,
     +         'Watts/cm^3',istatus)
     
           vid=ncvdef2(ncid,'sorpw_nbii',NCDOUBLE,2,sorpw_dims,istatus)
           call ncaptc2(ncid,vid,'long_name0',NCCHAR,38,
     +         'Radially Integrated Power from NBI+FUS',istatus)
           call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +         'Watts',istatus)
          !YuP[06-2016] Now the corresponding arrays contain
          ! a COMBINED NBI+FUSproduct source.  
	  ! But the netcdf names like 'sorpw_nbi' are not changed 
	  ! because Python plotting script must be changed in such case.

      endif
c

      if (urfmod.ne."disabled") then

         vid=ncvdef2(ncid,'mrfn',NCLONG,0,0,istatus) ! YuP[2017-11]added
         call ncaptc2(ncid,vid,'long_name',NCCHAR,59,
     +   'number of rf modes (sum over all wave types and all nharms)',
     +   istatus)

         vid=ncvdef2(ncid,'powurf',NCDOUBLE,2,dims_powurf,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,33,
     +        'URF power in each mode, and total',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,41,
     +        'powurf(1:mrfn)=power for individual modes',istatus)
         call ncaptc2(ncid,vid,'long_name2',NCCHAR,38,
     +        'powurf(mrfn+1)=power summed over modes',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +        'Watts',istatus)
         
         vid=ncvdef2(ncid,'rfpwr',NCDOUBLE,3,rfpwr_dims,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,37,
     +        'RF power densities (powrf(*,1:mrfn)):',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,45,
     +        'rfpwr(*,1:mrfn)=pwr den from individual modes',istatus)
         call ncaptc2(ncid,vid,'long_name2',NCCHAR,24,
     +        'rfpwr(*,mrfn+1)=powrft()',istatus)
         call ncaptc2(ncid,vid,'long_name3',NCCHAR,40,
     +        'Summed rf+nbi pwr den: sorpwt(*,mrfn+2)=',istatus)
         call ncaptc2(ncid,vid,'long_name4',NCCHAR,39,
     +        'Radially integrated: sorpwti(*,mrfn+3)=',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,37,
     +        'Watts/cm**3, except Watts for sorpwti',istatus)
         
         vid=ncvdef2(ncid,'powrf',NCDOUBLE,3,dims_powrf,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,47,
     +        'RF power densities due to mode (or harmonic for',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,39,
     +        'nharms.gt.1 cases [powrf(lrza,nmodsa)])',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,11,
     +        'Watts/cm**3',istatus)

         vid=ncvdef2(ncid,'nrfspecies',NCLONG,1,dims_powrf(2),istatus) !nmodsa
         !YuP[2017-11] added
         call ncaptc2(ncid,vid,'long_name',NCCHAR,60,
     +   'nrfspecies(nmodsa)= general species index for each wave type',
     +   istatus)
         call check_err(istatus)
         
         vid=ncvdef2(ncid,'powrfl',NCDOUBLE,3,dims_powrf,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,46,
     +        'RF power densities due to salphal(1:nmodsdim)',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,40,
     +        '(For multi-harmonic or multi-mode cases)',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,11,
     +        'Watts/cm**3',istatus)
         
         vid=ncvdef2(ncid,'powurfl',NCDOUBLE,2,dims_powrf(2),istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,46,
     +        'Tot RF power absrbd due to salphal(1:nmodsdim)',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,40,
     +        '(For multi-harmonic or multi-mode cases)',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +        'Watts',istatus)
         
         vid=ncvdef2(ncid,'powrfc',NCDOUBLE,3,dims_powrf,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,50,
     +     'Coll RF power densities due to salphac(1:nmodsdim)',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,40,
     +        '(For multi-harmonic or multi-mode cases)',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,11,
     +        'Watts/cm**3',istatus)
         
         vid=ncvdef2(ncid,'powurfc',NCDOUBLE,2,dims_powrf(2),istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,49,
     +     'Tot Coll RF pwr absrbd due to salphac(1:nmodsdim)',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,40,
     +        '(For multi-harmonic or multi-mode cases)',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +        'Watts',istatus)
         
         vid=ncvdef2(ncid,'powrft',NCDOUBLE,2,dims_powrft,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,50,
     +     'RF power densities summed over modes or harmonics,',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,43,
     +        'due to urf, collisional and add. linear abs',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,11,
     +        'Watts/cm**3',istatus)
         
      endif  !On urfmod
     
      
      vid=ncvdef2(ncid,'curtor',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,36,
     +           'Toroidal current density at min B pt',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,10,
     +           'Amps/cm**2',istatus)

      vid=ncvdef2(ncid,'ccurtor',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +           'Area Integrated toroidal current',istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,39,
     +           'accounting for pol variation of tor cur',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,4,
     +           'Amps',istatus)

      vid=ncvdef2(ncid,'curpol',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,36,
     +           'Poloidal current density at min B pt',istatus)
       call ncaptc2(ncid,vid,'units',NCCHAR,10,
     +           'Amps/cm**2',istatus)

      vid=ncvdef2(ncid,'ccurpol',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,44,
     +           'Integrated poloidal current density at min B',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,4,
     +           'Amps',istatus)

      
      if (kelecg.ne.0) then ! e as gen.species
         vid=ncvdef2(ncid,'currm_e',NCDOUBLE,2,r0_dims,istatus)
      else ! ion as general species
         vid=ncvdef2(ncid,'currm_i',NCDOUBLE,2,r0_dims,istatus)
      endif
      
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,38,
     +           'Parallel elec current density at min B',istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,43,
     +           'Electrons, or first gen species if kelecg=0',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,10,
     +           'Amps/cm**2',istatus)

      if (ngen.eq.1) then
         vid=ncvdef2(ncid,'curr',NCDOUBLE,2,r0_dims,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,20,
     +        'FSA Parallel current',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,44,
     +        'i.e., Par curr per poloidal area between FSs',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,10,
     +        'Amps/cm**2',istatus)
         !YuP[07-31-2014] Added:
         vid=ncvdef2(ncid,'energym',NCDOUBLE,2,r_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +           'Energy per particle, from f0 at midplane',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'keV',istatus)
      else  !  ngen.ge.2
         vid=ncvdef2(ncid,'curr',NCDOUBLE,3,r0k_dims,istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,20,
     +        'FSA Parallel current',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,44,
     +        'i.e., Par curr per poloidal area between FSs',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,10,
     +        'Amps/cm**2',istatus)
         !YuP[07-31-2014] Added:
         vid=ncvdef2(ncid,'energym',NCDOUBLE,3,r0k_dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +           'Energy per particle, from f0 at midplane',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'keV',istatus)

      endif !ngen

      vid=ncvdef2(ncid,'restp',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,19,
     +           '<E_phi/R>/<j_phi/R>',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,12,
     +           'cgs, seconds',istatus)

      vid=ncvdef2(ncid,'restnp',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,45,
     +          'neoclassical resist <E_parall*B>/<j_parall*B>',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,12,
     +           'cgs, seconds',istatus)

      vid=ncvdef2(ncid,'sptzrp',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,41,
     +           'Spitzer resistivity, incl Zeff dependence',istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,41,
     +           'Eq. 4.2-77 of ONETWO Manual, Eq. 5.66 H&H',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,12,
     +           'cgs, seconds',istatus)

      vid=ncvdef2(ncid,'rovsc',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,31,
     +           'Connor resistivity over Spitzer',istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,42,
     +           'J.W. Connor et al, Nucl Fus 13, 211 (1973)',istatus)

      vid=ncvdef2(ncid,'rovsc_hi',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,39,
     +           'Connor resistivity over Spitzer, hi eps',istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,42,
     +           'J.W. Connor et al, Nucl Fus 13, 211 (1973)',istatus)

      vid=ncvdef2(ncid,'zreskim',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,44,
     +           'Hirshman/Sigmar/Kim resistivity over Spitzer',istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,43,
     +           'Reference is Hir/Sig (~1988) and Kim theses',istatus)

      vid=ncvdef2(ncid,'taueeh',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,36,
     +           'Hinton-Hazeltine(Eq 5.4)-ONETWO-taue',istatus)

      vid=ncvdef2(ncid,'nuestar',NCDOUBLE,2,r0_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +           'ONETWO Manual, Eq. 4.2-30',istatus)
         
      vid=ncvdef2(ncid,'powers',NCDOUBLE,4,powers_dims,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,55,
     +    'Component by component FSA powers to gen species k vs t',
     +     istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,53,
     +  'powers(*,1,k,t)=due to collisions with Maxw electrons',istatus)
      call ncaptc2(ncid,vid,'long_name2',NCCHAR,48,
     +   'powers(*,2,k,t)=due to collisions with Maxw ions',istatus)
      call ncaptc2(ncid,vid,'long_name3',NCCHAR,25,
     +   'powers(*,3,k,t)=Ohmic E.v',istatus)
      call ncaptc2(ncid,vid,'long_name4',NCCHAR,51,
     +   'powers(*,4,k,t)=due to collisions with general spec',istatus)
      call ncaptc2(ncid,vid,'long_name5',NCCHAR,24,
     +   'powers(*,5,k,t)=RF power',istatus)
      call ncaptc2(ncid,vid,'long_name6',NCCHAR,35,
     +   'powers(*,6,k,t)=Ion particle source',istatus)
      call ncaptc2(ncid,vid,'long_name7',NCCHAR,34,
     +   'powers(*,7,k,t)=losses by lossmode',istatus)
      call ncaptc2(ncid,vid,'long_name8',NCCHAR,33,
     +   'powers(*,8,k,t)=losses by torloss',istatus)
      call ncaptc2(ncid,vid,'long_name9',NCCHAR,30,
     +   'powers(*,9,k,t)=Runaway losses',istatus)
      call ncaptc2(ncid,vid,'long_name10',NCCHAR,45,
     +   'powers(*,10,k,t)=Synchrotron radiation losses',istatus)
      call ncaptc2(ncid,vid,'long_name11',NCCHAR,39,
     +   'powers(*,11,k,t)=Setting neg. j to zero',istatus)
      call ncaptc2(ncid,vid,'long_name12',NCCHAR,40,
     +   'powers(*,12,k,t)=Phenomenological losses',istatus)
      call ncaptc2(ncid,vid,'long_name13',NCCHAR,22,
     +   'powers(*,13,k,t)=Total',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,11,
     +   'Watts/cm**3',istatus)
         
      vid=ncvdef2(ncid,'powers_int',NCDOUBLE,3,powers_dims(2),istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,58,
     +    'Vol int of FSA powers, respectively, to gen species k vs t',
     +     istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +   'Watts',istatus)

      if (sigmamod.eq.'enabled') then ! define netcdf names
      vid=ncvdef2(ncid,'sigftt',NCDOUBLE,2,fus_dims(2),istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +        'Total fusion Rx rates vs time',istatus)
      endif


      if (netcdfshort.eq.'enabled') then
c        Do nothing: no storage defined.
      elseif ( (netcdfshort.eq.'longer_f').or.
     +         (netcdfshort.eq.'lngshrtf')     ) then ! define storage

         if (ngen.eq.1) then    !maintaining backwards compatability
            vid=ncvdef2(ncid,'f',NCDOUBLE,4,dimsf(1:4),istatus) ! ngen=1
            ! here dimsf={ydimf,xdimf,rdim,tdim(or tsavedim)}
            call ncaptc2(ncid,vid,'long_name',NCCHAR,21,
     +           'Distribution function',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,28,
     +           'vnorm**3/(cm**3*(cm/sec)**3)',istatus)
         else  !ngen.ge.2
            vid=ncvdef2(ncid,'f',NCDOUBLE,5,dimsg(1:5),istatus) ! ngen>1
                !Additional dim included for ngen.gt.1 cases
            ! here dimsg={ydimf,xdimf,rdim,gdim,tdim(or tsavedim)}
            call ncaptc2(ncid,vid,'long_name',NCCHAR,21,
     +           'Distribution function',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,28,
     +           'vnorm**3/(cm**3*(cm/sec)**3)',istatus)
            call ncaptc2(ncid,vid,'comment',NCCHAR,44,
     +           'Additional dimension added for multi-species',istatus)
         endif  !on ngen
         
      else     !disabled, Standard o/p: f at last time step
      
         if (ngen.eq.1) then    !maintaining backwards compatability
            vid=ncvdef2(ncid,'f',NCDOUBLE,3,dimsf(1:3),istatus)
            if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
            write(*,*)'dims=',dims
            write(*,*)'dimsf=',dimsf
            write(*,*)'dimsg=',dimsg
            endif
            ! here dimsf={ydimf,xdimf,rdim}
            call ncaptc2(ncid,vid,'long_name',NCCHAR,21,
     +           'Distribution function',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,28,
     +           'vnorm**3/(cm**3*(cm/sec)**3)',istatus)
            call ncaptc2(ncid,vid,'comment',NCCHAR,39,
     +           'Facility set up only for single species',istatus)
         else  !ngen.ge.2
            vid=ncvdef2(ncid,'f',NCDOUBLE,4,dimsg(1:4),istatus)
                !Additional dim included for ngen.gt.1 cases
            ! here dimsg={ydimf,xdimf,rdim,gdim}
            call ncaptc2(ncid,vid,'long_name',NCCHAR,21,
     +           'Distribution function',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,28,
     +           'vnorm**3/(cm**3*(cm/sec)**3)',istatus)
            call ncaptc2(ncid,vid,'comment',NCCHAR,44,
     +           'Additional dimension added for multi-species',istatus)
         endif  !on ngen
         
      endif  !on netcdfshort : storage defined


cBH100912:  Option to output specific curr and rf pwr at each time step
      if (netcdfshort.eq.'long_jp') then

      if (ngen.eq.1) then
      vid=ncvdef2(ncid,'currv',NCDOUBLE,3,currv_dims(1:3),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,44,
     +           'Specific Current Density j_u(u) at each step',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,43,
     +           'Amps/cm^2 (int:0,1 over dx =current density',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,34,
     +           'Facility set up for single species',istatus)

      vid=ncvdef2(ncid,'pwrrf',NCDOUBLE,3,currv_dims(1:3),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,47,
     +        'Specific RF Power Density pwrrf(u) at each step',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,44,
     +          'W/cm^3 (int:0,1 over dx =RF power density',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,34,
     +           'Facility set up for single species',istatus)
     
      !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
      vid=ncvdef2(ncid,'favr_thet0',NCDOUBLE,3,currv_dims(1:3),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,52,
     &  'Distr function at midplane averaged over pitch angle',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,28,
     &  'vnorm**3/(cm**3*(cm/sec)**3)',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,50,
     +  'Facility set up for single species. Each time step',istatus)

      else  !ngen.ge.2
      
      vid=ncvdef2(ncid,'currv',NCDOUBLE,4,currv_dimsg(1:4),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,44,
     +           'Specific Current Density j_u(u) at each step',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,43,
     +           'Amps/cm^2 (int:0,1 over dx =current density',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,23,
     +           'Setup for multi-species',istatus)

      vid=ncvdef2(ncid,'pwrrf',NCDOUBLE,4,currv_dimsg(1:4),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,47,
     +        'Specific RF Power Density pwrrf(u) at each step',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,41,
     +          'W/cm^3 (int:0,1 over dx =RF power density',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,23,
     +           'Setup for multi-species',istatus)
     
      !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
      vid=ncvdef2(ncid,'favr_thet0',NCDOUBLE,4,currv_dimsg(1:4),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,52,
     &  'Distr function at midplane averaged over pitch angle',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,28,
     &  'vnorm**3/(cm**3*(cm/sec)**3)',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,39,
     +  'Setup for multi-species. Each time step',istatus)
     
      endif  !on ngen

      !YuP[2022-03-19] Added fl()==F_par reduced distr.func.
      vid=ncvdef2(ncid,'fl',NCDOUBLE,3,dims_fl(1:3),istatus) !netcdfshort='long_jp'
      call ncaptc2(ncid,vid,'long_name',NCCHAR,34,
     &  'Reduced distr function Fpar(v_par)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,26,
     &  'vnorm*Fpar_cgs(v_parallel)',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,50,
     &  'Facility set up for single species. EACH time step',istatus)
     
      else !(netcdfshort.ne.'long_jp')  ! Setup for last time step only

      if (ngen.eq.1) then
      
      vid=ncvdef2(ncid,'currv',NCDOUBLE,2,currv_dims(1:2),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
     +           'Specific Current Density j_u(u)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,44,
     +           'Amps/cm^2 (int:0,1 over dx =current density)',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,34,
     +           'Facility set up for single species',istatus)

      vid=ncvdef2(ncid,'pwrrf',NCDOUBLE,2,currv_dims(1:2),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,34,
     +           'Specific RF Power Density pwrrf(u)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,42,
     +          'W/cm^3 (int:0,1 over dx =RF power density)',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,34,
     +           'Facility set up for single species',istatus)

      !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
      vid=ncvdef2(ncid,'favr_thet0',NCDOUBLE,2,currv_dims(1:2),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,52,
     &  'Distr function at midplane averaged over pitch angle',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,28,
     &  'vnorm**3/(cm**3*(cm/sec)**3)',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,34,
     +  'Facility set up for single species',istatus)
     
      else  !ngen.ge.2
      
      vid=ncvdef2(ncid,'currv',NCDOUBLE,3,currv_dimsg(1:3),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
     +           'Specific Current Density j_u(u)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,44,
     +           'Amps/cm^2 (int:0,1 over dx =current density)',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,23,
     +           'Setup for multi-species',istatus)

      vid=ncvdef2(ncid,'pwrrf',NCDOUBLE,3,currv_dimsg(1:3),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,34,
     +           'Specific RF Power Density pwrrf(u)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,42,
     +          'W/cm^3 (int:0,1 over dx =RF power density)',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,23,
     +           'Setup for multi-species',istatus)
     
      !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
      vid=ncvdef2(ncid,'favr_thet0',NCDOUBLE,3,currv_dimsg(1:3),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,52,
     &  'Distr function at midplane averaged over pitch angle',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,28,
     &  'vnorm**3/(cm**3*(cm/sec)**3)',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,24,
     +  'Setup for multi-species.',istatus)

      endif  !on ngen
      
      !YuP[2022-03-19] Added fl()==F_par reduced distr.func.
      vid=ncvdef2(ncid,'fl',NCDOUBLE,2,dims_fl(1:2),istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,34,
     &  'Reduced distr function Fpar(v_par)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,26,
     &  'vnorm*Fpar_cgs(v_parallel)',istatus)
      call ncaptc2(ncid,vid,'comment',NCCHAR,50,
     &  'Facility set up for single species. Last time step',istatus)
     
      endif  !  On netcdfshort.eq.'long_jp'


c--------------------------
c     Last time step data
c--------------------------

cBH100912
c$$$      vid=ncvdef2(ncid,'currv',NCDOUBLE,2,currv_dims,istatus)
c$$$      call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
c$$$     +           'Specific Current Density j_u(u)',istatus)
c$$$      call ncaptc2(ncid,vid,'units',NCCHAR,44,
c$$$     +           'StatA/cm^2 (int:0,1 over dx =current density',istatus)
c$$$      call ncaptc2(ncid,vid,'comment',NCCHAR,39,
c$$$     +           'Facility set up only for single species',istatus)
c$$$
c$$$      vid=ncvdef2(ncid,'pwrrf',NCDOUBLE,2,currv_dims,istatus)
c$$$      call ncaptc2(ncid,vid,'long_name',NCCHAR,34,
c$$$     +           'Specific RF Power Density pwrrf(u)',istatus)
c$$$      call ncaptc2(ncid,vid,'units',NCCHAR,45,
c$$$     +          'StatA/cm^2 (int:0,1 over dx =RF power density',istatus)
c$$$      call ncaptc2(ncid,vid,'comment',NCCHAR,39,
c$$$     +           'Facility set up only for single species',istatus)


      if (sigmamod.eq.'enabled') then  ! define netcdf names
         vid=ncvdef2(ncid,'fuspwrvt',NCDOUBLE,1,fus_dims(2),istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,38,
     +        'Total fusion power, for four reactions',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,18,
     +        'At final time step',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +        'Watts',istatus)

         vid=ncvdef2(ncid,'fuspwrv',NCDOUBLE,2,fus_dims(1),istatus)
         call ncaptc2(ncid,vid,'long_name0',NCCHAR,46,
     +        'Fusion power versus radius, for four reactions',istatus)
         call ncaptc2(ncid,vid,'long_name1',NCCHAR,18,
     +        'At final time step',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,11,
     +        'Watts/cm**3',istatus)
      endif


c.......................................................................
cl    1.1.4 end the define-mode and start the data-mode
c     p. 51-2 of manual

c-YuP:      call ncendf(ncid,istatus)
      istatus= NF_ENDDEF(ncid) !-YuP: NetCDF-f77
      call check_err(istatus)

C-----------------------------------------------------------------------
cl    1.2 Write data:   First call to netcdfrw2
c

c --- set the time-step counter ==> numrec1
      numrec1=1
      numrecsave=1   ! for (netcdfshort.eq.'lngshrtf')
      start(4)=numrec1 !=1 here;  used for 'time', etc.
      start1(4)=numrec1
      startg(5)=numrec1 ! here numrec1=1 =numrecsave=1
      start_rk(3)=numrec1
      start_r0k(3)=numrec1
      start_powurf(2)=numrec1
      start_rfpwr(3)=numrec1
      start_powrf(3)=numrec1
      start_powrft(2)=numrec1
      start_powers(4)=numrec1
      start_fus(3)=numrec1
      start_xr(3)=numrec1
      start_npa(3)=numrec1

c --- initialize data file ---
c     First get variable_id: 
c             nf_inq_varid(netcdf_id,variable_name,integer_info)
c     Then write data with nc variable_put:
c             nf_put_var... ()
c
c     function nf_put_var...(ncid,variable_id,index,val)
c     p. 52,, 54-65 of netcdf-3 manual
c

c-YuP:      vid=ncvid(ncid,'version',istatus)
      istatus= NF_INQ_VARID(ncid,'version',vid)  
      ll=length_char(version)
      call ncvptc2(ncid,vid,1,ll,version,ll,istatus)

c-YuP:      vid=ncvid(ncid,'mnemonic',istatus)
      istatus= NF_INQ_VARID(ncid,'mnemonic',vid)  
c      call ncvptc2(ncid,vid,1,64,mnemonic,64,istatus)
      ll=length_char(mnemonic)
      call ncvptc2(ncid,vid,1,ll,mnemonic,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'ampfmod',vid)
      ll=length_char(ampfmod) !YuP[2019-06-21] was a BUG
      call ncvptc2(ncid,vid,1,ll,ampfmod,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'urfmod',vid)
      ll=length_char(urfmod)
      call ncvptc2(ncid,vid,1,ll,urfmod,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'rdcmod',vid)
      ll=length_char(rdcmod)
      call ncvptc2(ncid,vid,1,ll,rdcmod,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'frmod',vid)
      ll=length_char(frmodp)
      call ncvptc2(ncid,vid,1,ll,frmodp,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'beamplse',vid)
      ll=length_char(beamplsep)
      call ncvptc2(ncid,vid,1,ll,beamplsep,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'transp',vid)
      ll=length_char(transp)
      call ncvptc2(ncid,vid,1,ll,transp,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'tavg',vid)
      ll=length_char(tavg)
      call ncvptc2(ncid,vid,1,ll,tavg,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'f4d_out',vid)
      ll=length_char(f4d_out)
      call ncvptc2(ncid,vid,1,ll,f4d_out,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'f3d_out',vid)
      ll=length_char(f3d_out)
      call ncvptc2(ncid,vid,1,ll,f3d_out,ll,istatus)


      istatus= NF_INQ_VARID(ncid,'netcdfshort',vid)
      ll=length_char(netcdfshort)
      call ncvptc2(ncid,vid,1,ll,netcdfshort,ll,istatus)

c-YuP:      vid=ncvid(ncid,'eqdskin',istatus)
      istatus= NF_INQ_VARID(ncid,'eqdskin',vid)  
c      call ncvptc2(ncid,vid,1,64,eqdskin,64,istatus)
      ll=length_char(eqdskin)
      call ncvptc2(ncid,vid,1,ll,eqdskin,ll,istatus)

c-YuP:      vid=ncvid(ncid,'ngen',istatus)
      istatus= NF_INQ_VARID(ncid,'ngen',vid)  
      call ncvpt_int2(ncid,vid,1,1,ngen,istatus)

c-YuP:      vid=ncvid(ncid,'ntotal',istatus)
      istatus= NF_INQ_VARID(ncid,'ntotal',vid)  
      call ncvpt_int2(ncid,vid,1,1,ntotal,istatus)

ccc      vid=ncvid(ncid,'kspeci',istatus)
      istatus= NF_INQ_VARID(ncid,'kspeci',vid)  
      call ncvptc2(ncid,vid,start,kspeci_count,kspeci,8,istatus)

c-YuP:      vid=ncvid(ncid,'bnumb',istatus)
      istatus= NF_INQ_VARID(ncid,'bnumb',vid)  
      call ncvpt_doubl2(ncid,vid,1,ntotal,bnumb,istatus)

c-YuP:      vid=ncvid(ncid,'fmass',istatus)
      istatus= NF_INQ_VARID(ncid,'fmass',vid)  
      call ncvpt_doubl2(ncid,vid,1,ntotal,fmass,istatus)

c-YuP:      vid=ncvid(ncid,'lrzmax',istatus)
      istatus= NF_INQ_VARID(ncid,'lrzmax',vid)  
      call ncvpt_int2(ncid,vid,1,1,lrzmax,istatus)

c-YuP:      vid=ncvid(ncid,'radcoord',istatus)
      istatus= NF_INQ_VARID(ncid,'radcoord',vid)  
      call ncvptc2(ncid,vid,1,8,radcoord,8,istatus)

c-YuP:      vid=ncvid(ncid,'rya',istatus)
      istatus= NF_INQ_VARID(ncid,'rya',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,rya(1),istatus)
      
      istatus= NF_INQ_VARID(ncid,'Rp',vid)  ! rpcon(1:lrz) array
      call ncvpt_doubl2(ncid,vid,1,lrzmax,rpcon,istatus)      
      call check_err(istatus)

      istatus= NF_INQ_VARID(ncid,'Rm',vid)  ! rmcon(1:lrz) array
      call ncvpt_doubl2(ncid,vid,1,lrzmax,rmcon,istatus)      
      call check_err(istatus)


c-YuP:      vid=ncvid(ncid,'rhomax',istatus)
      istatus= NF_INQ_VARID(ncid,'rhomax',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,rhomax,istatus)

c-YuP:      vid=ncvid(ncid,'radmaj',istatus)
      istatus= NF_INQ_VARID(ncid,'radmaj',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,radmaj,istatus)

      istatus= NF_INQ_VARID(ncid,'rpmconz',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax+1,rpmconz(0),istatus)

c-YuP:      vid=ncvid(ncid,'btor',istatus)
      istatus= NF_INQ_VARID(ncid,'btor',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,btor,istatus)

c-YuP:      vid=ncvid(ncid,'toteqd',istatus)
      istatus= NF_INQ_VARID(ncid,'toteqd',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,toteqd/3.e9,istatus)

c-YuP:      vid=ncvid(ncid,'rgeomp',istatus)
      istatus= NF_INQ_VARID(ncid,'rgeomp',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,rgeomp,istatus)

c-YuP:      vid=ncvid(ncid,'r0geomp',istatus)
      istatus= NF_INQ_VARID(ncid,'r0geomp',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,r0geomp,istatus)

c-YuP:      vid=ncvid(ncid,'rmag',istatus)
      istatus= NF_INQ_VARID(ncid,'rmag',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,rmag,istatus)

c-YuP:      vid=ncvid(ncid,'zmag',istatus)
      istatus= NF_INQ_VARID(ncid,'zmag',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,zmag,istatus)

c-YuP:      vid=ncvid(ncid,'eqsym',istatus)
      istatus= NF_INQ_VARID(ncid,'eqsym',vid)  
      call ncvptc2(ncid,vid,1,8,eqsym,8,istatus)

c-YuP:      vid=ncvid(ncid,'zshift',istatus)
      istatus= NF_INQ_VARID(ncid,'zshift',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,zshift,istatus)

c-YuP:      vid=ncvid(ncid,'eps0',istatus)
      istatus= NF_INQ_VARID(ncid,'eps0',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,eps0,istatus)

c-YuP:      vid=ncvid(ncid,'elong',istatus)
      istatus= NF_INQ_VARID(ncid,'elong',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,zgeomp/rgeomp,istatus)

c-YuP:      vid=ncvid(ncid,'area',istatus)
      istatus= NF_INQ_VARID(ncid,'area',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,area(1),istatus)

c-YuP:      vid=ncvid(ncid,'darea',istatus)
      istatus= NF_INQ_VARID(ncid,'darea',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,darea(1),istatus)

c-YuP:      vid=ncvid(ncid,'vol',istatus)
      istatus= NF_INQ_VARID(ncid,'vol',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,vol(1),istatus)

c-YuP:      vid=ncvid(ncid,'dvol',istatus)
      istatus= NF_INQ_VARID(ncid,'dvol',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,dvol(1),istatus)

c-YuP:      vid=ncvid(ncid,'equilpsi',istatus)
      istatus= NF_INQ_VARID(ncid,'equilpsi',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,equilpsi(1),istatus)

      istatus= NF_INQ_VARID(ncid,'psivalm',vid)
      call ncvpt_doubl2(ncid,vid,1,lrzmax,psivalm(1),istatus)

c-YuP:      vid=ncvid(ncid,'psimag',istatus)
      istatus= NF_INQ_VARID(ncid,'psimag',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,psimag,istatus)

c-YuP:      vid=ncvid(ncid,'psilim',istatus)
      istatus= NF_INQ_VARID(ncid,'psilim',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,psilim,istatus)

c-YuP:      vid=ncvid(ncid,'dpsi',istatus)
      istatus= NF_INQ_VARID(ncid,'dpsi',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,dpsi(1),istatus)

c-YuP:      vid=ncvid(ncid,'h_r',istatus)
      istatus= NF_INQ_VARID(ncid,'h_r',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,h_r(1),istatus)

c-YuP:      vid=ncvid(ncid,'qsafety',istatus)
      istatus= NF_INQ_VARID(ncid,'qsafety',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,qsafety(1),istatus)
      
      if (eqmod.eq."enabled") then
c-YuP:      vid=ncvid(ncid,'curreq',istatus)
         istatus= NF_INQ_VARID(ncid,'curreq',vid)  
         call ncvpt_doubl2(ncid,vid,1,lrzmax,curreq(1),istatus)
      endif

c-YuP:      vid=ncvid(ncid,'lrz',istatus)
      istatus= NF_INQ_VARID(ncid,'lrz',vid)  
      call ncvpt_int2(ncid,vid,1,1,lrz,istatus)

c-YuP:      vid=ncvid(ncid,'lrindx',istatus)
      istatus= NF_INQ_VARID(ncid,'lrindx',vid)  
      call ncvpt_int2(ncid,vid,1,lrz,lrindx(1),istatus)

c-YuP:      vid=ncvid(ncid,'jx',istatus)
      istatus= NF_INQ_VARID(ncid,'jx',vid)  
      call ncvpt_int2(ncid,vid,1,1,jx,istatus)

c-YuP:      vid=ncvid(ncid,'x',istatus)
      istatus= NF_INQ_VARID(ncid,'x',vid)  
      call ncvpt_doubl2(ncid,vid,1,jx,x,istatus)

      istatus= NF_INQ_VARID(ncid,'xl',vid)  ![2022-03-19] for saving fl()
      call ncvpt_doubl2(ncid,vid,1,jfl,xlm,istatus)

      istatus= NF_INQ_VARID(ncid,'enerkev',vid) 
      call ncvpt_doubl2(ncid,vid,1,jx,enerkev(1:jx,1),istatus) ![2020-10-20] For k=1 species

      istatus= NF_INQ_VARID(ncid,'uoc',vid)
      call ncvpt_doubl2(ncid,vid,1,jx,uoc,istatus)

c-YuP:      vid=ncvid(ncid,'dx',istatus)
      istatus= NF_INQ_VARID(ncid,'dx',vid)  
      call ncvpt_doubl2(ncid,vid,1,jx,dx,istatus)

c-YuP:      vid=ncvid(ncid,'cint2',istatus)
      istatus= NF_INQ_VARID(ncid,'cint2',vid)  
      call ncvpt_doubl2(ncid,vid,1,jx,cint2,istatus)

c-YuP:      vid=ncvid(ncid,'vnorm',istatus)
      istatus= NF_INQ_VARID(ncid,'vnorm',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,vnorm,istatus)

c-YuP:      vid=ncvid(ncid,'enorm',istatus)
      istatus= NF_INQ_VARID(ncid,'enorm',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,enorm,istatus)

c-YuP:      vid=ncvid(ncid,'iy',istatus)
      istatus= NF_INQ_VARID(ncid,'iy',vid)  
      call ncvpt_int2(ncid,vid,1,1,iymax,istatus)
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)

      call pack21(y,1,iymax,1,lrors,wkpack,iymax,lrors)
      istatus= NF_INQ_VARID(ncid,'y',vid)  
      call ncvpt_doubl2(ncid,vid,start,y_count,wkpack,istatus)

      call pack21(dy,1,iymax,1,lrors,wkpack,iymax,lrors)
      istatus= NF_INQ_VARID(ncid,'dy',vid)  
      call ncvpt_doubl2(ncid,vid,start,y_count,wkpack,istatus)

      call pack21(cynt2,1,iymax,1,lrors,wkpack,iymax,lrors)
      istatus= NF_INQ_VARID(ncid,'cynt2',vid)  
      call ncvpt_doubl2(ncid,vid,start,y_count,wkpack,istatus)

      istatus= NF_INQ_VARID(ncid,'iy_',vid)  
      call ncvpt_int2(ncid,vid,1,y_count(2),iy_,istatus)
      !changed to y_count(2) which can be either lrz or lz
      !depending on cqlpmod

c-YuP:      vid=ncvid(ncid,'itl',istatus)
      istatus= NF_INQ_VARID(ncid,'itl',vid)  
      call ncvpt_int2(ncid,vid,1,lrz,itl_,istatus)

c-YuP:      vid=ncvid(ncid,'itu',istatus)
      istatus= NF_INQ_VARID(ncid,'itu',vid)  
      call ncvpt_int2(ncid,vid,1,lrz,itu_,istatus)

c-YuP:      vid=ncvid(ncid,'lz',istatus)
      istatus= NF_INQ_VARID(ncid,'lz',vid)  
      call ncvpt_int2(ncid,vid,1,1,lz,istatus)

      call pack21(z,1,lz,1,lrzmax,wkpack,lz,lrzmax)
c-YuP:      vid=ncvid(ncid,'z',istatus)
      istatus= NF_INQ_VARID(ncid,'z',vid)  
      call ncvpt_doubl2(ncid,vid,start(2:3),z_count(2:3),wkpack,istatus)

      call pack21(dz,1,lz,1,lrzmax,wkpack,lz,lrzmax)
c-YuP:      vid=ncvid(ncid,'dz',istatus)
      istatus= NF_INQ_VARID(ncid,'dz',vid)  
      call ncvpt_doubl2(ncid,vid,start(2:3),z_count(2:3),wkpack,istatus)

      call pack21(solrz,1,lz,1,lrzmax,wkpack,lz,lrzmax)
c-YuP:      vid=ncvid(ncid,'solrz',istatus)
      istatus= NF_INQ_VARID(ncid,'solrz',vid)  
      call ncvpt_doubl2(ncid,vid,start(2:3),z_count(2:3),wkpack,istatus)

      call pack21(solzz,1,lz,1,lrzmax,wkpack,lz,lrzmax)
c-YuP:      vid=ncvid(ncid,'solzz',istatus)
      istatus= NF_INQ_VARID(ncid,'solzz',vid)  
      call ncvpt_doubl2(ncid,vid,start(2:3),z_count(2:3),wkpack,istatus)

      call pack21(pol,1,lz,1,lrzmax,wkpack,lz,lrzmax)
c-YuP      vid=ncvid(ncid,'pol',istatus)
      istatus= NF_INQ_VARID(ncid,'pol',vid)  
      call ncvpt_doubl2(ncid,vid,start(2:3),z_count(2:3),wkpack,istatus)

      call pack21(bbpsi,1,lz,1,lrzmax,wkpack,lz,lrzmax)
c-YuP:      vid=ncvid(ncid,'bbpsi',istatus)
      istatus= NF_INQ_VARID(ncid,'bbpsi',vid)  
      call ncvpt_doubl2(ncid,vid,start(2:3),z_count(2:3),wkpack,istatus)

      call ipack21(imax,1,lz,1,lrzmax,iwkpack,lz,lrzmax) !YuP[2019-06-19] was wkpack. BUG?
      istatus= NF_INQ_VARID(ncid,'imax',vid)  
      call ncvpt_int2(ncid,vid,start(2:3),z_count(2:3),iwkpack,istatus)
ccc      call ncvpt_int2(ncid,vid,start(2:3),z_count(2:3),imax,istatus) !directly
      !YuP: the result is quite different when using pack21 or writing directly

      call ipack21(lmax,1,iymax,1,lrzmax,item1,iymax,lrzmax)
c-YuP:      vid=ncvid(ncid,'lmax',istatus)
      istatus= NF_INQ_VARID(ncid,'lmax',vid)  
      call ncvpt_int2(ncid,vid,start,tau_count,item1,istatus)
      
      call pack21(zboun,1,iymax,1,lrzmax,wkpack,iymax,lrzmax)
c-YuP:      vid=ncvid(ncid,'zboun',istatus)
      istatus= NF_INQ_VARID(ncid,'zboun',vid)  
      call ncvpt_doubl2(ncid,vid,start,tau_count,wkpack,istatus)

c-YuP:      vid=ncvid(ncid,'zmaxpsi',istatus)
      istatus= NF_INQ_VARID(ncid,'zmaxpsi',vid)  
      call ncvpt_doubl2(ncid,vid,start,tau_count(2),zmaxpsi(1),istatus)
      
      call pack21(tau,1,iymax,1,lrzmax,wkpack,iymax,lrzmax)
c-YuP:      vid=ncvid(ncid,'tau',istatus)
      istatus= NF_INQ_VARID(ncid,'tau',vid)  
      call ncvpt_doubl2(ncid,vid,start,tau_count,wkpack,istatus)
      
c-YuP:      vid=ncvid(ncid,'dtau',istatus)
      istatus= NF_INQ_VARID(ncid,'dtau',vid)  
      do ll=1,lrzmax
cBH         do l=1,lz
cBH            do i=1,iymax
cBH               temp1(i,l)=dtau(i,l,ll)
cBH            enddo
cBH         enddo
cBH         call pack21(temp1,0,iyp1,0,jxp1,wkpack,iymax,lz)
cBH  Actually, with dynamic dimensioning to exact size, can
cBH  leave out following step:
cBH         call pack21(dtau(1,1,ll),1,iymax,1,lz,wkpack,iymax,lz)
         start1(3)=ll
cBH         call ncvpt_doubl2(ncid,vid,start1,z_count1,wkpack,istatus)
         call ncvpt_doubl2(ncid,vid,start1(1:3),z_count1(1:3),
     +        dtau(1:iymax,1:lz,ll),istatus)
      enddo
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)

      istatus= NF_INQ_VARID(ncid,'beampon',vid)
      call ncvpt_doubl2(ncid,vid,1,1,beamponp,istatus)

      istatus= NF_INQ_VARID(ncid,'beampoff',vid)
      call ncvpt_doubl2(ncid,vid,1,1,beampoffp,istatus)

      istatus= NF_INQ_VARID(ncid,'tavg1',vid)
      call ncvpt_doubl2(ncid,vid,1,ntavga,tavg1,istatus)

      istatus= NF_INQ_VARID(ncid,'tavg2',vid)
      call ncvpt_doubl2(ncid,vid,1,ntavga,tavg2,istatus)

      istatus= NF_INQ_VARID(ncid,'ndeltarho',vid)
      call ncvptc2(ncid,vid,1,8,ndeltarho,8,istatus)
      
      if( (ndeltarho.ne."disabled").and.(ndelta_op.eq."enabled"))then 
         !YuP[2020-10-20] Removed lossmode from if() below. 
         !deltarho-related arrays are not used in sub.losscone, for now.
!YuP      if ((ndeltarho.ne.'disabled'.or.lossmode(1).eq.'simplban').and.
!YuP     +     ndelta_op.eq."enabled") then
cBH         vid=ncvid(ncid,'deltarho',istatus)
         istatus= NF_INQ_VARID(ncid,'deltarho',vid)
c$$$         do ll=1,lrzmax
c$$$            do l=1,lz
c$$$               do i=1,iymax
c$$$                  temp1(i,l)=deltarho(i,l,ll)
c$$$               enddo
c$$$            enddo
c$$$            call pack21(temp1,0,iyp1a,0,jxp1a,tem2,iymax,lz)
c$$$            start1(3)=ll
c$$$            call ncvpt(ncid,vid,start1,z_count1,tem2,istatus)
c$$$         enddo
c  Don't need above temp1 flail, since deltarho dynamically
c  dimensioned to size deltarho(iymax,lz,lrzmax)
cBH         call ncvpt(ncid,vid,start,z_count,deltarho,istatus)
         call ncvpt_doubl2(ncid,vid,start,z_count,deltarho,istatus)

         
Cdeltarhop         WRITE(*,*)'netcdfrw2:deltap_start,deltap_count=',
Cdeltarhop     +       deltap_start(1:3),deltap_count(1:3)
Cdeltarhop         vid=ncvid(ncid,'deltarhop',istatus)
Cdeltarhop         call ncvpt(ncid,vid,deltap_start,deltap_count,deltarhop,
Cdeltarhop     +        istatus)
        
cBH         vid=ncvid(ncid,'r_delta',istatus)
cBH         call ncvpt(ncid,vid,1,nr_delta,r_delta(1),istatus)
         istatus= NF_INQ_VARID(ncid,'r_delta',vid)
         call ncvpt_doubl2(ncid,vid,1,nr_delta,r_delta(1),istatus)
         
cBH         vid=ncvid(ncid,'z_delta',istatus)
cBH         call ncvpt(ncid,vid,1,nz_delta,z_delta(1),istatus)
         istatus= NF_INQ_VARID(ncid,'z_delta',vid)
         call ncvpt_doubl2(ncid,vid,1,nz_delta,z_delta(1),istatus)
         
cBH         vid=ncvid(ncid,'t_delta',istatus)
cBH         call ncvpt(ncid,vid,1,nt_delta,t_delta(1),istatus)
         istatus= NF_INQ_VARID(ncid,'t_delta',vid)
         call ncvpt_doubl2(ncid,vid,1,nt_delta,t_delta(1),istatus)
         
cBH         vid=ncvid(ncid,'deltarz',istatus)
cBH         call ncvpt_doubl2(ncid,vid,delta_start,delta_count,deltarz,istatus)
         istatus= NF_INQ_VARID(ncid,'deltarz',vid)
         call ncvpt_doubl2(ncid,vid,delta_start,delta_count,deltarz,
     +        istatus)
        WRITE(*,*)'netcdfrw2:start,z_count1=',
     +       start(1:3),z_count1(1:3)
        WRITE(*,*)'netcdfrw2:delta_start,delta_count=',
     +       delta_start(1:3),delta_count(1:3)

cBH         vid=ncvid_doubl2(ncid,'delta_bdb0',istatus)
cBH         call ncvpt(ncid,vid,delta_start,delta_count,delta_bdb0,
cBH     +        istatus)
         istatus= NF_INQ_VARID(ncid,'delta_bdb0',vid)
         call ncvpt_doubl2(ncid,vid,delta_start,delta_count,delta_bdb0,
     +        istatus)
         
      endif  ! On ndeltarho

c-YuP:      vid=ncvid(ncid,'bthr',istatus)
      istatus= NF_INQ_VARID(ncid,'bthr',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,bthr(1),istatus)

c-YuP:      vid=ncvid(ncid,'btoru',istatus)
      istatus= NF_INQ_VARID(ncid,'btoru',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,btoru(1),istatus)

c-YuP:      vid=ncvid(ncid,'btor0',istatus)
      istatus= NF_INQ_VARID(ncid,'btor0',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,btor0(1),istatus)

c-YuP:      vid=ncvid(ncid,'bmidplne',istatus)
      istatus= NF_INQ_VARID(ncid,'bmidplne',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,bmidplne(1),istatus)

c-YuP:      vid=ncvid(ncid,'softxry',istatus)
      istatus= NF_INQ_VARID(ncid,'softxry',vid)  
      call ncvptc2(ncid,vid,1,8,softxry,8,istatus)

      if (softxry .ne. "disabled") then
 
         if (x_sxr(1).ne.zero  .or. z_sxr(1).ne.zero) then

c-YuP:                vid=ncvid(ncid,'x_sxr',istatus)
            istatus= NF_INQ_VARID(ncid,'x_sxr',vid)  
            call ncvpt_doubl2(ncid,vid,1,nv,x_sxr(1),istatus)

c-YuP:                vid=ncvid(ncid,'z_sxr',istatus)
            istatus= NF_INQ_VARID(ncid,'z_sxr',vid)  
            call ncvpt_doubl2(ncid,vid,1,nv,z_sxr(1),istatus)

         else

c-YuP:                vid=ncvid(ncid,'rd',istatus)
            istatus= NF_INQ_VARID(ncid,'rd',vid)  
            call ncvpt_doubl2(ncid,vid,1,nv,rd(1),istatus)

c-YuP:                vid=ncvid(ncid,'thetd',istatus)
            istatus= NF_INQ_VARID(ncid,'thetd',vid)  
            call ncvpt_doubl2(ncid,vid,1,nv,thetd(1),istatus)

         endif

c-YuP:      vid=ncvid(ncid,'nv',istatus)
         istatus= NF_INQ_VARID(ncid,'nv',vid)  
         call ncvpt_int2(ncid,vid,1,1,nv,istatus)

c-YuP:      vid=ncvid(ncid,'nen',istatus)
         istatus= NF_INQ_VARID(ncid,'nen',vid)  
         call ncvpt_int2(ncid,vid,1,1,nen,istatus)

c-YuP:      vid=ncvid(ncid,'msxr',istatus)
         istatus= NF_INQ_VARID(ncid,'msxr',vid)  
         call ncvpt_int2(ncid,vid,1,1,msxr,istatus)
         
c-YuP:      vid=ncvid(ncid,'enmin',istatus)
         istatus= NF_INQ_VARID(ncid,'enmin',vid)  
         call ncvpt_doubl2(ncid,vid,1,1,enmin,istatus)
         
c-YuP:      vid=ncvid(ncid,'enmax',istatus)
         istatus= NF_INQ_VARID(ncid,'enmax',vid)  
         call ncvpt_doubl2(ncid,vid,1,1,enmax,istatus)

c-YuP:      vid=ncvid(ncid,'en_',istatus)
         istatus= NF_INQ_VARID(ncid,'en_',vid)  
         call ncvpt_doubl2(ncid,vid,1,nen,en_(1),istatus)

c-YuP:      vid=ncvid(ncid,'eflux',istatus)
         istatus= NF_INQ_VARID(ncid,'eflux',vid)  
         call pack21(eflux,1,nena,1,nva,wkpack,nen,nv)
         call ncvpt_doubl2(ncid,vid,start_xr,count_xr,wkpack,istatus)

c-YuP:      vid=ncvid(ncid,'efluxt',istatus)
         istatus= NF_INQ_VARID(ncid,'efluxt',vid)  
         call ncvpt_doubl2(ncid,vid,start_xr(2),count_xr(2),efluxt(1),
     +        istatus)

      endif  ! On softxry .ne. "disabled"

      istatus= NF_INQ_VARID(ncid,'npa_diag',vid)
      call ncvptc2(ncid,vid,1,8,npa_diag,8,istatus)
      
      !YuP[2019-09] Moved this section outside of 
      ! if (npa_diag .ne. "disabled") section.
      ! Let the *.nc file contain this information
      ! regardless of npa_dia, 
      ! in case we want to use neutral density for other purpose.
      istatus= NF_INQ_VARID(ncid,'ipronn',vid)
      call ncvptc2(ncid,vid,1,8,ipronn,8,istatus)
      
      !YuP[2019-09] added [[
      istatus= NF_INQ_VARID(ncid,'gamafac',vid)
      call ncvptc2(ncid,vid,1,8,gamafac,8,istatus)
      istatus= NF_INQ_VARID(ncid,'pellet',vid)
      call ncvptc2(ncid,vid,1,8,pellet,8,istatus)
      istatus= NF_INQ_VARID(ncid,'imp_type',vid)
      call ncvpt_int2(ncid,vid,1,1,imp_type,istatus)
      istatus= NF_INQ_VARID(ncid,'nstates',vid)
      call ncvpt_int2(ncid,vid,1,1,nstates,istatus)
      istatus= NF_INQ_VARID(ncid,'fmass_imp',vid)
      call ncvpt_doubl2(ncid,vid,1,1,fmass_imp,istatus)
      istatus= NF_INQ_VARID(ncid,'pellet_M0',vid)
      call ncvpt_doubl2(ncid,vid,1,1,pellet_M0,istatus)
      istatus= NF_INQ_VARID(ncid,'pellet_Cablation',vid)
      call ncvpt_doubl2(ncid,vid,1,1,pellet_Cablation,istatus)
      if(nstates.gt.0)then ! If no impurities, nstates=0. 
        istatus= NF_INQ_VARID(ncid,'bnumb_imp',vid)
        call ncvpt_doubl2(ncid,vid,1,nstates+1,bnumb_imp(0:nstates),
     &  istatus)
      endif
      !YuP[2019-09] added ]]


      if (npa_diag .ne. "disabled") then
 
         if (x_npa(1).ne.zero  .or. z_npa(1).ne.zero) then

            istatus= NF_INQ_VARID(ncid,'x_npa',vid)
            call ncvpt_doubl2(ncid,vid,1,nv_npa,x_npa(1),istatus)

            istatus= NF_INQ_VARID(ncid,'z_npa',vid)
            call ncvpt_doubl2(ncid,vid,1,nv_npa,z_npa(1),istatus)

         else

            istatus= NF_INQ_VARID(ncid,'rd_npa',vid)
            call ncvpt_doubl2(ncid,vid,1,nv_npa,rd_npa(1),istatus)

            istatus= NF_INQ_VARID(ncid,'thetd_npa',vid)
            call ncvpt_doubl2(ncid,vid,1,nv_npa,thetd_npa(1),istatus)

         endif

         istatus= NF_INQ_VARID(ncid,'nv_npa',vid)
         call ncvpt_int2(ncid,vid,1,1,nv_npa,istatus)

         istatus= NF_INQ_VARID(ncid,'nen_npa',vid)
         call ncvpt_int2(ncid,vid,1,1,nen_npa,istatus)
         
         istatus= NF_INQ_VARID(ncid,'npaproc',vid)
         call ncvpt_int2(ncid,vid,1,1,npaproc,istatus)
         
         istatus= NF_INQ_VARID(ncid,'enmin_npa',vid)
         call ncvpt_doubl2(ncid,vid,1,1,enmin_npa,istatus)
         
         istatus= NF_INQ_VARID(ncid,'enmax_npa',vid)
         call ncvpt_doubl2(ncid,vid,1,1,enmax_npa,istatus)

         istatus= NF_INQ_VARID(ncid,'npa_process',vid)
c         do ii=1,npaproc
c            npa_proc(ii)=npa_process(ii)
c         enddo
         call ncvptc2(ncid,vid,start_npaproc,count_npaproc,
     +               npa_process,8*npaproc,istatus)

         istatus= NF_INQ_VARID(ncid,'atten_npa',vid)
         call ncvptc2(ncid,vid,1,8,atten_npa,8,istatus)
        
         istatus= NF_INQ_VARID(ncid,'ennscal',vid)
         call ncvpt_doubl2(ncid,vid,1,npaproc,ennscal(1),istatus)

         istatus= NF_INQ_VARID(ncid,'en_',vid)
         call ncvpt_doubl2(ncid,vid,1,nen_npa,en_(1),istatus)
         
         istatus= NF_INQ_VARID(ncid,'enn',vid)
         call pack21(enn,1,lrza,1,npaproca,wkpack,lrzmax,npaproc)
         call ncvpt_doubl2(ncid,vid,start_npaenn,count_npaenn,wkpack,
     +        istatus)
         
         istatus= NF_INQ_VARID(ncid,'eflux_npa',vid)
         call pack21(eflux,1,nena,1,nva,wkpack,nen_npa,nv_npa)
         call ncvpt_doubl2(ncid,vid,start_npa,count_npa,wkpack,istatus)

         istatus= NF_INQ_VARID(ncid,'efluxt',vid)
         call ncvpt_doubl2(ncid,vid,start_npa(2),count_npa(2),efluxt(1),
     +        istatus)

      endif  ! On npa_diag .ne. "disabled"


      istatus= NF_INQ_VARID(ncid,'sigmamod',vid)  
      call ncvptc2(ncid,vid,1,8,sigmamod,8,istatus)
      call check_err(istatus)
      
      call check_err(istatus)

      if (sigmamod .eq. "enabled") then ! n=0: save values

         istatus= NF_INQ_VARID(ncid,'isigmas',vid)  
         call ncvpt_int2(ncid,vid,1,4,isigmas,istatus)

         istatus= NF_INQ_VARID(ncid,'isigsgv1',vid)  
         call ncvpt_int2(ncid,vid,1,1,isigsgv1,istatus)

         istatus= NF_INQ_VARID(ncid,'isigsgv2',vid)  
         call ncvpt_int2(ncid,vid,1,1,isigsgv2,istatus)

         istatus= NF_INQ_VARID(ncid,'mmsv',vid)  
         call ncvpt_int2(ncid,vid,1,1,mmsv,istatus)

      endif  !  On sigmamod



c.......................................................................
c     Time-Dependent data (numrec1=1)
c     (Continue for numrec1.gt.1 with additional write below.)
c.......................................................................
c     Time-Dependent data (numrec1=1)

      istatus= NF_INQ_VARID(ncid,'time',vid)
      call ncvpt_doubl2(ncid,vid,start(4),1,timet,istatus)

      istatus= NF_INQ_VARID(ncid,'bctshift',vid)
      call ncvpt_doubl2(ncid,vid,1,1,bctshift,istatus)

      istatus= NF_INQ_VARID(ncid,'bctimescal',vid)
      call ncvpt_doubl2(ncid,vid,1,1,bctimescal,istatus)
      
      !YuP[2018-09-28] added for 'lngshrtf' option,
      !for saving f() distr.func. at selected t steps only.
      if((netcdfshort.eq.'lngshrtf').and.isave.ne.0) then  !isave/nsavet set in tdchief
        !YuP[2019-06-08] Added (netcdfshort.eq.'lngshrtf') in if()
         istatus= NF_INQ_VARID(ncid,'nsave',vid) !here n=0
         call ncvpt_int2(ncid,vid,isave,1,nsave,istatus)
         istatus= NF_INQ_VARID(ncid,'tsave',vid) !here n=0
         call ncvpt_doubl2(ncid,vid,startg(5),1,timet,istatus) !here startg(5)=1
      endif

      do ll=0,lrzmax
         tr(ll)=reden(kelec,ll)
      enddo
      istatus= NF_INQ_VARID(ncid,'den_e',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r00_count,tr(0:lrzmax),
     & istatus)

      call pack21(reden,1,ntotala,0,lrza,wkpack,ntotal,lrzmax)
      istatus= NF_INQ_VARID(ncid,'density',vid)  
      call ncvpt_doubl2(ncid,vid,start(2),species_count,wkpack,istatus)

      istatus= NF_INQ_VARID(ncid,'zeff',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,zeff,istatus)

      !YuP[07-31-2014] Added:
      istatus= NF_INQ_VARID(ncid,'consn',vid) ! for k=1 only, for now
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,consn(1:lrza),
     & istatus)
      !YuP[07-31-2014] Added:
      istatus= NF_INQ_VARID(ncid,'bscurr_e_gen',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,bscurm(1:lrza,1,2),
     + istatus)
      !YuP[07-31-2014] Added:
      istatus= NF_INQ_VARID(ncid,'bscurr_e_maxw',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,bscurm(1:lrza,1,1),
     + istatus)
      !YuP[07-31-2014] Added:
      istatus= NF_INQ_VARID(ncid,'bscurr_i_gen',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,bscurm(1:lrza,2,2),
     + istatus)
      !YuP[07-31-2014] Added:
      istatus= NF_INQ_VARID(ncid,'bscurr_i_maxw',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,bscurm(1:lrza,2,1),
     + istatus)
      !YuP[2019-12-23] Added:
      istatus= NF_INQ_VARID(ncid,'currpar_starnue',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,
     & currpar_starnue_n(1:lrza),istatus)
      !YuP[2019-12-23] Added:
      istatus= NF_INQ_VARID(ncid,'currpar_starnue0',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,
     & currpar_starnue0_n(1:lrza),istatus)


      call pack21(temp,1,ntotala,0,lrza,wkpack,ntotal,lrzmax)
      istatus= NF_INQ_VARID(ncid,'temp',vid)  
      call ncvpt_doubl2(ncid,vid,start(2),species_count,wkpack,istatus)

      call pack21(energy,1,ntotala,1,lrza,wkpack,ntotal,lrzmax) 
      istatus= NF_INQ_VARID(ncid,'energy',vid) !<..>_FSA
      call ncvpt_doubl2(ncid,vid,start(2),species_count,wkpack,istatus)

      if (ngen.eq.1) then
      call pack21(wpar,1,ngena,1,lrza,wkpack,ngen,lrz)
      istatus= NF_INQ_VARID(ncid,'wpar',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,wkpack,istatus)

      call pack21(wperp,1,ngena,1,lrza,wkpack,ngen,lrz)
      istatus= NF_INQ_VARID(ncid,'wperp',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,wkpack,istatus)

      else  !  ngen.ge.2

      do ll=1,lrz
         do k=1,ngen
            tem1(ll+(k-1)*lrz)=wpar(k,lrindx(ll))
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'wpar',vid)  
      call ncvpt_doubl2(ncid,vid,start_rk,count_rk,tem1,istatus)

      do ll=1,lrz
         do k=1,ngen
            tem1(ll+(k-1)*lrz)=wperp(k,lrindx(ll))
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'wperp',vid)  
      call ncvpt_doubl2(ncid,vid,start_rk,count_rk,tem1,istatus)

      endif !  on ngen

      istatus= NF_INQ_VARID(ncid,'elecfld',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r00_count,elecfld,istatus)

      istatus= NF_INQ_VARID(ncid,'edreicer',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,elecr,istatus)

      tr=zero
      do ll=1,lrz
         tr(ll)=vfluxz(lrindx(ll))
      enddo
      istatus= NF_INQ_VARID(ncid,'runaway_rate',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,tr(1:lrz),istatus)

      tr=zero
      do ll=1,lrors !YuP[2021-04] lrz --> lrors
         tr(ll)=denra(1,ll) !YuP[2018-09-24]
      enddo
      istatus= NF_INQ_VARID(ncid,'denra',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,tr(1:lrz),istatus)

      tr=zero
      do ll=1,lrors !YuP[2021-04] lrz --> lrors
         tr(ll)=curra(1,ll)/3.e9  !Scaling from statA/cm**2 ==> A/cm**2
      enddo
      istatus= NF_INQ_VARID(ncid,'curra',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,tr(1:lrz),istatus)

      tr=zero !YuP[2019-06-08]was call bcast(tr(1),zero,lrzmax)
      do ll=1,lrors ! !YuP[2021-04] (lrz) --> (lrors)
         tr(ll)=ucrit(1,ll) !YuP[2018-09-24]
      enddo
      istatus= NF_INQ_VARID(ncid,'ucrit',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,tr(1:lrz),istatus)

      istatus= NF_INQ_VARID(ncid,'knockon',vid)  
      call ncvptc2(ncid,vid,1,8,knockon,8,istatus)

      if (knockon.ne."disabled") then
         
         tr=zero
         do ll=1,lrz
            tr(ll)=eoe0(1,ll) !YuP[2018-09-24]
         enddo
         istatus= NF_INQ_VARID(ncid,'eoe0',vid)  
         call ncvpt_doubl2(ncid,vid,start(3:4),r_count(1:2),tr(1:lrz),
     &     istatus)
         
         istatus= NF_INQ_VARID(ncid,'srckotot',vid)  
         call ncvpt_doubl2(ncid,vid,start(3:4),r_count(1:2),srckotot,
     &     istatus)
         
         istatus= NF_INQ_VARID(ncid,'denfl',vid)  
         call ncvpt_doubl2(ncid,vid,start(3:4),r_count(1:2),denfl,
     &     istatus)
         
      endif                     ! on knockon
      
      !if(transp.eq."enabled")then !could add this condition later.
      if(read_data.eq.'nimrod')then
         istatus= NF_INQ_VARID(ncid,'dbb2',vid) !(deltaB/B)^2 from data files
         call ncvpt_doubl2(ncid,vid,start(3:4),count(3:4),dbb2,
     &     istatus)
      endif
      !endif !transp
      
c
c...new Freya stuff
c
      if (frmodp.eq."enabled") then
           istatus= NF_INQ_VARID(ncid,'hibrz',vid)
           call ncvpt_doubl2(ncid,vid,start_hibr,count_hibr,
     +                 hibrzp,istatus)
     
           istatus= NF_INQ_VARID(ncid,'sorpw_nbi',vid)
           call ncvpt_doubl2(ncid,vid,start_sorpw,count_sorpw,
     +                 sorpw_nbi,istatus)
     
           istatus= NF_INQ_VARID(ncid,'sorpw_nbii',vid)
           call ncvpt_doubl2(ncid,vid,start_sorpw,count_sorpw,
     +                 sorpw_nbii,istatus)

      endif
c


      if (rdcmod.ne."disabled") then
         if ((mrfn+3)*lrz .gt. iyjx2) stop
     +     'netcdfrw2:  Need  (mrfn+3)*lrz<(iy+2)*(jx+2)'
         call bcast(tem1,zero,(mrfn+3)*lrz)
         do ll=1,lrz
            do kk=1,mrfn
               kkk=kk
               tem1(ll+(kk-1)*lrz)=sorpw_rf(kk,lrindx(ll))
            enddo
            tem1(ll+kkk*lrz)=sorpw_rfi(1,lrindx(ll))
            tem1(ll+(kkk+1)*lrz)=sorpwt(lrindx(ll))
            tem1(ll+(kkk+2)*lrz)=sorpwti(lrindx(ll))
         enddo
         istatus= NF_INQ_VARID(ncid,'rfpwr',vid) 
      call ncvpt_doubl2(ncid,vid,start_rfpwr,count_rfpwr,tem1,istatus)

      endif  !On rdcmod


      if (urfmod.ne."disabled") then
      if ((mrfn+3)*lrz .gt. iyjx2) stop
     +     'netcdfrw2:  Need  (mrfn+3)*lrz<(iy+2)*(jx+2)'
      istatus= NF_INQ_VARID(ncid,'mrfn',vid)  !-YuP[2017]added
      call ncvpt_int2(ncid,vid,1,1,mrfn,istatus)
      !number of rf modes (sum over all wave types and all nharms)
      call bcast(tem1,zero,(mrfn+3)*lrz)
cBH120223:  Removed erroneous divide by dvol(ll) from powrf/powrft
      do ll=1,lrz
         do kk=1,mrfn
            kkk=kk
            tem1(ll+(kk-1)*lrz)=powrf(lrindx(ll),kk)
         enddo
         tem1(ll+kkk*lrz)=powrft(lrindx(ll))
         tem1(ll+(kkk+1)*lrz)=sorpwt(lrindx(ll))
         tem1(ll+(kkk+2)*lrz)=sorpwti(lrindx(ll))
      enddo
      istatus= NF_INQ_VARID(ncid,'rfpwr',vid)  
      call ncvpt_doubl2(ncid,vid,start_rfpwr,count_rfpwr,tem1,istatus)

      call bcast(tem1,zero,mrfn+1)
      do kk=1,mrfn
         tem1(kk)=powurf(kk)
      enddo
      tem1(mrfn+1)=powurf(0)
      istatus= NF_INQ_VARID(ncid,'powurf',vid)  
      call ncvpt_doubl2(ncid,vid,start_powurf,count_powurf,tem1,istatus)

cBH120202:  Should only store mrfn*lrz, at most: NEEDS ADJUSTMENT
       
      call bcast(tem1,zero,nmodsa*lrz)
      do ll=1,lrz
         do kk=1,nmodsa
            tem1(ll+(kk-1)*lrz)=powrfl(lrindx(ll),kk)
         enddo
      enddo

      istatus= NF_INQ_VARID(ncid,'powrfl',vid)  
      call ncvpt_doubl2(ncid,vid,start_powrf,count_powrf,tem1,istatus)

      istatus= NF_INQ_VARID(ncid,'powurfl',vid)  
      call ncvpt_doubl2(ncid,vid,start_powrf(2),count_powrf(2),
     +           powurfl(1),istatus)

       
cBH120202:  Should only store mrfn*lrz, at most: NEEDS ADJUSTMENT
      call bcast(tem1,zero,nmodsa*lrz)
      do ll=1,lrz
         do kk=1,nmodsa
            tem1(ll+(kk-1)*lrz)=powrf(lrindx(ll),kk)
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'powrf',vid)
      call ncvpt_doubl2(ncid,vid,start_powrf,count_powrf,tem1,istatus)

cBH120202:  Should only store mrfn*lrz, at most: NEEDS ADJUSTMENT
      call bcast(tem1,zero,nmodsa*lrz)
      do ll=1,lrz
         do kk=1,nmodsa
            tem1(ll+(kk-1)*lrz)=powrfc(lrindx(ll),kk)
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'powrfc',vid)
      call ncvpt_doubl2(ncid,vid,start_powrf,count_powrf,tem1,istatus)

      istatus= NF_INQ_VARID(ncid,'powurfc',vid)
      call ncvpt_doubl2(ncid,vid,start_powrf(2),count_powrf(2),
     +           powurfc(1),istatus)

      call bcast(tem1,zero,lrz)
      do ll=1,lrz
            tem1(ll)=powrft(lrindx(ll))
      enddo
      istatus= NF_INQ_VARID(ncid,'powrft',vid)
      call ncvpt_doubl2(ncid,vid,start_powrft,count_powrft,tem1,istatus)

      istatus= NF_INQ_VARID(ncid,'nrfspecies',vid)  !YuP[11-2017]
      call ncvpt_int2(ncid,vid,1,count_powrf(2),nrfspecies,istatus)
      endif  !On urfmod

      istatus= NF_INQ_VARID(ncid,'curtor',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,curtor,istatus)

      istatus= NF_INQ_VARID(ncid,'ccurtor',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,ccurtor(1),istatus)

      istatus= NF_INQ_VARID(ncid,'curpol',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,curpol,istatus)

      istatus= NF_INQ_VARID(ncid,'ccurpol',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,ccurpol(1),istatus)

      if (ngen.eq.1) then
      
         k=1
         do ll=1,lrzmax
            tr(ll)=curr(k,ll)/3.e9  !Scaling statA/cm**2 ==> A/cm**2
         enddo
         istatus= NF_INQ_VARID(ncid,'curr',vid)
         call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)
         !YuP[07-31-2014] Added:
         do ll=1,lrzmax
            tr(ll)=energym(k,ll)
         enddo
         istatus= NF_INQ_VARID(ncid,'energym',vid)
         call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)

      else  !  ngen.ge.2
         do k=1,ngen
         do ll=1,lrzmax
               tem1(ll+(k-1)*lrzmax)=curr(k,ll)/3.e9
            ! 1/3e9 is for statA/cm**2 ==> A/cm**2
         enddo
         enddo
         istatus= NF_INQ_VARID(ncid,'curr',vid)
         call ncvpt_doubl2(ncid,vid,start_r0k,count_r0k,tem1,istatus)
         !YuP[07-2017] added
         do k=1,ngen
         do ll=1,lrzmax
            tem1(ll+(k-1)*lrzmax)=energym(k,ll) !YuP[07-31-2014] Added
         enddo
         enddo
         istatus= NF_INQ_VARID(ncid,'energym',vid)
         call ncvpt_doubl2(ncid,vid,start_r0k,count_r0k,tem1,istatus)
	 
      endif ! ngen>1


      istatus= NF_INQ_VARID(ncid,'efflag',vid)  
      call ncvptc2(ncid,vid,1,8,efflag,8,istatus)

      kk=1
      if (kelecg.ne.0) kk=kelecg
      do ll=1,lrzmax
         tr(ll)=currm(kk,ll)/3.e9  !Scaling statA/cm**2 ==> A/cm**2
      enddo
      if (kelecg.ne.0) then ! e as gen.species
         istatus= NF_INQ_VARID(ncid,'currm_e',vid)
      else ! ion as general species
         istatus= NF_INQ_VARID(ncid,'currm_i',vid)
      endif
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)

      do ll=1,lrzmax
         tr(ll)=restp(nch(ll),ll)
      enddo
      istatus= NF_INQ_VARID(ncid,'restp',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)

      do ll=1,lrzmax
         tr(ll)=restnp(nch(ll),ll)
      enddo
      istatus= NF_INQ_VARID(ncid,'restnp',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)

      do ll=1,lrzmax
         tr(ll)=sptzrp(nch(ll),ll)
      enddo
      istatus= NF_INQ_VARID(ncid,'sptzrp',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)

      istatus= NF_INQ_VARID(ncid,'rovsc',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,rovsc,istatus)

      istatus= NF_INQ_VARID(ncid,'rovsc_hi',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,rovsc,istatus)

      istatus= NF_INQ_VARID(ncid,'zreskim',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,zreskim,istatus)

      istatus= NF_INQ_VARID(ncid,'taueeh',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,taueeh,istatus)

      istatus= NF_INQ_VARID(ncid,'nuestar',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,starnue,istatus)

      !YuP[2019-09] added time-dep. data. HERE n=0 [[
      !Note:       start(2:4)= {1,         1,   numrec1} HERE: n=0
      !Note: impur_count(1:3)= {nstates+1, lrz, 1      }
      if(nstates.gt.0)then ! If no impurities, nstates=0. 
        istatus= NF_INQ_VARID(ncid,'dens_imp_allstates',vid)
        call ncvpt_doubl2(ncid,vid,start(3:4),impur_count(2:3),
     &     dens_imp_allstates, istatus)
        istatus= NF_INQ_VARID(ncid,'dens_imp',vid)
        call ncvpt_doubl2(ncid,vid,start(2:4),impur_count(1:3),
     &     dens_imp, istatus)
        istatus= NF_INQ_VARID(ncid,'pellet_rho',vid)
        call ncvpt_doubl2(ncid,vid,start(4:4),impur_count(3:3),
     &     pellet_rho, istatus)
        istatus= NF_INQ_VARID(ncid,'Gablation',vid)
        call ncvpt_doubl2(ncid,vid,start(4:4),impur_count(3:3),
     &     Gablation, istatus)
        istatus= NF_INQ_VARID(ncid,'pellet_Mrem',vid)
        call ncvpt_doubl2(ncid,vid,start(4:4),impur_count(3:3),
     &     pellet_Mrem, istatus)
      endif
      !YuP[2019-09] added time-dep. data ]]

cBH110320c     Following only set up for ngen=1
cBH110320      if (ngen.ge.2)
cBH110320     1     WRITE(*,*)'netcdfrw2: Tot pwrs only set up for ngen=1'
      if (13*lrz*ngen .gt. iyjx2) stop 
     &        'netcdfrw2: Need 13*lrz*ngen<(iy+2)*(jx+2)'
      tem1=0.d0 !YuP[2023] moved zero-ing of tem1 here.
      do k=1,ngen
         kkk=(k-1)*13*lrz
         !call bcast(tem1,zero,13*lrz*ngen) !YuP[2023] Why here in k-loop?
         do ll=1,lrz
            kk=1
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,-1,ll)
            kk=2
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,0,ll)
            kk=3
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,2,ll)
            kk=4
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,1,ll)
            kk=5
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,3,ll)
            kk=6
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,5,ll)
            kk=7
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,6,ll)
            kk=8
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,7,ll)
            kk=9
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,8,ll)
            kk=10
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,11,ll)
            kk=11
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,10,ll)
            kk=12
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,12,ll)
            kk=13
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,4,ll)
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'powers',vid)  
      call ncvpt_doubl2(ncid,vid,start_powers,count_powers,tem1,istatus)
     
c     Following only set up for ngen=1
cBH110320      if (ngen.ge.2)
cBH110320     1     WRITE(*,*)'netcdfrw2: Tot pwrs only set up for ngen=1'
      if (13*ngen .gt.iyjx2) stop
     &     'netcdfrw2: Need 13*ngen<(iy+2)*(jx+2)'
      tem1=0.d0 !YuP[2022-12-16] moved zero-ing of tem1 here.
	WRITE(*,*) '8.1.1', maxval(entrintr),minval(entrintr)
      do k=1,ngen
            kkk=(k-1)*13
            !call bcast(tem1,zero,13*ngen) !YuP[2022-12-16] Why here in k-loop?
            kk=1
            tem1(kk+kkk)=entrintr(k,-1)
            kk=2
            tem1(kk+kkk)=entrintr(k,0)
            kk=3
            tem1(kk+kkk)=entrintr(k,2)
            kk=4
            tem1(kk+kkk)=entrintr(k,1)
            kk=5
            tem1(kk+kkk)=entrintr(k,3)
            kk=6
            tem1(kk+kkk)=entrintr(k,5)
            kk=7
            tem1(kk+kkk)=entrintr(k,6)
            kk=8
            tem1(kk+kkk)=entrintr(k,7)
            kk=9
            tem1(kk+kkk)=entrintr(k,8)
            kk=10
            tem1(kk+kkk)=entrintr(k,11)
            kk=11
            tem1(kk+kkk)=entrintr(k,10)
            kk=12
            tem1(kk+kkk)=entrintr(k,12)
            kk=13
            tem1(kk+kkk)=entrintr(k,4)
      enddo
      istatus= NF_INQ_VARID(ncid,'powers_int',vid)  
      call ncvpt_doubl2(ncid,vid,start_powers(2),count_powers(2),
     ~                  tem1,istatus)


      if (sigmamod .eq. "enabled") then ! here n=0: save sigftt
         istatus= NF_INQ_VARID(ncid,'sigftt',vid)  
         do lsig=1,4
            tem2(lsig)=sigftt(nch(1),lsig)
         enddo
         call ncvpt_doubl2(ncid,vid,start_fus(2),count_fus(2),
     ~                     tem2,istatus)
      endif
      
      if (netcdfshort.eq.'long_jp') then ! Here n=0

         if (ngen.eq.1) then
         
            k=1
            do ll=1,lrz
               do j=1,jx
                  i=j+(lrindx(ll)-1)*jx
                  tem1(i)=currv(j,k,lrindx(ll))/3.e9
               enddo
            enddo
            istatus= NF_INQ_VARID(ncid,'currv',vid)  
            call ncvpt_doubl2(ncid,vid,start(2:3),count(2:3),tem1,
     &       istatus)
            
            do ll=1,lrz
               do j=1,jx
                  i=j+(lrindx(ll)-1)*jx
                  tem1(i)=pwrrf(j,k,lrindx(ll))
               enddo
            enddo
            istatus= NF_INQ_VARID(ncid,'pwrrf',vid) 
            call ncvpt_doubl2(ncid,vid,start(2:3),count(2:3),tem1,
     &       istatus)

            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
            ! tam4 contains pitch angle averaged equatorial distribution function.
            !For possible future use:
            ! tam5 to contain pitch angle integrated distribution <f> such
            ! that integral{<f> d(gamma)} = fsa density.
            ! tam5=2*pi*x*cnorm2*gamma*int_i{sinn*dy*vptb*f_code}/zmaxpsi.
            do ll=1,lrors !YuP[2022-02-11] was lrz
              tam4=0.d0
              tam5=0.d0
              do j=1,jx
                do i=1,iymax
                  tam4(j)=tam4(j)+f(i,j,k,ll)*cynt2(i,ll) !integral over i
                  tam5(j)=tam5(j)+f(i,j,k,ll)*cynt2(i,ll)
     &                                       *vptb(i,lrindx(ll))
                  !YuP[2021-02-26] Corrected vptb(i,ll)->vptb(i,lrindx(ll))
                  !This is important for CQLP (when lr_=1, but l_=1:ls)
                  !Probably should not be using vptb in CQLP version at all.
                enddo
                tam4(j)=tam4(j)/twoint(ll) != INTEGR{f*sin(theta0)dtheta0} /2
                tam5(j)=tam5(j)/zmaxpsi(lrindx(ll))*x(j)*gamma(j)*cnorm2
              enddo ! j
              do j=1,jx
                 i=j+(ll-1)*jx
                 tem1(i)=tam4(j) ! Save favr_thet0(j), each ll
              enddo
            enddo ! ll
            istatus= NF_INQ_VARID(ncid,'favr_thet0',vid) 
            call ncvpt_doubl2(ncid,vid,start(2:3),count(2:3),tem1,
     &      istatus) 
            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2

         else  !  ngen.ge.2

         do k=1,ngen
            do ll=1,lrz
               do j=1,jx
                  i=j+(lrindx(ll)-1)*jx
                  tem1(i)=currv(j,k,lrindx(ll))/3.e9
               enddo
            enddo
            start(4)=k ! and start(3) is not changed, it is 1
            !Note: count(3)=lrz
            !Note: count(4)=1 
            istatus= NF_INQ_VARID(ncid,'currv',vid)  
            call ncvpt_doubl2(ncid,vid,start(2:4),count(2:4),tem1,
     &                        istatus)
            
            do ll=1,lrz
               do j=1,jx
                  i=j+(lrindx(ll)-1)*jx
                  tem1(i)=pwrrf(j,k,lrindx(ll))
               enddo
            enddo
            start(4)=k ! and start(3) is not changed, it is 1
            !Note: count(3)=lrz
            !Note: count(4)=1 
            istatus= NF_INQ_VARID(ncid,'pwrrf',vid) 
            call ncvpt_doubl2(ncid,vid,start(2:4),count(2:4),tem1,
     &                        istatus)
     
            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
            ! tam4 contains pitch angle averaged equatorial distribution function.
            !For possible future use:
            ! tam5 to contain pitch angle integrated distribution <f> such
            ! that integral{<f> d(gamma)} = fsa density.
            ! tam5=2*pi*x*cnorm2*gamma*int_i{sinn*dy*vptb*f_code}/zmaxpsi.
            do ll=1,lrors !YuP[2022-02-11] was lrz
              tam4=0.d0
              tam5=0.d0
              do j=1,jx
                do i=1,iymax
                  tam4(j)=tam4(j)+f(i,j,k,ll)*cynt2(i,ll) !integral over i
                  tam5(j)=tam5(j)+f(i,j,k,ll)*cynt2(i,ll)
     &                                       *vptb(i,lrindx(ll))
                  !YuP[2021-02-26] Corrected vptb(i,ll)->vptb(i,lrindx(ll))
                  !This is important for CQLP (when lr_=1, but l_=1:ls)
                  !Probably should not be using vptb in CQLP version at all.

                enddo
                tam4(j)=tam4(j)/twoint(ll) != INTEGR{f*sin(theta0)dtheta0} /2
                tam5(j)=tam5(j)/zmaxpsi(lrindx(ll))*x(j)*gamma(j)*cnorm2
              enddo ! j
              do j=1,jx
                 i=j+(ll-1)*jx
                 tem1(i)=tam4(j) ! Save favr_thet0(j), each ll
              enddo
            enddo ! ll
            start(4)=k ! and start(3) is not changed, it is 1
            !Note: count(3)=lrz
            !Note: count(4)=1 
            istatus= NF_INQ_VARID(ncid,'favr_thet0',vid) 
            call ncvpt_doubl2(ncid,vid,start(2:4),count(2:4),tem1,
     &      istatus) 
            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
     
         enddo  !  on k=1,ngen
         
         endif  !  on ngen
         
         !YuP[2022-03-19] Added fl()==F_par reduced distr.func.
         !One species, for now. 
         istatus= NF_INQ_VARID(ncid,'fl',vid) !here: 'long_jp', n=0
         call ncvpt_doubl2(ncid,vid,start(1:2),count_fl(1:2),
     &        fl(1:jfl,1:lrors),istatus) !count_fl={jfl; lrz or lz; 1}
         
         start(4)=1 ! restored
         
      endif  ! on netcdfshort.eq.'long_jp'  ! Here n=0


cyup      if ( netcdfshort.eq.'longer_f' ) then  !endif at line 3090
      if ( (netcdfshort.eq.'longer_f').or.
     +     (netcdfshort.eq.'lngshrtf')    ) then
         !---> Here n=0. Should we record f() even if isave=0?
         !---> If-yes, do not use ".and.(isave.ne.0)" as in n>0 case.
         !write(*,*)'netcdfrw2: netcdfshort,n,isave=',netcdfshort,n,isave
         istatus= NF_INQ_VARID(ncid,'f',vid)  
         if (ngen.eq.1) then
         
            if(netcdfshort.eq.'lngshrtf') then
              startf(4)=numrecsave !YuP[2018-09-28]netcdfshort.eq.'lngshrtf'
            else ! 'longer_f'
              startf(4)=numrec1 ! usual (saved at every t step) Here numrec1=1
            endif
            do ll=1,lrors !YuP[2022-02-11] was lrz
               do j=1,jx ! 0,jx+1 !1,jx
                  do i=1,iymax ! 0,iy+1 !1,iy
                     temp1(i,j)=f(i,j,1,ll)
                  enddo
               enddo
               call pack21(temp1(1:iymax,1:jx),1,iymax,1,jx,
     &                     wkpack(1:iymax*jx),iymax,jx)
               !YuP[2021-03-11] Changed iy-->iymax 
               !(just in case if iy is changed by iy=iy_(l_) somewhere)
               startf(3)=ll
               call ncvpt_doubl2(ncid,vid,startf(1:4),countf(1:4),
     &          wkpack,istatus)
               !So, here startf={1,   1,   ll,   numrec1(or numrecsave)}
               !         countf={iy,  jx,   1,    1}
               !         dimsf= {ydimf,xdimf,rdim, tdim(or tsavedim)}
            enddo
            startf(4)=numrec1 ! restore (=1 here, anyway)

         else  !ngen.ge.2

            if(netcdfshort.eq.'lngshrtf') then
              startg(5)=numrecsave !YuP[2018-09-28]netcdfshort.eq.'lngshrtf'
            else ! 'longer_f'
              startg(5)=numrec1 ! usual (saved at every t step) here numrec1=1
            endif
            do k=1,ngen
            do ll=1,lrors !YuP[2022-02-11] was lrz
               do j=1,jx ! 0,jx+1 !1,jx
                  do i=1,iymax ! 0,iy+1 !1,iy
                     temp1(i,j)=f(i,j,k,ll)
                  enddo
               enddo
               call pack21(temp1(1:iymax,1:jx),1,iymax,1,jx,
     &                     wkpack(1:iymax*jx),iymax,jx)
               !YuP[2021-03-11] Changed iy-->iymax 
               !(just in case if iy is changed by iy=iy_(l_) somewhere)
               startg(3)=ll
               startg(4)=k
               call ncvpt_doubl2(ncid,vid,startg,countg,wkpack,istatus)
               !So, here startg={1,   1,   ll,  k,   numrec1(or numrecsave)}
               !         countg={iy,jx, 1,  1,   1}
               !         dimsg= {ydimf,xdimf,rdim,gdim,tdim(or tsavedim)}
            enddo  !  On ll
            enddo  !  On k
            startg(3)=1  ! restore
            startg(4)=1  ! restore
            startg(5)=numrec1 ! restore (=1 here, anyway)
            
         endif  ! on ngen

            
      endif  ! on netcdfshort.eq.'longer_f'
         
         
         
         
c --- output large arrays only if nstop=0
      if (nstop.eq.0) then 

c$$$         istatus= NF_INQ_VARID(ncid,'currv',vid)  
c$$$         k=1
c$$$         do ll=1,lrz
c$$$            do j=1,jx
c$$$               i=j+(lrindx(ll)-1)*jx
c$$$               tem1(i)=currv(j,k,lrindx(ll))/3.e9
c$$$            enddo
c$$$         enddo
c$$$         call ncvpt_doubl2(ncid,vid,start(2),count(2),tem1,istatus)
c$$$
c$$$         istatus= NF_INQ_VARID(ncid,'pwrrf',vid)  
c$$$         k=1
c$$$         do ll=1,lrz
c$$$            do j=1,jx
c$$$               i=j+(lrindx(ll)-1)*jx
c$$$               tem1(i)=pwrrf(j,k,lrindx(ll))
c$$$            enddo
c$$$         enddo
c$$$         call ncvpt_doubl2(ncid,vid,start(2),count(2),tem1,istatus)
         
         if ( (netcdfshort.ne.'enabled')  .and. 
     +        (netcdfshort.ne.'longer_f') .and.
     +        (netcdfshort.ne.'lngshrtf')       ) then ! here n=0 (nstop=0)
     
            istatus= NF_INQ_VARID(ncid,'f',vid)  
cBH011221: This storage is set up for constant iy as function of radius.
cBH011221: Needs generalizing. Should we store in reg array to max(iy_)?
cBH011221: For now, simply stop.
!            do ll=1,lrz
!               if (iy_(ll).ne.iymax) 
!     +              stop 'netcdfrw2: Cant handle iy.ne.iymax' 
!                    !YuP[2021-03] Could be relaxed now?
!            enddo

            if (ngen.eq.1) then
            
               k=1 !  
               if (tavg.ne."disabled") then
                 PRINT *,'netcdfrw2 WARNING: favg is saved (NOT f)'
                 PRINT *,'netcdfrw2 WARN:  but if time(nstop)<tavg1(1)'
                 PRINT *,'netcdfrw2 WARNING: then favg = f.'
               endif
               do ll=1,lrors !YuP[2022-02-11] was lrz
                  if (tavg.eq."disabled") then
                     do j=1,jx ! 0,jx+1 !1,jx
                     do i=1,iymax ! 0,iy+1 !1,iy
                        temp1(i,j)=f(i,j,k,ll)
                        !if(gone(i,j,k,lrindx(ll)).lt.-0.1) then
                        !   temp1(i,j)=em90
                        !endif
                     enddo
                     enddo
                  else  !On tavg = enabled
                     ! Bob, here is the nstop=0 part, 
                     ! so how can we have favg?
                     do j=1,jx ! 0,jx+1 !1,jx
                     do i=1,iymax ! 0,iy+1 !1,iy
                        temp1(i,j)=favg(i,j,k,ll)
                     !if(gone(i,j,k,lrindx(ll)).lt.-0.1) temp1(i,j)=em90
                     enddo
                     enddo
                     WRITE(*,'(a,i6,2e13.4)')
     +               'nstop0.netcdfrw2/tavg=en ll,sumij(favg),sumij(f)',
     +                 ll,sum(temp1),sum(f(:,:,k,ll))
                  endif  !On tavg
c                 temp1 dimensnd 0:iyp1,0:jxp1. Pack in to (1:iy,1:jx)
                  call pack21(temp1(1:iymax,1:jx),1,iymax,1,jx,
     &                        wkpack(1:iymax*jx),iymax,jx)
                  !YuP[2021-03-11] Changed iy-->iymax 
                  !(just in case if iy is changed by iy=iy_(l_) somewhere)
                  startf(3)=ll
                  call ncvpt_doubl2(ncid,vid,startf(1:3),countf(1:3),
     +                              wkpack,istatus)
               !So, here startf={1,   1,   ll}
               !         countf={iy,  jx,  1}
               !         dimsf= {ydimf,xdimf,rdim}
               enddo  !On ll
               startf(3)=1 !restore
               
            else  !  ngen.ge.2
            
               do k=1,ngen
               if (tavg.eq."disabled") then
               do ll=1,lrors !YuP[2022-02-11] was lrz
                  do j=1,jx ! 0,jx+1 !1,jx
                     do i=1,iymax ! 0,iy+1 !1,iy
                        temp1(i,j)=f(i,j,k,ll)
                     enddo
                  enddo
                  call pack21(temp1(1:iymax,1:jx),1,iymax,1,jx,
     &                        wkpack(1:iymax*jx),iymax,jx)
                  !YuP[2021-03-11] Changed iy-->iymax 
                  !(just in case if iy is changed by iy=iy_(l_) somewhere)
                  startg(3)=ll
                  startg(4)=k
                 call ncvpt_doubl2(ncid,vid,startg(1:4),countg(1:4),
     +                             wkpack,istatus)
               !So, here startg={1,   1,   ll,  k}
               !         countg={iy,jx, 1,  1}
               !         dimsg= {ydimf,xdimf,rdim,gdim}
               enddo  ! on ll
               else  ! On tavg
               do ll=1,lrors ! was lrz
                  do j=1,jx ! 0,jx+1 !1,jx
                     do i=1,iymax ! 0,iy+1 !1,iy
                        temp1(i,j)=favg(i,j,k,ll)
                     enddo
                  enddo
                  call pack21(temp1(1:iymax,1:jx),1,iymax,1,jx,
     &                        wkpack(1:iymax*jx),iymax,jx)
                  !YuP[2021-03-11] Changed iy-->iymax 
                  !(just in case if iy is changed by iy=iy_(l_) somewhere)
                  startg(3)=ll
                  startg(4)=k
                  call ncvpt_doubl2(ncid,vid,startg(1:4),countg(1:4),
     +                              wkpack,istatus)
               !So, here startg={1,   1,   ll,  k}
               !         countg={iy,jx, 1,  1}
               !         dimsg= {ydimf,xdimf,rdim,gdim}
               enddo  ! on ll
               endif  !On tavg
               enddo  ! on k
               startg(3)=1  ! restore
               startg(4)=1  ! restore
               
            endif  !  on ngen
               
         endif  !  on netcdfshort
      endif  ! on nstop.eq.0
      
c --- endif ((kopt.eq.0) .and. (n.eq.0)) ---
      endif







c$$$C-----------------------------------------------------------------------
c$$$c
c$$$cl    2. Restart from previous run
c            (THIS COMMENTED OUT SECTION IS PART OF FPET
C             ROUTINES USED AS A TEMPLATE FOR THIS CQL3D
C             NETCDF SUBROUTINE SET.  BobH, 2000)
c$$$c        The following data is NOT read in:
c$$$c          vnorm,x,y,z,dvol,flux,press,hflux,pdens,flux_a,flux_b,
c$$$c          hflux_a,hflux_b,pdens0
c$$$c
c$$$
c$$$c --- begin if ---
c$$$      if ((kopt.ne.0) .and. (n.eq.0)) then 
c$$$
c$$$c.......................................................................
c$$$cl    2.1 Open previous netCDF file
c$$$
c$$$      ncid = ncopn(filename,NCWRITE,istatus)
c$$$
c$$$c.......................................................................
c$$$cl    2.2 read in dimension IDs and sizes
c$$$
c$$$      xdim = ncdid(ncid,'xdim',istatus)
c$$$      ydim = ncdid(ncid,'ydim',istatus)
c$$$      rdim = ncdid(ncid,'zdim',istatus)
c$$$      tdim = ncdid(ncid,'taudim',istatus)
c$$$
c$$$c --- inquire about dimension sizes ---
c$$$c     ncdinq(netCDF_id, dimension_id_from_ncdid, returned_dim_name,
c$$$c     returned_dim_size)
c$$$c     Note: for unlimited dimension, returned_dim_size=current maximum
c$$$c     which is the same as the maximum record number
c$$$
c$$$      call ncdinq(ncid,ydim,name,iyp,istatus)
c$$$      call ncdinq(ncid,xdim,name,jxp,istatus)
c$$$      call ncdinq(ncid,rdim,name,kzp,istatus)
c$$$
c$$$c --- stop if dimension sizes don't match ---
c$$$      if ((iyp.ne.iy) .or. (jxp.ne.jx) .or. (kzp.ne.kz)) then
c$$$         write(6,*) "set the following parameter values in RUN_FPET"
c$$$         write(6,*) "  iy = ",iyp,"  jx = ",jxp, "  kz = ",kzp
c$$$         stop "non-matching dimensions in NCDFWRITE"
c$$$      endif
c$$$
c$$$c --- set the time-step counter ==> numrec1
c$$$      call ncdinq(ncid,tdim,name,numrec1,istatus)
c$$$      start(4)=numrec1
c$$$
c$$$c.......................................................................
c$$$cl    2.3 Read datas
c$$$c
c$$$c     Here we read in only what's needed to re-start
c$$$c     the run from the previous time-step.
c$$$c
c$$$c     ncvgt(netCDF_id, variable_id_from_ncvid, vector_of_starting_index,
c$$$c     vector_of_lengths, returned_variable, integer_info)
c$$$c
c$$$      vid = ncvid(ncid,'tau',istatus)
c$$$      call ncvgt(ncid,vid,start(4),1,tau,istatus)
c$$$
c$$$      vid = ncvid(ncid,'dtau',istatus)
c$$$      call ncvgt(ncid,vid,1,1,dtau,istatus)
c$$$
c$$$      vid = ncvid(ncid,'elecfld',istatus)
c$$$      call ncvgt(ncid,vid,start(3),count(3),zarray1d(1),istatus)
c$$$      do k=start(3),start(3)+count(3)-1
c$$$        elecfld(k)=zarray1d(k)
c$$$      enddo
c$$$      .   .   .   .  .   .   .    .    .     .
c$$$
c$$$c --- endif ((kopt.ne.0) .and. (n.eq.0)) ---
c$$$      endif

C-----------------------------------------------------------------------









c
cl    3. Periodic save at each time-step (numrec1.gt.1)
c

c --- begin if ---
      if (n.gt.0) then  !endif at line 3376

c.......................................................................
cl    3.1 set the time-step counter ==> numrec1

      ! This is for saving data at EACH t step:
      numrec1=numrec1+1
      start(4)=numrec1
      start1(4)=numrec1
      startg(5)=numrec1
      start_powurf(2)=numrec1
      start_rfpwr(3)=numrec1
      start_powrf(3)=numrec1
      start_powrft(2)=numrec1
      start_powers(4)=numrec1
      start_fus(3)=numrec1
      start_xr(3)=numrec1
      start_npa(3)=numrec1
      start_rk(3)=numrec1
      start_r0k(3)=numrec1
      
      if((netcdfshort.eq.'lngshrtf').and.(isave.ne.0))then
      !YuP[2018-09-28] Saving distr.func. at selected t steps only.
      !Increment index only for steps specified by nsave() array:
      numrecsave=numrecsave+1 ! for (netcdfshort.eq.'lngshrtf')
      startgsave(4)=numrecsave ! for (netcdfshort.eq.'lngshrtf')
      startg(5)=numrecsave ! for (netcdfshort.eq.'lngshrtf')
      !YuP[2018-09-28] added for 'lngshrtf' option,
      !for saving f() distr.func. at selected t steps only.
      istatus= NF_INQ_VARID(ncid,'tsave',vid) ! here n>0
      call ncvpt_doubl2(ncid,vid,startgsave(4),1,timet,istatus) !can use startg(5) here
      !Note: for netcdfshort.ne.'lngshrtf' startgsave(4) remains =1
      !and 'tsave' was recorded for n=0 only
      endif

c.......................................................................
cl    3.2 Variables saved at each time-step (numrec1.gt.1)

      istatus= NF_INQ_VARID(ncid,'time',vid)  
      call ncvpt_doubl2(ncid,vid,start(4),1,timet,istatus)
      

      do ll=0,lrzmax
         tr(ll)=reden(kelec,ll)
      enddo
      istatus= NF_INQ_VARID(ncid,'den_e',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r00_count,tr(0:lrzmax),
     & istatus)

      call pack21(reden,1,ntotala,0,lrza,wkpack,ntotal,lrzmax)
      istatus= NF_INQ_VARID(ncid,'density',vid)  
      call ncvpt_doubl2(ncid,vid,start(2),species_count,wkpack,istatus)

      istatus= NF_INQ_VARID(ncid,'zeff',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,zeff,istatus)

      !YuP[07-31-2014] Added:
      istatus= NF_INQ_VARID(ncid,'consn',vid) ! for k=1, for now
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,consn(1:lrza),
     & istatus)
      !YuP[07-31-2014] Added:
      istatus= NF_INQ_VARID(ncid,'bscurr_e_gen',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,bscurm(1:lrza,1,2),
     + istatus)
      !YuP[07-31-2014] Added:
      istatus= NF_INQ_VARID(ncid,'bscurr_e_maxw',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,bscurm(1:lrza,1,1),
     + istatus)
      !YuP[07-31-2014] Added:
      istatus= NF_INQ_VARID(ncid,'bscurr_i_gen',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,bscurm(1:lrza,2,2),
     + istatus)
      !YuP[07-31-2014] Added:
      istatus= NF_INQ_VARID(ncid,'bscurr_i_maxw',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,bscurm(1:lrza,2,1),
     + istatus)
      !YuP[2019-12-23] Added:
      istatus= NF_INQ_VARID(ncid,'currpar_starnue',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,
     & currpar_starnue_n(1:lrza),istatus)
      !YuP[2019-12-23] Added:
      istatus= NF_INQ_VARID(ncid,'currpar_starnue0',vid)
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,
     & currpar_starnue0_n(1:lrza),istatus)


      istatus= NF_INQ_VARID(ncid,'temp',vid)  
      call pack21(temp,1,ntotala,0,lrza,wkpack,ntotal,lrzmax)
      call ncvpt_doubl2(ncid,vid,start(2),species_count,wkpack,istatus)

      istatus= NF_INQ_VARID(ncid,'energy',vid)
      call pack21(energy,1,ntotala,1,lrza,wkpack,ntotal,lrzmax)
      call ncvpt_doubl2(ncid,vid,start(2),species_count,wkpack,istatus)


      if (ngen.eq.1) then
      call pack21(wpar,1,ngena,1,lrza,wkpack,ngen,lrz)
      istatus= NF_INQ_VARID(ncid,'wpar',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,wkpack,istatus)

      call pack21(wperp,1,ngena,1,lrza,wkpack,ngen,lrz)
      istatus= NF_INQ_VARID(ncid,'wperp',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,wkpack,istatus)

      else  !  ngen.ge.2

      do ll=1,lrz
         do k=1,ngen
            tem1(ll+(k-1)*lrz)=wpar(k,lrindx(ll))
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'wpar',vid)  
      call ncvpt_doubl2(ncid,vid,start_rk,count_rk,tem1,istatus)

      do ll=1,lrz
         do k=1,ngen
            tem1(ll+(k-1)*lrz)=wperp(k,lrindx(ll))
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'wperp',vid)  
      call ncvpt_doubl2(ncid,vid,start_rk,count_rk,tem1,istatus)

      endif ! ngen

      istatus= NF_INQ_VARID(ncid,'elecfld',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r00_count,elecfld,istatus)

      istatus= NF_INQ_VARID(ncid,'edreicer',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,elecr,istatus)

      tr=zero
      do ll=1,lrz
         tr(ll)=vfluxz(lrindx(ll))
      enddo
      istatus= NF_INQ_VARID(ncid,'runaway_rate',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,tr(1:lrz),istatus)

      tr=zero
      do ll=1,lrz
         tr(ll)=denra(1,ll) !YuP[2018-09-24]
      enddo
      istatus= NF_INQ_VARID(ncid,'denra',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,tr(1:lrz),istatus)

      tr=zero
      do ll=1,lrz
         tr(ll)=curra(1,ll)/3.e9
      enddo
      istatus= NF_INQ_VARID(ncid,'curra',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,tr(1:lrz),istatus)

      tr=zero
      do ll=1,lrors !YuP[2021-04] lrz --> lrors
         tr(ll)=ucrit(1,ll) !YuP[2018-09-24]
      enddo
      istatus= NF_INQ_VARID(ncid,'ucrit',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r_count,tr(1:lrz),istatus)

      if (knockon.ne."disabled") then
         
         tr=zero
         do ll=1,lrz
            tr(ll)=eoe0(1,ll) !YuP[2018-09-24]
         enddo
         istatus= NF_INQ_VARID(ncid,'eoe0',vid)  
         call ncvpt_doubl2(ncid,vid,start(3:4),r_count,tr(1:lrz),
     &    istatus)
         
         istatus= NF_INQ_VARID(ncid,'srckotot',vid)  
         call ncvpt_doubl2(ncid,vid,start(3:4),r_count,srckotot,istatus)
         
         istatus= NF_INQ_VARID(ncid,'denfl',vid)  
         call ncvpt_doubl2(ncid,vid,start(3:4),r_count,denfl,istatus)
         
      endif                     ! on knockon

      !if(transp.eq."enabled")then !could add this condition later.
      if(read_data.eq.'nimrod')then
         istatus= NF_INQ_VARID(ncid,'dbb2',vid) !(deltaB/B)^2 from data files
         call ncvpt_doubl2(ncid,vid,start(3:4),count(3:4),dbb2,
     &     istatus)
      endif
      !endif !transp

c
c...new Freya stuff
c
      if (frmodp.eq."enabled") then
         istatus= NF_INQ_VARID(ncid,'hibrz',vid)
         call ncvpt_doubl2(ncid,vid,start_hibr,count_hibr,
     +                 hibrzp,istatus)
     
         istatus= NF_INQ_VARID(ncid,'sorpw_nbi',vid)
         call ncvpt_doubl2(ncid,vid,start_sorpw,count_sorpw,
     +                 sorpw_nbi,istatus)
     
         istatus= NF_INQ_VARID(ncid,'sorpw_nbii',vid)
         call ncvpt_doubl2(ncid,vid,start_sorpw,count_sorpw,
     +                     sorpw_nbii,istatus)
      endif
c

      if (rdcmod.ne."disabled") then

         if ((mrfn+3)*lrz .gt. iyjx2) stop
     +        'netcdfrw2:  Need  (mrfn+3)*lrz<(iy+2)*(jx+2)'
         call bcast(tem1,zero,(mrfn+3)*lrz)
cBH120223:  Removed erroneous divide by dvol(ll) from powrf/powrft
         do ll=1,lrz
            do kk=1,mrfn
               kkk=kk
               tem1(ll+(kk-1)*lrz)=sorpw_rf(kk,lrindx(ll))
            enddo
            tem1(ll+kkk*lrz)=sorpw_rfi(1,lrindx(ll)) !For 1 gen species 
            tem1(ll+(kkk+1)*lrz)=sorpwt(lrindx(ll))
            tem1(ll+(kkk+2)*lrz)=sorpwti(lrindx(ll))
         enddo
c     -YuP:      vid=ncvid(ncid,'rfpwr',istatus)
        istatus= NF_INQ_VARID(ncid,'rfpwr',vid) 
        call ncvpt_doubl2(ncid,vid,start_rfpwr,count_rfpwr,tem1,istatus)
         
      endif  !On rdcmod


      if (urfmod.ne."disabled") then

      if ((mrfn+3)*lrz .gt. iyjx2) stop
     +     'netcdfrw2:  Need  (mrfn+3)*lrz<(iy+2)*(jx+2)'
      call bcast(tem1,zero,(mrfn+3)*lrz)
cBH120223:  Removed erroneous divide by dvol(ll) from powrf/powrft
      do ll=1,lrz
         do kk=1,mrfn
            kkk=kk
            tem1(ll+(kk-1)*lrz)=powrf(lrindx(ll),kk)
         enddo
         tem1(ll+kkk*lrz)=powrft(lrindx(ll))
         tem1(ll+(kkk+1)*lrz)=sorpwt(lrindx(ll))
         tem1(ll+(kkk+2)*lrz)=sorpwti(lrindx(ll))
      enddo
      istatus= NF_INQ_VARID(ncid,'rfpwr',vid)  
      call ncvpt_doubl2(ncid,vid,start_rfpwr,count_rfpwr,tem1,istatus)

      call bcast(tem1,zero,mrfn+1)
      do kk=1,mrfn
         tem1(kk)=powurf(kk)
      enddo
      tem1(mrfn+1)=powurf(0)
      istatus= NF_INQ_VARID(ncid,'powurf',vid)  
      call ncvpt_doubl2(ncid,vid,start_powurf,count_powurf,tem1,istatus)

cBH120202:  Should only store mrfn*lrz, at most: NEEDS ADJUSTMENT
      call bcast(tem1,zero,nmodsa*lrz)
      do ll=1,lrz
         do kk=1,nmodsa
            tem1(ll+(kk-1)*lrz)=powrfl(lrindx(ll),kk)
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'powrfl',vid)  
      call ncvpt_doubl2(ncid,vid,start_powrf,count_powrf,tem1,istatus)

      istatus= NF_INQ_VARID(ncid,'powurfl',vid)  
      call ncvpt_doubl2(ncid,vid,start_powrf(2),count_powrf(2),
     +           powurfl(1),istatus)
       
cBH120202:  Should only store mrfn*lrz, at most: NEEDS ADJUSTMENT
      call bcast(tem1,zero,nmodsa*lrz)
      do ll=1,lrz
         do kk=1,nmodsa
            tem1(ll+(kk-1)*lrz)=powrf(lrindx(ll),kk)
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'powrf',vid)
      call ncvpt_doubl2(ncid,vid,start_powrf,count_powrf,tem1,istatus)


cBH120202:  Should only store mrfn*lrz, at most: NEEDS ADJUSTMENT
      call bcast(tem1,zero,nmodsa*lrz)
      do ll=1,lrz
         do kk=1,nmodsa
            tem1(ll+(kk-1)*lrz)=powrfc(lrindx(ll),kk)
         enddo
      enddo

      istatus= NF_INQ_VARID(ncid,'powrfc',vid)
      call ncvpt_doubl2(ncid,vid,start_powrf,count_powrf,tem1,istatus)

      istatus= NF_INQ_VARID(ncid,'powurfc',vid)
      call ncvpt_doubl2(ncid,vid,start_powrf(2),count_powrf(2),
     +           powurfc(1),istatus)

      call bcast(tem1,zero,lrz)
      do ll=1,lrz
            tem1(ll)=powrft(lrindx(ll))
      enddo
      istatus= NF_INQ_VARID(ncid,'powrft',vid)
      call ncvpt_doubl2(ncid,vid,start_powrft,count_powrft,tem1,istatus)

      endif ! urfmod.ne."disabled"


      istatus= NF_INQ_VARID(ncid,'curtor',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,curtor,istatus)

      istatus= NF_INQ_VARID(ncid,'ccurtor',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,ccurtor(1),istatus)

      istatus= NF_INQ_VARID(ncid,'curpol',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,curpol,istatus)

      istatus= NF_INQ_VARID(ncid,'ccurpol',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,ccurpol(1),istatus)

      if (ngen.eq.1) then
         k=1
         do ll=1,lrzmax
            tr(ll)=curr(k,ll)/3.e9
         enddo
         istatus= NF_INQ_VARID(ncid,'curr',vid)
         call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)
         !YuP[07-31-2014] Added:
         do ll=1,lrzmax
            tr(ll)=energym(k,ll)
         enddo
         istatus= NF_INQ_VARID(ncid,'energym',vid)
         call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)
      else  !  ngen.ge.2
         do k=1,ngen
         do ll=1,lrzmax
               tem1(ll+(k-1)*lrzmax)=curr(k,ll)/3.e9
         enddo
         enddo
         istatus= NF_INQ_VARID(ncid,'curr',vid)
         call ncvpt_doubl2(ncid,vid,start_r0k,count_r0k,tem1,istatus)
         !YuP[07-2017] added
         !do k=1,ngen
         !do ll=1,lrzmax
         !      tem1(ll+(k-1)*lrzmax)=den_fsa(k,ll)
         !enddo
         !enddo
         !istatus= NF_INQ_VARID(ncid,'den_fsa',vid)
         !call ncvpt_doubl2(ncid,vid,start_r0k,count_r0k,tem1,istatus)
         !YuP[07-2017] added
         do k=1,ngen
         do ll=1,lrzmax
               tem1(ll+(k-1)*lrzmax)=energym(k,ll)
         enddo
         enddo
         istatus= NF_INQ_VARID(ncid,'energym',vid)
         call ncvpt_doubl2(ncid,vid,start_r0k,count_r0k,tem1,istatus)
      endif ! ngen

      kk=1
      if (kelecg.ne.0) kk=kelecg
      do ll=1,lrzmax
         tr(ll)=currm(kk,ll)/3.e9
      enddo
      if (kelecg.ne.0) then ! e as gen.species
         istatus= NF_INQ_VARID(ncid,'currm_e',vid)
      else ! ion as general species
         istatus= NF_INQ_VARID(ncid,'currm_i',vid)
      endif
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)

      do ll=1,lrzmax
         tr(ll)=restp(nch(ll),ll)
      enddo
      istatus= NF_INQ_VARID(ncid,'restp',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)

      do ll=1,lrzmax
         tr(ll)=restnp(nch(ll),ll)
      enddo
      istatus= NF_INQ_VARID(ncid,'restnp',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)

      do ll=1,lrzmax
         tr(ll)=sptzrp(nch(ll),ll)
      enddo
      istatus= NF_INQ_VARID(ncid,'sptzrp',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,tr(1),istatus)

      istatus= NF_INQ_VARID(ncid,'rovsc',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,rovsc,istatus)

      istatus= NF_INQ_VARID(ncid,'rovsc_hi',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,rovsc_hi,istatus)

      istatus= NF_INQ_VARID(ncid,'zreskim',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,zreskim,istatus)

      istatus= NF_INQ_VARID(ncid,'taueeh',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,taueeh,istatus)

      istatus= NF_INQ_VARID(ncid,'nuestar',vid)  
      call ncvpt_doubl2(ncid,vid,start(3),r0_count,starnue,istatus)

      !YuP[2019-09] added time-dep. data. HERE n>0 [[
      !Note:       start(2:4)= {1,         1,   numrec1} HERE: n>0
      !Note: impur_count(1:3)= {nstates+1, lrz, 1      }
      if(nstates.gt.0)then ! If no impurities, nstates=0. 
        istatus= NF_INQ_VARID(ncid,'dens_imp_allstates',vid)
        call ncvpt_doubl2(ncid,vid,start(3:4),impur_count(2:3),
     &     dens_imp_allstates, istatus)
        istatus= NF_INQ_VARID(ncid,'dens_imp',vid)
        call ncvpt_doubl2(ncid,vid,start(2:4),impur_count(1:3),
     &     dens_imp, istatus)
        istatus= NF_INQ_VARID(ncid,'pellet_rho',vid)
        call ncvpt_doubl2(ncid,vid,start(4:4),impur_count(3:3),
     &     pellet_rho, istatus)
        istatus= NF_INQ_VARID(ncid,'Gablation',vid)
        call ncvpt_doubl2(ncid,vid,start(4:4),impur_count(3:3),
     &     Gablation, istatus)     
        istatus= NF_INQ_VARID(ncid,'pellet_Mrem',vid)
        call ncvpt_doubl2(ncid,vid,start(4:4),impur_count(3:3),
     &     pellet_Mrem, istatus)
      endif
      !YuP[2019-09] added time-dep. data ]]
      
cBH110320c     Following only set up for ngen=1
cBH110320      if (ngen.ge.2)
cBH110320     1     WRITE(*,*)'netcdfrw2: Tot pwrs only set up for ngen=1'
      if (13*lrz .gt. iyjx2) stop 'netcdfrw2: Need 13*lrz<(iy+2)*(jx+2)'
      tem1=0.d0
      do k=1,ngen
         kkk=(k-1)*13*lrz
         !call bcast(tem1,zero,13*lrz) !YuP[2023]
         do ll=1,lrz
            kk=1
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,-1,ll)
            kk=2
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,0,ll)
            kk=3
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,2,ll)
            kk=4
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,1,ll)
            kk=5
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,3,ll)
            kk=6
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,5,ll)
            kk=7
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,6,ll)
            kk=8
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,7,ll)
            kk=9
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,8,ll)
            kk=10
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,11,ll)
            kk=11
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,10,ll)
            kk=12
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,12,ll)
            kk=13
            tem1(ll+(kk-1)*lrz+kkk)=entr(k,4,ll)
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'powers',vid)        
      
      call ncvpt_doubl2(ncid,vid,start_powers,count_powers,tem1,istatus)
     
cBH110320c     Following only set up for ngen=1
cBH110320      if (ngen.ge.2)
cBH110320    1     WRITE(*,*)'netcdfrw2: Tot pwrs only set up for ngen=1'
      if (lrz .gt. iyjx2) stop 'netcdfrw2: Need lrz<(iy+2)*(jx+2)'
      tem1=0.d0
      do k=1,ngen
            kkk=(k-1)*13
            !call bcast(tem1,zero,13) YuP[2023]
            kk=1
            tem1(kk+kkk)=entrintr(k,-1)
            kk=2
            tem1(kk+kkk)=entrintr(k,0)
            kk=3
            tem1(kk+kkk)=entrintr(k,2)
            kk=4
            tem1(kk+kkk)=entrintr(k,1)
            kk=5
            tem1(kk+kkk)=entrintr(k,3)
            kk=6
            tem1(kk+kkk)=entrintr(k,5)
            kk=7
            tem1(kk+kkk)=entrintr(k,6)
            kk=8
            tem1(kk+kkk)=entrintr(k,7)
            kk=9
            tem1(kk+kkk)=entrintr(k,8)
            kk=10
            tem1(kk+kkk)=entrintr(k,11)
            kk=11
            tem1(kk+kkk)=entrintr(k,10)
            kk=12
            tem1(kk+kkk)=entrintr(k,12)
            kk=13
            tem1(kk+kkk)=entrintr(k,4)
      enddo  !  on k=1,ngen
      istatus= NF_INQ_VARID(ncid,'powers_int',vid)  
      call ncvpt_doubl2(ncid,vid,start_powers(2),count_powers(2),
     ~                  tem1,istatus)

      if (sigmamod .eq. "enabled") then ! n>0: save sigftt
         istatus= NF_INQ_VARID(ncid,'sigftt',vid)  
         do lsig=1,4
            tem2(lsig)=sigftt(nch(1),lsig)
         enddo
         call ncvpt_doubl2(ncid,vid,start_fus(2),count_fus(2),
     ~                     tem2,istatus)
         call check_err(istatus)
      endif ! sigmamod
      

c      if (npa_diag .eq. "ncdf_all") then   NEED at least do first and last
      if (npa_diag .ne. "disabled" .and. numrec1.ge.2) then

         istatus= NF_INQ_VARID(ncid,'eflux_npa',vid)
         call pack21(eflux,1,nena,1,nva,wkpack,nen_npa,nv_npa)
         call ncvpt_doubl2(ncid,vid,start_npa,count_npa,wkpack,istatus)

         istatus= NF_INQ_VARID(ncid,'efluxt',vid)  
         call ncvpt_doubl2(ncid,vid,start_npa(2),count_npa(2),efluxt(1),
     +        istatus)

      endif

      if (softxry .eq. "ncdf_all") then

         istatus= NF_INQ_VARID(ncid,'eflux',vid)  
         call pack21(eflux,1,nena,1,nva,wkpack,nen,nv)
         call ncvpt_doubl2(ncid,vid,start_xr,count_xr,wkpack,istatus)

         istatus= NF_INQ_VARID(ncid,'efluxt',vid)  
         call ncvpt_doubl2(ncid,vid,start_xr(2),count_xr(2),efluxt(1),
     +        istatus)

      endif
      
          
      if (npa_diag .eq. "ncdf_all") then

         istatus= NF_INQ_VARID(ncid,'eflux_npa',vid)
         call pack21(eflux,1,nena,1,nva,wkpack,nen_npa,nv_npa)
         call ncvpt_doubl2(ncid,vid,start_npa,count_npa,wkpack,istatus)

         istatus= NF_INQ_VARID(ncid,'efluxt',vid)
         call ncvpt_doubl2(ncid,vid,start_npa(2),count_npa(2),efluxt(1),
     +        istatus)

      endif  ! On npa_diag .eq. "ncdf_all"


cyup      if ( netcdfshort.eq.'longer_f' ) then
      if ( (netcdfshort.eq.'longer_f').or.
     +     ((netcdfshort.eq.'lngshrtf').and.(isave.ne.0))  ) then
     
         !YuP[2018-09-27] Added option to save f() only at nsave() steps,
         !rather than at every step.
         !write(*,*)'netcdfrw2: netcdfshort,n,isave=',netcdfshort,n,isave
         ! Here: n>0
         istatus= NF_INQ_VARID(ncid,'f',vid)  
         
         if (ngen.eq.1) then
         
           if(netcdfshort.eq.'lngshrtf') then
             startf(4)=numrecsave !YuP[2018-09-28]netcdfshort.eq.'lngshrtf'
           else ! 'longer_f'
             startf(4)=numrec1 ! usual (saved at every t step)
           endif
           do ll=1,lrors !YuP[2022-02-11] was lrz
              do j=1,jx ! 0,jx+1 !1,jx
                do i=1,iymax ! 0,iy+1 !1,iy
                  temp1(i,j)=f(i,j,1,ll)
                enddo
              enddo
              call pack21(temp1(1:iymax,1:jx),1,iymax,1,jx,
     &                    wkpack(1:iymax*jx),iymax,jx)
              !YuP[2021-03-11] Changed iy-->iymax 
              !(just in case if iy is changed by iy=iy_(l_) somewhere)
              startf(3)=ll
              call ncvpt_doubl2(ncid,vid,startf(1:4),countf(1:4),
     +                          wkpack,istatus)
               !So, here startf={1,   1,   ll,  numrec1(or numrecsave)}
               !         countf={iy,  jx,  1,   1}
               !         dimsf= {ydimf,xdimf,rdim,tdim(or tsavedim)}
           enddo
           startf(4)=numrec1 ! restore
         
         else ! ngen>1
         
           if(netcdfshort.eq.'lngshrtf') then
             startg(5)=numrecsave !YuP[2018-09-28]netcdfshort.eq.'lngshrtf'
           else ! 'longer_f'
             startg(5)=numrec1 ! usual (saved at every t step)
           endif
           do k=1,ngen
            do ll=1,lrors !YuP[2022-02-11] was lrz
               do j=1,jx ! 0,jx+1 !1,jx
                  do i=1,iymax ! 0,iy+1 !1,iy
                     temp1(i,j)=f(i,j,k,ll)
                  enddo
               enddo
               call pack21(temp1(1:iymax,1:jx),1,iymax,1,jx,
     &                     wkpack(1:iymax*jx),iymax,jx)
               !YuP[2021-03-11] Changed iy-->iymax 
               !(just in case if iy is changed by iy=iy_(l_) somewhere)
               startg(3)=ll
               startg(4)=k
               call ncvpt_doubl2(ncid,vid,startg,countg,wkpack,istatus)
               !So, here startg={1,   1,   ll,  k,   numrec1(or numrecsave)}
               !         countg={iy,jx, 1,  1,   1}
               !         dimsg= {ydimf,xdimf,rdim,gdim,tdim(or tsavedim)}
            enddo  !  On ll
           enddo  !  On k
           startg(3)=1  ! restore
           startg(4)=1  ! restore
           startg(5)=numrec1 ! restore
         
         endif ! ngen
         
      endif ! netcdfshort

cBH131030:  Could replace above k=1 output of distn by following
cBH131030:  output of f for all k, but for now leave it out as
cBH131030:  the netcdf file many easily get unwieldly large.
cYuP[2018-09-28] BUT WE MUST DO IT because 'f' was defined 
cYuP             with either dims() or dimg(), depending on ngen.
        

      if (netcdfshort.eq.'long_jp') then ! Here n>0  !YuP[2020-02-06]
      
         if (ngen.eq.1) then
         
         !Note: start(2:4)={1,  1, numrec1} !(saved at every t step)
         !Note: count(2:4)={jx,lrz,1}
         k=1
         do ll=1,lrz
            do j=1,jx
               i=j+(lrindx(ll)-1)*jx
               tem1(i)=currv(j,k,lrindx(ll))/3.e9
            enddo
         enddo
         istatus= NF_INQ_VARID(ncid,'currv',vid)  
         call ncvpt_doubl2(ncid,vid,start(2:4),count(2:4),tem1,istatus)
         do ll=1,lrz
            do j=1,jx
               i=j+(lrindx(ll)-1)*jx
               tem1(i)=pwrrf(j,k,lrindx(ll))
            enddo
         enddo
         istatus= NF_INQ_VARID(ncid,'pwrrf',vid)  
         call ncvpt_doubl2(ncid,vid,start(2:4),count(2:4),tem1,istatus)

            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
            ! tam4 contains pitch angle averaged equatorial distribution function.
            !For possible future use:
            ! tam5 to contain pitch angle integrated distribution <f> such
            ! that integral{<f> d(gamma)} = fsa density.
            ! tam5=2*pi*x*cnorm2*gamma*int_i{sinn*dy*vptb*f_code}/zmaxpsi.
            do ll=1,lrors !YuP[2022-02-11] was lrz
              tam4=0.d0
              tam5=0.d0
              do j=1,jx
                do i=1,iymax
                  tam4(j)=tam4(j)+f(i,j,k,ll)*cynt2(i,ll) !integral over i
                  tam5(j)=tam5(j)+f(i,j,k,ll)*cynt2(i,ll)
     &                                       *vptb(i,lrindx(ll))
                  !YuP[2021-02-26] Corrected vptb(i,ll)->vptb(i,lrindx(ll))
                  !This is important for CQLP (when lr_=1, but l_=1:ls)
                  !Probably should not be using vptb in CQLP version at all.
                enddo
                tam4(j)=tam4(j)/twoint(ll) != INTEGR{f*sin(theta0)dtheta0} /2
                tam5(j)=tam5(j)/zmaxpsi(lrindx(ll))*x(j)*gamma(j)*cnorm2
              enddo ! j
              do j=1,jx
                 i=j+(ll-1)*jx
                 tem1(i)=tam4(j) ! Save favr_thet0(j), each ll
              enddo
            enddo ! ll
            istatus= NF_INQ_VARID(ncid,'favr_thet0',vid) 
            call ncvpt_doubl2(ncid,vid,start(2:4),count(2:4),tem1,
     &      istatus) 
            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2

         else  !  ngen.ge.2
         
         do k=1,ngen
           startg(4)=k ! and start(2) is not changed, it is 1
           startg(5)=numrec1
           countg(3)=lrz ! and countg(4)=1;  countg(5)=1  
           !So here startg(2:5)={1,  1,   k,   numrec1}
           !        countg(2:5)={jx,lrz,  1,   1      }
           !and recall that 
           !   currv_dimsg(1:4)={xdim,rdim,gdim,tdim }
           do ll=1,lrz
            do j=1,jx
               i=j+(lrindx(ll)-1)*jx
               tem1(i)=currv(j,k,lrindx(ll))/3.e9
            enddo
           enddo
           istatus= NF_INQ_VARID(ncid,'currv',vid)  
           call ncvpt_doubl2(ncid,vid,startg(2:5),countg(2:5),tem1,
     &     istatus)
           do ll=1,lrz
            do j=1,jx
               i=j+(lrindx(ll)-1)*jx
               tem1(i)=pwrrf(j,k,lrindx(ll))
            enddo
           enddo
           istatus= NF_INQ_VARID(ncid,'pwrrf',vid)  
           call ncvpt_doubl2(ncid,vid,startg(2:5),countg(2:5),tem1,
     &     istatus)
     
            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
            ! tam4 contains pitch angle averaged equatorial distribution function.
            !For possible future use:
            ! tam5 to contain pitch angle integrated distribution <f> such
            ! that integral{<f> d(gamma)} = fsa density.
            ! tam5=2*pi*x*cnorm2*gamma*int_i{sinn*dy*vptb*f_code}/zmaxpsi.
            do ll=1,lrors !YuP[2022-02-11] was lrz
              tam4=0.d0
              tam5=0.d0
              do j=1,jx
                do i=1,iymax
                  tam4(j)=tam4(j)+f(i,j,k,ll)*cynt2(i,ll) !integral over i
                  tam5(j)=tam5(j)+f(i,j,k,ll)*cynt2(i,ll)
     &                                       *vptb(i,lrindx(ll))
                  !YuP[2021-02-26] Corrected vptb(i,ll)->vptb(i,lrindx(ll))
                  !This is important for CQLP (when lr_=1, but l_=1:ls)
                  !Probably should not be using vptb in CQLP version at all.
                enddo
                tam4(j)=tam4(j)/twoint(ll) != INTEGR{f*sin(theta0)dtheta0} /2
                tam5(j)=tam5(j)/zmaxpsi(lrindx(ll))*x(j)*gamma(j)*cnorm2
              enddo ! j
              do j=1,jx
                 i=j+(ll-1)*jx
                 tem1(i)=tam4(j) ! Save favr_thet0(j), each ll
              enddo
            enddo ! ll
            istatus= NF_INQ_VARID(ncid,'favr_thet0',vid) 
            call ncvpt_doubl2(ncid,vid,startg(2:5),countg(2:5),tem1,
     &      istatus) 
            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
     
         enddo  !  on k=1,ngen
         startg(4)=1 ! restore
         startg(5)=numrec1 ! restore
         countg(3)=1 ! restore
         
         endif  !  on ngen

         !YuP[2022-03-19] Added fl()==F_par reduced distr.func.
         !One species, for now. 
         istatus= NF_INQ_VARID(ncid,'fl',vid) !here: 'long_jp', n>0
         call ncvpt_doubl2(ncid,vid,start(2:4),count_fl(1:3),
     &        fl(1:jfl,1:lrors),istatus) !count_fl={jfl; lrz or lz; 1}
                                         !start(4)=numrecl

      endif ! 'long_jp'  ! Here n>0   !YuP[2020-02-06]
         


c --- endif (n.gt.0) ---     if at l 3378
      endif








c.......................................................................
cl    3.3 Variables saved only at last time-step

c --- begin if ---
      if (n.eq.nstop) then  !endif at line 4139

c --- full update of data file (last time-step only) ---

      if (sigmamod.eq.'enabled') then ! n=nstop: save fuspwrvt,flux_neutron_f,...
         istatus= NF_INQ_VARID(ncid,'fuspwrvt',vid)  
         call ncvpt_doubl2(ncid,vid,1,4,fuspwrvt,istatus)
         call check_err(istatus)
         istatus= NF_INQ_VARID(ncid,'fuspwrv',vid)  
         do ii=1,4
            do ll=1,lrzmax
               ielem=ll+(ii-1)*lrzmax
               tem2(ielem)=fuspwrv(ii,ll)
            enddo
         enddo
         call ncvpt_doubl2(ncid,vid,start_fus,count_fus,tem2,istatus)
         call check_err(istatus)
      endif

      if(ampfmod.eq.'enabled')then !elecfldn(0:,0:,0:) saved at the last t step
         istatus= NF_INQ_VARID(ncid,'elecfldn',vid)
         if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
         write(*,*)'netcdfrw2: n=1,it=0,elecfldn(:,n,0)*300=',
     +                              elecfldn(0:lrz+1,1,0)*300
         write(*,*)'netcdfrw2: n=1,it=1,elecfldn(:,n,1)*300=',
     +                              elecfldn(0:lrz+1,1,1)*300
         endif
         call ncvpt_doubl2(ncid,vid,start_elecfldn,count_elecfldn,
     +        elecfldn(0,0,0),istatus)
      endif      

      if ( (netcdfshort.ne.'enabled')  .and. 
     +     (netcdfshort.ne.'longer_f') .and.
     +     (netcdfshort.ne.'lngshrtf')        ) then ! here: n=nstop
      
         istatus= NF_INQ_VARID(ncid,'f',vid)  
cBH011221: This storage is set up for constant iy as function of radius.
cBH011221: Needs generalizing. Should we store in reg array to max(iy_)?
cBH011221: For now, simply stop.
!         do ll=1,lrors !YuP[2022-02-11] was lrz
!            if (iy_(ll).ne.iymax) 
!     +           stop 'netcdfrw2: Cant handle iy.ne.iymax'
!                 !YuP[2021-03] Could be relaxed now?
!         enddo

         if (ngen.eq.1) then
         
            k=1 !  
            if (tavg.ne."disabled") then
              PRINT *,'netcdfrw2 WARNING: favg is saved (NOT f)'
              PRINT *,'netcdfrw2 WARNING: but if time(nstop) < tavg1(1)'
              PRINT *,'netcdfrw2 WARNING: then favg == f.'
            endif
            WRITE(*,*)
     +        'netcdfrw2[netcdfshort=disabled]:Write f into mnemonic.nc'
            WRITE(*,*)'netcdfrw2_4060: For checkup SUM(f),SUM(gone)=', 
     +                 SUM(f),SUM(gone)
            WRITE(*,*)'netcdfrw2_4060: For checkup MIN(f),MAX(f)=', 
     +                 MINVAL(f),MAXVAL(f)
            do ll=1,lrors !YuP[2022-02-11] was lrz
               if (tavg.eq."disabled") then
                  do j=1,jx ! 0,jx+1 !1,jx
                  do i=1,iymax ! 0,iy+1 !1,iy
                     temp1(i,j)=f(i,j,k,ll)
                     !YuP if(gone(i,j,k,lrindx(ll)).lt.-0.1) temp1(i,j)=em90
                     !YuP[2019-07-05] Better leave it "raw". 
                     ! The negative values will be adjusted after restart
                     ! in other subroutines.
                  enddo
                  enddo
               else  !On tavg = enabled
                  do j=1,jx ! 0,jx+1 !1,jx
                  do i=1,iymax ! 0,iy+1 !1,iy
                     temp1(i,j)=favg(i,j,k,ll)
                     !if(gone(i,j,k,lrindx(ll)).lt.-0.1) temp1(i,j)=em90
                     !Note: because of gone, the saved favg can be 
                     !smaller than the original favg (removed loss cone).
                     !Comment the above if(gone...) 
                     !if you want to leave favg intact.
                  enddo
                  enddo
                  WRITE(*,'(a,i6,2e13.4)')
     +            'nstop.netcdfrw2/tavg=en. ll, sumij(favg), sumij(f)',
     +             ll,sum(temp1),sum(f(:,:,k,ll))
               endif  !On tavg
!               WRITE(*,*)'netcdfrw2_4060: ll,MAX,SUM(temp1)=', 
!     &                 ll,MAXVAL(temp1),SUM(temp1)
               startf(3)=ll
               call pack21(temp1(1:iymax,1:jx),1,iymax,1,jx,
     &                     wkpack(1:iymax*jx),iymax,jx)
               call ncvpt_doubl2(ncid,vid,startf(1:3),countf(1:3),
     +                           wkpack,istatus)
                !YuP[2021-03-11] Changed iy-->iymax 
                !(just in case if iy is changed by iy=iy_(l_) somewhere)
!!3               call ncvpt_doubl2(ncid,vid,startf(1:3),countf(1:3),
!!3     +                           temp1(1:iy,1:jx),istatus)
               !So, here startf={1,   1,   ll}
               !         countf={iy,  jx,  1}
               !         dimsf= {ydimf,xdimf,rdim}
            enddo ! ll
            startf(3)=1 ! restore
            
         else  !  ngen.ge.1

            do k=1,ngen
            do ll=1,lrors !YuP[2022-02-11] was lrz
               if (tavg.eq."disabled") then
               do j=1,jx ! 0,jx+1 !1,jx
                  do i=1,iymax ! 0,iy+1 !1,iy
                     temp1(i,j)=f(i,j,k,ll)
                  enddo
               enddo
               else  !On tavg
               do j=1,jx ! 0,jx+1 !1,jx
                  do i=1,iymax ! 0,iy+1 !1,iy
                     temp1(i,j)=favg(i,j,k,ll)
                  enddo
               enddo
               endif  !On tavg
               call pack21(temp1(1:iymax,1:jx),1,iymax,1,jx,
     &                     wkpack(1:iymax*jx),iymax,jx)
               !YuP[2021-03-11] Changed iy-->iymax 
               !(just in case if iy is changed by iy=iy_(l_) somewhere)
               startg(3)=ll
               startg(4)=k
               call ncvpt_doubl2(ncid,vid,startg(1:4),countg(1:4),
     +                           wkpack,istatus)
               !So, here startg={1,   1,   ll,  k}
               !         countg={iy,jx, 1,  1}
               !         dimsg= {ydimf,xdimf,rdim,gdim}
            enddo ! on ll
            enddo ! on k
            startg(3)=1 ! restore
            startg(4)=1 ! restore

         endif !  on ngen

      endif


      if (softxry .ne. "disabled" .and. softxry.ne."ncdf_all") then
 
         start_xr(3)=2
         istatus= NF_INQ_VARID(ncid,'eflux',vid)  
         call pack21(eflux,1,nena,1,nva,wkpack,nen,nv)
         call ncvpt_doubl2(ncid,vid,start_xr,count_xr,wkpack,istatus)

         istatus= NF_INQ_VARID(ncid,'efluxt',vid)  
         call ncvpt_doubl2(ncid,vid,start_xr(2),count_xr(2),efluxt(1),
     +        istatus)

      endif

      if (netcdfshort.ne.'long_jp') then ! Here n=nstop !YuP[2020-02-06]
         if (ngen.eq.1) then
         k=1
         do ll=1,lrz
            do j=1,jx
               i=j+(lrindx(ll)-1)*jx
               tem1(i)=currv(j,k,lrindx(ll))/3.e9
            enddo
         enddo
         istatus= NF_INQ_VARID(ncid,'currv',vid)  
         call ncvpt_doubl2(ncid,vid,start(2:3),count(2:3),tem1,istatus)
         do ll=1,lrz
            do j=1,jx
               i=j+(lrindx(ll)-1)*jx
               tem1(i)=pwrrf(j,k,lrindx(ll))
            enddo
         enddo
         istatus= NF_INQ_VARID(ncid,'pwrrf',vid)  
         call ncvpt_doubl2(ncid,vid,start(2:3),count(2:3),tem1,istatus)
         
            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
            ! tam4 contains pitch angle averaged equatorial distribution function.
            !For possible future use:
            ! tam5 to contain pitch angle integrated distribution <f> such
            ! that integral{<f> d(gamma)} = fsa density.
            ! tam5=2*pi*x*cnorm2*gamma*int_i{sinn*dy*vptb*f_code}/zmaxpsi.
            do ll=1,lrors !YuP[2022-02-11] was lrz
              tam4=0.d0
              tam5=0.d0
              do j=1,jx
                do i=1,iymax
                  tam4(j)=tam4(j)+f(i,j,k,ll)*cynt2(i,ll) !integral over i
                  tam5(j)=tam5(j)+f(i,j,k,ll)*cynt2(i,ll)
     &                                       *vptb(i,lrindx(ll))
                  !YuP[2021-02-26] Corrected vptb(i,ll)->vptb(i,lrindx(ll))
                  !This is important for CQLP (when lr_=1, but l_=1:ls)
                  !Probably should not be using vptb in CQLP version at all.
                enddo
                tam4(j)=tam4(j)/twoint(ll) != INTEGR{f*sin(theta0)dtheta0} /2
                tam5(j)=tam5(j)/zmaxpsi(lrindx(ll))*x(j)*gamma(j)*cnorm2
              enddo ! j
              do j=1,jx
                 i=j+(ll-1)*jx
                 tem1(i)=tam4(j) ! Save favr_thet0(j), each ll
              enddo
            enddo ! ll
            istatus= NF_INQ_VARID(ncid,'favr_thet0',vid) 
            call ncvpt_doubl2(ncid,vid,start(2:3),count(2:3),tem1,
     &      istatus) 
            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2

         else  !  ngen.ge.2

         do k=1,ngen
         do ll=1,lrz
            do j=1,jx
               i=j+(lrindx(ll)-1)*jx
               tem1(i)=currv(j,k,lrindx(ll))/3.e9
            enddo
         enddo
         start(4)=k ! and start(3) is not changed, it is 1
         !Note: count(3)=lrz
         !Note: count(4)=1 
         istatus= NF_INQ_VARID(ncid,'currv',vid)  
         call ncvpt_doubl2(ncid,vid,start(2:4),count(2:4),tem1,
     &    istatus)
         do ll=1,lrz
            do j=1,jx
               i=j+(lrindx(ll)-1)*jx
               tem1(i)=pwrrf(j,k,lrindx(ll))
            enddo
         enddo
         start(4)=k ! and start(3) is not changed, it is 1
         !Note: count(3)=lrz
         !Note: count(4)=1 
         istatus= NF_INQ_VARID(ncid,'pwrrf',vid)  
         call ncvpt_doubl2(ncid,vid,start(2:4),count(2:4),tem1,
     &    istatus)
     
            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2
            ! tam4 contains pitch angle averaged equatorial distribution function.
            !For possible future use:
            ! tam5 to contain pitch angle integrated distribution <f> such
            ! that integral{<f> d(gamma)} = fsa density.
            ! tam5=2*pi*x*cnorm2*gamma*int_i{sinn*dy*vptb*f_code}/zmaxpsi.
            do ll=1,lrors !YuP[2022-02-11] was lrz
              tam4=0.d0
              tam5=0.d0
              do j=1,jx
                do i=1,iymax
                  tam4(j)=tam4(j)+f(i,j,k,ll)*cynt2(i,ll) !integral over i
                  tam5(j)=tam5(j)+f(i,j,k,ll)*cynt2(i,ll)
     &                                       *vptb(i,lrindx(ll))
                  !YuP[2021-02-26] Corrected vptb(i,ll)->vptb(i,lrindx(ll))
                  !This is important for CQLP (when lr_=1, but l_=1:ls)
                  !Probably should not be using vptb in CQLP version at all.
                enddo
                tam4(j)=tam4(j)/twoint(ll) != INTEGR{f*sin(theta0)dtheta0} /2
                tam5(j)=tam5(j)/zmaxpsi(lrindx(ll))*x(j)*gamma(j)*cnorm2
              enddo ! j
              do j=1,jx
                 i=j+(ll-1)*jx
                 tem1(i)=tam4(j) ! Save favr_thet0(j), each ll
              enddo
            enddo ! ll
            istatus= NF_INQ_VARID(ncid,'favr_thet0',vid) 
            call ncvpt_doubl2(ncid,vid,start(2:4),count(2:4),tem1,
     &      istatus) 
            !YuP[2020-04-07] Added INTEGR{f*sin(theta0)dtheta0} /2    
     
         enddo  !  on k=1,ngen
         endif  !  on ngen
         start(4)=1 ! restored
         
         !YuP[2022-03-19] Added fl()==F_par reduced distr.func.
         !One species, for now. Here - saved at last time step only.
         istatus= NF_INQ_VARID(ncid,'fl',vid) !here: 'ne.long_jp', n=nstop
         call ncvpt_doubl2(ncid,vid,start(1:2),count_fl(1:2),
     &        fl(1:jfl,1:lrors),istatus) !count_fl={jfl; lrz or lz; 1}
         
      endif !(netcdfshort.ne.'long_jp') ! Here n=nstop !YuP[2020-02-06]

c --- endif (n.eq.nstop) ---
      endif  !if at l 4016

C-----------------------------------------------------------------------
c
cl    4. Close netCDF file
c
      if (n.eq.nstop) then
         istatus = NF_CLOSE(ncid) !-YuP: NetCDF-f77
         call check_err(istatus)         
      endif

      return
      end subroutine netcdfrw2
!=======================================================================
c       
c
      subroutine check_err(iret)
      integer iret
      include 'netcdf.inc'
      if (iret .ne. NF_NOERR) then
         PRINT *, 'netCDF error:', iret
         PRINT *, NF_STRERROR(iret) ! print error explanation
      stop 'netCDF error'
      endif
      end

c
c
      subroutine netcdfmain
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     Controls some netcdf output (could make it all).
c     Presently, just output of velocity-space flux data.
c.......................................................................
      save
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

CMPIINSERT_IF_RANK_NE_0_RETURN
 ! save data on mpirank.eq.0 only

      
      if (n.ne.nstop) return
      if ((netcdfvecs.eq."irzplt" .and. mplot(l_).eq."enabled")
     +     .or. (netcdfvecs.eq."all")) then
        
         if (netcdfvecal.ne."disabled") then
            igrid=0
            if(netcdfvecal.eq."x-theta") igrid=1
            call netcdfvec(4,igrid)
         endif
         if (netcdfvecrf.ne."disabled") then
            igrid=0
            if(netcdfvecrf.eq."x-theta") igrid=1
            call netcdfvec(3,igrid)
         endif
         if (netcdfvece.ne."disabled") then
            igrid=0
            if(netcdfvece.eq."x-theta") igrid=1
            call netcdfvec(2,igrid)
         endif
         if (netcdfvecc.ne."disabled") then
            igrid=0
            if(netcdfvecc.eq."x-theta") igrid=1
            call netcdfvec(1,igrid)
         endif

      endif

      return
      end
c
c
      subroutine netcdfvec(lefct,igrid)
      implicit integer (i-n), real*8 (a-h,o-z)

c...................................................................
c
c     This subroutine writes out flux data in netCDF format.
c     It is called if one of netcdfvecXX = .ne."disabled".
c     igrid=1:  indicates netcdfvecXX="u-theta", then write the
c       x,theta components of the fluxes on the code x,theta grid 
c       (same as distribution functions).
c     igrid=0: indicates netcdfvecXX = .ne."disabled" and .ne."u-theta",
c       then write velocity-space fluxes suitable for
c       vector plot in (x-par,x-perp) coordinates, on an xpar,xperp
c       grid.
c     
c...................................................................

      save
      include 'param.h'
      include 'comm.h'
c     advnce.h contains statement functions,
c     including gfi,hfi used below:
      include 'advnce.h'
      include 'netcdf.inc'
      
      dimension ll_netcdf(lrza),rya_netcdf(lrza),
     +          itl_netcdf(lrza),itu_netcdf(lrza)

      integer ncid,vid,istatus
      integer ipxydim,jpxydim
      integer dims(4),count(4),start(4)
      integer chardim,char64dim
      integer xdim,ydim,rdim,gdim
      integer y_dims(2),y_count(2),y_start(2)

      character*8 target
      character*18 fluxcmpt(4)

      integer lfirst(4)
      data lfirst/4*0/
      data fluxcmpt/"collisional flux  ","electric fld flux ",
     +              "rf diffusion flux ","sum of fluxes     "/ 


      real*8, allocatable :: wkpack(:) ! local working array for pack21
     
      if ((lefct.lt.1) .or. (lefct.gt.4)) stop 'pltvec:lefct'

c...................................................................
c     Set up netCDF output on last time step, if
c     this is first call for a given value of lefct.
c     It is assumed that lefct is an integer.ge.1, and that
c     there are lrz (or nirzplt, if netcdfvecs="irzplt") 
c     calls to this routine (corresponding
c     to each of the FP'd flux surfaces for each
c     value of lefct.
c
c     lefct=1  ==> collisional flux
c     lefct=2  ==> electric fld flux
c     lefct=3  ==> rf diffusion flux
c     lefct=4  ==> sum of fluxes
c     
c     This routine can readily be adapted for
c     cqlpmod="enabled".
c...................................................................


      if (cqlpmod.eq."enabled") then
         stop "netcdfrw2: Need to adapt sub for cqlpmod.eq.enabled"
      endif

      if (.NOT.ALLOCATED(wkpack)) then ! allocate working array for pack21
         nwkpack=max(iyjx2, jpxy*ipxy) + 10   ! +10 just in case 
         allocate(wkpack(nwkpack),STAT=istat)
         call bcast(wkpack,zero,SIZE(wkpack))
      endif

c     lfirst(lefct) is an indicator of whether this is the first
c     call for given lefct.
      if (netcdfnm.ne."disabled" .and. n.eq.nstop
     +    .and. lfirst(lefct).ne.lefct) then

         lfirst(lefct)=lefct

         write(t_,235) mnemonic(1:length_char(mnemonic)),lefct
 235     format(a,"_flux_",i1,".nc")

         istatus = NF_CREATE(t_, NF_CLOBBER, ncid) !-YuP: NetCDF-f77
         call check_err(istatus)

         call ncaptc2(ncid,NCGLOBAL,'title',NCCHAR,18,
     +    fluxcmpt(lefct),istatus)

c        Put radial flux surface numbers in ll_netcdf(),
c        number of surfaces in n_netcdf, normalized radii in rya_netcdf(),
c        indices of lower and upper trapped-passing boundaries in
c        itl_netcdf() and itu_netcdf().
         if (netcdfvecs.eq."irzplt") then
            n_netcdf=nirzplt
            llcount=0
            do ll=1,lrz
               if (mplot(ll).eq."enabled") then
                  llcount=llcount+1
                  ll_netcdf(llcount)=ll
                  rya_netcdf(llcount)=rya(lrindx(ll))
                  itl_netcdf(llcount)=itl_(lrindx(ll))
                  itu_netcdf(llcount)=itu_(lrindx(ll))
               endif
            enddo
         else
            n_netcdf=lrz
            do ll=1,lrz
               ll_netcdf(ll)=ll
               rya_netcdf(ll)=rya(lrindx(ll))
               itl_netcdf(ll)=itl_(lrindx(ll))
               itu_netcdf(ll)=itu_(lrindx(ll))
           enddo
         endif

         if(igrid.eq.0) then

         istatus= NF_DEF_DIM(ncid, 'jpxydim', jpxy,     jpxydim)
         istatus= NF_DEF_DIM(ncid, 'ipxydim', ipxy,     ipxydim)
         istatus= NF_DEF_DIM(ncid, 'rdim',    n_netcdf, rdim)
         istatus= NF_DEF_DIM(ncid, 'gdim',    ngen,     gdim)
         istatus= NF_DEF_DIM(ncid, 'chardim',     8,    chardim)
         istatus= NF_DEF_DIM(ncid, 'char64dim',  64,    char64dim)

         dims(1)=jpxydim
         dims(2)=ipxydim
         dims(3)=rdim
         dims(4)=gdim

         start(1)=1
         start(2)=1
         start(3)=1
         start(4)=1

         count(1)=jpxy !case igrid=0
         count(2)=ipxy !case igrid=0
         count(3)=1    !case igrid=0
c sgbaek 
c         count(1)=1
         count(4)=1
c end of sgbaek

      vid=ncvdef2(ncid,'mnemonic',NCCHAR,1,char64dim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +        'Mnemonic run identifier',istatus)

      vid=ncvdef2(ncid,'grid_type',NCCHAR,1,chardim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,35,
     +        'Indicates xpar-prp or x-theta grid',istatus)

      vid=ncvdef2(ncid,'lrz',NCLONG,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,34,
     +        'Number of FPd radial surface bins',istatus)
         call check_err(istatus)

      vid=ncvdef2(ncid,'n_netcdf',NCLONG,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,44,
     +        'Number of radial surfaces for output (=rdim)',istatus)
         call check_err(istatus)

      vid=ncvdef2(ncid,'ll_netcdf',NCLONG,1,rdim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +        'Radial bin numbers where data is stored',istatus)
         call check_err(istatus)

      vid=ncvdef2(ncid,'jpxy',NCLONG,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,45,
     +        'Number of parallel momentum points (=jpxydim)',istatus)
         call check_err(istatus)
         
      vid=ncvdef2(ncid,'ipxy',NCLONG,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +     'Number of perp momentum points (=ipxydim)',istatus)
         call check_err(istatus)

      vid=ncvdef2(ncid,'xll',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +        'Lower parallel momentum-per-mass, normalized',istatus)
         
      vid=ncvdef2(ncid,'xlu',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +        'Upper parallel momentum-per-mass, normalized',istatus)
         
      vid=ncvdef2(ncid,'xpl',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +        'Lower perp momentum-per-mass, normalized',istatus)
         
      vid=ncvdef2(ncid,'xpu',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +        'Upper perp momentum-per-mass, normalized',istatus)

      vid=ncvdef2(ncid,'xpar',NCDOUBLE,1,jpxydim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +        'Normalized par momentum-per-mass',istatus)

      vid=ncvdef2(ncid,'xperp',NCDOUBLE,1,ipxydim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +        'Normalized perp momentum-per-mass',istatus)

      vid=ncvdef2(ncid,'rhomax',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
     +        'Generalized plasma minor radius',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'cms',istatus)
         
      vid=ncvdef2(ncid,'rya_netcdf',NCDOUBLE,1,rdim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +        'Normalized radial mesh at bin centers',istatus)

      if (ngen.eq.1) then !Maintaining backwards compatability
      vid=ncvdef2(ncid,'gamma_par',NCDOUBLE,3,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +        'Parallel momentum-space flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,16,
     +        'Normalized units',istatus)
         call ncaptc2(ncid,vid,'comment',NCCHAR,39,
     +        'Facility set up only for single species',istatus)
         
      vid=ncvdef2(ncid,'gamma_perp',NCDOUBLE,3,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +        'Perpendicular momentum-space flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,16,
     +        'Normalized units',istatus)
      else  !  ngen.ge.2
      vid=ncvdef2(ncid,'gamma_par',NCDOUBLE,4,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +        'Parallel momentum-space flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,16,
     +        'Normalized units',istatus)
         call ncaptc2(ncid,vid,'comment',NCCHAR,39,
     +        'Facility set up for one or more species',istatus)
         
      vid=ncvdef2(ncid,'gamma_perp',NCDOUBLE,4,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +        'Perpendicular momentum-space flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,16,
     +        'Normalized units',istatus)
         call ncaptc2(ncid,vid,'comment',NCCHAR,39,
     +        'Facility set up for one or more species',istatus)
      endif   !  on ngen

         else  ! igrid.ne.0 case


c     Maximum of iy_() as function of radius is iymax, set in ainsetpa.f:

cBH021028: This storage is set up for constant iy as function of radius.
cBH021028: Needs generalizing. Should we store in reg array to max(iy_)?
cBH021028: For now, simply stop.
         do ll=1,lrz
            if (iy_(ll).ne.iymax) 
     +           stop 'netcdfrw2/netcdfvec: Cant handle iy.ne.iymax'
            !YuP[2021-03] Could be relaxed now?
         enddo

         istatus= NF_DEF_DIM(ncid, 'xdim',    jx,       xdim) 
         istatus= NF_DEF_DIM(ncid, 'ydim',    iymax,       ydim)
         istatus= NF_DEF_DIM(ncid, 'rdim',    n_netcdf, rdim)
         istatus= NF_DEF_DIM(ncid, 'gdim',    ngen,     gdim)
         istatus= NF_DEF_DIM(ncid, 'chardim',     8,    chardim)
         istatus= NF_DEF_DIM(ncid, 'char64dim',  64,    char64dim)

         dims(1)=ydim
         dims(2)=xdim
         dims(3)=rdim
         dims(4)=gdim

         start(1)=1
         start(2)=1
         start(3)=1
         start(4)=1

         count(1)=iymax  !case igrid.ne.0
         count(2)=jx     !case igrid.ne.0
         count(3)=1      !case igrid.ne.0
         
         y_dims(1)=ydim
         y_dims(2)=rdim

         y_start(1)=1
         y_start(2)=1
         
         y_count(1)=iymax
         y_count(2)=1
         

      vid=ncvdef2(ncid,'mnemonic',NCCHAR,1,char64dim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +        'Mnemonic run identifier',istatus)

      vid=ncvdef2(ncid,'grid_type',NCCHAR,1,chardim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,36,
     +        'Indicates xpar-prp or x-theta grid',istatus)

      vid=ncvdef2(ncid,'lrz',NCLONG,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,34,
     +        'Number of FPd radial surface bins',istatus)
         call check_err(istatus)

      vid=ncvdef2(ncid,'n_netcdf',NCLONG,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,44,
     +        'Number of radial surfaces for output (=rdim)',istatus)
         call check_err(istatus)

      vid=ncvdef2(ncid,'ll_netcdf',NCLONG,1,rdim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +        'Radial bin numbers where data is stored',istatus)
         call check_err(istatus)

      vid=ncvdef2(ncid,'jx',NCLONG,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,35,
     +        'momentum-per-mass dimension (=xdim)',istatus)
         call check_err(istatus)
         
      vid=ncvdef2(ncid,'x',NCDOUBLE,1,xdim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +        'normalized momentum-per-mass',istatus)

      vid=ncvdef2(ncid,'vnorm',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +        'velocity (momentum-per-mass) norm',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +        'cms/sec',istatus)
         
      vid=ncvdef2(ncid,'enorm',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +        'Energy normalization',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'keV',istatus)
         
      vid=ncvdef2(ncid,'iy',NCLONG,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +        'max Pitch angle dimension (=ydim)',istatus)
         call check_err(istatus)
         
      vid=ncvdef2(ncid,'y',NCDOUBLE,2,y_dims(1:2),istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,36,
     +        'pitch angle at n_netcdf flux surfaces',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +        'radians',istatus)
         
      vid=ncvdef2(ncid,'itl_netcdf',NCLONG,1,rdim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +        'lower trapped-passing bndy index',istatus)
         
      vid=ncvdef2(ncid,'itu_netcdf',NCLONG,1,rdim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +        'upper trapped-passing bndy index',istatus)
         
      vid=ncvdef2(ncid,'rhomax',NCDOUBLE,0,0,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
     +        'Generalized plasma minor radius',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +        'cms',istatus)
         
      vid=ncvdef2(ncid,'rya_netcdf',NCDOUBLE,1,rdim,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +        'Normalized radial mesh at bin centers',istatus)
         
      if (ngen.eq.1) then  !Maintaining backwards compatability
      vid=ncvdef2(ncid,'gamma_x',NCDOUBLE,3,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +        'x-compnt momentum-space flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,16,
     +        'Normalized units',istatus)
         call ncaptc2(ncid,vid,'comment',NCCHAR,39,
     +        'Facility set up for one or more species',istatus)
         
      vid=ncvdef2(ncid,'gamma_theta',NCDOUBLE,3,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +        'Theta-compnt momentum-space flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,16,
     +        'Normalized units',istatus)
         call ncaptc2(ncid,vid,'comment',NCCHAR,39,
     +        'Facility set up for one or more species',istatus)
      else  !  ngen.ge.2
      vid=ncvdef2(ncid,'gamma_x',NCDOUBLE,4,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +        'x-compnt momentum-space flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,16,
     +        'Normalized units',istatus)
         call ncaptc2(ncid,vid,'comment',NCCHAR,39,
     +        'Facility set up for one or more species',istatus)
         
      vid=ncvdef2(ncid,'gamma_theta',NCDOUBLE,4,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,32,
     +        'Theta-compnt momentum-space flux',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,16,
     +        'Normalized units',istatus)
         call ncaptc2(ncid,vid,'comment',NCCHAR,39,
     +        'Facility set up for one or more species',istatus)
      endif  !  on ngen


         endif   !end igrid options



         istatus= NF_ENDDEF(ncid) !-YuP: NetCDF-f77
         call check_err(istatus)
         istatus = NF_CLOSE(ncid) !-YuP: NetCDF-f77

      endif
      
         

c     The following calculation of velocity space fluxes is copied
c     from subroutine pltvec.

      do 190 k=1,ngen
         if (tandem.eq."enabled" .and. k.eq.kionn) then
            xll=-xlwr
            xlu=xlwr
            xpl=0.
            xpu=xlwr
            xmaxq=xlwr
            target="ionmesh"
         else
            xll=-xmax
            xlu=xmax
            xpl=0.
            xpu=xmax
            xmaxq=xmax
            target="mainmesh"
         endif
         
c     If pltlim.ne."disabled", then plot in xpar,xprp-space up
c     to appropriate limit:
         if (pltlim.ne."disabled") then
            if (pltlim.eq.'x') then
               pltlimmm=pltlimm
            elseif (pltlim.eq.'u/c') then
               pltlimmm=pltlimm*cnorm
            elseif (pltlim.eq.'energy') then
               pltlimmm=sqrt((1.+pltlimm/restmkev)**2-1.)*cnorm
            endif
            xll=-pltlimmm
            xlu=pltlimmm
            xpl=0.
            xpu=pltlimmm
            xmaxq=pltlimmm
         endif
         
c     
         call bcast(da,zero,iyjxp1)
         call bcast(db,zero,iyjxp1)
         call bcast(dc,zero,iyjxp1)
         call bcast(dd,zero,iyp1jx)
         call bcast(de,zero,iyp1jx)
         call bcast(df,zero,iyp1jx)
         if (lefct .eq. 1) go to 50
         if (lefct .eq. 2) go to 60
         if (lefct .eq. 3) go to 80
         if (lefct .eq. 4) go to 100
 50      call coeffpad(k)



         go to 120
 60      if (abs(elecfld(lr_)) .lt. 1.e-09) go to 190
         call coefefad(k)



         go to 120
 80      continue
         xrf=0.
         if (n .lt. nonrf(k) .or. n .ge. noffrf(k)) go to 90
         call coefrfad(k,xrf)
 90      continue
         if (xrf.gt.0.) then



         endif
         go to 120
 100     continue
         call coefstup(k)



 120     continue
         
c...................................................................
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
c     
c     first the velocity flux coefficients for gfi..
c...................................................................

         call coefmidv(da,1)
         call coefmidv(db,2)
         call coefmidv(dc,3)
         
c...................................................................
c     the theta flux coefficients for hfi..
c...................................................................
         
         call coefmidt(dd,1)
         call coefmidt(de,2)
         call coefmidt(df,3)
         if (lefct .eq. 3 .and. xrf .eq. 0) go to 190
         call bcast(temp5(0,0),zero,iyjx2)
         call bcast(temp4(0,0),zero,iyjx2)
c
c        In the following, tam2 and tam3 are u-space fluxes Gamma_x and
c          sin(theta)*Gamma_theta (in code units)
c          interpolated onto the code theta,x-mesh. 
c           
c          For igrid.eq.0, temp5 and temp4 are the 
c          parallel, perpendicular components of flux Gamma,
c          respectively, on the x,theta-mesh.
c
c          For igrid.eq.1, temp5,temp4 are x,theta components, 
c          respectively, of flux Gamma on the theta,x-mesh.
c
         if (implct .eq. "enabled") then
            do 140 i=2,iymax-1
               do 141 j=2,jxm1
                  tam2(j)=-(gfi(i,j,k)+gfi(i,j-1,k))*.5/xsq(j)
                  tam3(j)=-(hfi(i,j)+hfi(i-1,j))*.5*xi(j)
 141           continue
               if (igrid.eq.1) then
                  do j=2,jxm1
                     temp5(i,j)=tam2(j)
                     temp4(i,j)=tam3(j)/sinn(i,l_)
                 enddo
               else
                  do j=2,jxm1
                     temp5(i,j)=tam2(j)*coss(i,l_)-tam3(j)
                     temp4(i,j)=tam2(j)*sinn(i,l_)+tam3(j)/tann(i,l_)
                  enddo
               endif
 140        continue
         else
            temp1(0:iymax+1,0:jx+1)=f_(0:iymax+1,0:jx+1,k,l_)
            temp2(0:iymax+1,0:jx+1)=fxsp(0:iymax+1,0:jx+1,k,l_)
            !YuP[2021-03-11] Changed iy-->iymax
            !(just in case if iy is changed by iy=iy_(l_) somewhere)
            do 240 j=2,jxm1
               do 241 i=2,iymax-1
                  temp6(i,j)=-(gfu(i,j,k)+gfu(i,j-1,k))*.5/xsq(j)
 241           continue
 240        continue
            call dcopy(iyjx2,temp2(0,0),1,temp1(0,0),1)
            call dcopy(iyjx2,f(0,0,k,l_),1,temp2(0,0),1)
            do 242 i=2,iymax-1
               do 243 j=2,jxm1
                  tam3(j)=-(hfu(i,j)+hfu(i-1,j))*.5*xi(j)
 243           continue
               if (igrid.eq.1) then
                  do j=2,jxm1
                     temp5(i,j)=temp6(i,j)
                     temp4(i,j)=tam3(j)/sinn(i,l_)
                  enddo
               else
                  do j=2,jxm1
                     temp5(i,j)=temp6(i,j)*coss(i,l_)-tam3(j)
                     temp4(i,j)=temp6(i,j)*sinn(i,l_)+tam3(j)/tann(i,l_)
                  enddo
               endif
 242        continue
         endif
         do 244 j=1,jx
            temp5(itl,j)=temp5(itl-1,j)
            temp5(itu,j)=temp5(itu+1,j)
            temp4(itl,j)=temp4(itl-1,j)
            temp4(itu,j)=temp4(itu+1,j)
 244     continue
         
c     
c     Above gives:
c     for igrid.eq.0, 
c       flux flux_perp(u,theta) ~ temp4, flux_par ~ temp5.
c       These are output to the netcdf file.
c     for igrid.eq.1, 
c       flux flux_u(theta,u)~temp5, flux_theta~temp4.
c
c

         if (igrid.eq.0) then

c
c       The following calls to prppr and dcopy put 
c       flux_par into xhead, flux_perp into yhead, on an x,y-grid.
c       (xhead(jpxy,ipxy) has temp1 ptr, 
c        yhead(jpxy,ipxy) has temp4 ptr),
c        and these are output to the netcdf file.

         call dcopy(iyjx2,temp5(0,0),1,temp3(0,0),1)         
         call prppr(target,"norm",xll,xlu,xpl,xpu) 
c sgbaek, following Line 258 in pltvec.f, I am commenting out the following and
c replace with the two lines
c         call dcopy(iyjx2,temp2(0,0),1,temp1(0,0),1)
         ipxjpx=jpxy*ipxy
         call dcopy(ipxjpx,fpn,1,xhead,1)
         
         call dcopy(iyjx2,temp4(0,0),1,temp3(0,0),1)         
         call prppr(target,"norm",xll,xlu,xpl,xpu)         
c sgbaek
c         call dcopy(iyjx2,temp2(0,0),1,temp4(0,0),1)
         call dcopy(ipxjpx,fpn,1,yhead,1)


c sgbaek, I am writing the following
!         WRITE(*,*)'sgbaek pltvec: jpxy,xpar=',jpxy,(xpar(j),j=1,jpxy)
!         WRITE(*,*)'sgbaek pltvec: ipxy,xperp=',ipxy,(xperp(i),i=1,ipxy)
!         WRITE(*,*) 'sgbaek temp1(0,0)', temp1(0,0)
!         WRITE(*,*) 'sgbaek temp1', temp1
!         WRITE(*,*) 'sgbaek xhead', xhead
!         WRITE(*,*) 'sgbaek xhead print rows  each column'
!         do i=1,ipxy
!            WRITE(*,*) 'i, what size', i,size(xhead),size(xhead,dim=1)
!            WRITE(*,122) (xhead(j,i), j=1,jpxy)
!            WRITE(*,*) '....'
!         enddo
!         WRITE(*,*) '....'
 122     format(101(1x,e17.10))
!         WRITE(*,*) 'sgbaek temp4(0,0)', temp4(0,0)
!         WRITE(*,*) 'sgbaek temp4', temp4
!         WRITE(*,*) 'sgbaek yhead', yhead
!         WRITE(*,*) 'sgbaek, size xhead, size yhead',
!     &                  size(xhead), size(yhead)
!         WRITE(*,*) 'sgbaek size1', size(xhead,dim=1), size(yhead,dim=1)
!         WRITE(*,*) 'sgbaek size2', size(xhead,dim=2), size(yhead,dim=2)
!         WRITE(*,*) 'sgbaek temp1 size1', 
!     &                   size(temp1,dim=1), size(temp4,dim=1)
!         WRITE(*,*) 'sgbaek temp1 size2', 
!     &                   size(temp1,dim=2), size(temp4,dim=2)
cc end of sgbaek write commands       

         endif  !end igrid.eq.0
         
         
         
         
c     gamma_par ~ temp1, gamma_prp ~ temp4
c     For netcdf data:  for each l_, compress into 
c     jpxy*ipxy array and output.
         
         
         if (netcdfnm.ne."disabled" .and. n.eq.nstop) then
            
            
            write(t_,236) mnemonic(1:length_char(mnemonic)),lefct
 236        format(a,"_flux_",i1,".nc")
c     Open the correct netcdf file, to get ncid.
            istatus = NF_OPEN(t_, NF_WRITE, ncid) !-YuP: NetCDF-f77
            
c     Write some data at head of the file:

            if (igrid.eq.0) then

            istatus= NF_INQ_VARID(ncid,'mnemonic',vid)  
            ll=length_char(mnemonic)
            call ncvptc2(ncid,vid,1,ll,mnemonic,ll,istatus)

            istatus= NF_INQ_VARID(ncid,'grid_type',vid)  
            call ncvptc2(ncid,vid,1,8,'xpar-prp',8,istatus)

            istatus= NF_INQ_VARID(ncid,'lrz',vid)  
            call ncvpt_int2(ncid,vid,1,1,lrz,istatus)
            
            istatus= NF_INQ_VARID(ncid,'n_netcdf',vid)  
            call ncvpt_int2(ncid,vid,1,1,n_netcdf,istatus)
            
            istatus= NF_INQ_VARID(ncid,'ll_netcdf',vid)  
            call ncvpt_int2(ncid,vid,1,n_netcdf,ll_netcdf,istatus)
            
            istatus= NF_INQ_VARID(ncid,'jpxy',vid)  
            call ncvpt_int2(ncid,vid,1,1,jpxy,istatus)
            
            istatus= NF_INQ_VARID(ncid,'ipxy',vid)  
            call ncvpt_int2(ncid,vid,1,1,ipxy,istatus)
            
            istatus= NF_INQ_VARID(ncid,'xll',vid)  
            call ncvpt_doubl2(ncid,vid,1,1,xll,istatus)
            
            istatus= NF_INQ_VARID(ncid,'xlu',vid)  
            call ncvpt_doubl2(ncid,vid,1,1,xlu,istatus)
            
            istatus= NF_INQ_VARID(ncid,'xpl',vid)  
            call ncvpt_doubl2(ncid,vid,1,1,xpl,istatus)
            
            istatus= NF_INQ_VARID(ncid,'xpu',vid)  
            call ncvpt_doubl2(ncid,vid,1,1,xpu,istatus)
            
            istatus= NF_INQ_VARID(ncid,'xpar',vid)  
            call ncvpt_doubl2(ncid,vid,1,jpxy,xpar,istatus)
            
            istatus= NF_INQ_VARID(ncid,'xperp',vid)  
            call ncvpt_doubl2(ncid,vid,1,ipxy,xperp,istatus)
            
            istatus= NF_INQ_VARID(ncid,'rhomax',vid)  
            call ncvpt_doubl2(ncid,vid,1,1,rhomax,istatus)
            
            istatus= NF_INQ_VARID(ncid,'rya_netcdf',vid)  
            call ncvpt_doubl2(ncid,vid,1,n_netcdf,rya_netcdf(1),istatus)
            
            
            
c     Write the fluxes:
c     (First, figure out bin number for netcdf file.)
            do ll=1,n_netcdf
               if (l_ .eq. ll_netcdf(ll)) start(3)=ll
            enddo

c sgbaek to commpare the pack21 operation in the u-theta grid case            
c           call pack21(temp5(0:iy+1,0:jx+1),0,iyp1,0,jxp1,
c    &                  wkpack(1:iy*jx),iy,jx) ! for gamma_x
 
            call pack21(xhead(1:jpxy,1:ipxy),1,jpxy,1,ipxy,
     &                  wkpack(1:jpxy*ipxy),jpxy,ipxy)

c sgbaek, I want to see the start and count variables
            istatus= NF_INQ_VARID(ncid,'gamma_par',vid)  
            call ncvpt_doubl2(ncid,vid,start,count,
     &                      wkpack(1:jpxy*ipxy),istatus)
!            WRITE(*,*) 'sgbaek start', start
!            WRITE(*,*) 'sgbaek count', count
!            WRITE(*,*) 'sgbaek ncid',ncid
!            WRITE(*,*) 'sgbaek vid', vid
           
!            WRITE(*,*) 'sgbaek l_', l_
!            WRITE(*,*) 'sgbaek pack21 xhead size',
!     &                      size(xhead(1:jpxy,1:ipxy)),size(xhead,dim=1)
!            WRITE(*,*) 'sgbaek size wkpack(1:jpxy*ipxy), wkpack(:)', 
!     &                     size(wkpack(1:jpxy*ipxy)),size(wkpack)
!            WRITE(*,*) 'sgbaek xhead(:,1)',xhead(:,1)
!            WRITE(*,*) 'sgbaek wkpack(1:jpxy), wkpack=',
!     &                    wkpack(1:jpxy), wkpack
            
     
            call pack21(yhead(1:jpxy,1:ipxy),1,jpxy,1,ipxy,
     &                  wkpack(1:jpxy*ipxy),jpxy,ipxy)

            istatus= NF_INQ_VARID(ncid,'gamma_perp',vid)  
            call ncvpt_doubl2(ncid,vid,start,count,wkpack,istatus)
!            WRITE(*,*) 'sgbaek pack21 yhead',size(wkpack)


        



            else   ! igrid.ne.0 case



            istatus= NF_INQ_VARID(ncid,'mnemonic',vid)  
            ll=length_char(mnemonic)
            call ncvptc2(ncid,vid,1,ll,mnemonic,ll,istatus)

            istatus= NF_INQ_VARID(ncid,'grid_type',vid)  
            call ncvptc2(ncid,vid,1,7,'x-theta',7,istatus)

            istatus= NF_INQ_VARID(ncid,'lrz',vid)  
            call ncvpt_int2(ncid,vid,1,1,lrz,istatus)
            
            istatus= NF_INQ_VARID(ncid,'n_netcdf',vid)  
            call ncvpt_int2(ncid,vid,1,1,n_netcdf,istatus)
            
            istatus= NF_INQ_VARID(ncid,'ll_netcdf',vid)  
            call ncvpt_int2(ncid,vid,1,n_netcdf,ll_netcdf,istatus)
            
            istatus= NF_INQ_VARID(ncid,'jx',vid)  
            call ncvpt_int2(ncid,vid,1,1,jx,istatus)
            
            istatus= NF_INQ_VARID(ncid,'x',vid)  
            call ncvpt_doubl2(ncid,vid,1,jx,x,istatus)
            
            istatus= NF_INQ_VARID(ncid,'vnorm',vid)  
            call ncvpt_doubl2(ncid,vid,1,1,vnorm,istatus)
            
            istatus= NF_INQ_VARID(ncid,'enorm',vid)  
            call ncvpt_doubl2(ncid,vid,1,1,enorm,istatus)
            
            istatus= NF_INQ_VARID(ncid,'iy',vid)  
            call ncvpt_int2(ncid,vid,1,1,iymax,istatus)
             
            istatus= NF_INQ_VARID(ncid,'itl_netcdf',vid)  
            call ncvpt_int2(ncid,vid,1,n_netcdf,itl_netcdf(1),istatus)
             
            istatus= NF_INQ_VARID(ncid,'itu_netcdf',vid)  
            call ncvpt_int2(ncid,vid,1,n_netcdf,itu_netcdf(1),istatus)
            
            istatus= NF_INQ_VARID(ncid,'rhomax',vid)  
            call ncvpt_doubl2(ncid,vid,1,1,rhomax,istatus)
             
            istatus= NF_INQ_VARID(ncid,'rya_netcdf',vid)  
            call ncvpt_doubl2(ncid,vid,1,n_netcdf,rya_netcdf(1),istatus)
           
            
c     Write the fluxes (and the pitch angle mesh y):
c     (First, figure out bin number for netcdf file.)
            do ll=1,n_netcdf
               if (l_ .eq. ll_netcdf(ll)) then
                  start(3)=ll
                  y_start(2)=ll
               endif
            enddo
            start(4)=k  !For ngen.ge.2 case.
            
            istatus= NF_INQ_VARID(ncid,'y',vid)  
            call ncvpt_doubl2(ncid,vid,y_start(1:2),y_count(1:2),y(1,l_)
     &       ,istatus)
            
            call pack21(temp5(0:iyp1,0:jx+1),0,iyp1,0,jxp1,
     &                  wkpack(1:iymax*jx),iymax,jx) ! for gamma_x
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)

            istatus= NF_INQ_VARID(ncid,'gamma_x',vid)  
            call ncvpt_doubl2(ncid,vid,start,count,wkpack,istatus)
            
            call pack21(temp4(0:iyp1,0:jx+1),0,iyp1,0,jxp1,
     &                  wkpack(1:iymax*jx),iymax,jx) ! for gamma_theta
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)
            istatus= NF_INQ_VARID(ncid,'gamma_theta',vid)  
            call ncvpt_doubl2(ncid,vid,start,count,wkpack,istatus)
            
            endif   !end igrid options

            istatus = NF_CLOSE(ncid) !-YuP: NetCDF-f77
            
         endif
         
 190  continue  ! on k=1,ngen
      
      return
      end subroutine netcdfvec
c     
c     
      integer function length_char(string)
c     Returns length of string, ignoring trailing blanks,
c     using the fortran intrinsic len().
      character*(*) string
      do i=len(string),1,-1
         if(string(i:i) .ne. ' ') goto 20
      enddo
 20   length_char=i
      return
      end function length_char






C=YuP=> ADDED: conversion from Netcdf-2 to Netcdf-3 or higher ==========
C These routines/function convert old routines:

      integer function ncvdef2(NCID,VARNAM,VARTYP,NDIMS,VDIMS,istatus)
      ! vid=NCVDEF() is renamed to vid=ncvdef2() in *.f files
      INCLUDE 'netcdf.inc'
      CHARACTER*(*) VARNAM
      INTEGER istatus,vid,NCID, VARTYP,XTYPE, NDIMS,VDIMS(*)
      if (VARTYP.eq.NCFLOAT)  XTYPE=NF_FLOAT    ! 32 BITS
      if (VARTYP.eq.NCDOUBLE) XTYPE=NF_DOUBLE   ! 64 BITS
      if (VARTYP.eq.NCCHAR)   XTYPE=NF_CHAR
      if (VARTYP.eq.NCBYTE)   XTYPE=NF_BYTE
      if (VARTYP.eq.NCSHORT)  XTYPE=NF_SHORT    ! 16 BITS
      if (VARTYP.eq.NCLONG)   XTYPE=NF_INT      ! 32 BITS
      istatus = NF_DEF_VAR(NCID,VARNAM,XTYPE,NDIMS,VDIMS,vid)
      ncvdef2 = vid
      end
      
      subroutine ncaptc2(NCID, vid, NAME, ATTYPE, LEN, TEXT, istatus)
      INCLUDE 'netcdf.inc'
      CHARACTER*(*) NAME
      CHARACTER*(*) TEXT
      INTEGER istatus,vid,NCID, LEN, ATTYPE
      istatus = NF_PUT_ATT_TEXT(NCID, vid, NAME, LEN, TEXT)
      return 
      end
        
      subroutine ncvptc2(NCID, vid, START, COUNTS, TEXT, LEN, istatus)
      INCLUDE 'netcdf.inc'
      CHARACTER*(*) TEXT
      INTEGER istatus,vid,NCID, LEN, START(*), COUNTS(*)
      istatus = NF_PUT_VARA_TEXT(NCID, vid, START, COUNTS, TEXT)
      return 
      end

      subroutine ncvpt_doubl2(NCID, vid, START, COUNTS,  DVALS, istatus)
      INCLUDE 'netcdf.inc'
      INTEGER istatus,vid,NCID, START(*), COUNTS(*)
      REAL*8 DVALS(*)
      istatus=NF_PUT_VARA_DOUBLE(NCID, vid, START, COUNTS, DVALS)
      return 
      end

      subroutine ncvpt_int2(NCID, vid, START, COUNTS,  IVALS, istatus)
      INCLUDE 'netcdf.inc'
      INTEGER istatus,vid,NCID, START(*), COUNTS(*)
      INTEGER IVALS(*)
      istatus=NF_PUT_VARA_INT(NCID, vid, START, COUNTS, IVALS)
      return 
      end
      
      
c      subroutine NCPOPT  
       !!!-YuP: call NCPOPT() commented in files
       ! No similar routine in NetCDF-3


c     OTHER NF_PUT_*** routines available in NetCDF-3 or higher:

c     VAR entire variable access:
c     INTEGER FUNCTION  NF_PUT_VAR_DOUBLE(NCID, VARID, DVAL)

c     VAR1 single value access:
c     INTEGER FUNCTION  NF_PUT_VAR1_INT(NCID, VARID, INDEX, IVAL)

c     VARA array or array section access:
c     INTEGER FUNCTION  NF_PUT_VARA_INT(NCID, VARID, START, COUNT, IVALS)

c     VARS strided access to a subsample of values:
c     INTEGER FUNCTION  NF_PUT_VARS_INT(NCID, VARID, START, COUNT, STRIDE, IVALS)

c     VARM mapped access to values not contiguous in memory:
c     INTEGER FUNCTION  NF_PUT_VARM_INT(NCID, VARID, START, COUNT, STRIDE, IMAP, IVALS)
 

c======================================================================
c======================================================================

      subroutine f4dwrite
      implicit integer (i-n), real*8 (a-h,o-z)
      save
      include 'param.h'
      include 'comm.h'

CMPIINSERT_INCLUDE

c     Write 4D distribution, f(v,pitch,R,Z) to netCDF file.
c     First use of this file is in the FIDA syntheic diagnostic.
c     Work with Deyong Liu, Bill Heidbrink.
c     BH120328

c     nr_f4d,nz_f4d are R,Z grid dimensions set in namelist
c     nv_f4d,nt_f4d are normalized vel and pitch angle grids.
c     Grids, at this time, are taken to be equispaced.

c     The distribution function f4d, and the grids will be allocated
c     according to the namelist input, and will be deallocated at
c     the end of the subroutine.

      real*8,dimension(:),allocatable:: f4dr,f4dz,f4dv,f4dt,f4ddv,f4ddt
      real*8,dimension(:,:,:,:),allocatable:: f4d

CMPIINSERT_IF_RANK_NE_0_RETURN

      WRITE(*,*)
      WRITE(*,*)'Entering subroutine f4dwrite'

c     Only set up for one (the first) general species:
      if (ngen.gt.1) then
         WRITE(*,*)
         WRITE(*,*)'WARNING: Output f4d dist ftn for 1st gen spec only'
         WRITE(*,*)
         k=1 ! First general species only, for now
      endif

c.......................................................................
c     Allocate temporary storage, to be deallocated at end of subroutine
c.......................................................................
      allocate(f4dr(nr_f4d),STAT=istat1)
      allocate(f4dz(nz_f4d),STAT=istat2)
      allocate(f4dv(nv_f4d),STAT=istat3)
      allocate(f4dt(nt_f4d),STAT=istat4)
      allocate(f4ddv(nv_f4d),STAT=istat5)
      allocate(f4ddt(nt_f4d),STAT=istat6)
      allocate(f4d(nr_f4d,nz_f4d,nv_f4d,nt_f4d),STAT=istat7)
      if (istat1.ne.0 .or. istat2.ne.0 .or. istat3.ne.0 .or. 
     +    istat4.ne.0 .or. istat5.ne.0 .or.
     +    istat6.ne.0 .or. istat7.ne.0) then
         WRITE(*,*)'f4dwrite: allocation problem'
         STOP
      endif

c.......................................................................
c     Form grids:  R,Z grids fitting plasma; v in [0,1], pitch in [0,pi]
c     See eqrhopsi of calc of rmaxcon,rmincon,zmaxcon,zmincon, slightly
c     inside the LCFS.
c     Future FOW calcs can move grid out to input the chamber wall.
c.......................................................................

      dr=(rmaxcon-rmincon)/(nr_f4d-1)
      dz_=(zmaxcon-zmincon)/(nz_f4d-1)
      dv=one/(nv_f4d-1)
      dt=pi/(nt_f4d-1)

      do ir=1,nr_f4d
         f4dr(ir)=rmincon+(ir-1)*dr
      enddo
      do iz=1,nz_f4d
         f4dz(iz)=zmincon+(iz-1)*dz_
      enddo
      do iv=1,nv_f4d
         f4dv(iv)=(iv-1)*dv
      enddo
      do it=1,nt_f4d
         f4dt(it)=(it-1)*dt
      enddo

      call bcast(f4d,zero,nr_f4d*nz_f4d*nv_f4d*nt_f4d)

c.......................................................................
c     Form volume elements in velocity space
c.......................................................................

      f4ddv(1)=f4dv(2)**3/24.
      do iv=2,nv_f4d
         f4ddv(iv)=f4dv(iv)**2*dv  !v=0 vol element.....
      enddo
      f4ddt(1)=0.25*pi*f4dt(2)**2
      f4ddt(nt_f4d)=f4ddt(1)
      do it=2,nt_f4d-1
         f4ddt(it)=twopi*sin(f4dt(it))*dt
      enddo


c.......................................................................
c     For each R,Z, find rhoin,polang for use with tdfinterp.
c     Coding is similar to that in tdnpa0.
c     Only set up for eqsource="eqdsk".  Easy to generalize.
c.......................................................................

      if (eqsource.ne."eqdsk") then
         WRITE(*,*)
         WRITE(*,*) 'f4dwrite only set up for eqsource=eqdsk'
         WRITE(*,*) 'Easy to generalize, if useful'
         STOP
      endif


      if(fow.eq.'disabled')then
      
c        Change sign of eqpsi, to get asceding order necessary
c        for coeff1.  (Remember this when using the spline coeffs.)
         do j=1,nconteqn
            eqpsi(j)=-eqpsi(j)
         enddo
         itab(1)=1 ! can check rho value in tab(1) with debugger.
         itab(2)=0
         itab(3)=0
         do ir=1,nr_f4d
            rr=f4dr(ir)
            do iz=1,nz_f4d
               zz=f4dz(iz)
               !epsi(,) has convention of being max at mag axis.
               !eqpsi is sign reversed from this, giving min a mag axis.
               !Therefore, ppsi value is sign reversed here.
               ppsi=-terp2(rr,zz,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1         epsirz,nnra,0,0)
               !(Check sign of ppsi and eqpsi with debugger)
               call terp1(nconteqn,eqpsi,eqrho,d2eqrho,ppsi,1,tab,itab)
               rhoin=tab(1)/rhomax
c              If rhoin.gt.1, point is outside the LCFS.  Leave f4d=0.
               !  Poloidal angle (rmag is major radius of magnetic axis).
               if (rhoin.lt.one) then 
                  arg1=zz
                  arg2=rr-rmag
                  if(arg1.eq.zero .and. arg2.eq.zero)  then
                     polang=0.
                  else
                     polang=atan2(arg1,arg2)
                     if (polang.lt.zero) polang=polang+twopi
                  endif
                !WRITE(*,'(a,3e12.3)')'f4dwrite : rr,zz,rhoin=',rr,zz,rhoin
                  do iv=1,nv_f4d
                     vn=f4dv(iv)*vnorm
                     do it=1,nt_f4d
                        pitch=f4dt(it)
                        call tdfinterp(vn,pitch,rhoin,polang,
     +                              f4d(ir,iz,iv,it)) !,tau_b)
                     enddo
                  enddo
               endif  ! On rhoin
            enddo ! iz
            !pause
         enddo ! ir
      
      elseif(fow.eq.'hybrid' .or. fow.eq.'full')then 
! Not applicable in this version
!         do ir=1,nr_f4d
!            rloc=f4dr(ir)  ! local R
!         do iz=1,nz_f4d
!            zloc=f4dz(iz)  ! local Z
!            do iv=1,nv_f4d
!               vn=f4dv(iv)*vnorm ! local v
!            do it=1,nt_f4d 
!               pitch=f4dt(it)    ! local pitch
!               call fow_tdfinterp(k,vn,pitch,rloc,zloc,f4d(ir,iz,iv,it))
!            enddo ! it
!            enddo ! iv
!         enddo ! iz
!         enddo ! ir
         STOP 'f4dwrite: only setup for fow=disabled'
      else
         STOP 'f4dwrite: only setup for fow=disabled'
      endif ! fow




c.......................................................................

c  Some printing:
c$$$      do ir=1,nr_f4d
c$$$         iz=33
c$$$         WRITE(*,*) 'f4d(ir,iz,1,:), ir,iz=', ir,iz 
c$$$         WRITE(*,100) (f4d(ir,iz,1,i),i=1,nt_f4d)
c$$$         WRITE(*,*) 'f4d(ir,iz,2,:), ir,iz=', ir,iz 
c$$$         WRITE(*,100) (f4d(ir,iz,2,i),i=1,nt_f4d)
c$$$         WRITE(*,*) 'f4d(ir,iz,:,1), ir,iz=', ir,iz 
c$$$         WRITE(*,100) (f4d(ir,iz,j,1),j=1,nv_f4d)
c$$$         WRITE(*,*) 'f4d(ir,iz,:,nt_f4d), ir,iz=', ir,iz 
c$$$         WRITE(*,100) (f4d(ir,iz,j,nt_f4d),j=1,nv_f4d)
c$$$         iz=40
c$$$         WRITE(*,*) 'f4d(ir,iz,1,:), ir,iz=', ir,iz 
c$$$         WRITE(*,100) (f4d(ir,iz,1,i),i=1,nt_f4d)
c$$$         WRITE(*,*) 'f4d(ir,iz,2,:), ir,iz=', ir,iz 
c$$$         WRITE(*,100) (f4d(ir,iz,2,i),i=1,nt_f4d)
c$$$         WRITE(*,*) 'f4d(ir,iz,:,1), ir,iz=', ir,iz 
c$$$         WRITE(*,100) (f4d(ir,iz,j,1),j=1,nv_f4d)
c$$$         WRITE(*,*) 'f4d(ir,iz,:,nt_f4d), ir,iz=', ir,iz 
c$$$         WRITE(*,100) (f4d(ir,iz,j,nt_f4d),j=1,nv_f4d)
c$$$      enddo  ! On ir
c$$$ 100  format(10ES10.2)


c  Fix some odd values:  Need to check tdfinterp
      do ir=1,nr_f4d
         do iz=1,nz_f4d
            iv=1
            avg=zero
            do it=1,nt_f4d
               avg=avg+f4d(ir,iz,iv+1,it)
            enddo  ! On it
            avg=avg/nt_f4d
            do it=1,nt_f4d
               f4d(ir,iz,iv,it)=avg
            enddo  ! On it
            iv=nv_f4d
            it=nt_f4d
            f4d(ir,iz,iv,it)=f4d(ir,iz,iv,it-1)  !Might not be needed
         enddo  ! On iz
      enddo  ! On ir

c$$$      ir=20
c$$$      iz=33
c$$$      it=nt_f4d
c$$$      WRITE(*,*)'f4d(ir,iz,:,it-1):',(f4d(ir,iz,j,it-1),j=1,nv_f4d)
c$$$      WRITE(*,*)'f4d(ir,iz,:,it):',(f4d(ir,iz,j,it),j=1,nv_f4d)

c.......................................................................



c     Change back eqpsi sign.
      do j=1,nconteqn
         eqpsi(j)=-eqpsi(j)
      enddo

      if (eqsym.eq."none") then
         WRITE(*,*)
         WRITE(*,*)'************************************************'
         WRITE(*,*)'WARNING:  NPA CALC ASSUMING UP-DOWN SYMM'
         WRITE(*,*)'          and eqsym=none.  Need to check coding'
         WRITE(*,*)'************************************************'
         WRITE(*,*)
      endif

c.......................................................................
c     Write data into netcdf file f4d.nc
c.......................................................................

      call ncwritef4d(f4dr,f4dz,f4dv,f4dt,f4ddv,f4ddt,f4d)

      deallocate(f4dr,f4dz,f4dv,f4dt,f4ddv,f4ddt,f4d)

      return

      end



c======================================================================
c======================================================================

      subroutine f3dwrite
      implicit integer (i-n), real*8 (a-h,o-z)
      save
      include 'param.h'
      include 'comm.h'

CMPIINSERT_INCLUDE

c     Write 3D distribution, f(pol.angle,vpar,mu) at fixed rho to data file.
c     Original purpose: To be used by COGENT, as a boundary condition 
c     at specified rho (flux surface).
c     Work with Ron Cohen, Mikhail Dorf
c     YuP August 2013 [based on f4dwrite, BH]

c     npol_f3d is poloidal angle grid size set in namelist
c     nvpar_f3d, nmu_f3d are (vpar,mu) grid sizes.
c     All grids are equispaced.

c     The distribution function f3d, and the grids will be allocated
c     according to the namelist input, and will be deallocated at
c     the end of the subroutine.

      real*8,dimension(:),allocatable:: f3d_pol,f3d_b, f3d_vpar,f3d_mu
      real*8,dimension(:,:,:),allocatable:: f3d, f3d_tau

CMPIINSERT_IF_RANK_NE_0_RETURN

      WRITE(*,*)
      WRITE(*,*)'Entering subroutine f3dwrite'

c     Only set up for one (the first) general species:
      if (ngen.gt.1) then
         WRITE(*,*)
         WRITE(*,*)'WARNING: Output f3d dist ftn for 1st gen spec only'
         WRITE(*,*)
         k=1 ! First general species only, for now
      endif

c.......................................................................
c     Allocate temporary storage, to be deallocated at end of subroutine
c.......................................................................
      allocate(f3d_pol(npol_f3d)) ! Pol.angle grid
      allocate(f3d_b(npol_f3d))  ! Magnetic field at pol.angle grid points
      allocate(f3d_vpar(nvpar_f3d)) ! Vpar
      allocate(f3d_mu(nmu_f3d))  ! 0.5*m*Vper^2/B 
      allocate(f3d(npol_f3d, nvpar_f3d, nmu_f3d),STAT=istat)
      if (istat.ne.0) then
         WRITE(*,*)'f3dwrite: allocation problem'
         STOP
      endif
      allocate(f3d_tau(npol_f3d, nvpar_f3d, nmu_f3d),STAT=istat)
      if (istat.ne.0) then
         WRITE(*,*)'f3dwrite: allocation problem'
         STOP
      endif
      
      proton_kg= 1.672621777e-27   ! [kg]
      e_C=       1.602176565e-19   ! [C]
      epsvac=    8.854187817e-12   ! [F/m]
      
      ! Define reference units - to be used by COGENT. 
      ! See GKModel.pdf pp.6-7
      units_n= 1.0e20 ! [m^-3] n_tilda == units.number_density
      units_T= 1.0e3  ! [eV]   T_tilda == units.temperature
      units_L= 1.0    ! [m]    L_tilda == units.length
      units_m= 1.0    ! [amu]  m_tilda == units.mass
      units_B= 1.0    ! [T]    B_tilda == units.magnetic_field
      
      ! Derived reference values - for cross-check 
      !(these values are printed out by COGENT into a log file)
      ref_v=sqrt(units_T*e_C/(units_m*proton_kg)) ![m/s] sqrt(T_tilda/m_tilda)
      ref_tau=units_L/ref_v               ![sec]         Transit time
      ref_mu= units_T*e_C/(2*units_B)     ![Joule/Tesla] Magn.moment
      ref_distfunc= units_n/(pi*ref_v**3) ![s^3/m^6]  Distribution function 
      ref_potential=units_T               ![V] Electric potential
      ref_gyrofreq=e_C*units_B/(units_m*proton_kg)  ![1/sec] Gyro-frequency
      ref_debyelen=sqrt(epsvac*units_T/(units_n*e_C)) ![m] Debye length
      ref_psi= units_B*(units_L)**2       ! [Wb] Poloidal flux
      
c      if (eqsource.eq."miller") then
      !......................... local/input for Miller equilibrium:
      delta_e=eq_miller_deltaedge ! Triangularity of LCFS (at r=radmin)
      akappa= eq_miller_kappa  ! Vertical elongation (const for all surfaces)
      drr0=   eq_miller_drr0   ! dR0/dr  See REF: describes Shafranov shift
      fpsieq= btor*radmaj      ! B*R assumed constant (not dep. on r)
      !......................... 
c      endif

      rhoin=f3d_rho ! Values of f are evaluated at this rho (coupling rho)
      

      !===> Form grids:  
      dpol=    twopi/(npol_f3d-1)
      v_parallel_max= vnorm
      dvpar= 2*v_parallel_max/(nvpar_f3d-1)
      do ipol=1,npol_f3d
         f3d_pol(ipol)= (ipol-1)*dpol ! [0;2*pi]
      enddo
      do iv=1,nvpar_f3d
         f3d_vpar(iv)= -vnorm +(iv-1)*dvpar ! [-vnorm; +vnorm]  (cm/sec)
      enddo

      ! Find the Nearest flux surf. ir-index:
      !Careful with use of lookup_tdf:  last argument is index
      !of the array element just ABOVE the first argument. 
      call lookup_tdf(rhoin,rovera,lrzmax,rweightr,rweightl,ir)
      ir=ir-1 ! now, rhoin is between rovera(ir) and rovera(ir+1)
      ! rweightr,rweightl are weight coeffs for interpolation.
      
      !===> First, Find min(B) at this rhoin, and form mu-grid
      polang=0.d0 ! Midplane, outer-most point (largest R)
      ip=1 ! Nearest index in cql3d-poloidal grid "pol"
      !-> Find the local (R,Z)-coord by interpolation:
      rloc= rweightl*solrz(ip,ir)+rweightr*solrz(ip,ir+1)
      zloc= rweightl*solzz(ip,ir)+rweightr*solzz(ip,ir+1)
      ! For a given (rloc,zloc), determine components of B:
              ! Convert (R,Z)->(r,polang)
              call eq_miller_RZ_to_rpol(rmag,zmag, delta_e,akappa,drr0,
     +             rloc,zloc, rminor,polang) 
              ! Get PSI, BR, BPHI, BZ values:
              call eq_miller(rminor, polang, 
     +             rmag, zmag, cursign, bsign, radmin, psimag,psilim,
     +             delta_e, akappa, drr0, fpsieq,
     +             dpsidr, Rs,Zs, PSI_tmp, BR0, BPHI0, BZ0)
      Rmax=rloc  ! major radius, outermost point at this flux surface 
      Bmin=sqrt(BR0**2+BZ0**2+BPHI0**2)  !-> |B| found
      ! Define max(mu)= 0.5*m*Vmax^2/Bmin and grid step dmu at this rhoin:
      amu_max= 0.5*fmass(k)*vnorm2/Bmin
      dmu= amu_max/(nmu_f3d-1)
      ! Also get the value of dpsi/dr (physics value) at given r:
      dpsidr0= -cursign*abs(dpsidr) !When cursign<0 -> physics dpsi/dr>0

      ! Form grid for mu:
      do imu=1,nmu_f3d
         f3d_mu(imu)= (imu-1)*dmu ! [0; 0.5*fmass(k)*vnorm2/Bmin]  (cgs)
      enddo
      
      !===> Find values of magnetic field at pol.angle grid points.
      Rmin=Rmax ! Rmin will be found (innermost major radius)
      do ipol=1,npol_f3d
         polang=f3d_pol(ipol) 
         ! polang is NOT the theta angle as in Miller definitions of 
         ! flux surfaces. The Miller theta is polang_miller below.
         ! But: for COGENT, we need such f3d_pol, as in 
         ! polrz= atan2(zloc-zmag,rloc-rmag), see below.
         ! In COGENT, the polang will be defined as polrz, 
         ! to find the nearest grid point and to get the distr.function.
         zzmag=1.d0 ! changed to -1 when eqsym.ne."none" and polang>pi
         if (eqsym.ne."none" .and. polang.gt.pi) then
           polang=2*pi-polang   !assumes up-down symm
           zzmag=-1.d0 ! sign of (z-zmag) for this point
           ! The grids pol(l,lr),solrr(l,lr),solzz(l,lr)
           ! are defined only in one half of poloidal plane.
           ! The zzmag value indicates - which half.
         endif
         ! Find the Nearest index in cql3d-poloidal grid "pol":
         call lookup_tdf(polang,pol(1,ir),lz,pweightr,pweightl,ip)
         ip=ip-1 ! now, polang is between pol(ip,ir) and pol(ip+1,ir)
         !-> Find the local (R,Z)-coord by interpolation:
         rloc= 
     +      rweightl*(pweightl*solrz(ip,ir)  +pweightr*solrz(ip+1,ir))
     +     +rweightr*(pweightl*solrz(ip,ir+1)+pweightr*solrz(ip+1,ir+1))
         zloc= 
     +      rweightl*(pweightl*solzz(ip,ir)  +pweightr*solzz(ip+1,ir))
     +     +rweightr*(pweightl*solzz(ip,ir+1)+pweightr*solzz(ip+1,ir+1))
         zloc=zloc*zzmag ! zzmag=-1 when eqsym.ne."none" and polang>pi
         polrz= atan2(zloc-zmag,rloc-rmag)
         if(polrz.lt.0.d0) polrz=polrz+twopi
         Rmin=min(Rmin,rloc)
         ! For a given (rloc,zloc), determine components of B:
c         if (eqsource.eq."miller") then
               ! Convert (R,Z)->(r,polang_miller)
               call eq_miller_RZ_to_rpol(rmag,zmag, delta_e,akappa,drr0,
     +              rloc,zloc, rminor,polang_miller) 
               ! Get PSI, BR, BPHI, BZ values:
               call eq_miller(rminor, polang_miller, 
     +              rmag, zmag, cursign, bsign, radmin, psimag,psilim,
     +              delta_e, akappa, drr0, fpsieq,
     +              dpsidr, Rs,Zs, PSI_tmp, BR0, BPHI0, BZ0)
c         endif          
         if (eqsource.eq."eqdsk"   .or.
     +           eqsource.eq."topeol"  .or.
     +           eqsource.eq."tsc"   ) then
              if(eqmod.eq."enabled")then
               call equilib(rloc,zloc,1,PSI_tmpe,BRe,BPHIe,BZe)
               ! Call with index=1 for PSI,BR,BPHI,BZ return (cgs units)           
               !                   for a given Requil,Zequil coordinate.
              else ! eqmod.eq."disabled" (circular-shaped plasma eq. model)
                STOP 'f3dwrite: not set for eqmod.eq."disabled" '
              endif
      WRITE(*,'(a,2f12.3,7f11.3)')
     +'R,Z,polrz,f3d_pol,pol_miller,  BR0,BRe,BZ0,BZe',
     + rloc,zloc, polrz*180/pi, polang*180/pi, polang_miller*180/pi,
     + BR0,BRe,BZ0,BZe
c         else
c               WRITE(*,*)'fow_setb: wrong eqsource? eqsource=',eqsource
c               stop
         endif
         f3d_b(ipol)=sqrt(BR0**2+BZ0**2+BPHI0**2)  !-> |B| found
      enddo ! ipol
      !---> COUPLING minor radius (for coupling of CQL3D with COGENT):
      rcoupling= 0.5*(Rmax-Rmin)
      ! Also evaluate triangularity delta at r=rcoupling:
      delta_c= delta_e*(rcoupling/radmin) !Be sure it's same as in sub.eq_miller
      s_delta= delta_c/sqrt(1.d0-delta_c**2)
      ! Characteristic of change of elongation:
      s_kappa= 0.d0 !always 0 (kappa=const in r), see sub.eq_miller
      !-----------------------------------
      WRITE(*,*)'f3d_rho,rcoupling=',f3d_rho,rcoupling
      WRITE(*,*)'radmaj,fpsieq,dpsidr0=',radmaj,fpsieq,dpsidr0
      !pause
      

      !===> Find and save distribution function at given rhoin.
      call bcast(f3d, zero, npol_f3d*nvpar_f3d*nmu_f3d) ! initialize
      ! tau_bounce [sec]
      call bcast(f3d_tau, zero, npol_f3d*nvpar_f3d*nmu_f3d) ! initialize

c.......................................................................
c     For a given f3d_rho, and each pol.angle, use tdfinterp to find f.
c     Coding is similar to that in tdnpa0.
c     Only set up for eqsource="eqdsk".  Easy to generalize.
c.......................................................................

      if (eqsource.ne."eqdsk") then
         WRITE(*,*)
         WRITE(*,*) 'f3dwrite only set up for eqsource=eqdsk'
         WRITE(*,*) 'Easy to generalize, if useful'
         STOP
      endif

      if(fow.eq.'disabled')then ! .or. fow.eq.'hybrid')then
         
         ! For tests only:
         fpeak= f(2,1,1,ir) ! Peak value of distr.func, for reference.
         dvpeak=0.01 ! Width of a peak in vel.space, norm-ed by vnorm
         ! end test
         do ipol=1,npol_f3d
            polang=f3d_pol(ipol)
            bmag=f3d_b(ipol)  ! |B| [gauss]
            do ivpar=1,nvpar_f3d
            v_par= f3d_vpar(ivpar)*bsign !to make Vpar=(V.b), for COGENT
            !NOTE: v_par (or cospitch) definition is different in COGENT!
            v_par2= v_par*v_par
            do imu=1,nmu_f3d ! over adiabatic inv. 0.5*m*Vper^2/B [cgs]
               umu= f3d_mu(imu)
               v_per2= umu*2.0*bmag/fmass(k)
               v_per=sqrt(v_per2)
               vn=sqrt(v_par2+v_per2) !momentum-per-rest-mass (cm/sec)
               cospitch= v_par/vn
               cospitch=max(cospitch,-1.d0) ! to be sure cos>=-1
               cospitch=min(cospitch,+1.d0) ! to be sure cos<=+1
               pitch=acos(cospitch) !acos() is in [0,pi]
               polang=f3d_pol(ipol)
               call tdfinterp(vn,pitch,rhoin,polang,
     +              f3d(ipol,ivpar,imu)) !, f3d_tau(ipol,ivpar,imu) )
               ! For tests only:
               polang=f3d_pol(ipol)
               if (abs(polang -280.*pi/180.).lt.(dpol*0.5)
     +            .and. abs(pitch-pi*0.45).lt.0.02 ) then
                   WRITE(*,*)'ipol,polang=', ipol,polang*180/pi
                  ! zero every pol.angle, except selected point (ipol=..)
                  ! In vel.space, set it as a ring distribution at high v
                  ! (no thermal part)
                  fsource=fpeak*exp( -((vn/vnorm -0.78)/dvpeak)**2 )
                  if (fsource.ge.fpeak*1e-8 )then
                    call tdfinterp(vn,pitch,rhoin,polang,
     +                   f3d(ipol,ivpar,imu)) !, f3d_tau(ipol,ivpar,imu) )
                    polang=f3d_pol(ipol)
                  else
                    f3d_tau(ipol,ivpar,imu)= 0.d0
                  endif
                  f3d(ipol,ivpar,imu)= fsource
               else
                  f3d(ipol,ivpar,imu)= 0.d0
                  f3d_tau(ipol,ivpar,imu)= 0.d0
               endif
               ! end test
            enddo
            enddo
         enddo ! ipol
             
      else
         STOP 'f3dwrite: only setup for fow=disabled' ! or hybrid'
      endif ! fow

      if (eqsym.eq."none") then
         WRITE(*,*)
         WRITE(*,*)'************************************************'
         WRITE(*,*)'WARNING:  f3dwrite ASSUMING UP-DOWN SYMM'
         WRITE(*,*)'          and eqsym=none.  Need to check coding'
         WRITE(*,*)'************************************************'
         WRITE(*,*)
      endif

c.......................................................................
c     Write data into a file - normalization/units data for COGENT:
      if(f3d_format.eq.'ascii') then
        iunit=33
        open(unit=iunit,file='cql3d_cogent_units.txt',STATUS='UNKNOWN')
        write(iunit,'(A)')'--- Normalization/units data for COGENT ---'
        write(iunit,'(A,1pe16.7)')
     +   'units_n== units.number_density [m^-3]=',units_n
        write(iunit,'(A,1pe16.7)')
     +   'units_T== units.temperature    [eV]  =',units_T
        write(iunit,'(A,1pe16.7)')
     +   'units_L== units.length         [m]   =',units_L
        write(iunit,'(A,1pe16.7)')
     +   'units_m== units.mass           [amu] =',units_m
        write(iunit,'(A,1pe16.7)')
     +   'units_B== units.magnetic_field [T]   =',units_B
        write(iunit,'(A)')'  '
        write(iunit,'(A)')'--- Derived reference values --------------'
        
        write(iunit,'(A,1pe16.7)')
     +   'ref_v [m/s] = sqrt(units_T*e[C]/(units_m*proton[kg])) =',ref_v

        write(iunit,'(A,1pe16.7)')
     +   'Transit time [sec] = units_L/ref_v   =', ref_tau

        write(iunit,'(A,1pe16.7)')
     +   'Magn.moment [Joule/Tesla] = units_T*e[C]/(2*units_B) =',ref_mu

        write(iunit,'(A,1pe16.7)')
     +   'Distr.func.[sec^3/m^6] = units_n/(pi*ref_v**3)=',ref_distfunc

         write(iunit,'(A,1pe16.7)')
     +   'Electric potential [V] = units_T=', ref_potential
     
        write(iunit,'(A,1pe16.7)')
     +   'Gyro-frequency [1/sec] = e[C]*units_B/(units_m*proton[kg])=',
     +   ref_gyrofreq
     
        write(iunit,'(A,1pe16.7)')
     +   'Gyro-radius [m] = ref_v/ref_gyrofreq=',
     +   ref_v/ref_gyrofreq

        write(iunit,'(A,1pe16.7)')
     +   'Debye length [m] = sqrt(eps0*units_T/(units_n*e[C]))=',
     +   ref_debyelen

        write(iunit,'(A,1pe16.7)')
     +   'Pol.flux [Wb] = units_B*(units_L)^2 =',ref_psi

        write(iunit,'(A)')'  '
        write(iunit,'(A)')'--- CONSTANTS USED IN THE ABOVE ------------'
        write(iunit,'(A)')'proton[kg]= 1.672621777e-27'
        write(iunit,'(A)')'e[C]=       1.602176565e-19'
        write(iunit,'(A)')'eps0[F/m]=  8.854187817e-12'
        write(iunit,'(A)')'  '
        write(iunit,'(A)')'--- INPUT values for Miller equilibrium ----'
        !  Note conversion factors from cgs to mks
        write(iunit,'(A,1pe16.7)')'rmag[m]/units_L =', rmag/100/units_L
        write(iunit,'(A,1pe16.7)')'zmag[m]/units_L =', zmag/100/units_L
        write(iunit,'(A,1pe16.7)')
     +  'major R at Geom.center of LCFS: radmaj[m]/units_L =',
     +   radmaj/100/units_L
        write(iunit,'(A,1pe16.7)')
     +  'minor r at LCFS: radmin[m]/units_L =', radmin/100/units_L
        write(iunit,'(A,1pe16.7)')
     +  'minor r at coupling surface: rcoupling[m]/units_L =',
     +   rcoupling/100/units_L
        write(iunit,'(A)')'  '
        write(iunit,'(A,1pe16.7)')
     +  '|Btor*R|[T*m]/(units_B*units_L) =', 
     +   fpsieq*1.e-6/(units_B*units_L)
        write(iunit,'(A,f3.0)')
     +  'sign of Btor:  bsign=', bsign
        write(iunit,'(A)')'  '
        write(iunit,'(A,1pe16.7)')
     +  '(d(psi)/dr)[Wb/m]/(units_B*units_L) at coupling surface',
     +  dpsidr0/1.e6/(units_B*units_L)
        write(iunit,'(A)')'  '
        write(iunit,'(A,1pe16.7)')
     +  'Pol.flux at mag.axis: psimag[Wb]/ref_psi =',psimag/1.e8/ref_psi
        write(iunit,'(A)')'  '
        write(iunit,'(A,1pe16.7)')
     +  'Pol.flux at LCFS:     psilim[Wb]/ref_psi =',psilim/1.e8/ref_psi
        write(iunit,'(A)')'  '
        write(iunit,'(A,f3.0)')
     +  'sign of plasma current in tor.direction cursign=', cursign
        write(iunit,'(A)')'  '
        write(iunit,'(A,1pe16.7)')'Vertical Elongation kappa=', akappa
        write(iunit,'(A,1pe16.7)')'S_kappa=', s_kappa
        write(iunit,'(A,1pe16.7)')
     +  'Triangularity delta (at r=rcoupling):', delta_c
        write(iunit,'(A,1pe16.7)')'S_delta=', s_delta
        write(iunit,'(A,1pe16.7)')'dR0/dr for Shafranov shift:',drr0
        write(iunit,'(A,1pe16.7)')
     +  'R at geom.center of coupling surf. 
     +   Rc= Rmag+drr0*rcoupling [m]/units_L',
     +   (Rmag+drr0*rcoupling)/100/units_L
        write(iunit,'(A)')'  '
        write(iunit,'(A,1pe16.7)')
     +  'Dist.func at polang=0,vpar=0,mu=0:  
     +   (fcql/vnorm**3)*10^12/ref_distfunc =',
     +   (f3d(1,nvpar_f3d/2,1)/vnorm3)*1.e12/ref_distfunc
        write(iunit,'(A)')'  '
        write(iunit,'(A,1pe16.7)')
     +  'v_parallel_max[m/s]/ref_v:', v_parallel_max/100/ref_v
        write(iunit,'(A)')'  '
        write(iunit,'(A,1pe16.7)')
     +  'mu_max[Joule/Tesla]/ref_mu:', amu_max*1.e-3/ref_mu
        write(iunit,'(A)')'  '

        write(iunit,'(A)')'Species name,type  mass/proton[kg]  Charge 
     +         T[eV]/units_T  n[m^-3]/units_n'
        do k=1,ntotal ! over all species
          ! Local temperature and density:
          temp_loc= rweightl*temp(k,ir) +rweightr*temp(k,ir+1)  ! keV
          den_loc=  rweightl*reden(k,ir)+rweightr*reden(k,ir+1) ! cm^-3
          write(iunit,'(2A, 2(1pe16.7), 2(1pe16.7))')
     +    kspeci(1,k),kspeci(2,k),
     +    fmass(k)*1.e-3/proton_kg/units_m,   bnumb(k),  
     +    temp_loc*1.e3/units_T,    den_loc*1.e6/units_n
        enddo
        
        write(iunit,'(A)')'  '
        close(iunit)
        
      endif

      renorm= (1.e12/vnorm3)/ref_distfunc
      do ipol=1,npol_f3d
      do ivpar=1,nvpar_f3d
      do imu=1,nmu_f3d
       f3d_norm= f3d(ipol,ivpar,imu)*renorm
       f3d_norm= max(f3d_norm,1.d-99)
       f3d(ipol,ivpar,imu)= f3d_norm
      enddo
      enddo
      enddo

c     Write data into a file
      if(f3d_format.eq.'ascii') then
        iunit=33
        open(unit=iunit,file='cql3d_f3d.txt',STATUS='UNKNOWN')
        
        write(iunit,'(A)')'Contains distr.func. calculated by CQL3D:' 
        write(iunit,'(A)')'f(pol.angle,vpar,mu) at fixed rho==f3d_rho'
        write(iunit,'(A)')'To be used as a boundary condition by COGENT'
        write(iunit,'(A)')'Grids and distr.function are expressed  '
        write(iunit,'(A)')'  in mks units and normalized by ref.values.'
        write(iunit,'(A)')'Example for magnetic field: [Tesla]/units_B'
        write(iunit,'(A)')'Example for f: [m^-3 (m/s)^-3]/ref_distfunc'
        write(iunit,'(A)')'radcoord=',radcoord
        write(iunit,'(A)')
     +  'See cql3d_cogent_units.txt for definition of
     +   units_B, ref_v, ref_mu, ref_distfunc'
        write(iunit,'(A)')'  Scalars in this file:'
        write(iunit,'(A)')'npol_f3d  = number of poloidal angle points'
        write(iunit,'(A)')'nvpar_f3d = number of Vpar points'
        write(iunit,'(A)')'nmu_f3d = number of mu=0.5*m*Vper^2/B points'
        write(iunit,'(A)')'f3d_rho   = rho value'
        write(iunit,'(A)')'(definition of rho is according to radcoord)'
        write(iunit,'(A)')'vnorm= max.velocity of CQL3D v-grid [m/s]'
        write(iunit,'(A)')'mass(1)/proton[kg] = Mass of species#1'
        write(iunit,'(A)')'  1D grids [format: (1pe16.7)]:'
        write(iunit,'(A)')
     +      'f3d_pol(ipol=1,npol_f3d)      [rad]'
        write(iunit,'(A)')
     +      'f3d_vpar(ivpar=1,nvpar_f3d)   [m/s]/ref_v'
        write(iunit,'(A)')
     +      'f3d_mu(imu=1,nmu_f3d)         [Joule/Tesla]/ref_mu'
        write(iunit,'(A)')'  1D array magn.field [format: (1pe16.7)]:'
        write(iunit,'(A)')
     +      'f3d_b(ipol=1,npol_f3d)        [T]/units_B'
        write(iunit,'(A)')'  3D array dist.func. [format: 5(1pe16.7)]:'
        write(iunit,'(A)')
     +      'f3d(ipol,ivpar,imu)           [m^-3 (m/s)^-3]/ref_distfunc'
        write(iunit,'(A)')'--------------------------------------------'
        write(iunit, '(i5)') npol_f3d   !number of poloidal angle points
        write(iunit, '(i5)') nvpar_f3d  !number of Vpar points
        write(iunit, '(i5)') nmu_f3d    !number of mu=0.5*m*Vper^2/B points
        write(iunit, '(1pe16.7)') f3d_rho      ! rho value (rcoupling/a)
        write(iunit, '(1pe16.7)') vnorm/100    ! [m/s]
        write(iunit, '(1pe16.7)') fmass(k)*1.e-3/proton_kg ! Mass of species#1
        ! Grids:
        write(iunit,'(1pe16.7)') 
     +       ( f3d_pol(ipol),              ipol= 1,npol_f3d )
        write(iunit,'(1pe16.7)') 
     +       ( f3d_vpar(ivpar)/100/ref_v,  ivpar=1,nvpar_f3d )
        write(iunit,'(1pe16.7)') 
     +       ( f3d_mu(imu)*1.e-3/ref_mu,   imu=  1,nmu_f3d )
        ! Magnetic field at pol.angle grid points, [T]/units_B:
        write(iunit,'(1pe16.7)') 
     +       ( f3d_b(ipol)*1.e-4/units_B,  ipol= 1,npol_f3d ) 
        ! 3D distribution function, [m^-3 (m/s)^-3]/ref_distfunc
        write(iunit,1000) ( (( 
     +         f3d(ipol,ivpar,imu),
     +                                     ipol= 1,npol_f3d  ),
     +                                     ivpar=1,nvpar_f3d ),
     +                                     imu=  1,nmu_f3d   )
     
        close(iunit)

        open(unit=iunit,file='cql3d_f3d_taub.txt',STATUS='UNKNOWN')
        write(iunit,'(A)')'tau_bounce [sec] over (ipol,ivpar,imu) grid.'
        write(iunit,'(A)')'Other data - in cql3d_f3d.txt (for COGENT)'
        write(iunit,'(A)')'--------------------------------------------'
        write(iunit, '(i5)') npol_f3d   !number of poloidal angle points
        write(iunit, '(i5)') nvpar_f3d  !number of Vpar points
        write(iunit, '(i5)') nmu_f3d    !number of mu=0.5*m*Vper^2/B points
        write(iunit, '(1pe16.7)') f3d_rho      ! rho value (rcoupling/a)
        write(iunit, '(1pe16.7)') vnorm/100    ! [m/s]
        write(iunit, '(1pe16.7)') fmass(k)*1.e-3/proton_kg ! Mass of species#1
        ! Grids:
        write(iunit,'(1pe16.7)') 
     +       ( f3d_pol(ipol),              ipol= 1,npol_f3d )
        write(iunit,'(1pe16.7)') 
     +       ( f3d_vpar(ivpar)/100/ref_v,  ivpar=1,nvpar_f3d )
        write(iunit,'(1pe16.7)') 
     +       ( f3d_mu(imu)*1.e-3/ref_mu,   imu=  1,nmu_f3d )
        write(iunit,1000) ( (( 
     +         f3d_tau(ipol,ivpar,imu),
     +                                     ipol= 1,npol_f3d  ),
     +                                     ivpar=1,nvpar_f3d ),
     +                                     imu=  1,nmu_f3d   )
        close(iunit)
        

      else
        STOP 'f3dwrite: only ready for f3d_format=ascii'
      endif

 1000 format(5(1pe16.7))

c.......................................................................

      deallocate(f3d,f3d_tau, f3d_pol, f3d_b, f3d_vpar, f3d_mu)

      return

      end ! f3dwrite

c======================================================================
c======================================================================

      subroutine ncwritef4d(f4dr,f4dz,f4dv,f4dt,f4ddv,f4ddt,f4d)
      implicit integer (i-n), real*8 (a-h,o-z)

      include 'param.h'
      include 'comm.h'
c --- include file for netCDF declarations 
c --- (obtained from NetCDF distribution)
      include 'netcdf.inc'
CMPIINSERT_INCLUDE
      
c     nr_f4d,nz_f4d are R,Z grid dimensions set in namelist
c     nv_f4d,nt_f4d are dims of normalized vel and of pitch angle grids.
      real*8, dimension(:) :: f4dr(nr_f4d),f4dz(nz_f4d),
     +                        f4dv(nv_f4d),f4dt(nt_f4d),
     +                        f4ddv(nv_f4d),f4ddt(nt_f4d)
      real*8, dimension(nr_f4d,nz_f4d,nv_f4d,nt_f4d) :: f4d  !dims in comm.h



      integer ncid,vid,istatus
      integer chardim,char64dim
      integer dim_nr_f4d,dim_nz_f4d,dim_nv_f4d,dim_nt_f4d
      integer dims(4),start(4),count(4)

      character*128 ltitle

      data start/1,1,1,1/

CMPIINSERT_IF_RANK_NE_0_RETURN

      count(1)=nr_f4d
      count(2)=nz_f4d
      count(3)=nv_f4d
      count(4)=nt_f4d

c     Create netCDF file
      write(t_,1000) mnemonic(1:length_char(mnemonic))
 1000 format(a,"_f4d.nc")
      istatus = NF_CREATE(t_, NF_CLOBBER, ncid)
      call check_err(istatus)

c     Define dimensions
      istatus= NF_DEF_DIM(ncid, 'dim_nr_f4d', nr_f4d, dim_nr_f4d)
      istatus= NF_DEF_DIM(ncid, 'dim_nz_f4d', nz_f4d, dim_nz_f4d)
      istatus= NF_DEF_DIM(ncid, 'dim_nv_f4d', nv_f4d, dim_nv_f4d)
      istatus= NF_DEF_DIM(ncid, 'dim_nt_f4d', nt_f4d, dim_nt_f4d)
      istatus= NF_DEF_DIM(ncid, 'chardim',      8,    chardim)
      istatus= NF_DEF_DIM(ncid, 'char64dim',   64,    char64dim)

c     Define vector of dimensions
      dims(1)=dim_nr_f4d
      dims(2)=dim_nz_f4d
      dims(3)=dim_nv_f4d
      dims(4)=dim_nt_f4d

c     Define netCDF variables
      
      ltitle='CQL3D 4D Distn Function (R,Z,v,pitch): '//version
      if( length_char(ltitle).gt.128 ) stop 'Adjust ltitle in f4dwrite'
      call ncaptc2(ncid,NCGLOBAL,'title',NCCHAR,length_char(ltitle),
     +     ltitle,istatus)

      vid=ncvdef2(ncid,'version',NCCHAR,1,char64dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +          'CQL3D version number',istatus)

      vid=ncvdef2(ncid,'mnemonic',NCCHAR,1,char64dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +          'Mnemonic run identifier',istatus)

      vid=ncvdef2(ncid,'vnorm',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +          'velocity (momentum-per-mass) norm',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +                     'cms/sec',istatus)

      vid=ncvdef2(ncid,'enorm',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +                     'Energy normalization',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'keV',istatus)

      vid=ncvdef2(ncid,'rmag',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +          'Magnetic axis major radius',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'zmag',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,31,
     +          'Magnetic axis vertical position',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'eqsym',NCCHAR,1,chardim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +          'Indicator for symmetrization',istatus)

      vid=ncvdef2(ncid,'zshift',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +          'Vertical shift of equilibrium per eqsym',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +          'cms',istatus)

      vid=ncvdef2(ncid,'rmincon',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +          'Minimum major radius at outer contour',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +          'cms',istatus)

      vid=ncvdef2(ncid,'rmaxcon',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +          'Maximum major radius at outer contour',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'zmincon',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +          'Minimum axial (z-dirn) at outer contour',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +          'cms',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'zmaxcon',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +          'Maximum axial (Z-dirn) at outer contour',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +          'cms',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'nr_f4d',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +          'Major radius grid dimension(=dim_nr_f4d)',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'nz_f4d',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,35,
     +          'Axial Z-grid dimension(=dim_nz_f4d)',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'nv_f4d',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,42,
     +          'Normalized vel grid dimension(=dim_nv_f4d)',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'nt_f4d',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +          'Pitch angle grid dimension(=dim_nt_f4d)',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'f4dr',NCDOUBLE,1,dim_nr_f4d,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +          'Major Radius mesh (presently equispaced)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,'cms',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'f4dz',NCDOUBLE,1,dim_nz_f4d,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,35,
     +          'Axial Z-mesh (presently equispaced)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,'cms',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'f4dv',NCDOUBLE,1,dim_nv_f4d,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +          'Normalized to vnorm vel mesh (equispaced)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,8,'unitless',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'f4dt',NCDOUBLE,1,dim_nt_f4d,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +          'Pitch angle mesh (presently equispaced)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,'radians',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'f4ddv',NCDOUBLE,1,dim_nv_f4d,istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,7,
     +           'v**2*dv',istatus)
      call ncaptc2(ncid,vid,'long_name2',NCCHAR,29,
     +          'Normalized vel volume element',istatus)

      vid=ncvdef2(ncid,'f4ddt',NCDOUBLE,1,dim_nt_f4d,istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,22,
     +          '2*pi*sin(pitch)*dpitch',istatus)
      call ncaptc2(ncid,vid,'long_name2',NCCHAR,36,
     +          'vel space pitch angle volume element',istatus)
      
      vid=ncvdef2(ncid,'f4d',NCDOUBLE,4,dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +          'Distribution function on R,Z,v,pitch grid',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,28,
     +          'vnorm**3/(cm**3*(cm/sec)**3)',istatus)
      call check_err(istatus)

c     End the define-mode
      istatus= NF_ENDDEF(ncid)
      call check_err(istatus)


c.......................................................................
c     Put data in the file
c.......................................................................

      istatus= NF_INQ_VARID(ncid,'version',vid)
      ll=length_char(version)
      call ncvptc2(ncid,vid,1,ll,version,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'mnemonic',vid)
      ll=length_char(mnemonic)
      call ncvptc2(ncid,vid,1,ll,mnemonic,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'vnorm',vid)
      call ncvpt_doubl2(ncid,vid,1,1,vnorm,istatus)

      istatus= NF_INQ_VARID(ncid,'enorm',vid)
      call ncvpt_doubl2(ncid,vid,1,1,enorm,istatus)

      istatus= NF_INQ_VARID(ncid,'rmag',vid)
      call ncvpt_doubl2(ncid,vid,1,1,rmag,istatus)

      istatus= NF_INQ_VARID(ncid,'zmag',vid)
      call ncvpt_doubl2(ncid,vid,1,1,zmag,istatus)

      istatus= NF_INQ_VARID(ncid,'eqsym',vid)
      call ncvptc2(ncid,vid,1,8,eqsym,8,istatus)

      istatus= NF_INQ_VARID(ncid,'zshift',vid)
      call ncvpt_doubl2(ncid,vid,1,1,zshift,istatus)

      istatus= NF_INQ_VARID(ncid,'rmincon',vid)
      call ncvpt_doubl2(ncid,vid,1,1,rmincon,istatus)

      istatus= NF_INQ_VARID(ncid,'rmaxcon',vid)
      call ncvpt_doubl2(ncid,vid,1,1,rmaxcon,istatus)

      istatus= NF_INQ_VARID(ncid,'zmincon',vid)
      call ncvpt_doubl2(ncid,vid,1,1,zmincon,istatus)

      istatus= NF_INQ_VARID(ncid,'zmaxcon',vid)
      call ncvpt_doubl2(ncid,vid,1,1,zmaxcon,istatus)

      istatus= NF_INQ_VARID(ncid,'nr_f4d',vid)
      call ncvpt_int2(ncid,vid,1,1,nr_f4d,istatus)

      istatus= NF_INQ_VARID(ncid,'nz_f4d',vid)
      call ncvpt_int2(ncid,vid,1,1,nz_f4d,istatus)

      istatus= NF_INQ_VARID(ncid,'nv_f4d',vid)
      call ncvpt_int2(ncid,vid,1,1,nv_f4d,istatus)

      istatus= NF_INQ_VARID(ncid,'nt_f4d',vid)
      call ncvpt_int2(ncid,vid,1,1,nt_f4d,istatus)

      istatus= NF_INQ_VARID(ncid,'f4dr',vid)
      call ncvpt_doubl2(ncid,vid,1,nr_f4d,f4dr,istatus)

      istatus= NF_INQ_VARID(ncid,'f4dz',vid)
      call ncvpt_doubl2(ncid,vid,1,nz_f4d,f4dz,istatus)

      istatus= NF_INQ_VARID(ncid,'f4dv',vid)
      call ncvpt_doubl2(ncid,vid,1,nv_f4d,f4dv,istatus)

      istatus= NF_INQ_VARID(ncid,'f4dt',vid)
      call ncvpt_doubl2(ncid,vid,1,nt_f4d,f4dt,istatus)

      istatus= NF_INQ_VARID(ncid,'f4ddv',vid)
      call ncvpt_doubl2(ncid,vid,1,nv_f4d,f4ddv,istatus)

      istatus= NF_INQ_VARID(ncid,'f4ddt',vid)
      call ncvpt_doubl2(ncid,vid,1,nt_f4d,f4ddt,istatus)

      istatus= NF_INQ_VARID(ncid,'f4d',vid)
      call ncvpt_doubl2(ncid,vid,start,count,f4d,istatus)

      return

      end
      


