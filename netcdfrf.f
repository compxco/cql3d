c
c
      subroutine netcdfrf(kopt,krf)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

      include 'param.h'
      include 'comm.h'

c     This subroutine uses netCDF-2 subroutines. 
c     http://www.unidata.ucar.edu/packages/netcdf/index.html
c     (Had a problem with linking to netCDF-3 subroutines
c      on account of embedded underscores in their names). 
c
c     This subroutine either reads an rf data file (see below),
c       or creates/writes a netCDF file with name "mnemonic"_rf.nc,
c       for RF data from cql3d.
c      (Additional standard netCDF output is given 
c       in file "mnemonic".nc).
c
c     Action of the subroutine is controlled by kopt,krf, 
c       time step counter n:
c       kopt=0, n=0, initialize "mnemonic"_rf.nc and write data
c       kopt=1, n.gt.0, write data (only time-dep data, unless n.eq.nstop)
c       kopt=2, read basic ray data from file rffile(krf)
c       kopt=3 is a limited part of kopt=2, 
c            to only read the values of nray(krf),nharm(krf),freqcy(krf)
c       kopt=4 is a limited part of kopt=2
      ! kopt=2 part is called by urfread with krf=1:mrf to read all data.
      ! BUT if kopt=3 or 4, this part is called by urfsetup 
      ! to only read the values of nray(krf),nharm(krf),freqcy(krf), 
      ! and also nrayelt(:,krf) (when kopt=4).
c
c     krf= wave-type number, presently for reading input files
c            for up to 3 modes, with names specified by rffile(1:3);
c            by default, rffile()=rayech.nc, and/or rayrfw.nc,
c                                 and/or raylh.nc, depending on
c                                 namelist values of ech,lh,fw
c                                 See cqlinput for alternate namings.
c     BH060314:  Generalized to multi-wave-types/multi-harmonic, up to
c                maximum specified by parameter nmodsa.
c
c     YuP-100404: writing output ray data is now set up for all krf=1:mrf.
c     The data is saved into files "mnemonic"_krf###.nc, for each krf.

c     Data written consists of  ray data and ql diffusion 
c     coefficient urfb (if netcdfshort.ne.'enabled' .and. .ne.'longer_b'
c     .and. .ne. 'long_urf'),
c     in which case urfb will be stored on the last step.
c     If netcdfshort.eq.'longer_b', the urfb stored at every time step.
c     if netcdfshort='long_urf', then all urf coefs (urfb,urfc,
c     urfe,urff) are stored at last time step.
c     Only data for one mode (or harmonic) is stored,
c     for the time being.
c     Also, some mesh data is included,  for possible
c     convenience in using this data file without
c     the general "mnemonic".nc file, e.g., with idl or ncbrowse.
c
c
c     NOTE:  Some of code comments are vestigial to previous code
c            fragments used in this code construction.
c            Take them with a grain of salt.
c
c
c --- include file for netCDF declarations 
c --- (obtained from NetCDF distribution)
      include 'netcdf.inc'
CMPIINSERT_INCLUDE

      integer numrec(nmodsa) !-YuP-> added: as a function of krf
      common/netcdfrf_numrec/numrec ! counter for recording data
      
c --- some stuff for netCDF file ---
      character*128 name
      character*3  krf_index               !-YuP-> to form file name 
      integer ncid,vid,istatus
      integer xdim,ydim,rdim,r0dim,twodim,tdim
      integer nraydim,neltdim
      integer cdim

      integer dims(4),count(4),start(4)
      integer count1(4),start1(4)
      integer ray_dims(3),ray_count(3)
      integer r0_dims(2),r0_count(2)
      integer y_dims(2),y_count(2)
      integer tau_dims(2),tau_count(2)

      complex*16 cei

      data start/1,1,1,1/, start1/1,1,1,1/
      
CMPIINSERT_IF_RANK_NE_0_RETURN
c This subroutine is only called from MPI rank=0.

      cei=(0.,1.)



      WRITE(*,'(a,4i6)')'Begin netcdfrf, n,kopt,krf,irfn(krf):',
     +                                   n,kopt,krf,irfn(krf)

      if(n.eq.0) call ibcast(numrec,1,SIZE(numrec)) ! counter for data recording
      
      
      
c --- begin if ---
      if ((kopt.eq.0 .or. kopt.eq.1) .and. 
     +    (irffile(krf).ne."combine2"))  then         !To line 1076

      !---- Form file name for each krf:
      WRITE(krf_index(1:3),'(3I1)') 0,0,0
      IF(                 krf.LE.9  ) WRITE(krf_index(3:3),'(I1)') krf
      IF(krf.GE.10  .AND. krf.LE.99 ) WRITE(krf_index(2:3),'(I2)') krf
      IF(krf.GE.100 .AND. krf.LE.999) WRITE(krf_index(1:3),'(I3)') krf 
      if(krf.GE.1000) stop "Not setup for krf>999"
      t_= mnemonic(1:length_char(mnemonic))//'_krf'//krf_index//'.nc'
      !---- These files are created-then-closed at n=0, 
      !---- opened-then-closed for writing data at n>0, 
      !---- and opened-then-closed at n=nstop

C-----------------------------------------------------------------------
C     Only set up for cqlpmod.ne."enabled",ngen=1, for the time being.
C-----------------------------------------------------------------------
      if (cqlpmod.eq."enabled" ) then  !-YuP: removed  .or. ngen.ne.1   ???
         WRITE(*,*) 'WARNING: netcdfrf subroutine not implemented'
         return
      endif

c     Maximum iy as function of radius:
c$$$      iy=0
c$$$      do l=1,lrz
c$$$         iy=max(iy,iy_(l))
c$$$      enddo

      nray_krf= nray(irfn(krf)) !YuP[2020-09-23] 
        !Such arrays as freqcy, delpwr, cwexde, etc, -
        !they are not functions of krf (wave type; Example: krf=1:mrf=1:2),
        !but rather they are functions of krfn (wave modes; Example: mrfn=3+3).
        !Initially, they are read for each krf, 
        !but then duplicated into each krfn index.
        !This is done in subr.urfread, at the end.
        !Here, netcdfrf(kopt=0/1, krf) is called for recording data.
        !So,  krf is wave type (=1 and 2 in our example).
        !Then we need to record them like
        !cwexde(is,iray, irfn(krf)),
        !where irfn(krf=1)=1 and irfn(krf=2)=4 in our example,
        !irfn(krf) is pointing to beginning harmonic of each wave type
        !== index (in 1:mrfn) of the lowest harmonic for each wave type 

c     Maximum number of ray elements per ray, for this specific krf (wave type):
      neltmax=0
      do iray=1,nray_krf  !-YuP: nray(1) -> nray_krf
         neltmax=max(neltmax,nrayelt(iray,irfn(krf))) ! nrayelt(*,1)->nrayelt(*,irfn(krf))
      enddo
      
c     Following counting vectors set up to facilitate writing
c     of the various arrays. Set up here to ensure initialization
c     in each call to the subroutine.

c$$$      count(1)=iy
      count(1)=iymax
      count(2)=jx
      count(3)=lrz
      count(4)=1

c$$$      count1(1)=iy
      count1(1)=iymax
      count1(2)=jx
      count1(3)=1
      count1(4)=1

      ray_count(1)=neltmax ! depends on krf
      ray_count(2)=nray_krf !-YuP: nray(1) -> nray_krf
      ray_count(3)=2

      r0_count(1)=lrzmax
      r0_count(2)=1

c$$$      y_count(1)=iy
      y_count(1)=iymax
      y_count(2)=lrz

      tau_count(1)=iy
      tau_count(2)=lrzmax


c.......................................................................
cl    1. Initialize, creating new netcdf file
c

c --- begin if ---
      !WRITE(*,*)'netcdfrf: n,kopt,krf= ',n,kopt,krf
      if ((kopt.eq.0) .and. (n.eq.0)) then 

C-----------------------------------------------------------------------
c
cl     1.1 create netCDF file and define dimensions,variables 
c          and attributes
c

c.......................................................................
cl    1.1.1 create netCDF file (Entering define mode.)
c     integer function nccre(filename,overwrite?,error_code)
c     Ref to page 46 of NetCDF-2 manual.
c     CLOBber old file, if it exists.
c     istatus is 0, if no errors.
      !NF_CLOBBER means overwrite old file
      istatus = NF_CREATE(t_, NF_CLOBBER, ncid) !-YuP: NetCDF-f77
      call check_err(istatus)
      WRITE(*,*)' netcdfrf 1.After NF_CREATE istatus=',istatus

c.......................................................................
cl    1.1.2 define dimensions
c     p. 67 of netcdf-2 manual
c     integer function ncddef(ncid,dim_name,dim_siz,error_code)
c       returns dimension id.

c-YuP:      xdim=ncddef(ncid,'x',jx,istatus)
c-YuP:      ydim=ncddef(ncid,'y',iymax,istatus)
c-YuP:      rdim=ncddef(ncid,'r',lrz,istatus)
c-YuP:      r0dim=ncddef(ncid,'r0',lrzmax,istatus)
c-YuP:      twodim=ncddef(ncid,'two',2,istatus)
c-YuP:      neltdim=ncddef(ncid,'neltmax',neltmax,istatus)
c-YuP:      nraydim=ncddef(ncid,'nrays',nray(1),istatus)
c-YuP:      cdim=ncddef(ncid,'cdim',8,istatus)

      istatus= NF_DEF_DIM(ncid, 'x',      jx ,      xdim)  !-YuP: NetCDF-f77
      istatus= NF_DEF_DIM(ncid, 'y',      iymax,    ydim)
      istatus= NF_DEF_DIM(ncid, 'r',      lrz,      rdim)
      istatus= NF_DEF_DIM(ncid, 'r0',     lrzmax,   r0dim)
      istatus= NF_DEF_DIM(ncid, 'two',    2,        twodim)
      istatus= NF_DEF_DIM(ncid, 'neltmax',neltmax,  neltdim) ! depends on krf
      istatus= NF_DEF_DIM(ncid, 'nrays',  nray_krf,nraydim) ! 
      istatus= NF_DEF_DIM(ncid, 'cdim',   8,        cdim)  !-YuP: NetCDF-f77

c     unlimited dimension for time, dimension name= 'time'
      istatus= NF_DEF_DIM(ncid, 'time',NF_UNLIMITED,tdim) !-YuP: NetCDF-f77
      call check_err(istatus)
      WRITE(*,*)' netcdfrf 2.After NF_DEF_DIM/time istatus=',istatus

c     define vector of dimensions [for urfb(1:iy,1:jx,1:lrz,irfn(krf))], unlimited last
      dims(1)=ydim
      dims(2)=xdim
      dims(3)=rdim
      dims(4)=tdim

      ray_dims(1)=neltdim ! depends on krf
      ray_dims(2)=nraydim ! depends on krf
      ray_dims(3)=twodim

      r0_dims(1)=r0dim
      r0_dims(2)=tdim

      y_dims(1)=ydim
      y_dims(2)=rdim

      tau_dims(1)=ydim
      tau_dims(2)=r0dim

c.......................................................................
cl    1.1.3 define variables

c     Note, the variable IDs (denoted below as vid) are
c     not saved here in this subroutine; rather, the IDs
c     are retrieved from the netCDF data file, as needed,
c     by calling the netCDF routine ncvid.

c     netCDF variable_define:
c     integer function ncvdef2(ncid,variable_name,variable_type,
c                number_of_dimensions,
c                vector_for_length_per_dimension,error_code)
c       returns varid.
c     Note: Unlimited dimension must be last
c     Refer to p. 77, netcdf-2 manual

c     NCDOUBLE for REAL*8.  This is independent of the
c       internal representation of data on a particular
c       machine, e.g., 32- or 64-bit based calculations.


c--------------------------
c     Ray data
c--------------------------

      vid=ncvdef2(ncid,'nray',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,14,
     +            'Number of rays',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'freqcy',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,14,
     +            'Wave frequency',istatus)
      call check_err(istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,2,
     +           'Hz',istatus)

      vid=ncvdef2(ncid,'lh',NCCHAR,1,cdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +          'Indicates raylh LH ray data file used',istatus)

      vid=ncvdef2(ncid,'fw',NCCHAR,1,cdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,37,
     +          'Indicates rayfw FW ray data file used',istatus)

      vid=ncvdef2(ncid,'ech',NCCHAR,1,cdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +          'Indicates rayech ECH ray data file used',istatus)

      vid=ncvdef2(ncid,'nharm1',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,21,
     +            'First harmonic number',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'nharms',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name0',NCCHAR,39,
     +            'Number of harmonics, starting at nharm1',istatus)
      call ncaptc2(ncid,vid,'long_name1',NCCHAR,27,
     +            ' Only .gt.1 for one rf mode',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'nrayelt',NCLONG,1,nraydim,istatus) ! dep. on krf
      call ncaptc2(ncid,vid,'long_name',NCCHAR,35,
     +            'Number of ray elements for each ray',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'ws',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +           'poloidal distance along a ray',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'spsi',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +           'radial-like variable',istatus)

      vid=ncvdef2(ncid,'wr',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,12,
     +           'major radius',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'wphi',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,14,
     +           'toroidal angle',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'radians',istatus)

      vid=ncvdef2(ncid,'wz',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,15,
     +           'vertical height',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'wnpar',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +           'parallel refractive index',istatus)

      vid=ncvdef2(ncid,'wnper',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +           'perpendicular refractive index',istatus)

      vid=ncvdef2(ncid,'delpwr',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +           'power in ray channel',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,29,
     +           'erg/sec [BH180207:MKS, check]',istatus)

      vid=ncvdef2(ncid,'urfpwrl',NCDOUBLE,2,ray_dims,istatus) !YuP[2020-09-24] added
      call ncaptc2(ncid,vid,'long_name',NCCHAR,64,
     +'linear power absorption: dp=urfpwrl[i,irayelt]*delpwr[i,irayelt]'
     & ,istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,22,
     +           'erg/sec(after *delpwr)',istatus)

      vid=ncvdef2(ncid,'urfpwrc',NCDOUBLE,2,ray_dims,istatus) !YuP[2020-09-24] added
      call ncaptc2(ncid,vid,'long_name',NCCHAR,64,
     +'collis power absorption: dp=urfpwrc[i,irayelt]*delpwr[i,irayelt]'
     & ,istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,22,
     +           'erg/sec(after *delpwr)',istatus)

      vid=ncvdef2(ncid,'sdpwr',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +           'power to ions (if incl)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,8,
     +           'ergs/sec',istatus)

      vid=ncvdef2(ncid,'wdnpar',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +           'effective width in npar',istatus)

c     Added 3rd dimension equal to 2 accomodates complex data.
      vid=ncvdef2(ncid,'cwexde',NCDOUBLE,3,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +           'Complex Ex/E Polarization',istatus)

      vid=ncvdef2(ncid,'cweyde',NCDOUBLE,3,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +           'Complex Ey/E Polarization',istatus)

      vid=ncvdef2(ncid,'cwezde',NCDOUBLE,3,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +           'Complex Ez/E Polarization',istatus)

      vid=ncvdef2(ncid,'fluxn',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +           'fluxn, Stix norm, |E|=1',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,13,
     +           'ergs/sec/cm^2',istatus)

      vid=ncvdef2(ncid,'sbtot',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +           'Magnetic field strength',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +           'gauss',istatus)

      vid=ncvdef2(ncid,'sene',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,17,
     +           'Density along ray',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,14,
     +           'particles/cm^3',istatus)

      vid=ncvdef2(ncid,'salphac',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +           'Collisional damping wavenumber',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +           '1/cms',istatus)

      vid=ncvdef2(ncid,'salphal',NCDOUBLE,2,ray_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +           'Linear damping wavenumber',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,5,
     +           '1/cms',istatus)


c--------------------------
c     Additional Time-independent data
c--------------------------


      vid=ncvdef2(ncid,'lrzmax',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +            'Number of radial surfaces',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'rya',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,22,
     +           'Normalized radial mesh',istatus)

      vid=ncvdef2(ncid,'rhomax',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)

      vid=ncvdef2(ncid,'lrz',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +            'Number of FPd radial surfaces',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'lrindx',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +           'Radial indices of FPd surfaces',istatus)

      vid=ncvdef2(ncid,'jx',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +            'momentum-per-mass dimension',istatus)
      call check_err(istatus)
      
      vid=ncvdef2(ncid,'mrf',NCLONG,0,0,istatus) !-YuP: added
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +            'number of rf wave types',istatus)
      call check_err(istatus)
      
      vid=ncvdef2(ncid,'mrfn',NCLONG,0,0,istatus) !-YuP[2017]added
      call ncaptc2(ncid,vid,'long_name',NCCHAR,59,
     + 'number of rf modes (sum over all wave types and all nharms)',
     + istatus)
      call check_err(istatus)
      vid=ncvdef2(ncid,'krf',NCLONG,0,0,istatus) !-YuP: added
      call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +            'This file: rf wave type number',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'x',NCDOUBLE,1,xdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +           'normalized momentum-per-mass',istatus)

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
      call ncaptc2(ncid,vid,'long_name',NCCHAR,21,
     +            'Pitch angle dimension',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'y',NCDOUBLE,2,y_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,11,
     +           'pitch angle',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +           'radians',istatus)

      vid=ncvdef2(ncid,'iy_',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,36,
     +            'Pitch angle dimension at each radius',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'itl',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,26,
     +           'lower trapped-passing bndy',istatus)

      vid=ncvdef2(ncid,'itu',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,26,
     +           'upper trapped-passing bndy',istatus)

      vid=ncvdef2(ncid,'tau',NCDOUBLE,2,tau_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +           'tau_bounce * abs(speed)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'cms',istatus)


c--------------------------
c     Time-dependent data
c--------------------------

      vid=ncvdef2(ncid,'time',NCDOUBLE,1,tdim,istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +                     'seconds',istatus)

      if (netcdfshort.eq.'enabled') then
c         Do nothing: no storage defined.
      elseif (netcdfshort.eq.'longer_b') then
         vid=ncvdef2(ncid,'urfb',NCDOUBLE,4,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,41,
     +        'u^2 * BA QL Diff Coeff * v_par*tau_bounce',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,12,
     +        'cgs/vnorm**4',istatus)
      elseif (netcdfshort.eq.'long_urf') then
         vid=ncvdef2(ncid,'urfb',NCDOUBLE,3,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,42,
     +        'u^2 * BA<QL Duu Coeff> * v_par0*tau_bounce',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,12,
     +        'cgs/vnorm**4',istatus)
         vid=ncvdef2(ncid,'urfc',NCDOUBLE,3,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,51,
     +     'u * BA<c/(psi**0.5*c0) QL Dtu> * v_par0*tau_bounce',istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +        'Above: c=cos(theta),c0=c at B0, t=theta',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,12,
     +        'cgs/vnorm**3',istatus)
c         vid=ncvdef2(ncid,'urfe',NCDOUBLE,3,dims,istatus)
c         call ncaptc2(ncid,vid,'long_name',NCCHAR,42,
c     +        'u * BA<c*s/(psi*c0) QL Dut> * v_par0*tau_bounce',istatus)
c         call ncaptc2(ncid,vid,'units',NCCHAR,12,
c     +        'cgs/vnorm**3',istatus)
c         vid=ncvdef2(ncid,'urff',NCDOUBLE,3,dims,istatus)
c         call ncaptc2(ncid,vid,'long_name',NCCHAR,55,
c     +        'BA<c**2*s/(psi**(3/2)*c0**2) QL Dtt> * v_par*tau_bounce',
c     +        istatus)
c         call ncaptc2(ncid,vid,'units',NCCHAR,12,
c     +        'cgs/vnorm**2',istatus)
!YuP[03/18/2015] urfe,urff are expressed through urfb,urfc
! No need to save them.
      else ! netcdfshort='disabled'; Standard o/p: f at last time step
         vid=ncvdef2(ncid,'urfb',NCDOUBLE,3,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,42,
     +        'u^2 * BA<QL Duu Coeff> * v_par0*tau_bounce',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,25,
     +        'code units:  cgs/vnorm**4',istatus)
      endif
      call check_err(istatus)
      WRITE(*,*)' netcdfrf 3.After NF_DEF_DIM/urfb istatus=',istatus


c.......................................................................
cl    1.1.4 end the define-mode and start the data-mode
c     p. 51-2 of netcdf manual

      istatus= NF_ENDDEF(ncid) !-YuP: NetCDF-f77
      call check_err(istatus)
      WRITE(*,*)' netcdfrf 4.After NF_ENDDEF istatus=',istatus

c.......................................................................

cl    1.2 Write data
c
      start(4)=numrec(krf) ! counter for data recording: initially 1 

c --- initialize data file ---
c     First get variable_id: 
c        integer function ncvid(ncid,variable_name,error_code)
c        returns varid
c     Then write data with nc variable_put:
c        ncvpt(ncid,varid,start_indices,counts,values,error_code)
c
c     p. 79, 89 of netcdf-2 manual
c

c--------------------------
c     Time-independent data
c--------------------------

c-YuP:      vid=ncvid(ncid,'lrzmax',istatus)
      istatus= NF_INQ_VARID(ncid,'lrzmax',vid)  
      call ncvpt_int2(ncid,vid,1,1,lrzmax,istatus)

c-YuP:      vid=ncvid(ncid,'rya',istatus)
      istatus= NF_INQ_VARID(ncid,'rya',vid)  
      call ncvpt_doubl2(ncid,vid,1,lrzmax,rya(1),istatus)

c-YuP:      vid=ncvid(ncid,'rhomax',istatus)
      istatus= NF_INQ_VARID(ncid,'rhomax',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,rhomax,istatus)

c-YuP:      vid=ncvid(ncid,'lrz',istatus)
      istatus= NF_INQ_VARID(ncid,'lrz',vid)  
      call ncvpt_int2(ncid,vid,1,1,lrz,istatus)

c-YuP:      vid=ncvid(ncid,'lrindx',istatus)
      istatus= NF_INQ_VARID(ncid,'lrindx',vid)  
      call ncvpt_int2(ncid,vid,1,lrz,lrindx(1),istatus)

c-YuP:      vid=ncvid(ncid,'jx',istatus)
      istatus= NF_INQ_VARID(ncid,'jx',vid)  
      call ncvpt_int2(ncid,vid,1,1,jx,istatus)

      istatus= NF_INQ_VARID(ncid,'mrf',vid)  !-YuP: added
      call ncvpt_int2(ncid,vid,1,1,mrf,istatus)

      istatus= NF_INQ_VARID(ncid,'mrfn',vid)  !-YuP[2017]added
      call ncvpt_int2(ncid,vid,1,1,mrfn,istatus)
      !number of rf modes (sum over all wave types and all nharms)
      istatus= NF_INQ_VARID(ncid,'krf',vid)  !-YuP: added
      call ncvpt_int2(ncid,vid,1,1,krf,istatus)

c-YuP:      vid=ncvid(ncid,'x',istatus)
      istatus= NF_INQ_VARID(ncid,'x',vid)  
      call ncvpt_doubl2(ncid,vid,1,jx,x,istatus)

c-YuP:      vid=ncvid(ncid,'vnorm',istatus)
      istatus= NF_INQ_VARID(ncid,'vnorm',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,vnorm,istatus)

c-YuP:      vid=ncvid(ncid,'enorm',istatus)
      istatus= NF_INQ_VARID(ncid,'enorm',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,enorm,istatus)

c-YuP:      vid=ncvid(ncid,'iy',istatus)
      istatus= NF_INQ_VARID(ncid,'iy',vid)  
      call ncvpt_int2(ncid,vid,1,1,iymax,istatus)

      call pack21(y,1,iy,1,lrors,tem1,iymax,lrors)
c-YuP:      vid=ncvid(ncid,'y',istatus)
      istatus= NF_INQ_VARID(ncid,'y',vid)  
      call ncvpt_doubl2(ncid,vid,start,y_count,tem1,istatus)

c-YuP:      vid=ncvid(ncid,'iy_',istatus)
      istatus= NF_INQ_VARID(ncid,'iy_',vid)  
      call ncvpt_int2(ncid,vid,1,count(3),iy_,istatus)

c-YuP:      vid=ncvid(ncid,'itl',istatus)
      istatus= NF_INQ_VARID(ncid,'itl',vid)  
      call ncvpt_int2(ncid,vid,1,count(3),itl_,istatus)

c-YuP:      vid=ncvid(ncid,'itu',istatus)
      istatus= NF_INQ_VARID(ncid,'itu',vid)  
      call ncvpt_int2(ncid,vid,1,count(3),itu_,istatus)
      
      call pack21(tau,1,iy,1,lrzmax,tem1,iy,lrzmax)
c-YuP:      vid=ncvid(ncid,'tau',istatus)
      istatus= NF_INQ_VARID(ncid,'tau',vid)  
      call ncvpt_doubl2(ncid,vid,start,tau_count,tem1,istatus)

      endif  !kopt.eq.0 .and. n.eq.0

c--------------------------------------------------------
c     Time-Dependent data:   (for all kopt=1,2 and all n)
c--------------------------------------------------------

      istatus= NF_INQ_VARID(ncid,'time',vid)   
      call ncvpt_doubl2(ncid,vid,start(4),1,timet,istatus)
      
      if ( netcdfshort.eq.'longer_b' ) then
         istatus= NF_INQ_VARID(ncid,'urfb',vid)  
         do ll=1,lrz
            start1(3)=ll
            call ncvpt_doubl2(ncid,vid,start1,count1,
     +           urfb(1,1,lrindx(ll),irfn(krf)), istatus) !-YuP: (1)->(irfn(krf))
         enddo
      endif

cUNNECESSARY      call ncclos(ncid,istatus)
      istatus=NF_CLOSE(ncid)


c$$$C-----------------------------------------------------------------------
c$$$c
c$$$cl    2. Restart from previous run
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
c$$$      xdim = ncdid(ncid,'x',istatus)
c$$$      ydim = ncdid(ncid,'y',istatus)
c$$$      rdim = ncdid(ncid,'z',istatus)
c$$$      tdim = ncdid(ncid,'tau',istatus)
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
c$$$c --- set the time-step counter ==> numrec(krf)
c$$$      call ncdinq(ncid,tdim,name,numrec(krf),istatus)
c$$$      start(4)=numrec(krf)
c$$$
c$$$c.......................................................................
c$$$cl    2.3 Read data
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
cl    3. Periodic save at each time-steps
c
c --- begin if ---
      if (n.gt.0 .or. nstop.eq.0) then    !Down to line 774
      istatus = NF_OPEN(t_, NF_WRITE, ncid) !-YuP: NetCDF-f77
c.......................................................................
cl    3.1 increment the counter for data recording:
      numrec(krf)=numrec(krf)+1 
      start(4)=numrec(krf)
      start1(4)=numrec(krf)
c.......................................................................
cl    3.2 Variables saved at each time-step

      istatus= NF_INQ_VARID(ncid,'time',vid)  
      call ncvpt_doubl2(ncid,vid,start(4),1,timet,istatus)
      
      if ( netcdfshort.eq.'longer_b' ) then
         istatus= NF_INQ_VARID(ncid,'urfb',vid)  
         do ll=1,lrz
            start1(3)=ll
            call ncvpt_doubl2(ncid,vid,start1,count1,
     +           urfb(1,1,lrindx(ll),irfn(krf)), istatus) !-YuP: (1)->(irfn(krf))
         enddo
      endif
cUNNECESSARY      call ncclos(ncid,istatus)
      istatus=NF_CLOSE(ncid)
c --- endif (n.gt.0 .or. nstop.eq.0) ---
      endif

c.......................................................................
cl    3.3 Variables saved only at last time-step

c --- begin if ---
      if (n.eq.nstop) then        !Down to line 1086
      istatus = NF_OPEN(t_, NF_WRITE, ncid) !-YuP: NetCDF-f77

c.....................................................................
c     Adjust data to standard external format specified in urfread_.f. 
c     (See urfread_.f. Perform inverse transformation)
c.....................................................................
c
        do iray=1,nray_krf
 
        do  is=1,nrayelt(iray,irfn(krf))
           wdnpar(is,iray,irfn(krf))=
     &        abs(wdnpar(is,iray,irfn(krf)))/wdscale(krf)
           !YuP: Careful! wdscale() was not duplicated into each mode,
           !so it is still a function of krf 
        enddo

c     Change sign of sbtot and wnpar, if bsign.lt.0.
c     (see urfwrite_.f and cqlinput_help)
        if (eqsource.eq."eqdsk" .or. eqsource.eq.'mirror1')then
        if (bsign.lt.0.d0) then
           sbsign=sign(one,bsign)
           if (iray.eq.1) then
              WRITE(*,*)'netcdfrf:eqsource,bsign,sbsign,bsign1(krf) = ',
     1                            eqsource,bsign,sbsign,bsign1(krf)
           endif
           !YuP: Careful! bsign1() was not duplicated into each mode,
           !so it is still a function of krf 
           do is=1,nrayelt(iray,irfn(krf))
           sbtot(is,iray,irfn(krf))=sbtot(is,iray,irfn(krf))/bsign1(krf)
              wnpar(is,iray,irfn(krf))=sbsign*wnpar(is,iray,irfn(krf))
              cwexde(is,iray,irfn(krf))=sbsign*cwexde(is,iray,irfn(krf))
              cweyde(is,iray,irfn(krf))=sbsign*cweyde(is,iray,irfn(krf))
           enddo
        endif
        endif
        
c     fluxn is renormalized:
        do is=1,nrayelt(iray,irfn(krf))
        fluxn(is,iray,irfn(krf))=fluxn(is,iray,irfn(krf))*(8.*pi)/clight
        enddo  !  is

c     shift z-position of ray data, if eqdsk has been
c     vertically shifted in subroutine equilib:
        if (zshift.ne.0) then
           do is=1,nrayelt(iray,irfn(krf))
              wz(is,iray,irfn(krf))=wz(is,iray,irfn(krf))+zshift
           enddo
        endif

        enddo     !  iray

c..................................................................

c--------------------------
c     Put data
c--------------------------
      istatus= NF_INQ_VARID(ncid,'nray',vid)  
      call ncvpt_int2(ncid,vid,1,1,nray_krf,istatus)

      istatus= NF_INQ_VARID(ncid,'freqcy',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,freqcy(irfn(krf)),istatus)
!      write(*,*)'netcdfrf: kopt,krf,freqcy(irfn(krf))=',
!     & kopt,krf,freqcy(irfn(krf))

      istatus= NF_INQ_VARID(ncid,'lh',vid)  
      call ncvptc2(ncid,vid,1,8,lh,8,istatus)

      istatus= NF_INQ_VARID(ncid,'fw',vid)  
      call ncvptc2(ncid,vid,1,8,fw,8,istatus)

      istatus= NF_INQ_VARID(ncid,'ech',vid)  
      call ncvptc2(ncid,vid,1,8,ech,8,istatus)

      istatus= NF_INQ_VARID(ncid,'nharm1',vid)  
      call ncvpt_int2(ncid,vid,1,1,nharm1(krf),istatus)
           !YuP: Careful! nharm1() was not duplicated into each mode,
           !so it is still a function of krf 

      istatus= NF_INQ_VARID(ncid,'nharms',vid)  
      call ncvpt_int2(ncid,vid,1,1,nharms(krf),istatus)
           !YuP: Careful! nharms() was not duplicated into each mode,
           !so it is still a function of krf 

      istatus= NF_INQ_VARID(ncid,'nrayelt',vid)  
      call ncvpt_int2(ncid,vid,1,nray_krf,nrayelt(:,irfn(krf)),istatus)

cBH170920: Added (1,1,krf) in below section of coding down to l 982, to get
cBH170920: krf dependent data going in to the respective krf data file.
      call pack21
     1  (ws(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'ws',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1  (spsi(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! dep. on krf
      istatus= NF_INQ_VARID(ncid,'spsi',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1  (wr(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'wr',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1  (wphi(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'wphi',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1  (wz(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'wz',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1 (wnpar(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'wnpar',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1 (wnper(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'wnper',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1(delpwr(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'delpwr',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1(urfpwrl(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,
     & neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'urfpwrl',vid)   !YuP[2020-09-24] added
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1(urfpwrc(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,
     & neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'urfpwrc',vid)   !YuP[2020-09-24] added
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1(sdpwr(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'sdpwr',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1(wdnpar(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'wdnpar',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      ii=0
      do j=1,nray_krf
         do i=1,neltmax ! depends on krf
            ii=ii+1
            urftmp(ii)=0.5*(cwexde(i,j,irfn(krf))
     &               +conjg(cwexde(i,j,irfn(krf))))
         enddo
      enddo
      do j=1,nray_krf
         do i=1,neltmax ! depends on krf
            ii=ii+1
            urftmp(ii)=-cei*0.5*(cwexde(i,j,irfn(krf))
     &                    -conjg(cwexde(i,j,irfn(krf))))
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'cwexde',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      ii=0
      do j=1,nray_krf
         do i=1,neltmax ! depends on krf
            ii=ii+1
            urftmp(ii)=0.5*(cweyde(i,j,irfn(krf))
     &        +conjg(cweyde(i,j,irfn(krf))))
         enddo
      enddo
      do j=1,nray_krf
         do i=1,neltmax ! depends on krf
            ii=ii+1
            urftmp(ii)=-cei*0.5*(cweyde(i,j,irfn(krf))
     &        -conjg(cweyde(i,j,irfn(krf))))
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'cweyde',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      ii=0
      do j=1,nray_krf
         do i=1,neltmax ! depends on krf
            ii=ii+1
            urftmp(ii)=0.5*(cwezde(i,j,irfn(krf))
     &       +conjg(cwezde(i,j,irfn(krf))))
         enddo
      enddo
      do j=1,nray_krf
         do i=1,neltmax ! depends on krf
            ii=ii+1
            urftmp(ii)=-cei*0.5*(cwezde(i,j,irfn(krf))
     &        -conjg(cwezde(i,j,irfn(krf))))
         enddo
      enddo
      istatus= NF_INQ_VARID(ncid,'cwezde',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)


      call pack21
     1 (fluxn(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'fluxn',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1 (sbtot(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'sbtot',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1  (sene(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'sene',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1(salphac(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,
     & neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'salphac',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)

      call pack21
     1  (salphal(1,1,irfn(krf)),1,nrayelts,1,nrayn,urftmp,
     & neltmax,nray_krf) ! depends on krf
      istatus= NF_INQ_VARID(ncid,'salphal',vid)  
      call ncvpt_doubl2(ncid,vid,start,ray_count,urftmp,istatus)


c --- full update of data file (last time-step only) ---

      if (netcdfshort.ne.'enabled' .and. netcdfshort.ne.'longer_b') then
         istatus= NF_INQ_VARID(ncid,'urfb',vid)  
         do ll=1,lrz
            start1(3)=ll
            call ncvpt_doubl2(ncid,vid,start1,count1,
     +           urfb(1,1,lrindx(ll),irfn(krf)),istatus) !-YuP: (1)->(krf)
         enddo
         
         if (netcdfshort.eq.'long_urf') then
            istatus= NF_INQ_VARID(ncid,'urfc',vid) 
            do ll=1,lrz
               start1(3)=ll
               call ncvpt_doubl2(ncid,vid,start1,count1,
     +              urfc(1,1,lrindx(ll),irfn(krf)),istatus) !-YuP: (1)->(krf)
            enddo
c            istatus= NF_INQ_VARID(ncid,'urfe',vid) 
c            do ll=1,lrz
c               start1(3)=ll
c               call ncvpt_doubl2(ncid,vid,start1,count1,
c     +              urfe(1,1,lrindx(ll),irfn(krf)),istatus) !-YuP: (1)->(krf)
c            enddo
c            istatus= NF_INQ_VARID(ncid,'urff',vid) 
c            do ll=1,lrz
c               start1(3)=ll
c               call ncvpt_doubl2(ncid,vid,start1,count1,
c     +              urff(1,1,lrindx(ll),irfn(krf)),istatus) !-YuP: (1)->(krf)
c            enddo
!YuP[03/18/2015] urfe,urff are expressed through urfb,urfc
! No need to save them.
         endif  !On netcdfshort
      endif  !On netcdfshort

c..................................................................
c     Re-Adjust data to internal format, 
c     in case this is not last step. 
c     (See urfread_.f)
c..................................................................
c
      do iray=1,nray_krf

        do  is=1,nrayelt(iray,irfn(krf))
           wdnpar_old= wdnpar(is,iray,irfn(krf))
           wdnpar(is,iray,irfn(krf))=
     &        wdscale(krf)*abs(wdnpar(is,iray,irfn(krf)))
           !YuP: Careful! wdscale() was not duplicated into each mode,
           !so it is still a function of krf 
        enddo

c     Change sign of sbtot and wnpar, if bsign.lt.0.
c     (see cqlinput_help)
        if (eqsource.eq."eqdsk" .or. eqsource.eq.'mirror1')then
        if (bsign.lt.0.d0) then
           sbsign=sign(one,bsign)
           if (iray.eq.1) then
              WRITE(*,*)'netcdfrf:eqsource,bsign,sbsign,bsign1(krf) = ',
     1                            eqsource,bsign,sbsign,bsign1(krf)
           endif
           !YuP: Careful! bsign1() was not duplicated into each mode,
           !so it is still a function of krf 
           do is=1,nrayelt(iray,irfn(krf))
           sbtot(is,iray,irfn(krf))=bsign1(krf)*sbtot(is,iray,irfn(krf))
              if (sbtot(is,iray,krf).lt.0.) 
     1           WRITE(*,*)'urfread: Sign Problem with sbtot:is,iray=',
     2           is,iray
              wnpar(is,iray,irfn(krf))=sbsign*wnpar(is,iray,irfn(krf))
              cwexde(is,iray,irfn(krf))=sbsign*cwexde(is,iray,irfn(krf))
              cweyde(is,iray,irfn(krf))=sbsign*cweyde(is,iray,irfn(krf))
           enddo
        endif
        endif
        
c     fluxn is renormalized to be as in Stix or Bekefi:
        do is=1,nrayelt(iray,irfn(krf))
        fluxn(is,iray,irfn(krf))=fluxn(is,iray,irfn(krf))*clight/(8.*pi)
        enddo

c     shift z-position of ray data, if eqdsk has been
c     vertically shifted in subroutine equilib:
        if (zshift.ne.0) then
           do is=1,nrayelt(iray,irfn(krf))
              wz(is,iray,irfn(krf))=wz(is,iray,irfn(krf))-zshift
           enddo
        endif

      enddo  !  iray

 
c..................................................................
      istatus = NF_CLOSE(ncid) !-YuP: NetCDF-f77
      call check_err(istatus)


c --- endif (n.eq.nstop) ---
      endif


C-----------------------------------------------------------------------
c --- endif ((kopt.eq.0) .or. (kopt.eq.1))
      endif




C-----------------------------------------------------------------------
C
C     READ NETCDF RAY DATA FILE.    kopt = 2,3,4
C     (Parallels urfread_.f)
C
C-----------------------------------------------------------------------
c --- begin if ---
      if (kopt.eq.2  .or.  kopt.eq.3  .or.  kopt.eq.4) then !Down to 1443
      ! read data;
      ! This part is called by urfread with krf=1:mrf to read all data.
      ! BUT if kopt=3 or 4, this part is called by urfsetup 
      ! to only read the values of nray(krf),nharm(krf),freqcy(krf), 
      ! and also nrayelt(:,krf) (when kopt=4).

c     Open existing netCDF file

c-YuP      ncid=ncopn(rffile(krf),NCNOWRITE,istatus)
c-YuP      write(*,*)'after ncopn ncid=',ncid,'istatus',istatus
c-YuP      if (istatus.ne.0) then
c-YuP         write(*,*)'   ***   Problem opening rf .nc data file   ***'
c-YuP         Stop
c-YuP      endif      
      istatus = NF_OPEN(rffile(krf), 0, ncid) !-YuP: NetCDF-f77
      if (istatus .NE. NF_NOERR) then         !-YuP: NetCDF-f77
         WRITE(*,*)'   ***   Problem opening rf .nc data file   ***'
         Stop
      endif                                   !-YuP: NetCDF-f77
               
c     Get dimension ID from dimension name
c-YuP:      neltdim=ncdid(ncid,'neltmax',istatus)
c-YuP:      nraydim=ncdid(ncid,'nrays',istatus)
c-YuP:      twodim=ncdid(ncid,'two',istatus)
      istatus= NF_INQ_DIMID(ncid,'neltmax',neltdim) ! dep. on krf 
      istatus= NF_INQ_DIMID(ncid,'nrays',  nraydim) ! dep. on krf
      istatus= NF_INQ_DIMID(ncid,'two',    twodim)  !-YuP: NetCDF-f77 get twodim

c     Query netcdf file for dimensions:
c-YuP:      call ncdinq(ncid,neltdim,name,neltmax,istatus)
c-YuP:      call ncdinq(ncid,nraydim,name,nrays,istatus)
c-YuP:      call ncdinq(ncid,twodim,name,ntwo,istatus)
      istatus= NF_INQ_DIM(ncid,neltdim,name,neltmax)  ! dep. on krf 
      istatus= NF_INQ_DIM(ncid,nraydim,name,nrays)    ! dep. on krf
      istatus= NF_INQ_DIM(ncid,twodim,name,ntwo)     !-YuP: NetCDF-f77 get ntwo

      !-YuP: Why needed ???      COMMENTING it out:
cBH110329  Now using urftmp temporary storage rather than temp1, etc.
cBH110329      if (neltmax*nrays*2 .gt. (iy+2)*(jx+2)) then
cBH110329        WRITE(*,*)'STOP in netcdfrf, neltmax*nrays*2.gt.(iy+2)*(jx+2)'
cBH110329        WRITE(*,*)'netcdfrf: neltmax,neltmax*nrays*2,(iy+2)*(jx+2)=',
cBH110329     +                       neltmax,neltmax*nrays*2,(iy+2)*(jx+2)
cBH110329c        STOP 'Increase iy or jx'
cBH110329      endif

c     Read data:
c     The following presupposes that the rank (number of dimensions)
c     is known for the input data. That is, the data conforms
c     to the above output data format. 
c     (Checks could be placed in the code.)

      istatus= NF_INQ_VARID(ncid,'nray',vid)  
      istatus= NF_GET_VAR1_INT(ncid,vid,(1),nray(krf)) !-YuP: NetCDF-f77
      
c     Variable nharm in toray.nc and 3d.nc, is called nharm1 in
c     the cql3d mnemonic_rf.nc file.  Allow for this.
      !!!-YuP       call NCPOPT(NCVERBOSL)
      istatus= NF_INQ_VARID(ncid,'nharm',vid)  
      !!!-YuP       call NCPOPT(NCVERBOS+NCFATAL)
      if (istatus.ne.0) then
      istatus= NF_INQ_VARID(ncid,'nharm1',vid)  
      endif
      istatus= NF_GET_VAR1_INT(ncid,vid,(1),nharm(krf)) !-YuP: NetCDF-f77

      istatus= NF_INQ_VARID(ncid,'freqcy',vid)  
      istatus= NF_GET_VAR1_DOUBLE(ncid,vid,(1),freqcy(krf)) !-YuP: NetCDF-f77

      omega(krf)=2.*pi*freqcy(krf) !ok to leave here.
      !YuP[2020-09-23] moved/copied this definition for omega to urfread.
      !In urfread, it is duplicated to all wave modes (krfmode=1:mrfn)

      if(kopt.eq.3) goto 300 !-> Finish reading (for urfsetup)
                             !   Close file. ----------------------
                             
c..................................................................
c     Check code dimensions are sufficient for given ray data
c     Send message if nray.gt.nrayn
c..................................................................
c     Several files, corresponding to ray data from different wave
c     modes, may be read.
      if (nray(krf).gt.nrayn) then
        nr=nrayn
        WRITE(*,200) nray(krf),nr
        nray(krf)=nrayn
      endif
 200  format("Ray tracing code provides more rays than CQL parameter",/,
     1  "nrayn is equipped to handle. Recompile CQL with nrayn=",i5,/,
     1  "CQL will proceed with the calculation with nray reset to",i5)
c..................................................................

      ray_count(1)=neltmax ! depends on krf
      ray_count(2)=nrays   ! depends on krf
      ray_count(3)=2

      istatus= NF_INQ_VARID(ncid,'nrayelt',vid)  
      istatus= NF_GET_VARA_INT(ncid,vid,(1),ray_count(2),nrayelt(:,krf))

      
      if(kopt.eq.4) goto 300 !-> Finish reading (for urfsetup)
                             !   Close file. ----------------------

      
      if (nrays.gt.nrayn .or. neltmax.gt.nrayelts) then
         WRITE(*,*)'netcdfrf:  nrays,neltmax= ', nrays,neltmax
         WRITE(*,*)'netcdfrf:  Need to be .le. nrayn,nrayelts'
         WRITE(*,*)'netcdfrf:  nrayn,nrayelts= ',nrayn,nrayelts
         STOP
      endif

c-YuP:      vid=ncvid(ncid,'ws',istatus)
      istatus= NF_INQ_VARID(ncid,'ws',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(ws(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c-YuP:      vid=ncvid(ncid,'spsi',istatus)
      istatus= NF_INQ_VARID(ncid,'spsi',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(spsi(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c-YuP:      vid=ncvid(ncid,'wr',istatus)
      istatus= NF_INQ_VARID(ncid,'wr',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(wr(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c-YuP:      vid=ncvid(ncid,'wphi',istatus)
      istatus= NF_INQ_VARID(ncid,'wphi',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(wphi(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c-YuP:      vid=ncvid(ncid,'wz',istatus)
      istatus= NF_INQ_VARID(ncid,'wz',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(wz(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c-YuP:      vid=ncvid(ncid,'wnpar',istatus)
      istatus= NF_INQ_VARID(ncid,'wnpar',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(wnpar(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c-YuP:      vid=ncvid(ncid,'wnper',istatus)
      istatus= NF_INQ_VARID(ncid,'wnper',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(wnper(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

      istatus= NF_INQ_VARID(ncid,'delpwr',vid)  
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(delpwr(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c     The toray .nc file puts out variable sdpwri rather
c     than sdpwr.  But toray always has sdpwri=0.
c     Trap around the difficulty, if there is no sdpwr variable.
      !!!-YuP       call NCPOPT(NCVERBOS)
c-YuP:      vid=ncvid(ncid,'sdpwr',istatus)
      istatus= NF_INQ_VARID(ncid,'sdpwr',vid)  
      if (istatus.ne.0) then
         WRITE(*,*)
         WRITE(*,*)'***************************************************'
         WRITE(*,*)'netcdfrf: sdpwr istatus =',istatus
         WRITE(*,*)'sdpwr variable not found, but not needed from toray'
         WRITE(*,*)'***************************************************'
         WRITE(*,*)
      else
c-YuP:         call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
         call unpack21(sdpwr(1,1,krf),1,nrayelts,1,nrayn,
     1        urftmp,neltmax,nray(krf)) ! depends on krf
      endif
      !!!-YuP       call NCPOPT(NCVERBOS+NCFATAL)

c     The toray .nc file does not contain variable wdnpar.
c     If there is trouble reading wdnpar (istatus.ne.0),
c     then wdnpar is set equal to the standard value from
c     toray, corresponding to the DIII-D half angle at half-power EC
c     power pattern equal to 1.7 degrees.
      !!!-YuP       call NCPOPT(NCVERBOS)
c-YuP:      vid=ncvid(ncid,'wdnpar',istatus)
      istatus= NF_INQ_VARID(ncid,'wdnpar',vid)  
      if (istatus.ne.0) then
         WRITE(*,*)
         WRITE(*,*)'***************************************************'
         WRITE(*,*)'netcdfrf: wdnpar istatus =',istatus
         WRITE(*,*)'wdnpar variable not found, set equal to standard'
         WRITE(*,*)'DIII-D value 0.006297 for half-width at half-power'
         WRITE(*,*)'equal to 1.7 degrees.'
         WRITE(*,*)'***************************************************'
         WRITE(*,*)
         do j=1,nray(krf)
            do i=1,neltmax ! depends on krf
               wdnpar(i,j,krf)=0.006297
            enddo
         enddo
      else
c-YuP:         call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
         istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
         call unpack21(wdnpar(1,1,krf),1,nrayelts,1,nrayn,
     1        urftmp,neltmax,nray(krf)) ! depends on krf
      endif
      !!!-YuP       call NCPOPT(NCVERBOS+NCFATAL)
      
c-YuP:      vid=ncvid(ncid,'cwexde',istatus)
      istatus= NF_INQ_VARID(ncid,'cwexde',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      ii=0
      len_ii=nray(krf)*neltmax ! depends on krf
      do j=1,nray(krf)
         do i=1,neltmax ! depends on krf
            ii=ii+1
            cwexde(i,j,krf)=cmplx(urftmp(ii),urftmp(ii+len_ii))
         enddo
      enddo

c-YuP:      vid=ncvid(ncid,'cweyde',istatus)
      istatus= NF_INQ_VARID(ncid,'cweyde',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      ii=0
      len_ii=nray(krf)*neltmax ! depends on krf
      do j=1,nray(krf)
         do i=1,neltmax ! depends on krf
            ii=ii+1
            cweyde(i,j,krf)=cmplx(urftmp(ii),urftmp(ii+len_ii))
         enddo
      enddo

c-YuP:      vid=ncvid(ncid,'cwezde',istatus)
      istatus= NF_INQ_VARID(ncid,'cwezde',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      ii=0
      len_ii=nray(krf)*neltmax ! depends on krf
      do j=1,nray(krf)
         do i=1,neltmax ! depends on krf
            ii=ii+1
            cwezde(i,j,krf)=cmplx(urftmp(ii),urftmp(ii+len_ii))
         enddo
      enddo

c-YuP:      vid=ncvid(ncid,'fluxn',istatus)
      istatus= NF_INQ_VARID(ncid,'fluxn',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(fluxn(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c-YuP:      vid=ncvid(ncid,'sbtot',istatus)
      istatus= NF_INQ_VARID(ncid,'sbtot',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(sbtot(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c-YuP:      vid=ncvid(ncid,'sene',istatus)
      istatus= NF_INQ_VARID(ncid,'sene',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(sene(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c-YuP:      vid=ncvid(ncid,'salphac',istatus)
      istatus= NF_INQ_VARID(ncid,'salphac',vid)  
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(salphac(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c-YuP:      vid=ncvid(ncid,'salphal',istatus)
      istatus= NF_INQ_VARID(ncid,'salphal',vid)  
cBH050318      call unpack21(salphal(1,1,krf),1,nrayelts,1,nrayn,
cBH050318     1     urftmp,neltmax,nray(krf))
c-YuP:      call ncvgt(ncid,vid,start,ray_count,urftmp,istatus)
      istatus= NF_GET_VARA_DOUBLE(ncid,vid,start,ray_count,urftmp) !-YuP: NetCDF-f77
      call unpack21(salphal(1,1,krf),1,nrayelts,1,nrayn,
     1     urftmp,neltmax,nray(krf)) ! depends on krf

c..................................................................
c     Adjust data to internal cql3d format. 
c     (See urfread_.f)
c..................................................................
c
      do iray=1,nray(krf)

        wdnpar_old_min=minval(wdnpar(1:nrayelt(iray,krf),iray,krf))
        wdnpar_old_max=maxval(wdnpar(1:nrayelt(iray,krf),iray,krf))
        do  is=1,nrayelt(iray,krf)
           wdnpar(is,iray,krf)=wdscale(krf)*abs(wdnpar(is,iray,krf))
        enddo
        wdnpar_new_min=minval(wdnpar(1:nrayelt(iray,krf),iray,krf))
        wdnpar_new_max=maxval(wdnpar(1:nrayelt(iray,krf),iray,krf))
!YuP        write(*,'(a,2i4,4e12.3)')
!YuP     +   'netcdfrf(2): iray,krf, min/max for wdnpar_old,new:',
!YuP     +        iray, krf, wdnpar_old_min,wdnpar_old_max,
!YuP     +                   wdnpar_new_min,wdnpar_new_max

c     Change sign of sbtot and wnpar, if bsign.lt.0.
c     (see cqlinput_help and further explanation in urfread_.f)
        if (eqsource.eq."eqdsk" .or. eqsource.eq.'mirror1')then
        if (bsign.lt.0.d0) then
           sbsign=sign(one,bsign)
           if (iray.eq.1) then
              bsign1(krf)=sign(one,sbtot(1,iray,krf))
              WRITE(*,*)'netcdfrf:eqsource,bsign,sbsign,bsign1(krf) = ',
     1                            eqsource,bsign,sbsign,bsign1(krf)
           endif
           do is=1,nrayelt(iray,krf)
              sbtot(is,iray,krf)=bsign1(krf)*sbtot(is,iray,krf)
              if (sbtot(is,iray,krf).lt.0.) 
     1          WRITE(*,*)'urfread: Sign Problem with sbtot, is,iray=',
     2          is,iray
              wnpar(is,iray,krf)=sbsign*wnpar(is,iray,krf)
              cwexde(is,iray,krf)=sbsign*cwexde(is,iray,krf)
              cweyde(is,iray,krf)=sbsign*cweyde(is,iray,krf)
           enddo
        endif
        endif
        
c     fluxn is renormalized to be as in Stix or Bekefi:
        do is=1,nrayelt(iray,krf)
           fluxn(is,iray,krf)=fluxn(is,iray,krf)*clight/(8.*pi)
        enddo

c     shift z-position of ray data, if eqdsk has been
c     vertically shifted in subroutine equilib:
        if (zshift.ne.0) then
           do is=1,nrayelt(iray,krf)
              wz(is,iray,krf)=wz(is,iray,krf)-zshift
           enddo
        endif
          
      enddo   ! iray


300   istatus = NF_CLOSE(ncid) !-YuP: close file
      call check_err(istatus)

c --- endif --- (kopt = 2  or  3 or 4)
      endif


      return
      end





C==========================================================================



      subroutine netcdf_rdcb(krf)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

c     This subroutine uses netCDF-2 subroutines. 
c     http://www.unidata.ucar.edu/packages/netcdf/index.html
c     (Had a problem with linking to netCDF-3 subroutines
c      on account of embedded underscores in their names). 
c
c     This subroutine writes an DC file of urfb0 diffusion coeffs.
c       into a netCDF file with name "mnemonic"_rdc."krf".nc,
c       where "krf" is numeric value of krf.
c
c --- include file for netCDF declarations 
c --- (obtained from NetCDF distribution)
      include 'netcdf.inc'

c --- some stuff for netCDF file ---
      character*128 name
      integer ncid,vid,istatus
      integer xdim,ydim,rdim,r0dim,tdim
      integer dims(4),start(4),count(4)
      integer start1(3),count1(3)
      integer y_dims(2),y_count(2)
      integer tau_dims(2),tau_count(2)

      data start/1,1,1,1/, start1/1,1,1/
      
CMPIINSERT_IF_RANK_NE_0_RETURN

      WRITE(*,*)'Begin of netcdf_rdc, krf=',krf

c$$$C-----------------------------------------------------------------------
c$$$C     Only set up for cqlpmod.ne."enabled",ngen=1, for the time being.
c$$$C-----------------------------------------------------------------------
c$$$
c$$$      if (ngen.ne.1) then
c$$$         WRITE(*,*) 'WARNING: netcdf_rdcb subroutine not implemented'
c$$$         WRITE(*,*) '         for ngen.gt.1'
c$$$         return
c$$$      endif

cBH180515:  Valid for multiple rdc files and ngen.ge.1

c     Following counting vectors set up to facilitate writing
c     of the various arrays. Set up here to ensure initialization
c     in each call to the subroutine.

      count(1)=iymax
      count(2)=jx
      count(3)=lrz
      count(4)=1

      count1(1)=iymax
      count1(2)=jx
      count1(3)=1

      y_count(1)=iymax
      y_count(2)=lrz

      tau_count(1)=iy
      tau_count(2)=lrzmax


c.......................................................................
cl    1.1.1 create netCDF filename (Entering define mode.)
c     integer function nccre(filename,overwrite?,error_code)
c     Ref to page 46 of NetCDF-2 manual.
c     CLOBber old file, if it exists.
c     istatus is 0, if no errors.

      write(t_,100) mnemonic(1:length_char(mnemonic)),krf
 100  format(a,"_rdc.",i1,".nc")
      WRITE(*,*)'netcdf_rdcb:t_ = ',t_(1:length_char(t_))
c-YuP:      ncid=nccre(t_,NCCLOB,istatus)
      istatus = NF_CREATE(t_, NF_CLOBBER, ncid) !-YuP: NetCDF-f77
      call check_err(istatus)

c.......................................................................
cl    1.1.2 define dimensions
c     p. 67 of netcdf-2 manual
c     integer function ncddef(ncid,dim_name,dim_siz,error_code)
c       returns dimension id.

c-YuP:      xdim=ncddef(ncid,'x',jx,istatus)
c-YuP:      ydim=ncddef(ncid,'y',iymax,istatus)
c-YuP:      rdim=ncddef(ncid,'r',lrz,istatus)
c-YuP:      r0dim=ncddef(ncid,'r0',lrzmax,istatus)
      istatus= NF_DEF_DIM(ncid, 'x',  jx ,    xdim)  !-YuP: NetCDF-f77
      istatus= NF_DEF_DIM(ncid, 'y',  iymax,  ydim)
      istatus= NF_DEF_DIM(ncid, 'r',  lrz,    rdim)
      istatus= NF_DEF_DIM(ncid, 'r0', lrzmax, r0dim) !-YuP: NetCDF-f77


c     unlimited dimension for time, dimension name= 'time'
c     ncddef(ncid,'time',NF_UNLIMITED,error_code)
c-YuP:      tdim=ncddef(ncid,'time',NCUNLIM,istatus)
      istatus= NF_DEF_DIM(ncid, 'time',NF_UNLIMITED,tdim) !-YuP: NetCDF-f77

c     define vector of dimensions [for urfb(iy,jx,lrz,mrfn)], unlimited last
      dims(1)=ydim
      dims(2)=xdim
      dims(3)=rdim
      dims(4)=tdim

      y_dims(1)=ydim
      y_dims(2)=rdim

      tau_dims(1)=ydim
      tau_dims(2)=r0dim

c--------------------------
c     Additional Time-independent data
c--------------------------


      vid=ncvdef2(ncid,'lrzmax',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +            'Number of radial surfaces',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'rya',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,22,
     +           'Normalized radial mesh',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'rpconz',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +           'Major radius at outside of flux surface',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'rhomax',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'lrz',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +            'Number of FPd radial surfaces',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'lrindx',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +           'Radial indices of FPd surfaces',istatus)

      vid=ncvdef2(ncid,'jx',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +            'momentum-per-mass dimension',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'x',NCDOUBLE,1,xdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +           'normalized momentum-per-mass',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'vnorm',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +           'velocity (momentum-per-mass) norm',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +                     'cms/sec',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'enorm',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +                     'Energy normalization',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'keV',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'iy',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,21,
     +            'Pitch angle dimension',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'y',NCDOUBLE,2,y_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,11,
     +           'pitch angle',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +           'radians',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'iy_',NCLONG,1,rdim,istatus)

      vid=ncvdef2(ncid,'itl',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,26,
     +           'lower trapped-passing bndy',istatus)

      vid=ncvdef2(ncid,'itu',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,26,
     +           'upper trapped-passing bndy',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'tau',NCDOUBLE,2,tau_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +           'tau_bounce * abs(speed)',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'cms',istatus)


      vid=ncvdef2(ncid,'rdcb',NCDOUBLE,3,dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,40,
     +           'u^2 * BA QL Diff Coeff * v_par*tau_bounce',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,12,
     +           'cgs/vnorm**4',istatus)
      call check_err(istatus)

      if (netcdfshort.eq.'long_urf') then

         vid=ncvdef2(ncid,'rdcc',NCDOUBLE,3,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,51,
     +     'u * BA<c/(psi**0.5*c0) QL Dtu> * v_par0*tau_bounce',istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +        'Above: c=cos(theta),c0=c at B0, t=theta',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,12,
     +        'cgs/vnorm**3',istatus)
         vid=ncvdef2(ncid,'rdce',NCDOUBLE,3,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,42,
     +        'u * BA<c*s/(psi*c0) QL Dut> * v_par0*tau_bounce',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,12,
     +        'cgs/vnorm**3',istatus)
         vid=ncvdef2(ncid,'rdcf',NCDOUBLE,3,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,55,
     +        'BA<c**2*s/(psi**(3/2)*c0**2) QL Dtt> * v_par*tau_bounce',
     +        istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,12,
     +        'cgs/vnorm**2',istatus)
         call check_err(istatus)

      endif  ! On netcdfshort

c.......................................................................
cl    1.1.4 end the define-mode and start the data-mode
c     p. 51-2 of manual

c-YuP:      call ncendf(ncid,istatus)
      istatus= NF_ENDDEF(ncid) !-YuP: NetCDF-f77
      call check_err(istatus)


C-----------------------------------------------------------------------
cl    1.2 Write data
c

c --- set the time-step counter ==> numrec_rdcb
      numrec_rdcb=1
      start(4)=numrec_rdcb

c --- initialize data file ---
c     First get variable_id: 
c        integer function ncvid(ncid,variable_name,error_code)
c        returns varid
c     Then write data with nc variable_put:
c        ncvpt(ncid,varid,start_indices,counts,values,error_code)
c
c     p. 79, 89 of netcdf-2 manual


c--------------------------
c     Time-independent data
c--------------------------

c-YuP:      vid=ncvid(ncid,'lrzmax',istatus)
      istatus= NF_INQ_VARID(ncid,'lrzmax',vid)  
      call ncvpt_int2(ncid,vid,1,1,lrzmax,istatus)

c-YuP:      vid=ncvid(ncid,'rya',istatus)
      istatus= NF_INQ_VARID(ncid,'rya',vid)  
      call ncvpt_doubl2(ncid,vid,1,tau_count(2),rya(1),istatus)

c-YuP:      vid=ncvid(ncid,'rpconz',istatus)
      istatus= NF_INQ_VARID(ncid,'rpconz',vid)  
      call ncvpt_doubl2(ncid,vid,1,tau_count(2),rpconz(1),istatus)

c-YuP:      vid=ncvid(ncid,'rhomax',istatus)
      istatus= NF_INQ_VARID(ncid,'rhomax',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,rhomax,istatus)

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

c-YuP:      vid=ncvid(ncid,'vnorm',istatus)
      istatus= NF_INQ_VARID(ncid,'vnorm',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,vnorm,istatus)

c-YuP:      vid=ncvid(ncid,'enorm',istatus)
      istatus= NF_INQ_VARID(ncid,'enorm',vid)  
      call ncvpt_doubl2(ncid,vid,1,1,enorm,istatus)

c-YuP:      vid=ncvid(ncid,'iy',istatus)
      istatus= NF_INQ_VARID(ncid,'iy',vid)  
      call ncvpt_int2(ncid,vid,1,1,iymax,istatus)

      if (iy*lrors.gt.iyjx2) stop 'netcdfrf:  Need to set jx>lrza'
      call pack21(y,1,iy,1,lrors,tem1,iymax,lrors)
c-YuP:      vid=ncvid(ncid,'y',istatus)
      istatus= NF_INQ_VARID(ncid,'y',vid)  
      call ncvpt_doubl2(ncid,vid,start,y_count,tem1,istatus)

c-YuP:      vid=ncvid(ncid,'iy_',istatus)
      istatus= NF_INQ_VARID(ncid,'iy_',vid)  
      call ncvpt_int2(ncid,vid,1,lrz,iy_,istatus)

c-YuP:      vid=ncvid(ncid,'itl',istatus)
      istatus= NF_INQ_VARID(ncid,'itl',vid)  
      call ncvpt_int2(ncid,vid,1,lrz,itl_,istatus)

c-YuP:      vid=ncvid(ncid,'itu',istatus)
      istatus= NF_INQ_VARID(ncid,'itu',vid)  
      call ncvpt_int2(ncid,vid,1,lrz,itu_,istatus)
      
      call pack21(tau,1,iy,1,lrzmax,tem1,iy,lrzmax)
c-YuP:      vid=ncvid(ncid,'tau',istatus)
      istatus= NF_INQ_VARID(ncid,'tau',vid)  
      call ncvpt_doubl2(ncid,vid,start,tau_count,tem1,istatus)

c     rdcb coefficient write

c-YuP:      vid=ncvid(ncid,'rdcb',istatus)
      istatus= NF_INQ_VARID(ncid,'rdcb',vid)  
      do ll=1,lrz
         start1(3)=ll
         call ncvpt_doubl2(ncid,vid,start1,count1,rdcb(1,1,lrindx(ll),
     +              krf),istatus)
         if (istatus.ne.NF_NOERR) WRITE(*,*) 'netcdf_rdc, ll= :',ll
         call check_err(istatus)
      enddo

c     Additional rfc.... coefficients write
      if (netcdfshort.eq.'long_urf') then

      istatus= NF_INQ_VARID(ncid,'rdcc',vid)  
      do ll=1,lrz
         start1(3)=ll
         call ncvpt_doubl2(ncid,vid,start1,count1,rdcc(1,1,lrindx(ll),
     +              krf),istatus)
         if (istatus.ne.NF_NOERR) WRITE(*,*) 'netcdf_rdc, ll= :',ll
         call check_err(istatus)
      enddo

      istatus= NF_INQ_VARID(ncid,'rdce',vid)  
      do ll=1,lrz
         start1(3)=ll
         call ncvpt_doubl2(ncid,vid,start1,count1,rdce(1,1,lrindx(ll),
     +              krf),istatus)
         if (istatus.ne.NF_NOERR) WRITE(*,*) 'netcdf_rdc, ll= :',ll
         call check_err(istatus)
      enddo

      istatus= NF_INQ_VARID(ncid,'rdcf',vid)  
      do ll=1,lrz
         start1(3)=ll
         call ncvpt_doubl2(ncid,vid,start1,count1,rdcf(1,1,lrindx(ll),
     +              krf),istatus)
         if (istatus.ne.NF_NOERR) WRITE(*,*) 'netcdf_rdc, ll= :',ll
         call check_err(istatus)
      enddo


      endif  ! On netcdfshort

c     Close netcdf file
c-YuP:      call ncclos(ncid,istatus)
      istatus = NF_CLOSE(ncid) !-YuP: NetCDF-f77
      
      call check_err(istatus)

      return
      end

