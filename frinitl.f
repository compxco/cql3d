c
c
      subroutine frinitl
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'frname_decl.h'
      character*8 machinei

      ep100=1.d+100

c..................................................................
c     Jump past the "fr" namelist if "fr" is disabled.
c..................................................................

c     some preset to have clean namelist output
        do i=1,nap
          do j=1,kb
            ashape(i,j) = 'disabled'
          end do
        end do
        do j=1,kb
           bshape(j)='disabled'
        enddo

c..................................................................
c     This routine sets defaults and reads in data for the
c     neutral beam (NFREYA) code as it exists in ONETWO.
c..................................................................

c
c     NEUTRAL BEAM HEATING PARAMETERS
c--------------------------------------------------------------------

c******************************************************************
c     BELOW TO **** NOT USED IN CQL3D  **** ONETWO DIVERGENCE

c     timbplt    Times (up to 5) to produce data for Freya -like plots
c     of beam deposition. o/p is processed by nubplt.
c     Defaulted to off.
c     timbplt(1).le.time0.and.beamon.lt.time0 gives o/p
c     for initial time.
c     beamon      Time (sec) at which beam heating is turned on
c     btime       Time interval (sec) during which beam heating is on
c     ibcur       Flag for neutral beam driven current
c     1, include beam driven current
c     0, neglect beam driven current
c     ibcx        Flag for secondary charge exchange between fast ions
c     and thermal neutrals
c     1, include secondary charge exchange
c     0, neglect secondary charge exchange
c     ibslow      Flag for neutral beam slowing down
c     1, include neutral beam slowing down
c     0, neglect neutral beam slowing down
c     fionx       Allows testing for non-classical slowing down
c     (see subroutine slow1).
c     nameb       Name of neutral species in beam
c     'h', 'd', 't', 'dt'
c     Must be primary species #1 or #2.
c     relnub      Minimum relative change in ion density or electron
c     temperature between neutral beam calculations (suggest 0.10)
c      tfusbb      'thermal fusion balance for beams', fraction by which
c      the net energy gain from thermal fusion must exceed
c      the net energy loss for automatic beam turnoff.
c      If tfusbb = 0., automatic beam turnoff is not done.
c      iddcal     Flag controlling treatment of beam effects on fuscal,
c      the calculated fusion neutron rate:
c      0 = do not include knock-on or beam-d neutrons in fuscal
c      1 = include only knock-on neutrons in fuscal
c      2 = include only beam-d neutrons in fuscal
c      3 = include both knock-on and beam-d neutrons in fuscal
c     fdbeam     Fraction of deuterium atoms in neutral beam.

c     ABOVE NOT USED IN CQL3D** ONETWO DIVERGENCE**
c******************************************************************
c     ranseed    Starting seed for random number generator used in the
c     Freya determination of the beam deposition.
c     npart       Number of particles followed into plasma
c     (suggest 10000)
c     npskip     Ratio of number of particles followed into plasma
c     to number of source particles (suggest 1)
c     iborb      Flag for modeling orbit effects on beam-
c     generated fast ions
c     1, model orbit effects
c     0, do not model orbit effects
c     itrapfi    Flag for trapped fast ions.  If iborb=1 then
c     setting itrapfi=1 will modify the beam driven
c     current by the initial trapped ion fraction.
c     (Subsequent pitch angle diffusion is not!!! taken into
c     account).  itrapfi=0 neglects this effect.  If iborb=0,
c     itrapfi has no affect.  itrapfi=0 is default
c     iexcit    excited beam state option
c     0,  do not use excited state cross sections (default)
c     1,  use hexnb package but do not include excitations in
c     its calculation of cross sections
c     2,  use hexnb package, include excitations in calculation
c     of cross sections
c     inubpat   two-dimensional beam deposition option
c     0,  do not calculate beam deposition in (r,z) coordinates
c     1,  calculate beam deposition on eqdisk (r,z) coordinates,
c     write third excited state population to file 'beamdep'
c     2,  same as inubpat=1 except rescale (r,z) grid according
c     to npat (see below)
c     if inubpat.gt.0, iexcit is automatically set, iexcit=2
c     npat      modified (r,z) grid dimensions, used only for inubpat=2
c     npat(1)=number of elements in 'r' (<=2*nnra)
c     npat(2)=number of elements in 'z' (<=2*nnza)
c     defaults, npat(1)=nnra, npat(2)=nnza
c     mf not read in - ONETWO DIVERGENCE
c     mf         Number of flux zones plus 1 (max = 81)
c
c     In the following list the index ib designates the beam injector,
c     while ie refers to one of the three energy components.
c     iap refers to one of the aperatures.
c
c     nbeams          Number of neutral beam injectors (.le.kb)
c     nsourc          Number of sources per beamline.
c     If 1, source is centered on beamline axis.
c     If nsourc=2, distinguish between the beamline
c     axis and the source centerline (optical axis).
c     The two sources are assumed to be mirror images
c     through the beamline axis.
c     In either case, the exit grid plane is perpendicula
c     to the beamline axis, and contains the source
c     exit grid center(s).
c     If nsourc=2, the alignment of the sources w.r.t.
c     the beamline axis is specified through bhofset,
c     bvofset, and bleni (described further below).
c     naptr           Total number of aperatures encountered by a particle
c     as is moves from the source into the plasma chamber
c     Maximum is specified by parameter nap (=10).
c     First set of aperatures encountered by the particle
c     are assumed centered on the source axis, and subseq
c     aperatures are centered on the beamline axis; the
c     distinction is made through ashape.
c     anglev(ib)      Vertical angle (degrees) between optical axis
c     and horizontal plane; a positive value indicates
c     particles move upward
c     angleh(ib)      Horizontal angle (degrees) between optical axis and
c     vertical plane passing through pivot point and
c     toroidal axis; a zero value denotes perpendicular
c     injection, while a positive value indicates par-
c     ticles move in the co-current direction
c     bvofset(ib)     Vertical offset from beamline axis to center
c     of each source (cm; used only for nsourc=2)
c     bhofset(ib)     Horizontal offset from beamline axis to center
c     of each source (cm; used only for nsourc=2)
c     bleni(ib)       Length along source centerline (source optical axis)
c     source to intersection point with the beamline axis
c     sfrac1(ib)      Fraction of source current per beamline coming
c     from upper source (used only for nsourc=2)
c     bcur(ib)        Total current (a) in ion beam (used only if bptor
c     is zero)
c     bptor(ib)       Total power (w) through aperture into torus; when
c     nonzero, bptor takes precedence over bcur
c     bshape(ib)      Beam shape
c     'circ' : circular
c     'rect' : rectangular
c     'rect-lps':  rect. long pulse source (DIII-D only)
c     a choice of short or long pulse sources is
c     available by injector (not by source).  on
c     or both injectors may be long pulse by
c     setting one or both to 'rect-lps'
c     CAUTION:  DIII-D sources are defaulted to lps specs.
c     It is the user's responsibility to overide these for
c     sps configuration(s).
c     bheigh(ib)      Height of source (cm)
c     Default is bshape(1)=bshape(2)='rect-lps'
c     bwidth(ib)      Width of source (cm); diameter for
c     circular source.
c     bhfoc(ib)       Horizontal focal length of source (cm)
c     bvfoc(ib)       Vertical focal length of source (cm)
c     bhdiv(ib)       Horizontal divergence of source (degrees)
c     bvdiv(ib)       Vertical divergence of source (degrees)
c     ebkev(ib)       Maximum particle energy in source (keV)
c     fbcur(ie,ib)    Fraction of current at energy ebkeV/ie
c     ashape(iap,ib)  Aperture shape.
c     Prefix 's-' indicates source axis centered.
c     Prefix 'b-' indicates beamline axis centered.
c     's-circ'          'b-circ'
c     's-rect'          'b-rect'
c     's-vert'          'b-vert'
c     's-horiz'         'b-horiz'
c     'b-d3d'
c     (circ=circular aperature, rect=rectagular,
c     vert=limits vertical height of source particles,
c     horiz=limits horizontal height of source particles,
c     d3d= special DIII-D polygonal aperature)
c     aheigh(iap,ib)  Height of aperture (cm)
c     awidth(iap,ib)  Width of aperture (cm); diameter for circular
c     aperture
c     alen(iap,ib)    Length from source to aperature for 's-type' aperatur
c     and from exit grid plane along beamline axis for
c     'b-type' aperatures.
c     blenp(ib)       Distance along beamline axis from source exit
c     plane to the fiducial "pivot" point.
c     rpivot(ib)      Radial position of pivot (cm)
c     zpivot(ib)      Axial position of pivot (cm)
c
c     hdepsmth        set this parm to a positive value (gt.0.0) to turn of
c     the smoothing of hibrz and hdepz in sub postnub.
c     If this option is used then enough zones must be spec
c     for adequate resolution (zones=number of radial grid
c     and enough injected neutrals must be followed to minimize
c     the statistical noise enough so that no greatly uneven
c     profiles result! this option was added because the smoothing
c     of the profiles by subroutine smooth can lead to unphysical
c     forms of the birth and deposition profiles.
c--------------------------------------------------------------------
c     ONETWO DIVERGENCE BELOW
      nprim=1  !number of primary species 
             !(see the scan over kfrsou(ib) in frset to determine nprim)
      nimp=0   !If impurities are present, should include at least =1.
      machinei="iter"
      if (machinei.eq."iter") go to 1660
      do 1650 i=1,5
        timbplt(i)=1000.
 1650 continue
      beamon = 1000.  ! Not used for anything
      btime  = 0.     ! Not used for anything
      ibcur = 1
      itrapfi=0
      ibcx = 1
      ibslow = 1
      fionx = 0.0
      nameb = 'h'
      relnub=.10 !Not used in CQL3D
cNLremoved      tfusbb = 0.
cNLremoved      iddcal = 3
cNLremoved      fdbeam = 0.150e-3

 1660 continue
cBH091020:  Adding above defaults which were skipped for some unknown
cBH091020:  reason, for machinei='iter'
cBH091020:  Simply zeroing.
      beamon = 0.  ! Not used for anything
      btime  = 0.  ! Not used for anything
      ibcur = 0
      itrapfi=0
      ibcx = 0
      ibslow = 0
      fionx = 0.0
c      nameb = 'h'
      relnub=0. !Not used in CQL3D
      
      nameb = 'd'
      ranseed=dble(7**7) !YuP[2021-01-10] Added dble
c     ONETWO DIVERGENCE
      npart= 150000 ! YuP-101211: default value. Was: npart=nap
      hdepsmth=-1.0
c---smoothing is normally on by above line
      npskip = 1
c     ONETWO DIVERGENCE BELOW
      iborb = 0
      inubpat=0
      npat(1)=nnra
      npat(2)=nnza
      mf = 41
      nbeams = 1
      ne_tk=0 !YuP[2022-12-15] Added initialization
!     ne_tk= number of equi-width energy bins used in forming
!      n*sigma array.  required for nonzero rotation cases. 
      ds_tk=0.d0 !Also, for rotation !YuP[2022-12-19] Added initialization
      fe_tk=0.d0 !Also, for rotation !YuP[2022-12-19] Added initialization

c
c     ONETWO DIVERGENCE BELOW
      nsourc=1
c
      if (ke.lt.3) then
         WRITE(*,*)'frinitl:  parameter ke needs to be .ge.3'
         stop
      endif
c
c     DIII beam input
      if(machinei.ne.'doub-iii')go to 9020
      naptr=2
      do 1700 i=1,kb
        anglev(i)  = 0.
        angleh(i)  = 13.5
        bvofset(i)=39.75
        bhofset(i)=0.0
        bleni(i)=553.88
        bcur(i)    = 110.
        bptor(i)   = 3.5e6
        bshape(i)  = 'rect'
        bheigh(i)  = 10.
        bwidth(i)  = 40.
        bhfoc(i)   = 480.
        bvfoc(i)   = 550.
        bhdiv(i)   = 1.4
        bvdiv(i)   = .45
        ebkev(i)   = 80.0
        if (ke.lt.3) then
           WRITE(*,*)'frinitl:  parameter ke needs to be .ge.3'
           stop
        else
           fbcur(1,i) = .6
           fbcur(2,i) = .3
           fbcur(3,i) = .1
        endif
        if (ke.ge.4) then
           do k=4,ke
              fbcur(k,i)=0.
           enddo
        endif
        sfrac1(i)  = 0.5
        blenp(i) = 486.13
        rpivot(i)  = 270.
        zpivot(i)  = 89.
        if (nap.lt.2) then
           WRITE(*,*)'frinitl:  parameter nap needs to be .ge.2'
           stop
        else
           ashape(1,i)='s-vert'
           ashape(2,i)='b-horiz'
           aheigh(1,i)=10.1
           aheigh(2,1)=0.0
           awidth(1,i)=0.0
           awidth(2,i)=32.
           alen(1,i)=442.*1.0026
           alen(2,i)=456.*1.0026
        endif
        if (nap.ge.3) then
           do k=3,nap
              ashape(k,i)='disabled'
              aheigh(k,i)=0.0
              awidth(k,i)=0.0
              alen(k,i)=0.0
           enddo
        endif
 1700 continue
      go to 9030
c
c     ITER BEAM INPUT.
 9020 naptr=4
      do 1705  i=1,kb
        anglev(i)=0.0
        angleh(i)=19.5
        bvofset(i)=0.0
        bhofset(i)=42.074
        bleni(i)=556.808
        bcur(i)=110.
        bptor(i)=10.e6
        bshape(i)='rect-lps'
        bheigh(i)=48.
        bwidth(i)=12.
        bhdiv(i)=.50
        bvdiv(i)=1.3
        if (ke.lt.3) then
           WRITE(*,*)'frinitl:  parameter ke needs to be .ge.3'
           stop
        else
           fbcur(1,i)=0.7
           fbcur(2,i)=0.2
           fbcur(3,i)=0.1
        endif
        if (ke.ge.4) then
           do k=4,ke
              fbcur(k,i)=0.0
           enddo
        endif
        bhfoc(i)=ep100
        bvfoc(i)=1.e3
        ebkev(i)=75.
        sfrac1(i)=0.5d0
        blenp(i)=539.
        rpivot(i)=286.6
        zpivot(i)=0.0d0
        if (nap.lt.4) then
           WRITE(*,*)'frinitl:  parameter nap needs to be .ge.4'
           stop
        else
           ashape(1,i)='s-rect'
           ashape(2,i)='s-rect'
           ashape(3,i)='b-d3d'
           ashape(4,i)='b-circ'
           aheigh(1,i)=47.8
           aheigh(2,i)=48.0
           aheigh(3,i)=0.0
           aheigh(4,i)=0.0
           awidth(1,i)=13.8
           awidth(2,i)=17.7
           awidth(3,i)=0.0
           awidth(4,i)=50.9
           alen(1,i)=186.1
           alen(2,i)=346.0
           alen(3,i)=449.0
           alen(4,i)=500.
        endif
        if (nap.ge.5) then
           do k=5,nap
              ashape(k,i)='disabled'
              aheigh(k,i)=0.0
              awidth(k,i)=0.0
              alen(k,i)=0.0
           enddo
        endif
           
 1705 continue
 9030 continue
c---  
c---following parms are used in subroutine hexnb
c---  
      kdene=1
      kdeni=1
      kdenz=1
      ksvi=0
      ksvz=0
      ksve=0
      krad=1
      ngh=10
      ngl=10
      iexcit=0
      ilorent=0
      mstate=4

c     note izstrp=0 implies coronal equilibrium for impurity j in hexnb
      do 1720 j=1,kimp
        izstrp(j)=0
 1720 continue


      frmod="disabled"
      !gyro-radius correction for NBI deposition
      fr_gyro="disabled"
      frplt="enabled"
      nfrplt=300
      bmsprd=.03
      multiply="disabled"
      multiplyn=0

      beamplse="disabled"  
      beampon=0.d0
      beampoff=0.d0

      !Not used in GitHub version (yet): 
      src_nbi_e="disabled" !YuP[2022-06-30]
      ! Add source to e_general matching NBI sources

c     defaults for reading NUBEAM particle birth pt list
      read_birth_pts="disabled"
      nbirth_pts_files=1
      nbirth_pts=250000
      do i=1,24
         birth_pts_files(i)="notset"
      enddo

c     For removing NBI source at all psi outside of psicutoff:     
      psicutoff=0.d0 ! if 0.0, no removal is done
      ! The value of psicutoff can be determined from  
      ! screen output of rho and psi values. 
      ! Select one of psi(lr) values, set it in cqlinput.


      return
      end
