c
c
      subroutine tdnpa0(rb,ene,icall,iplotnbi)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
cVT....V.Tang, NPA diagnostic 9-25-05.............................
cVT    Currently, I have only made changes to the non-circular section
cVT    Checks out 5/12/06, Version 1.0, NPA Mod
cVT...............................................................
cBH100425: Modified and simplified treatment:  see BH100423 below.
cBH100425: Results changed substantially.  CX cross-section updated
cBH100425: from IAEA ALLADIN data base, nominally good from 0.12 eV
cBH100425: to 630 keV.
c.......................................................................
c     [Earlier SXR routines modified for NPA. Comments from tdsxr0, etc.]
c     This subroutine calculates and plots a SXR///NPA energy spectrum.
c
c     Toroidal geometry with circular or non-circular flux surfaces 
c     is assumed (depending on eqmod namelist variable).
c     The detector is assumed to lie in the x-z plane at y=0.  

c     The radial flux surface bins are traversed by the viewing cord up
c     to four times as in the case where the viewing cord is incident 
c     from the outer equatorial plane, grazes the inside wall of the
c     tokamak, and continues on to the outside equatorial plane.
c
c     For efficiency, contributions to the SXR spectrum are computed
c     only once for each radial bin (defined by input rb) crossed 
c     by the viewing cord.
c     The first task in the computation is obtaining the lengths of
c     the viewing chord in each radial bin, the weighted direction
c     of the viewing cord w.r.t. the B-field, and the weighted poloidal
c     angle on the flux surface.  
c     These quantities are indexed by (bin-index,crossing-index 1:4)
c     [ibin1,ibin2 as below], and are then used to obtain the
c     contributions to the SXR flux.
c     
c     [BH100423:  The CX cross-section is simple, compared to that
c                 for XRays:  forward scattering only, with no energy
c                 change, i.e., scattering cross-section per energy
c                 and solid angle has delta-functions in energy and
c                 solid angle.
c                 Hence, NPA is a relatively short calculation compared
c                 to SXR, so  the calculation can be performed
c                 as a simple integration at each step along the
c                 viewing cord.]
c
c     rb=array of minor radius bin boundaries from 0. to radmin, in cms.
c     Dimension is 0:lrzmax.
c     zeff=array of effective charge, dimension lrzmax.
c     rd_npa= minor radius of detector (cms.),  .ge.radmin.
c     thetd_npa= poloidal location of detector (radians).
c     nv_npa= number of viewing chords.
c     thet1_npa(1:nv_npa)=angle in the poloidal plane including the 
c          antenna, measured from the z-axis (degs).
c     thet2_npa(1:nv_npa)=toroidal angle of the viewing coord, measured 
c           from the x-z plane (y=0.) in right-hand-sense with 
c           respect to the z-axis direction (degs).
c     We have detector direction vector x,y,z-components:
c           detec_x=sin(thet1_npa)*cos(thet2_npa)
c           detec_y=sin(thet1_npa)*sin(thet2_npa)
c           detec_z=cos(thet1_npa)
c     enmin_npa,enmax_npa= are minimum and maximum detector energies (keV).
c       (same for all chords).
c     fds_npa= specifies the step size along the viewing cord, as a fraction
c     of the average radial width of each radial bin (0.2 is a
c     reasonable value).
c
c     icall=  "first" means set up the time-independent and viewing
c     chord-independent piece of the sxr diagnostic, otherwise skip it.
c     icall will be reset to "notfrst", upon exit from this subroutine.
c     NPA:  only presently called with icall="notfrst" [BH100425].
c     NPA:  References to icall can be removed. [BH100425].
c
c     Ionic charge in included in the cross-section calculation using
c     the input variable zeff(lr_).
c.......................................................................


      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      character*8 icall,icalls,iplotnbi
      
      real*8,dimension(lrzmax)::ene
      real*8,dimension(0:lrzmax)::rb
      real*8,dimension(lrzmax,4,nen_npa)::atten
      real*8,dimension(lrzmax,nen_npa)::stoplamda
      real*8,dimension(nen_npa)::vel
      real*8,dimension(nen_npa,npaproc)::sigmanpa,emitnpa
      real*8,dimension(nen_npa,nv_npa)::eflux_npa
      dimension efluxwk(nen_npa,lrzmax) ! local working array
      
      integer::jj
      real*8, allocatable, dimension(:):: tempp4,tempp5,tempp6
        
      print*, 'tdnpa0::Doing NPA diagnostic, CQL at step', n 
      eflux_npa(:,:)=0. 
        
      if(nv_npa.gt.nva) stop 'nv_npa.gt.nva in tdnpa0'
      
      if(nen_npa.gt.nena) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE (*,1000) nen_npa, nena    
CMPIINSERT_ENDIF_RANK  
 1000    format("nen_npa = ",i3, "is too large. Reset to ",i5)
         nen_npa=nena
      endif

c.......................................................................
c     Allocate temporary storage, to be deallocated at end of subroutine
c.......................................................................
cBH121115      itempp=int(15*lrzmax*nv_npa/fds)
      itempp=int(100*lrzmax*nv_npa/fds)
      allocate(tempp4(itempp),STAT=istat4)
      allocate(tempp5(itempp),STAT=istat5)
      allocate(tempp6(itempp),STAT=istat6)
      if (istat4.ne.0 .or. istat5.ne.0 .or. istat6.ne.0) then
         STOP 'tdnpa0: tempp4-tempp6 allocation problem'
      endif
c.......................................................................


      den=(enmax_npa-enmin_npa)/(nen_npa-1.) !energy bin width(keV)

cBH   Take en_() to be the center of each energy bin
      do 5  ien=1,nen_npa
         en_(ien)=enmin_npa+(ien-1)*den  ! (keV)
 5    continue

      do 223 ien=1,nen_npa
         vel(ien)=(2.*en_(ien)*ergtkev/fmass(1))**0.5  ! For ion#1 !!!
 223  continue 
      
c.......................................................................
c     Obtain CX cross-section(s) versus energy
c.......................................................................
      
c     if (icall .eq. "first") 
      call tdsetnpa(sigmanpa)
              
        
cVT..............................................
cVT....call lam subroutine to get the needed attenuation cross
cVT    sections for NPA diagnostic
cVT.................................................................

      call lam(en_,ene,stoplamda)     
     
      icalls=icall
      iistep=0


c.......................................................................
c*********main loop over viewing angles      ***************************
c.......................................................................

      do 420  nn=1,nv_npa

c.......................................................................
c     Start source at detector position and calculate viewing cosines
c.......................................................................

c     if x_npa(1) and z_npa(1) are both zero, then rd_npa,thetd_npa 
c     are used instead.
       xs_(2)=0.0d0
       if (x_npa(1).eq.zero .and. z_npa(1).eq.zero) then
          xs_(1)= rmag + rd_npa(nn)*cos(thetd_npa(nn)*pio180)
          xs_(3)= zmag + rd_npa(nn)*sin(thetd_npa(nn)*pio180)
          ! YuP June 2013: In above, changed radmaj -> rmag, added zmag
       else
          xs_(1)=x_npa(nn)
          xs_(3)=z_npa(nn)
       endif
       print*,'tdnpa0: starting x,y,z',xs_(1),xs_(2),xs_(3)

c.......................................................................
c     Calculate viewing cosines.
c.......................................................................

        alphad(1)=sin(thet1_npa(nn)*pio180)*cos(thet2_npa(nn)*pio180)
        alphad(2)=sin(thet1_npa(nn)*pio180)*sin(thet2_npa(nn)*pio180)
        alphad(3)=cos(thet1_npa(nn)*pio180)
       
c.......................................................................
c     Stepping along the sightline obtaining distance in each radial
c     bin, the weighted angle with respect to the b-field, and the
c     average polidal position in the radial bin....
c.......................................................................

        if(fds_npa.ge.1.)  stop 'tdfds_npa step size in npa too large'
        ibin1=lrzmax
        ibin2=1

c.......................................................................
c     ibin1= the radial bin of the source point along the viewing cord,
c     numbered from 1 for the bin with inner edge at the magnetic
c     axis, to lrzmax for the bin with outer limit equal to radmin/
c     (rhomax for eqmod.eq.'enabled').
c     (Bin ibin occupies the region from .ge.rb(ibin-1) to
c     .lt.rb(ibin)).
c     Emission will only be calculated for each crossing of a radial
c     bin.  Need to keep track of several possible crossings per bin.
c     ibin2= refers to successive classes of radial motion along the
c     viewing cord.
c     = 1, then this is first pass from larger to smaller minor radius.
c     2, minor radius is increasing as the viewing cord is transitted
c     3, viewing cord has not passed out of the plasma, and the
c     minor radius is decreasing.
c     4, minor radius increasing again.
c.......................................................................

        s=0.0
        istart=1
        istep=0
        
cVT indicators for when we switch ibin2 and ibin1 so we can properly 
cVT account for attenuation;
        iclass12=0 
        iclass23=0
        ichangebin13=0
        ichangebin24=0
cVT...............................................
        
        do 99  j=1,4
          do 991  i=1,lrzmax  ! Initialize:
            ibin(i,j)=0
            sxry(i,j)=0.0
            sang(i,j)=0.0
            spol(i,j)=0.0
cVT attenuation factor
            do 992 kk=1,nen_npa	
               atten(i,j,kk)=0.0
 992        continue   
cVT................................       
 991     continue
 99      continue

c.......................................................................
c     main loop for determination of view chord increments
c.......................................................................

 100    continue

        ds=fds_npa*min(rb(ibin1)-rb(ibin1-1),radmin/lrzmax)
        s=s+ds
        istep=istep+1  ! counts for each view chord
        do 102  j=1,3
 102    xs_(j)=xs_(j)+alphad(j)*ds

c.......................................................................
c     Save view cords for plotting at end of subroutine
c.......................................................................

        iistep=iistep+1  ! counts up over all view chords
        if (iistep.le.itempp) then
           lensxr(nn)=istep
           tempp4(iistep)=xs_(1)
           tempp5(iistep)=xs_(2)
           tempp6(iistep)=xs_(3)
        else
           stop 'tdnpa0: TOO MANY STEPS to save in NPA cords'
        endif

c.......................................................................
c     Toroidal angle of source point
c.......................................................................

        phis=atan2(xs_(2),xs_(1))

c.......................................................................
c     Beginning of if-statement for circular/noncircular
c.......................................................................

        if (eqmod.ne."enabled")  then

CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)
           WRITE(*,*)'tdnpa0:CAUTION: Circular SXR NOT ADJUSTED FOR NPA'
           WRITE(*,*)
CMPIINSERT_ENDIF_RANK  

c.......................................................................
c     Minor radius of source point
c.......................................................................

          rs=sqrt((xs_(1)-radmaj*cos(phis))**2
     1      +(xs_(2)-radmaj*sin(phis))**2 +xs_(3)**2)

c.......................................................................
c     Continue stepping inwards until the plasma is entered,
c     or more than 1000/fds_npa steps are taken.
c.......................................................................

          if(istep.gt.(1000/fds_npa).and.istart.eq.1)  then
CMPIINSERT_IF_RANK_EQ_0
             WRITE(*,*) 'Warning: NPA sightline',nn,
     +                  " missed or didn't reach plasma"
CMPIINSERT_ENDIF_RANK  
             go to 420
          endif
          if(rs.ge.radmin.and.istart.eq.1)  go to 100
          istart=0

c.......................................................................
c     Exit from this loop if the source point leaves the plasma.
c.......................................................................

          if(rs.ge.radmin)  go to 200

c.......................................................................
c     Decide which radial bin the source point
c     is in:
c.......................................................................

          if(ibin2.eq.2.or.ibin2.eq.4)  go to 130

c.......................................................................
c     ibin2 is equal to 1 or 3. Thus at previous step the bin disignator
c     was in a decreasing mode.
c     Check if in the same bin as previous step.
c.......................................................................

          rs1=rs-rb(ibin1)
          rs2=rs-rb(ibin1-1)
          if(rs1*rs2.le.0..and.rs1.ne.zero)  go to 150

c.......................................................................
c     Have changed bins.
c.......................................................................

          if(rs1.ge.0.)  go to 115
c.......................................................................
c     Continuing in decreasing mode.
c.......................................................................
          ibin1=ibin1-1
          if(ibin1.lt.1)  stop 'stop 1 in tdnpa0'
          go to 150
c..................................................................
c     Have changed directions in minor radius
c..................................................................

 115      ibin2=ibin2+1
          ibin1=ibin1+1
          go to 150

c..................................................................
c     ibin2 is equal to 2 or 4.  At previous step bin designator was in
c     increasing mode.
c     Check if source is in same bin as previous step.
c..................................................................

 130      rs1=rs-rb(ibin1-1)
          rs2=rs-rb(ibin1)
          if(rs1*rs2.le.0..and.rs2.ne.zero)  go to 150

c.......................................................................
c     Have changed bins
c.......................................................................

          if(rs1.lt.0.0)  go to 135

c.......................................................................
c     Continuing on is same direction
c.......................................................................

          ibin1=ibin1+1
          if(ibin1.gt.lrzmax)  stop 'stop 2 in tdnpa0'
          go to 150

c.......................................................................
c     Have changed directions
c.......................................................................

 135      ibin2=ibin2+1
          ibin1=ibin1-1

c.......................................................................
c     Add in ds, and increment weighted angles.
c.......................................................................
 150      if(ibin2.gt.4)  stop 'stop 3 in tdnpa0'
          sxry(ibin1,ibin2)=sxry(ibin1,ibin2)+ds
c.......................................................................
c     Angle between viewing dirn. and toroidal (approx. b-field) direction
c     (Detector views sxr's emitted towards it).
c.......................................................................

          angle=acos(-sin(phis)*alphad(1)+cos(phis)*alphad(2))
          sang(ibin1,ibin2)=sang(ibin1,ibin2)+ds*angle

c.......................................................................
c     Poloidal angle
c.......................................................................

          arg1=xs_(3)
          arg2=sqrt(xs_(1)**2+xs_(2)**2)-radmaj
          if(arg1.eq.zero .and. arg2.eq.zero)  then
            polang=0.
          else
            polang=atan2(arg1,arg2)
          endif

c.......................................................................
c     assume up-down symmetry
c.......................................................................

          polang=abs(polang)
          if(polang.gt.pi)  polang=pi
          spol(ibin1,ibin2)=spol(ibin1,ibin2)+ds*polang

c.......................................................................





c.......................................................................
c     If non-circular:   (Only checked out implementation for NPA.)
c.......................................................................

        else  ! eqmod.ne.disabled


          if ((psimag-psilim).le.zero) stop 'psimag.lt.psilim in tdnpa0'
          tr(0)=zero
          do 160  l=1,lrzmax
 160      tr(l)=psimag-psivalm(l)  !Increases from 0. to psimag-psilim at edge.
          xx=xs_(1)
          yy=xs_(2)
          rr=sqrt(xx**2+yy**2)
          zz=xs_(3)
c.......................................................................
c     Check if on eqdsk grid
c.......................................................................

          iongrid=0
          if ((rr.lt.er(nnr).and. rr.gt.er(1)) .and.
     +         (zz.lt.ez(nnz).and.zz.gt.ez(1))) iongrid=1
          if (iongrid.eq.0) go to 165 
c.......................................................................
c     Minor radius of source point
c.......................................................................

          ppsi=terp2(rr,zz,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1      epsirz,nnra,0,0)
          apsi=psimag-ppsi     !Increases from 0. to psimag-psilim at edge.
          apsi=max(apsi,zero)  !Sometimes there might be a small
                               !error in psimag causing a problem
                               !near the magnetic axis [BH, 121115].

c.......................................................................
c     Continue stepping inwards until the plasma is entered,
c     or more than 1000/fds_npa steps are taken.
c.......................................................................

 165      if(istep.gt.(1000/fds_npa).and.istart.eq.1)  then
CMPIINSERT_IF_RANK_EQ_0
             WRITE(*,*) 'Warning: npa sightline',nn,
     +                  " missed or didn't reach plasma"
CMPIINSERT_ENDIF_RANK  
             go to 420
          endif
          if(((iongrid.eq.0).or.(ppsi.le.psilim)).and.
     +                                        istart.eq.1)  go to 100
          !Correction, similar to YuP[2021-11-05] in tdsxr0:
          !YuP[2023-08-10] Flaw in logic: assumes that the sightline
          !enters plasma at psi=psilim surface, 
          !which is outside of rya(lrzmax).
          !The procedure below is checking
          !where the point is relative to bracket [tr(lrz-1); tr(lrz)]
          !while it is not there yet. As a result, 
          !the value of ibin1+1 will exceed lrzmax.
          !Temporary fixed it by adding a limit
          !ibin1=min(ibin1,lrzmax)  [see below, after label 170]

          istart=0  ! in the plasma, i.e., no longer starting.

c.......................................................................
c     Exit from this loop if the source point leaves the plasma.
c.......................................................................

          if(ppsi.le.psilim)  go to 200

c.......................................................................
c     Decide which radial bin the source point
c     is in:
c.......................................................................

          if(ibin2.eq.2.or.ibin2.eq.4)  go to 180

c.......................................................................
c     ibin2 is equal to 1 or 3. Thus at previous step the bin designator
c     was in a decreasing mode.
c     Check if in the same bin as previous step.
c.......................................................................

          rs1=apsi-tr(ibin1)     !  ibin1 is initialized = lrzmax
          rs2=apsi-tr(ibin1-1)
          if(rs1*rs2.le.0..and.rs1.ne.zero)  go to 190

c.......................................................................
c     Have changed bins.
c.......................................................................

          if(rs1.ge.zero)  go to 170  ! Changed to increasing mode
c.......................................................................
c     Continuing in decreasing mode.
c.......................................................................
          ichangebin13=1 !cVT
          ibin1=ibin1-1
          if(ibin1.lt.1)  stop 'stop 4 in tdnpa0'
          go to 190
c.......................................................................
c     Have changed directions in minor radius
c.......................................................................

 170      ibin2=ibin2+1
          ibin1=ibin1+1
          iclass12=1 !cVT
          !Correction, similar to YuP[2021-11-05] in tdsxr0:
          ibin1=min(ibin1,lrzmax) !YuP[2023-08-10] added
          if(ibin1.gt.lrzmax)then
             write(*,*)' tdnpa0/170: rs1,rs2=',rs1,rs2, ibin1,ibin2
             stop 'stop 170/ibin1>lrz in tdnpa0'
          endif

          go to 190

c.......................................................................
c     ibin2 is equal to 2 or 4.  At previous step bin designator was in
c     increasing mode.
c     Check if source is in same bin as previous step.
c.......................................................................

 180      rs1=apsi-tr(ibin1-1)
          rs2=apsi-tr(ibin1)
          if(rs1*rs2.le.zero.and.rs2.ne.zero)  go to 190

c.......................................................................
c     Have changed bins
c.......................................................................

          if(rs1.lt.zero)  go to 185

c.......................................................................
c     Continuing on in same direction
c.......................................................................
          ichangebin24=1 !cVT
          ibin1=ibin1+1
          !Correction, similar to YuP[2021-11-05] in tdsxr0:
          ibin1=min(ibin1,lrzmax) !YuP[2023-08-10] added
          if(ibin1.gt.lrzmax)  stop 'stop 5 in tdnpa0'
          go to 190

c.......................................................................
c     Have changed directions
c.......................................................................

 185      ibin2=ibin2+1
          ibin1=ibin1-1
          iclass23=1 !cVT

c.......................................................................
c     Add in ds, and increment weighted angles.
c.......................................................................

 190      if(ibin2.gt.4)  stop 'stop 6 in tdnpa0'
          sxry(ibin1,ibin2)=sxry(ibin1,ibin2)+ds ! cumulative length (cms)
                                                 ! in each radial bin.
          do 195 ien=1,nen_npa

cBH          print*,'tdnpa0::Stepping thru energies for atten' 
cVT check bin changes and increment attenuation factor depending on case
cVT stoplamda stands for lamda and it's the mean-free path for a
cVT flux surface as a function of neutral energy (n_en)   
          if (iclass12.eq.1) then !changed ibin2, class from 1 to 2  or 3 to 4
          atten(ibin1,ibin2,ien)=ds/stoplamda(ibin1,ien)
     1     +atten(ibin1-1,ibin2-1,ien)  !going from into plasma to out...
cVT                                     ibin1 flux sufrace now going up
c          print*,'tdnpa0::ibin2 change from 1 to 2 or 3 to 4 in atten'
c          print*,'tdnpa0::Previous class attenuation factor for 
c     1     energy bin:',en_m(ien),atten(ibin1-1,ibin2-1,ien)
          elseif (iclass23.eq.1) then !changed ibin2, class from 2 to 3
          atten(ibin1,ibin2,ien)=ds/stoplamda(ibin1,ien)
     1      +atten(ibin1+1,ibin2-1,ien) !going from out of plasma to in
cVT                         again...inbin flux sufraces now going down

c          print*,'tdnpa0::ibin2 change from 2 to 3 in atten'
c          print*,'tdnpa0::Previous class attenuation factor for energy 
c     1     bin:',en_m(ien),atten(ibin+1,ibin2-1,ien)

          elseif (ichangebin13.eq.1) then !decreasing ibin1, in class 1 or 3
          atten(ibin1,ibin2,ien)=ds/stoplamda(ibin1,ien)
     1       +atten(ibin1+1,ibin2,ien) !get result from last flux surface
c          print*,'tdnpa0::ibin1 has decreased by 1 in atten'

          elseif (ichangebin24.eq.1) then !increasing ibin1, in class 2 or 4
          atten(ibin1,ibin2,ien)=ds/stoplamda(ibin1,ien)
     1     +atten(ibin1-1,ibin2,ien) !get result from last flux sufrace
c          print*,'tdnpa0::ibin1 has increased by 1 in atten'

          else !no change in bins or direction
          atten(ibin1,ibin2,ien)=ds/stoplamda(ibin1,ien)
     1     +atten(ibin1,ibin2,ien)
          endif

 195      continue

          iclass12=0
          iclass23=0
          ichangebin13=0
          ichangebin24=0

cVT........................................................................

c.......................................................................
c     Angle between (negative) viewing dirn, and b-field direction,
c     (considering b-field oriented with positive toroidal component).
c     (Detector views sxr's emitted towards it).
c.......................................................................

c.......................................................................
c     Determine the pitch angle (cos(theta)) at birth. Compute
c     B dot v / mag B/ mag v
c.......................................................................

          dpsidr=terp2(rr,zz,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1      epsirz,nnra,1,0)
          dpsidz=terp2(rr,zz,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1      epsirz,nnra,0,1)

c.......................................................................
c     Determine f(psi)
c.......................................................................

          call eqfpsi(ppsi,fpsi_,fppsi_)
          fpsi_=abs(fpsi_)

c.......................................................................
c     Determine components of B.
c.......................................................................
cBHandYuP110715: Account for bsign/cursign
          xcomp=(-cursign*dpsidz*xx-bsign*fpsi_*yy)/rr**2
          ycomp=(-cursign*dpsidz*yy+bsign*fpsi_*xx)/rr**2
          zcomp=+cursign*dpsidr/rr

c.......................................................................
c     Compute the dot product and the cosine
c.......................................................................

          dotprd=-(xcomp*alphad(1)+ycomp*alphad(2)+zcomp*alphad(3))
          bbmag=sqrt(xcomp**2+ycomp**2+zcomp**2)
cBHandYuP110715: cospitch measured with vpar pos in pos vphi dirn
cBHandYuP110715: (CCW dirn viewed from top).
          cospitch=bsign*dotprd/bbmag
          cospitch=max(cospitch,-1.d0) ! to be sure cos>=-1
          cospitch=min(cospitch,+1.d0) ! to be sure cos<=+1
          sinpitch=sqrt(1.d0-cospitch**2)
          angle=acos(cospitch)
          sang(ibin1,ibin2)=sang(ibin1,ibin2)+ds*angle

c.......................................................................
c     Poloidal angle   (rmag is major radius of magnetic axis).
c.......................................................................

          arg1=zz-zmag
          arg2=rr-rmag
          if(arg1.eq.zero .and. arg2.eq.zero)  then
            polang=0.
          else
            polang=atan2(arg1,arg2)
          endif

c.......................................................................
c     assume up-down symmetry
c.......................................................................

          polang=abs(polang)
          if(polang.gt.pi)  polang=pi
          spol(ibin1,ibin2)=spol(ibin1,ibin2)+ds*polang

c.......................................................................
c     end of circular/non-circular if-statement
c.......................................................................

        endif


c.......................................................................
c     Radial bin.
c.......................................................................

        ibin(ibin1,ibin2)=ibin1
        go to 100

c.......................................................................
c     *******Have finished sightline and attenuation determination******
c.......................................................................

 200    continue

c        do ien=1,nen_npa
c           do l=1,lrzmax
c            write(*,*)'l,ien,atten(l,j=1:4,ien)=',l,ien,atten(l,1:4,ien)
c           enddo
c        enddo


c.......................................................................
c     Obtain weighted angle between viewing and toroidal directions,
c     and poloidal angle, in each of the lrzmax radial bins.
c.......................................................................

        do 210  j=1,ibin2  ! Radial bin class, ibin2.le.4
          do 211  i=1,lrzmax
            if(sxry(i,j).eq.zero)  go to 211
            sang(i,j)=sang(i,j)/sxry(i,j)
            spol(i,j)=spol(i,j)/sxry(i,j)
            if (spol(i,j) .gt. pi) spol(i,j)=pi
 211      continue
 210    continue

c        do l=1,lrzmax
c           write(*,*)'l,sxry(l,j=1:4)=',l,sxry(l,1:4)
c        enddo
c        do l=1,lrzmax
c           write(*,*)'l,sang(l,j=1:4)=',l,sang(l,1:4)
c        enddo
c        do l=1,lrzmax
c           write(*,*)'l,spol(l,j=1:4)=',l,spol(l,1:4)
c        enddo


c.......................................................................
c     Obtain energy flux for each energy and sightline
c.......................................................................

        call bcast(emitnpa,zero,nen_npa*npaproc)
        call bcast(efluxwk(1,1),zero,nen_npa*lrzmax)
        mpisz= nen_npa ! number of elements in efluxwk(:,i) 
           
c       set atten bins that will be double counted to some 
c       very high attenuation

        atten(1,2,:)=100
        atten(lrzmax,3,:)=100
        atten(1,4,:)=100
        
        do i=1,lrzmax   !lrzmax
CMPIINSERT_MPIWORKER_I
CMPIINSERT_IF_RANK_EQ_MPIWORKER
        do j=1,ibin2  ! Radial bin class, ibin2.le.4
              if(sxry(i,j).eq.zero)  go to 311
c     Obtain neutral emittance emitnpa (#/sec*cm**3*ster*eV),
c     for each npa_process().
c     enn density pertains to each npa_process(), generally
c     a namelist input neutral density.  For npa_process(5)=
c     'radrecom', enn(1:lrzmax,5) is set to the electron density.
              call tdnpa(vel,sang(i,j),ibin(i,j),spol(i,j),
     +             sigmanpa,emitnpa)
              do kk=1,npaproc
              do 320 ien=1,nen_npa
c              if (j.EQ.1 .AND. i.EQ.1) THEN  
c     VT specifically prohibit double counting of l=1 surface
c     VT.......... have to interpolate values for emitnpa since the grid 
c     VT           for emitnpa is velocity and we want energy....
                 
                 if (npa_process(kk).ne.'notset') then
                 if (atten_npa.eq."enabled") then
                     efluxwk(ien,i)= efluxwk(ien,i)+
     1                   enn(ibin(i,j),kk)
     1                   *sxry(i,j)*emitnpa(ien,kk)*exp(-atten(i,j,ien))
                 else
                     efluxwk(ien,i)= efluxwk(ien,i)+
     1                   enn(ibin(i,j),kk)
     1                   *sxry(i,j)*emitnpa(ien,kk)
                 endif
                 endif  ! On npa_process(kk)
c     VT................................................................
c                 print*,"eflux_npa(ien,nn)=",eflux_npa(ien,nn)
c                 print*,"enn(ibin(i,j),1)=",enn(ibin(i,j),1)
c                 print*,"enn(ibin(i,j),2)=",enn(ibin(i,j),2)
c                 print*,"sxry(i,j)=",sxry(i,j)
c                 print*,"exp(-attend..)=",exp(-atten(i,j,ien))
c                 if (ien.eq.1)print*,"emitnpa(,1)=",emitnpa(1:nen_npa,1)
c                 print*,"----------------------------"

 320        continue !ien=1,nen_npa
            enddo  ! kk=1,npaproc
            
 311    continue 
        enddo ! j=1,ibin2
CMPIINSERT_ENDIF_RANK
CMPIINSERT_RECV_EFLUXWK
CMPIINSERT_SEND_EFLUXWK
        enddo ! i=1,lrzmax 

        call bcast(eflux_npa(1,nn),zero,nen_npa) 
        
        do ien=1,nen_npa  ! for each energy bin
        do i=1,lrzmax ! sum-up contributions from all flux surfaces
           eflux_npa(ien,nn)=eflux_npa(ien,nn)+ efluxwk(ien,i)
        enddo ! i=1,lrzmax
        enddo ! ien=1,nen

CMPIINSERT_BARRIER 
CMPIINSERT_BCAST_EFLUX_NPA



      
c.......................................................................
c     Integrate energy flux spectra from enmin_npa to enmax_npa.
c     Assume the flux varies as exp(-e/t) between flux values,
c     to improve accuracy.
c.......................................................................

        inegsxr(nn)=nen_npa
        efluxt(nn)=0.0
        do 400  i=1,nen_npa
c          write(*,*)'tdnpa0:i,nn,eflux_npa(i,nn),zero=',
c     +          i,nn,eflux_npa(i,nn),zero
          if (eflux_npa(i,nn).le.zero) then
            inegsxr(nn)=i-1
            go to 402
          endif
cBH100811         if (i.eq.nen) go to 402
cBH100811  Added following line.
         if (i.eq.nen_npa) go to 402
c         write(*,*)'tdnpa0: i=',i
cBH100419:  Wonder why I have this (also in tdsxr0.f)??
cBH100811:  An answer:  get out-of-bounds for i=nen_npa.
cBH100419          if (i.eq.nen_npa) go to 402

c         Avoid some overflows:
          if(eflux_npa(i+1,nn).lt.zero) go to 402
          if (eflux_npa(i,nn).lt.em100*em100) go to 400
cBH160507     if(eflux_npa(i,nn).eq.eflux_npa(i+1,nn))  go to 401
cBH160507 Allowing for possible increase in eflux_npa with en.
cBH160507 Could modify to fork with eflux_npa(i:i+1) are too close 
          if(eflux_npa(i,nn).le.eflux_npa(i+1,nn))  go to 401
          if(eflux_npa(i+1,nn).eq.zero)  go to 401

          t=(en_(i+1)-en_(i))/log(eflux_npa(i,nn)/eflux_npa(i+1,nn))
cBH160507     c=eflux_npa(i,nn)*exp(en_(i)/t)
cBH160507          write(*,*)'tdnpa0.f: i,t,c,eflux_npa(i,nn)=',
cBH160507     +                         i,t,c,eflux_npa(i,nn)
cBH160507     efluxt(nn)=efluxt(nn)+c*t*(exp(-en_(i)/t)-exp(-en_(i+1)/t))
          efluxt(nn)=efluxt(nn)+t*(eflux_npa(i,nn)-eflux_npa(i+1,nn))
     +      *1.e3
          go to 400
 401      efluxt(nn)=efluxt(nn)+eflux_npa(i,nn)*(en_(i+1)-en_(i))*1.e3
 400    continue
 402    continue
 
c        write(*,*)
c        write(*,*)'tdnpa0: emitnpa(1:30,1:npaproc)'
c        do i=1,npaproc
c           write(*,*)'tdnpa0: emitnpa(*,i)',(emitnpa(jj,i),jj=1,30)
c        enddo
c        write(*,*)
        
c        print*,'npa_mesh=[',en_(1:nen_npa),']'
c        print*,'eflux_npa=[',eflux_npa,']'
c        write(*,*)'tdnpa0: efluxt(1:nn)=',efluxt(1:nn)
c     
        icall="notfirst"

c.......................................................................
c*********END loop over viewing angles      **************************
c.......................................................................
c        print*,'------------------------------'
c        print*,'Done with sightline',nn

 420  continue

c$$$      iistep=0
c$$$      do nn=1,nv
c$$$         print*,'tempp4 x=[',tempp4(iistep+1:iistep+lensxr(nn)),']'
c$$$         print*,'tempp5 y=[',tempp5(iistep+1:iistep+lensxr(nn)),']'
c$$$         print*,'tempp6 z=[',tempp6(iistep+1:iistep+lensxr(nn)),']'
c$$$         iistep=iistep+lensxr(nn)
c$$$      enddo
c$$$          iiii=int(nen_npa)
c$$$          print*, "tdnpa0::atten for Neutral energy:",en_(iiii)
c$$$          print*,"atten_ibin21=[",atten(1:lrzmax,1,iiii)
c$$$          print*,"atten_ibin22=[",atten(1:lrzmax,2,iiii)
c$$$          print*,"atten_ibin23=[",atten(1:lrzmax,3,iiii)
c$$$          print*,"atten_ibin24=[",atten(1:lrzmax,4,iiii)
c$$$          iiii=int(nen_npa/2)
c$$$          print*, "tdnpa0::atten for Neutral energy:",en_(iiii)
c$$$          print*,"atten_ibin21=[",atten(1:lrzmax,1,iiii)
c$$$          print*,"atten_ibin22=[",atten(1:lrzmax,2,iiii)
c$$$          print*,"atten_ibin23=[",atten(1:lrzmax,3,iiii)
c$$$          print*,"atten_ibin24=[",atten(1:lrzmax,4,iiii)





c.......................................................................
c     Plot eflux_npa........
c.......................................................................

      nenaa=nen_npa
cBH100418:  copied following call here.  Probably should restore
cBH100418:  icall logic for npa....
cBH100418:      call tdsxrvw

cBH100418: Follwing call to tdsxrplt was commented out by VT

cBH100418: Normalize nps flux
cBH100418:      do nn=1,nv_npa
cBH100418:         do i=2,nen_npa
cBH100418:            eflux_npa(i,nn)=eflux_npa(i,nn)/eflux_npa(1,nn)
cBH100418:         enddo
cBH100418:         eflux_npa(1,nn)=1.0
cBH100418:      enddo

c     iplt3d set in tdchief
!YuP      if (iplt3d.ne.0 .or. n.eq.0 .or. n.eq.nstop 
!YuP     +     .and. (iplotnbi.eq.'yes' .or. iplotsxr.eq.'yes')  ) then
!YuP[2022-03-20] Removing iplotsxr.eq.'yes' 
!YuP Looks like it got here by mistake. No handle to iplotsxr
      if (iplt3d.ne.0 .or. n.eq.0 .or. n.eq.nstop 
     +     .and. (iplotnbi.eq.'yes')  ) then
         call tdsxrplt(en_,eflux_npa,nen_npa,nenaa,
     +        efluxt,nv_npa,inegsxr,softxry,npa_diag,lnwidth)

c.......................................................................
c     Plot NPA view cords in poloidal cross-section, if eqmod="enabled"
c.......................................................................

         if (eqmod.eq."enabled") then
            nv=nv_npa
            call tdsxrvw(tempp4,tempp5,tempp6)
         endif
      endif ! On iplt3d, etc.

      deallocate(tempp4,tempp5,tempp6,STAT=istat1)

c.......................................................................
c     Put eflux_npa into eflux in order to pass the data to netcdfrw2.
c     (It would be cleaner to put eflux_npa in common, as done for
c     eflux.  BH100816)
c.......................................................................

      do nn=1,nv_npa
         do ien=1,nen_npa
            eflux(ien,nn)=eflux_npa(ien,nn)
         enddo
      enddo

      return
      end
