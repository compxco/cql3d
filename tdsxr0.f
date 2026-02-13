c
c
      subroutine tdsxr0(rb,ene,icall,iplotsxr)
      implicit integer (i-n), real*8 (a-h,o-z)


c.......................................................................
c     This subroutine calculates and plots a SXR energy spectrum.
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
c
c
c     rb=array of minor radius bin boundaries from 0. to radmin, in cms.
c        It is only used in the circular plasma case, and/or for 
c        determination of the step size along the sight line for
c        both circ and eqdsk equilibria.
c     Dimension is 0:lrzmax.
c     zeff=array of effective charge, dimension lrzmax.
c     nv = number of viewing chords.
c     x_sxr(1:nv) = x-coordinate of detector
c     z_sxr(1:nv) = z-coordinate of detector
c
c     if x_sxr(1) and z_sxr(1) are both zero, then rd,thetd are used instead
c
c     rd(1:nv) = minor radius of detector (cms.),  .ge.radmin.
c     thetd(1:nv) = poloidal location of detector (radians).
c     thet1(1:nv) =angle in the poloidal plane including the 
c          antenna, measured from the z-axis (degs).
c     thet2(1:nv) = toroidal angle of the viewing coord, measured 
c           from the x-z plane (y=0.) in right-hand-sense with 
c           respect to the z-axis direction (degs).
c     We have detector direction vector x,y,z-components:
c           detec_x=sin(thet1)*cos(thet2)
c           detec_y=sin(thet1)*sin(thet2)
c           detec_z=cos(thet2)
c     enmin,enmax= are minimum and maximum detector energies (keV).
c       (same for all chords).
c     fds= specifies the step size along the viewing cord, as a fraction
c     of the average radial width of each radial bin (0.2 is a
c     reasonable value).
c
c     icall=  "first" means set up the time-independent and viewing
c     chord-independent piece of the sxr diagnostic, otherwise skip it.
c     icall will be reset to "notfrst", upon exit from this subroutine.
c
c     Ionic charge in included in the cross-section calculation using
c     the input variable zeff(lr_).
c.......................................................................


      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      character*8 icall,iplotsxr,icalls
      dimension rb(0:lrzmax),ene(lrzmax)
      dimension efluxwk(nen,lrzmax) ! local working array
      real*8, allocatable, dimension(:):: tempp4,tempp5,tempp6

      if(nv.gt.nva) stop 'nv.gt.nva in tdsxr0'

c.......................................................................
c     Allocate temporary storage, to be deallocated at end of subroutine
c.......................................................................
cBH121115      itempp=int(25*lrzmax*nv/fds)
      itempp=int(100*lrzmax*nv/fds)
      allocate(tempp4(itempp),STAT=istat4)
      allocate(tempp5(itempp),STAT=istat5)
      allocate(tempp6(itempp),STAT=istat6)
      if (istat4.ne.0 .or. istat5.ne.0 .or. istat6.ne.0) then
         STOP 'tdsxr0: tempp4-tempp6 allocation problem'
      endif
c.......................................................................
      

      
      icalls=icall
      iistep=0

c.......................................................................
c*********loop over viewing angles      **************************
c.......................................................................
      do 420  nn=1,nv

c.......................................................................
c     Start source at detector position and calculate viewing cosines
c.......................................................................

       xs_(2)=0.0
       if (x_sxr(1).eq.zero .and. z_sxr(1).eq.zero) then
          xs_(1)=radmaj+rd(nn)*cos(thetd(nn)*pio180)
          xs_(3)=rd(nn)*sin(thetd(nn)*pio180)
       else
          xs_(1)=x_sxr(nn)
          xs_(3)=z_sxr(nn)
       endif
c.......................................................................
c     Calculate viewing cosines.
c.......................................................................

        alphad(1)=sin(thet1(nn)*pio180)*cos(thet2(nn)*pio180)
        alphad(2)=sin(thet1(nn)*pio180)*sin(thet2(nn)*pio180)
        alphad(3)=cos(thet1(nn)*pio180)

c.......................................................................
c     Stepping along the sightline obtaining distance in each radial
c     bin, the weighted angle with respect to the b-field, and the
c     average polidal position in the radial bin....
c.......................................................................

        if(fds.ge.1.)  stop 'fds step size in sxr too large'
        ibin1=lrzmax
        ibin2=1

c.......................................................................
c     ibin1= the radial bin of the source point along the viewing cord,
c     numbered from 1 for the bin with inner edge at the magnetic
c     axis, to lrzmax for the bin with outer limit equal to radmin.
c     (Bin ibin occupies the region from .ge.rb(ibin-1) to
c     .lt.rb(ibin)).
c     Emission will only be calculated for each crossing of a radial
c     bin.  Need to keep track of several possible crossings per bin.
c     ibin2= refers to successive classes of radial motion along the
c     viewing cord.
c     = 1, then this is first pass from larger to smaller minor radius.
c       2, minor radius is increasing as the viewing cord is transitted
c       3, viewing cord has not passed out of the plasma, and the
c          minor radius is decreasing.
c       4, minor radius increasing again.
c.......................................................................

        s=0.0
        istart=1
        istep=0
        do 99  j=1,4
          do 991  i=1,lrzmax
            ibin(i,j)=0
            sxry(i,j)=0.0
            sang(i,j)=0.0
            spol(i,j)=0.0
 991      continue
 99     continue

c.......................................................................
c     main loop for determination of view chord increments
c.......................................................................

 100    continue
 
c990131        ds=fds*amin1(rb(ibin1)-rb(ibin1-1),radmin/lrzmax)
cBH081106        ds=fds*min(rb(ibin1)-rb(ibin1-1),radmin/lrzmax)
        ds=fds*min(rb(ibin1)-rb(ibin1-1),(rb(ibin1)-rb(0))/lrzmax)
        s=s+ds
        istep=istep+1  ! counts for each view chord
        do 102  j=1,3
 102    xs_(j)=xs_(j)+alphad(j)*ds
!        write(*,'(a,2i5,1p3e11.3)')' tdsxr0: nn,istep, X,Y,Z=',
!     &      nn,istep, xs_(1:3)

c.......................................................................
c     Save view cords for plotting at end of subroutine
c.......................................................................

        iistep=iistep+1  ! counts up over all view chords

c        write(*,*)'tdsxr0: nn,istep,iistep,ibin1,ds=',
c     &       nn,istep,iistep,ibin1,ds

        if (iistep.le.itempp) then
           lensxr(nn)=istep
           tempp4(iistep)=xs_(1)
           tempp5(iistep)=xs_(2)
           tempp6(iistep)=xs_(3)
        else
           stop 'tdsxr0: TOO MANY STEPS to save in SXR cords'
        endif

c.......................................................................
c     Toroidal angle of source point
c.......................................................................

        phis=atan2(xs_(2),xs_(1))



c.......................................................................
c     Beginning of if-statement for circular/noncircular
c     Non-circular (eqmod.eq."enabled") case starts at line 349.
c.......................................................................

        if (eqmod.ne."enabled")  then

c.......................................................................
c     Minor radius of source point
c.......................................................................

          rs=sqrt((xs_(1)-radmaj*cos(phis))**2
     1      +(xs_(2)-radmaj*sin(phis))**2 +xs_(3)**2)

c.......................................................................
c     Continue stepping inwards until the plasma is entered,
c     or more than 1000/fds steps are taken.
c.......................................................................

cBH180207          if(istep.gt.(1000/fds).and.istart.eq.1)  then
cBH180207: Needed a few more steps for a difficult case.
          if(istep.gt.(2000/fds).and.istart.eq.1)  then
CMPIINSERT_IF_RANK_EQ_0
             WRITE(*,*) 'Warning: SXR sightline',nn,
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
          if(ibin1.lt.1)  stop 'stop 1 in tdsxr0'
          go to 150
c..................................................................
c     Have changed directions in minor radius
c..................................................................

 115      ibin2=ibin2+1
          ibin1=ibin1+1
          if(ibin1.gt.lrzmax)  stop 'stop 115/ibin1>lrz in tdsxr0'
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
          if(ibin1.gt.lrzmax)  stop 'stop 2 in tdsxr0'
          go to 150

c.......................................................................
c     Have changed directions
c.......................................................................

 135      ibin2=ibin2+1
          ibin1=ibin1-1

c.......................................................................
c     Add in ds, and increment weighted angles.
c.......................................................................
 150      if(ibin2.gt.4)  stop 'stop 3 in tdsxr0'
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
c     If non-circular:
c.......................................................................

        else


          if ((psimag-psilim).le.0.0) stop 'psimag.lt.psilim in tdsxr0'
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
c     or more than 1000/fds steps are taken.
c.......................................................................

cBH180207     165   if(istep.gt.(1000/fds).and.istart.eq.1)  then
cBH180207: Needed a few more steps for a difficult case.
 165      if(istep.gt.(2000/fds).and.istart.eq.1)  then
CMPIINSERT_IF_RANK_EQ_0
             WRITE(*,*) 'Warning: SXR sightline',nn,
     +                  " missed or didn't reach plasma"
CMPIINSERT_ENDIF_RANK  
             go to 420
          endif
          
          if(((iongrid.eq.0).or.(ppsi.le.psilim)).and.
     +                                        istart.eq.1)  go to 100
          !YuP[2021-11-05] Flaw in logic: assumes that the sightline
          !enters plasma at psi=psilim surface, 
          !which is outside of rya(lrzmax).
          !The procedure below is checking
          !where the point is relative to bracket [tr(lrz-1); tr(lrz)]
          !while it is not there yet. As a result, 
          !the value of ibin1+1 will exceed lrzmax.
          !Temporary fixed it by adding a limit
          !ibin1=min(ibin1,lrzmax)  [see below, after label 170]

          istart=0  ! in the plasma
          !write(*,*)' tdsxr0/L.404 Entering plasma. rr=',rr

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
          !write(*,*)' psimag-psilim=', psimag-psilim
          !write(*,*)' tr(19),tr(20)=', tr(19),tr(20)
          !write(*,*)' apsi=',apsi
          !write(*,*)' tdsxr0/L.427  rs1,rs2=',rs1,rs2
          if(rs1*rs2.le.zero.and.rs1.ne.zero) go to 190 !in [ibin1,ibin1+1)

c.......................................................................
c     Have changed bins.
c.......................................................................

          if(rs1.ge.zero)  go to 170  ! Changed to increasing mode
c.......................................................................
c     Continuing in decreasing mode.
c.......................................................................
          ibin1=ibin1-1
          if(ibin1.lt.1) then
             stop 'stop 4 in tdsxr0'
          endif
          go to 190
          
c.......................................................................
c     Have changed directions in minor radius
c.......................................................................
 170      ibin2=ibin2+1
          ibin1=ibin1+1
          ibin1=min(ibin1,lrzmax) !YuP[2021-11-05] added
          if(ibin1.gt.lrzmax)then
             write(*,*)' tdsxr0/170: rs1,rs2=',rs1,rs2, ibin1,ibin2
             stop 'stop 170/ibin1>lrz in tdsxr0'
          endif
          go to 190

c.......................................................................
c     ibin2 is equal to 2 or 4.  At previous step bin designator was in
c     increasing mode.
c     Check if source is in same bin as previous step.
c.......................................................................

 180      rs1=apsi-tr(ibin1-1)
          rs2=apsi-tr(ibin1)
          if(rs1*rs2.le.zero .and. rs2.ne.zero)  go to 190

c.......................................................................
c     Have changed bins
c.......................................................................

          if(rs1.lt.zero)  go to 185

c.......................................................................
c     Continuing on in same direction
c.......................................................................

          ibin1=ibin1+1
          ibin1=min(ibin1,lrzmax) !YuP[2021-11-05] added
          if(ibin1.gt.lrzmax)  stop 'stop 5 in tdsxr0'
          go to 190

c.......................................................................
c     Have changed directions
c.......................................................................

 185      ibin2=ibin2+1
          ibin1=ibin1-1

c.......................................................................
c     Add in ds, and increment weighted angles.
c.......................................................................
 190      if(ibin2.gt.4)  stop 'stop 6 in tdsxr0'
          sxry(ibin1,ibin2)=sxry(ibin1,ibin2)+ds
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

          xcomp=(-cursign*dpsidz*xx-bsign*fpsi_*yy)/rr**2
          ycomp=(-cursign*dpsidz*yy+bsign*fpsi_*xx)/rr**2
          zcomp=+cursign*dpsidr/rr

c.......................................................................
c     Compute the dot product and the cosine
c.......................................................................

          dotprd=-(xcomp*alphad(1)+ycomp*alphad(2)+zcomp*alphad(3))
          bbmag=sqrt(xcomp**2+ycomp**2+zcomp**2)
cBHandYuP110715: cospitch measured with vpar pos in pos vphi
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
c     *******     Have finished sightline determination   ******
c.......................................................................

 200    continue

c.......................................................................
c     Obtain weighted angle between viewing and toroidal directions,
c     and poloidal angle, in each or the lrzmax radial bins.
c.......................................................................

        do 210  j=1,ibin2
          do 211  i=1,lrzmax
            if(sxry(i,j).eq.zero)  go to 211
            sang(i,j)=sang(i,j)/sxry(i,j)
            spol(i,j)=spol(i,j)/sxry(i,j)
            if (spol(i,j) .gt. pi) spol(i,j)=pi
 211      continue
 210    continue

c.......................................................................
c     Integrations over velocity space for several detector energies.
c     Set up energies linearly from enmin to enmax.
c     (Check nen.le.nena)
c.......................................................................

        if(nen.le.nena)  go to 299
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,1000) nen, nena
CMPIINSERT_ENDIF_RANK  
 1000   format("nen = ",i3, "is too large. Reset automatically to ",i5)
        nen=nena
 299    den=(enmax-enmin)/(nen-1.)
 
        do 300  ien=1,nen
          en_(ien)=enmin+(ien-1)*den

c.......................................................................
c     kev
c     Units of m0*c**2
c.......................................................................

          enk(ien)=en_(ien)/restmkev  !YuP[01-2011] was 512.

c.......................................................................
c     Legendre decomposition of XR cross sections due to ei  and ee
c     collisions.  Done only once for each energy. 
c     Results stored in sigsxr(1:jx,1:msxr,1:nen,2) by sub tdsetsxr.
c.......................................................................

          if (icall .eq. "first") call tdsetsxr(ien)
 300    continue ! ien=1,nen

c.......................................................................
c     Obtain energy flux for each energy and sightline
c.......................................................................

        call bcast(efluxwk(1,1),zero,nen*lrzmax)
        mpisz= nen ! number of elements in efluxwk(nen,i) 
        
        do i=1,lrzmax
CMPIINSERT_MPIWORKER_I
CMPIINSERT_IF_RANK_EQ_MPIWORKER
        do j=1,ibin2
        if(sxry(i,j).ne.zero)  then
          do ien=1,nen !should be the innermost: setting plegg at ien=1
              call tdsxr(ien,sang(i,j),spol(i,j),ibin(i,j),
     +                   dedotei,dedotee) !-> out
              efluxwk(ien,i)= efluxwk(ien,i)+
     1          1.6e-12*enk(ien)*sxry(i,j)*ene(ibin(i,j))*
     2          (zeff(ibin(i,j))*dedotei +dedotee)
              ! efluxwk(ien,i): summation over j=1,ibin2 only
          enddo ! ien=1,nen
        endif
        enddo ! j=1,ibin2
CMPIINSERT_ENDIF_RANK
CMPIINSERT_RECV_EFLUXWK
CMPIINSERT_SEND_EFLUXWK
        enddo ! i=1,lrzmax
         
        call bcast(eflux(1,nn),zero,nen)
        
        do ien=1,nen  ! for each energy bin
        do i=1,lrzmax ! sum-up contributions from all flux surfaces
          eflux(ien,nn)= eflux(ien,nn) + efluxwk(ien,i)
        enddo ! i=1,lrzmax
        enddo ! ien=1,nen

CMPIINSERT_BARRIER 
CMPIINSERT_BCAST_EFLUX

c.......................................................................
c     Integrate energy flux spectra from enmin to enmax.
c     Assume the flux varies as exp(-e/t) between flux values,
c     to improve accuracy.
c.......................................................................

        inegsxr(nn)=nen
        efluxt(nn)=0.0
        
        do 400  ien=1,nen
          if (eflux(ien,nn).le.0.) then
            inegsxr(nn)=ien-1
            go to 402
          endif
          if (ien.eq.nen) go to 402

c         Avoid some overflows:
          if (eflux(ien+1,nn).lt.0.) go to 402
          if (eflux(ien,nn).lt.em100*em100) go to 400
cBH160507     if(eflux(ien,nn).eq.eflux(ien+1,nn))  go to 401
cBH160507 Allowing for possible increase in eflux_npa with en.
cBH160507 Could modify to fork with eflux_npa(i:i+1) are too close
	  if(eflux(ien,nn).le.eflux(ien+1,nn))  go to 401
	  if(eflux(ien+1,nn).eq.zero)  go to 401

c990131          t=(en_(ien+1)-en_(ien))/alog(eflux(ien,nn)/eflux(ien+1,nn))
          t=(en_(ien+1)-en_(ien))/log(eflux(ien,nn)/eflux(ien+1,nn))
cBH160507          c=eflux(ien,nn)*exp(en_(ien)/t)
cBH160507          write(*,*)'tdnpa0.f: i,t,c,eflux_npa(i,nn)=',
cBH160507     +                         i,t,c,eflux_npa(i,nn)
cBH160507          efluxt(nn)=efluxt(nn)+c*t*
cBH160507     +      (exp(-en_(ien)/t)-exp(-en_(ien+1)/t))*1.e3
	  efluxt(nn)=efluxt(nn)+t*(eflux(ien,nn)-eflux(ien+1,nn))*1.e3
          go to 400
 401      efluxt(nn)=efluxt(nn)+eflux(ien,nn)*(en_(ien+1)-en_(ien))*1.e3
 400    continue !   ien=1,nen
 402    continue
c
        icall="notfirst"

c.......................................................................
c*********END loop over viewing angles      **************************
c.......................................................................

 420  continue ! nn=1,nv

c.......................................................................
c     Plot eflux........
c.......................................................................

      nenaa=nena
      if (iplotsxr.eq.'yes')
     +    call tdsxrplt(en_,eflux,nen,nenaa,
     +            efluxt,nv,inegsxr,softxry,npa_diag,lnwidth)

c.......................................................................
c     Plot SXR view cords in poloidal cross-section, if eqmod="enabled"
c.......................................................................

      if (eqmod.eq."enabled" .and. icalls.eq."first" .and.
     +     iplotsxr.eq.'yes') then
         call tdsxrvw(tempp4,tempp5,tempp6)
      endif

      deallocate(tempp4,tempp5,tempp6,STAT=istat1)

      return
      end
