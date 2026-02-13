c
c
      subroutine freyasou(qx,qy,qz,qr,vx,vy,vz,iqts,curdep,
     1  bmsprd,multiply,multiplyn,ibstart)
      implicit integer (i-n), real*8 (a-h,o-z)

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      character*8 multiply,method

      dimension qx(*),qy(*),qz(*),qr(*),vx(*),vy(*),vz(*)
      dimension llbrth(0:lrza), curbrth(lrza)
      dimension d2vphipl(lrza),workk(3*lrza+1)

      dimension ranvc(:)
      pointer ranvc
      dimension px(:)
      pointer px
      dimension py(:)
      pointer py
      dimension pz(:)
      pointer pz
      dimension lbrth(:)
      pointer lbrth
      dimension psipt(:)
      pointer psipt
      dimension pr(:)
      pointer pr

c
c..................................................................
c
c     This routine computes the bounce averaged source suitable for
c     time advancement from the ion birth points as determined by the
c     NFREYA code as it exists in the transport code ONETWO. For the
c     moment the value of psi at which the particles are born is
c     assumed to be the same as the bounce average of psi. qx qy and
c     qz are the locations of birth in a cartesian system, and vx,vy and
c     and vz are the velocity coordinates. 
c     kfrsou(ibstart+1) is the species index.
c
c..................................................................
c     psivalm(l) is the value of psi that delimits the EXACT OUTER EDGE
c     of flux volume l. The volume, area and incremental volume and
c     area arrays, vol, area, dvol and darea (functions of l) are
c     computed with this boundary in mind. Thus there is a PRECISE
c     correspondence between the volume between psival(l-1) and
c     psival(l) and dvol(l).
c..................................................................

      itab(1)=1
      itab(2)=0
      itab(3)=0
      method="meth1"

      twopi2=twopi*twopi
      
      k=kfrsou(ibstart)
        
      ! q/mc for source particles:
      qmc= charge*bnumb(k)/(fmass(k)*clight) 
      qmca=abs(qmc)
      do 10 l=1,lrz
        tr(l)=psimag-psivalm(l)
 10   continue
      tr(lrz)=psimag-psilim

c..................................................................
c     If variable multiply.ne."disabled" and multiplyn.ge.1, 
c     we assume that each particle birth
c     point represents the mean value of a normally distributed
c     beamlet with standard deviation equal to beamsprd. If multiply
c     is "disabled" then the code goes with the actual NFREYA deposition
c     with no alterations.
c     In the case that multiply is activated, multiply particles are
c     born distributed as described above. Generate space for these new
c     particles and for the random number vector.
c..................................................................

      ma=multiplyn
      if (multiply.eq."disabled") ma=1
      ipts=iqts*ma
      mrans=ipts*3

      write(*,*)'freyasou: allocating ipts 6 times, ipts=',ipts
      allocate(px(ipts),STAT=istat)
      call bcast(px,zero,SIZE(px))
      allocate(py(ipts),STAT=istat)
      call bcast(py,zero,SIZE(py))
      allocate(pz(ipts),STAT=istat)
      call bcast(pz,zero,SIZE(pz))
      allocate(lbrth(ipts),STAT=istat)
      call ibcast(lbrth,0,SIZE(lbrth)) !-YuP: was 'zero', which is 0.d0
      allocate(psipt(ipts),STAT=istat)
      call bcast(psipt,zero,SIZE(psipt))
      allocate(pr(ipts),STAT=istat)
      call bcast(pr,zero,SIZE(pr))
      if (multiply.ne."disabled") then
         allocate(ranvc(mrans),STAT=istat)
         call bcast(ranvc,zero,SIZE(ranvc))


c..................................................................
c     Call random number generator - scale to obtain desired  
c     standard deviation
c..................................................................

c     iseed=123457        ***old**bh
c     call rnset(iseed)   ***old**bh
c     call rnnoa(mrans,ranvc)   ***old**bh
        call frnnoa(mrans,ranvc) !here: for multiply.ne."disabled"
        bmra=bmsprd*radmin
        call dscal(mrans,bmra,ranvc,1)

c..................................................................
c     Determine the birth points....
c..................................................................

        mpar=0
        mran=0
        do 30 ipar=1,ipts
          jpar=(ipar-1)/multiplyn+1
          px(ipar)=qx(jpar)+ranvc((ipar-1)*3+1)
          py(ipar)=qy(jpar)+ranvc((ipar-1)*3+2)
          pz(ipar)=qz(jpar)+ranvc((ipar-1)*3+3)
 30     continue
      else
        do 45 ipar=1,ipts
          px(ipar)=qx(ipar)
          py(ipar)=qy(ipar)
          pz(ipar)=qz(ipar)
 45     continue
      endif  !on multiply
      write(*,*)'freyasou: After allocate'

c      write(*,*)
c      write(*,*)'freyasou:i,qx(i),qy(i),qz(i),qr(i),vx(i),vy(i),vz(i):'
c      do i=1,5    
c         write(*,*) i,qx(i),qy(i),qz(i),qr(i),vx(i),vy(i),vz(i)
c      enddo

c..................................................................
c     Mark birth particles according to their birth flux surface bin.
c     lbrth(ipar) is the label of the relevant flux surface.
c..................................................................

      do 60 ipar=1,ipts
      
        rbrth2= px(ipar)**2 +py(ipar)**2
        rbrth=sqrt(rbrth2)
        pr(ipar)=rbrth
        psibrth=terp2(rbrth,pz(ipar),nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1    epsirz,nnra,0,0)
        psipt(ipar)=psibrth
        apsi=psimag-psibrth
        if (psibrth.lt.psilim) then  !i.e., outside LCFS
          lbrth(ipar)=0
        else
          lbrth(ipar)=luf_bin(apsi,tr(1),lrz)
        endif

        if(fr_gyrop.eq.'enabled')then ! YuP[03/13/2015]
           ! Note: fr_gyrop=fr_gyro through frnfreya call.
           ! Adjust coordinate from local position to g.c. position
           ! First, Determine components of B:
           dpsidr=terp2(rbrth,pz(ipar),nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1       epsirz,nnra,1,0)
           dpsidz=terp2(rbrth,pz(ipar),nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1       epsirz,nnra,0,1)
           call eqfpsi(psibrth,fpsi_,fppsi_)
!           Bpol_factor=0d0   !Test: zero Bpol effect, leaving only btor.
           Bpol_factor=1.d0
           
           if(rbrth.gt.1.d-8)then !YuP[2018-01-24] Added,similar to do_100 below
             bxcomp=(-cursign*Bpol_factor*dpsidz*px(ipar)-bsign*fpsi_*
     1          py(ipar))/rbrth2
             bycomp=(-cursign*Bpol_factor*dpsidz*py(ipar)+bsign*fpsi_*
     1          px(ipar))/rbrth2
             bzcomp=+cursign*Bpol_factor*dpsidr/rbrth
           else ! in a mirror machine, can get to R=0
             go to 60 ! skip, for now
           endif
           
           if (multiply.eq."disabled") then
              jpar=ipar
           else
              jpar=(ipar-1)/multiplyn+1
           endif
           bmag2= bxcomp**2+bycomp**2+bzcomp**2 ! B^2
           !===> This part is to add gyro-correction
           !    (to find g.c. position from a local position of an ion)
           one_wcb= 1.d0/(qmc*bmag2) ! = 1/(omega_c*B)
           ! Components of rho_gyro are found 
           ! from  rho_gyro= (1/omega_c)[v x B]/|B|  
           rhox_gyro= one_wcb*(vy(jpar)*bzcomp - vz(jpar)*bycomp)
           rhoy_gyro= one_wcb*(vz(jpar)*bxcomp - vx(jpar)*bzcomp)
           rhoz_gyro= one_wcb*(vx(jpar)*bycomp - vy(jpar)*bxcomp)
           ! The vector rho_gyro is directed from ion position 
           ! towards the gyro-center (for a positive charge q)
           !-> Gyro-radius correction: 
           px(ipar)= px(ipar) +rhox_gyro ! Now px is at g.c.
           py(ipar)= py(ipar) +rhoy_gyro ! Now py is at g.c.
           pz(ipar)= pz(ipar) +rhoz_gyro ! Now pz is at g.c.
           ! Update other values. Now - at g.c. position:
           rbrth2= px(ipar)**2 +py(ipar)**2
           rbrth=sqrt(rbrth2)
           pr(ipar)=rbrth
           psibrth= terp2(rbrth,pz(ipar),nnr,er,nnz,ez,epsi,
     1                    epsirr,epsizz,epsirz,nnra,0,0)
           psipt(ipar)=psibrth ! at g.c. now
           apsi=psimag-psibrth
           if (psibrth.lt.psilim) then  !i.e., g.c. outside LCFS
              !Or maybe still use the actual particle position for lbrth?
              lbrth(ipar)=0
           else
              lbrth(ipar)=luf_bin(apsi,tr(1),lrz)
           endif
           ! Also, update gx,qy,qz - it will update xpts,ypts,zpts
           qx(ipar)=px(ipar)
           qy(ipar)=py(ipar)
           qz(ipar)=pz(ipar)  ! Need to check multiply.eq."enabled" case
        endif ! fr_gyro
          
 60   continue !ipar=1,ipts

      
c      do ipar=1,10
c         write(*,*)'freyasou: ipar,psipt(ipar),lbrth(ipar)'//
c     +        'vx(ipar),vy(ipar),vz(ipar)=,',
c     +        ipar,psipt(ipar),lbrth(ipar),vx(ipar),vy(ipar),vz(ipar)
c      enddo




c..................................................................
c     Subtract rotation velocity vphipl from the toroidal 
c     velocity of the particles, if n.ge.nonvphi .and. n.lt.noffvphi.
c..................................................................

      if ((n+1).ge.nonvphi .and. (n+1).lt.noffvphi) then
        if (multiply.ne."disabled") stop 'rerun with multiply=disabled'
        sum_vphipl= sum(abs(vphipl(1:lrz))) !YuP[2022-11-17] To verify
        if(sum_vphipl.gt.0.d0) then 
          !YuP[2022-11-17] Added verification of vphipl - skip if no rotation
          i1p(1)=4
          i1p(2)=4
          itab(1)=1
          itab(2)=0
          itab(3)=0
          call  coeff1(lrzmax,equilpsp(1),vphipl,d2vphipl,i1p,1,workk)
          do 70 ipar=1,ipts
            psibrth=psipt(ipar)
            !prevent possibility of unhealthy extrapolation
            if (psibrth.lt.psilim) psibrth=psilim
            if(psibrth.gt.(-equilpsp(1))) psibrth=-equilpsp(1)
            psibrth=-psibrth
            call terp1(lrzmax,equilpsp(1),vphipl,d2vphipl,psibrth,
     +      1,tab,itab)
            if(rmag.gt.1.d-8)then
              vphiplas=tab(1)*(pr(ipar)/rmag) ! rmag=0 in mirror machine
              cosp=px(ipar)/pr(ipar)
              sinp=py(ipar)/pr(ipar)
              vphi=-sinp*vx(ipar)+cosp*vy(ipar)
              vphip=vphi-vphiplas
              vrp=cosp*vx(ipar)+sinp*vy(ipar)
              vx(ipar)=vrp*cosp-vphip*sinp
              vy(ipar)=vrp*sinp+vphip*cosp
              !vz(ipar)=vz(ipar)
            else
              goto 70 ! skip, for now
            endif
c          if (ipar.lt.10) then
c             write(*,*)'freyasou: ipar,psibrth,vphiplas,'//
c     +            'vx(ipar),vy(ipar),vz(ipar)=,',
c     +            ipar,psibrth,vphiplas,vx(ipar),vy(ipar),vz(ipar)
c          endif
 70       continue ! ipar=1,ipts
        endif !(sum_vphipl.gt.0.d0) YuP[2022-11-17]skip if no rotation
      endif ! ((n+1).ge.nonvphi .and. (n+1).lt.noffvphi)

c.......................................................................
cBH110614:  Loop structure is changed.  Main loop is now over
cBH110614:  particles, and data is accumulated for each flux
cBH110614:  surface as advance through the particle (ipar) loop.
c
cxx     Begin loop over the flux surfaces: Recall that information
cxx     pertaining to individual flux surfaces (including S) is
cxx     stored on disk. This introduces some awkwardness in this
cxx     routine (such as the existence of do loop 60).
c.......................................................................

c.......................................................................
c     curnorm will be the integral over the source over velocity and
c     configuration space volume. To get the actual current, the
c     source arrays which are being computed below will have to
c     be divided by curnorm (giving total current=1.) and then
c     multiplied by curdep, curdep being the total number of
c     particles deposited per second in the plasma.
c     asor will (after rescaling at the end of the routine) have in it
c     the local current (part/sec) for the flux surface in question.
c.......................................................................

      curnorm=0.
      do 200 ll=1,lrz
        call tdnflxs(ll) !-> get l_,lr_, etc.
        llbrth(ll)=0
cBH171014        call bcast(source(0,0,kfrsou,ll),zero,iyjx2)
c       k=kfrsou(ibstart) ! already done at very beginning
        call bcast(source(0,0,k,ll),zero,iyjx2)

c..................................................................
c     Begin loop over the birth particles.
c     Determine the value of psi and R at birth
c..................................................................

        k=kfrsou(ibstart) ! already done at very beginning
        do 100 ipar=1,ipts

          if (lbrth(ipar).ne.ll) go to 100
          llbrth(ll)=llbrth(ll)+1
          rbrth=pr(ipar)
          rbrth2= rbrth*rbrth
          psibrth=psipt(ipar)
!          write(*,*)'freyasou, do 100, ll,ipar,ipts=',ll,ipar,ipts
!          write(*,*)'freyasou, do 100, rbrth,psibrth=',rbrth,psibrth

c..................................................................
c     Determine the pitch angle (cos(theta)) at birth. Compute
c     B dot v / mag B/ mag v
c..................................................................

          dpsidr=terp2(rbrth,pz(ipar),nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1      epsirz,nnra,1,0)
          dpsidz=terp2(rbrth,pz(ipar),nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1      epsirz,nnra,0,1)

c..................................................................
c     Determine f(psi)
c..................................................................

          call eqfpsi(psibrth,fpsi_,fppsi_)

c..................................................................
c     Determine components of B.
c..................................................................
          if(rbrth.gt.1.d-8)then 
cBHandYuP110715: Account for bsign/cursign

          bxcomp=(-cursign*dpsidz*px(ipar)-bsign*fpsi_*py(ipar))/rbrth2
          bycomp=(-cursign*dpsidz*py(ipar)+bsign*fpsi_*px(ipar))/rbrth2
          bzcomp=+cursign*dpsidr/rbrth
          else ! in a mirror machine, can get to R=0
             go to 100 ! skip, for now
          endif

c.......................................................................
c     Compute the dot product and the cosine
c     BH110614:  But, pitch angle is measure from B-field,
c     BH110614:  however, it is pos. in pos. tor. dirn, by convention.
c     BH110614:  In present NSTX study, bsign=-1., but get pos cospitch
c     BH110614:  for part with pos vphi.  Maybe ok, by logic needs
c     BH110614:  verifying.  Also, effect of cursign?
c.......................................................................

          if (multiply.eq."disabled") then
            jpar=ipar
          else
            jpar=(ipar-1)/multiplyn+1
          endif
          dotprd=bxcomp*vx(jpar)+bycomp*vy(jpar)+bzcomp*vz(jpar)
          vmag=sqrt(vx(jpar)**2+vy(jpar)**2+vz(jpar)**2)
          bmag2= bxcomp**2+bycomp**2+bzcomp**2 ! B^2
          bmag=sqrt(bmag2)
cBHandYuP110715: cospitch measured with vpar pos in pos vphi
cBHandYuP110715: (CCW dirn viewed from top).
          cospitch=bsign*dotprd/bmag/vmag ! local cos(pitch)
c          write(*,*)'freyasou: ll,ipar,vmag,bmag,cospitch=',
c     +                         ll,ipar,vmag,bmag,cospitch

c..................................................................
c     Translate to midplane coordinates. This requires knowledge
c     of B-ratio = B(at birth)/B(at midplane). We require rcon,
c     the value of R at which the flux surface of birth pierces
c     the midplane.
c     Determine the first mesh point iless such the epsi(iless,nzc)
c     is less than psibrth (the epsi associated with the contour of
c     interest).
c     Two methods can be used; default 1-d splines (meth1)
c     Other method is probably much slower (meth2).
c..................................................................

          if (method.eq."meth2") then   !NOT TRUE, method hardwired.
            do 110 i=imag+1,nnr
              if (epsi(i,nzc).lt. psibrth) then
                iless=i
                go to 111
              elseif (i.eq.nnr) then
                call frwrong(3)
              endif
 110        continue
 111        continue

c..................................................................
c     Do a Newton's iteration to find the R=rcon where the contour
c     intersects the Z=0. axis.
c..................................................................

            iter=0
            imore=iless-1
            rcon=er(iless)
            epsitst=epsi(iless,nzc)
 115        continue
            dpsidr=terp2(rcon,zero,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1        epsirz,nnra,1,0)
            a=(psibrth-epsitst)/dpsidr

c..................................................................
c     New value for rcon..
c..................................................................

            rcon=rcon+a
            epsitst=terp2(rcon,zero,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1        epsirz,nnra,0,0)
            err=abs((psibrth-epsitst)/(epsi(imore,nzc)+epsi(iless,nzc)))
            if (err .lt. 1.e-3) go to 120
            iter=iter+1
            if (iter.gt.5 .and. err.gt..01) call frwrong(4)
            if (iter.gt.5) go to 120
            go to 115
 120        continue


c..................................................................
c     Determine B0, i.e., B at midplane
c..................................................................

cBH990429            dpsidr=terp2(rcon,rbrth,0.,nnr,er,
cBH990429                         nnz,ez,epsi,epsirr,epsizz,
            dpsidr=terp2(rcon,zero,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1        epsirz,nnra,1,0)
            bzero=sqrt(fpsi_**2+dpsidr**2)/rcon

          elseif (method.eq."meth1") then   !TRUE

            psb=-psibrth
            call terp1(lrzmax,equilpsp(1),bmdplne,d2bmdpl,psb,
     +        1,tab,itab)
            bzero=tab(1)

          endif

          bratio=bmag/bzero
c         bratio, physically, is .ge.1.  Due to inaccuracies, it can
c         numerically get .lt.1.   
c         Allowing for some tolerance, these cases are adjusted.
          if (bratio.lt.1.) then
            bratio=1.0
            if (bratio.lt.0.99)
     &      write(*,*)'freyasou: bratio=[should be=1, else stop]',bratio
cBH041215            if (bratio.le.0.9) stop 'freyasou: bratio.le.0.9'
          endif
          tmdplne=asin(sqrt((1.-cospitch**2)/bratio))
          if (cospitch.lt.0.) then
            tmdplne=pi-tmdplne
          endif

c..................................................................
c     Determine the indices of the midplane meshes to which the
c     velocity coordinates at the birth point transform.
c     i for theta, j for velocity.
c     Locate the flux surface to which this particle will be
c     assigned.
c..................................................................

          j=luf_bin(vmag/vnorm,xmidpt,jx)
          if (j.gt.jx) go to 100
          i=luf_bin(tmdplne,ymid(1,l_),iy)
          if (i.gt.iy) then
             write(*,*)'freyasou: ll,i,j,iy,ipar,tmdplne=',
     +            ll,i,j,iy,ipar,tmdplne
             call frwrong(5)
          endif

c..................................................................
c     Increment the source by the birth particle weight.
c     2. in denominator added because zmaxpsi(lr_), vptb are half orbit
c     quantities in CQL
c..................................................................

cVickieLynch060824 found this should be before do 100:          k=kfrsou
          aa=bmdplne(lr_)/(cynt2(i,l_)*cint2(j)
     1      *dpsi(lr_)*twopi*vptb(i,lr_)*one_*symm)
          !YuP[03/27/2015] Changed 2.0->symm,  
          !which is symm=2 in case of up-dn symmetrical equilibrium.
          !In this case,   
          !for passing particles vptb*symm corresponds to the whole orbit;
          !for trapped: vptb*symm corresponds to half orbit (from tip to tip)

c..................................................................
c     Symmetrize in trapped region...
c..................................................................

          if (i.ge.itl_(l_) .and. i.le.itu_(l_)) then   !trapped
            source(i,j,k,ll)=source(i,j,k,ll)+aa/2. ! ZOW
            ! Factor 1/2 because vptb for trapped - over 1/4 physical orbit
            !(in case of up-dn symmetrical equil) 
            source(iy+1-i,j,k,ll)=source(i,j,k,ll)  ! ZOW
          else
            source(i,j,k,ll)=source(i,j,k,ll)+aa    ! ZOW
          endif
 100   continue ! ipar=1,ipts
c=========================================== Done LOOP IN PARTICLES 

 
        s=0.
        do 150 j=1,jx
          do 160 i=1,iy
            s=s+source(i,j,k,ll)*cynt2(i,l_)*cint2(j)*vptb(i,lr_)
 160      continue
 150    continue

c..................................................................
c     asor(k,1,lr_) will temporarily hold the non-normalized current
c..................................................................

        asor(k,1,lr_)=s/zmaxpsi(lr_)*one_
        asorz(k,1,lr_)=asor(k,1,lr_)
        curnorm=curnorm+dvol(lr_)*asor(k,1,lr_)
        call tdtoaray
 200  continue !ll=1,lrz

c_cray    call hpdeallc(pxptr,ier,0)
c_cray    call hpdeallc(pyptr,ier,0)
c_cray    call hpdeallc(pzptr,ier,0)
c_cray    call hpdeallc(prptr,ier,0)
c_cray    call hpdeallc(lbrthptr,ier,0)
c_cray    call hpdeallc(psiptptr,ier,0)
c_pc      Using unix library libU77.a.  
c_pc      call free(pxptr)
c_pc      call free(pyptr)
c_pc      call free(pzptr)
c_pc      call free(prptr)
c_pc      call free(lbrthptr)
c_pc      call free(psiptptr)
      deallocate(px,STAT=istat)
      deallocate(py,STAT=istat)
      deallocate(pz,STAT=istat)
      deallocate(pr,STAT=istat)
      deallocate(lbrth,STAT=istat)
      deallocate(psipt,STAT=istat)
      if (multiply.ne."disabled") then
         deallocate(ranvc,STAT=istat)
      endif
      call bcast(tr3(1),one,lrz)

c..................................................................
c     Smooth the source if input variable smooth .ge. .001
c..................................................................
     
      call frsmooth(k,curnorm)
      if(curnorm.le.0.d0)then !YuP[2022-12] added warning
CMPIINSERT_IF_RANK_EQ_0      
      WRITE(*,'(a,1pe11.3)')' freyasou(565) WARNING: curnorm=',
     &  curnorm
CMPIINSERT_ENDIF_RANK
      endif

c..................................................................
c     Compute the factor necessary to scale asor so that the source
c     current is equal to curdep (a quantity determined by NFREYA).
c     If frsmooth has been utilized, tr3 has been altered as has
c     curnorm.
c..................................................................

      scalfact=curdep/curnorm
CMPIINSERT_IF_RANK_EQ_0      
      WRITE(*,'(a,1p3e11.3)')' freyasou(580) scalfact,curdep,curnorm=',
     &  scalfact,curdep,curnorm
CMPIINSERT_ENDIF_RANK

c..................................................................
c     Scale all the sources (at each flux surface) by scalfact*tr3.
c     This forces the source current equal to curdep.
c..................................................................

      do 300 ll=1,lrz
        call tdnflxs(ll)
        curbrth(ll)=dfloat(llbrth(ll))/dvol(lr_)*scalfact
        scal_s=scalfact*tr3(ll)
        !call dscal(iyjx2,scal_s,source(0:iy+1,0:jx+1,k,ll),1) !issues in certain debug mode
        do jj=0,jx+1      
        do ii=0,iy+1
          source(ii,jj,k,ll)=source(ii,jj,k,ll)*scal_s
        enddo
        enddo
        asor(k,1,lr_)=asorz(k,1,lr_)*scalfact*tr3(ll)
        xlncur(k,lr_)=asor(k,1,lr_)*zmaxpsi(lr_)
        xlncurt(lr_)=xlncur(k,lr_)
        !YuP: note that asorz is NOT rescaled;
        ! but - it seems it is not used further?
        call sourcpwr(k) !YuP[2020] Maybe no need to call here. 
                         !It is also called later from achiefn-->sourcee
        call tdtoaray
CMPIINSERT_IF_RANK_EQ_0      
        sum_s=0.d0
        do jj=0,jx+1      
        do ii=0,iy+1
          sum_s=sum_s+source(ii,jj,k,ll)
        enddo
        enddo
        WRITE(*,'(a,i3,1pe12.4)') 
     &   'freyasou_end. lr,sum_ij(source)=',ll,sum_s
CMPIINSERT_ENDIF_RANK
 300  continue !ll

      return
      end subroutine freyasou
c======================================================================
c======================================================================


      subroutine frnnoa(mrans,ranvc)
      implicit integer (i-n), real*8 (a-h,o-z)
c     Generate a vector of mrans normal (0,1) pseudo-random numbers
c     using a method based on the central limit theorem
c     and power residue method.  Output becomes truly normal
c     as k (below) goes to infinity.
c     General formula for the deviate is
c     y=((sum of x(i),i=1,k)-k/2.)/sqrt(k/12.),
c     where x(i) are uniformally distributed on 0,1.
c     Method is borrowed through ibm.  They use k=12.
c
c     parameter (k=12,rtkd12=1.00)
c
      dimension ranvc(mrans)

cYuP110316      integer iflag

cYuP110316      iflag=0
      seed=0.d0  ! I.E., continue RN sequence in RANDOM_my

c990131      iseed=123457
c990131      call  ranset(iseed)
c     drand() is a unix library function (see Absoft f77 manual).
c     drand(0) returns next real*8 random number of the
c     sequence.  The input parameter must be integer. 
c     Presumably each realization of the
c     sequence is the same.  Rand(iflag.ne.0) could be
c     used to start a different sequence.

cBH060821:
c     random_number(harvest) is [0.,1.] r.n., an f90 intrinsic sub
c     call random_seed(size,put,get) see f90 intrinsics
c     call random_seed() initializes the rn generator.

c      call random_seed()

cBH060821:  BUT, instead of random_number will use RANDOM_my,
c           which is from the GA portlib.f (see function in zfreya.f)
      k=12
      rtkd12=1.0
      do 100 i=1,mrans
        a=0.0
        do 10 j=1,12
c990131          a=a+ranf()
cBH060821          a=a+drand(iflag)
cBH060821           call random_number(harvest)
cBH060821           a=a+harvest

cYuP110316           a=a+RANDOM_my(iflag)
           a=a+RANDOM_my(seed)  !input should be real*8
c-YuP           a=a+drand(iflag) ! YuP: unresolved drand
 10     continue
        ranvc(i)=(a-0.5*k)/rtkd12
 100  continue
      return
      end
