c
c
      subroutine pltcont(k,pltcase,tt_,itype)
      !YuP[2018-02-07] added input itype, to identify what is plotted.
      ! itype=1 for plots of f(),
      !      =2 for df
      !      =3 for source (from subr.souplt)
      !      =4 for urfb
      !      =5 for vlfb
      !      =6 for vlhb
      !      =7 for rdcb
      !      =8 for loss region (from pltlosc)
      !      =9 for Drr coeff (from subr.drrplt) !Added [2021-08]
      implicit integer (i-n), real*8 (a-h,o-z)
      save
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      parameter(nconta=100)
      common/contours/cont(nconta),tempcntr(nconta)
      integer pltcase
      character*(*) tt_
      character*64 tx_,ty_
      REAL*4 wk_tam(jx)
      real*8 wkd(jx) 
c...  
cmnt  This routine performs contour plots of distributions
cmnt  given in temp1(0:iyp1,0,jxp1) as a function of v,theta,
cmnt  as specified by x,y.
cmnt    pltcase=1: geometric progression of plot contours,
cmnt    pltcase.ne.1: contours equispaced for max. at temp(k,lr_),
!                     or temppar(k,ls_) for CQLP
cmnt    k gives species index.
cmnt    tt_ is contour plot heading.
cmnt    Additional annotation of the plot can be added from
cmnt    the calling routine.
c...
C
C     PASSING ARRAYS TO PGFUNC1, FOR PGPLOT PGCONX:
      pointer wx,wy
      REAL*4 wx(:),wy(:), xpt,ypt
      REAL*4 RCONT,RXMAXQ,RTEMP1,RXPTS,RYPTS
      DIMENSION RCONT(NCONTA),RTEMP1(iymax,jx),RXPTS(2),RYPTS(2)
      COMMON /PGLOCAL1/wx,wy,IIY,JXQ
C     wx IS V-NORM ARRAY, wy IS THETA ARRAY.  TYPE REAL.
      real*4 RTAB1(iymax),RTAB2(iymax) ! local
      !YuP[2021-03-11] Changed iy-->iymax in declarations
      !(just in case if iy is changed by iy=iy_(l_) somewhere)

      !YuP[2018-01-27] Local, for PGPLOT:
      parameter(npar=200, nprp=100)
      real*8 vpar(npar),vprp(nprp) ! rectangular grid for plots 
      real*8 fparprp(npar,nprp) !f(i,j) will be interpolated to this grid
      real*4 BRIGHT,CONTRA,FMIN,FMAX,RVMIN,RVMAX,
     +       TRPG(6),RTEMP2(npar,nprp)
      REAL*4 RCONTLOG(NCONTA),FLOGMIN,FLOGMAX

      REAL*4 :: R40=0.,R4P2=.2,R4P8=.8,R4P65=.65,R4P9=.9
      REAL*4 :: R42=2.,R45=5.,R46=6.,R47=7.,R48=8.

      EXTERNAL PGFUNC1


CMPIINSERT_IF_RANK_NE_0_RETURN
 ! make plots on mpirank.eq.0 only

      if (noplots.eq."enabled1") return

      if(ASSOCIATED(wx)) then
        ! wx and wy are already allocated => do nothing 
      else ! Not allocated yet
        allocate(wx(jx))
        allocate(wy(iymax))
      endif

      if (ncont.gt.nconta) stop 'in pltcont'

      dmin=0.
      dmax=0.

c-----YuP[2018-01-08] revised to match cqlinput_help:
c**    pltlim= "disabled",  plots versus x (u/vnorm) from
c**                    x=0. to 1. (default:pltlim="disabled",pltlimm=1.)
c**            "x",    plot 1d and 2d plots versus x 
c**                    from 0. to pltlimm.
c**            "u/c",  plot 1d and 2d plots versus u/c
c**                    from 0. to pltlimm.
c**            "energy", plot 1d plots verus energy (kev)
c**                    from 0. to pltlimm (kev).
cyup                   BUT, for 2d plots, use u/c units, not keV
      if (pltlim.eq."disabled") then ! whole range in x(j)
         jxq=jx
         xmaxq=x(jxq)
         !iyjxq=iymax*jxq !YuP[2021-03] What for? Commented
         tx_='v_parallel (u/vnorm)'
         ty_='v_perp (u/vnorm)'
      elseif (pltlim.eq.'u/c' .or. pltlim.eq.'energy') then
         if (pltlim.eq.'u/c') then
            pltlimmm=pltlimm
         else
            pltlimmm=sqrt((1.+pltlimm/restmkev)**2-1.)
         endif
         jxq=min(luf(pltlimmm,uoc,jx),jx)
         xmaxq=uoc(jxq)
         !iyjxq=iy*jxq !YuP[2021-03] What for? Commented
         tx_='v_parallel (u/c)'
         ty_='v_perp (u/c)'
      elseif (tandem.eq."enabled" .and. fmass(k).gt.1.e-27) then
         jxq=jlwr
         xmaxq=xlwr
         !iyjxq=iy*jlwr !YuP[2021-03] What for? Commented
         tx_='v_parallel (u/vnorm)'
         ty_='v_perp (u/vnorm)'
      else ! 'x'
         pltlimmm=pltlimm
         jxq=min(luf(pltlimmm,x,jx),jx)
         xmaxq=x(jxq)
         !iyjxq=iy*jxq !YuP[2021-03] What for? Commented
         tx_='v_parallel (u/vnorm)'
         ty_='v_perp (u/vnorm)'
      endif

      if (pltlim.eq.'u/c' .or. pltlim.eq.'energy') then
         do j=1,jxq
            tam1(j)=uoc(j) !==x(j)/cnorm
         enddo
      else ! 'disabled', 'x', or tandem 
         do j=1,jxq
            tam1(j)=x(j)
         enddo
      endif

C     

      DO J=1,JXQ
         DO I=1,iy_(l_) !YuP[2021-03] For meshy="fixed_mu" iy_(l_)<iy at some l_
            DMIN=MIN(temp1(I,J),DMIN) !temp1() contains f(), or other arrays
            DMAX=MAX(temp1(I,J),DMAX)
            RTEMP1(I,J)=temp1(I,J)
         ENDDO
      ENDDO


      admin=abs(dmin)
      
      if( (itype.eq.1 .and. pltd.eq.'color')   .or. 
     +    (itype.eq.1 .and. pltd.eq.'df_color')      .or.      
     +    (itype.eq.2 .and. pltd.eq.'df_color'   )   .or. 
     +    (itype.eq.3 .and. pltso.eq.'color')  .or. 
     +    (itype.eq.3 .and. pltso.eq.'first_cl')  .or.
     +    (itype.eq.4 .and. plturfb.eq.'color')   .or.
     +    (itype.eq.7 .and. pltrdc.eq.'onecolor')   .or.
     +    (itype.eq.7 .and. pltrdc.eq.'allcolor')   .or.
     +    (itype.eq.9 .and. pltdrr.eq.'color')    .or. 
     +    (itype.eq.9 .and. pltdrr.eq.'first_cl') 
     &                                                  ) then 
          !Other itype can be added later
     
      !YuP[2018-01-27] Added: interpolate f(i,j) to fparprp(npar,nprp)
      ! over the rectangular (vpar,vprp) grid.
      ! Set rect. grid, then interpolate f(i,j) to this grid
      f_zero=1.d-100
      vmax= xmaxq  ! either u/c or u/vnorm units
      vmin=-xmaxq  ! either u/c or u/vnorm units
      dvpar=(vmax-vmin)/(npar-1)
      do ipar=1,npar
        vpar(ipar)= vmin+dvpar*(ipar-1) ! either u/c or u/vnorm units
      enddo
      dvprp=(vmax-0.d0)/(nprp-1)
      do iprp=1,nprp
        vprp(iprp)= 0.d0+dvprp*(iprp-1) ! either u/c or u/vnorm units
      enddo
      ! Interpolate f(i,j) to fparprp(npar,nprp)
      ! ADJUST f(i,j) - to eliminate zero values.
      ! This is important if we want to consider LOG10(f).
      ! For now, simply set it to f_zero
      fparprp= f_zero ! ~zero level ! initialize
      dxlast=tam1(jxq)-tam1(jxq-1) ! will be used for (vpar,vprp) points
                           ! outside of tam1(jxq) semi-circle
      do iprp=1,nprp
      do ipar=1,npar
         ! for a given (vpar,vprp) find the four nearest points 
         ! in (i,j) grid 
         !(y(i)= pitch angle in radians)
         vloc= sqrt(vpar(ipar)**2 + vprp(iprp)**2) !either u/c or u/vnorm units
         ploc= atan2(vprp(iprp),vpar(ipar)) ! pitch angle [rad]
         !If vprp=0, the result is 0 (if vpar >0) or pi (if vpar <0).
         !-1-> For the given vloc, find the nearest j index in tam1(j) grid:
         call lookup_tdf(vloc,tam1,jxq,rweightu,rweightl,jloc) 
         ! vloc is between tam1(jloc-1) and tam1(jloc) grid points
         ! rweightu,rweightl are the weight factors for lin. interpolation.
         !-2-> For the given ploc, find the nearest i index
         !     in y0pi(i) pitch grid.
         call lookup_tdf(ploc,y,iy_(l_),pweightu,pweightl,iloc)
         !write(*,*)ploc,iloc
         ! ploc is between y(iloc-1) and y(iloc).
         !-3-> Interpolate values of f(i,j) from four points
         !     to (vpar,vprp) point, with ~zero outside of tam1(jxq) border
         if(vloc.gt.tam1(jxq)+dxlast)then
           ! Far outside of tam1(jxq) semi-circle
           fparprp(ipar,iprp)=f_zero ! basically zero (not defined)
         elseif(vloc.gt.tam1(jxq))then
           ! Outside of tam1(jxq) semi-circle,
           ! but close (vloc is less than tam1(jxq)+dxlast).
           ! Make a gradual drop to ~zero level
           fmm= temp1(iloc-1,jxq) 
           f0m= temp1(iloc,  jxq)
           fm0= f_zero ! ~zero level
           f00= f_zero
           floc= rweightl*( pweightl*fmm + pweightu*f0m )
     +          +rweightu*( pweightl*fm0 + pweightu*f00 )
           fparprp(ipar,iprp)=floc !! set          
         else ! vloc.le.tam1(jxq)i.e. interior to tam1(jxq) semi-circle
           fmm= temp1(iloc-1,jloc-1) 
           f0m= temp1(iloc,  jloc-1)
           fm0= temp1(iloc-1,jloc)
           f00= temp1(iloc,  jloc)
           floc= rweightl*( pweightl*fmm + pweightu*f0m )
     +          +rweightu*( pweightl*fm0 + pweightu*f00 )
           fparprp(ipar,iprp)=floc !! set          
         endif
         ! adjust - to eliminate neg. values:
         fparprp(ipar,iprp)=max(fparprp(ipar,iprp),f_zero) 
         ! Because for PGIMAG, we need LOG10() scale.
      enddo
      enddo
      
      endif ! on color option
      
c**bh 
c
      if(dmax.le.0)  go to 999
c**bh 
c
c     In the case abs(dmin).le.contrmin*dmax, then plot contours
c     occur at levels:
c     cont(j)=dmax * contrmin**(1-(j-.5)/ncont),  j=1,ncont
c     This can be described as a geometric progression of values
c     from (near) dmax down to contrmin*dmax.
c
      if (dmin.ge.0. .or. admin.lt.contrmin*dmax) then
         k2=1
         if(pltcase.eq.1) then
            smin=log(contrmin*dmax)
            if (admin/dmax .gt. contrmin) smin=log(admin)
            smax=log(dmax)
            dcont=(smax-smin)/dfloat(ncont)
            cont(1)=smin+.5*dcont
            do 20 kc=2,ncont
               cont(kc)=cont(kc-1)+dcont
 20         continue
            do 30 kc=1,ncont
               cont(kc)=exp(cont(kc))
 30         continue
         else !pltcase>1

c     Modifying above case, which is usual situation for a function
c     for which dmin .ge.0. .or. not very  negative, to contours which
c     will be equispaced for a Maxwellian distribution with temperature
c     equal to that defined for the distribution:
            if(cqlpmod.ne."enabled")then
              temp_kev= temp(k,lr_)  !CQL3D
            else ! cqlpmod.ne."enabled"
              temp_kev= temppar(k,ls_) ! CQLP [2021-03-02] added
            endif
            emax=-temp_kev*log(contrmin)
            if (emax.gt.enorm) emax=enorm
            gammax=1.+emax/restmkev
            uocmax=sqrt(gammax**2-1.)
            do j=1,ncont
               cont(j)=j/dfloat(ncont)*uocmax
            enddo
            do j=1,ncont
               cont(j)=dmax*
     +              exp(-restmkev*(sqrt(1.+cont(j)**2)-1.)/temp_kev)
            enddo
         endif !pltcase

      else
        if (dmax .gt. 0.) then
          k2=ncont*.1+1
          ncontp=ncont-k2+1
          ncontm=k2-1
          smaxp=log(dmax)
          sminp=log(contrmin*dmax)
        else
          ncontm=ncont
          ncontp=1
          k2=1
        endif
        sminm=log(-contrmin*dmin)
        if (dmax/dmin.gt.contrmin) sminm=log(-dmax)
        smaxm=log(-dmin)
        dcontp=(smaxp-sminp)/dfloat(ncontp)
        dcontm=(smaxm-sminm)/dfloat(ncontm)
        cont(1)=smaxm-.5*dcontm
        do 40 kc=2,ncontm
          cont(kc)=cont(kc-1)-dcontm
 40     continue
        do 50 kc=1,ncontm
          cont(kc)=-exp(cont(kc))
 50     continue
        if (dmax .gt. 0.) then
          cont(k2)=sminp-.5*dcontp
          do 60 kc=k2+1,ncont
            cont(kc)=cont(kc-1)+dcontp
 60       continue
          do 70 kc=k2,ncont
            cont(kc)=exp(cont(kc))
 70       continue
        endif
      endif !dmax
      
      
      if (k2.eq.0) k2=1 ! YuP: bug? was if(k.eq.0)
      do 71 js=1,ncont
        tempcntr(js)=cont(js)
 71   continue

      DO J=1,JXQ
         wx(J)=TAM1(J)
      ENDDO
      DO I=1,iymax
         wy(I)=y(i,l_)
      ENDDO
      DO JS=1,NCONT
         RCONT(JS)=CONT(JS)
         !write(*,*)'pltcont: t_,lr_,JS,CONT(JS)=',lr_,JS,CONT(JS)
         if(CONT(JS).gt.0.d0)then
           if(pltcase.eq.1) then !YuP[2021-08-04] added branch
             RCONTLOG(JS)=LOG10(CONT(JS)) !Can be used for (as an option):
           !CALL PGCONT(RTEMP2,npar,nprp,1,npar,1,nprp,RCONTLOG,-NCONT,TR)
           ! where RTEMP2(ipar,iprp)=log10(fparprp(ipar,iprp)) 
           ! The LOG10 scale is needed for PGIMAG()
           else
             RCONTLOG(JS)= CONT(JS) !
           endif !pltcase
         else
           RCONTLOG(JS)=-100.
         endif       
      ENDDO
      IIY=iy_(l_) !YuP: What for?
      RXMAXQ=XMAXQ

      CALL PGSVP(R4P2,R4P8,R4P65,R4P9)
        IF ( RXMAXQ.eq.0. ) THEN
           RXMAXQ=1.
        ENDIF
      CALL PGSWIN(-RXMAXQ,RXMAXQ,R40,RXMAXQ)
      !CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
       !moved PGBOX AFTER plotting contours,
       !otherwise ticks are overlapped by colored levels
      !CALL PGLAB(tx_,ty_,tt_)

      if( (itype.eq.1 .and. pltd.eq.'color')   .or. 
     +    (itype.eq.1 .and. pltd.eq.'df_color')      .or.
     +    (itype.eq.2 .and. pltd.eq.'df_color'   )   .or. 
     +    (itype.eq.3 .and. pltso.eq.'color')  .or. 
     +    (itype.eq.3 .and. pltso.eq.'first_cl')  .or.
     +    (itype.eq.4 .and. plturfb.eq.'color')   .or.
     +    (itype.eq.7 .and. pltrdc.eq.'onecolor')   .or.
     +    (itype.eq.7 .and. pltrdc.eq.'allcolor')   .or.
     +    (itype.eq.9 .and. pltdrr.eq.'color')      .or. 
     +    (itype.eq.9 .and. pltdrr.eq.'first_cl')  
     &                                                 ) then 
          !Other itype can be added later

      !BH,YuP[2018-01-26] Version 2 for color map of distr.func.
      !Set up the color map.
      BRIGHT=0.5 ! 0.8 gives yellow_low -- red_upper (no blue)
      CONTRA=0.8 !0.5 gives light-blue background 
                 !1.0 gives dark-blue(almost black)
      CALL PALETT(2, CONTRA, BRIGHT)
      !First arg: 
      ! 1- gray scale
      ! 2- rainbow (ok)
      ! 3- heat    (bad: gives black background)
      ! 4- weird IRAF ( really weird: black background and random colors)
      ! 5- AIPS ( not so bad, but 2 probably is the best)
      FMIN=max(f_zero,dmin) ! or could be a fraction of it.
      FMAX=dmax ! or could be a fraction of it.
      !dmin, dmax are found above, for TEMP1(I,J)=f(I,J) (lin. scale). 
      !      but RCONT() levels are log()
      
      !Set the coordinate transformation matrix TR(1:6): 
      !world coordinate = pixel number.
      ! From PGPLOT manual:
      !The transformation matrix TR is used to calculate the world
      !coordinates of the center of the "cell" that represents each
      !array element. The world coordinates of the center of the cell
      !corresponding to array element A(I,J) are given by:
      !          X = TR(1) + TR(2)*I + TR(3)*J
      !          Y = TR(4) + TR(5)*I + TR(6)*J
      !
      ! Based on our X==Vpar= RVMIN  + dvpar*(ipar-1)
      !              Y==Vprp=  0     + dvprp*(iprp-1)
      ! we set:
      RVMIN=vmin ! vmin=-xmaxq  ! either u/c or u/vnorm units
      TRPG(1) = RVMIN-dvpar
      TRPG(2) = dvpar
      TRPG(3) = 0.0
      TRPG(4) = 0.0-dvprp
      TRPG(5) = 0.0
      TRPG(6) = dvprp
      if(pltcase.eq.1) then !YuP[2021-08-04] added branch
        do ipar=1,npar
        do iprp=1,nprp
         RTEMP2(ipar,iprp)=log10(fparprp(ipar,iprp)) ! to REAL*4
         ! LOG scale - better for color plots?
         !FLOGMIN=MIN(FLOGMIN,RTEMP2(ipar,iprp))
        enddo
        enddo
        ! Draw the map with PGIMAG. 
        ! Valid only for RTEMP2(ipar,iprp) over rectangular grid!
        FLOGMAX=LOG10(dmax)
        FLOGMIN=LOG10(dmax*contrmin) ! => FLOGMIN=FLOGMAX-LOG10R 
        CALL PGIMAG(RTEMP2,npar,nprp,1,npar,1,nprp,FLOGMIN,FLOGMAX,TRPG)
        CALL pgwedg('RI',R42,R45, FLOGMIN,FLOGMAX, 'log10(..)') !Colorbar
      else ! pltcase>1
        do ipar=1,npar
        do iprp=1,nprp
         RTEMP2(ipar,iprp)= fparprp(ipar,iprp) ! to REAL*4
         ! LIN scale scale in this case
        enddo
        enddo
        CALL PGIMAG(RTEMP2,npar,nprp,1,npar,1,nprp,FMIN,FMAX,TRPG)
        CALL pgwedg('RI',R42,R45, FMIN,FMAX, '(lin.scale)') !Colorbar
      endif ! pltcase
      
      
      endif !----------- on color option
      
      

      ! Overlay contours, with black color:
      CALL PGSCI(1) ! 1==black 

      ! Plot original f(i,j) in (v,pitch) coord 
      ! (lin.scale of f, but log scale of RCONT)
      CALL PGCONX(RTEMP1,iymax,jx,1,iy_(l_),1,JXQ, RCONT,NCONT,PGFUNC1)

      ! Or plot the interpolated fparprp(ipar,iprp) 
      ! over (vpar,vprp) rectangular grid:
      !CALL PGSLW(1) !lnwidth=1 thin line
      !CALL PGCONT(RTEMP2,npar,nprp,1,npar,1,nprp,RCONTLOG,-NCONT,TR)
      CALL PGSLW(3) !restore lnwidth=3 normal line width
      CALL PGBOX('BCNST',R40,0,'BCNST',R40,0) !with ticks on sides
      CALL PGLAB(tx_,ty_,tt_) !labels


      t0t=sin(thb(l_))/cos(thb(l_))  ! PLOT t-p boundary (ZOW cone)
      CALL PGSLS(2) ! 2-> dashed
      if (t0t .lt. 1.) then
         RXPTS(1)=0.
         RYPTS(1)=0.
         RXPTS(2)=XMAXQ
         RYPTS(2)=XMAXQ*T0T
         CALL PGLINE(2,RXPTS,RYPTS)
         RXPTS(2)=-XMAXQ
         CALL PGLINE(2,RXPTS,RYPTS)
      else
         RXPTS(1)=0.
         RYPTS(1)=0.
         RXPTS(2)=XMAXQ/T0T
         RYPTS(2)=XMAXQ
         CALL PGLINE(2,RXPTS,RYPTS)
         RXPTS(2)=-XMAXQ/T0T
         CALL PGLINE(2,RXPTS,RYPTS)
      endif
      CALL PGSLS(1) ! 1-> restore solid line


      !YuP[03-2016] Added: plot v=vnorm line
      !plot v=vth line, for the k-th gen. species
      if (pltlim.eq.'u/c' .or. pltlim.eq.'energy') then        
        do i=1,iy_(l_)
        RTAB1(i)= coss(i,l_)/cnorm ! v_par/c  (cnorm is c/vnorm)
        RTAB2(i)= sinn(i,l_)/cnorm ! v_perp/c
        !YuP[2021-03-02] changed lr_ to l_ (important for CQLP)
        enddo
      else ! v/vnorm units
        do i=1,iy_(l_)
        RTAB1(i)= coss(i,l_) ! v_par/vnorm
        RTAB2(i)= sinn(i,l_) ! v_perp/vnorm
        !YuP[2021-03-02] changed lr_ to l_ (important for CQLP)
        enddo
      endif
      CALL PGLINE(iy_(l_),RTAB1,RTAB2)
      CALL PGSCI(2) !red color
      do i=1,iy_(l_)  ![2022-02-10] Added dots at y-mesh points.
      CALL PGPT1(RTAB1(i),RTAB2(i),17) !Marker at each grid point.
      enddo
      CALL PGSCI(1) ! black color restore
      
      !plot v=vth line, for the k-th gen. species
c..................................................................
c     Note: the do loop below uses vth(),
c     vth is the thermal velocity =sqrt(T/m) (at t=0 defined in ainpla).
c     But, T==temp(k,lr) can be changed in profiles.f, 
c     in case of iprote (or iproti) equal to "prbola-t" or "spline-t"
c..................................................................
      if(cqlpmod .ne. "enabled")then
          vth_l= vth(k,lr_)
      else !(cqlpmod.eq."enabled") ! YuP[2021-02-26]
          vth_l= vthpar(k,ls_)
      endif
      if (pltlim.eq.'u/c' .or. pltlim.eq.'energy') then        
          do i=1,iy_(l_)
          RTAB1(i)= (vth_l/clight)*coss(i,l_) ! vth_par/c
          RTAB2(i)= (vth_l/clight)*sinn(i,l_) ! vth_perp/c
          !YuP[2021-03-02] changed lr_ to l_ (important for CQLP)
          enddo
      else ! v/vnorm units
          do i=1,iy_(l_)
          RTAB1(i)= (vth_l/vnorm)*coss(i,l_) ! vth_par/vnorm
          RTAB2(i)= (vth_l/vnorm)*sinn(i,l_) ! vth_perp/vnorm
          !YuP[2021-03-02] changed lr_ to l_ (important for CQLP)
          enddo
      endif
      ! Five different line styles are available:
      ! 1 (full line), 2 (dashed), 3 (dot-dash-dot-dash), 4 (dotted),
      CALL PGSLS(4) 
      CALL PGLINE(iy_(l_),RTAB1,RTAB2)
      CALL PGSLS(1) ! 1-> restore solid line
      CALL PGSLW(lnwidth) !lnwidth=3 line width in units of 0.005


      if (k.eq.0) return
      rr=rpcon(lr_) !rovera(lr_)*radmin  ! YuP[03-2016] changed to rpcon
      write(t_,150) n,timet
        CALL PGMTXT('B',R46,R40,R40,t_)
      write(t_,151) rovera(lr_),rr
        CALL PGMTXT('B',R47,R40,R40,t_)
      if(cqlpmod.ne."enabled")then
        write(t_,153) rya(lr_), rpcon(lr_), lr_
      else ! (cqlpmod.eq."enabled")
        write(t_,154) rya(lr_), l_ !YuP[2021-03-02] added, for CQLP
      endif
        CALL PGMTXT('B',R48,R40,R40,t_)
 150  format("time step n=",i5,5x,"time=",1pe10.2," secs")
 151  format( "r/a=",1pe10.3,5x,"radial position (R)=",1pe12.4," cm")
 153  format( "rya=",1pe10.3,5x,"R=rpcon=",1pe12.4," cm,  Surf#",i4)
 154  format( "rya=",1pe10.3,5x, "l_(index along B)=",i5)
 999  return
      end
C
C
      subroutine PGFUNC1(VISBLE,yplt,xplt,zplt)
      !Used in subr. pltcont and pltstrml
      INTEGER VISBLE
      REAL*4 xplt,yplt,zplt
      pointer wx,wy
      REAL*4 wx(:),wy(:)
      REAL*4 XWORLD,YWORLD
C
C 
      INCLUDE 'param.h'
      include 'comm.h'

      COMMON /PGLOCAL1/wx,wy,IIY,JXQ

C     wx IS V-NORM ARRAY, wy IS THETA ARRAY.  TYPE REAL.
C     xplt (yplt) IS FRACTIONAL INDEX IN V-NORM (THETA) ARRAY.
C     THIS SUBROUTINE MOVES PEN TO NORMALIZED V_PARALLEL,V_PERP COORDS.

      IIX=INT(xplt)
      IF (IIX.GE.1 .AND. IIX.LT.jx) THEN
         XX=wx(IIX) + (xplt-IIX)*(wx(IIX+1)-wx(IIX))
      ELSEIF (IIX.LE.1) THEN
         XX=wx(1)
      ELSE
         XX=wx(jx)
      ENDIF

      IIY=INT(yplt)
      IF (IIY.GE.1 .AND. IIY.LT.iymax) THEN
         YY=wy(IIY) + (yplt-IIY)*(wy(IIY+1)-wy(IIY))
      ELSEIF (IIY.LE.1) THEN
         YY=wy(1)
      ELSE
         YY=wy(iymax)
      ENDIF

      XWORLD=XX*COS(YY)
      YWORLD=XX*SIN(YY)

C      write(*,*) 'visble,x,y,z,xworld,yworld',visble,x,y,z,xworld,yworld

      IF (VISBLE.EQ.0) THEN
         CALL PGMOVE(XWORLD,YWORLD)
      ELSE
         CALL PGDRAW(XWORLD,YWORLD)
      ENDIF

      RETURN
      END



      
c====================================================================
c====================================================================
      SUBROUTINE PALETT(TYPE, CONTRA, BRIGHT)
C-----------------------------------------------------------------------
C Set a "palette" of colors in the range of color indices used by
C PGIMAG.
c From pgdemo4.f
C-----------------------------------------------------------------------
      INTEGER TYPE
      REAL*4 CONTRA, BRIGHT
C
      REAL*4 GL(2), GR(2), GG(2), GB(2)
      REAL*4 RL(9), RR(9), RG(9), RB(9)
      REAL*4 HL(5), HR(5), HG(5), HB(5)
      REAL*4 WL(10), WR(10), WG(10), WB(10)
      REAL*4 AL(20), AR(20), AG(20), AB(20)
C
      DATA GL /0.0, 1.0/
      DATA GR /0.0, 1.0/
      DATA GG /0.0, 1.0/
      DATA GB /0.0, 1.0/
C
      DATA RL /-0.5, 0.0, 0.17, 0.33, 0.50, 0.67, 0.83, 1.0, 1.7/
      DATA RR / 0.0, 0.0,  0.0,  0.0,  0.6,  1.0,  1.0, 1.0, 1.0/
      DATA RG / 0.0, 0.0,  0.0,  1.0,  1.0,  1.0,  0.6, 0.0, 1.0/
      DATA RB / 0.0, 0.3,  0.8,  1.0,  0.3,  0.0,  0.0, 0.0, 1.0/
C
      DATA HL /0.0, 0.2, 0.4, 0.6, 1.0/
      DATA HR /0.0, 0.5, 1.0, 1.0, 1.0/
      DATA HG /0.0, 0.0, 0.5, 1.0, 1.0/
      DATA HB /0.0, 0.0, 0.0, 0.3, 1.0/
C
      DATA WL /0.0, 0.5, 0.5, 0.7, 0.7, 0.85, 0.85, 0.95, 0.95, 1.0/
      DATA WR /0.0, 1.0, 0.0, 0.0, 0.3,  0.8,  0.3,  1.0,  1.0, 1.0/
      DATA WG /0.0, 0.5, 0.4, 1.0, 0.0,  0.0,  0.2,  0.7,  1.0, 1.0/
      DATA WB /0.0, 0.0, 0.0, 0.0, 0.4,  1.0,  0.0,  0.0, 0.95, 1.0/
C
      DATA AL /0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5,
     :         0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0/
      DATA AR /0.0, 0.0, 0.3, 0.3, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0,
     :         0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0/
      DATA AG /0.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8,
     :         0.6, 0.6, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.0, 0.0/
      DATA AB /0.0, 0.0, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.9, 0.9,
     :         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0/
C
      IF (TYPE.EQ.1) THEN
C        -- gray scale
         CALL PGCTAB(GL, GR, GG, GB, 2, CONTRA, BRIGHT)
      ELSE IF (TYPE.EQ.2) THEN
C        -- rainbow
         CALL PGCTAB(RL, RR, RG, RB, 9, CONTRA, BRIGHT)
      ELSE IF (TYPE.EQ.3) THEN
C        -- heat
         CALL PGCTAB(HL, HR, HG, HB, 5, CONTRA, BRIGHT)
      ELSE IF (TYPE.EQ.4) THEN
C        -- weird IRAF
         CALL PGCTAB(WL, WR, WG, WB, 10, CONTRA, BRIGHT)
      ELSE IF (TYPE.EQ.5) THEN
C        -- AIPS
         CALL PGCTAB(AL, AR, AG, AB, 20, CONTRA, BRIGHT)
      END IF
      END
