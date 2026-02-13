
c
c
      subroutine pltfluxs
      implicit integer (i-n), real*8 (a-h,o-z)

c...................................................................
c     Plots velocity (momentum-per-mass)  fluxes versus u, 
c     for selected values of theta.
c     Do combined fluxes, and individual fluxes, per pltflux1 vector.
c...................................................................
c
c     Modified from Graflib to pgplot calls by Yuri Petrov, 090727,
c     using PGPLOT + GRAFLIBtoPGPLOT.f routines (put in pltmain.f).
c

      save
      include 'param.h'
      include 'comm.h'

      include 'advnce.h'

      REAL*4 RILIN !-> For PGPLOT (text output positioning)
      REAL*4 :: R41=1.,R40=0.,R4P2=.2,R4P3=.3,R4P8=.8,R4P9=.9
      !YuP[2019-10-28] see PGSVP(R4P2,R4P8,R4P3,R4P9) below

      real*8 wkd(jx) 
      CHARACTER*64 TX_
      
c     pltflux1(1)=1. ==> sum of fluxes
c              2         collisions
c              3         parallel electric field
c              4         rf
c              5         synchrotron
c              6         Bremssstrahlung (+ phenomenological energy loss)
c              7         Krook operator slowing down

c...................................................................
c     Outer loop is over the various plots, as enabled by pltflux1
c...................................................................

      do 500 kk=1,7

      if (pltflux1(kk).ne.1.) go to 500

      do 400 k=1,ngen

c     Do velocity and theta fluxes separately:
      do 410 kkk=1,2
c
        call bcast(da,zero,iyjxp1)
        call bcast(db,zero,iyjxp1)
        call bcast(dc,zero,iyjxp1)
        call bcast(dd,zero,iyp1jx)
        call bcast(de,zero,iyp1jx)
        call bcast(df,zero,iyp1jx)

        go to (10,20,30,40,50,60,70),kk
 10     call coefstup(k)
        write(t_,910) k
        go to 100
 20     call coeffpad(k)
        write(t_,920) k
        go to 100
 30     if (elecfld(lr_)  .lt. 1.e-09) go to 500
        call coefefad(k)
        write(t_,930) k
        go to 100
 40     continue
        if (urfmod.eq."disabled" .and. vlfmod.eq."disabled"
     +       .and. vlhmod.eq."disabled") go to 500
        xrf=0.
        if (n .lt. nonrf(k) .or. n .ge. noffrf(k)) go to 41

        call coefrfad(k,xrf)
        if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
          write(*,'(a,2i6,e12.2)')'pltfluxs->coefrfad: n,lr_,sum(dbb)=',
     +    n,lr_,sum(dbb)
        endif

 41     continue
        if (xrf.gt.0.) then
          write(t_,940) k
        endif
        go to 100

 50     continue
        if (syncrad.eq."disabled") go to 500
        call coefsyad(k)
        write(t_,950) k
        go to 100

 60     continue
        if (bremsrad.eq."disabled"
     +       .and. torloss(k).eq."disabled") go to 500
        call coefegad(k)
        write(t_,960) k
        go to 100

 70     call coefload(k)
        write(t_,970) k



 100    continue

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
c     first the velocity flux..
c...................................................................

        call coefmidv(da,1)
        call coefmidv(db,2)
        call coefmidv(dc,3)

c...................................................................
c     the theta flux..
c...................................................................

        call coefmidt(dd,1)
        call coefmidt(de,2)
        call coefmidt(df,3)


c...................................................................
c     Fluxes
c...................................................................


        if (kkk.eq.1) then
        do 200 j=1,jx
           do 199 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
              !Note that for meshy="fixed_mu" iy_(l_) can be less than iy
              temp1(i,j)=gfi(i,j,k)
 199       continue
 200    continue
        else
         do 202 j=1,jx
           do 201 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
              !Note that for meshy="fixed_mu" iy_(l_) can be less than iy
              temp1(i,j)=hfi(i,j)
 201       continue
 202    continue
        endif
          


c...  
cmnt  Plot fluxes as a function of velocity for various
cmnt  angles.
c...  
      iii=(itl+iyh)/2
      trpmd=y(iii,l_)
      do 210 i=1,iyh
        if (y(i,l_) .ge. trpmd) goto 220
 210   continue
 220   continue
      midtrp=i
      imsh(1)=1
      imsh(2)=itl
      imsh(3)=midtrp
      imsh(4)=iyh
      imsh(5)=iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
      !Note that for meshy="fixed_mu" iy_(l_) can be less than iy

c        if (tandem.eq."enabled" .and. fmass(k).gt.1.e-27) then
c          jxq=jlwr
c          xmaxq=xlwr
c          iyjxq=iy*jlwr
c        else
c          jxq=jx
c          xmaxq=xmax
c          iyjxq=iyjx
c        endif

        if (pltlim.eq."disabled") then
           jxq=jx
           xmaxq=x(jxq)
           !iyjxq=iyjx !YuP: Not used?
           tx_='u/vnorm'        
           do j=1,jxq
              tam1(j)=x(j)
           enddo
        endif

        if (tandem.eq."enabled" .and. fmass(k).gt.1.e-27) then
           jxq=jlwr
           xmaxq=xlwr
           !iyjxq=iy*jlwr !YuP: note used?
           pltlim='x' ! YuP: reset?
           pltlimm=xmaxq
           tx_='u/vnorm'        
           do j=1,jxq
              tam1(j)=x(j)
           enddo
c       If pltlim .ne. "disabled", plot versus
c       'x', 'u/c', or 'energy', up to maximum pltlimm.
        elseif (pltlim.eq.'x') then
           jxq=min(luf(pltlimm,x,jx),jx)
           xmaxq=x(jxq)
           !iyjxq=iy*jxq !YuP: not used?
           tx_='u/vnorm'        
           do j=1,jxq
              tam1(j)=x(j)
           enddo
        elseif (pltlim.eq.'u/c') then
           jxq=min(luf(pltlimm,uoc,jx),jx)
           xmaxq=uoc(jxq)
           !iyjxq=iy*jxq !YuP: not used?
           tx_='u/c'        
           do j=1,jxq
              tam1(j)=uoc(j)
           enddo
        elseif (pltlim.eq.'energy') then
           pltlimmm=pltlimm
           wkd(1:jx)=enerkev(1:jx,k)
           jxq=min(luf(pltlimmm,wkd,jx),jx)
           xmaxq=enerkev(jxq,k) !YuP[2018-01-08] added 2nd index (k)
           !iyjxq=iy*jxq !YuP: not used?
           tx_='Energy (keV)'        
           do j=1,jxq
              tam1(j)=enerkev(j,k) !YuP[2018-01-08] added 2nd index (k)
           enddo
        endif


        call bcast(tam4,zero,jxq)
        do 240 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
          !Note that for meshy="fixed_mu" iy_(l_) can be less than iy
          do 230 j=1,jxq
            tam4(j)=tam4(j)+temp1(i,j)*cynt2(i,l_)/twoint(l_)
 230       continue
 240     continue
        call aminmx(tam4,1,jxq,1,emin,emax,kmin,kmax)
        do 260 iu=1,5
          i=imsh(iu)
          do 250 j=1,jxq
            tam2(j)=temp1(i,j)
 250       continue
          call aminmx(tam2,1,jxq,1,fmin,fmax,kmin,kmax)
          if (fmin .lt. emin) emin=fmin
          if (fmax .gt. emax) emax=fmax
 260     continue
        emax=emax*1.03
        emin=emax/1.e+12

        CALL PGPAGE ! new page
        CALL PGSVP(R4P2,R4P8,R4P3,R4P9) !(.2,.8,.3,.9)
        CALL PGSCH(R41) ! set character size; default is 1.
        call GSWD2D("linlog$",tam1(1),tam1(jxq),emin,emax)
        !-YuP:   call GSCVLB(1)
        !-YuP:   call GSCVFT(0.)
        do 310 iu=1,6
          if (iu .eq. 6) then
            do 270 j=1,jxq
              tam2(j)=tam4(j)
 270         continue
            text(1)="avg"
            goto 290
          endif
          do 280 j=1,jxq
            tam2(j)=f(imsh(iu),j,k,l_)
 280       continue
          if (iu .eq. 1) text(1)="pll"
          if (iu .eq. 2) text(1)="trp/ps"
          if (iu .eq. 3) text(1)="midtrp"
          if (iu .eq. 4) text(1)="perp"
          if (iu .eq. 5) text(1)="pll-pi"
 290       continue
          !-YuP:   call GSCVTX(loc(text))
          do 300 j=1,jxq
            if(tam2(j).le.emin) tam2(j)=emin
 300      continue
          call GPCV2D(tam1,tam2,jxq)
          xu=float(iu)
          !-YuP:   call GSCVFT(xu/6.)
 310    continue
        !-YuP:   call GSCVLB(0)

       CALL PGLAB(tx_,' ',' ') ! YuP/added: horizontal axis label
        
c     Write previously set title
        RILIN=1.
        CALL PGSCH(R4P8) ! set character size; default is 1.
        CALL PGMTXT('T',RILIN,R40,R40,t_) ! Top
        
        rr=rpcon(lr_) !rovera(lr_)*radmin  ! YuP[03-2016] changed to rpcon
        write(t_,10020) k
10020 format("Flux vs. velocity for some angles; species number = ",i3)
        RILIN=3.
        CALL PGMTXT('B',RILIN,R40,R40,t_) ! Bottom
        
        write(t_,10010) n,timet
10010 format("time step (n) is",i5,5x,"time=",e14.6," secs")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)

        write(t_,10011) rovera(lr_),rr
10011 format("r/a=",e14.6,5x,"radial position (R) =",e14.6," cm")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        
        write(t_,10030) 
10030 format("pll    ---- theta = 0 radians")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) 

        write(t_,10031) 
10031 format("pll-pi ---- theta = pi radians")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) 

        write(t_,10032) y(itl,l_)
10032 format("trp/ps ---- theta = ",e13.5," radians")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) 

        write(t_,10033) trpmd
10033 format("midtrp ---- theta = ",e13.5," radians")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) 

        write(t_,10034) y(iyh,l_)
10034 format("perp   ---- theta = ",e13.5," radians")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) 

        write(t_,10035) 
10035 format("avg    ---- theta averaged over pi radians")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) 
        
        CALL PGSCH(R41) ! recover default 1.0 fontsize

 410    continue
 400    continue ! k species
 500  continue ! skipping handle
 
      CALL PGSCH(R41) ! recover default 1.0 fontsize

 910  format("species no.",i2,5x,"combined velocity space fluxes")
 920  format("species no.",i2,5x,"Fokker-Planck velocity space flux")
 930  format("species no.",i2,5x,"electric field velocity space flux")
 940  format("species no.",i2,5x,"RF velocity space flux")
 950  format("species no.",i2,5x,"synchrotron velocity space flux")
 960  format("species no.",i2,5x,"Brems+phenom velocity space flux")
 970  format("species no.",i2,5x,"Krook velocity space flux")

      return
      end
