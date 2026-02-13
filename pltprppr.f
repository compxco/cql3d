c
c
      subroutine pltprppr
      implicit integer (i-n), real*8 (a-h,o-z)
c...  
cmnt  This routine plots the parallel distribution function.
c...
c
c     Modified from Graflib to pgplot calls by Yuri Petrov, 090727,
c     using PGPLOT + GRAFLIBtoPGPLOT.f routines (put in pltmain.f).
c
c     Need to re-work this a bit, for plot of pitch angle averaged
c     reduced distributions (knockon=.ne."disabled" case).
c  
      include 'param.h'
      include 'comm.h'

      REAL*4 RILIN !-> For PGPLOT (text output positioning)
      REAL*4 :: R40=0.,R41=1.
      REAL*4 :: R4P2=.2,R4P8=.8,R4P25=.25,R4P95=.95 !.2,.8,.25,.95
      !YuP[2019-10-28]
      
      character*8 target
      if (noplots.eq."enabled1") return

c     Return, if using a theta average distribution
c     (which may conflict with call fle_fsa or fle_pol, below).
      if (flemodel.ne."pol"  .and. flemodel.ne."fsa") return

      do 20 k=1,ngen
        if (tandem.eq."enabled" .and. k.eq.kionn) then
          xll=-xlwr
          xlu=xlwr
          xpl=0.
          xpu=xlwr
          target="ionmesh"
        else
          xll=-xmax
          xlu=xmax
          xpl=0.
          xpu=xmax
          if (xprpmax.ne.1.) xpu=xprpmax
          target="mainmesh"
        endif
c       Obtain equatorial plane pitch-avg'd distribution for ko model
c       case.  (Else setup interfere's with subroutine sourceko).
!        if ((knockon.ne."disabled").and.(n.ge.nonko).and.(n.le.noffko)
!     &                                                          ) then
            !YuP[2020-12-22] Added (n.ge.nonko).and.(n.le.noffko)
            !In cqlinput, knockon could be set to 'enabled' (or other),
            !but effectively it is disabled because of large 
            !value of nonko=10000 (example).
c           call fle("setup",0)
c           call fle("calc",1)
!            write(*,*)'pltprppr: Not ready for knockon=enabled'
        if (lrz.eq.1) then
           call fle_pol("setup",0) ! lp=0 is of no importance during setup
           if(cqlpmod.ne."enabled")then !YuP[2021-03-08] added if()
             call fle_pol("calc",1) ! lp=1 means midplane position
           else !(cqlpmod.eq."enabled")  !YuP[2021-03-08] added for CQLP
             call fle_pol("calc",l_) ! lp=l_ means position along B line (CQLP)
           endif
        else
           call fle_fsa("setup")
           call fle_fsa("calc")
        endif

c     plot (fll(+xpar)-fll(-xpar)), and xpar*(fll(+xpar)-fll(-xpar))
        jpxyh=(jfl+1)/2
        jpxyhm=jpxyh-1
        tem1(1)=0.0
        tem2(1)=0.0
        do 30  jp=1,jpxyhm
          ![2022-03-19] Now fl() is computed&saved for each l_ 
          tem1(jp)=fl(jpxyh+jp-1,l_)-fl(jpxyh-jp,l_) ! fll(+xpar)-fll(-xpar)
          tem2(jp)=xlm(jpxyh+jp-1)*tem1(jp) ! xpar*(fll(+xpar)-fll(-xpar))
!           write(*,'(a,i4,2e11.3)')'lr_, xpar, fll(+xpar)-fll(-xpar)=', 
!     &          lr_,xlm(jpxyh+jp-1),tem1(jp)
 30     continue
        call aminmx(tem1,1,jpxyhm,1,fmin1,fmax1,kmin,kmax)
        call aminmx(tem2,1,jpxyhm,1,fmin2,fmax2,kmin,kmax)
        fmin=min(fmin1,fmin2)
        fmax=max(fmax1,fmax2)
        
        CALL PGPAGE !--------------------------------------- new page(s)
        CALL PGSVP(R4P2,R4P8,R4P25,R4P95) !(.2,.8,.25,.95)!YuP[2019-10-28]
        ! (XLEFT, XRIGHT, YBOT, YTOP)
        CALL PGSCH(R41) ! set character size; default is 1.
        call GSWD2D("linlin$",0.d0,xlu,fmin,fmax)
        call GPCV2D(xlm(jpxyh),tem1,jpxyhm) !       fll(+xpar)-fll(-xpar)
        call GPCV2D(xlm(jpxyh),tem2,jpxyhm) ! xpar*(fll(+xpar)-fll(-xpar))
        
        write(t_,10011) lr_,k
10011 format("asymmetric cmpt of f_par, and xpar*cmpt, lr,species:",
     &              1x,i3,',',i3)
        RILIN=5.
        CALL PGSCH(R4P8) ! set character size; default is 1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        
        write(t_,10012)
10012 format("(f_par normed so int(-1,+1)=equatorial ne)")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        
        if(cqlpmod.eq."enabled")then !YuP[2021-03-03] added for CQLP:
        write(t_,10032) l_,sz(l_)
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        endif
       
        CALL PGPAGE !--------------------------------------- new page(s)
        ![2022-03-19] Now fl() is computed&saved for each l_ :
        call aminmx(fl(1,l_),1,2*jpxyhm,1,fmin,fmax,kmin,kmax)
        fmin=1.d-08*fmax
        CALL PGSVP(R4P2,R4P8,R4P25,R4P95) !(.2,.8,.25,.95)!YuP[2019-10-28]
        CALL PGSCH(R41) ! set character size; default is 1.
        call GSWD2D("linlog$",xll,xlu,fmin,fmax)
        do 10 jj=1,2*jpxyhm
          ![2022-03-19] Now fl() is computed&saved for each l_ :
          if (fl(jj,l_) .lt. fmin ) fl(jj,l_)=fmin
 10     continue
        call GPCV2D(xlm(1:2*jpxyhm),fl(1:2*jpxyhm,l_),2*jpxyhm)
        !YuP[2022-03-19] added explicit range for xlm();
        !Note that xlm array is allocated as xlm(0:jfl),
        !and fl is allocated as fl(0:jfl,1:lrors)
        
        write(t_,10013) k
10013 format("parallel distribution function for species:",1x,i5)
        RILIN=3.
        CALL PGSCH(R4P8) ! set character size; default is 1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        
        write(t_,10014) 
10014 format("(normed so int(-1,+1)=equatorial ne)")        
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        
        write(t_,10020)
10020 format( "(log plot)")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        
        rr=rpcon(lr_) !rovera(lr_)*radmin  ! YuP[03-2016] changed to rpcon
        write(t_,10030) n,timet
10030 format("time step (n) is",i5,5x,"time=",e14.6," secs")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)

        write(t_,10031) rovera(lr_),rr
10031 format("r/a=",e14.6,5x,"radial position (R) =",e14.6," cm")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        
        if(cqlpmod.eq."enabled")then !YuP[2021-03-03] added for CQLP:
        write(t_,10032) l_,sz(l_)
10032 format("Index along B, l=",i4, 4x,  
     &       "Parallel position s=",1pe14.6," cm")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        endif
        
        CALL PGSCH(R41) ! recover default 1.0 fontsize
        
 20   continue

      return
      end
