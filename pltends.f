c
c
      subroutine pltends
      implicit integer (i-n), real*8 (a-h,o-z)
c
c     Plot energy, density, parallel current and density conservation
c     constant vs. time., at a given s, distance along the magnetic field.
c     A cqlp.eq.'enabled' routine.
c
c     Modified from Graflib to pgplot calls by Yuri Petrov, 090727,
c     using PGPLOT + GRAFLIBtoPGPLOT.f routines (put in pltmain.f).
c
      include 'param.h'
      include 'comm.h'

      REAL*4 RILIN !-> For PGPLOT (text output positioning)
      REAL*4 :: R40=0.,R4MP2=-.2,R4P5=.5
      REAL*4 :: R4P2=.2,R4P8=.8,R4P6=.6,R4P9=.9 !.2,.8,.6,.9
      REAL*4 :: R41=1.,R42=2.,R43=3.,R44=4.,R45=5.,R46=6.,R41P8=1.8
      !YuP[2019-10-28] see PGSVP(R4P2,R4P8,R4P6,R4P9) and similar calls below
      
c     Conversion to real*4 function for PGPLOT
      REAL*4 RBOUND
      REAL*4 RJXA1(jx),RJXA2(jx),RJXA3(jx)
      REAL*4 :: R4P3=.3,R4P65=.65,R4P95=.95
      REAL*4 RPG1,RPG2


      dimension wk_nch(nonch)
c
      if (noplots.eq."enabled1") return
      if (pltend.eq."disabled") return
      dgts=1.e-8
      rr=rpcon(lr_) !rovera(lr_)*radmin  ! YuP[03-2016] changed to rpcon
      do 220 k=1,ngen
        if (pltend.eq."notplts") then
          goto 100
        elseif (pltend.eq."last") then
          if (n.lt.nstop) goto 100
        endif
c...  
cmnt  Generate plot "endn"
c...  
        CALL PGPAGE    
        call aminmx(pdens(1,k,l_),1,nch(l_),1,emin,emax,kmin,kmax)
        if (abs(emin-emax).lt.emax*dgts) emax=emin+.001*abs(emin)
        if(emax.gt.0.) emax=emax*1.02 ! extend the upper limit        
        if(emin.gt.0.) emin=emin*0.98 ! reduce the lower limit        
        !write(*,*) 'pltends-1:',ptime(1,l_),ptime(nch(l_),l_)
        CALL PGSVP(R4P2,R4P8,R4P6,R4P9) !(.2,.8,.6,.9) !---> 1st subplot
        CALL PGSCH(R41) ! set character size; default is 1.
        call GSWD2D("linlin$",ptime(1,l_),ptime(nch(l_),l_),emin,emax)
        call GPCV2D(ptime(1,l_),pdens(1,k,l_),nch(l_)) ! density(time)
        write(t_,10120) k
10120   format("density(s) of species",i2)
        RILIN=0.
        CALL PGLAB('time (sec)',t_,' ')
        !CALL PGSCH(R4P8) ! set character size; default is 1.
        !CALL PGMTXT('LV',RILIN,R40,R40,t_) ! Left-Vertical

        call aminmx(pengy(1,k,l_),1,nch(l_),1,emin,emax,kmin,kmax)
        if (abs(emin-emax).lt.emax*dgts) emax=emin+.001*abs(emin)
        if(emax.gt.0.) emax=emax*1.02 ! extend the upper limit        
        if(emin.gt.0.) emin=emin*0.98 ! reduce the lower limit        
        !write(*,*) 'pltends-2:', ptime(1,l_),ptime(nch(l_),l_),emin,emax
        CALL PGSVP(R4P2,R4P8,R4P2,R4P5) !(.2,.8,.2,.5)!----> 2nd subplot
        CALL PGSCH(R41) ! set character size; default is 1.
        call GSWD2D("linlin$",ptime(1,l_),ptime(nch(l_),l_),emin,emax)
        call GPCV2D(ptime(1,l_),pengy(1,k,l_),nch(l_)) ! energy(time)
        write(t_,10110) k
10110   format("energy(s) of species",i2)
        CALL PGLAB('time (sec)',t_,' ')
        RILIN=0.
        !CALL PGSCH(R4P8) ! set character size; default is 1.
        !CALL PGMTXT('LV',RILIN,R40,R40,t_) ! Left-Vertical

        write(t_,10140) denpar(k,ls_),enrgypa(k,ls_)
10140   format("local density(s) (/cm**2) = ",1pe13.6,
     +    ";  energy(s) (kev) =",1pe13.6)
        RILIN=5.5
        CALL PGSCH(R4P8) ! set character size; default is 1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) ! Bottom
     
        write(t_,10150) n,timet,k
10150   format("time step (n) is",i5,5x,"time=",1pe13.6," secs",
     +         "   Species k=",i2)
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) ! Bottom

        write(t_,10152) rovera(lr_),rr
10152   format("r/a=",e14.6,5x,"radial position (R)=",e14.6," cm")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) ! Bottom

        write(t_,10153) sz(l_)
10153   format("parallel position (s) =",1pe14.6," cm")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) ! Bottom
        
c...  
cmnt  Generate plot "curr"
c...  
        CALL PGPAGE ! new page
        call aminmx(pcurr(1,k,l_),1,nch(l_),1,emin,emax,kmin,kmax)
        if (abs(emin-emax).lt.emax*dgts) emax=emin+.001*abs(emin)
        if(emax.gt.0.) emax=emax*1.05 ! extend the upper range
        !write(*,*) 'pltends-3:', ptime(1,l_),ptime(nch(l_),l_),emin,emax
        CALL PGSVP(R4P2,R4P8,R4P6,R4P9) !(.2,.8,.6,.9)
        CALL PGSCH(R41) ! set character size; default is 1.
        call GSWD2D("linlin$",ptime(1,l_),ptime(nch(l_),l_),emin,emax)
        call GPCV2D(ptime(1,l_),pcurr(1,k,l_),nch(l_)) ! current_dens(time)
        CALL PGLAB('time (sec)',' ',' ')
        curramp=currm(k,l_)/3.e9
        write(t_,10160) k,curramp
10160   format("Local in s current density",
     &    " of species ",i2," =",1pe14.5,
     &    " Amps/cm**2")
        RILIN=5.5
        CALL PGSCH(R4P8) ! set character size; default is 1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) ! Bottom

        write(t_,10032) l_,sz(l_)
10032   format("Index along B, l=",i4, 4x,  
     &       "Parallel position s=",1pe14.6,"cm")
        RILIN=RILIN+2.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
     

 100    continue

        if (pltend.eq."tplts") goto 220
        
        !YuP[2021-03] Revised, to be similar to plots in pltendn:
        CALL PGPAGE  !----- J(v) and CumIntegral[J(v)dv]
        jxq=jx
        call aminmx(currv(1,k,l_),1,jxq,1,fnmin,fnmax,kmin,kmax)
        if (abs(fnmin-fnmax).lt.fnmax*dgts) fnmax=fnmin+.001*abs(fnmin)

        CALL PGSVP(R4P2,R4P8,R4P6,R4P9) !(.2,.8,.6,.9)!----> 1st subplot
c       Convert from statAmps/cm**2 to Amps/cm**2, dividing by 3.e9
        DO J=1,JXQ
           RJXA1(J)=RBOUND(x(j))
           RJXA2(J)=RBOUND(currv(j,k,l_))/3.e9
        ENDDO
        RPG1=RBOUND(fnmin)/3.e9 
        RPG2=RBOUND(fnmax)/3.e9 
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN
           RPG2= RPG1+1.e-16
        ENDIF
        CALL PGSWIN(RJXA1(1),RJXA1(JXQ),RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGLINE(jxq,RJXA1,RJXA2)
        CALL PGLAB('u/unorm','Par Curr Den: j(u/unorm)',' ')

        !YuP[2021-03-23] Added cumulative integration here, locally.
        !Note that currvs() does not have l_ index, 
        !so it should be recomputed for each given l_, from currv() :
        currvs(1,k)=dx(1)*currv(1,k,l_)
        do j=2,jx
           currvs(j,k)=currvs(j-1,k)+currv(j,k,l_)*dx(j)
        enddo
        call aminmx(currvs(1,k),1,jxq,1,fnmin,fnmax,kmin,kmax)
        if (abs(fnmin-fnmax).lt.fnmax*dgts) fnmax=fnmin+.001*abs(fnmin)
        CALL PGSVP(R4P2,R4P8,R4P2,R4P5)
c       Convert from statAmps/cm**2 to Amps/cm**2, dividing by 3.e9
        DO J=1,JXQ
           RJXA1(J)=RBOUND(x(j))
           RJXA2(J)=RBOUND(currvs(j,k))/3.e9 
        ENDDO
        RPG1=RBOUND(fnmin)/3.e9 
        RPG2=RBOUND(fnmax)/3.e9 
        CALL PGSWIN(RJXA1(1),RJXA1(JXQ),RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGLINE(jxq,RJXA1,RJXA2)
        CALL PGLAB('u/unorm','Int_0,u[j(u/unorm)]',' ')

        write(t_,10183) k,RJXA2(jx) !currvs(jx,k)/3.e9 
10183   format("Species:",i2,"  Current= ",1pe11.4," Amps/cm\u2\d")
        RILIN=5.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        
        if(cqlpmod.eq."enabled")then !YuP[2021-03-03] added for CQLP:
        write(t_,10032) l_,sz(l_)
!10032   format("Index along B, l=",i4, 4x,  
!     &       "Parallel position s=",1pe14.6,"cm") !defined above
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        endif

 220  continue ! k=1,ngen
 
      CALL PGSCH(R41) ! restore to default font size      
c
c     Plot the density conservation diagnostic vs time
c
      if (pltend.eq."notplts") then
        if (n.lt.nstop) return
      elseif (pltend.eq."last") then
        if (n.lt.nstop) return
      endif
c...  
cmnt  Generate plot "consn(l_)"
c...  
      CALL PGPAGE ! new page
      call aminmx(consnp(1,l_),1,nch(l_),1,emin,emax,kmin,kmax)
      !write(*,*) 'pltends-6:', ptime(1,l_),ptime(nch(l_),l_),emin,emax
      CALL PGSVP(R4P2,R4P8,R4P6,R4P9) !(.2,.8,.6,.9)
      CALL PGSCH(R41) ! set character size; default is 1.
      call GSWD2D("linlin$",ptime(1,l_),ptime(nch(l_),l_),emin,emax)
      call GPCV2D(ptime(1,l_),consnp(1,l_),nch(l_))      
      CALL PGLAB('time (sec)','consn(l_) conservation diag',' ')
      
      write(t_,10250) consn(l_)
10250 format("consn(l_)=",1pe12.4)
      RILIN=5.
      CALL PGSCH(R4P8) ! set character size; default is 1.
      CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
        
      write(t_,10251) 
10251 format("Perfect conservation should yield  machine accuracy,")
      RILIN=RILIN+1.
      CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
        
      write(t_,10252) 
10252 format("or about 1.e-14:")
      RILIN=RILIN+1.
      CALL PGMTXT('B',RILIN,R4MP2,R40,t_)

      write(t_,10150) n,timet
      RILIN=RILIN+1.
      CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
        
      write(t_,10152) rovera(lr_),rr
      RILIN=RILIN+1.
      CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
        
      write(t_,10153) sz(l_)
      RILIN=RILIN+1.
      CALL PGMTXT('B',RILIN,R40,R40,t_) 
      
      
      do 280 k=1,ngen
        CALL PGPAGE ! new page(s)
c$$$    Possibly write t_ greater than present dimension character*512:
c$$$        write(t_,10260) k,(sgaint(i,k,l_),i=1,8)
c$$$        call gptx2d(t_)
c$$$10260   format("total gain (+) or loss (-) to date for species",i3,";",
c$$$     &    "(in particles/cm**2 - units of line density)",";",
c$$$     &    "forcing nonnegative f x-sweep (implct=disabled)",e12.5,";",
c$$$     &    "forcing nonnegative f y-sweep (implct=disabled)",e12.5,";",
c$$$     &    "due to particle source=",e12.5,";","due to runaway=",e12.5,
c$$$     +    ";",
c$$$     &    "due to lossmode(k)=",e12.5,";","due to torloss(k)=",e12.5,
c$$$     +    ";",
c$$$     &    "due to fusion losses=",e12.5,";",
c$$$     &    "forcing nonnegative distribution (implct=enabled)=",e12.4,
c$$$     +    ";","$")
 280  continue
        write(t_,10150) n,timet
        RILIN=0.
        CALL PGSCH(R4P8) ! set character size; default is 1.
        CALL PGMTXT('T',RILIN,R40,R40,t_) ! Top

        write(t_,10152) rovera(lr_),rr
        RILIN=RILIN-1.
        CALL PGMTXT('T',RILIN,R40,R40,t_) ! Top

        write(t_,10153) sz(l_)
        RILIN=RILIN-1.
        CALL PGMTXT('T',RILIN,R40,R40,t_) ! Top

        CALL PGSCH(R41) ! restore to default font size      
      return
      end subroutine pltends
