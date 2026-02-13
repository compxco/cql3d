c
c
      subroutine pltendn
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c
c     Plot energy, density, electric field, parallel current,
c     and density conservation constant vs. time.
c
      include 'param.h'
      include 'comm.h'
c     Conversion to real*4 function for PGPLOT
      REAL*4 RBOUND
c     PGPLOT REAL*4 Variables:
      REAL*4 RILIN
      REAL*4 RPG1,RPG2
      REAL*4 RNONCHA1(nonch),RNONCHA2(nonch)
      REAL*4 RJXA1(jx),RJXA2(jx),RJXA3(jx)

      REAL*4 :: R40=0.,R4P2=.2,R4P65=.65,R4P95=.95,R4MP2=-0.2
      REAL*4 :: R4P3=.3,R4P5=.5,R4P6=.6,R4P8=.8,R4P9=.9
      REAL*4 :: R41=1.,R42=2.,R43=3.,R44=4.,R45=5.,R46=6.,R41P8=1.8

      dimension wk_nch(nonch)
      real*8 wkd(jx) 
      CHARACTER*64 TX_

cCH090220      data em33/1.d-35/
      data em33/1.d-33/
c      write(*,*)'pltendn:  HERE0'

c
      if (noplots.eq."enabled1") return
      if (pltend.eq."disabled") return
      dgts=1.e-8
      rr=rpcon(lr_) !rovera(lr_)*radmin  ! YuP[03-2016] changed to rpcon

      do 220 k=1,ngen
        if (l_ .ne. lmdpln_) go to 100
        if (pltend.eq."notplts") then
          goto 100
        elseif (pltend.eq."last") then
          if (n.lt.nstop) goto 100
        endif
c...  
cmnt  Generate plot "endn"
c...  
10150   format("time step (n) is",i5,5x,"time=",1pe12.4," secs") !,"   Species k=",i2)
10151   format("r/a=",1pe12.4,5x,"radial position (R) =",1pe12.4," cm")
c$$$c...  
cmnt  Generate plot "elecfld" and "curr"
c...  
        CALL PGPAGE

cBH070419
c        write(*,*)'pltendn:  HERE1'
c        write(*,*)'pltendn: nch(l_),pefld(1:nch(l_),1)=',
c     +                      nch(l_),pefld(1:nch(l_),1)

        call aminmx(pefld(1,l_),1,nch(l_),1,emin,emax,kmin,kmax)
        if (abs(emin-emax).lt.abs(emax)*dgts) emax=emin+.001*abs(emin)
        if(emax.gt.0.) emax=emax*1.05 ! extend the upper range
        CALL PGSVP(R4P2,R4P8,R4P65,R4P95)

        DO I=1,NCH(L_)
           RNONCHA1(I)=RBOUND(ptime(i,l_))
           RNONCHA2(I)=RBOUND(pefld(i,l_))
        ENDDO
        RPG1=RBOUND(emin)
        RPG2=RBOUND(emax)
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN
           RPG2= RPG1+1.e-16
        ENDIF
c        write(*,*)'Elec Fld: RNONCHA1(1),RNONCHA1(NCH(L_)),RPG1,RPG2',
c     +             RNONCHA1(1),RNONCHA1(NCH(L_)),RPG1,RPG2   
        CALL PGSWIN(RNONCHA1(1),RNONCHA1(NCH(L_)),RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)
        CALL PGLAB(' ','Elec Fld (V/cm)',' ')


        call aminmx(pcurr(1,k,l_),1,nch(l_),1,emin,emax,kmin,kmax)
        if (abs(emin-emax).lt.emax*dgts) emax=emin+.001*abs(emin)
        if(emax.gt.0.) emax=emax*1.05 ! extend the upper range
        DO I=1,NCH(L_)
           RNONCHA1(I)=RBOUND(ptime(i,l_))
           RNONCHA2(I)=RBOUND(pcurr(i,k,l_))
        ENDDO
        RPG1=RBOUND(emin)
        RPG2=RBOUND(emax)
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN
           RPG2= RPG1+1.e-16
        ENDIF
        CALL PGSVP(R4P2,R4P8,R4P3,R4P6)
c        write(*,*)'Curr Den: RNONCHA1(1),RNONCHA1(NCH(L_)),RPG1,RPG2',
c     +             RNONCHA1(1),RNONCHA1(NCH(L_)),RPG1,RPG2   
        CALL PGSWIN(RNONCHA1(1),RNONCHA1(NCH(L_)),RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGLAB('Time (sec)','Curr Den (A/cm^2)',' ')
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)

C       Have to fix following up to plot a mark:
C        CALL PGLINE(nch(l_),ptime(1,l_),currv_(k,lr_))
C        CALL PGSLS(3)
C        CALL PGLINE(nch(l_),ptime(1,l_),currr(k,lr_))
C        CALL PGSLS(1)     ! Reset line style back to full.


        curramp=pcurr(nch(l_),k,l_)
        RILIN=5.
        if (efswtchn.eq."neo_hh") then
           write(t_,10164) elecfld(lr_)
           RILIN=RILIN+1.
           CALL PGMTXT('B',RILIN,R40,R40,t_)
           write(t_,10165) k,curramp
           RILIN=RILIN+1.
           CALL PGMTXT('B',RILIN,R40,R40,t_)
           write(t_,10166) 
           RILIN=RILIN+1.
           CALL PGMTXT('B',RILIN,R40,R40,t_)
        else
           write(t_,10160) elecfld(lr_)
           RILIN=RILIN+1.
           CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
           write(t_,10161) k,curramp
           RILIN=RILIN+1.
           CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
        endif

10160   format("Electric field =",1pe12.4," (V/cm)")
10161   format("FSA current den of species",i2,
     &       " =",1pe12.4," Amps/cm**2")
10164   format("Electric field =",1pe12.4,"(V/cm)")
10165   format("FSA current den of species",i2,
     +       " =",1pe12.4,"Amps/cm**2")
10166   format("(efswtchn=neo_hh)")


        if(cqlpmod.ne."enabled")then !YuP[2021-03-30] CQL3D/CQLP treatment
          fnu0=2.0/tauee(lr_) !for printout in plots
        else !cqlpmod.eq."enabled"
          fnu0=2.0/tauee(ls_) !for printout in plots
        endif
        if (entr(k,3,l_).ge. 1.e-20 .and. kelecg.ne.0) then
          if (lrzmax.gt.1.and.n.gt.0) then
            areaovol=darea(lr_)/dvol(lr_)
          else
            if (eqmod.eq."enabled") then
              areaovol=1./twopi*onovrp(1,lr_)
            else
              areaovol=1./twopi/radmaj
            endif
          endif
          cdeffncy=curr(k,lr_)/entr(k,3,l_)/3.e9*areaovol
          !fnu0=2.0/tauee(lr_)
          xj=curr(k,lr_)/reden(k,lr_)/charge/vth(kelec,lr_)
          xp=entr(k,3,l_)/reden(k,lr_)/vth(kelec,lr_)**2/fmass(k)/fnu0
     +      *1.e+7
          xe=xj/xp

          fnuc=8.*pi*reden(k,lr_)*charge**4*gama(kelec,kelec)
     +      /(fmass(kelec)**2*clight**3)
          xc=curr(k,lr_)/(entr(k,3,l_)*1.e7)
     +        /(charge/(fnuc*fmass(kelec)*clight))


          write(t_,10170) cdeffncy
          RILIN=RILIN+2.
          CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
10170     format("Current drive efficiency j/(2*pi*R*prf) =",
     +         1pe12.4,' A/W')

          if (k .eq. kelecg) then
             write(t_,10171) xj
             RILIN=RILIN+1.
             CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
10171        format("Electron current (units ne*q*vth(kelec,lr_)) = ",
     +            1pe12.4)
             write(t_,10172) xp
             RILIN=RILIN+1.
             CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
10172        format("power (units: ne*vth(kelec,lr_)**2*me*nu0) =",
     +            1pe12.4)
             write(t_,10173) xe
             RILIN=RILIN+1.
             CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
10173        format("efficiency (j/p) (Fisch 1978 units) = ",
     +            1pe12.4)
             write(t_,10174) xc
             RILIN=RILIN+1.
             CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
10174        format("efficiency (j/p) (e/(m*c*nu_c units) = ",
     +            1pe12.4)
          endif
       endif
c..................................................................
c     Note: the printout below uses vth(),
c     vth is the thermal velocity =sqrt(T/m) (at t=0 defined in ainpla).
c     But, T==temp(k,lr) can be changed in profiles.f, 
c     in case of iprote (or iproti) equal to "prbola-t" or "spline-t"
c..................................................................
       if (k .eq. kelecg) then
          write(t_,10175) vth(kelec,lr_)
10175     format("vth(kelec,lr_) = sqrt(T/m) = ",1pe12.4," cm/sec")
          RILIN=RILIN+1.
          CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
          write(t_,10176) fnu0
10176     format("nu0 = ",1pe12.4," Hz")
          RILIN=RILIN+1.
          CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
       endif

 100   continue
       if (pltend.eq."tplts") goto 220
c...  
cmnt  Generate plot "currv"
c...  

c       If pltlim .ne. "disabled", plot versus
c       'x', 'u/c', or 'energy', up to maximum pltlimm.

        if (tandem.eq."enabled" .and. fmass(k).gt.1.e-27) then
           jxq=jlwr
           xmaxq=xlwr
           iyjxq=iy*jlwr
           tx_='u/vnorm'        
           do j=1,jxq
              tam1(j)=x(j)
           enddo
        elseif (pltlim.eq.'x') then
           jxq=min(luf(pltlimm,x,jx),jx)
           xmaxq=x(jxq)
           tx_='u/vnorm'        
           do j=1,jxq
              tam1(j)=x(j)
           enddo
        elseif (pltlim.eq.'u/c') then
           jxq=min(luf(pltlimm,uoc,jx),jx)
           xmaxq=uoc(jxq)
           tx_='u/c'        
           do j=1,jxq
              tam1(j)=uoc(j)
           enddo
        elseif (pltlim.eq.'energy') then
           pltlimmm=pltlimm
           wkd(1:jx)=enerkev(1:jx,k)
           jxq=min(luf(pltlimmm,wkd,jx),jx)
           xmaxq=enerkev(jxq,k) !YuP[2018-01-08] added 2nd index (k)
           tx_='Energy (keV)'        
           do j=1,jxq
              tam1(j)=enerkev(j,k) !YuP[2018-01-08] added 2nd index (k)
           enddo
        else ! pltlim='disabled' - whole range in x
           jxq=jx
           xmaxq=xmax
           tx_='u/vnorm'        
           do j=1,jx
              tam1(j)=x(j)
           enddo
        endif

        CALL PGPAGE  !----- J(v) and CumIntegral[J(v)dv]

        call aminmx(currv(1,k,l_),1,jxq,1,fnmin,fnmax,kmin,kmax)
        if (abs(fnmin-fnmax).lt.fnmax*dgts) fnmax=fnmin+.001*abs(fnmin)

        CALL PGSVP(R4P2,R4P8,R4P6,R4P9)
c       Convert from statAmps/cm**2 to Amps/cm**2, dividing by 3.e9
        DO J=1,JXQ
           RJXA1(J)=RBOUND(tam1(j))
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
        CALL PGLAB(tx_,'Par Curr Den: j(u/unorm)',' ')

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
           RJXA1(J)=RBOUND(tam1(j))
           RJXA2(J)=RBOUND(currvs(j,k))/3.e9 
        ENDDO
        RPG1=RBOUND(fnmin)/3.e9 
        RPG2=RBOUND(fnmax)/3.e9 
        CALL PGSWIN(RJXA1(1),RJXA1(JXQ),RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGLINE(jxq,RJXA1,RJXA2)
        CALL PGLAB(tx_,'Int_0,u[j(u/unorm)]',' ')

        write(t_,10183) k,currvs(jx,k)/3.e9 
10183   format("Species:",i2,"  Current=",1pe11.4," Amps/cm\u2\d")
        RILIN=5.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        
        if(cqlpmod.eq."enabled")then !YuP[2021-03-03] added for CQLP:
        write(t_,10032) l_,sz(l_)
10032   format("Index along B, l=",i4, 4x,  
     &       "Parallel position s=",1pe14.6,"cm")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        endif



c       if (pltlim.eq.'u/c') then
c          write(t_,10184) 
c       elseif (pltlim.eq.'energy') then
c          write(t_,10185) 
c       else
c          write(t_,10186) 
c       endif
10184  format("u/c")
10185  format("energy (keV)")
10186  format("u/unorm")
c       CALL PGLAB(t_,' ',' ')


       if (nrf .eq. 0.and. urfmod.eq."disabled"
     +     .and. rdcmod.eq."disabled") go to 220
       if (colmodl.eq.4 .and. k.eq.ngen) goto 220
c...  
cmnt  Generate plot "pwrrf"
c...  
        CALL PGPAGE
        call aminmx(pwrrf(1,k,l_),1,jxq,1,fnmin,fnmax,kmin,kmax)
        if (abs(fnmin-fnmax).lt.fnmax*dgts) fnmax=fnmin+.001*abs(fnmin)
        CALL PGSVP(R4P2,R4P8,R4P6,R4P9)
        DO J=1,JXQ
           RJXA1(J)=RBOUND(tam1(j))
           RJXA2(J)=RBOUND(pwrrf(j,k,l_))
        ENDDO
        RPG1=RBOUND(fnmin)
        RPG2=RBOUND(fnmax)
        if (abs(RPG1-RPG2).le.(2.1*em33)) RPG2=RPG1+10.*em33
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN
           RPG2= RPG1+1.e-16
        ENDIF
        CALL PGSWIN(RJXA1(1),RJXA1(JXQ),RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGLINE(jxq,RJXA1,RJXA2)
        CALL PGLAB(' ','RF Pwr Den: p(u/unorm)',' ')


        call aminmx(pwrrfs(1,k,l_),1,jxq,1,fnmin,fnmax,kmin,kmax)
        if (abs(fnmin-fnmax).lt.fnmax*dgts) fnmax=fnmin+.001*abs(fnmin)
        CALL PGSVP(R4P2,R4P8,R4P2,R4P5)
        DO J=1,JXQ
           RJXA1(J)=RBOUND(tam1(j))
           RJXA2(J)=RBOUND(pwrrfs(j,k,l_))
        ENDDO
        RPG1=RBOUND(fnmin)
        RPG2=RBOUND(fnmax)
        !if (abs(RPG1-RPG2).le.(2.1*em33)) RPG2=RPG1+10.*em33
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN
           RPG2= RPG1+1.e-16
        ENDIF
        CALL PGSWIN(RJXA1(1),RJXA1(JXQ),RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGLINE(jxq,RJXA1,RJXA2)
        CALL PGLAB(' ','Int_0,u[p(u/unorm)]',' ')
        write(t_,10193) k,pwrrfs(jx,k,l_)

        RILIN=6.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
10193   format("Species:",i2,"  Power =",1pe11.4," Watts/cc")

       if (pltlim.eq.'u/c') then
          write(t_,10184) 
       elseif (pltlim.eq.'energy') then
          write(t_,10185) 
       else
          write(t_,10186) 
       endif
       CALL PGLAB(t_,' ',' ')

 220  continue ! k=1,ngen
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

      CALL PGPAGE
        do i=1,nch(l_)
          wk_nch(i)=consnp(i,l_)
        enddo
        call aminmx(wk_nch,1,nch(l_),1,emin,emax,kmin,kmax)
        CALL PGSVP(R4P2,R4P8,R4P5,R4P9)
        DO I=1,NCH(L_)
           RNONCHA1(I)=RBOUND(ptime(i,l_))
           RNONCHA2(I)=RBOUND(consnp(i,l_))
        ENDDO
        RPG1=RBOUND(emin)
        RPG2=RBOUND(emax)
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN
           RPG2= RPG1+1.e-16
        ENDIF
c        write(*,*)'consn(l_): RNONCHA1(1),RNONCHA1(nch(l_)),RPG1,RPG2',
c     +                        RNONCHA1(1),RNONCHA1(nch(l_)),RPG1,RPG2   
        CALL PGSWIN(RNONCHA1(1),RNONCHA1(nch(l_)),RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)
        CALL PGLAB('time (sec)','consn(l_) conservation diag',' ')

      write(t_,10250) consn(l_)
        RILIN=5.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
      write(t_,10251) 
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
      write(t_,10252) 
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)

10250 format("consn(l_)=",1pe12.4)
10251 format("Perfect conservation should yield  machine accuracy,")
10252 format("or about 1.e-14:")

        write(t_,10150) n,timet !,k ! consnp does not dep. on k
        RILIN=RILIN+2.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
        write(t_,10151) rovera(lr_),rr
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)


      return
      end subroutine pltendn






C======================================================================
C======================================================================

      subroutine plt_fow_cons ! developed for FOW, but can also be used for ZOW
      implicit integer (i-n), real*8 (a-h,o-z)
      save
      ! Plot the change in Total N of ptcles as a func of time.
      ! Calculations are done in sub. fow_cons()
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE
c     Conversion to real*4 function for PGPLOT
      REAL*4 RBOUND
c     PGPLOT REAL*4 Variables:
      REAL*4 RILIN
      REAL*4 RPG1,RPG2, RPGmin, RPGmax
      REAL*4 RLRZAP1(0:LRZA) ! for rho coord.
      REAL*4 RLRZAP(0:LRZA)  ! for R coord.
      REAL*4 RLRZAP11(0:LRZA) ! vertical axis: conservation of ptcl
      REAL*4 :: R41P44=1.44,R41P8=1.8,R4P5=.5,R40=0.,R41=1.
      REAL*4 :: R4P2=.2,R4P6=.6,R4P8=.8,R4P9=.9
      
      REAL*4 RNONCHA1(nonch),RNONCHA2(nonch)
      real*8 wk_cons(nonch)

      character*16 t_horiz

      data em33/1.d-33/

CMPIINSERT_IF_RANK_NE_0_RETURN
 ! make plots on mpirank.eq.0 only

      if (noplots.eq."enabled1") return
      if (pltend.eq."disabled") return
      if (lrzmax .le. 2) return

c..................................................................
c     Determine the x-axis for plots (psi or rho - both normalized).
c..................................................................

      if (pltvs.eq."psi") then
        do 20 l=1,lrzmax
          tr(l)=(equilpsi(0)-equilpsi(l))/equilpsi(0)
 20     continue
        write(t_horiz,'(a3)') 'psi'
      else
        do 30 l=1,lrzmax
          tr(l)=rz(l)/radmin
 30     continue
        write(t_horiz,'(a6,a8,a1)') 'rho (=', radcoord, ')'
      endif

      DO ir=1,LRZMAX
         RLRZAP(ir)=rpcon(ir) ! outermost-midplane R of flux surf.
         RLRZAP1(ir)=tr(ir)   ! rho
         !RLRZAP11(ir)= consnp(nonch,ir)
      ENDDO

      ! If the horizontal coord is rho, set the limits to [0.,1.]
      RPGmin=RLRZAP1(1)
      RPGmax=RLRZAP1(LRZMAX)
      if(RPGmin.le.0.2) RPGmin=0. ! Lower limit in plots: extend to 0.
      if(RPGmax.ge.0.8 .and. RPGmax.lt.1.) RPGmax=1. ! Upper limit: extend to 1.


      CALL PGPAGE
      
        CALL PGSAVE !-1
        ! FIRST(lower) SUBPLOT: Conservation vs. t, for different rho
        CALL PGSVP(R4P2,R4P8,R4P2,R4P5)
        !Find min/max over all time steps and all radial points
        emin= 1.d16
        emax=-1.d16
        do ll=1,lrz
        do itime=1,nch(ll)
           emin=min(emin,consnp(itime,ll))
           emax=max(emax,consnp(itime,ll))
        enddo
        enddo
        if(emax.gt.0.) emax=emax*1.05 ! give extra 5% in plot limit
        RPG1=RBOUND(emin)
        RPG2=RBOUND(emax)
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN
           RPG2= RPG1+1.e-16
        ENDIF
        CALL PGSWIN(RBOUND(ptime(1,1)),RBOUND(ptime(n+1,1)),RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGSLS(1)  ! 1-> solid; 2-> dashed; 3-> -.-.- ;
        do ll=1,lrz
          do itime=1,n+1 !!YuP was: nonch
           RNONCHA1(itime)=RBOUND(ptime(itime,1))
           RNONCHA2(itime)=RBOUND(consnp(itime,ll))
          ENDDO
          CALL PGLINE(n+1,RNONCHA1,RNONCHA2)
        enddo
        CALL PGSCH(R41)
        CALL PGLAB(' ',
     +       'consn(t) at different rho',' ')
        ! Label for horizontal axis (better control of position):
        CALL PGSCH(R41P44)
        CALL PGMTXT('B',R41P8,R4P5,R4P5, 'time (sec)') 
!        write(t_,10250) conserv_nptcl(k,n+1)
!10250   format("ratio(t_end)=",1pe12.4)
!        CALL PGMTXT('B',3.,R40,R40,t_)
        CALL PGUNSA !-1


        CALL PGSAVE !-2
        ! SECOND(upper) SUBPLOT: Conservation vs. rho, at different t
        CALL PGSVP(R4P2,R4P8,R4P6,R4P9)
        !Find min/max over all time steps and all radial points
        emin= 1.d16
        emax=-1.d16
        do ll=1,lrz
        do itime=1,nch(ll)
           emin=min(emin,consnp(itime,ll))
           emax=max(emax,consnp(itime,ll))
        enddo
        enddo
        if(emax.gt.0.) emax=emax*1.05 ! give extra 5% in plot limit
        RPG1=RBOUND(emin)
        RPG2=RBOUND(emax)
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN
           RPG2= RPG1+1.e-16
        ENDIF
        CALL PGSWIN(RPGmin,RPGmax,RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGSLS(1)  ! 1-> solid; 2-> dashed; 3-> -.-.- ;
        DO it=1,n+1 !nonch ! Plot all curves (for all time steps)
           !RNONCHA1(it)=RBOUND(ptime(it,1))
cBH171231           RLRZAP11(1:LRZMAX)= consnp(it,1:LRZMAX)
            RLRZAP11(1:LRZ)= consnp(it,1:LRZ)
cBH171231           CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP11(1)) 
           CALL PGLINE(lrz,RLRZAP1(1),RLRZAP11(1)) 
        ENDDO   
        CALL PGSCH(R41)
        CALL PGLAB(' ',
     +  '(dentot-xlndn0-sgaint1)/(.5*(xlndn0+dentot))',' ')
        ! Label for horizontal axis (better control of position):
        CALL PGSCH(R41P44)
        CALL PGMTXT('B',R41P8,R4P5,R4P5,t_horiz) 
!        write(t_,3002) k,kspeci(1,k),kspeci(2,k),n
! 3002   format("species no. ",i2,4x,a8,2x,a8,2x," time step n=",i5)  
        CALL PGSCH(R41)
!        CALL PGMTXT('T',1.,R40,R40,t_)
        CALL PGUNSA !-2


      return
      end subroutine plt_fow_cons
