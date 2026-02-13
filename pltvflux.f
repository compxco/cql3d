c
c
      subroutine pltvflux
      implicit integer (i-n), real*8 (a-h,o-z)
c
c     Plot the fraction of the total particle density that is fluxing
c     up in velocity vs. v.  Mark the v=vth(k,lr_) position.
c..................................................................
c     vth is the thermal velocity =sqrt(T/m) (at t=0 defined in ainpla).
c     But, T==temp(k,lr) can be changed in profiles.f, 
c     in case of iprote (or iproti) equal to "prbola-t" or "spline-t"
c..................................................................
      include 'param.h'
      include 'comm.h'

      REAL*4 RTAM1(jx),RTAM2(jx), vth_mark
      REAL*4 REMAX,REMIN,RILIN, RXMAXQ
      REAL*4 :: R4P2=.2,R4P8=.8,R4P5=.5,R4P9=.9
      REAL*4 :: R40=0.,R41=1.,R41P44=1.44,R4MP2=-.2

      real*8 wkd(jx) 
      character*64 tx_
c
      if (noplots.eq."enabled1") return

      do 100 k=1,ngen !-----------------------------------------

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
         if(cqlpmod .ne. "enabled")then
           vth_l= vth(k,lr_)
         else !(cqlpmod.eq."enabled") ! YuP[2021-02-26]
           vth_l= vthpar(k,ls_)
         endif

         if (pltlim.eq."disabled") then
            jxq=jx
            xmaxq=x(jxq)
            tx_='u/vnorm'        
            vth_mark=vth_l/vnorm
            do j=1,jxq-1
               TAM1(j)=0.5*(x(j)+x(j+1))
            enddo
         endif
         if (tandem.eq."enabled" .and. fmass(k).gt.1.e-27) then
            jxq=jlwr
            xmaxq=xlwr
            tx_='u/vnorm'        
            vth_mark=vth_l/vnorm
            do j=1,jxq-1
               TAM1(j)=0.5*(x(j)+x(j+1))
            enddo
         ! If pltlim .ne. "disabled", plot versus
         ! 'x', 'u/c', or 'energy', up to maximum pltlimm.
         elseif (pltlim.eq.'x') then
            jxq=min(luf(pltlimm,x,jx),jx)
            xmaxq=x(jxq)
            tx_='u/vnorm'        
            vth_mark=vth_l/vnorm
            do j=1,jxq-1
               TAM1(j)=0.5*(x(j)+x(j+1))
            enddo
         elseif (pltlim.eq.'u/c') then
            jxq=min(luf(pltlimm,uoc,jx),jx)
            xmaxq=uoc(jxq)
            TX_='u/c\dlight\u'
            vth_mark=vth_l/clight
            do j=1,jxq-1
               TAM1(j)=0.5*(uoc(j)+uoc(j+1))
            enddo
         elseif (pltlim.eq.'energy') then
            pltlimmm=pltlimm
            wkd(1:jx)=enerkev(1:jx,k)
            jxq=min(luf(pltlimmm,wkd,jx),jx)
            xmaxq=enerkev(jxq,k) !YuP[2018-01-08] added 2nd index (k)
            do j=1,jxq-1
               TAM1(j)=0.5*(enerkev(j,k)+enerkev(j+1,k)) 
            enddo
            TX_='Energy (keV)'
            vth_mark=fmass(k)*vth_l**2/ergtkev ! in keV units
         endif
         RXMAXQ=XMAXQ
      
        do 183 j=1,jxq-1 !YuP: use jxq-1 (was jx)
          !cYuP tam1(j)=(x(j)+x(j+1))*.5*vnorm/vth(kelec,lr_) ! v/Vth
                   !YuP: or should it be vth(k,lr_) ?
          tam2(j)=vflux(j,k,l_)
 183    continue
        call aminmx(tam2,1,jxq-1,1,emin,emax,kmin,kmax) !YuP: use jxq-1 (was jx)

        DO J=1,jxq-1 !YuP: use jxq-1 (was jx)
           RTAM1(J)=TAM1(J) !
           RTAM2(J)=TAM2(J)
        ENDDO
        REMIN=EMIN
        REMAX=EMAX
        TEST=ABS((EMAX-EMIN)/EMAX)
        !write(*,*) 'pltvflux: n,emax,emin,test ',n,emax,emin,test
        !write(*,*)'RTAM1=',RTAM1
        !write(*,*)'RTAM2=',RTAM2
        IF (TEST .lt. 0.0001) THEN
           REMIN=REMAX-0.0001*ABS(REMAX)
        ENDIF
        
        CALL PGPAGE
        CALL PGSVP(R4P2,R4P8,R4P5,R4P9)
        IF ( Remax-Remin .le. 1.e-16 ) THEN ! YuP [02-23-2016]
           Remax= Remin+1.e-16
        ENDIF
        CALL PGSWIN(R40,RXMAXQ,Remin,Remax) !YuP: use jxq-1 (was jx)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGSAVE
        CALL PGSCH(R41P44)
        CALL PGLAB(tx_, ' (\gt\dei\u/n) (dn/dt)', 
     +       'Normalized v-flux (\gt\dei\u/n)(dn/dt)')
        CALL PGUNSA
        CALL PGLINE(jxq-1,RTAM1,RTAM2) !!YuP: use jxq-1 (was jx)
        
        !YuP: add line v=vth(k,lr) 
        RTAM1(1)=vth_mark ! in different units, dep. on pltlim
        RTAM1(2)=vth_mark
        RTAM2(1)=Remin
        RTAM2(2)=Remax
        CALL PGSLS(2) ! 2-> dashed
        CALL PGLINE(2,RTAM1,RTAM2) ! vertical line 
        CALL PGSLS(1) ! 1-> solid (restore)


        RILIN=7.5
        CALL PGMTXT('B',RILIN,R4MP2,R40,'Dashed line: u = Vthermal')
        
        write(t_,186) k,lr_,n
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
        
        if(cqlpmod.eq."enabled")then !YuP[2021-03-03] added for CQLP:
        write(t_,10032) l_,sz(l_)
10032   format("Index along B, l=",i4, 4x,  
     &       "Parallel position s=",1pe14.6," cm")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
        endif
        
        write(t_,1861) 
        RILIN=RILIN+2.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
        write(t_,185) tam1(1),tam2(1),tam1(2),tam2(2)
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
        do 184 j=jxq*2/3,jxq,16 !YuP: use jxq-1 (was jx)
          if (j+11 .gt. jxq) go to 184 !YuP: use jxq-1 (was jx)
          write(t_,185) tam1(j),tam2(j),tam1(j+8),tam2(j+8)
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
 184    continue
        write(t_,185) tam1(jxq-2),tam2(jxq-2),tam1(jxq-1),tam2(jxq-1)
        !YuP: use jxq-1 (was jx)

        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)

 185    format(2(1pe12.4,2x,1pe12.4,5x))

 186    format("Species k = ",i4,"     Surf.#",i5,"     Time step:",i9)
 1861   format(2(2x,"v/vth",9x,"normalized v-flux",2x))
 
 100  continue ! k species -------------------------------------
 
      CALL PGSCH(R41) ! restore character size; default is 1.

      return
      end
