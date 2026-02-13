c
c
      subroutine pltpower
      implicit integer (i-n), real*8 (a-h,o-z)
c
c     This routine plots the power due to the various physical mechanism
c     vs. time
c
      include 'param.h'
      include 'comm.h'
      REAL*4 RILIN
      REAL*4 RPG1,RPG2
      REAL*4 RNONCHA1(nonch),RNONCHA2(nonch)
      REAL*4 :: R40=0.,R4P2=.2,R4P8=.8,R4P4=.4,R4P95=.95

c
      if (noplots.eq."enabled1") return

      do 100 k=1,ngen
        if (colmodl.eq.4 .and. k.eq.ngen) goto 100
        CALL PGPAGE
        emin=ep90
        emax=-em90
        do 192 lu=-1,11
          call aminmx(pentr(1,k,lu,l_),1,nonch,1,hmin,hmax,idmin,idmax)
          if (hmin .lt. emin) emin=hmin
          if (hmax .gt. emax) emax=hmax
c     do 191 nc=1,nonch
c     h=pentr(nc,k,lu,l_)
c     if (h .lt. emin) emin=h
c     if (h .gt. emax) emax=h
c191  continue
 192    continue
        emax=emax+.05*(abs(emax))
        emin=emin-.05*(abs(emin))
        CALL PGSVP(R4P2,R4P8,R4P4,R4P95)
        DO I=1,NCH(L_)
           RNONCHA1(I)=ptime(i,l_)
        ENDDO
        RPG1=emin
        RPG2=emax
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN
           RPG2= RPG1+1.e-16
        ENDIF
        CALL PGSWIN(RNONCHA1(1),RNONCHA1(NCH(L_)),RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGLAB('Time (sec)','Power Den (W/cm\u3\d)',' ')
        ! Note: pentr(nch(l_),k,is,l_)=entr(k,is,l_)

        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,4,l_) !total - sum of all power sources
        ENDDO
        CALL PGSLS(1) ! 1-> solid
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)
        
        CALL PGSCI(4) !blue color
        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,9,l_) !Power computed from df/dt. 
        ENDDO                          !Should equal entr(k,4,l_).
        CALL PGSLS(2) ! 2-> dashed
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)
        CALL PGSCI(1) !restore black color

        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,3,l_) !RF - radio frequency heating
        ENDDO
        CALL PGSCI(2) !red color for RF
        CALL PGSLS(3) ! 3-> -.-.- 
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)
        CALL PGSCI(1) !restore black color

        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,2,l_) !ohmic - electric field term
        ENDDO
        CALL PGSLS(4) ! 4-> .....
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)

        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,1,l_) !collisional transfer from gens.
        ENDDO
        CALL PGSLS(1) ! 1-> solid
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)

        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,5,l_) !particle sources
        ENDDO
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)

        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,6,l_) !loss-lossmode
        ENDDO
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)

        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,7,l_) !losses-torloss
        ENDDO
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)

        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,-1,l_) !collisional transfer from Maxw. elec.
        ENDDO
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)

        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,0,l_) !collisional transfer from Maxw. ions
        ENDDO
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)

        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,8,l_) !losses due to runaway
        ENDDO
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)

        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,10,l_) !setting neg f to zero
        ENDDO
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)
        
        if (k .ne. kelecg .or. syncrad .eq. "disabled") go to 300
        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,11,l_) !synchrotron rad losses
        ENDDO
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)        
 300    continue
 
        if(tauegy(k,lr_).gt.0.)  then
        DO I=1,NCH(L_)
           RNONCHA2(I)=pentr(i,k,12,l_) !phenomenological energy losses
        ENDDO
        CALL PGLINE(nch(l_),RNONCHA1,RNONCHA2)        
        endif
        
c$$$ 200    format("COMPONENT BY COMPONENT POWER FLOW: SPECIES",i2,";",
c$$$     1    "ohmic--- electric field term",";",
c$$$     2    "col-mxe--- coulomb coll. with Maxwellian elec.",";",
c$$$     2    "col-mxi--- coulomb coll. with Maxwellian ions",";",
c$$$     3    "RF--- radio frequency heating",";",
c$$$     4    "total--- sum of all power sources ",";",
c        9     df/dt
c$$$     5    "fusion--- fusion power output",";",
c$$$     6    "loss-o--- particle losses due to lossmode(k)",";",
c$$$     1    "loss-t--- losses due to torloss(k)",";",
c$$$     1    "runaway--- losses over high velocity terminator",";",
c$$$     1    "synchrd---synchrotron radiation losses",";",
c$$$     1    "los-egy---phenomenological energy loss","$")

        RILIN=4.
        write(t_,210) k ! Gen. species number
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        
        CALL PGSCI(4) !blue color
        write(t_,211) entr(k,4,l_),entr(k,9,l_) !sum over all components
                                        !and from df/dt (should be same)
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        CALL PGSCI(1) !restore black color
        
        write(t_,212) entr(k,-1,l_) !collisional transfer from Maxwellian elec.
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        write(t_,213) entr(k,0,l_)  !collisional transfer from Maxwellian ions
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        write(t_,214) entr(k,1,l_)  !collisional transfer from gens.
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        write(t_,215) entr(k,2,l_)  !ohmic drive
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
          CALL PGSCI(2) !red color
          write(t_,216) entr(k,3,l_)  !RF drive
          RILIN=RILIN+1.
          CALL PGMTXT('B',RILIN,R40,R40,t_)
          CALL PGSCI(1) !restore black color
        write(t_,217) entr(k,5,l_)  !particle sources
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        write(t_,218) entr(k,6,l_),entr(k,7,l_) !loss-lossmode;losses-torloss
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        write(t_,219) entr(k,8,l_)  !losses due to runaway
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        write(t_,220) entr(k,10,l_) !setting neg f to zero
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_) 
        write(t_,221) entr(k,11,l_) !synchrotron rad losses
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        write(t_,222) entr(k,12,l_) !phenomenological energy losses
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        
        if(cqlpmod.eq."enabled")then !YuP[2021-03-03] added for CQLP:
        write(t_,10032) l_,sz(l_)
10032   format("Index along B, l=",i4, 4x,  
     &       "Parallel position s=",1pe14.6,"cm")
        RILIN=RILIN+1.
        CALL PGMTXT('B',RILIN,R40,R40,t_)
        endif
        
 210    format("Species k=",i2, "    Final powers in Watts/cc are:")
 211    format("sum over all comp=",1pe10.2, 3x,
     +         "From df/dt(--)",1pe10.2)
 212    format("collisional transfer from Maxwellian elec.=",1pe10.2)
 213    format("collisional transfer from Maxwellian ions=",1pe10.2)
 214    format("collisional transfer from gens.=",1pe10.2)
 215    format("ohmic drive=",1pe10.2)
 216    format("RF drive (-.-) =",1pe10.2)
 217    format("particle sources=",1pe10.2)
 218    format("loss-lossmode(k)=", 1pe10.2, 3x,
     +         "losses-torloss(k)=",1pe10.2)
 219    format("losses due to runaway=",1pe10.2)
 220    format("setting neg f to zero=",1pe10.2)
 221    format("synchrotron rad losses=",1pe10.2)
 222    format("phenomenological energy losses=",1pe10.2)

 100  continue ! k=1,ngen
 
      return
      end
