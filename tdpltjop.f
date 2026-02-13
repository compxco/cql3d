
c
c
      subroutine tdpltjop
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      REAL*4 RILIN
      REAL*4 RPG1,RPG2, RPGmin, RPGmax
      REAL*4 RLRZAP1(0:LRZA),RLRZAP11(0:LRZA),RLRZAP12(0:LRZA),
     +     RLRZAP13(0:LRZA),RLRZAP14(0:LRZA)
      REAL*4 RLRZAP(0:LRZA)
      !YuP[2019-09] REAL*4 LNWIDTH ! It is integer, in name_decl.h

      REAL*4 :: R40=0.,R41P44=1.44,R42P5=2.5,R4P5=.5,R41P5=1.5,R41P2=1.2
      REAL*4 :: R41=1.,R42=2.,R43=3.,R44=4.,R45=5.,R46=6.,R41P8=1.8
      REAL*4 :: R47=7.,R48=8.,R45P5=5.5
      REAL*4 :: R4P2=.2,R4P8=.8,R4P6=.6,R4P9=.9
      REAL*4 :: R4P05=.05,R4P95=.95,R41P3=1.3 
      REAL*4 :: R4MP2=-0.2 

      character*16 t_horiz
      
      real*8 wk_lrz(lrza),
     +       sorpw_nbi_all(lrza), sorpw_nbii_all(lrza) ! local

c..................................................................
c     This routine plots a number of current drive diagnostics
c     for NB, LH and ECH excitation. The x axis of the plots are
c     rho/rhomax or psi/psimax depending on the value of bcd
c     word psival. Below "fi" means fast ions (usually in conjunction
c     with NBI. "fi+e" would include the electron contribution to
c     the fast ions, this would be either from Coulomb drag on the
c     electrons by the fast ions or RF excitation.
c..................................................................

CMPIINSERT_IF_RANK_NE_0_RETURN
 ! make plots on mpirank.eq.0 only

      if (noplots.eq."enabled1") return

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
          tr(l)=rya(l) !YuP: was rz(l)/radmin
          !write(*,*)'lr,radmin,rz(lr),rya(lr)=',l,radmin,rz(l),rya(l)
 30     continue
        write(t_horiz,'(a6,a8,a1)') 'rho (=', radcoord, ')'
      endif
      
      do l=1,lrzmax ! for plots of profiles vs R
          RLRZAP(l)=rpcon(l) ! R_outermost of flux surf.
      enddo


c..................................................................
c     plots of current..
c..................................................................
      ! Choose which j_bs component to plot: 
      ! thermal(=maxwellian) or non-thermal(=general species)
      if(kelecg.ne.0)then  
         kke=2 ! I_bs for e_general (non-thermal)
      else
         kke=1 ! I_bs for e_maxw
      endif
      if(niong.ne.0)then 
         kki=2 ! I_bs for i_general (non-thermal)
      else
         kki=1 ! I_bs for i_maxw
      endif
      if (jhirsh.eq.0) then
         ! In this case, j_bs is only calculated for maxwellian part.
         ! (In fact, it is done for electrons only)
         kke=1
         kki=1
      endif

      fmin=0.
      fmax=0.
      call aminmx(currtz(1),1,lrzmax,1,fmin,fmax,kmin,kmax)
      call aminmx(currtpz(1),1,lrzmax,1,gmin,gmax,kmin,kmax)
      if (gmin.lt.fmin) fmin=gmin
      if (fmax.lt.gmax) fmax=gmax
      call aminmx(bscurm(1,1,kke),1,lrzmax,1,gmin,gmax,kmin,kmax)
      if (gmin.lt.fmin) fmin=gmin
      if (fmax.lt.gmax) fmax=gmax
      call aminmx(bscurm(1,2,kki),1,lrzmax,1,gmin,gmax,kmin,kmax)
      if (gmin.lt.fmin) fmin=gmin
      if (fmax.lt.gmax) fmax=gmax
      if (fmax-fmin.lt.1.e-8) fmin=fmax-.1*abs(fmax)-1.e-5
      RPG1=min(fmin,0.) !Make lower limit 0 when fmin>0 
      RPG2=max(fmax,0.)
      
      DO I=1,LRZMAX
         RLRZAP1(I)=tr(i) ! rho or psi
         RLRZAP11(I)=currtz(i)
         RLRZAP12(I)=currtpz(i)
         RLRZAP13(I)=bscurm(i,1,kke) !bscurm(*,1,*) is for electrons
         RLRZAP14(I)=bscurm(i,2,kki) !bscurm(*,2,*) is for ions
      ENDDO
      
      ! If the horizontal coord is rho, set the limits to [0.,1.]
      RPGmin=RLRZAP1(1)
      RPGmax=RLRZAP1(LRZMAX)
      if(RPGmin.le.0.2) RPGmin=0. ! Lower limit in plots: extend to 0.
      if(RPGmax.ge.0.8 .and. RPGmax.lt.1.) RPGmax=1. ! Upper limit: extend to 1.
      
cyup      call aminmx(totcurz(1),1,lrzmax,1,gmin,gmax,kmin,kmax)
cyup      if (gmin.lt.fmin) fmin=gmin
cyup      if (fmax.lt.gmax) fmax=gmax
cyup [May-2014] Do not plot the total current anymore, 
cyup because in present setup, it may include
cyup a bootstrap analytical current (jhirsh88/99) for electrons.
cyup But if bootcalc='method1' or bootcalc='method2',
cyup totcurz() will also include a numerical bootstrap current 
cyup for a given general species. In such a case, totcurz() will 
cyup include the bootstrap current from both analytical model
cyup and numerical model; makes no sense.
cyup For now, make plots of current density profiles from:
cyup general ions (if any), general electrons (it will be in "fi+e").
cyup Also, plotted profiles of bootstrap current (jhirsh88/99) 
cyup separately for electrons and ions, based on maxwellian T,n profiles
cyup or, if available, for general electrons or ions;
cyup they are designated as "bs_e" and "bs_i" in the plots.
      
      write(t_,4040)
 4040 format("Current summed over all species",";",
     1  "fi - fast ion current",3x,
     1  "fi+e - fi + electrons"
     1  ,";","bs - Bootstrap current",3x,"tot - total current","$")


        CALL PGPAGE
        
        ! FIRST PANEL: profiles vs rho
        CALL PGSVP(R4P2,R4P8,R4P6,R4P9)
        CALL PGSAVE
        CALL PGSCH(R41P44)
        CALL PGMTXT('T',R42P5,R4P5,R4P5,'CURRENT (AMPS/CM\u2\d)')
        CALL PGUNSA
c..................................................................
c     currents, printed and plotted next
c..................................................................
 4012 format("fi [solid]=  ",1pe10.3,5x,"fi+e[--]= ",  1pe10.3)
 4013 format("bs_e[-.-]= ",1pe10.3,5x,"bs_i[.....]= ", 1pe10.3, " Amps")
        write(t_,4012) currtza,currtpza
        CALL PGMTXT('T',R42,R40,R40,t_)
        write(t_,4013) bscurma(1,kke),bscurma(2,kki) ! (e,*), (i,*)
        CALL PGMTXT('T',R41,R40,R40,t_)
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN ! YuP [02-23-2016]
           RPG2= RPG1+1.e-16
        ENDIF
        CALL PGSWIN(RPGmin,RPGmax,RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        ! fi (general ions):
        CALL PGSLS(1) !solid line
        CALL PGSCI(1) !black color
        CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP11(1))
        ! fi+e [can be general ions (if any) + general electrons, 
        ! or general ions + screening current from e; see eleccomp='enabled' option]:
        CALL PGSLS(2) ! ---
        CALL PGSCI(2) !red color
        CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP12(1))
        ! Bootstrap for e, based on jhirsh88/99 models:
        CALL PGSLS(3) ! -.-
        CALL PGSCI(3) !green color
        CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP13(1))
        ! Bootstrap for ions, based on jhirsh88/99 models:
        CALL PGSLS(4) ! ...
        CALL PGSCI(4) !blue color
        CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP14(1))
        CALL PGSLS(1) ! Restore solid line
        CALL PGSCI(1) !black color restored        
cyup        ! Total: [YuP: do not plot anymore]
cyup        DO I=1,LRZMAX
cyup           RLRZAP13(I)=totcurz(i)
cyup        ENDDO
cyup        CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP13(1))
        CALL PGSAVE
        CALL PGSCH(R41P44)
        CALL PGLAB(' ','curr density (A/cm\u2\d)',' ')
        CALL PGMTXT('B',R41P8,R4P5,R4P5,t_horiz)
        CALL PGUNSA

        ! SECOND PANEL: same profiles vs R
        CALL PGSVP(R4P2,R4P8,R4P2,R4P5)
        RPGmin=min(RLRZAP(1),rmag)
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN ! YuP [02-23-2016]
           RPG2= RPG1+1.e-16
        ENDIF
        CALL PGSWIN(RPGmin,RLRZAP(lrzmax),RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        ! fi (general ions):
        CALL PGSLS(1) ! solid
        CALL PGSCI(1) ! color
        CALL PGLINE(lrzmax,RLRZAP(1),RLRZAP11(1))
        ! fi+e [can be general ions (if any) + general electrons, 
        ! or general ions + screening current from e; see eleccomp='enabled' option]:
        CALL PGSLS(2) ! ---
        CALL PGSCI(2) !red color
        CALL PGLINE(lrzmax,RLRZAP(1),RLRZAP12(1))
        ! Bootstrap for e, based on jhirsh88/99 models:
        CALL PGSLS(3) ! -.-
        CALL PGSCI(3) !green color
        CALL PGLINE(lrzmax,RLRZAP(1),RLRZAP13(1))
        ! Bootstrap for ions, based on jhirsh88/99 models:
        CALL PGSLS(4) ! ...
        CALL PGSCI(4) !blue color
        CALL PGLINE(lrzmax,RLRZAP(1),RLRZAP14(1))
        CALL PGSLS(1) ! Restore solid line
        CALL PGSCI(1) !black color restored
        CALL PGSAVE
        CALL PGSCH(R41P44)
        CALL PGLAB(' ','curr density (A/cm\u2\d)',' ')
        CALL PGMTXT('B',R41P8,R4P5,R4P5,'R (=rpcon)  (cm)')
        CALL PGUNSA



c..................................................................
c     Partially integrated in rho (or psi)
c..................................................................

        CALL PGPAGE
        CALL PGSVP(R4P2,R4P8,R4P2,R4P6)

        CALL PGSAVE
        CALL PGSCH(R41P44)
        CALL PGMTXT('T',R48,R4P5,R4P5,
     +              'CURRENT (AMPS)')
        CALL PGMTXT('T',R47,R4P5,R4P5,
     +              '(INTEGRATED UP TO RHO or PSI)')
        CALL PGUNSA

c..................................................................
c     plots of partial integration of currents
c..................................................................


      fmin=0.
      fmax=0.

      call aminmx(currtzi(1),1,lrzmax,1,fmin,fmax,kmin,kmax)
      call aminmx(currtpzi(1),1,lrzmax,1,gmin,gmax,kmin,kmax)
      if (gmin.lt.fmin) fmin=gmin
      if (fmax.lt.gmax) fmax=gmax
cyup      call aminmx(totcurzi(1),1,lrzmax,1,gmin,gmax,kmin,kmax)
cyup      if (gmin.lt.fmin) fmin=gmin
cyup      if (fmax.lt.gmax) fmax=gmax
      if (fmax-fmin.lt.1.e-8) fmin=fmax-.1*abs(fmax)-1.e-5
      if(fmax.gt.0.) fmax=fmax*1.05 ! extend the upper range
      RPG1=fmin
      RPG2=fmax
      RPG1=min(fmin,0.) !Make lower limit 0 when fmin>0 

      DO I=1,LRZMAX
         RLRZAP1(I)=tr(i)
      ENDDO
      ! If the horizontal coord is rho, set the limits to [0.,1.]
      RPGmin=RLRZAP1(1)
      RPGmax=RLRZAP1(LRZMAX)
      if(RPGmin.le.0.2) RPGmin=0. ! Lower limit in plots: extend to 0.
      if(RPGmax.ge.0.8 .and. RPGmax.lt.1.) RPGmax=1. ! Upper limit: extend to 1.

        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN ! YuP [02-23-2016]
           RPG2= RPG1+1.e-16
        ENDIF
        CALL PGSWIN(RPGmin,RPGmax,RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)

        CALL PGSLS(4) ! dotted 
        CALL PGSLW(lnwidth)  ! thin line
        CALL PGSCI(4) ! blue color
        CALL PGMTXT('T',R46,R40,R40,
     &   'Blue/dotted: using currz(k,lr) over ionic general species') !YuP[2019-12-20]
        DO I=1,LRZMAX
           RLRZAP11(I)=currtzi(i) 
           !currtzi(ll)=currtzi(ll-1)+darea(ll)*currtz(ll)
           !where currtz(lr_)=currt(lr_)/3.e+9 
           ! is the sum over ionic general species.
        ENDDO
        CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP11(1))
        
        CALL PGSLS(2) !dashed --
        CALL PGSLW(lnwidth*2)  ! bold line
        CALL PGSCI(2) ! red color
        CALL PGMTXT('T',R45,R40,R40,
     &   'Red/dashed: using curr(k,lr) over all general species') !YuP[2019-12-20]
        DO I=1,LRZMAX
           RLRZAP12(I)=currtpzi(i)
           !currtpzi(ll)=currtpzi(ll-1)+darea(ll)*currtpz(ll)
           !where currtpz(lr_)=currtp(lr_)/3.e+9 
           ! is the sum over general species k of curr(k,lr_)
        ENDDO
        CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP12(1))
cyup        CALL PGSLS(3)
cyup        DO I=1,LRZMAX
cyup           RLRZAP13(I)=totcurzi(i)
cyup        ENDDO
cyup        CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP13(1))
        CALL PGSLS(1) ! restore solid line
        CALL PGSLW(lnwidth) ! restore
        CALL PGSCI(1) ! black color restored
        CALL PGSAVE
        CALL PGSCH(R41P44)
        CALL PGLAB(' ','current  (Amps)',' ')
        CALL PGMTXT('B',R41P8,R4P5,R4P5,t_horiz)
        CALL PGUNSA


c..................................................................
c     Source Power...
c..................................................................

C-----------------------------------------------------------------------
C     IF NO POWER NO PLOT
C-----------------------------------------------------------------------
C%OS  

cBH171014:  Need to check whether more than kfrsou(1) needed for
cBH171014:  more than 1 beamline species.
c     Adjust for kfrsou=0 (can occur when no NBI):
cyup      if (kfrsou(1).ne.0) then
cyup         kfrsou1=kfrsou(1)
cyup      else
cyup         kfrsou1=1
cyup      endif
cyup: kfrsou1 is not used anymore [2018-06-27],
cyup: Instead, plot/print values for each k_gen species
cyup: or print/plot values of the sum over k_gen, see below.
      
      !YuP[2018-06-27] Added, for plots: sum of sorpw_nbi over all k_gen
      sorpw_nbi_all= 0.d0 ! initialize
      sorpw_nbii_all=0.d0 ! initialize
      do l=1,lrzmax
      do k=1,ngen
         sorpw_nbi_all(l)= sorpw_nbi_all(l) +sorpw_nbi(k,l)
         sorpw_nbii_all(l)=sorpw_nbii_all(l)+sorpw_nbii(k,l)
         !YuP: might include KO source pwr, if e_general is present
      enddo
      enddo

      !write(*,*)'tdpltjop: sorpwtza=', sorpwtza, maxval(sorpwt)
      IF (abs(sorpwtza)+maxval(powurf).le.1.e-25) goto 809 !IF NO POWER NO PLOT
      !YuP[2022-05-10] Sometimes sorpwtza is negative 
      !(from instability in distr func)
      ! although powurf is positive. Added abs(), and also
      ! added maxval(powurf) for the if() check above.
C%OS  
      ! PRINT OUT OF SOURCE POWER (WATTS/CC): 
      ! sorpwt(l),sorpw_nbi(k,l),sorpw_rf(1,l),sorpw_rf(2,l),sorpw_rf(3,l)
      CALL PGPAGE  
      CALL PGSVP(R4P05,R4P95,R4P05,R4P95)
      RILIN=0.
      CALL PGSAVE
      CALL PGSCH(R41P44)
      CALL PGMTXT('T',-RILIN,R4P5,R4P5,
     +     'SOURCE POWER: (WATTS/CC)')
      RILIN=RILIN+3.
      CALL PGSCH(R41) !(1.)!YuP[2019-10-28]
      write(t_,6013) pltvs
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      RILIN=RILIN+2.
      write(t_,6014) pltvs
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      RILIN=RILIN+2.

 6013 format(4x,a8,"NBI(orKO)+RF  NBI(or KO)",
     +  "   RF(1)      RF(2)       RF(3)")
 6014 format(4x,a8,"   (sorpwt)   (sorpw_nbi)",
     +  "      (sorpw_rf for gen.species 1,2,3)")

c     Start printing results on first page
      do 10  l=1,min(40,lrzmax)
        if (ngen.eq.1) then
          !YuP[2018-06-27] print the sum of sorpw_nbi over all k_gen
          write(t_,6015) tr(l),sorpwt(l), sorpw_nbi_all(l),sorpw_rf(1,l)
        elseif (ngen.eq.2) then 
          !YuP[2018-06-27] print the sum of sorpw_nbi over all k_gen
          write(t_,6016) tr(l),sorpwt(l), sorpw_nbi_all(l),
     1      sorpw_rf(1,l),sorpw_rf(2,l)
        else ! ngen=3
          !YuP[2018-06-27] print the sum of sorpw_nbi over all k_gen
          write(t_,6017) tr(l),sorpwt(l), sorpw_nbi_all(l),
     1      sorpw_rf(1,l),sorpw_rf(2,l),sorpw_rf(3,l)
        endif
        CALL PGMTXT('T',-RILIN,R40,R40,t_)
        RILIN=RILIN+1.
 10   continue

c     Continue printing results on second page, if lrzmax.gt.40
      if (lrzmax.gt.40) then
         CALL PGPAGE
         CALL PGSVP(R4P05,R4P95,R4P05,R4P95)
         RILIN=0.+3.
         do l=41,lrzmax
           !YuP[2018-06-27] print the sum of sorpw_nbi over all k_gen
            if (ngen.eq.1) then
               write(t_,6015) tr(l),sorpwt(l),sorpw_nbi_all(l),
     1              sorpw_rf(1,l)
            elseif (ngen.eq.2) then
               write(t_,6016) tr(l),sorpwt(l),sorpw_nbi_all(l),
     1              sorpw_rf(1,l),sorpw_rf(2,l)
            else
               write(t_,6017) tr(l),sorpwt(l),sorpw_nbi_all(l),
     1              sorpw_rf(1,l),sorpw_rf(2,l),sorpw_rf(3,l)
            endif
            CALL PGMTXT('T',-RILIN,R40,R40,t_)
            RILIN=RILIN+1.
         enddo
      endif
      
 6015 format(1x, 1pe9.3, 3(1x,e10.3) )
 6016 format(1x, 1pe9.3, 4(1x,e10.3) )
 6017 format(1x, 1pe9.3, 5(1x,e10.3) )
 

      RILIN=RILIN+2.
      write(t_,6022) sorpwtza !sorpwtza=sorpwti(lrzmax) !NBI+RF, all gen.species
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      RILIN=RILIN+1.
      write(t_,6023) sorpw_nbii_all(lrzmax) ! NBI only (sum over k_gen)
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      RILIN=RILIN+1.
      if (ngen.ge.1) then
         write(t_,6024) sorpw_rfi(1,lrzmax) ! RF(1st gen.species) only
         CALL PGMTXT('T',-RILIN,R40,R40,t_)
         RILIN=RILIN+1.
      endif
      if (ngen.ge.2) then
         write(t_,6025) sorpw_rfi(2,lrzmax) ! RF(2nd gen.species) only
         CALL PGMTXT('T',-RILIN,R40,R40,t_)
         RILIN=RILIN+1.
      endif
      if (ngen.ge.3) then
         write(t_,6026) sorpw_rfi(3,lrzmax) ! RF(3rd gen.species) only
         CALL PGMTXT('T',-RILIN,R40,R40,t_)
         RILIN=RILIN+1.
      endif
      CALL PGUNSA      
 6022 format("Power integr.over rad. (RF+NBI(or KO), all gen.species)=",
     ~       1pe12.4,"Watts")
 6023 format("Power from NBI(or KO) (sorpw_nbii)=",1pe12.4,"Watts")
 6024 format("Power from RF  (sorpw_rfi) Gen.species no.1 =",
     ~       1pe12.4,"Watts")
 6025 format("Power from RF  (sorpw_rfi) Gen.species no.2 =",
     ~       1pe12.4,"Watts")
 6026 format("Power from RF  (sorpw_rfi) Gen.species no.3 =",
     ~       1pe12.4,"Watts")



      ! PRINT OUT OF DEPOSITED POWER (WATTS/CC): 
      ! powrft(l), powrf(l,1),...,powrf(l,5)
      CALL PGPAGE
      CALL PGSVP(R4P05,R4P95,R4P05,R4P95)
      RILIN=0.
      CALL PGSAVE
      CALL PGSCH(R41P3) !PGSCH(R41P44) ! set character size; default is 1.
      CALL PGMTXT('T',-RILIN,R4P5,R4P5,
     +     'DEPOSITED POWER: (WATTS/CC)')
      RILIN=RILIN+3.
      CALL PGSCH(R41) !(1.) ! set character size; default is 1. !YuP[2019-10-28]
      write(t_,6113) pltvs
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      RILIN=RILIN+2.
      write(t_,6114) pltvs
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      RILIN=RILIN+2.

 6113 format(1x,a8,"TOTAL",
     +  "       RF1         RF2        RF3         RF4        RF5")
 6114 format(1x,a8,"(powrft)",
     +  "           (powrf(*,harmonic) for harmonics = 1-5)")

c     Start printing results on first page
      do l=1,min(40,lrzmax)
        if (mrfn.eq.1) then
          write(t_,6115) tr(l),powrft(l),powrf(l,1)
        elseif (mrfn.eq.2) then
          write(t_,6116) tr(l),powrft(l),powrf(l,1),powrf(l,2)
        elseif (mrfn.eq.3) then
          write(t_,6117) tr(l),powrft(l),powrf(l,1),powrf(l,2),
     ~                         powrf(l,3)
        elseif (mrfn.eq.4) then
          write(t_,6118) tr(l),powrft(l),powrf(l,1),powrf(l,2),
     ~                         powrf(l,3),powrf(l,4)
        else
          write(t_,6119) tr(l),powrft(l),powrf(l,1),powrf(l,2),
     ~                         powrf(l,3),powrf(l,4),powrf(l,5)
        endif
        CALL PGMTXT('T',-RILIN,R40,R40,t_)
        RILIN=RILIN+1.
      enddo

c     Continue printing results on second page, if lrzmax.gt.40
      if (lrzmax.gt.40) then
         CALL PGPAGE
         CALL PGSVP(R4P05,R4P95,R4P05,R4P95)
         RILIN=0.+3.
         do l=41,lrzmax
           if (mrfn.eq.1) then
             write(t_,6115) tr(l),powrft(l),powrf(l,1)
           elseif (mrfn.eq.2) then
             write(t_,6116) tr(l),powrft(l),powrf(l,1),powrf(l,2)
           elseif (mrfn.eq.3) then
             write(t_,6117) tr(l),powrft(l),powrf(l,1),powrf(l,2),
     ~                            powrf(l,3)
           elseif (mrfn.eq.4) then
             write(t_,6118) tr(l),powrft(l),powrf(l,1),powrf(l,2),
     ~                            powrf(l,3),powrf(l,4)
           else
             write(t_,6119) tr(l),powrft(l),powrf(l,1),powrf(l,2),
     ~                            powrf(l,3),powrf(l,4),powrf(l,5)
           endif
           CALL PGMTXT('T',-RILIN,R40,R40,t_)
           RILIN=RILIN+1.
         enddo
      endif

 6115 format(f5.3, 2(1x,1pe9.2) )
 6116 format(f5.3, 3(1x,1pe9.2) )
 6117 format(f5.3, 4(1x,1pe9.2) )
 6118 format(f5.3, 5(1x,1pe9.2) )
 6119 format(f5.3, 6(1x,1pe9.2) )
      
      RILIN=RILIN+2.
      write(t_,6122) sorpwtza
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      RILIN=RILIN+1.
      write(t_,6123) powurf(0)
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      RILIN=RILIN+1.
      
      do krf=1,mrfn ! Print-out for all harmonics now (was 5 only)
         write(t_,6131) krf,nharm(krf),powurf(krf)
         CALL PGMTXT('T',-RILIN,R40,R40,t_)
         RILIN=RILIN+1.
      enddo
 6131 format("      mode/harmonic krf, nharm(krf), powurf(krf)=",
     +              2i4,1pe12.4)
 
!      if (mrfn.ge.1) then
!         write(t_,6124) powurf(1)
!         CALL PGMTXT('T',-RILIN,R40,R40,t_)
!         RILIN=RILIN+1.
!      endif
!      if (mrfn.ge.2) then
!         write(t_,6125) powurf(2)
!         CALL PGMTXT('T',-RILIN,R40,R40,t_)
!         RILIN=RILIN+1.
!      endif
!      if (mrfn.ge.3) then
!         write(t_,6126) powurf(3)
!         CALL PGMTXT('T',-RILIN,R40,R40,t_)
!         RILIN=RILIN+1.
!      endif
!      if (mrfn.ge.4) then
!         write(t_,6127) powurf(4)
!         CALL PGMTXT('T',-RILIN,R40,R40,t_)
!         RILIN=RILIN+1.
!      endif
!      if (mrfn.ge.5) then
!         write(t_,6128) powurf(5)
!         CALL PGMTXT('T',-RILIN,R40,R40,t_)
!         RILIN=RILIN+1.
!      endif
      
      write(t_,6129) powurfc(0)
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      RILIN=RILIN+1.
      write(t_,6130) powurfl(0)
      CALL PGMTXT('T',-RILIN,R40,R40,t_)

      CALL PGUNSA      

 6122 format
     + ("Power sources integr. over rad. (RF+NBI, all gen.species)=",
     ~       1pe12.4,"W")
 6123 format("Power from intern ray diagnostic[powurf(0)]=",1pe12.4,"W")
 6124 format("                mode/harmonic 1 [powurf(1)]=",1pe12.4)
 6125 format("                mode/harmonic 2 [powurf(2)]=",1pe12.4)
 6126 format("                mode/harmonic 3 [powurf(3)]=",1pe12.4)
 6127 format("                mode/harmonic 4 [powurf(4)]=",1pe12.4)
 6128 format("                mode/harmonic 5 [powurf(5)]=",1pe12.4)
 6129 format("Power by collisions (from ray data)    =",1pe12.4,"W")
 6130 format("Power by linear damping (from ray data)=",1pe12.4,"W")


c..................................................................
c     plots of sources and deposited power density [W/cm^3]..
c..................................................................

      fmin=0.
      fmax=0.
      gmin=0.
      gmax=0.
      call aminmx(powrft(1),1,lrzmax,1,gmin,gmax,kmin,kmax)
      call aminmx(sorpwt(1),1,lrzmax,1,fmin,fmax,kmin,kmax)
      if (gmin.lt.fmin) fmin=gmin
      if (gmax.gt.fmax) fmax=gmax
      !if (fmax-fmin.lt.1.e-16) fmin=fmax-.1*abs(fmax)-1.e-5
      if(fmax.gt.0.) fmax=fmax*1.05 ! extend the upper range
      RPG1=fmin ! could be negative because of numerical errors
      RPG2=fmax
      RPG1=min(fmin,0.) !Make lower limit 0 when fmin>0 
      
      do l=1,lrzmax
        tr1(l)=powrft(l)
      enddo

      CALL PGPAGE ! START FSA SOURCE POWER DEN
      CALL PGSVP(R4P2,R4P8,R4P2,R4P6)
      CALL PGSAVE
      CALL PGSCH(R41P44) ! set character size; default is 1.
      CALL PGSLW(LNWIDTH) ! line thickness/width
      CALL PGMTXT('T',R46,R4P5,R4P5,
     +              'FSA SOURCE POWER DEN: (WATTS/CM\u3\d)')
      CALL PGSCI(1) !black color
      CALL PGMTXT('T',R45,R40,R40,
     ~   "Solid: NBI(or KO)+RF for all gen.sp.[sorpwt]")
      CALL PGSCI(2) !red color
      CALL PGMTXT('T',R44,R40,R40,
     ~   "Dashed: NBI (or KO) [sorpw_nbi]")
      CALL PGSCI(1) !black color
      CALL PGMTXT('T',R43,R40,R40,
     ~   "Solid-bold: total absorbed RF power [powrft]")
      CALL PGMTXT('T',R42,R40,R40,
     ~   "Other: RF general species (each) [sorpw_rf]")
      CALL PGUNSA
 
        DO I=1,LRZMAX
           RLRZAP1(I)=tr(i)
        ENDDO
      ! If the horizontal coord is rho, set the limits to [0.,1.]
      RPGmin=RLRZAP1(1)
      RPGmax=RLRZAP1(LRZMAX)
      if(RPGmin.le.0.2) RPGmin=0. ! Lower limit in plots: extend to 0.
      if(RPGmax.ge.0.8 .and. RPGmax.lt.1.) RPGmax=1. ! Upper limit: extend to 1.
      IF ( RPG2-RPG1 .le. 1.e-16 ) THEN ! YuP [02-23-2016]
         RPG2= RPG1+1.e-16
      ENDIF
      CALL PGSWIN(RPGmin,RPGmax,RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGSLS(1) ! 1-> solid
        CALL PGSCI(1) !black color
        DO I=1,LRZMAX
           RLRZAP11(I)=sorpwt(i) ! solid: NBI(or KO)+RF(all gen.species)
        ENDDO
        CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP11(1))
        CALL PGSLS(2) ! 2-> dashed
        CALL PGSCI(2) ! red color
	
      do k=1,ngen ! NBI and rf sources for general species
           DO I=1,LRZMAX
             RLRZAP12(I)=sorpw_nbi(k,I) ! dashed: NBI only
             !YuP[2018-06-27] For each k_gen now
           ENDDO
           CALL PGSLS(2) !NBI:  dashed, for each k=k_gen 
           CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP12(1)) ! NBI, each k
         DO I=1,LRZMAX
            RLRZAP12(I)=sorpw_rf(k,I) 
         ENDDO
         CALL PGSLS(k+2) ! 3-> -.-.- ;   4-> .....
         CALL PGSCI(4) !blue color: RF
         CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP12(1))
      enddo ! k=1,ngen
      !
      CALL PGSLS(1) ! solid
      CALL PGSCI(1) !black color
      CALL PGSLW(LNWIDTH+1) ! bold
      DO I=1,LRZMAX
         RLRZAP13(I)=powrft(i)
      ENDDO
      CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP13(1)) !solid bold: total rf
      CALL PGSLS(1)
      CALL PGSCI(1) !black color, restore
      CALL PGSLW(LNWIDTH) !
      CALL PGSAVE
      CALL PGSCH(R41P44)
      CALL PGLAB(' ','power density (W/cm\u3\d)',' ')
      CALL PGMTXT('B',R41P8,R4P5,R4P5,t_horiz)
      CALL PGUNSA
      RILIN=6.
      write(t_,10150) n,timet 
      CALL PGMTXT('B',RILIN,R4MP2,R40,t_) ! n,time
10150 format("time step (n) is",i5,5x,"time=",1pe12.4," secs")      
      ! DONE FSA SOURCE POWER 

      
      !YuP[08-2017] Make a separate page for only RF-power density
      !(because, if the power level is too small, the curve can be too low)
      CALL PGPAGE ! START RF-only POWER DEN
      CALL PGSVP(R4P2,R4P8,R4P2,R4P6)
      CALL PGSAVE
      CALL PGSCH(R41P44) ! set character size; default is 1.
      CALL PGSLW(LNWIDTH) ! line thickness/width
      CALL PGMTXT('T',R46,R4P5,R4P5,
     +              'FSA RF POWER DEN: (WATTS/CM\u3\d)')
      CALL PGMTXT('T',R43,R40,R40,
     ~   "Solid-bold: total absorbed RF power [powrft]")
      CALL PGMTXT('T',R42,R40,R40,
     ~   "Other: RF general species (each) [sorpw_rf]")
      CALL PGUNSA
      ! If the horizontal coord is rho, set the limits to [0.,1.]
      RPGmin=RLRZAP1(1)
      RPGmax=RLRZAP1(LRZMAX)
      if(RPGmin.le.0.2) RPGmin=0. ! Lower limit in plots: extend to 0.
      if(RPGmax.ge.0.8 .and. RPGmax.lt.1.) RPGmax=1. ! Upper limit: extend to 1.
      ! Vertical axis limits:
      call aminmx(powrft(1),1,lrzmax,1,gmin,gmax,kmin,kmax)
      RPG1=gmin ! could be negative because of numerical errors
      RPG2=gmax
      !YuP[2018-06-27] Added min/max estimate from sorpw_rf
      do k=1,ngen ! rf sources for general species
        wk_lrz(1:lrzmax)=sorpw_rf(k,1:lrzmax)
        call aminmx(wk_lrz,1,lrzmax,1,gmin,gmax,kmin,kmax)
        RPG1=min(RPG1,gmin) 
        RPG2=max(RPG2,gmax)
      enddo
      RPG1=min(RPG1,0.) !Make lower limit 0 when gmin>0 
      RPG2=RPG2*1.2 ! give 20% extra
      IF ( RPG2-RPG1 .le. 1.e-16 ) THEN ! YuP [02-23-2016]
         RPG2= RPG1+1.e-16
      ENDIF
      CALL PGSWIN(RPGmin,RPGmax,RPG1,RPG2)
      CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
      do k=1,ngen ! rf sources for general species
         DO I=1,LRZMAX
            RLRZAP12(I)=sorpw_rf(k,I) 
         ENDDO
         CALL PGSLS(k+2) ! 3-> -.-.- ;   4-> .....
         CALL PGSCI(4) !blue color: RF
         CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP12(1))
      enddo ! k=1,ngen
      !
      CALL PGSLS(1) ! solid
      CALL PGSCI(4) !!blue color: RF 
      CALL PGSLW(LNWIDTH+1) ! bold
      DO I=1,LRZMAX
         RLRZAP13(I)=powrft(i)
      ENDDO
      CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP13(1)) !solid bold: total rf
      CALL PGSCI(1) ! black color restored
      CALL PGSLS(1)
      CALL PGSLW(LNWIDTH) !
      CALL PGSAVE
      CALL PGSCH(R41P44)
      CALL PGLAB(' ','power density (W/cm\u3\d)',' ')
      CALL PGMTXT('B',R41P8,R4P5,R4P5,t_horiz)
      CALL PGUNSA
      ! DONE RF-only POWER 


c..................................................................
c     plots of partial integration of powers [Watts]
c..................................................................

      fmin=0.
      fmax=0.
      call aminmx(sorpwti(1),1,lrzmax,1,fmin,fmax,kmin,kmax)
      if(fmax.gt.0.) fmax=fmax*1.05 ! extend the upper range
      
      do 50 kk=1,mrfn
        call aminmx(powurfi(1,kk),1,lrzmax,1,gmin,gmax,kmin,kmax)
        if (gmin.lt.fmin) fmin=gmin
        if (fmax.lt.gmax) fmax=gmax
 50   continue
      if (fmax-fmin.lt.1.e-8) fmin=fmax-.1*abs(fmax)-1.e-5
      RPG1=fmin
      RPG2=fmax
      RPG1=min(fmin,0.) !Make lower limit 0 when fmin>0 
      
        CALL PGPAGE
        CALL PGSVP(R4P2,R4P8,R4P2,R4P6)
        CALL PGSAVE
        CALL PGSCH(R41P44)
        CALL PGSCI(1) !black color
        CALL PGMTXT('T',R46,R4P5,R4P5,
     +              'SOURCE POWER (integr. up to rho or psi) (WATTS)')
        CALL PGMTXT('T',R45,R40,R40,
     ~   "Solid: NBI(or KO)+RF for all gen.sp.[sorpwti]")
        CALL PGSCI(2) !red color
        CALL PGMTXT('T',R44,R40,R40,
     ~   "Dashed: NBI(or KO) [sorpw_nbii]")
        CALL PGSCI(1) !black color
        CALL PGMTXT('T',R43,R40,R40,
     ~   "Solid-bold: total absorbed RF [powurfi(*,0)]")
        CALL PGMTXT('T',R42,R40,R40,
     ~   "Other: RF general ions (each) [sorpw_rfi]")
        CALL PGUNSA
 
        DO I=1,LRZMAX
           RLRZAP1(I)=tr(i)
        ENDDO
        
      ! If the horizontal coord is rho, set the limits to [0.,1.]
      RPGmin=RLRZAP1(1)
      RPGmax=RLRZAP1(LRZMAX)
      if(RPGmin.le.0.2) RPGmin=0. ! Lower limit in plots: extend to 0.
      if(RPGmax.ge.0.8 .and. RPGmax.lt.1.) RPGmax=1. ! Upper limit: extend to 1.
        
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN ! YuP [02-23-2016]
           RPG2= RPG1+1.e-16
        ENDIF
        CALL PGSWIN(RPGmin,RPGmax,RPG1,RPG2)
        CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        CALL PGSLS(1) ! 1-> Solid line
        CALL PGSCI(1) !black color
        DO I=1,LRZMAX
           RLRZAP11(I)=sorpwti(i) !solid: NBI(or KO)+RF(all gen.species)
        ENDDO
        CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP11(1))
      do k=1,ngen ! rf sources for general species
           CALL PGSLS(2) ! 2-> dashed
           CALL PGSCI(2) !red color
           DO I=1,LRZMAX
           RLRZAP12(I)=sorpw_nbii(k,I) !dashed: NBI, each k_gen
           ENDDO
           CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP12(1)) ! NBI
         DO I=1,LRZMAX
            RLRZAP12(I)=sorpw_rfi(k,I) 
         ENDDO
         CALL PGSCI(1) !black color
         CALL PGSLS(k+2) ! 3-> -.-.- ;   4-> .....
         CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP12(1))
      enddo ! k=1,ngen
      CALL PGSLS(1) ! solid
      CALL PGSCI(1) !black color
      CALL PGSLW(LNWIDTH+1) ! bold
      DO I=1,LRZMAX
         RLRZAP13(I)=powurfi(i,0) ! = SUM_harmonics{powurfi(l,harmonics)}
      ENDDO
      CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP13(1)) !solid bold: total rf
      CALL PGSLS(1) ! restore: solid line
      CALL PGSLW(LNWIDTH) ! restore
      CALL PGSAVE
      CALL PGSCH(R41P44)
      CALL PGLAB(' ','Int power: 0 to psi (or rho)',' ')
      CALL PGMTXT('B',R41P8,R4P5,R4P5,t_horiz)
      RILIN=6.
      write(t_,10150) n,timet 
      CALL PGMTXT('B',RILIN,R4MP2,R40,t_) ! n,time
      CALL PGUNSA ! restore


c..................................................................
c     plots of partial integration of RF-only powers [Watts]
      !YuP[08-2017] Make a separate page for RF-power only
      !(because, if the power level is too small, the curve can be too low)
c..................................................................

      call aminmx(powurfi(1,0),1,lrzmax,1,gmin,gmax,kmin,kmax)
      if (gmax-gmin.lt.1.e-8) gmin=gmax-.1*abs(gmax)-1.e-5
      RPG1=gmin
      RPG2=gmax*1.2 ! give 20% more
      RPG1=min(gmin,0.) !Make lower limit 0 when gmin>0 
      
        CALL PGPAGE
        CALL PGSVP(R4P2,R4P8,R4P2,R4P6)
        CALL PGSAVE
        CALL PGSCH(R41P3)
        CALL PGMTXT('T',R46,R4P5,R4P5,
     +              'RF POWER (integr. up to rho or psi) (WATTS)')
        CALL PGMTXT('T',R43,R40,R40,
     ~   "Solid-bold: total absorbed RF [powurfi(*,0)]")
        CALL PGMTXT('T',R42,R40,R40,
     ~   "Other: RF general species (each) [sorpw_rfi]")
     
        CALL PGUNSA
         
      ! If the horizontal coord is rho, set the limits to [0.,1.]
      RPGmin=RLRZAP1(1)
      RPGmax=RLRZAP1(LRZMAX)
      if(RPGmin.le.0.2) RPGmin=0. ! Lower limit in plots: extend to 0.
      if(RPGmax.ge.0.8 .and. RPGmax.lt.1.) RPGmax=1. ! Upper limit: extend to 1.
        
      IF ( RPG2-RPG1 .le. 1.e-16 ) THEN ! YuP [02-23-2016]
          RPG2= RPG1+1.e-16
      ENDIF
      CALL PGSWIN(RPGmin,RPGmax,RPG1,RPG2)
      CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
      CALL PGSLS(1) ! 1-> Solid line
      !
      do k=1,ngen ! rf sources for general species
         DO I=1,LRZMAX
            RLRZAP12(I)=sorpw_rfi(k,I) 
         ENDDO
         CALL PGSLS(k+2) ! 3-> -.-.- ;   4-> .....
         CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP12(1))
      enddo ! k=1,ngen
      CALL PGSLS(1) ! solid
      CALL PGSLW(LNWIDTH+1) ! bold
      DO I=1,LRZMAX
         RLRZAP13(I)=powurfi(i,0) ! = SUM_harmonics{powurfi(l,harmonics)}
      ENDDO
      CALL PGLINE(lrzmax,RLRZAP1(1),RLRZAP13(1)) !solid bold: total rf
      CALL PGSLS(1) ! restore: solid line
      CALL PGSLW(LNWIDTH) ! restore
      CALL PGSAVE
      CALL PGSCH(R41P44)
      CALL PGLAB(' ','Int power: 0 to psi (or rho)',' ')
      CALL PGMTXT('B',R41P8,R4P5,R4P5,t_horiz)
      CALL PGUNSA ! restore

c..................................................................
c     Print out figures af merit and other average quantities.
c..................................................................

 809  CONTINUE

      CALL PGSCH(R41) ! restore to default font size      
      return
      end
