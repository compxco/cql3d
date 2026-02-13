c
c
      subroutine pltfvsv
      implicit integer (i-n), real*8 (a-h,o-z)
      save
      include 'param.h'
      include 'comm.h'

      REAL*4 RTAM1(jx),RTAM2(jx)
      REAL*4 REMAX,REMIN
      REAL*4 :: R4P2=.2,R4P8=.8,R4P35=.35,R4P9=.9
      REAL*4 :: R40=0.,R41P44=1.44
      REAL*4 :: R47=7.,R48=8.,R49=9.,R410=10.,R412=12.

      CHARACTER*64 TX_
      real*8 wkd(jx) 

c...  
cmnt  This routine plots f as a function of velocity or energy 
cmnt  for various angles.
c...  
      if (noplots.eq."enabled1") return


      do 120 k=1,ngen ! ---------------------------------------------
      
         if (pltlim.eq."disabled") then
            jxq=jx
            xmaxq=x(jxq)
            do j=1,jxq
               tam1(j)=x(j)
            enddo
            TX_='u/vnorm'
         endif
         if (tandem.eq."enabled" .and. fmass(k).gt.1.e-27) then
            jxq=jlwr
            xmaxq=xlwr
            do j=1,jxq
               tam1(j)=x(j)
            enddo
            TX_='u/vnorm'
         ! If pltlim .ne. "disabled", plot versus
         ! 'x', 'u/c', or 'energy', up to maximum pltlimm.
         elseif (pltlim.eq.'x') then
            jxq=min(luf(pltlimm,x,jx),jx)
            xmaxq=x(jxq)
            do j=1,jxq
               tam1(j)=x(j)
            enddo
            TX_='u/vnorm'
         elseif (pltlim.eq.'u/c') then
            jxq=min(luf(pltlimm,uoc,jx),jx)
            xmaxq=uoc(jxq)
            do j=1,jxq
               tam1(j)=uoc(j)
            enddo
            TX_='u/c\dlight\u'
         elseif (pltlim.eq.'energy') then
            pltlimmm=pltlimm
            wkd(1:jx)=enerkev(1:jx,k)
            jxq=min(luf(pltlimmm,wkd,jx),jx)
            xmaxq=enerkev(jxq,k) !YuP[2018-01-08] added 2nd index (k)
            do j=1,jxq
               tam1(j)=enerkev(j,k) !YuP[2018-01-08] added 2nd index (k)
            enddo
            TX_='Energy (keV)'
         endif
        
         imsh(1)=1
         imsh(2)=itl
         imsh(3)=iyh
         imsh(4)=iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
         !Note that for meshy="fixed_mu" iy_(l_) can be less than iy
         imsh(5)=0
         
         call bcast(tam4,zero,jx)
         call bcast(tam5,zero,jx)
c     tam4 contains pitch angle averaged equatorial distribution
c          function.
c     tam5 to contain pitch angle integrated distribution <f> such
c          that integral{<f> d(gamma)} = fsa density.
c          tam5=2*pi*x*cnorm2*gamma*int_i{sinn*dy*vptb*f_code}/zmaxpsi.
         do 40 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
            !Note that for meshy="fixed_mu" iy_(l_) can be less than iy
            do 30 j=1,jx
               tam4(j)=tam4(j)+f(i,j,k,l_)*cynt2(i,l_)
               tam5(j)=tam5(j)+f(i,j,k,l_)*cynt2(i,l_)*vptb(i,lr_)
               !YuP[2021-02-26] Corrected vptb(i,l_)->vptb(i,lr_)
               !This is important for CQLP (when lr_=1, but l_=1:ls)
               !Probably should not be using vptb in CQLP version at all.
 30         continue
 40      continue
         do 41 j=1,jx
            tam4(j)=tam4(j)/twoint(l_)
            tam5(j)=tam5(j)/zmaxpsi(lr_)*x(j)*gamma(j)*cnorm2
 41      continue
c-----------------------------
c  Save pitch angle averaged fsa distribution 
c  (for seperate print and output). (SC Sept./96)
cbh970403        itstp=n_(l_)/nplot
cbh971027         itstp=n_(l_)/nplot+1
        if (iplot.gt.0 .and. iplot.le.5) then
           do 35 j=1,jx
              f_aveth(j,k,l_,iplot)=tam5(j)
 35        continue
        endif
c-----------------------------

        if (pltfvs.eq."enabled") then

        call aminmx(tam4,1,jxq,1,emin,emax,kmin,kmax)
        do 60 iu=1,5
           if (iu.ne.5) then
              i=imsh(iu)
              do j=1,jxq
                 tam2(j)=f(i,j,k,l_)
              enddo
           else
              do j=1,jxq
                 tam2(j)=tam4(j)
              enddo
           endif
           call aminmx(tam2,1,jxq,1,fmin,fmax,kmin,kmax)
           if (fmin .lt. emin) emin=fmin
           if (fmax .gt. emax) emax=fmax
 60     continue
        emax=emax*1.03
!        emin=emax/1.e+12
        emin=emax*contrmin


        DO J=1,JXQ
           RTAM1(J)=TAM1(J)
           TAM2(J)=ABS(TAM2(J))
           TAM2(J)=MAX(em300,TAM2(j))
           RTAM2(J)=LOG10(TAM2(J))
        ENDDO
        REMIN=LOG10(EMIN)
        REMAX=LOG10(EMAX)

        CALL PGPAGE

        CALL PGSVP(R4P2,R4P8,R4P35,R4P9)
        IF ( ReMAX-ReMIN .le. 1.e-16 ) THEN ! YuP [02-23-2016]
           ReMAX= ReMIN+1.e-16
        ENDIF
        CALL PGSWIN(Rtam1(1),Rtam1(jxq),Remin,Remax)
        CALL PGSAVE
        CALL PGSCH(R4P8)
        CALL PGBOX('BCNST',R40,0,'BCNSTL',R40,0)
        CALL PGUNSA
        CALL PGSAVE
        CALL PGSCH(R41P44)
        CALL PGLAB(TX_, 'f', 'Cuts of f vs. v, at cnst pitch angle')
        CALL PGUNSA

        do 110 iu=1,5
           if (iu.ne.5) then
              i=imsh(iu)
              do j=1,jxq
                 tam2(j)=f(i,j,k,l_)
              enddo
           else
              do j=1,jxq
                 tam2(j)=tam4(j)
              enddo
           endif
           if (iu .eq. 1) text(1)="pll$"
           if (iu .eq. 2) text(1)="trp/ps$"
           if (iu .eq. 3) text(1)="perp$"
           if (iu .eq. 4) text(1)="pll-pi$"
           if (iu .eq. 5) text(1)="avg$"
 90        continue
           do 100 j=1,jxq
              if(tam2(j).le.emin) tam2(j)=emin
              RTAM2(J)=LOG10(TAM2(J))
 100       continue
           CALL PGLINE(JXQ,RTAM1,RTAM2)
          xu=float(iu)
 110   continue





       rr=rpcon(lr_) !rovera(lr_)*radmin  ! YuP[03-2016] changed to rpcon
       if (pltlim.eq.'u/c') then
          write(t_,10021) 
       elseif (pltlim.eq.'energy') then
          write(t_,10022) 
       else
          write(t_,10020) 
       endif

       CALL PGMTXT('B',R47,R40,R40,t_)
       write(T_,10023) K,ENORM
       CALL PGMTXT('B',R48,R40,R40,t_)

       write(t_,10010) n,timet
       CALL PGMTXT('B',R49,R40,R40,t_)
       write(t_,10011) rovera(lr_),rr
       CALL PGMTXT('B',R410,R40,R40,t_)

       if(cqlpmod.eq."enabled")then !YuP[2021-03-03] added for CQLP:
        write(t_,10032) l_,sz(l_)
10032 format("Index along B, l=",i4, 4x,  
     &       "Parallel position s=",1pe14.6,"cm")
        CALL PGMTXT('B',R412,R40,R40,t_)
       endif

c        CALL PGEND
c        STOP
c        CALL PGPAGE

       endif


c     Plot pitch angle integrated distribution function <f>, normalized
c     so integral{<f> d(gamma)} = flux surface averaged density.
 120  continue
10010 format("time step (n)=",i5,5x,"time=",e14.6," secs")
10011 format("r/a=",1pe10.2,5x,"radial position(R)=",1pe10.3," cm")
10020 format("Distribution function vs. velocity for some angles")

10021 format("Distribution function vs. u/c for some angles")

10022 format("Distribution function vs. energy(kev) for some angles")

10023 format("Species number=",i1,",   enorm=",1pd10.2)
10030 format("pll ---- theta=0 radians",";",
     &     "pll-pi ---- theta=pi radians",";",
     &     "trp/ps ---- theta = ",e13.5," radians",";",
     &     "perp ---- theta = ",e13.5," radians",";",
     &     "avg ---- theta averaged over pi radians","$")

20010 format("time step (n) is",i5,5x,"time=",e14.6," secs",";",
     &     "r/a=",e14.6,5x,"radial position (R) =",e14.6," cm","$")
20020 format("Pitch angle integrated electron distribution vs v/vnorm"
     &     ,";","(normed so that int{<f> d(gamma)}=fsa n_e",
     &     ";","$")
20021 format("Pitch angle integrated electron distribution vs u/c "
     &     ,";","(normed so that int{<f> d(gamma)}=fsa n_e",
     &     ";","$")
20022 format("Pitch angle integrated electron distribution vs energy "
     &     ,";","(normed so that int{<f> d(gamma)}=fsa n_e",
     &     ";","$")
     
      return
      end
