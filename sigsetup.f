c
c
      subroutine sigsetup
      implicit integer (i-n), real*8 (a-h,o-z)
c
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

c     Written by m.g. mccoy, revised for cql3d by bobH (dec94).
c     Controls setting-up of fusion sigma-v-bar integration factors
c     (indep of time step).

      if (sigmamod.ne."enabled") return
      if (msig .eq. 0) return   !msig is number of different reactions,
                                !the sum of isigmas()
c     Presently , up to msig=4 nuclear rxs plus simple impact and CX 
c     are treated.  Storage only setup for any 4 of these processes.
      if (msig.gt.4) then
        write(*,1000)
 1000   format('Need to check code for msig.gt.4, resetting sigmamod')
        sigmamod="disabled"
        return
      endif
c
c     Various arrays must be determined for later use...
c

      do 10 j=1,jx
   10 iind(j)=j*(j-1)/2
      temc1(1) = 0.
      temc1(iy)=pi
      iym1=iy-1
      hy2 = temc1(iy)/dfloat(iym1)
      do 20 i = 2,iym1
      temc1(i) = temc1(i-1)+hy2
   20 continue
      do 30 i = 1,iy
      temc2(i) = cos(temc1(i))
      temc3(i) = sin(temc1(i))
   30 continue
      do 40 i = 2,iym1
      temc4(i) = twopi*temc3(i)*(temc1(i+1)-temc1(i-1))/2.
   40 continue
      temc4(1)=pi*temc1(2)**2/4.
      temc4(iy)=temc4(1)
c     The Legendre coefficients pleg are used only in this subroutine
c     for following calculation of csv.
      do 60  i = 1,iy
      pleg(0,i) = 1.
      if (mmsv .eq. 0) go to 60
      pleg(1,i) = temc2(i)
      do 50 m = 2,mmsv
      pleg(m,i)=((2*m-1)*temc2(i)*pleg(m-1,i)-(m-1)*pleg(m-2,i))/m
   50 continue
   60 continue
      na=0
309   continue
      na=na+1

c     The four nuclear reactions presently treated are:
c     isigmas(1)=1, then compute d+t => alpha+n
c            (2)=1               d+he3 => alpha+p  
c            (3)=1               d+d => t+p
c            (4)=1               d+d => he3+n

c     For each of rxs isigmas(1:4), there must be two species,
c     each either Maxwellian or a General species.
c     The imaxwln(1:2,1:4) and igenrl(1:2,1:4) give species
c     indices in the 1:ntotal namelist list which are available
c     as the first or second collisional body, for each of the
c     reactions, indexed 1:4, (indep of whether calcd).
c     imaxwln(first or second maxwl species, reaction number)
c     igenrl(first or second general species, reaction number)
c     D+ mass: 3.3436e-24, bnumb=1.
c     T+ mass: 5.0074e-24, bnumb=1.
c     He3++ mass: 5.0064e-24, bnumb=2.
      call ibcast(imaxwln,0,8)
      call ibcast(igenrl,0,8)
      do 601 ks=1,ntotal
         if (abs ((fmass(ks) -3.3436e-24)/fmass(ks)) .gt. .01) go to 601
c     Here, if D:
         if (ks .le. ngen) go to 603
         ! Maxwellian D is available for:
         imaxwln(1,1)=ks ! D is 1st in reaction#1 (D+T  ->He4+n)
         imaxwln(1,2)=ks ! D is 1st in reaction#2 (D+He3->He4+p)
         imaxwln(1,3)=ks ! D is 1st in reaction#3 (D+D  ->T+p)
         imaxwln(1,4)=ks ! D is 1st in reaction#4 (D+D  ->He3+n)
         imaxwln(2,3)=ks ! D is 2nd in reaction#3 (D+D  ->T+p)
         imaxwln(2,4)=ks ! D is 2nd in reaction#4 (D+D  ->He3+n)
         go to 601
 603     igenrl(1,1)=ks
         igenrl(1,2)=ks
         igenrl(1,3)=ks
         igenrl(1,4)=ks
         igenrl(2,3)=ks
         igenrl(2,4)=ks
 601  continue
      !write(*,*)'Finished checking for D' 
      !WRITE(*,*)'sigsetup: imaxwln(1,1:4)=', imaxwln(1,1:4)
      !WRITE(*,*)'sigsetup: imaxwln(2,1:4)=', imaxwln(2,1:4)
      !pause
 
      do 605 ks=1,ntotal
         if (abs((fmass(ks) - 5.007e-24)/fmass(ks)) .gt. .01 .or.
     1        ((bnumb(ks)-1d0).gt. .01)) go to 605
c     Here, if Trit:
         if(ks .le. ngen) go to 609
         ! Maxwellian T is available for:
         !imaxwln(1,1)=ks  ! D is 1st in reaction#1 (D+T  ->He4+n) BH160720, by analogy with D settings.
         !YuP: But we are checking for T (species #ks), so cannot put 'ks' into imaxwln(1,1)
         imaxwln(2,1)=ks  ! T is 2nd in reaction#1 (D+T  ->He4+n)
         go to 605
 609     igenrl(2,1)=ks   ! Genrl T is 2nd in reaction#1 (D+T  ->He4+n)
         !igenrl(1,1)=ks   ! D is 1st in reaction#1 (D+T  ->He4+n) BH160720, by analogy with D settings.
         !YuP: But we are checking for T (species #ks), so cannot put 'ks' into igenrl(1,1)
 605  continue
      !write(*,*)'Finished checking for T' 
      !WRITE(*,*)'sigsetup: imaxwln(1,1:4)=', imaxwln(1,1:4)
      !WRITE(*,*)'sigsetup: imaxwln(2,1:4)=', imaxwln(2,1:4)
      !pause
 
      do 621 ks=1,ntotal
      if (abs((fmass(ks)-5.007e-24)/fmass(ks)) .gt. .01 .or.
     1        bnumb(ks) .ne. 2) go to 621
c     Here, if He3:
      if (ks .le. ngen) go to 623
      ! Maxwellian He3 is available for:
      !imaxwln(1,2)=ks  ! Maxwln D is 1st in reaction#2 (D+He3->He4+p) BH160720, by analogy with D settings.
      !YuP: But we are checking for He3 (species #ks), so cannot put 'ks' into imaxwln(1,2)
      imaxwln(2,2)=ks  ! He3 is 2nd in reaction#2 (D+He3->He4+p)
      go to 621
 623  igenrl(2,2)=ks   ! Genrl He3 is 2nd in reaction#2 (D+He3->He4+p)
      !igenrl(1,2)=ks   ! Genrl D is 1st in reaction#2 (D+He3->He4+p) BH160720, by analogy with D settings.
      !YuP: But we are checking for He3 (species #ks), so cannot put 'ks' into igenrl(1,2)
 621  continue

c.......................................................................
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'sigsetup: Fusion Reactants in four types of reactions:'
      WRITE(*,'(a,4i3)')
     +'sigsetup: 1st reactant (D,D,D,D)   igenrl(1,1:4)= ',igenrl(1,1:4)
      WRITE(*,'(a,4i3)')
     +'sigsetup: 2nd reactant (T,He3,D,D) igenrl(2,1:4)= ',igenrl(2,1:4)
      WRITE(*,'(a,4i3)')
     +'sigsetup: 1st reactant (D,D,D,D)   imaxwln(1,1:4)=',imaxwln(1,:)
      WRITE(*,'(a,4i3)')
     +'sigsetup: 2nd reactant (T,He3,D,D) imaxwln(2,1:4)=',imaxwln(2,:)
      !----------------------------------------------------------------
      WRITE(*,*)'sigsetup: indicator of calc. of reaction rates:'
      WRITE(*,'(a,4i3)')'sigsetup: isigmas(1:4)=', isigmas(1:4)
CMPIINSERT_ENDIF_RANK

c     If there is no general species, D, T or He3, 
c     or Maxwellian species that can react with general or Maxwellian, then
c     sigmamod is reset to "disabled".
c.......................................................................

      iigenrl=0
      do kk=1,4 ! over reaction types
         !YuP[07-2016] check every reaction type for presence
         ! of 1st and 2nd reactant, either as general or as Maxw. species:
         if( (igenrl(1,kk)+imaxwln(1,kk).ne.0) .and.
     +       (igenrl(2,kk)+imaxwln(2,kk).ne.0)      ) then
             iigenrl=iigenrl+1 ! Both reactants present, 
             ! either as General species or as Maxw. species
         endif
cBH120312         iigenrl=igenrl(1,kk)+igenrl(2,kk)
cYuP[07-2016]         iigenrl=iigenrl+igenrl(1,kk)+igenrl(2,kk)
cYuP[07-2016]     +                  +igenrlp(1,kk)+igenrlp(2,kk)
      enddo
      
      if (iigenrl.eq.0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
         WRITE(*,*) 'sigsetup:  No nuclear fusion species'
         WRITE(*,*) 'sigsetup:  Setting sigmamod=disabled'
         WRITE(*,*)
CMPIINSERT_ENDIF_RANK
         sigmamod='disabled'
      endif

cBH170609: Checked whether there were results changes for a T=genrl, DT case,
cBH170609: ngen=1 using sigsetup.f.YuP.mainline, sigsetup.f_BH, sigsetup.f.orig
cBH170609: ==> sigsetup.f_BH chose wrong distributions, and the other two
cBH170609:     gave same results. [ Tests in Mirror_NBI.5/170609_rerun/]

c
c     next a sigma table is constructed. the values in the
c     table depend upon knumb.
c        if knumb = 1, the reaction is d+t=>alpha(3.5MeV)+n(14.1MeV)
c        if knumb = 2, the reaction is d+he3=>alpha(3.6MeV)+p(14.7MeV)
c        if knumb = 3, the reaction is d+d=>t(1.01Mev)+p(3.02MeV)
c        if knumb = 4, the reaction is d+d=>he3(.82MeV)+n(2.45MeV)
c        if knumb = 5, the reaction is hydrogenic impact ionization
c        if knumb = 6 the reaction is charge exchange 
c     
c     The table is stored in csv(jx*(jx+1)/2,0:mmsv,msig)
c     csv is proportional to (equal?) to I_jj'l (Eq. 18, 
c     Mirin and McCoy, UCRL-98568 (April 6, 1998),  
c     and Comp. Phys. Comm.?), for each sigma considered.
c
      mtabm1=mtab-1

c     Loop over sigma-v calculations
      iq=0
      do 120 knumb=1,6
      if (isigmas(knumb) .eq. 0) go to 120
      iq=iq+1  ! Count number of sigma-v calcs.

      fi=(1.67165e-24)*vnorm**2/(ergtkev*2.)

c     For fusion, need the max energy in center of mass frame.
c     fi is normalization for relative velocity vnorm.
c     2*vnorm is max relative velocity, so max rel. energy = 4.*fi
c     For cntr of mass, fi=0.5*(mass_a*mass_b/(mass_a+mass_b))*vnorm**2
      if (knumb.eq.1 .or. knumb.eq.2) then
        fi=(6./5.)*fi
      elseif (knumb.eq.3 .or. knumb.eq.4) then
        fi=1.*fi
      else
        fi=fi
      endif

      engymax=4.*fi*xsq(jx)
      elmin=fi*xsq(2)
      delegy=(engymax-elmin)/dfloat(mtabm1)

      if (iq .gt. msig) go to 120  !Shouldn't be possible?
      els=elmin-delegy
c     svtab calculated here and used below by subroutine siggy, a table
c     look-up of the energy dependent reaction cross-section.
      do 70 j = 1,mtab
      els=els+delegy 
      svtab(j) = sigfn(els,knumb)
 70   continue

c     Loop over velocity
      jsum=0
      do 110 jv1=1,jx
      fact=1. 
      if (jv1 .eq. jx) fact=.5

c     Loop over velocity-prime
      do 100 jv2=1,jv1
      if (jv2 .eq. jx) fact=.5*fact
      jsum=jsum+1
      vicp=cint2(jv1)*cint2(jv2)*8.*pi**2
      vicp=vicp*fact
      sqsum=xsq(jv1)+xsq(jv2)
      twoprd=-2.*x(jv1)*x(jv2)

c     Loop over Legendre polynomials
      do 90 m = 0,mmsv
      div=2.*m+1.
      vicp2=vicp/div
      sum1=0.
      if (isigtst .eq. 1 .and. knumb .eq. 5) then
        do 81 jfee=1,iy
 81     sum1=sum1+sigvi/vnorm*pleg(m,jfee)*temc4(jfee)
      endif
      if (isigtst .eq. 1 .and. knumb .eq. 6) then
        do 82 jfee=1,iy
 82     sum1=sum1+sigvcx/vnorm*pleg(m,jfee)*temc4(jfee)
      endif
      if (knumb .gt. 4 .and. isigtst .eq. 1) go to 13
      do 80 jfee=1,iy
      usq=sqsum+twoprd*temc2(jfee)
c990131      usq=amax1(usq,0.)
      usq=max(usq,zero)
      egy=fi*usq
      suc=siggy(egy)
 80   sum1=sum1+sqrt(usq)*suc*pleg(m,jfee)*temc4(jfee)
 13   continue
 90   tamm1(m)=sum1*vicp2*vnorm/twopi  ! m=0,mmsv
      call dcopy(mmsv+1,tamm1,1,csv(jsum,0,iq),jxis)

 100  continue  !on jv2=1,jv1
 110  continue  !on jv1=1,jx
 120  continue  !on knumb=1,6

      return
      end
