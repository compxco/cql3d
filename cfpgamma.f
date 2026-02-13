c
c
      subroutine cfpgamma
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c.......................................................................
c     calculate the Coulomb logarithm and an energy dependent factor.
c     BH180807: Needs reviewing.  Most often gamaset=fixed value (16.
c     or 17.) is used.  Should re-investigate ln(Lambda) expressions.
c     Compare/code NRL formulary expressions.
c.......................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

c..................................................................
c     If log is set by fiat,  set it and move to gamefac
c..................................................................


      if (gamaset.gt.zero) then
         do 12 kk=1,ntotal
            do 13 k=1,ntotal
               gama(kk,k)=gamaset
 13         continue
 12      continue
         goto 99 !-> Continue,  setup gamefac()
      endif
      
      if (gamaset.lt.zero) then ! YuP[2019-07], YuP[2020-06-23] added
         ! Use expressions from NRL_FORMULARY_2016, p.34
!         te_ev= (2./3.)*energy(kelec,lr_)*1.d3 ! in eV
!         fmass_e= fmass(kelec)
!         dense=reden(kelec,lr_) ! Electron density [cm-3]
!         ! Thermal e-e collisions (no p-dependence here):
!         cNRL_ee= 23.5 -log(sqrt(dense) * te_ev**(-1.25))  
!     &                 -sqrt( 1d-5  +(1./16.)*(log(te_ev)-2.)**2 )
!         ! For e-i collisions, low Te case (lower than 10eV*Zi^2)
!         if( (te_ev .gt. Ti*1000*me/mi)  .and.  (te_ev .le. 10*Zi**2) )then
!            cNRL_ei= 23.d0 -log(sqrt(dense)*Zi*te_ev**(-1.5))  
!         endif
!         ! For e-i collisions, higher Te case
!         if(te_ev .gt. 10*Zi**2)then
!            cNRL_ei= 24.d0 -log(sqrt(dense)/te_ev) 
!         endif
        !---------- See below for e-i and i-i collisions
        do kk=1,ntotal
          if (cqlpmod.eq."enabled") then !YuP[2021-02-24]
          t_kk_ev=(2./3.)*enrgypa(kk,ls_)*1.d3
          dens_kk=denpar(kk,ls_)
          else
          t_kk_ev=(2./3.)*energy(kk,lr_)*1.d3 !Effective T[eV] for kk species
          dens_kk= reden(kk,lr_) !density [cm-3]
          endif
        !Note that if kk=Maxw.species, energy is updated at each time step
        !as 1.5*temp(kk,lr) (or relativistic expression), if temp() is changed.

        do k=1,ntotal
          if (cqlpmod.eq."enabled") then !YuP[2021-02-24]
          t_k_ev=(2./3.)*enrgypa(k,ls_)*1.d3
          dens_k=denpar(k,ls_)
          else
          t_k_ev=(2./3.)*energy(k,lr_)*1.d3 !Effective T[eV] for k species
          dens_k= reden(k,lr_) !density [cm-3]
          endif
          if(kk.eq.kelecg .or. kk.eq.kelecm)then
             ! kk=electrons (general or Maxwellian)
             if(k.eq.kelecg .or. k.eq.kelecm)then
               ! k=electrons (general or Maxwellian)
               ! So, here: e-e 
               ! Select general-e species, if available, for Te and ne values:
               if(kk.eq.kelecg)then
                 gama(kk,k)= 23.5 -log(sqrt(dens_kk) *t_kk_ev**(-1.25))
     &                      -sqrt(1d-5 +(1./16.)*(log(t_kk_ev)-2.)**2)
               elseif(k.eq.kelecg)then
                 gama(kk,k)= 23.5 -log(sqrt(dens_k) *t_k_ev**(-1.25))
     &                      -sqrt(1d-5 +(1./16.)*(log(t_k_ev)-2.)**2)
               else !Both kk and k are Maxwellian e (should be k=kk here)
                 gama(kk,k)= 23.5 -log(sqrt(dens_k) *t_k_ev**(-1.25))
     &                      -sqrt(1d-5 +(1./16.)*(log(t_k_ev)-2.)**2)
               endif
             else
               ! k=ion species (and kk=electrons)
               ! So, here: e-i
               te_ev= t_kk_ev
               dense= dens_kk
               ti_ev= t_k_ev
               densi= dens_k
               zi= bnumb(k) ! Z of this ion
               zi2= zi*zi
               fmemi= fmass(kk)/fmass(k) ! me/mi ratio
               fmimp= fmass(k)/proton    ! mi/m_proton ratio
               if(te_ev.lt.ti_ev*fmemi)then  !very low Te
                gama(kk,k)=16.d0 -log(sqrt(densi)/ti_ev**1.5 *zi2*fmimp)
               elseif(te_ev.lt.10*zi2)then   !medium Te
                gama(kk,k)=23.d0 -log(sqrt(dense)/te_ev**1.5 *zi)  
               else !(te_ev .gt. 10*zi**2)
                gama(kk,k)=24.d0 -log(sqrt(dense)/te_ev) !high Te range
               endif
             endif ! k species
          else 
             ! kk=ion species
             if(k.eq.kelecg .or. k.eq.kelecm)then
               ! k=electrons (general or Maxwellian) (and kk=ions)
               ! So, here: i-e
               te_ev= t_k_ev
               dense= dens_k
               ti_ev= t_kk_ev
               densi= dens_kk
               zi= bnumb(kk) ! Z of this ion
               zi2= zi*zi
               fmemi= fmass(k)/fmass(kk)  ! me/mi ratio
               fmimp= fmass(kk)/proton    ! mi/m_proton ratio
               if(te_ev.lt.ti_ev*fmemi)then  !very low Te
                gama(kk,k)=16.d0 -log(sqrt(densi)/ti_ev**1.5 *zi2*fmimp)
               elseif(te_ev.lt.10*zi2)then   !medium Te
                gama(kk,k)=23.d0 -log(sqrt(dense)/te_ev**1.5 *zi)  
               else !(te_ev .gt. 10*zi**2)
                gama(kk,k)=24.d0 -log(sqrt(dense)/te_ev) !high Te range
               endif
             else
               ! k=ion species
               ! So, here: i-i
               !ti_ev= t_kk_ev
               !densi= dens_kk
               zikk= bnumb(kk) ! Z of this ion
               zikk2= zikk*zikk
               aw_kk= fmass(kk)/proton    ! mi/m_proton ratio
               zik=  bnumb(k)  ! Z of this ion
               zik2= zik*zik
               aw_k=  fmass(k)/proton     ! mi/m_proton ratio
               zztt= zikk*zik*(aw_kk+aw_k)/(aw_kk*t_k_ev+aw_k*t_kk_ev)
               gama(kk,k)= 23.d0 
     &        -log(zztt*sqrt(dens_kk*zikk2/t_kk_ev +dens_k*zik2/t_k_ev))
             endif ! k species
          endif ! kk species
        enddo ! k=1,ntotal
        enddo ! kk=1,ntotal
        goto 99 !-> Continue,  setup gamefac()
      endif ! (gamaset.lt.zero)
      

      !The rest, before 99_continue is for gamaset.eq.zero, 
      ! i.e. compute gama() internally, based on Killeen book, Eq.(2.1.5). 
      energ=ergtkev*energy(kelec,lr_)
      dense=reden(kelec,lr_)
      if (cqlpmod.eq."enabled") then
        energ=ergtkev*enrgypa(kelec,ls_)
        dense=denpar(kelec,ls_)
      endif

c..................................................................
c     If two electron species exist - do an average for deby calc.
c..................................................................

      if (colmodl.eq.1 .or. (colmodl.eq.3 .and. kelecm.ne.0)) then
        dense=reden(kelecm,lr_)
        energ=reden(kelecm,lr_)*energy(kelecm,lr_)*ergtkev/dense
        if (cqlpmod.eq."enabled") then
          dense=denpar(kelecm,ls_)
          energ=enrgypa(kelecm,ls_)*ergtkev
        endif

      elseif (kelecg.gt.1 .and. kelecm.gt. 1) then
        dense=reden(kelec,lr_)+reden(kelecm,lr_)
        km=kelecm
        energ=(reden(kelecg,lr_)*energy(kelecg,lr_)+reden(km,lr_)*
     1    energy(km,lr_))*ergtkev/dense
        if (cqlpmod.eq."enabled") then
          dense=denpar(kelec,ls_)+denpar(kelecm,ls_)
          energ=(denpar(kelecg,ls_)*enrgypa(kelecg,ls_)+denpar(km,ls_)*
     1      enrgypa(km,ls_))*ergtkev/dense
        endif
      endif
      deby=sqrt(energ/(dense*6.*pi*charge**2)) !YuP: Should we add Zeff here?
      do i=1,ntotal
        si=energy(i,lr_)/fmass(i)
        if (cqlpmod.eq."enabled") si=enrgypa(i,ls_)/fmass(i)
        do k=1,ntotal
          sk=energy(k,lr_)/fmass(k)
          if (cqlpmod.eq."enabled") sk=enrgypa(k,ls_)/fmass(k)
c990131          sf=amax1(si,sk)
          sf=max(si,sk)
          vikf=sqrt(sf*ergtkev*2.)
          gam3=gamt(i,k)*deby*vikf
c990131          gama(i,k)=(alog(gam3)-.5)
          gama(i,k)=(log(gam3)-.5)
        enddo ! k=1,ntotal
      enddo ! i=1,ntotal


 99   continue
 
CMPIINSERT_IF_RANK_EQ_0
!      if((n.eq.0).and.(lr_.eq.1 .or. lr_.eq.lrz))then
!      WRITE(*,*)'n,lr_=',n,lr_
!      do ktot1=1,ntotal
!      do ktot2=1,ntotal
!        WRITE(*,*)'k1,k2, ln(Lambda)==gama(k1,k2)=', 
!     +             ktot1,ktot2,  gama(ktot1,ktot2)
!        !YuP[2020-10-20] This is also printed from tdoutput,
!        !but only at the end of run. Commenting out.
!      enddo
!      enddo
!      endif
CMPIINSERT_ENDIF_RANK

c..................................................................
c     If kelecg.eq.1 and gamafac.eq."enabled", set energy
c       dependent factor gamefac(j,k) for Coulomb logarithm. 
c     The energy dependent Coulomb log will be
c       alog(flamcql + min(sqrt(gamma-1),1)*(flamrp-flamcql)),
c       where flamcql is the argument of cql e-e Coulomb log,
c       and flamrp is the argument of the Rosenbluth-Putvinski
c       (Nucl. Fus. 1997) high energy electron Coulomb log.
c       The sqrt(gamma-1)-factor is chosen in accord with
c       the energy factor in the the CQL Coulomb log
c       (cf., Killeen et al. book).  The CQL Coulomb log
c       reduces to the NRL, Te.gt.10eV expression.  
c     Some further incites into this factor can be obtained
c       from Physics Vade Mecum, H.L. Anderson, Ed.
c       2nd edition, AIP, NY (1989); Sect. 16.07.F, "energy
c       loss due to scattering from atomic electrons ... given
c       by Moller scattering, with I=(electronic charge)**2/(
c       Debye length) [Rosenbluth, personal comm. 1996].
c       (BH, 980505)  
c..................................................................

      do 14 j=1,jx
         !Set default values (1.0 means no energy-dependence in Coulomb log)
         gamefac(j,:)=1.0 !YuP[2019-07-26] k index added (: is 1:ntotal)
 14   continue

      if (kelecg.eq.1 .and. ngen.eq.1) then
        !YuP[2019-07-26] I think ngen.eq.1 condition could be dropped now.

        if(gamafac.eq."enabled") then !Factor that gives an energy-dependence
         flamcql=exp(gama(kelecg,kelecg))
         if (cqlpmod.eq."enabled") then !YuP[2021-02-24]
          omegape=5.64e4*sqrt(denpar(kelecg,ls_))
         else
          omegape=5.64e4*sqrt(reden(kelecg,lr_))
         endif         
         flamrp1=(2**.25)*fmass(kelecg)*clite2/(1.0546e-27*omegape)
         ! 1.0546e-27 is h^bar (Planck's constant /2pi) in [J*sec] 
         ! adjusted by 1e3*1e4 because m is in [gram]  and c is in [cm/sec]
         do  j=1,jx
            flam=flamcql+min(sqrt(gamma(j)-1.),one)*
     +           (flamrp1*gamma(j)**1.5 - flamcql)
            gamefac1=log(flam)/gama(kelecg,kelecg)
            do k=1,ntotal
               gamefac(j,k)= gamefac1
            !YuP[2019-07-26] k-index is added. Here, gamefac(j,k) are same 
            ! for all k, but in case of gamafac.eq."hesslow" (see below)
            ! they are different for e-on-e and e-on-i
            enddo
         enddo
        endif ! gamafac.eq."enabled"
! YuP[2019-07-25] Comments on the above definition:         
!    The gamefac() factor in CQL3D almost matches formula in the paper 
!    by Rosenbluth-Putvinski  (Nucl. Fus. 1997, see ln(Lambda) in Section 3),
!    except here gamefac() has a small correction at low energies.
!    The formula (on p.1358) is 
!     ln(Lambda) = ln{ 2^0.25 *m_e*c^2 *(1+p^2)^(3/4) /[(h/2pi)*Omega_pe] }
!     where p is the normalized momentum, so that 1+p^2 = gamma^2.
!    Specifically, if gamma>2, then the dependence is same as in the paper
!    ( ~ln(gamma**(3/2)), which is  ln[(1+p**2)**(3/4)] ).
!    But if gamma<2, then there is additional factor of sqrt(gamma-1),
!    so that gamefac ~ ln[ sqrt(gamma-1) * gamma**(3/2)].
!    Maybe it is done to "push" this factor to 1.0 when gamma=1 (non-relativ. limit).
!    Note that in the above
!      flam = L_ee + min(sqrt(gamma-1),1)*[A*gamma^1.5  -L_ee]
!    and
!      gamefac= ln(flam) / ln(L_ee)
!    So, when gamma->1, then flam=L_ee, and then gamefac=1.        
!    However, this formula in the paper is not consistent 
!    with a statement two lines down in the paper, which says 
!    "In practice, we set the Coulomb logarithm so that 
!      ln(Lambda) = 18 + ln(p). "      [for p=1000, it gives ln(Lambda)=25 ]
!    From the formula, at large p, we can get
!      ln(Lambda) = 21.6 + (3/2)*ln(p) [for p=1000, it gives ln(Lambda)=32 ]

        !--- New Version, following Hesslow et al, JPP-2018, Eqs.(2.10) 
        ! YuP[2019-07-25]
        !Note that gamefac(j,k) is used next to gama(kk,k),
        !where kk is for the "a" species (general) species,
        !and k is for the "b" species - either Maxwellian species
        ![see cfpcoefn or cfpcoefr, "anr1=..." at lines~200]
        !or general species (usually same as "a", or could be another general species)
        ![see cfpcoefn, lines ~646-761, inside do 490 loop].
        !Factor gamefac() is used when "a" is electrons(general species),
        !while "b" species could be ion(Maxwellian or general)
        !or it could be electron (general or Maxwellian).
        !All these cases are considered below, for the k values.
        if(gamafac.eq."hesslow") then
          !Define norm-ed momentum, corresp. to thermal electrons:
         if (cqlpmod.eq."enabled") then !YuP[2021-02-24]
          temp_kelecg=temppar(kelecg,ls_)
         else
          temp_kelecg=temp(kelecg,lr_)
         endif
          p_Te2= 2.d0*temp_kelecg/restmkev
          p_Te = sqrt(2.d0*temp_kelecg/restmkev) 
          ! Could use temp(kelecm), if kelecm>0 ???
          ! Note: restmkev = 510.998902d0 keV
          !-1-> Consider interaction of e(general) with e(general)
          k=kelecg ! "b" species
          gama_ee=gama(kelecg,k) != ln(L_ee) Coulomb log for e-on-e(general)
          do j=1,jx
            p_n = x(j)*vnorm/clight ! norm-ed momentum
            p_dep= (2*gamma(j)-2.d0)/p_Te2 ! p-dependent term
            gamefac(j,k)= 1.d0 + 0.2*log(1.d0 + p_dep**2.5)/gama_ee
            ! Note that when p-->0, then gamma=1 and then gamefac=1
          enddo
          !-2-> Consider interaction of e(general) with e(Maxw,if any)
          if(kelecm.ne.0)then
          k=kelecm ! "b" species
          gama_ee=gama(kelecg,k) != ln(L_ee) Coulomb log for e-on-e(Maxw)
          do j=1,jx
            p_n = x(j)*vnorm/clight ! norm-ed momentum
            p_dep= (2*gamma(j)-2.d0)/p_Te2 ! p-dependent term
            gamefac(j,k)= 1.d0 + 0.2*log(1.d0 + p_dep**2.5)/gama_ee
            ! Note that when p-->0, then gamma=1 and then gamefac=1
          enddo
          endif
          !-3-> Consider interaction of e with Maxwellian ion
          do k_b=1,nionm ! If no Maxw ion is present, nionm=0
             k=kionm(k_b) ! "b" species
             gama_ei=gama(kelecg,k) !Coulomb log for e-on-ion(Maxwellian)
             do j=1,jx
               p_n= x(j)*vnorm/clight ! norm-ed momentum
               p_dep= (2*p_n/p_Te) ! p-dependent term
               gamefac(j,k)= 1.d0 + 0.2*log(1.d0 + p_dep**5)/gama_ei
             enddo
          enddo ! k_b
          !-4-> Consider interaction of e with general ion (if none, niong=0)
          do k_b=1,niong ! If no general ion is present, niong=0
             k=kiong(k_b) ! "b" species
             gama_ei=gama(kelecg,k) !Coulomb log for e-on-ion(general)
             do j=1,jx
               p_n= x(j)*vnorm/clight ! norm-ed momentum
               p_dep= (2*p_n/p_Te) ! p-dependent term
               gamefac(j,k)= 1.d0 + 0.2*log(1.d0 + p_dep**5)/gama_ei
             enddo
          enddo ! k_b
          !Note: Cases 1 and 2 could be combined, 
          !and also cases 3 and 4 could be combined,
          !but we separated them for clarity of logic, 
          !and to reduce the usage of if() statements.
          !Note: These equations (following Hesslow) yield 
          !a different trend comparing to equations in gamafac="enabled"
          !section. In the latter case, the gamefac() provides a growth 
          !of Coulomb log with p_n at small values of p_n already, 
          !while in Hesslow version, the growth starts at p_n~5e-2.
          !At p_n~1e3, Hesslow equations give the values of Coulomb log
          ![i.e. gama()*gamefac() value] equal to 22 (for ee) or 26 (for ei),
          !while the original equation (gamafac="enabled")
          !gives values of 32 (for ee) and 33 (for ei) 
          ![the difference of 33-32 is from me/2 vs me factor,
          ! see comments on gamt() in sub.ainvnorm].
        endif ! gamafac.eq."hesslow"

	endif ! kelecg.eq.1 .and. ngen.eq.1     


      return
      end
