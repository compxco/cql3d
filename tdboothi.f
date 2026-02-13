!
!
      subroutine tdboothi
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      parameter(itlrza=3*lrza+1)
      include 'comm.h'
CMPIINSERT_INCLUDE

!     Calculation of Hirshman bootstrap current expression,
!     according to Phys. Fluids 31, p3150-2 (1988),
!     for arbitrary aspect ratio, banana regime, single
!     electron and ion species.
!     The Hirshman expressions are adapted for multiple
!     species in CQL3D by using Zeff and a combined
!     Ti and ion pressure piofr
!
!     bscurm(1:lrza,2,2) refers to bootstrap current density as
!     a function of radius for electrons:ions, thermal:nonthermal.
!     bscurmi(0:lrza,2,2) are radially integrated cumulative values,
!     and bscurma(2,2) are total radially integrated values.
!     (amps/cm**2) and amps.


      dimension workk(itlrza)

!     dimensions for "temp of r", "pres of r", refer to
!     tempofr(radius, e:i, thermal:nonthermal, 0th:1st:2nd deriv):
      dimension tempofr(lrza,2,2,3),presofr(lrza,2,2,3)

!..................................................................
!     This version currently is written for eqmod="enabled" -
!     the circular analogue will be added later. The formula for
!     the current is MKS. So all quantities will be put into MKS
!     for the evaluation of the current which will remain in
!     Amps/cm**2
!..................................................................
!
! --- statement functions (x = trapped particle fraction, z = charge #,
! --- see equations 11a,11b,11c,and equation for D(X) ):
!     (Snatched from ONETWO, 990823).  For jhirsh.eq.88 only.
      a31(xx,zz)  = ((zz+2.21)*zz+0.754)+xx*((zz+1.243)*zz+0.348)
      a32(zz)    =  0.884+2.074*zz
      alphai(xx) = -1.172/(1.0 + 0.462*xx)
!cBobH990823      df(xx,zz)   = (1.414+zz)*zz+(xx*((zz+1.243)*zz+3.48)
      ddf(xx,zz)   = (1.414+zz)*zz+(xx*((zz+1.243)*zz+0.348)
     &         + ((2.0*zz+2.657)*zz+0.754))*xx
!     (End-Snatch).
!
! --- Statement functions for Sauter, Angioni, Lin-Liu in banana limit
!     (O. Sauter, C. Angioni, and Y.R. Lin-Liu, "Neoclassical 
!      conductivity and bootstrap current formulas for general
!      axisymmetric equilibria and arbitrary collisionality regime",
!      Phys. of Plasmas 6, 2834 (1999)).
      sa31(xx,zz)=xx+xx*(1.4+xx*(-1.9+xx*(0.3+0.2*xx)))/(1.+zz) !-> L31

      f32ee(xx,zz)= (0.05+0.62*zz)*(xx-xx**4)/(zz+0.44*zz*zz)
     &            +(xx**2-xx**4-1.2*(xx**3-xx**4))/(1.+0.22*zz)
     &            +1.2*xx**4/(1.+0.5*zz) ! should use xx=ft32ee
     
      f32ei(xx,zz)=-(0.56+1.93*zz)*(xx-xx**4)/(zz+0.44*zz*zz)
     &            +4.95*(xx**2-xx**4-0.55*(xx**3-xx**4))/(1.+2.48*zz)
     &            -1.2*xx**4/(1.+0.5*zz) ! should use xx=ft32ei
      
      ! sa32==L32 in the paper. Not used anymore. 
      ! Instead, use f32ee(ft32ee,zz) + f32ei(ft32ei,zz),
      ! as in Eq.15a of Sauter paper.
      sa32(xx,zz)=(0.05+0.62*zz)*(xx-xx**4)/zz/(1.+0.44*zz)
     &            +(xx**2-xx**4-1.2*(xx**3-xx**4))/(1.+0.22*zz)
     &            -(0.56+1.93*zz)*(xx-xx**4)/zz/(1.+0.44*zz)
     &            +4.95*(xx**2-xx**4-0.55*(xx**3-xx**4))/(1.+2.48*zz)
!    &            +1.2*xx**4/(1.+0.5*zz)
!    &            -1.2*xx**4/(1.+0.5*zz)

      salpha0(xx)=-1.17*(1-xx)/(1.-0.22*xx-0.19*xx**2) !-> alfa0
      ! The functions above are from the referenced paper. 
      ! Note that in zero collisionality limit L34=L31, and alfa=alfa0


      if (eqmod.ne. "enabled") return

!..................................................................
!     Will use splines to determine derivatives of densities and
!     temperatures vs. psi (=psimag-equilpsi)
!..................................................................


      call bcast(bscurm,zero,size(bscurm))
      call bcast(bscurmi,zero,(lrza+1)*4)
      call bcast(bscurma,zero,4)

      if (bootst.ne."enabled" .and. jhirsh.ne.88) return 


      call bcast(tempofr,zero,lrza*12)
      call bcast(presofr,zero,lrza*12)

      i1p(1)=4
      i1p(2)=4

!..................................................................
!     Thermal electron logrithmic derivatives
!     Find profiles.
!     Use coeff1/terp1 to obtain radial derivative.
!..................................................................

      if (kelecm.ne.0) then

      do k=ngen+1,ntotal
      if (k.eq.kelecm) then
        do l=1,lrzmax
           tempofr(l,1,1,1)=temp(k,l)
           presofr(l,1,1,1)=reden(k,l)*temp(k,l)
           tr(l)=psimag-equilpsi(l)
           !pol.flux, in ascending order needed for coeff1 
           ! YuP[2019-12-12] migrated corrections from CQL3D-FOW version	   
        enddo
      endif
      enddo

      call coeff1(lrzmax,tr(1),tempofr(1,1,1,1),tempofr(1,1,1,3),
     &     i1p,1,workk)
      call coeff1(lrzmax,tr(1),presofr(1,1,1,1),presofr(1,1,1,3),
     &     i1p,1,workk)
      
      itab(1)=0
      itab(2)=1
      itab(3)=0
      do l=1,lrzmax
         call terp1(lrzmax,tr(1),tempofr(1,1,1,1),tempofr(1,1,1,3),
     &        tr(l),1,tab,itab)
         tempofr(l,1,1,2)=tab(2)/tempofr(l,1,1,1) ! (dTe/dpsi)/Te (thermal)
         call terp1(lrzmax,tr(1),presofr(1,1,1,1),presofr(1,1,1,3),
     &        tr(l),1,tab,itab)
         presofr(l,1,1,2)=tab(2)/presofr(l,1,1,1) ! (dPe/dpsi)/Pe (thermal)
         !......   test/printout .................................
!         k=kelecm
!         l0=l
!         lp=l+1
!         lp=min(lp,lrz) 
!         lm=l-1
!         lm=max(lm,1) 
!         dp_dpsi= (presofr(lp,1,1,1)-presofr(lm,1,1,1))/(tr(lp)-tr(lm))
!         write(*,*)'tdboothi k=kelecm: dp/dpsi/p=',presofr(l,1,1,2),
!     &    dp_dpsi*2./(presofr(lp,1,1,1)+presofr(lm,1,1,1))
         ! printout: inner points: same result (last digit difference)
         !......   test/printout .................................
      enddo

      endif

!.......................................................................
!     Thermal ion logrithmic derivatives
!     (Use density weighted ion temperature).
!     Find profiles.
!     Use coeff1/terp1 to obtain radial derivative.
!.......................................................................

      if (nionm.ne.0) then

      do k=ngen+1,ntotal
      do kk=1,nionm
         if (k.eq.kionm(kk)) then
            do l=1,lrzmax
               ! sum over all Maxw.ions: 
               tempofr(l,2,1,1)=tempofr(l,2,1,1)+reden(k,l) ! sum(n) 
               presofr(l,2,1,1)=presofr(l,2,1,1)+reden(k,l)*temp(k,l) ! sum(p)
            enddo
         endif
      enddo
      enddo

      do l=1,lrzmax
         ! Average T over all Maxw.ions: sum(p)/sum(n)
         tempofr(l,2,1,1)=presofr(l,2,1,1)/tempofr(l,2,1,1)
         ! YuP: Is it actually a good idea to have everage T over all ions?
         tr(l)=psimag-equilpsi(l)  
      enddo

      call coeff1(lrzmax,tr(1),tempofr(1,2,1,1),tempofr(1,2,1,3),
     &     i1p,1,workk)
      call coeff1(lrzmax,tr(1),presofr(1,2,1,1),presofr(1,2,1,3),
     &     i1p,1,workk)
      
      itab(1)=0
      itab(2)=1
      itab(3)=0
      do l=1,lrzmax
         call terp1(lrzmax,tr(1),tempofr(1,2,1,1),tempofr(1,2,1,3),
     &        tr(l),1,tab,itab)
         tempofr(l,2,1,2)=tab(2)/tempofr(l,2,1,1) ! (dTi/dpsi)/Ti (thermal)
         call terp1(lrzmax,tr(1),presofr(1,2,1,1),presofr(1,2,1,3),
     &        tr(l),1,tab,itab)
         presofr(l,2,1,2)=tab(2)/presofr(l,2,1,1) ! (dPi/dpsi)/Pi (thermal)
      enddo
        
      endif
      

!..................................................................
!     Nonthermal (general species) electron logrithmic derivatives
!     Find profiles.
!     Use coeff1/terp1 to obtain radial derivative.
!..................................................................

      if (kelecg.ne.0) then

      do k=1,ngen
      if (k.eq.kelecg) then
        do l=1,lrzmax
           tempofr(l,1,2,1)=(2./3.)*energy(k,l)
           presofr(l,1,2,1)=reden(k,l)*(2./3.)*energy(k,l)
           tr(l)=psimag-equilpsi(l) 
        enddo
      endif
      enddo

      call coeff1(lrzmax,tr(1),tempofr(1,1,2,1),tempofr(1,1,2,3),
     &     i1p,1,workk)
      call coeff1(lrzmax,tr(1),presofr(1,1,2,1),presofr(1,1,2,3),
     &     i1p,1,workk)
      
      itab(1)=0
      itab(2)=1
      itab(3)=0
      do l=1,lrzmax
         call terp1(lrzmax,tr(1),tempofr(1,1,2,1),tempofr(1,1,2,3),
     &        tr(l),1,tab,itab)
         tempofr(l,1,2,2)=tab(2)/tempofr(l,1,2,1) ! (dTe/dpsi)/Te (nonthermal)
!         !......   test/printout .................................
!         k=kelecg
!         if(l.eq.1)then
!            dt_drho= (temp(k,l+1)-temp(k,l))/(equilpsi(l+1)-equilpsi(l))
!         elseif(l.eq.lrzmax)then
!            dt_drho= (temp(k,l)-temp(k,l-1))/(equilpsi(l)-equilpsi(l-1))
!         else ! inner points:
!            dt_drho= (temp(k,l+1)-temp(k,l-1))/(equilpsi(l+1)-equilpsi(l-1))
!         endif
!         write(*,*)'tdboothi k=kelecg: dT/drho=',tab(2),dt_drho
!         ! printout: inner points: same result (last digit difference)
!         !......   test/printout .................................
         call terp1(lrzmax,tr(1),presofr(1,1,2,1),presofr(1,1,2,3),
     &        tr(l),1,tab,itab)
         presofr(l,1,2,2)=tab(2)/presofr(l,1,2,1) ! (dPe/dpsi)/Pe (nonthermal)
      enddo

      endif

!..................................................................
!     Nonthermal (general species) ion logrithmic derivatives
!     (Use density weighted ion temperature).
!     Find profiles.
!     Use coeff1/terp1 to obtain radial derivative.
!..................................................................

      if (niong.ne.0) then

      do k=1,ngen
      do kk=1,niong
         if (k.eq.kiong(kk)) then
            do l=1,lrzmax
               !densi=den_fsa(k,l) !!changed to <n>FSA [7-18-2014]
               ! The change in bs_i is in 4th-5th digit
               ! (in test for NSTX/scaleb=0.5) 
               ! and the profile is somewhat more coarse (noisy).
               ! so, change back to reden() :
               densi=reden(k,l)
               tempofr(l,2,2,1)=tempofr(l,2,2,1)+densi !sum gen.ions
               presofr(l,2,2,1)=presofr(l,2,2,1)
     &                      +densi*(2./3.)*energy(k,l) !energy() is FSA
            enddo
         endif
      enddo
      enddo

      do l=1,lrzmax
         ! Average over general ions:
         tempofr(l,2,2,1)=presofr(l,2,2,1)/tempofr(l,2,2,1) ! T=P/n
         ! Is it a good idea to "mix-up" all gen.ions?
         ! Maybe better - select the 1st gen.ion only, 
         ! or make calculations for each k-species separately?
         tr(l)=psimag-equilpsi(l) 
         !pol.flux, in ascending order needed for coeff1
      enddo

      call coeff1(lrzmax,tr(1),tempofr(1,2,2,1),tempofr(1,2,2,3),
     &     i1p,1,workk)
      call coeff1(lrzmax,tr(1),presofr(1,2,2,1),presofr(1,2,2,3),
     &     i1p,1,workk)
      
      itab(1)=0
      itab(2)=1
      itab(3)=0
      do l=1,lrzmax
         call terp1(lrzmax,tr(1),tempofr(1,2,2,1),tempofr(1,2,2,3),
     &        tr(l),1,tab,itab)
         tempofr(l,2,2,2)=tab(2)/tempofr(l,2,2,1) ! (dTi/dpsi)/Ti (nonthermal)
         call terp1(lrzmax,tr(1),presofr(1,2,2,1),presofr(1,2,2,3),
     &        tr(l),1,tab,itab)
         presofr(l,2,2,2)=tab(2)/presofr(l,2,2,1) ! (dPi/dpsi)/Pi (nonthermal)
      enddo
        
      endif

!.......................................................................
!     Loop over flux surfaces calculating bootstrap current
!     <j.B>/<|B|> (Amps/cm**2).  where <...> = Integral(...dlp/Bp)
!.......................................................................

      
!     Minus in following should probably be plus.
!     (10**8 webbers/(guass*cm**2)/(10**2 cm/m))*10**6 cm**3/m**3)*
!        10**3 eV/keV * 1.6e-19 joules/ev * 10**-4 m**2/cm**2
!        /10**2 cm/m, for cgs fpsi/bmidplne). = 1.6022e-10
      cnst=-1.6022e-10 ! old!
      
      ! YuP:
      cnst=cursign*1.6022e-8  ! The result will be in A/cm^2
      !YuP[2019-12-17] Added cursign in front.
      !Now the calculated bootstrap current bscurm() is always 
      !in the same direction as the Ohmic current 
      !(I_Ohm>0 corr. to cursign>0, which is in positive phi direction).
      ! Positive phi direction is counter-clockwise viewed from above.
      !Suggestion: We could also add bootsign factor in front,
      !for a more general control, and to match bscurm() with current
      !calculated with bootcalc='method1' or 'method2' option.
      !Also Note: 
      !The bootstrap-deformation of distr.functions of ions and electrons
      ! occurs in opposite way in Vpar axis, 
      ! but because of opposite charge signs, the bs currents
      ! produced by ions and electrons have same sign.

      !----------------- YuP[2019-12-18] get starnue().
      ! The problem is that usually starnue is calc-ed
      ! in sub.efields, but efields is only called when ampfmod.ne."enabled"
      ! or at n=0.
      !As a by-product, also calculate/update tauee(l),taueeh(l),sptzr(l).
      !Tests: Almost no increase in cpu time from this addition.
      call starnue_sptz
      !----------------- YuP[2019-12-18] Got starnue(),tauee(l),taueeh(l),sptzr(l)

      do l=1,lrzmax
         ft= trapfrac(l) !a function; moved here: one call per each l
         ft2= ft*ft
         ft6= ft**6
         zz=zeff(l)
         starnuee= starnue(l)  ! set to 0., for tests
         ! Is starnue(l) same as nue* in Sauter et al ? Yes.
         starnui= starnuee*(4.9/6.921)*zz**3 ! For now maybe ok.
         !(the problem is that starnui should use Z for each ion species,
         ! and here we have zz==Zeff)
         sqnui=  sqrt(starnui)
         sqnue=  sqrt(starnuee)
         ! YuP: For nue* correction, as in Eqs.(13-16), Sauter et al:
         trap_nucorr33= 1. +(0.55-0.1*ft)*sqnue 
     &                     +0.45*(1.-ft)*starnuee/zz**1.5
         trap_nucorr31= 1. +(1.-0.1*ft)*sqnue 
     &                     +0.5*(1.-ft)*starnuee/zz    
         trap_nucorr32ee= 1. +0.26*(1.-ft)*sqnue 
     &                       +0.18*(1.-0.37*ft)*starnuee/sqrt(zz)
         trap_nucorr32ei= 1. +(1+0.6*ft)*sqnue 
     &                       +0.85*(1.-0.37*ft)*starnuee*(1.+zz)
         trap_nucorr34= 1. +(1.-0.1*ft)*sqnue 
     &                     +0.5*(1.-0.5*ft)*starnuee/zz
         ! The above values are denominators in ft33= ft/trap_nucorr33, etc.
         ft33=   ft/trap_nucorr33 ! not used here (in paper, used for sigma)
         ft31=   ft/trap_nucorr31
         ft32ee= ft/trap_nucorr32ee
         ft32ei= ft/trap_nucorr32ei
         ft34=   ft/trap_nucorr34
         ! alpha(nui*)  in Sauter et.al. (Eq.17b)
         salpha=( (salpha0(ft)+0.25*(1.-ft2)*sqnui)/(1.+0.5*sqnui)
     &            +0.315*starnui**2*ft6 ) / (1.+0.15*starnui**2*ft6)  
         ! YuP [2015-07-07] In above: Corrected -0.315 to +0.315
         ! according to Erratum in Phys.Plasmas 2002,v.9,p.5140
         l_=l
         call eqflxavg_lz(l_,bbpsi(1,l_),bb0_aver,sum_a_ds_bp)
         !output: bb0_aver=<B/B0>= sum(bbpsi*ds/Bp)/sum(ds/Bp)
         !write(*,*) l,rya(l),bb0_aver,(psiavg(1,l)-bb0_aver)/psiavg(1,l)

!        k refers to electrons or ions.  kk to thermal or nonthermal.
         do kk=1,2

!cyup         fj0oB=-fpsi(l)*presofr(l,1,kk,1)*cnst/(psiavg(1,l)*bmidplne(l))
         ! YuP: rearranged to avoid pe/pe = 0/0 problem when e_gen are not present:
         fj0oB=-fpsi(l)*cnst/(bb0_aver*bmidplne(l))
         pe_kk= presofr(l,1,kk,1) ! pressure of e_maxw, or e_gen
         pi_kk= presofr(l,2,kk,1) ! pressure of i_maxw, or i_gen
CMPIINSERT_IF_RANK_EQ_0
!         WRITE(*,'(a,2i4,4e14.7)')
!     &   'tdboothi: lr,kk, starnuee, starnui, pe_kk,pi_kk',
!     &              l, kk, starnuee, starnui, pe_kk,pi_kk
CMPIINSERT_ENDIF_RANK
         
         if (jhirsh.eq.88) then

            xx=ft/(1.-ft)
 
            !tmp1=a31(xx,zz)/ddf(xx,zz)
            !tmp2=a32(zz)/ddf(xx,zz)

            k=1 ! electrons
            bscurm(l,k,kk)=fj0oB*xx*pe_kk*
     &                     ( a31(xx,zz)*presofr(l,k,kk,2)
     &                      -a32(zz)*tempofr(l,k,kk,2) )/ddf(xx,zz)

            k=2 ! ions
            bscurm(l,k,kk)=fj0oB*xx*pe_kk*a31(xx,zz)/ddf(xx,zz)*
     &                     tempofr(l,k,kk,1)/(tempofr(l,1,kk,1)*zz)*
     &                 (presofr(l,k,kk,2)+alphai(xx)*tempofr(l,k,kk,2))

         elseif(jhirsh.eq.99) then

            !tmp1=sa31(ft,zz)/ft
            !tmp2=sa32(ft,zz)/ft

            k=1 ! electrons
            bscurm(l,k,kk)=fj0oB*pe_kk*(sa31(ft31,zz)*presofr(l,k,kk,2)
     &           +(f32ee(ft32ee,zz)+f32ei(ft32ei,zz))*tempofr(l,k,kk,2))

            k=2 ! ions
            bscurm(l,k,kk)=fj0oB*(sa31(ft31,zz)*pi_kk*presofr(l,k,kk,2)
     &                  +0*salpha*sa31(ft34,zz)*pi_kk*tempofr(l,k,kk,2))
         ! YuP [2015-07-07] In above: Corrected Pe->Pi (ions), in 2nd term
         ! according to Erratum in Phys.Plasmas 2002,v.9,p.5140.
         ! There was a missing factor next to L34:
         ! (1-Rpe)/Rpe where usually Rpe=Pe/P, so that
         ! (1-Rpe)/Rpe = P/Pe -1 = Pi/Pe
         ! This factor should be next to (salpha*pe_kk), so that
         ! (salpha*pe_kk) -> (salpha*pe_kk)*Pi/Pe == (salpha*pi_kk)
         ! At 1st glance, there should be no change in result if Pe=Pi.
         ! But, in older runs, the value of bscurm for gen.ions 
         ! was using pe_kk for kk=2(general e, which are absent), 
         ! so that pe_kk was 0.d0 for kk=2.
         ! Now, after correction, pi_kk is not zero for kk=2.
         ! As a result, the value of bscurm for gen.ions is reduced.
         ! Is this term (with salpha) - a screening by electrons?
         ! If - yes, it's better to set it to zero, for comparison
         ! with "fi" current based on solution of FPE for ions
         ! that does not include a screening/entrainment by electrons.
         ! Or, alternatively, leave this term present,
         ! and then compare "bs_i"==bscurm with the "fi+e";
         ! "fi+e" includes a screening current, by Cordey-Start model
         ! (which may not be accurate).
         endif


         enddo ! kk=gen,maxw

      enddo ! lr

CMPIINSERT_IF_RANK_EQ_0
!      WRITE(*,*)' lr, Jbs_e_maxw, Jbs_e_gen,  Jbs_i_maxw, Jbs_i_gen '
!      do l=1,lrzmax
!         WRITE(*,'(i4,4e12.4)') l, bscurm(l,1,1),bscurm(l,1,2),
!     &                             bscurm(l,2,1),bscurm(l,2,2)
!      enddo
!      WRITE(*,*)'time step=',n
!      WRITE(*,*)'radcoord=',radcoord
!      WRITE(*,*)'kelecm,kelecg,nionm,niong=',kelecm,kelecg,nionm,niong
      !pause
CMPIINSERT_ENDIF_RANK

      return
      end subroutine tdboothi
      
!=======================================================================
!=======================================================================
      subroutine starnue_sptz 
      !YuP[2019-12-26] Made it as a separate subroutine.
      !Compute starnue(), tauee(), taueeh(), sptzr()
      ! Copied this part from sub.fields,
      ! The problem is that usually starnue is calc-ed in sub.efields,
      ! but efields is only called when ampfmod.ne."enabled"
      ! or at n=0.
      !Tests: Almost no increase in cpu time from this addition.
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
      
      if(cqlpmod.ne."enabled")then
       do l=1,lrzmax ! tauee,taueeh,starnue,vth are over full grid 
         call tdnflxs(lmdpln(l)) ! determine l_,lr_, etc.
         call cfpgamma !YuP[2020-01-07]Added, To get gama(k,kk) for each lr_
         !vth is the thermal velocity = sqrt(T/m)
         vth(kelec,l)=(temp(kelec,l)*ergtkev/fmass(kelec))**.5
         !tauee(l) is the electron-electron slowing down time at midplane
         !tauee = 0.532/nu^{ee}_s, where nu^{ee}_s is the low velocity
         !expression for ee-slowing in Book's NRL Plasma Formulary p.32,
         !0.532=2**(3/2)/(3*pi**0.5). As nu(ee)_s = 2/taueeh/Zeff, we
         !have tauee = 2**(3/2)/3/sqrt(pi)/2*taueeh*Zeff, thus
         !taueeh = (3*sqrt(pi/2)/Zeff)*tauee. 
         !Note taueeh = taue(NRL p.33)=2.9E-06*...
          tauee(l)=vth(kelec,l)**3*fmass(kelec)**2
     &     /(4.*pi*reden(kelec,l)*charge**4*gama(kelec,kelec))
         !Hinton-Hazeltine definition, Eq.(5.4), for e-i collisions.
         !1./taueeh = 4/3 sqrt(2*pi) ne*zeff*charge**4*gamma/(sqrt(me)*Te**1.5)
         !Thus: taueeh=3 sqrt(pi/2) / Zeff * tauee
         !Note: taueeh and nuestar same as ONETWO 4.2-20 and 4.2-30
         !(Eq. 4.2-28 and 4.2-39, in GA-A16178, Pfeiffer, Davidson, Miller, Waltz)
           taueeh(l)=tauee(l)*3.*sqrt(pi/2.)/zeff(l)
           starnue(l)=rgeom(l)*bmod0(l)/bthr0(l)/
     &            vth(kelec,l)/taueeh(l)/eps(l)**1.5
           !formula should be 0.5064*N(Z)*me/taueeh/ne/e**2, with N(Z=1)=1
           !Thus one should use game(e,e) as in taueeh, but as it is not
           !clear which to use we take the average between game(e,e) and game(e,i)
           !This expression for sptzr is given by Eq. 4.2-77 in ONETWO manual,
           !GA-A16178, the same as Eq. 5.66 of Hinton-Hazeltine, Rev. Mod. Phys.
       enddo ! l=1,lrzmax 
       do ll=1,lrors ! sptzr is not over full grid, it is over FPE grid
          call tdnflxs(ll)
          call cfpgamma !To get gama(k,kk) for each ll
          sptzr(ll)=zeff(lrindx(ll))*(0.29+0.46/(1.08+zeff(lrindx(ll))))
     &       *4.*sqrt(2.*pi)/3.*charge**2
     &       *0.5*(gama(kelec,kelec)+gama(kelec,kionn))
     &       *sqrt(fmass(kelec))/(temp(kelec,lrindx(ll))*ergtkev)**1.5
          write(*,'(a,2i3,2f7.2,1pe12.3)')
     &    'starnue_sptz: n,lr,Lee,Zeff,sptzr',
     &     n,ll,gama(kelec,kelec),zeff(lrindx(ll)),sptzr(ll)
       enddo ! l=1,lrzmax 
      !----------------- YuP[2019-12-18] Got starnue(),tauee(l),taueeh(l),sptzr(l)

      else !(cqlpmod.eq."enabled"): !YuP[2021-02-24] Added a separate branch
           ! change temp()-->temppar();  reden()--> denpar(), vth()-->vthpar()
       do l=1,lsmax ! tauee,taueeh,starnue,vthpar are over full grid 
         call tdnflxs(lmdpln(l)) !??? NEED TO CHECK
         call cfpgamma !YuP[2020-01-07]Added, To get gama(k,kk) for each ls_
         !vth is the thermal velocity = sqrt(T/m)
         vthpar(kelec,l)=(temppar(kelec,l)*ergtkev/fmass(kelec))**.5
         !tauee(l) is the electron-electron slowing down time at midplane
         !tauee = 0.532/nu^{ee}_s, where nu^{ee}_s is the low velocity
         !expression for ee-slowing in Book's NRL Plasma Formulary p.32,
         !0.532=2**(3/2)/(3*pi**0.5). As nu(ee)_s = 2/taueeh/Zeff, we
         !have tauee = 2**(3/2)/3/sqrt(pi)/2*taueeh*Zeff, thus
         !taueeh = (3*sqrt(pi/2)/Zeff)*tauee. 
         !Note taueeh = taue(NRL p.33)=2.9E-06*...
          tauee(l)=vthpar(kelec,l)**3*fmass(kelec)**2
     &     /(4.*pi*denpar(kelec,l)*charge**4*gama(kelec,kelec))
         !Hinton-Hazeltine definition, Eq.(5.4), for e-i collisions.
         !1./taueeh = 4/3 sqrt(2*pi) ne*zeff*charge**4*gamma/(sqrt(me)*Te**1.5)
         !Thus: taueeh=3 sqrt(pi/2) / Zeff * tauee
         !Note: taueeh and nuestar same as ONETWO 4.2-20 and 4.2-30
         !(Eq. 4.2-28 and 4.2-39, in GA-A16178, Pfeiffer, Davidson, Miller, Waltz)
           taueeh(l)=tauee(l)*3.*sqrt(pi/2.)/zeff(l)
           starnue(l)=rgeom(lrindx(l))*bmod0(lrindx(l))/bthr0(lrindx(l))
     &            /vthpar(kelec,l)/taueeh(l)/eps(lrindx(l))**1.5
         !---- YuP: In case of CQLP, lrz=1, 
         !---- so we use rgeom(lrz),bmod0(lrz),bthr0(lrz),eps(lrz)
           !formula should be 0.5064*N(Z)*me/taueeh/ne/e**2, with N(Z=1)=1
           !Thus one should use game(e,e) as in taueeh, but as it is not
           !clear which to use we take the average between game(e,e) and game(e,i)
           !This expression for sptzr is given by Eq. 4.2-77 in ONETWO manual,
           !GA-A16178, the same as Eq. 5.66 of Hinton-Hazeltine, Rev. Mod. Phys.
c           write(*,'(a,i3,1p2e12.3)')
c     &       ' starnue_sptz: l,gama(e,e),starnue(l)=',
c     &                       l,gama(kelec,kelec),starnue(l)
       enddo ! l=1,lsmax 
       do ll=1,lrors ! sptzr is not over full grid, it is over FPE grid
          call tdnflxs(ll)
          call cfpgamma !To get gama(k,kk) for each ll
          sptzr(ll)=zeff(lsindx(ll))*(0.29+0.46/(1.08+zeff(lsindx(ll))))
     &     *4.*sqrt(2.*pi)/3.*charge**2
     &     *0.5*(gama(kelec,kelec)+gama(kelec,kionn))
     &     *sqrt(fmass(kelec))/(temppar(kelec,lsindx(ll))*ergtkev)**1.5
       enddo ! l=1,lrors
      endif ! (cqlpmod.eq."enabled")
      
      return
      end subroutine starnue_sptz


!=======================================================================
!=======================================================================
      subroutine sptz(tempe,tempi, redene,redeni, fmassi, zeff,
     &                rbb,eps,trapfrac,  sptzr,pressau1,pressau2)
      !YuP[2021-02-08] Spitzer resistivity [cgs] sptzr
      ! and Sauter corrections pressau1,pressau2 
      ! for a given Te, Ti, ne,ni, Zeff;  rbb,eps,trapfrac.
      !
      !INPUT (no comm.h here, only through argument list): 
      ! tempe = Te [keV]  electron temp.
      ! tempi = Ti [keV]  ion temp. (needed for Coulomb log definition)
      ! redene= electron density [cm^-3]
      ! redeni= main-ion density [cm^-3] (for Coulomb log definition)
      ! fmassi= main-ion mass [g]   (needed for Coulomb log definition)
      ! zeff  = Zeff
      ! rbb== rgeom*(bmod0/bthr0) == r*B0/Bpol0  
      !       where r= minor radius [cm], 
      !       B0= total magn.field at outboard-midplane,
      !       Bpol0= poloidal magnetic field at outboard-midplane. 
      !       see corresponding arrays rgeom(lr),bmod0(lr),bthr0(lr)
      !       Note: rgeom(lr) = 0.5*(rpcon(lr) - rmcon(lr))
      !       is minor radius [cm], where 
      !       rpcon(lr)= major R at outboard-midplane, for surface lr,
      !       rmcon(lr)= major R at  inboard-midplane, for surface lr.
      !       The combination rgeom*(bmod0/bthr0) is a slowly changing
      !       function of minor radius (it is part of safety factor).
      ! eps= inverse aspect ratio r/R. See corresponding array eps(lr)
      ! trapfrac= trapped particle fraction (see corr.array trapfrac(lr))
      !(see e.g., Hirshman and Sigmar, Nucl.Fus.21,1079 (1981),Eq.4.54)
      ! Also see relevant paper 
      ! O. Sauter, C. Angioni and Y.R. Lin-Liu, Phys. of Plasmas 6, 
      !        2834 (1999).
      !BH210210: It would be good to give an approximate trapfrac.
      !BH210210: From ONETWO manual, Eqs 4.2-47 to 4.2-63, and
      !BH210210: assuming elliptical flux surfaces, one can get
      !BH210210: relatively simple expressions.  
      !BH210210: Approx result is f_trap=sqrt(1-Bmin/Bmax)
      !YuP: From printout for DIII-D equilibrium:
      !eps, sqrt(1.-Bmin/Bmax), trapfrac(exact) = 8.078E-03 1.266E-01 1.302E-01
      !eps, sqrt(1.-Bmin/Bmax), trapfrac(exact) = 1.053E-01 4.351E-01 4.512E-01
      !eps, sqrt(1.-Bmin/Bmax), trapfrac(exact) = 2.042E-01 5.785E-01 5.931E-01
      !eps, sqrt(1.-Bmin/Bmax), trapfrac(exact) = 3.599E-01 7.239E-01 6.733E-01
      !The 1st line is for plasma center, the last line is for plasma edge
      !
      !OUTPUT [cgs units]:
      ! sptzr = Spitzer resistivity, with dep. on Zeff
      ! pressau1= correction factor (as in Sauter1, starnue=0) low-coll. neoclassical
      ! pressau2= correction factor (as in Sauter2, starnue>0) with collisionality
      !     so that 
      !     sig_starnue0= 1/(pressau1*sptzr) ! sigma_nocollis (no-coll. neoclassical) [cgs]
      !     sig_starnue=  1/(pressau2*sptzr) ! sigma_collis  [cgs]
      ! What we probably need for NIMROD: 
      ! eta= sptzr*pressau2*9.e9 !Resistivity converted to [SI], 
      !      with neoclass.correction, collisional version 
      ! (see the end of this subr. on pressau1 and pressau2) 
      implicit none
      
      real*8 tempe,tempi,redene,redeni,fmassi,zeff,rbb,eps,trapfrac !INPUT
      real*8 sptzr,pressau1,pressau2 ! OUTPUT
      real*8 te_ev,ti_ev, vthe, taueeh, fmemi,fmimp, gamaee, gamaei  !local
      real*8 fmasse,proton,charge,ergtkev,pi !local
      real*8 const1,const2,zk033,za33,zb33,zc33,zg,zfcol,starnue !local
      real*8 starnue_s !local, for test
      
      fmasse= 9.1095d-28     !(e) [g]
      proton= 1.67262158d-24 !    [g] 
      charge= 4.8032d-10 !(e) [StatCoulomb]
      ergtkev=1.6022d-09 !energy(ergs) associated with 1 keV
      pi=3.141592653589793d0
      
      ! For Coulomb log definition, 
      ! use expressions from NRL_FORMULARY_2016, p.34
      ! (Be sure to set gamaset=-1. in cqlinput, which means: Use NRL definition)
      te_ev= tempe*1.d3 ! in eV
      ti_ev= tempi*1.d3 ! in eV
      fmemi= fmasse/fmassi ! me/mi ratio
      fmimp= fmassi/proton ! mi/m_proton ratio
      
      ! Thermal e-e collisions (no energy-dependence here):
      gamaee= 23.5 -log(sqrt(redene) * te_ev**(-1.25))  
     &             -sqrt( 1d-5  +(1./16.)*(log(te_ev)-2.)**2 )
      ! For e-i collisions:
      if(te_ev .lt. ti_ev*fmemi)then  !very low Te
         gamaei=16.d0 -log(sqrt(redeni)/ti_ev**1.5 *zeff**2*fmimp)
      elseif(te_ev .lt. 10*zeff**2)then   !medium Te
         gamaei=23.d0 -log(sqrt(redene)/te_ev**1.5 *zeff)  
      else !(te_ev .gt. 10*zeff**2)
         gamaei=24.d0 -log(sqrt(redene)/te_ev) !high Te range
      endif
      !write(*,*)'zeff,CL=',zeff,0.5*(gamaee+gamaei)
      
      !vth is the thermal velocity = sqrt(T/m), T is in keV
      vthe=  sqrt(tempe*ergtkev/fmasse)
      !tauee(l) is the electron-electron slowing down time at midplane
      !tauee = 0.532/nu^{ee}_s, where nu^{ee}_s is the low velocity
      !expression for ee-slowing in Book's NRL Plasma Formulary p.32,
      !0.532=2**(3/2)/(3*pi**0.5). As nu(ee)_s = 2/taueeh/Zeff, we
      !have tauee = 2**(3/2)/3/sqrt(pi)/2*taueeh*Zeff, thus
      !taueeh = (3*sqrt(pi/2)/Zeff)*tauee. 
      !Note taueeh = taue(NRL p.33)=2.9E-06*...
      taueeh= 3.*sqrt(pi/2.)*vthe**3*fmasse**2 / 
     &        (zeff*4.*pi*redene*charge**4*gamaee)
      !Hinton-Hazeltine definition, Eq.(5.4), for e-i collisions.
      !1./taueeh = 4/3 sqrt(2*pi) ne*zeff*charge**4*gamma/(sqrt(me)*Te**1.5)
      !Thus: taueeh=3 sqrt(pi/2) / Zeff * tauee
      !Note: taueeh and nuestar same as ONETWO 4.2-20 and 4.2-30
      !(Eq. 4.2-28 and 4.2-39, in GA-A16178, Pfeiffer, Davidson, Miller, Waltz)
      ! starnue(l)=rgeom(l)*bmod0(l)/bthr0(l)/vth(kelec,l)/taueeh(l)/eps(l)**1.5
      !formula should be 0.5064*N(Z)*me/taueeh/ne/e**2, with N(Z=1)=1
      !Thus one should use gamaee as in taueeh, but as it is not
      !clear which to use we take the average between gamaee and gamaei
      !This expression for sptzr is given by Eq. 4.2-77 in ONETWO manual,
      !GA-A16178, the same as Eq. 5.66 of Hinton-Hazeltine, Rev. Mod. Phys.
      sptzr= zeff*(0.29+0.46/(1.08+zeff))
     &      * (4.*sqrt(2.*pi)/3.) * charge**2
     &      * 0.5*(gamaee+gamaei)*sqrt(fmasse)/(tempe*ergtkev)**1.5 !OUTPUT
     
      !Now define corrections to Spitzer formula.
      
      !Based on subr.resthks() :
      zk033= 1.46+(1.83-1.46)/zeff ! BH210210: See ONETWO manual,
      !      Eq 4.2-49, which reduces to this formula.
      const1= (4.d0/3.d0)*(1.d0 -1.d0/zeff)
      const2= (1.d0/3.d0)*(4.d0/zeff -1.d0)
      za33= const1*0.47 + const2*0.68   !Table 4.2-2 of ONETWO Manual.
      zb33= const1*0.20 + const2*0.32
      zc33= const1*0.51 + const2*0.66 
      ! za33,zb33,zc33 are only needed for pressau2
      
      !Using 1.-xconn = trapfrac
      zg= trapfrac/(1.d0-trapfrac) !==(1.-xconn)/xconn

      starnue= rbb/(vthe*taueeh*eps**1.5) !only needed for pressau2
      !where rbb== rgeom*(bmod0/bthr0)
      
      !TEST:  From Sauter-1999 paper, under Eq.12:
      !starnue_s= 0.012*ne20*Zeff*qR/(Tkev^2 * eps^1.5)
      ! I guess R is in meters.
!      starnue_s= 
!     &    0.012*(redene*1.d-14)*zeff*(rbb/100)/(tempe**2 *eps**1.5)
!      write(*,'(a,1p4e11.3)')'eps, tempe, starnue, starnue_s= ',
!     & eps, tempe, starnue, starnue_s
      !From this test: When starnue and starnue_s are small (0.2--10),
      !they match well. When they are large (starnue~8000 [at low Te]),
      ! the value of starnue_s is ~2x larger (~17000). 
      ! Probably because of definition of Coulomb log.

      zfcol= zk033/(1.d0+za33*sqrt(starnue)+zb33*starnue)
     &            /(1.d0+zc33*starnue*eps**1.5)

      pressau1= (1.d0 + 1.24*zg) !OUTPUT: Collisionless limit

      pressau2= (1.d0 + 1.24*zg*(zfcol/zk033)) !OUTPUT: collisional
      !Note that pressau2 becomes pressau1 if starnue is set to 0.
      
      return
      end subroutine sptz
