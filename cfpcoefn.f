c
c
      subroutine cfpcoefn
      implicit integer (i-n), real*8 (a-h,o-z)
      save
ccc      real*8,dimension(iy):: prnt1,prnt2,prnt3
ccc      real*8,dimension(iy):: prnt4,prnt5,prnt6
c..................................................................
c     Subroutine to calculate bounce-averaged Fokker-Planck collision
c     coefficients (relativistic corrections added by MARK FRANZ--
c     U.S.A.F.)
c     If (cqlpmod.eq."enabled") then compute only the coefficients
c     at the orbit position l=l_ and do not perform the bounce-averages.
c..................................................................
      include 'param.h'
      include 'comm.h'
c
      data naccel/100/
      ialpha=2
!     ialpha=0  !BH180903: Made self-coll ener cons worse than ialpha=2
!     ialpha=1  !BH180903: Made self-coll ener cons worse than ialpha=2,0
      impcoef=0
      if (n.eq.1) then
        continue
      elseif (mod(n,ncoef).eq.1 .or. ncoef.eq.1) then
        continue
      else
        return
      endif
      impcoef=1
      nccoef=nccoef+1
      call bcast(cal(1,1,1,l_),zero,iymax*jx*ngen)
      call bcast(cbl(1,1,1,l_),zero,iymax*jx*ngen)
      call bcast(ccl(1,1,1,l_),zero,iymax*jx*ngen)
      call bcast(cdl(1,1,1,l_),zero,iymax*jx*ngen)
      call bcast(cel(1,1,1,l_),zero,iymax*jx*ngen)
      call bcast(cfl(1,1,1,l_),zero,iymax*jx*ngen)
      call bcast(eal(1,1,1,1,l_),zero,iymax*jx*ngen*2)
      call bcast(ebl(1,1,1,1,l_),zero,iymax*jx*ngen*2)
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)

c..................................................................
c     if only gen. species contributions are desired execute jump..
c..................................................................
      if (colmodl.eq.2 ) goto 110
c..................................................................
c     if no background species exist execute jump
c..................................................................
      if (ntotal.eq.ngen) goto 110
      iswwflag=0

      do 100 kbm=ngen+1,ntotal  ! Maxwellian species:
         !k=kbm 

c..................................................................
c     If this species is not to be included as a field (background)
c     species for calculation of the collision integral, jump out.
c..................................................................
        if (kfield(kbm).eq."disabled") go to 100

c..................................................................
c     Determine the Maxwellian distribution associated with
c     background species kbm. This will be a relativistic Maxwellian
c     for relativistic calculations.
c..................................................................

        temp_loc=temp(kbm,lr_)
        if (cqlpmod.eq."enabled") temp_loc=temppar(kbm,ls_)

!! YuP[08-2017] alternative to the following section: Use table cfpm() instead:
!! See if(cfp_integrals.eq.'enabled')  section
        if(cfp_integrals.eq.'disabled')then !Original method for integrals (slow)
        rstmss=fmass(kbm)*clite2/ergtkev
        reltmp=rstmss/temp_loc
	
        if (reltmp .gt. 100. .or. relativ .eq. "disabled") then
          ebk2=sqrt(pi/(2.*reltmp))
        else if (reltmp .lt. .01) then
          ebk2=2.*exp(reltmp)/reltmp**2
        else
          call cfpmodbe(reltmp,ebk1,ebk2)
        endif
        rnorm=reltmp/(4.*pi*cnorm3*ebk2)
        call bcast(temp1(0,0),zero,iyjx2)
c..................................................................
c     Need extra mesh points to represent ions on a mesh meant to
c     support electrons. Need more resolution near zero.
c     Split each velocity bin into nintg pieces and integrate.
c..................................................................
c990131        nintg0=41.*amax1(1.,sqrt(fmass(kbm)/1.67e-24))
cBH        nintg0=41.*max(1.d0,sqrt(fmass(kbm)/1.67e-24))
        nintg0=41.*max(1.d0,sqrt(fmass(kbm)/1.67d-24))
c990131        nintg=max0(nintg0,min0(51,int(x(2)/sqrt(.2*cnorm2/reltmp))))
        nintg=max(nintg0,min0(51,int(x(2)/sqrt(.2*cnorm2/reltmp))))
        nintg=2*(nintg/2)+1
        do 10 i=1,nintg
          sfac=4.+2.*(2*(i/2)-i)
          if (i.eq.1 .or. i.eq.nintg) sfac=1.
          sfac=sfac/3.
          do 11 j=2,jx
            xx=x(j)-(i-1)*dxm5(j)/dfloat(nintg-1)
            xx2=xx**2
            gg=sqrt(1.+xx2*cnorm2i) !cnorm2i=0 when relativ(ka).eq."disabled"
!!            ! YuP[07-2017] This is not quite correct - cnorm2i is set 
!!            ! for 'ka' general species, but here we consider 'kbm' Maxw.species.
!!            ! Should they be treated as relativistic or not?
!!            ! They can be different (relativistically) from 'ka' species.
!!            ! There should be a separate setting of relativ() for 
!!            ! Maxwellian species, too.
            gg2=gg**2

            if(cnorm2i*xx2-1.d-5 .gt. 0.d0) then
               expon=gg-1.
            else
               expon=.5*xx2/cnorm2
            endif
            fff=sfac*rnorm*exp(-expon*reltmp)*dxm5(j)/dfloat(nintg-1)
            temp1(1,j)=temp1(1,j)+xx*gg*fff
            temp1(2,j)=temp1(2,j)+xx*fff
            temp1(3,j)=temp1(3,j)+xx2*fff
            temp1(4,j)=temp1(4,j)+xx2*fff/gg
            temp1(5,j)=temp1(5,j)+(xx2/gg)**2*fff
            temp1(6,j)=temp1(6,j)+(xx2/gg2)**2*gg*fff
 11       continue
 10     continue
c..................................................................
        tam2(jx)=0.
        tam3(1)=0.
        tam5(1)=0.
        tam6(jx)=0.
        tam7(1)=0.
        tam9(1)=0.
c..................................................................
c     tam2(jx) and tam6(jx) represent integrals from xmax to infinity
c     The following coding performs this integration. In this case 21*xmax
c     represents infinity. The lack of this piece is most obvious when
c     ions are a general species and electrons are fixed Maxwellians.
c..................................................................
        do 15 ll2=1,21
          sfac=4.+2.*(2*(ll2/2)-ll2)
          if ( ll2.eq.1 .or. ll2.eq.21) sfac=1.
          sfac=sfac*x(jx)/60.
          do 16 ll1=1,20
            xx=(realiota(ll1)+.05*realiota(ll2-1))*x(jx)
            xx2=xx**2
            gg=sqrt(1.+xx2*cnorm2i) !cnorm2i=0 when relativ(ka).eq."disabled"

            if(cnorm2i*xx2-1.d-5 .gt. 0.d0) then
               expon=gg-1.
            else
               expon=.5*xx2/cnorm2
            endif
            fff=sfac*rnorm*exp(-expon*reltmp)
            tam2(jx)=tam2(jx)+xx*gg*fff
            tam6(jx)=tam6(jx)+xx*fff
 16       continue
 15     continue
        do 20 j=2,jx
          jj=jx+1-j
          jp=jj+1
          jm=j-1
c..................................................................
c     tam2 - M0; tam3 - N0; tam5 - E0
c     tam6 - M0';  tam7 - N0';   tam9 - E0'
c     see UCRL manual.
c     Eqns. 64-66 of UCRL-96510 by Mark R. Franz
c..................................................................
          tam2(jj)=tam2(jp)+temp1(1,jp)
          tam3(j)=tam3(jm)+temp1(3,j)
          tam5(j)=tam5(jm)+temp1(5,j)
          tam6(jj)=tam6(jp)+temp1(2,jp)
          tam7(j)=tam7(jm)+temp1(4,j)
          tam9(j)=tam9(jm)+temp1(6,j)
 20     continue
c..................................................................
        do 30 j=2,jx
          tam10(j)=cog(0,1)*(3.*tam7(j)+cnorm2i*(2.*xm(j,3)*tam6(j)-
     *      tam9(j)))*gamsqr(j)  ! ~Eq. 61, and cog(0,1)=4*pi/3
          tam11(j)=cog(0,1)*(xsq(j)*tam2(j)+
     *      gamsqr(j)*xm(j,-1)*tam5(j))  ! ~Eq. 62
          tam12(j)=cog(0,1)*(tam2(j)+1.5*xm(j,-1)*tam3(j)-
     *      .5*xm(j,-3)*tam5(j))  ! ~Eq. 63
 30     continue
        endif ! (cfp_integrals.eq.'disabled')
 
 
        !YuP[2020-07-16] alternative to the above section. Use table cfpm() instead.
        if(cfp_integrals.eq.'enabled')then
          ! For tam10(1:jx),tam11(1:jx),tam12(1:jx) - 
          ! Instead of the above calculations [cfp_integrals.eq.'disabled']
          ! use integrals stored in cfpm() array.
          kk=kbm-ngen
          call cfp_integrals_get(kk,temp_loc) !out: tam10(),tam11(),tam12()
        endif ! (cfp_integrals.eq.'enabled')
        !YuP[2020-07-16] Done For tam10,11,12 

c.......................................................................
c     Perform the bounce-average for the background species and introduce
c     the contribution to all general species coeff.
c.......................................................................

        if (cqlpmod .ne. "enabled") then
          call bavdens(kbm) !CQL3D only
        else ! (cqlpmod.eq."enabled")
          do 59 i=1,iy_(l_) !iymax !YuP[2021-03-11] iy-->iy_(l_)
            bavdn(i,lr_)=denpar(kbm,ls_) !CQLP
            bavpd(i,lr_)=denpar(kbm,ls_)*sinn(i,l_) !CQLP
            !YuP[2021-03] Logically, it would be better to use 
            ! bavdn(i,ls_) and bavpd(i,ls_) here, in CQLP part,
            !i.e., index ls_ instead of lr_.
            !However, it does not matter. bavdn(:,lr_) is only used
            !locally here, so the index lr_ or ls_ does not matter.
            !Just be sure that below, at cfl() = line, 
            !they are used with the same index. 
 59       continue
        endif

c     Below, eal and ebl are to save contributions to the collisional
c     coefficients cal() and cbl(i,j,ka,l_) for general species ka and
c     radial location l_, resulting from background electrons 
c     (eal and  ebl of (i,j,ka,1,l_)), and resulting from the sum of the
c     effects of the background ions, (eal and ebl of (i,j,ka,2,l_)). 
c     eal and ebl are used later for calculating powers from the
c     general  to the Max. species.

        do 80 ka=1,ngen  !Loop over gen species, adding bkgrnd coeffs
          anr1=gama(ka,kbm)*satioz2(kbm,ka)*one_  !ln(Lambda)*(Z_k/Z_kk)**2
          anr2=anr1*satiom(ka,kbm)              !*mass_kk/mass_k
          call bcast(tem1,zero,iymax*jx)
          call bcast(tem2,zero,iymax*jx)
          
          do 70 j=2,jx
            ttta=anr2*tam10(j)*gamefac(j,kbm) !if gamafac="enabled" or "hesslow", then
            tttb=anr1*tam11(j)*gamefac(j,kbm) !use gamefac for en dep gama
            tttf=anr1*tam12(j)*gamefac(j,kbm) !YuP[2019-07-26] kbm index added
            do 60 i=1,iymax !!YuP[2021-03-11] Changed iy-->iymax
              jj=i+(j-1)*iymax
              tem1(jj)=ttta*vptb(i,lr_)*bavdn(i,lr_) !saved for eal()
              tem2(jj)=tttb*vptb(i,lr_)*bavdn(i,lr_) !saved for ebl()
              cal(i,j,ka,l_)=cal(i,j,ka,l_)+tem1(jj)
              cbl(i,j,ka,l_)=cbl(i,j,ka,l_)+tem2(jj)
             cfl(i,j,ka,l_)=cfl(i,j,ka,l_)+tttf*vptb(i,lr_)*bavpd(i,lr_)
 60         continue
 70       continue

ccc          do i=1,iy
ccc             prnt4(i)=vptb(i,lr_)
ccc             prnt5(i)=bavpd(i,lr_)
ccc          enddo

          if(kbm.eq.kelecm) then
            call daxpy(iymax*jx,one,tem1,1,eal(1,1,ka,1,l_),1)
            call daxpy(iymax*jx,one,tem2,1,ebl(1,1,ka,1,l_),1)
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)
          else
            do 101 i=1,nionm
              if(kbm.eq.kionm(i)) then
                call daxpy(iymax*jx,one,tem1,1,eal(1,1,ka,2,l_),1)
                call daxpy(iymax*jx,one,tem2,1,ebl(1,1,ka,2,l_),1)

cBH180807:  Saving individual Maxwl ion components of coll operator
cBH180807:                call daxpy(iyjx,one,tem1,1,eal(1,1,ka,kbm+1,l_),1)
cBH180807:                call daxpy(iyjx,one,tem2,1,ebl(1,1,ka,kbm+1,l_),1)
cBH180807:  Need to increase dimensions of eal,ebl, and calc power transfer
cBH180807:  to each genrl species, and put into the .nc output file for
cBH180807:  use with radial transport moment codes.

              endif
 101        continue
          endif

 80     continue ! ka=1,ngen, line 187
 100  continue ! kbm=ngen+1,ntotal, line 49


!------------------------- contribution from partially ionized ions -----
!YuP[2019-07-26]--[2019-09]
      if(gamafac.eq."hesslow" .and. kelecg.eq.1)then
      
      !---1---> SCATTERING of free electron on partially screened ions
      ! These values and gscreen array are set in sub.set_gscreen_hesslow(imp_type):
      ! fmass_imp ! mass of impurity ion [gram]
      ! bnumb_imp(kstate) ! Z charge number of each ionization state
      !--- Distribution over charge states dens_imp(kstate,lr_) was found in tdchief
      ! Need to add: 
      ! temp_imp(kstate,lr_) ! T[kev] for each ionization state kstate, at given radial point 
      ! For now, assume all ionization states have same temperature, 
      ! equal to temper. of any Maxwellian ions, so that
      kion1=kionm(1)
      temp_imp(0:nstates,lr_)=temp(kion1,lr_) !BUT is it ok for kstate=0 (atom)?
      !Setup an effective Maxwellian distr. for impurity species,
      !find all relevant integrals
      rstmss= fmass_imp*clite2/ergtkev
      do kstate=0,nstates ! kstate=0 means neutral (atom)
        ! Note: kstate=0:nstates, with kstate=0 corresponding to a neutral,
        !       for which bnumb_imp(0)=0;
        !       and kstate= nstates corresponding to a fully ionized state,
        !       for which gscreen function (see below) is 0.
        temp_loc=temp_imp(kstate,lr_)
        !impurity ions - they are cold (non-relativistic)
          if(cfp_integrals.eq.'disabled')then
            reltmp= rstmss/temp_loc
            ebk2=sqrt(pi/(2.*reltmp))
            rnorm=reltmp/(4.*pi*cnorm3*ebk2)
            call bcast(temp1(0,0),zero,iyjx2)
            !Need extra mesh points to represent ions on a mesh meant to
            !support electrons. Need more resolution near zero.
            !Split each velocity bin into nintg pieces and integrate.
            nintg0=41.*max(1.d0,sqrt(fmass_imp/1.67d-24))
            nintg=max(nintg0,min0(51,int(x(2)/sqrt(.2*cnorm2/reltmp))))
            nintg=2*(nintg/2)+1
            do i=1,nintg
              sfac=4.+2.*(2*(i/2)-i)
              if (i.eq.1 .or. i.eq.nintg) sfac=1.
              sfac=sfac/3.
              do j=2,jx
                xx=x(j)-(i-1)*dxm5(j)/dfloat(nintg-1)
                xx2=xx**2
                gg=sqrt(1.+xx2*cnorm2i)
                gg2=gg**2
                expon=.5*xx2/cnorm2 ! non-relativ. limit
               fff=sfac*rnorm*exp(-expon*reltmp)*dxm5(j)/dfloat(nintg-1)
                temp1(1,j)=temp1(1,j)+xx*gg*fff
                temp1(2,j)=temp1(2,j)+xx*fff
                temp1(3,j)=temp1(3,j)+xx2*fff
                temp1(4,j)=temp1(4,j)+xx2*fff/gg
                temp1(5,j)=temp1(5,j)+(xx2/gg)**2*fff
                temp1(6,j)=temp1(6,j)+(xx2/gg2)**2*gg*fff
              enddo
            enddo
            tam2(jx)=0.
            tam3(1)=0.
            tam5(1)=0.
            !tam2(jx) represents integral from xmax to infinity
            !The following coding performs this integration. In this case 21*xmax
            !represents infinity. The lack of this piece is most obvious when
            !ions are a general species and electrons are fixed Maxwellians.
            do ll2=1,21
              sfac=4.+2.*(2*(ll2/2)-ll2)
              if ( ll2.eq.1 .or. ll2.eq.21) sfac=1.
              sfac=sfac*x(jx)/60.
              do ll1=1,20
                xx=(realiota(ll1)+.05*realiota(ll2-1))*x(jx)
                xx2=xx**2
                gg=sqrt(1.+xx2*cnorm2i)
                expon=.5*xx2/cnorm2 ! non-relativ. limit
                fff=sfac*rnorm*exp(-expon*reltmp)
                tam2(jx)=tam2(jx)+xx*gg*fff
              enddo
            enddo
     
            do j=2,jx
              jj=jx+1-j
              jp=jj+1
              jm=j-1
              tam2(jj)=tam2(jp)+temp1(1,jp)
              tam3(j)= tam3(jm)+temp1(3,j)
              tam5(j)= tam5(jm)+temp1(5,j)
            enddo
            do j=2,jx
              tam12(j)=cog(0,1)*(tam2(j)+1.5*xm(j,-1)*tam3(j)-
     &                               0.5*xm(j,-3)*tam5(j)  )  ! ~Eq. 63
            enddo

          else !if(cfp_integrals.eq.'enabled')then
            !YuP[2020-07-16] alternative to the above. Use table cfpm() instead.
            ! For tam10(1:jx),tam11(1:jx),tam12(1:jx) - 
            ! Instead of the above calculations [cfp_integrals.eq.'disabled']
            ! use integrals stored in cfpm() array.
            ktable=nmax+kstate+1 !so that ktable=(nmax+1):(nmax+nstates+1)
            call cfp_integrals_get(ktable,temp_loc) !out:tam10(),tam11(),tam12()
            !Here, we only need tam12(j)
          endif ! (cfp_integrals.eq.'enabled')

          !Perform the bounce-average for the background species and introduce
          !the contribution to all general species coeff.
          !ka=kelecg ! contribution will be added to cfl(i,j,ka,l_)
          Zion2=bnumb_imp(kstate)**2
          !Similar to anr1=gama(ka,k)*satioz2(k,ka)  !ln(Lambda0)*(Z_k/Z_e)**2
          !use anr1= gama(ka,kstate)*Zion2
          !i.e.,replace satioz2(i,ka)=(bnumb(i)/bnumb(ka))**2 by bnumb_imp(kstate)**2
          !Note: here bnumb(ka)=bnumb(kelecg)=-1
          do j=2,jx
           ! We only need to add contribution for "F" term - scattering
           ! of electron on ion.
           !Consider  coulomb_log_ei= gama(ka,kstate)*gamefac(j,kstate)
           !Note that for e-on-ions (according to Hesslow; see subr.cfpgamma)
           ! gama(kelecg,k)*gamefac(j,k) = gama_ei + 0.2*log(1.d0 + p_dep**5)
           !where gama_ei=gama(kelecg,kion) = 14.9-0.5*ln(ne20) +ln(T_kev),
           !i.e. it does not depend on a particular ion type.
           !So, instead of gama(ka,kstate)*gamefac(j,kstate)
           !we can use gama(ka,kion)*gamefac(j,kion)
           !where kion is kion=kionm(1), for example.
           kion1=kionm(1)
           coulomb_log_ei= gama(kelecg,kion1)*gamefac(j,kion1)
           tttf= tam12(j)*(Zion2*coulomb_log_ei +gscreen(kstate,j))
           !Note: for kstate=nstates we have z_state=z_imp (so z_bound=0),
           !so that gscreen(nstates,j)=0 for all j.
           !But the first term still works, as for a fully-ionized ion:
           ! Zion2*coulomb_log_ei
           do i=1,iy_(l_) !YuP[2021-03-11] Changed iy-->iy_(l_) 
             !Find bounce-av. density of this ionization state;
             !similar to subr.bavdens, only replace reden(k,lr_)-->dens_imp(kstate,lr_)
             bavpd_imp=batot(i,lr_)*sinn(i,lmdpln_)*dens_imp(kstate,lr_)
             cfl(i,j,kelecg,l_)= cfl(i,j,kelecg,l_)
     &                          +tttf*vptb(i,lr_)*bavpd_imp
             ! ka=kelecg here (electron_general)
           enddo ! i
          enddo ! j
          
      enddo !kstate=0,nstates

      !---2---> SLOWING-DOWN of free electron on bound electrons.
      ! See the slowing down term (Eq. 2.31 in Hesslow, JPP-2018).
      ! We need to add the second term (SUM_j), because 
      ! the first term (free e on free e) is already added, as usual.
      ! For the slowing down (drag force)
      ! the only coll. coefficient that must be changed is A.
      ! fmass(k) here is for bound electrons.
      !Normally, for interaction of free electrons (general species)
      !with Maxwellian electrons, we define the distribution function
      !of Maxwellian electrons through temp() and reden() values.
      !Now, we consider interaction of free electrons with bound electrons.
      !We need to add extra term to the A collisional (drag) coefficient.
      ![For reference: Eq. 61 in UCRL-96510 by Mark R. Franz].
      !So what is the temperature of bound electrons?
      !It is needed for calculation of integrals below,
      !such as M0, N0, E0.
      !Should we treat bound electrons as being at T=0?   
      !For simplicity, we just set them to be at very low T, say 10eV.  
      temp_e_bound=10.d-3 ! in keV
      fmass_e=9.1095d-28 !fmass(kelecg)
      rstmss= fmass_e*clite2/ergtkev
      !ka=kelecg ! contribution will be added to cal(i,j,ka,l_)
      do kstate=0,nstates-1 ! kstate=0 means neutral (atom)
        ! Note: kstate=0:nstates, with kstate=0 corresponding to a neutral,
        !       for which bnumb_imp(0)=0;
        !       and kstate= nstates corresponding to a fully ionized state,
        !       for which hbethe function is 0 (because z_bound=0).
        temp_loc=temp_e_bound
          if(cfp_integrals.eq.'disabled')then
            reltmp= rstmss/temp_loc
            !bound e - they are cold (non-relativistic)
            ebk2=sqrt(pi/(2.*reltmp))
            rnorm=reltmp/(4.*pi*cnorm3*ebk2)
            call bcast(temp1(0,0),zero,iyjx2)
            !Need extra mesh points to represent cold e on a mesh meant to
            !support general electrons. Need more resolution near zero.
            !Split each velocity bin into nintg pieces and integrate.
            nintg0=41
            nintg=max(nintg0,min0(51,int(x(2)/sqrt(.2*cnorm2/reltmp))))
            nintg=2*(nintg/2)+1
            do i=1,nintg
              sfac=4.+2.*(2*(i/2)-i)
              if (i.eq.1 .or. i.eq.nintg) sfac=1.
              sfac=sfac/3.
              do j=2,jx
                xx=x(j)-(i-1)*dxm5(j)/dfloat(nintg-1)
                xx2=xx**2
                gg=sqrt(1.+xx2*cnorm2i) ! could set to 1.
                gg2=gg**2
                expon=.5*xx2/cnorm2 ! non-relativ. limit
               fff=sfac*rnorm*exp(-expon*reltmp)*dxm5(j)/dfloat(nintg-1)
                !temp1(1,j)=temp1(1,j)+xx*gg*fff !not needed here?
                temp1(2,j)=temp1(2,j)+xx*fff
                !temp1(3,j)=temp1(3,j)+xx2*fff !not needed here?
                temp1(4,j)=temp1(4,j)+xx2*fff/gg
                !temp1(5,j)=temp1(5,j)+(xx2/gg)**2*fff !not needed here?
                temp1(6,j)=temp1(6,j)+(xx2/gg2)**2*gg*fff
              enddo
            enddo
            tam2(jx)=0. !not needed here?
            tam3(1)=0. !not needed here?
            tam5(1)=0. !not needed here?
            tam6(jx)=0.
            tam7(1)=0.
            tam9(1)=0.         
            !tam6(jx) represents integral from xmax to infinity
            !The following coding performs this integration. In this case 21*xmax
            !represents infinity. The lack of this piece is most obvious when
            !ions are a general species and electrons are fixed Maxwellians.
            do ll2=1,21
              sfac=4.+2.*(2*(ll2/2)-ll2)
              if ( ll2.eq.1 .or. ll2.eq.21) sfac=1.
              sfac=sfac*x(jx)/60.
              do ll1=1,20
                xx=(realiota(ll1)+.05*realiota(ll2-1))*x(jx)
                xx2=xx**2
                gg=sqrt(1.+xx2*cnorm2i)
                expon=.5*xx2/cnorm2 ! non-relativ. limit
                fff=sfac*rnorm*exp(-expon*reltmp)
                !tam2(jx)=tam2(jx)+xx*gg*fff ! not needed here?
                tam6(jx)=tam6(jx)+xx*fff 
              enddo
            enddo
            do j=2,jx
              jj=jx+1-j
              jp=jj+1
              jm=j-1
              !tam2(jj)=tam2(jp)+temp1(1,jp) ! not needed here?
              !tam3(j)= tam3(jm)+temp1(3,j) ! not needed here?
              !tam5(j)= tam5(jm)+temp1(5,j) ! not needed here?
              tam6(jj)=tam6(jp)+temp1(2,jp)
              tam7(j)= tam7(jm)+temp1(4,j)
              tam9(j)= tam9(jm)+temp1(6,j) 
              !tam6 - M0';  tam7 - N0';   tam9 - E0' 
              !Eqns. 64-66 of UCRL-96510 by Mark R. Franz 
            enddo
            do j=2,jx
              tam10(j)=cog(0,1)*(3.*tam7(j)+cnorm2i*(2.*xm(j,3)*tam6(j)-
     &                                           tam9(j)))*gamsqr(j)  
              ! ~Eq. 61, (inside {} brackets), and cog(0,1)=4*pi/3
            enddo
          else !if(cfp_integrals.eq.'enabled')then
            !YuP[2020-07-16] alternative to the above. Use table cfpm() instead.
            ! For tam10(1:jx),tam11(1:jx),tam12(1:jx) - 
            ! Instead of the above calculations [cfp_integrals.eq.'disabled']
            ! use integrals stored in cfpm() array.
            ktable=nmax+nstates+2 !(T-table for bound electrons at 10eV)
            call cfp_integrals_get(ktable,temp_loc) !out:tam10(),tam11(),tam12()
            !Here, we only need tam10(j).
            !Since we assumed that bound e are at same T=10eV (in each kstate)
            !we could compute tam10 just once, say, for kstate=0.
          endif ! (cfp_integrals.eq.'enabled')

          !Perform the bounce-average for the background species and introduce
          !the contribution to all general species coeff.
          !Instead of ne*gama(ka,k)*satioz2(k,ka)*satiom(ka,k) 
          !now we must use SUM[n_kstate*hbethe(kstate)].
          !Note: here bnumb(ka)=bnumb(kelecg)=-1, so satioz2(k,ka)=1.
          !Also here: satiom(ka,k)==mass_kk/mass_k =1.
          !Use bounce-av. density of this ionization state;
          !similar to subr.bavdens, only replace reden(k,lr_)-->dens_imp(kstate,lr_)
          call bcast(tem1,zero,iymax*jx)
          do j=2,jx
           ! We only need to add contribution for the "A" term -  
           ! drag of (free)electron on (bound)electrons.
           ! Replace this: ttta= gama(ka,k)*tam10(j)*gamefac(j,k) with this:
           ttta= tam10(j)*dens_imp(kstate,lr_)*hbethe(kstate,j)
           !Note that density is included here.
           !It is multiplied by hbethe(kstate,j), which contains 
           !z_bound= z_imp-z_state ! Nej in paper (number of bound electrons)
           do i=1,iymax !YuP[2021-03-11] Changed iy-->iymax 
             jj=i+(j-1)*iymax
             tem1(jj)= ttta*vptb(i,lr_) !for contributing to eal() below
             cal(i,j,kelecg,l_)= cal(i,j,kelecg,l_)+tem1(jj) !free_e on bound_e
             ! ka=kelecg here (electron_general)
           enddo ! i
          enddo ! j
          !??? Not sure where to add tem1() contribution:
          !eal(*,*,ka,1,l_) is for transfer of energy to Maxwellian e [k=kelecm].
          !eal(*,*,ka,2,l_) is for transfer of energy to Maxwellian i [k=kionm()],
          !and all such ions are summed-up (not for individual ion species).
          !Maybe we consider that in this case the energy is transferred to 
          !the partially ionized ion (kstate) ?
          !call daxpy(iyjx,one,tem1,1,eal(1,1,ka,2,l_),1) 
          
      enddo !kstate=0,nstates

      endif ! gamafac.eq."hesslow" .and. kelecg.eq.1

!YuP[2019-07-26] 
!------------------------- contribution from partially ionized ions -----



c.......................................................................
c     At this point, contributions to the FP coll coeffs for each 
c     general species due to the Maxwl background species have been
c     added to cal(,,), etc.
c.......................................................................


 110  continue !Branch around Maxwl contribs if nmax=0, or colmodl=2
ccc      kk=1
ccc      do i=1,iy
ccc         prnt1(i)=cal(i,2,kk,l_)
ccc         prnt2(i)=cbl(i,2,kk,l_)
ccc         prnt3(i)=ccl(i,2,kk,l_)
ccc         prnt4(i)=cdl(i,2,kk,l_)
ccc         prnt5(i)=cel(i,2,kk,l_)
ccc         prnt6(i)=cfl(i,2,kk,l_)
ccc      enddo

c..................................................................
c     Calculate coefficients and their bounce-averages for a general species
c..................................................................

      if (colmodl.eq.1) goto 700
      if (colmodl.eq.4 .and. n.ge.naccel) then !Undocumented option
                                      ! Kerbel
        do 250 k=1,ngen
          if (k.ne.ngen) then
            do 220 j=1,jx
              do 210 i=1,iymax
                if(2.*f(i,j,k,l_)-f_(i,j,k,l_).gt.0.) then
                  fxsp(i,j,k,l_)= 2.*f(i,j,k,l_)-f_(i,j,k,l_)
                else
                  fxsp(i,j,k,l_)= .5*f(i,j,k,l_)
                endif
 210          continue
 220        continue
          else
            call diagescl(kelecg)
            call dcopy(iyjx2,f(0,0,ngen,l_),1,fxsp(0,0,ngen,l_),1)
          endif
 250    continue
      else
        do 280 k=1,ngen
          if (colmodl.eq.4 .and. k.eq.ngen) call diagescl(kelecg)
 280    continue
      endif
      madd=1
      if (machine.eq."mirror") madd=2

c.......................................................................
c     loop over orbit mesh s(l) (for general species only)
c     CQLP case: compute only s(l_)
c.......................................................................

      if (cqlpmod .ne. "enabled") then
        iorbstr=1
        iorbend=lz  ! Whole flux surface, eqsym="none", else half FS.
      else ! (cqlpmod.eq."enabled")
        iorbstr=l_
        iorbend=l_
      endif

cBH180906:  Make sure that expanded eal/ebl and new ecl arrays 
cBH180906:  for non-isotropic genrl distributions, saving ca,cb,cc
cBH180906:  coll contributions to each general species from
cBH180906:  itself and other species, are zeroed out.  Then,
cBH180906:  accumate the contributions below, use for calculating
cBH180906:  power flow in diagentr coding, and save powers into
cBH180906:  .nc output file.
cBH180906:  ecl() can be dimension to cover just the range of
cBH180906   general species k's.
cBH180906   The coding also needs adding to cfpcoefr.f.

      do 600 l=iorbstr,iorbend
        ileff=l
        if (cqlpmod.eq."enabled") ileff=ls_

        do 500 k=1,ngen
c.................................................................
c     Jump out if this species is not to be used as a background species
cBH180901: This comment doesn't make much sense, as kfield(k) here is not
cBH180901: related to a background species.  Maybe the comment is
cBH180901: mistakenly copied from above.
c..................................................................
          if (kfield(k).eq."disabled") go to 500

c     zero ca, cb,.., cf  : 
CPTR>>>REPLACE PTR-BCASTCACD
          ca=zero !call bcast(ca,zero,iyjx)
          cb=zero !call bcast(cb,zero,iyjx)
          cc=zero !call bcast(cc,zero,iyjx)
          cd=zero !call bcast(cd,zero,iyjx)
          ce=zero !call bcast(ce,zero,iyjx)
          cf=zero !call bcast(cf,zero,iyjx)
CPTR<<<END PTR-BCASTCACD

          mu1=0
          mu2=mx
          mu3=madd
          if (colmodl.eq.3) then
            mu1=1  !Skipping P_0 term
            mu2=mx
            mu3=1
          endif
          do 400 m=mu1,mu2,mu3
            if (colmodl.eq.4 .and. n.ge.naccel) then
              call dcopy(iyjx2,fxsp(0,0,k,l_),1,temp3(0,0),1)
            else
              call dcopy(iyjx2,f(0,0,k,l_),1,temp3(0,0),1)
            endif

c     compute V_m_b in tam1(j), for given l, m and gen. species b
c     coeff. of Legendre decomposition of f (using temp3)
            call cfpleg(m,ileff,1) !-> tam1
!            do j=1,jx
!              if((j/15)*15.eq.j)then
!              write(*,*)' after cfpleg: m, V_m=tam1(j)=',m,tam1(j)
!              endif
!            enddo

            ! tam2(1)=0.d0 !YuP: should be added ?
            tam2(jx)=0.
            tam3(1)=0.
            tam4(jx)=0.
            tam5(1)=0.
            tam6(jx)=0.
            tam7(1)=0.
            tam8(jx)=0.
            tam9(1)=0.

c     prepare arrays for computation of M_m, N_m, R_m and E_m
            do 301 j=2,jx
              jm=j-1
              tam20(j)=.5*dxm5(j)*
     *          (gamsqr(j)*xm(j,ix1(m))*tam1(j)+
     *          gamsqr(jm)*xm(jm,ix1(m))*tam1(jm))
              tam13(j)=.5*dxm5(j)*
     *          (gamsqr(j)*xm(j,ix2(m))*tam1(j)+
     *          gamsqr(jm)*xm(jm,ix2(m))*tam1(jm))
              tam14(j)=.5*dxm5(j)*
     *          (gamsqr(j)*xm(j,ix3(m))*tam1(j)+
     *          gamsqr(jm)*xm(jm,ix3(m))*tam1(jm))
              tam15(j)=.5*dxm5(j)*
     *          (gamsqr(j)*xm(j,ix4(m))*tam1(j)+
     *          gamsqr(jm)*xm(jm,ix4(m))*tam1(jm))
              tam16(j)=.5*dxm5(j)*
     *          (gamma(j)*xm(j,ix1(m))*tam1(j)+
     *          gamma(jm)*xm(jm,ix1(m))*tam1(jm))
              tam17(j)=.5*dxm5(j)*
     *          (gamma(j)*xm(j,ix2(m))*tam1(j)+
     *          gamma(jm)*xm(jm,ix2(m))*tam1(jm))
              tam18(j)=.5*dxm5(j)*
     *          (gamma(j)*xm(j,ix3(m))*tam1(j)+
     *          gamma(jm)*xm(jm,ix3(m))*tam1(jm))
              tam19(j)=.5*dxm5(j)*
     *          (gamma(j)*xm(j,ix4(m))*tam1(j)+
     *          gamma(jm)*xm(jm,ix4(m))*tam1(jm))
 301        continue

c     note: ialpha set to 0, 1, or 2 at top of subroutine, Using 2 now.
            if (m.ge.1 .or. ialpha.eq.2) goto 308
c..................................................................
c     The do loops 302 through 307 seek to take advantage of the
c     Maxwellian nature of f for v < vth/2. This is employed in
c     the integrations to obtain the functionals - Thus f is
c     assumed to be Maxwellian between velocity mesh points,
c     not linear. (not used)
c..................................................................
            do 302 j=2,jx
              xs=sqrt(temp(k,lr_)*ergtkev*0.5/fmass(k))
              if (cqlpmod.eq."enabled")
     +          xs=sqrt(temppar(k,ls_)*ergtkev*0.5/fmass(k))
              if (x(j)*vnorm.gt.xs) goto 303
 302        continue
 303        continue
            jthov2=j-1
c..................................................................
c     Determine the "local" Maxwellian between meshpoints.
c..................................................................
            do 304 j=1,jthov2
c990131              tam21(j)=alog(tam1(j)/tam1(j+1))/(gamm1(j+1)-gamm1(j))
              tam21(j)=log(tam1(j)/tam1(j+1))/(gamm1(j+1)-gamm1(j))
              tam22(j)=tam1(j)/exp(-(gamm1(j)*tam21(j)))
 304        continue
            rstmss=fmass(k)*clite2/ergtkev
            reltmp=rstmss/temp(k,lr_)
            if (cqlpmod.eq."enabled") reltmp=rstmss/temppar(k,ls_)
            call bcast(tam23,zero,jx*8)
            nintg=max0(21,min0(51,int(x(2)/sqrt(.2*cnorm2/reltmp))))
            nintg=2*(nintg/2)+1
            do 305 i=1,nintg
              sfac=4.+2.*(2*(i/2)-i)
              if (i.eq.1 .or. i.eq.nintg) sfac=1.
              sfac=sfac/3.
              do 306 j=2,jthov2
                xx=x(j)-(i-1)*dxm5(j)/dfloat(nintg-1)
                xx2=xx**2
                gg=sqrt(1.+xx2*cnorm2i)
                gg2=gg**2
cBH                if(cnorm2i*xx2-1.e-5 .gt. 0.d0) then
                if(cnorm2i*xx2-1.d-5 .gt. 0.d0) then
                   expon=gg-1.
                else
                   expon=.5*xx2/cnorm2
                endif
                fff=sfac*tam22(j-1)*exp(-expon*tam21(j-1))*dxm5(j)
     1            /dfloat(nintg-1)
                tam30(j)=tam30(j)+
     *            gg2*(xx/gg)**ix1(m)*fff
                tam23(j)=tam23(j)+
     *            gg2*(xx/gg)**ix2(m)*fff
                tam24(j)=tam24(j)+
     *            gg2*(xx/gg)**ix3(m)*fff
                tam25(j)=tam25(j)+
     *            gg2*(xx/gg)**ix4(m)*fff
                tam26(j)=tam26(j)+
     *            gg*(xx/gg)**ix1(m)*fff
                tam27(j)=tam27(j)+
     *            gg*(xx/gg)**ix2(m)*fff
                tam28(j)=tam28(j)+
     *            gg*(xx/gg)**ix3(m)*fff
                tam29(j)=tam29(j)+
     *            gg*(xx/gg)**ix4(m)*fff
 306          continue
 305        continue
            do 307 j=1,jthov2
              if (ialpha.eq.0) then
                alpha=(x(j)/x(jthov2))**2
              elseif (ialpha .eq. 1) then
                alpha=0.
              else
                alpha=1.
              endif
              tam20(j)=alpha*tam20(j)+(1.-alpha)*tam30(j)
              tam14(j)=alpha*tam14(j)+(1.-alpha)*tam24(j)
              tam16(j)=alpha*tam16(j)+(1.-alpha)*tam26(j)
              tam18(j)=alpha*tam18(j)+(1.-alpha)*tam28(j)
              tam13(j)=alpha*tam13(j)+(1.-alpha)*tam23(j)
              tam15(j)=alpha*tam15(j)+(1.-alpha)*tam25(j)
              tam17(j)=alpha*tam17(j)+(1.-alpha)*tam27(j)
              tam19(j)=alpha*tam19(j)+(1.-alpha)*tam29(j)
 307        continue

 308        continue  !End of branch on special integration

            do 310 j=2,jx
              jj=jx+1-j
              jp=jj+1
              jm=j-1
              tam2(jj)=tam2(jp)+tam20(jp)
              tam3(j) =tam3(jm)+tam13(j)
              tam4(jj)=tam4(jp)+tam14(jp)
              tam5(j) =tam5(jm)+tam15(j)
              tam6(jj)=tam6(jp)+tam16(jp)
              tam7(j) =tam7(jm)+tam17(j)
              tam8(jj)=tam8(jp)+tam18(jp)
              tam9(j) =tam9(jm)+tam19(j)
 310        continue

c..................................................................
c     tam2= v**(2+m) * M_m ; tam3= v**(1-m) * N_m ; tam4= v**m * R_m
c     tam5= v**(1-m) * E_m
c     tam6= v**(2+m) * Mh_m ; tam7= v**(1-m) * Nh_m ; tam8= v**m * Rh_m
c     tam9= v**(1-m) * Eh_m ; where Mh="M / gamma(ksi)", etc.
c.......................................................................
            do 330 j=1,jx
              tam2(j)=xm(j,ix5(m))*tam2(j)
              tam3(j)=xm(j,ix6(m))*tam3(j)
              tam4(j)=xm(j,ix7(m))*tam4(j)
              tam5(j)=xm(j,ix8(m))*tam5(j)
              tam6(j)=xm(j,ix5(m))*tam6(j)
              tam7(j)=xm(j,ix6(m))*tam7(j)
              tam8(j)=xm(j,ix7(m))*tam8(j)
              tam9(j)=xm(j,ix8(m))*tam9(j)
 330        continue

c..................................................................
c     sg(j) = B_m_b, needed for Rosenbluth potential g (g,h with relav corr)
c     sh(j) = A_m_b, needed for Rosenbluth potential h (h "=" g/gamma_prime)
c     sgx(j) = dsg/dv (not du), etc. for sgxx, ...
c.......................................................................
            do 340 j=2,jx
              sg(j)=cog(m,1)*(tam5(j)+tam2(j))-
     *          cog(m,2)*(tam3(j)+tam4(j))
              sgx(j)=cog(m,3)*tam2(j)-cog(m,4)*tam5(j)-
     *          cog(m,5)*tam4(j)+cog(m,6)*tam3(j)
              sgx(j)=gamma(j)*xi(j)*sgx(j)
              sgxx(j)=cog(m,7)*(tam5(j)+tam2(j))-
     *          cog(m,8)*(tam3(j)+tam4(j))
              sgxx(j)=gamsqr(j)*x2i(j)*sgxx(j)
              sh(j)=cog(m,1)*(tam9(j)+tam6(j))-
     *          cog(m,2)*(tam7(j)+tam8(j))
              shx(j)=cog(m,3)*tam6(j)-cog(m,4)*tam9(j)-
     *          cog(m,5)*tam8(j)+cog(m,6)*tam7(j)
              shx(j)=gamma(j)*xi(j)*shx(j)
              shxx(j)=cog(m,7)*(tam9(j)+tam6(j))-
     *          cog(m,8)*(tam7(j)+tam8(j))
              shxx(j)=gamsqr(j)*x2i(j)*shxx(j)
              shxxx(j)=cog(m,9)*tam6(j)-cog(m,10)*tam9(j)-
     *          cog(m,11)*tam8(j)+cog(m,12)*tam7(j)
              shxxx(j)=gamcub(j)*x3i(j)*shxxx(j)
!              if((j/15)*15.eq.j)then
!                write(*,'(a,2i4,1p5e10.3)')
!     +          'm,j, urel[cm/s], enerkev, f, A_m, B_m[cgs]=',
!     +           m,j, x(j)*vnorm, enerkev(j,k), 
!     +           temp3(1,j)/vnorm**3, sh(j)*vnorm, sg(j)*vnorm ![cgs]
!!     +      1e6*temp3(1,j)/(vnorm/1e2)**3, 1e6*sh(j)*(vnorm/1e2), 1e6*sg(j)*(vnorm/1e2) ![SI]
!              endif
 340        continue
 
!            gn=0.d0 !Test
!            do j=1,jx
!            do i=1,iy
!            !Test/check: compute density n[m^-3]
!            !1e6*  to convert cm^-3 to m^-3,
!            ! vnorm/1e2 gives m/s;  temp3 contains f_code
!            gn= gn+ (1e6*temp3(i,j)/(vnorm/1e2)**3)
!     +              *(cint2(j)*cynt2(i,l_)*(vnorm/1e2)**3)
!            enddo
!            enddo
!            write(*,*)'n[m^-3]=',gn 
!            !Test: Yes, verified.
!            pause

c.......................................................................
c     compute A_a, B_a, ..., F_a as in GA report GA-A20978 p.11, 
c     Nov. 1992  (CQL3D Manual, Harvey and McCoy, IAEA TCM Montreal):
c     Mildly relativistic coll FP coeffs per M. Franz, UCRL-96510, 1987.
c.......................................................................
            fmmp1=m*(m+1)
            do 350 j=1,jx
              tam2(j)=-gamcub(j)*xi(j)*fmmp1*sh(j)
     *          +.5*gamsqr(j)*(2.+fmmp1)*shx(j)
     *          -gammi(j)*x(j)*shxx(j)
     *          -.5*gamm2i(j)*xsq(j)*shxxx(j)
              tam3(j)=.5*xsq(j)*sgxx(j)
              tam4(j)=.5*gamma(j)*(sgx(j)-gamma(j)*xi(j)*sg(j))
              tam5(j)=.5*gamcub(j)*x2i(j)*fmmp1*sh(j)
     *          -gamsqr(j)*xi(j)*shx(j)
     *          -.5*gammi(j)*shxx(j)
              tam6(j)=.5*gamma(j)*xi(j)*sgx(j)
              tam7(j)=.5*gamsqr(j)*x2i(j)*sg(j)
 350        continue

c$$$c     sum over m
c$$$            do 380 iii=1,imax(ileff,lr_)
c$$$              do 370 ii=0,1
c$$$                if (madd.eq.2 .and. ii.eq.0) goto 370
c$$$                i=iii*ii-(iy+1-iii)*(ii-1)
c$$$                do 360 j=2,jx
c$$$                  ca(i,j)=ca(i,j)+ss(i,ileff,m,lr_)*tam2(j)*gamefac(j)
c$$$                  cb(i,j)=cb(i,j)+ss(i,ileff,m,lr_)*tam3(j)*gamefac(j)
c$$$                  cc(i,j)=cc(i,j)+ssy(i,ileff,m,lr_)*tam4(j)*gamefac(j)
c$$$                  cd(i,j)=cd(i,j)+sinz(i,ileff,lr_)*
c$$$     *              ssy(i,ileff,m,lr_)*tam5(j)*gamefac(j)
c$$$                  ce(i,j)=ce(i,j)+sinz(i,ileff,lr_)*
c$$$     *              ssy(i,ileff,m,lr_)*tam4(j)*gamefac(j)
c$$$                  cf(i,j)=cf(i,j)+sinz(i,ileff,lr_)*
c$$$     *              (ss(i,ileff,m,lr_)*tam6(j)+ssyy(i,ileff,m,lr_)
c$$$     +              *tam7(j))*gamefac(j)
c$$$ 360            continue
c$$$ 370          continue
c$$$ 380        continue
c$$$c     end of loop over m
c     sum over m: Add contribution from each m
              !Note: for CQLP, ileff= l= l_=ls_, index along s
              do 380 iii=1,imax(ileff,lr_)
                if(symtrap.eq."enabled")then !YuP[2021-04-02] added symtrap condition
                  !Note that this skipping (goto 380)
                  !is only valid for bounce-averaging process.
                  !It should not be applied for CQLP (which runs with 
                  !symtrap='disabled' only)
                  if(iii.ge.itl+1 .and. mod(m,2).eq.1) goto 380 !YuP-111202
                endif
                !YuP-111202: This removes a bug in the calculation of
                !YuP-111202: collisional contribution to C,D, and F.
                !YuP-111202: Check YuP Email to BH, 111201
                !YuP-111202: 
                !In trap region: no contribution from m=1,3,5...
                !The contribution from m=0,2,4,... will provide proper
                !parity in theta0-(pi/2): even parity for ca,cb,cf; 
                !odd parity for cc,cd,ce (they are ~ dPm/dtheta). 
                !No further symmetrization needed.
                ! [but only valid for CQL3D]
                do 370 ii=0,1
                i=iii*ii-(iy_(l_)+1-iii)*(ii-1) ! i=iy+1-iii or i=iii
                !YuP[2021-03-11] iy-->iy_(l_)
                !CQLP: l=index along field line, 
                !      and here l takes the single value of l_
                !CQL3D: l=index along pol.angle, scanning 1:lz range,
                !      while l_ is fixed (given radial index)
                do 360 j=2,jx
                 ca(i,j)=ca(i,j)+ss(i,ileff,m,lr_)*tam2(j)*gamefac(j,k) !YuP[2019-07-26] k index added
                 cb(i,j)=cb(i,j)+ss(i,ileff,m,lr_)*tam3(j)*gamefac(j,k) !YuP[2019-07-26] k index added
                 cc(i,j)=cc(i,j)+ssy(i,ileff,m,lr_)*tam4(j)*gamefac(j,k) !YuP[2019-07-26] k index added
                  cd(i,j)=cd(i,j)+sinz(i,ileff,lr_)*
     *              ssy(i,ileff,m,lr_)*tam5(j)*gamefac(j,k) !YuP[2019-07-26] k index added
                  ce(i,j)=ce(i,j)+sinz(i,ileff,lr_)*
     *              ssy(i,ileff,m,lr_)*tam4(j)*gamefac(j,k) !YuP[2019-07-26] k index added
                  cf(i,j)=cf(i,j)+sinz(i,ileff,lr_)*
     *              (ss(i,ileff,m,lr_)*tam6(j)+ssyy(i,ileff,m,lr_)
     +              *tam7(j))*gamefac(j,k) !YuP[2019-07-26] k index added
 360            continue ! j
 370          continue ! ii
 380          continue ! iii

 400      continue ! m, starting at l 313

          if (madd.eq.2 .or. symtrap.ne."enabled") goto 430

c     symmetrize in trap region (if symtrap.eq."enabled")
          do 420 i=itl+1,iyh
            iu=iy_(l_)+1-i !YuP[2021-03-11] iy-->iy_(l_)
            !CQLP: l=index along field line, 
            !      and here l takes the single value of l_
            !CQL3D: l=index along pol.angle, scanning 1:lz range,
            !      while l_ is fixed (given radial index)
            do 410 j=2,jx
              ca(i,j)=.5*(ca(i,j)+ca(iu,j))
              cb(i,j)=.5*(cb(i,j)+cb(iu,j))
              cf(i,j)=.5*(cf(i,j)+cf(iu,j))
              xq=sign(half,cc(i,j))
              xr=sign(half,cd(i,j))
              xs=sign(half,ce(i,j))
              cd(i,j)=xr*(abs(cd(i,j))+abs(cd(iu,j)))
              cc(i,j)=xq*(abs(cc(i,j))+abs(cc(iu,j)))
              ce(i,j)=xs*(abs(ce(i,j))+abs(ce(iu,j)))
              ca(iu,j)=ca(i,j)
              cb(iu,j)=cb(i,j)
              cc(iu,j)=-cc(i,j)
              cd(iu,j)=-cd(i,j)
              ce(iu,j)=-ce(i,j)
              cf(iu,j)=cf(i,j)
 410        continue
 420      continue
 430      continue

c.......................................................................
c     add contribution to each gen. species A_kk,.., including charge,
c     ln(Lambda) and mass coefficients.
c.......................................................................

          do 490 kk=1,ngen
            if (colmodl.eq.4) then
              if (kk.eq.kelecg .and. k.eq.kelecg) goto 490
              if (kk.ne.kelecg .and. k.eq.ngen) goto 490
            endif
            anr1=gama(kk,k)*satioz2(k,kk)*one_
            if (anr1.lt.em90) goto 490
CPTR>>>REPLACE PTR-DSCALCACD
            call dscal(iymax*jx,anr1,ca(1,1),1) !-YuP: size of ca..cf: iy*jx
            call dscal(iymax*jx,anr1,cb(1,1),1)
            call dscal(iymax*jx,anr1,cc(1,1),1)
            call dscal(iymax*jx,anr1,cd(1,1),1)
            call dscal(iymax*jx,anr1,ce(1,1),1)
            call dscal(iymax*jx,anr1,cf(1,1),1)
            call dscal(iymax*jx,satiom(kk,k),ca(1,1),1)
            call dscal(iymax*jx,satiom(kk,k),cd(1,1),1)
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)
CPTR<<<END PTR-DSCALCACD

            
c.......................................................................
c     Note: At this point, ca, ..,cf(i,j) are the coeff. from gen. species k
c     at a given orbit position l.
c.......................................................................

            if (cqlpmod .ne. "enabled") then

c     Perform the bounce averaging
              do 480 i=1,imax(l,lr_)
                !CQLP: l=index along field line, 
                !      and here l takes the single value of l_
                !CQL3D: l=index along pol.angle, here scanning 1:lz range,
                !      while l_ is fixed (given radial index)
                ii=iy_(l_)+1-i !YuP[2021-03-11] iy-->iy_(l_) Here: CQL3D
                ax=abs(coss(i,l_))*dtau(i,l,lr_)
                ay=tot(i,l,lr_)/sqrt(bbpsi(l,lr_))
                az=ay*tot(i,l,lr_)

cBH091031                ax1=ax
cBH091031                !i.e, not bounce pt interval:
cBH091031                if (l.eq.lz .or. lmax(i,lr_).ne.l) goto 440
cBH091031                ax1=ax+dtau(i,l+1,lr_)*abs(coss(i,l_))
cBH091031 440            continue
                if (eqsym.ne."none") then       !i.e. up-down symm
                   !if not bounce interval
                   if(l.eq.lz .or. l.ne.lmax(i,lr_)) then
                      ax1=ax
                   else !bounce interval: additional contribution
                      ax1=ax+dtau(i,l+1,lr_)*abs(coss(i,l_))
                   endif
                else  !eqsym="none"
                   if (l.lt.lz_bmax(lr_) .and. l.eq.lmax(i,lr_))then
                      !trapped, with tips between l and l+1 (above midplane)
                      ax1=ax+dtau(i,l+1,lr_)*abs(coss(i,l_)) 
                      !-YuP  Note: dtau(i,l+1,lr_)=0
                   elseif (l.gt.lz_bmax(lr_) .and. l.eq.lmax(i+iyh,lr_))
     +                then 
                      !trapped, with tips between l and l-1 (below midplane)     
                      ax1=ax+dtau(i,l-1,lr_)*abs(coss(i,l_)) !NB:l-1
                      !-YuP  Note: dtau(i,l-1,lr_)=0
                   else 
                      !passing (i<itl), or trapped but with tips at other l;
                      !also, at l=lz_bmax, includes last trapped particle i=itl
                      !(for such particle, lmax(itl)=lz_bmax; see micxinil)
                      ax1=ax
                   endif
                endif

                do 450 j=2,jx
                  cal(i,j,kk,l_)=cal(i,j,kk,l_)+ax1*ca(i,j)
                  cbl(i,j,kk,l_)=cbl(i,j,kk,l_)+ax1*cb(i,j)
                  ccl(i,j,kk,l_)=ccl(i,j,kk,l_)+ax*tot(i,l,lr_)*cc(i,j)
                  cdl(i,j,kk,l_)=cdl(i,j,kk,l_)+ax*ay*cd(i,j)
                  cel(i,j,kk,l_)=cel(i,j,kk,l_)+ax*ay*ce(i,j)
                  cfl(i,j,kk,l_)=cfl(i,j,kk,l_)+ax*az*cf(i,j)
cBH180906: This is a point to accumalate contributions to expanded
cBH180906: eal/ebl/ecl arrays saving coll contributions to each general
cBH180906: species from itself and other species.
 450            continue
                if (madd.eq.2) goto 470
                do 460 j=2,jx
                  cal(ii,j,kk,l_)=cal(ii,j,kk,l_)+ax1*ca(ii,j)
                  cbl(ii,j,kk,l_)=cbl(ii,j,kk,l_)+ax1*cb(ii,j)
                  ccl(ii,j,kk,l_)=ccl(ii,j,kk,l_)+ax*tot(ii,l,lr_)
     1              *cc(ii,j)
                  cdl(ii,j,kk,l_)=cdl(ii,j,kk,l_)+ax*ay*cd(ii,j)
                  cel(ii,j,kk,l_)=cel(ii,j,kk,l_)+ax*ay*ce(ii,j)
                  cfl(ii,j,kk,l_)=cfl(ii,j,kk,l_)+ax*az*cf(ii,j)
cBH180906: This is a point to accumalate contributions to expanded
cBH180906: eal/ebl/ecl arrays saving coll contributions to each general
cBH180906: species from itself and other species.
 460            continue
 470            continue
 480          continue ! i=1,imax(l,lr_)

            else ! (cqlpmod.eq."enabled")
              do 485 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_) Here: CQLP
                !CQLP: l=index along field line, 
                !      and here l takes the single value of l_
                !CQL3D: l=index along pol.angle, scanning 1:lz range,
                !      while l_ is fixed (given radial index)
                do 486 j=2,jx
                  cal(i,j,kk,l_)=cal(i,j,kk,l_)+ca(i,j) 
                  cbl(i,j,kk,l_)=cbl(i,j,kk,l_)+cb(i,j)
                  ccl(i,j,kk,l_)=ccl(i,j,kk,l_)+cc(i,j)
                  cdl(i,j,kk,l_)=cdl(i,j,kk,l_)+cd(i,j)
                  cel(i,j,kk,l_)=cel(i,j,kk,l_)+ce(i,j)
                  cfl(i,j,kk,l_)=cfl(i,j,kk,l_)+cf(i,j)
 486            continue
 485          continue
            endif

CPTR>>>REPLACE PTR-DSCAL2
            fscal= one/anr1
            call dscal(iymax*jx,fscal,ca(1,1),1)
            call dscal(iymax*jx,fscal,cb(1,1),1)
            call dscal(iymax*jx,fscal,cc(1,1),1)
            call dscal(iymax*jx,fscal,cd(1,1),1)
            call dscal(iymax*jx,fscal,ce(1,1),1)
            call dscal(iymax*jx,fscal,cf(1,1),1)
            call dscal(iymax*jx,one/satiom(kk,k),ca(1,1),1)
            call dscal(iymax*jx,one/satiom(kk,k),cd(1,1),1)
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)
CPTR<<<END PTR-DSCAL2

 490      continue ! kk=1,ngen

c     end of loop over gen. species k
 500    continue ! k=1,ngen

ccc      kk=1
ccc      do i=1,iy
ccc         prnt1(i)=ca(i,2)
ccc         prnt2(i)=cb(i,2)
ccc         prnt3(i)=cc(i,2)
ccc         prnt4(i)=cd(i,2)
ccc         prnt5(i)=ce(i,2)
ccc         prnt6(i)=cf(i,2)
ccc      enddo
      lll=l_  !to create point to stop


c     end of loop over orbit l
 600  continue

ccc      kk=1
ccc      do i=1,iy
ccc         prnt1(i)=cal(i,2,kk,l_)
ccc         prnt2(i)=cbl(i,2,kk,l_)
ccc         prnt3(i)=ccl(i,2,kk,l_)
ccc         prnt4(i)=cdl(i,2,kk,l_)
ccc         prnt5(i)=cel(i,2,kk,l_)
ccc         prnt6(i)=cfl(i,2,kk,l_)
ccc      enddo

 700  continue
c..................................................................
c     define needed coefficients at pass/trapped boundary
c..................................................................
      if (cqlpmod .ne. "enabled") then
        do 2001 k=1,ngen
          do 2002 j=1,jx
            cal(itl,j,k,l_)=0.25*vptb(itl,lr_) * ( cal(itl-1,j,k,l_)/
     /        vptb(itl-1,lr_)+2.*cal(itl+1,j,k,l_)/vptb(itl+1,lr_) +
     +        cal(itu+1,j,k,l_)/vptb(itu+1,lr_) )
            cbl(itl,j,k,l_)=0.25*vptb(itl,lr_) * ( cbl(itl-1,j,k,l_)/
     /        vptb(itl-1,lr_)+2.*cbl(itl+1,j,k,l_)/vptb(itl+1,lr_) +
     +        cbl(itu+1,j,k,l_)/vptb(itu+1,lr_) )
            cal(itu,j,k,l_)=cal(itl,j,k,l_)
            cbl(itu,j,k,l_)=cbl(itl,j,k,l_)
 2002     continue
 2001   continue
      endif
c..................................................................
      if (madd .eq. 2) call cfpsymt
      

      do 2000 k=1,ngen
        fscal= one/tnorm(k)  !tnorm=vnorm**3/(GAM1*one_), see ainvnorm.f
        call dscal(iymax*jx,fscal,cal(1,1,k,l_),1)
        call dscal(iymax*jx,fscal,cbl(1,1,k,l_),1)
        call dscal(iymax*jx,fscal,ccl(1,1,k,l_),1)
        call dscal(iymax*jx,fscal,cdl(1,1,k,l_),1)
        call dscal(iymax*jx,fscal,cel(1,1,k,l_),1)
        call dscal(iymax*jx,fscal,cfl(1,1,k,l_),1)
        call dscal(iymax*jx,fscal,eal(1,1,k,1,l_),1)
        call dscal(iymax*jx,fscal,ebl(1,1,k,1,l_),1)
        call dscal(iymax*jx,fscal,eal(1,1,k,2,l_),1)
        call dscal(iymax*jx,fscal,ebl(1,1,k,2,l_),1)
        !YuP[2021-03-11] Changed iy-->iymax 
        !(just in case if iy is changed by iy=iy_(l_) somewhere)

c..................................................................
c     For the case that colmodl=3, a positive definite operator
c     is not guaranteed. This is a bastardized, hybrid collisional
c     model (see the input information in the input deck or the
c     user manual) and should be used with care. Caveat emptor!
c     In any case if we get negative diffusion from the model,
c     it is set to zero to keep the code from blowing up.
c..................................................................

        if (colmodl.eq.3) then
          do 2004 j=1,jx
            do 2005 i=1,iymax
            !YuP[2021-03-11] Changed iy-->iymax 
            !(just in case if iy is changed by iy=iy_(l_) somewhere)
              if(cbl(i,j,k,l_).lt.0.) then
                 cbl(i,j,k,l_)=em100
              endif
              if(cfl(i,j,k,l_).lt.0.) then
                 cfl(i,j,k,l_)=em100
              endif
 2005       continue
 2004     continue
        endif
 2000 continue
 
!           k=1
!           write(*,*)'   n,k,l_=',n,k,l_
!           gam1=4.*pi*(charge*bnumb(k))**4/fmass(k)**2
!           do j=1,jx
!             if((j/15)*15.eq.j)then
!                write(*,'(a,i4,1pe12.4,1p7e11.3)')
!     +          'j, urel[cm/s], enerkev, A,B,C,D,E,F[cgs]=',
!     +           j, x(j)*vnorm, enerkev(j,k), 
!     +           cal(2,j,1,1)*tnorm(k),
!     +           cbl(2,j,1,1)*vnorm*tnorm(k),
!     +           ccl(2,j,1,1)*tnorm(k),
!     +           cdl(2,j,1,1)*tnorm(k),
!     +           cel(2,j,1,1)*vnorm*tnorm(k),
!     +           cfl(iyh,j,1,1)*tnorm(k)/vnorm
!             endif
!             ! *tnorm is to remove fscal factor applied above.
!           enddo
           !pause


c Bug (from Gary Kerbel, Oct 31, 2005) needs checking:
c Take difference of eal (linear part for electron + 2nd part from ions)
c and cal (sum of linear and nonlinear) ==> just NL component, species
c scattering from itself.  If Maxwl, NL is same as L.  In limit of
c small grid, (cal-eal)/eal should equal 1.
c But it doesn't.  Check kerbel email, Oct 31, 2005.

      return
      end
      
c=======================================================================
c=======================================================================
      subroutine cfp_integrals_maxw
      !YuP[2020-07-02] Adapted from FOW version:
      !YuP[08-2017] Calculate certain integrals 
      ! needed for subr. cfpcoefn.
      ! These integrals describe a contribution to BA coll.coefs
      ! from local collisions of general species with the background 
      ! Maxwellian species (search "kbm=ngen+1,ntotal").
      ! These integrals only depend on mass (fmass)
      ! and local temperature of these (Maxwellian) species.
      ! So, instead of calculating them over and over again
      ! at each point along particle orbit (of the general species),
      ! calculate them once as a table over temperature grid 
      ! ("T-grid" below), and then reuse them by matching a local T
      ! along orbit with the nearest values in the T-grid.
      ! These integrals may need to be updated if the temp() of 
      ! the Maxwellian species is varied in time (as a prescribed form).
      ! This subr. is called from tdchief, just after call_profiles,
      ! before "do 600 k=1,ngen" loop.
      ! The speedup from using this subroutine is factor of 5x-9x !
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
        
      ! INPUT through comm.h: temp(ntotala,0:lrza),fmass(ntotala),etc
      
      ! Local working arrays:
      real*8 wk1(jx),wk2(jx),wk3(jx),wk4(jx),wk5(jx),wk6(jx)

      ! OUTPUT: 
      ! temp_min(nmaxw_sp), temp_max(nmaxw_sp) ==
      !           min/max of temp() for each Maxw.species, to set 
      !           a uniform T-grid within [temp_min; temp_max] range;
      !           Different for each Maxwellian species!
      ! cfpm(3,nmaxw_sp,ntemp,jx) ==
      !           storage of three integrals, 
      !           for each Maxw.species (1:nmaxw_sp),
      !           set as a table over 1:ntemp uniform T-grid,
      !           saved as a function of j index (func. of momentum). 
      ! Set ntemp=100 ! 100 points is sufficient to represent 
      ! all possible values of temp(kbm,lr) profile; (Better use 200)
      ! so typically 3*6*100*200=360000 data points.
      !
      !YuP[2020-07-16] The number of Maxw species is extended 
      !   to include impurities.
      !In nmaxw_sp=nmax+nstates+2, the contribution is from:
      ! nmax= Regular Maxw. species (fully-ionized ions or/and electrons). 
      ! nstates+1= Partially-ionized impurity ions
      !   (kstate=1:nstates for ionized states, and kstate=0 for neutral state)
      !    For now, it is assumed that all ionization states have
      !    same temperature, equal to T for main ions, 
      !    temp(kion1,lr), where kion1=kionm(1).
      ! Additionally, +1 for bound electrons in impurity ions; 
      !    It is assumed that all bound electrons have temp_e_bound=10eV.
      !Note: if no impurity ions are present, nstates is 0.
      
      !cnorm2i=0. !YuP[07-2016] Bad way to control the relativistic nature 
                  !of background species. Value 0 means non-relativistic.
      cnorm2ii=1./cnorm2 !treat Maxw. species relativistically, see notes below.
      !Note: cnorm=clight/vnorm, cnorm2=cnorm^2
      
      do 10 kbm=ngen+1,ngen+nmaxw_sp !Maxw. species index 
         kk=kbm-ngen !Counter of Maxw. species, starting from 1
        ! These values depend on kbm:
        if(kk.le.ntotal-ngen)then !Or simply kk.le.nmax (regular Maxw.species)
          !kk= 1 : nmax
          fmass_sp=fmass(kbm)
          temp_min_kbm=MINVAL(temp(kbm,:)) !min value of T for Maxw. species kbm
          temp_max_kbm=MAXVAL(temp(kbm,:)) !max value of T for Maxw. species kbm
        elseif(kk.le.nmax+nstates+1)then
          !impurity ions with different ionization states
          !kk= (nmax+1 : nmax+nstates+1) covers all ionization states
          ! with kstate=0:nstates (including neutral state).
          fmass_sp=fmass_imp ! All ioniz.states have mass fmass_imp
          ! For now, assume all ionization states have same temperature, 
          ! equal to temper. of any Maxwellian ions, so that
          if(kionm(1).ne.0)then
            kion1=kionm(1)
          else
            WRITE(*,*)'cfp_integrals_maxw: kionm(1)=0; choose other.'
            stop 'cfp_integrals_maxw'
          endif
          !temp_imp(0:nstates,lr_)=temp(kion1,lr_) !BUT is it ok for kstate=0 (atom)?
          temp_min_kbm=MINVAL(temp(kion1,:)) !min value of T, for kion1
          temp_max_kbm=MAXVAL(temp(kion1,:)) !max value of T
          ! In future, we can generalize to MINVAL(temp_imp(kstate,:) )
          ! where kstate= kk-nmax-1 (range: kstate=0:nstates)
        elseif(kk.eq.nmax+nstates+2)then !recall that nmaxw_sp= nmax+nstates+2
          ! Bound electrons in impurity ions.
          fmass_sp=9.1095d-28 !(e)
          ! All bound electrons are assumed to be at T=10eV
          temp_e_bound=10.d-3 ! in keV
          temp_min_kbm=temp_e_bound !min value of T 
          temp_max_kbm=temp_e_bound !max value of T 
        endif
        ! min/max values of temp() could have changed; Check:
        if( (temp_min(kk).eq.temp_min_kbm).and.
     &      (temp_max(kk).eq.temp_max_kbm)     ) then
           !write(*,*)'cfp_integrals: skipped update for kbm=',kbm
           goto 10 !No change; next kbm
        endif
        !Otherwise, update these arrays, recalculate cfpm() tables:
        temp_min(kk)=temp_min_kbm ! min value of T for Maxw. species kbm
        temp_max(kk)=temp_max_kbm ! max value of T for Maxw. species kbm
        dtemp=(temp_max(kk)-temp_min(kk))/(ntemp-1) ! for a uniform T-grid
        if(dtemp.gt.0.d0)then
          ntemp1=ntemp
        else ! dtemp=0, which means a flat T profile; Consider itemp=1 only.
          ntemp1=1
        endif
        rstmss=fmass_sp*clite2/ergtkev ! get values from comm.h
        nintg0=41.*max(1.d0,sqrt(fmass_sp/1.67e-24)) !see below, nintg
        
        
        do itemp=1,ntemp1 ! scan the range [temp_min; temp_max],
                        ! calculate integrals
          ! Uniform T-grid:
          temp_loc= temp_min(kk)+dtemp*(itemp-1) ! [temp_min; temp_max]
          ! Determine the Maxwellian distribution associated with
          ! background species kbm. This will be a relativistic 
          ! Maxwellian for relativistic calculations.
          reltmp=rstmss/temp_loc ! m*c^2/T  ["theta_b" in manual/refs]
          if (reltmp .gt. 100.) then     ! low T(kbm)
            ebk2=sqrt(pi/(2.*reltmp))
          else if (reltmp .lt. .01) then ! high T(kbm)
            ebk2=2.*exp(reltmp)/reltmp**2
          else
            call cfpmodbe(reltmp,ebk1,ebk2)
          endif
          rnorm=reltmp/(4.*pi*cnorm3*ebk2)
          wk1=0.d0 ! 1:jx
          wk2=0.d0 ! 1:jx
          wk3=0.d0 ! 1:jx
          wk4=0.d0 ! 1:jx
          wk5=0.d0 ! 1:jx
          wk6=0.d0 ! 1:jx
          ! Need extra mesh points to represent ions on a mesh meant to
          ! support electrons. Need more resolution near zero.
          ! Split each velocity bin into nintg pieces and integrate.
          nintg=max(nintg0,min0(51,int(x(2)/sqrt(.2*cnorm2/reltmp))))
          nintg=2*(nintg/2)+1
          do ibm=1,nintg
            sfac=4.+2.*(2*(ibm/2)-ibm)
            if (ibm.eq.1 .or. ibm.eq.nintg) sfac=1.
            sfac=sfac*rnorm/(3.*(nintg-1))
            dnintg=(ibm-1)/dfloat(nintg-1)
            do jbm=2,jx
              xx=x(jbm)-dnintg*dxm5(jbm) ! get values from comm.h
              xx2=xx*xx
              xc2=xx2*cnorm2ii !YuP: it was cnorm2i=0 when relativ="disabled"
            ! YuP[07-2017] This is not quite correct - cnorm2i is set 
            ! for 'ka' general species, but here we consider 'kbm' Maxw.species.
            ! Should they be treated as relativistic or not?
            ! They can be different (relativistically) from 'ka' species.
            ! There should be a separate setting of relativ() for 
            ! Maxwellian species, too.
              gg2=1.d0+xc2 ! ==  1 + (v/c)^2
              gg=sqrt(gg2)
              if(xc2 .gt. 1.e-5) then ! relativistic case
                 expon=gg-1.d0
              else
                 expon=.5*xx2/cnorm2  
              endif
              fff=sfac*exp(-expon*reltmp)*dxm5(jbm)
              x4g2f=(xx2*xx2/gg2)*fff
              wk1(jbm)=wk1(jbm)+xx*gg*fff  ! for M0 integral
              wk2(jbm)=wk2(jbm)+xx*fff     ! for M0'
              wk3(jbm)=wk3(jbm)+xx2*fff    ! for N0
              wk4(jbm)=wk4(jbm)+xx2*fff/gg ! for N0'
              wk5(jbm)=wk5(jbm)+x4g2f      ! for E0
              wk6(jbm)=wk6(jbm)+x4g2f/gg   ! for E0'
            enddo ! jbm
          enddo ! ibm
          tam2(jx)=0. ! defined/stored in comm.h
          tam3(1)=0.
          tam5(1)=0.
          tam6(jx)=0.
          tam7(1)=0.
          tam9(1)=0.
          ! tam2(jx) and tam6(jx) represent integrals from xmax to Inf.
          ! The following coding performs this integration. 
          ! In this case 21*xmax represents infinity. 
          ! The lack of this piece is most obvious when ions are 
          ! a general species and electrons are fixed Maxwellians.
          do ll2=1,21
            sfac=4.+2.*(2*(ll2/2)-ll2)
            if ( ll2.eq.1 .or. ll2.eq.21) sfac=1.
            sfac=sfac*rnorm*x(jx)/60.
            do ll1=1,20
              xx=(realiota(ll1)+.05*realiota(ll2-1))*x(jx)
              xx2=xx**2
              xc2=xx2*cnorm2ii !was cnorm2i=0 when relativ="disabled"
              gg=sqrt(1.+xc2)
              if(xc2 .gt. 1.e-5) then  ! relativistic case
                 expon=gg-1.
              else
                 expon=.5*xx2/cnorm2  
                 !do not use xc2 here: can be 0 when relativ="disabled"
              endif
              fff=sfac*exp(-expon*reltmp)
              tam2(jx)=tam2(jx)+xx*gg*fff
              tam6(jx)=tam6(jx)+xx*fff
            enddo ! ll1
          enddo ! ll2
 
 
          do j=2,jx ! scan all j, calculate integrals(j)
          
              do jb=2,j ! 2:j
                jm=jb-1 ! 1:j-1
                tam3(jb)=tam3(jm)+wk3(jb) ! N0
                tam7(jb)=tam7(jm)+wk4(jb) ! N0'
                tam5(jb)=tam5(jm)+wk5(jb) ! E0
                tam9(jb)=tam9(jm)+wk6(jb) ! E0'
              enddo
              do jb= jx-1,j,-1 ! [jx-1 : j]
                jp=jb+1        ! [jx : j+1]
                tam2(jb)=tam2(jp)+wk1(jp) ! M0
                tam6(jb)=tam6(jp)+wk2(jp) ! M0'
              enddo
              tam10(j)=cog(0,1)*gamsqr(j)*
     &                (3.*tam7(j)+cnorm2ii*(2.*xm(j,3)*tam6(j)-tam9(j)))
              tam11(j)=cog(0,1)*
     &                (xsq(j)*tam2(j)+gamsqr(j)*xm(j,-1)*tam5(j))
              tam12(j)=cog(0,1)*
     &                (tam2(j)+1.5*xm(j,-1)*tam3(j)-.5*xm(j,-3)*tam5(j))

              ! Done - the integrals are found. Now save them into cfpm(1:3,1:nmax,1:ntemp,1:jx)
              cfpm(1,kk,itemp,j)= tam10(j)
              cfpm(2,kk,itemp,j)= tam11(j)
              cfpm(3,kk,itemp,j)= tam12(j)
              ! Note: kbm is from [ngen+1;ntotal] range,
              ! while cfpm is defined at kk=1:nmax indices.
              ! So, kk=kbm-ngen
          
          enddo ! j

        enddo ! itemp
        
 10   enddo ! kbm=ngen+1,ntotal ! Maxwellian species
      
      return
      end subroutine cfp_integrals_maxw
      


c=======================================================================
c=======================================================================
      subroutine cfp_integrals_get(kk,temp_loc) 
      !YuP[2020-07-16] Use Temperature-grid table for evaluation
      !of some integrals related to Maxwellian species.
      !OUTPUT: tam10(jx),tam11(jx),tam12(jx) [through comm.h]
      !See explanation in subr.cfp_integrals_maxw
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
      integer kk ! INPUT
      real*8 temp_loc ! INPUT
      integer itemp,itemp1,itemp2,j ! local
      real*8 dtemp,ttdt,tdt !local
      
      ! INPUT:
      !  kk=kbm-ngen ! Counter of Maxw. species, starting from 1.
      !  temp_loc= Temperature of species (from temp(kbm,lr), or other)
      ! INPUT from comm.h :
      !  temp_min(kk)=MINVAL(temp(kbm,:)) ! min of T for Maxw. species kbm
      !  temp_max(kk)=MAXVAL(temp(kbm,:)) ! max of T for Maxw. species kbm
      !     (For impurity ions, see subr.cfp_integrals_maxw)
      !  ntemp= number of 'itemp' points in T-grid
      !  cfpm(1:3,kk,itemp,j=1:jx) = Table values of integrals, 
      !    which were computed in subr.cfp_integrals_maxw
      
        ! Step size in a uniform T-grid:
        dtemp=(temp_max(kk)-temp_min(kk))/(ntemp-1) 
        ! Find the nearest itemp index, to match the given temp_loc and 
        ! temp_grid(itemp)= temp_min+dtemp*(itemp-1) in the T-grid: 
        if(dtemp.gt.0.d0)then
          !itemp= NINT( (temp_loc-temp_min(kk))/dtemp ) +1
          ! Note: NINT(0.49)=0,   and   NINT(0.51)=1
          ! Or better, a linear interpolation:
          ttdt= (temp_loc-temp_min(kk))/dtemp !can be 0.0 ..(ntemp-1.d0)
          itemp1= INT(ttdt)+1 !can be 1..(ntemp-1) [ntemp, if temp_loc=temp_max]
          itemp2= itemp1+1    !can be 2..ntemp   [ntemp+1, if temp_loc=temp_max]
          itemp2=min(itemp2,ntemp) !to make sure it doesnot exceed ntemp
          ! temp_loc is between temp_grid(itemp1) and temp_grid(itemp2)
          tdt= ttdt - (itemp1-1)
          ! tdt is to be used for a lin. interpolation, like this:
          ! tam10= cfpm(itemp1) + (cfpm(itemp2)-cfpm(itemp1))*tdt
          ! Now Get the values of integrals from the cfpm() table:
          do j=2,jx
            tam10(j)= cfpm(1,kk,itemp1,j) + 
     +               (cfpm(1,kk,itemp2,j)-cfpm(1,kk,itemp1,j))*tdt
            tam11(j)= cfpm(2,kk,itemp1,j) +
     +               (cfpm(2,kk,itemp2,j)-cfpm(2,kk,itemp1,j))*tdt
            tam12(j)= cfpm(3,kk,itemp1,j) +
     +               (cfpm(3,kk,itemp2,j)-cfpm(3,kk,itemp1,j))*tdt
          enddo    
        else ! dtemp=0, which means a flat T profile; Consider itemp=1 only.
          itemp=1
          do j=2,jx
            tam10(j)= cfpm(1,kk,itemp,j)
            tam11(j)= cfpm(2,kk,itemp,j)
            tam12(j)= cfpm(3,kk,itemp,j)
          enddo    
        endif

      return
      end subroutine cfp_integrals_get
      
      