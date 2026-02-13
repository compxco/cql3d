c
c
      subroutine diaggnde2
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     Same as subroutine diaggnde but use Legendre decomposition for
c     computing the density and parallel current, so that the density
c     is the same as the one used in computing the collision operator.
c     It also allows one to have about a constant startup density
c     for the CQLP case, when the y-mesh is not the same on each
c     l mesh point.
c
c     Main diagnostic routine; computes densities, currents,
c     energies, z=effective, J/P. Also redefies electron density
c     to maintain quasineutrality if necessary.
c..................................................................

      include 'param.h'
      include 'comm.h'
c.......................................................................

c..................................................................
cl    1. Compute integrals for 0th and 1st moments using the same 
c     Legendre decomposition as in computing the collision term
c     Compute the other terms including bounce-averaged term in the same
c     way as diaggnde.
c     tam11 will contain midplane density moment (using Legendre coeff.)
c     tam13 will contain parallel current - at midplane. (with Leg. coeff.)
c..................................................................

      ilegen=1
      if (cqlpmod.eq."enabled") ilegen=l_

      currmt(l_)=0.
      currmtp(l_)=0.
      if (l_ .eq. lmdpln_) then
        currt(lr_)=0.
        currtp(lr_)=0.
      endif

      do 90 k=1,ngen
        call bcast(tam2,zero,jx)
        call bcast(tam4,zero,jx)
        call bcast(tam5,zero,jx)

c     compute Legendre decomposition from temp3 to tam1
c     m=0 contribution
        call dcopy(iyjx2,f(0,0,k,l_),1,temp3(0,0),1)
        call cfpleg(0,ilegen,1)
        do 10 j=1,jx
          tam11(j)=tam1(j)*4.*pi
 10     continue
        if (mx .eq. 0) go to 11
c     m=1 contribution
        call cfpleg(1,ilegen,1)
        do 12 j=1,jx
          tam13(j)=tam1(j)*4.*pi/3.
 12     continue
 11     continue

c     Compute other terms integrating explicitly over theta
        if (l_ .eq. lmdpln_) then
          do 20 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
            do 21 j=1,jx
c     include vptb=|cos(th0)| * tau
              tam2(j)=tam2(j)+f(i,j,k,l_)*cynt2(i,l_)*
     1          abs(coss(i,lmdpln_))*tau(i,lr_)
              tam4(j)=tam4(j)+f(i,j,k,l_)*cynt2(i,l_)*
     1          abs(coss(i,lmdpln_))*tau(i,lr_)*
     1          (1.-sinn(i,l_)**2*psiba(i,lr_))
              tam5(j)=tam5(j)+f(i,j,k,l_)*cynt2(i,l_)*
     1          abs(coss(i,lmdpln_))*tau(i,lr_)*
     1          (sinn(i,l_)**2*psiba(i,lr_))
 21         continue
 20       continue
        endif

        gn=0.
        en=0.
        hn=0.
        sn=0.
        cn=0.
        wpar_=0.
        wperp_=0.
        do 40 j=1,jx

c..................................................................
c     midplane density
c..................................................................

          gn=gn+tam11(j)*cint2(j)

c..................................................................
c     miplane current
c     currvs contains partial sums of the current; cn is the full
c     integral.
c..................................................................


          currv(j,k,l_)=tam13(j)*x(j)*gammi(j)*cint2(j)*dxi(j)
          if (j.eq.1) then
            currvs(j,k)=dx(1)*currv(j,k,l_)
          else
            currvs(j,k)=currvs(j-1,k)+currv(j,k,l_)*dx(j)
          endif
          cn=cn+currv(j,k,l_)*dx(j)

c..................................................................
c     field line density (particles/cm**2)
c..................................................................

          hn=hn+tam2(j)*cint2(j)

c..................................................................
c     Flux surface averaged energy (per particle)
c..................................................................

          sn=sn+tam2(j)*cint2(j)*tcsgm1(j)

c..................................................................
c     Midplane average energy per particle
c..................................................................

          en=en+tam11(j)*cint2(j)*tcsgm1(j)

c..................................................................
c     flux surface averaged parallel and perpendicular energy density
c..................................................................

          wpar_=wpar_+tam4(j)*cint2(j)*tcsgm1(j)
          wperp_=wperp_+tam5(j)*cint2(j)*tcsgm1(j)

 40     continue

c..................................................................
c     If density computed is negative call exit
c     Energies are in Kev or Kev/cm**3 (after multiplication by
c     fions)
c..................................................................

        if (gn.le.0.)  call diagwrng(10)
        energym(k,l_)=en/gn*fions(k)
        enrgypa(k,ls_)=en/gn*fions(k)
        if (l_ .eq. lmdpln_) then
          energy(k,lr_)=sn/hn*fions(k)
cBH080502          wpar(k,lr_)=wpar_*fions(k)/zmaxpsi(lr_)*ergtkev
cBH080502          wperp(k,lr_)=wperp_*fions(k)/zmaxpsi(lr_)*ergtkev
          wpar(k,lr_)=wpar_*fions(k)/hn
          wperp(k,lr_)=wperp_*fions(k)/hn
        endif

c..................................................................
c     At timet=0. scale the
c     distribution function to desired initial density.
c     NOTE: The momentum mesh, x, extends from 0. to 1.  The maximum
c     actual value of momentum/rest mass  or speed is vnorm.
c     A non-normalized mesh would extend from 0. to vnorm.
c     The distribution function in the code, f, will thus differ
c     from the cgs f (f_cgs) by the following f/vnorm**3=f_cgs
c     In what follows we scale f so that its initial MIDPLANE
c     density is reden.
c     Thus if f is integrated out using the code normalized velocity
c     mesh the integral will be density reden at timet 0.
c..................................................................

        if (n.eq.0) then
c%OS  
          print *,'diaggnde2: l_= ',l_,'  gn= ',gn
c%OS  
          gni=1./gn
          hni=0.0
          if (l_ .eq. lmdpln_) hni=1./hn
          if (l_ .eq. lmdpln_) hnis(k,lr_)=hni

c..................................................................
c     This assures the midplane density of reden for f
c..................................................................

          zfact=hni*zmaxpsi(lr_)*reden(k,lr_)
          if (cqlpmod.eq."enabled") zfact=gni*denpar(k,ls_)
cBH070414  if (.not. nlrestrt) call dscal(iyjx2,zfact,f(0,0,k,l_),1)
          if (nlrestrt.eq."disabled") 
     1         call dscal(iyjx2,zfact,f(0,0,k,l_),1)

c..................................................................
c     xlndn is the field line density (particles/cm**2)
c     xlndn/zmaxpsi(lr_) will equal the flux surface averaged (volume
c     averaged) density.
c..................................................................

          if (l_ .eq. lmdpln_) then
            xlndn(k,lr_)=reden(k,lr_)*zmaxpsi(lr_)
cBH080502            wperp(k,lr_)=wperp(k,lr_)*reden(k,lr_)*hni*zmaxpsi(lr_)
cBH080502            wpar(k,lr_)=wpar(k,lr_)*reden(k,lr_)*hni*zmaxpsi(lr_)
            wpar(k,lr_)=wpar_*fions(k)/hn
            wperp(k,lr_)=wperp_*fions(k)/hn
          endif

c..................................................................
c     currm is the non explicitly s dependent parallel current. 
c     In CQL3D, this is equal to the current at the midplane.
c     note:
c     current(z)=currm(k,l_)*bbpsi(z,lr__)
c     currm(k,lmdpln_)=parallel current at midplane
c
c     curr is the flux surface averaged current.
c     To wit: curr*dA=the total number of particles/sec passing through
c     an area element associated with the cross sectional area of the
c     flux surface in question. Once again curr and currm are
c     PARALLEL current.
c..................................................................


          faccur=reden(k,lr_)*hni*zmaxpsi(lr_)*vnorm*bnumb(k)*charge
          if (cqlpmod.eq."enabled") faccur=denpar(k,ls_)*vnorm*
     *      bnumb(k)*charge*gni
          currm(k,l_)=faccur*cn

c..................................................................
c     psifct will contain the factor needed to modify midplane current
c     to flux surface averaged current.
c..................................................................

c..................................................................
c     old bug next
c     psifct=radmaj/(radmaj**2-rovera(lr_)**2*radmin**2)**.5
c     for eqmod=disabled: psifct=sqrt((1+eps)/(1-eps))
c..................................................................

          psifct=psiovr(lr_)/onovrp(1,lr_)
          if (l_ .eq. lmdpln_) curr(k,lr_)=currm(k,l_)*psifct
          fgni=faccur*psifct  !/3.e+9 YuP[2021-03-23] removed 1/3.e9
          !Why is currv converted to A/cm2 here, at n=0, but not at n>0?
          !Leaving this conversion to pltendn and pltends,
          !where currv is plotted 
          if (cqlpmod.eq."enabled") fgni=faccur !/3.e+9 YuP[2021-03-23]
          call dscal(jx,fgni,currv(1,k,l_),1)
          call dscal(jx,fgni,currvs(1,k),1)
          !YuP[2021-03-23] :
          !Note that currvs() does not have l_ index, 
          !so it should be recomputed for each given l_, from currv(),
          !in pltendn and pltends (the only two places where it is used)

c..................................................................
c     Standard, non-initialization time step logic follows..
c..................................................................

        else ! n>0
          faccur=one_*vnorm*bnumb(k)*charge
          psifct=psiovr(lr_)/onovrp(1,lr_)

c..................................................................
c     xlndn(k,lr_) is the field line density
c     reden(k,lr_) is the flux averaged density
c     denpar(k,ls_) is the local in s density
c     currm(k,l_) is the local in s current (s:along the magnetic field)
c     curr(k,lr_) is the flux surface average current at l_=lmdpln_.
c     currt(lr_) is the current summed over all ionic general species.
c     currmt(l_) is the local in s current summed over all ionic g. species
c     currtp(lr_) adds in general electron contribution to currt(l_).
c     currmtp(l_) adds in general local in s electron contribution to currm
c..................................................................

          fgni=faccur*psifct  !Here: n>0
          if (cqlpmod.eq."enabled") fgni=faccur
          !YuP[2021-03-23] Corrected fgni factor for CQLP [i.e., no psifct]
          call dscal(jx,fgni,currv(1,k,l_),1)
          call dscal(jx,fgni,currvs(1,k),1)
          currm(k,l_)=faccur*cn
          currmtp(l_)=currmtp(l_)+currm(k,l_)
          if(cqlpmod.eq."enabled")then !YuP[2021-03-03] Added if()
          !YuP: denpar and enrgypa are only applicable for CQLP
          denpar(k,ls_)=one_*gn
          if (sbdry.eq."periodic")then !YuP/was .and. transp.eq."enabled") then
            !YuP: For sbdry="periodic", we could always set this
            !(i.e., transp or no transp, should not matter):
            if (ls_ .eq. 1) then
              denpar(k,lsmax+1)=denpar(k,1)
              enrgypa(k,lsmax+1)=enrgypa(k,1)
            else if (ls_ .eq. lsmax) then
              denpar(k,0)=denpar(k,lsmax)
              enrgypa(k,0)=enrgypa(k,lsmax)
            endif
          endif
          endif !cqlpmod.eq."enabled"
          if (l_ .eq. lmdpln_) then
            curr(k,lr_)=currm(k,l_)*psifct
            currtp(lr_)=currtp(lr_)+curr(k,lr_)
            xlndn(k,lr_)=one_*hn
            reden(k,lr_)=one_*hn/zmaxpsi(lr_)
          endif

        endif

 90   continue

c..................................................................
c     Above end of do loop over general (time advanced) species
c     Below subtract electron general species electron current
c     contribution to currt(lr_), currmt(l_).
c..................................................................

      if (kelecg.gt.0) then
        if (l_ .eq. lmdpln_) currt(lr_)=currtp(lr_)-curr(kelecg,lr_)
        currmt(l_)=currmtp(l_)-currm(kelecg,l_)
      else
        if (l_ .eq. lmdpln_) currt(lr_)=currtp(lr_)
        currmt(l_)=currmtp(l_)
      endif

c.......................................................................
c     following variables depend only on midplane values,
c     i.e. are strictly radial quantities
c.......................................................................

      if (l_ .eq. lmdpln_) then

c..................................................................
c     Compute current drive efficiency; Amps/watt
c     bdre(lr_) - ions only;   bdrep(lr_) - ions and electrons (kelecg.ne.
c     sorpwt(lr_) is the total source power (RF+NBI for urf and fr module
c..................................................................

        sorpwt(lr_)=0.
        do 85 kk=1,ngen
          sorpwt(lr_)=sorpwt(lr_)+sorpw_rf(kk,lr_)+sorpw_nbi(kk,lr_) !-YuP
 85     continue

c..................................................................
c     lrzmax.gt.0 flags a multi-flux surface (CQL3D) run.
c     Define dA/DV
c..................................................................

        if (lrzmax.gt.1.and.n.gt.0) then
          areaovol=darea(lr_)/dvol(lr_)
        else
          if (eqmod.eq."enabled") then
            areaovol=1./twopi*onovrp(1,lr_)
          else
            areaovol=1./twopi/radmaj
          endif
        endif

c..................................................................
c     sorpwt(lr_) is the combined nbi and rf power (from urf module)
c..................................................................

        if (n.gt.0) then
          if (sorpwt(lr_).ge.1.e-25) then
            bdre(lr_)=areaovol*currt(lr_)/(sorpwt(lr_)*3.e+9)
            bdrep(lr_)=areaovol*currtp(lr_)/(sorpwt(lr_)*3.e+9)
          else
            bdre(lr_)=0.
            bdrep(lr_)=0.
          endif
        endif

c......................................................................
c     Compute Z-effective
c......................................................................

      if(cqlpmod.ne."enabled")then !YuP[2021-02]
       zeff(lr_)=0.
       zeff4(lr_)=0.d0 !Yup[2014-05-27] Initialize to 0.
      else ! (cqlpmod.eq."enabled")
       zeff(ls_)=0.
       zeff4(ls_)=0.d0 
      endif 
      zeff1=0.
      xq=0.
      if (izeff.eq."backgrnd") then
        do 80 k=1,ntotal
           if (k.eq.kelecg .or. k.eq.kelecm) goto 80
           xq=xq+1
           if(cqlpmod.ne."enabled")then !YuP[2021-02]
            zeff(lr_)=zeff(lr_)+bnumb(k)**2*reden(k,lr_)
            zeff4(lr_)=bnumb(k)**4*reden(k,lr_)+zeff4(lr_)
            zeff1=zeff1+bnumb(k)*reden(k,lr_)
           else ! (cqlpmod.eq."enabled")
            zeff(ls_)=zeff(ls_)+bnumb(k)**2*denpar(k,ls_)
            zeff4(ls_)=bnumb(k)**4*denpar(k,ls_)+zeff4(ls_)
            zeff1=zeff1+bnumb(k)*denpar(k,ls_)
           endif
 80     continue
        !YuP[2020-06-22] if(gamafac.eq."hesslow".and.(kelec.ne.0))then
        if(nstates.gt.0)then !YuP[2020-06-22] Changed the above logic to this.
          ! If no suitable impurity is present 
          ! (example: imp_type is set to 0 in cqlinput),
          ! then nstates remains 0 ==> Effectively, no impurities.
          ! This logic is more general.
          ! Add ions from impurities, all charge states
          do kstate=1,nstates ! Now additional ions from impur.source.
           dens_kstate=dens_imp(kstate,lr_)
           xq=xq+1
           zeff(lr_)= zeff(lr_) +dens_kstate*bnumb_imp(kstate)**2
           zeff4(lr_)=zeff4(lr_)+dens_kstate*bnumb_imp(kstate)**4
           zeff1=     zeff1     +dens_kstate*bnumb_imp(kstate)            
          enddo ! kstate
        endif ! nstates.gt.0
        
      elseif (izeff.eq."ion") then
         do kk=1,nionm
            k=kionm(kk)
            xq=xq+1
            if(cqlpmod.ne."enabled")then !YuP[2021-02]
             zeff(lr_)=zeff(lr_)+bnumb(k)**2*reden(k,lr_)
             zeff4(lr_)=bnumb(k)**4*reden(k,lr_)+zeff4(lr_)
             zeff1=zeff1+bnumb(k)*reden(k,lr_)
            else ! (cqlpmod.eq."enabled")
             zeff(ls_)=zeff(ls_)+bnumb(k)**2*denpar(k,ls_)
             zeff4(ls_)=bnumb(k)**4*denpar(k,ls_)+zeff4(ls_)
             zeff1=zeff1+bnumb(k)*denpar(k,ls_)
            endif
         enddo
         
      else
         stop 'diaggnde2: Problem with izeff specification'
      endif
      
      if(cqlpmod.ne."enabled")then !YuP[2021-02]
       zeff4(lr_)=zeff4(lr_)/xq
       zeff(lr_)=zeff(lr_)/zeff1
      else ! (cqlpmod.eq."enabled")
       zeff4(ls_)=zeff4(ls_)/xq
       zeff(ls_)=zeff(ls_)/zeff1
      endif

c.......................................................................
c     end of strictly radial dependent variables
c.......................................................................

      endif

c.......................................................................
c     Compute the analytic (Cordey-Start) electron current degradation
c     factor for NBI type calculations. This assumes that electrons
c     are not a general species. If they are, the electron contribution
c     was computed above.
c.......................................................................

      if (kelecg.eq.0.and. eleccomp.eq."enabled") then
        if(cqlpmod.ne."enabled")then !YuP[2021-02]
          factor=diagcfac(eps(lr_),zeff(lr_))
        else ! (cqlpmod.eq."enabled") 
          factor=diagcfac(eps(lr_),zeff(ls_))
        endif
        if (l_ .eq. lmdpln_) currtp(lr_)=currt(lr_)*factor
        currmtp(l_)=currmt(l_)*factor
      endif

      if (l_ .ne. lmdpln_) go to 99

c..................................................................
c     Compute J/P which includes ion and electron currents
c..................................................................

      if (sorpwt(lr_).ge.1.e-25) then
        bdrep(lr_)=areaovol*currtp(lr_)/(sorpwt(lr_)*3.e+9)
      else
        bdrep(lr_)=0.
      endif

c..................................................................
c     Redefine electron density to maintain quasineutrality.
c..................................................................

      s=0.
      if (kelecm.eq.0 .or. kelecg.ne.0) goto 72
      if (qsineut .ne. "disabled") then
        if (qsineut.eq."maxwel") goto 71
        do 74 k=1,ngen
          s=s+xlndn(k,lr_)*zmaxpsii(lr_)*bnumb(k)
 74     continue
 71     do 70 k=ngen+1,ntotal
          if (k.eq.kelecm) goto 70
          s=s+reden(k,lr_)*bnumb(k)
 70     continue
        reden(kelecm,lr_)=s
        locquas="disabled"
      endif
 72   continue

c..................................................................
c     determine density as a function of poloidal angle.
c..................................................................

      call diagdenz

 99   continue
c
      return
      end
