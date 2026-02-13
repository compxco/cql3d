c
c
      subroutine tddiag
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     This routine computes radial diagnostics (total rf power 
c     and current)
c.......................................................................

      include 'param.h'
      include 'comm.h'

c.......................................................................
c     Compute the total current in AMPS, (currza).
c.......................................................................

      if (cqlpmod.eq."enabled") then
        print *," WARNING in tddiag: routine"," not yet ready for CQLP"
        !Except, it can compute this along field line:
        call starnue_sptz !Get starnue(),tauee(),taueeh(),sptzr(); all l_
        return
      endif


c.......................................................................
c     Bootstrap current calculation:
c     If jhirsh=88, use Hirshman '88 banana regime calc;
c     If jhirsh=99, use Sauter, Angioni, and Lin-Liu, PoP, 2834 (1999),
!         Corrected according to Erratum in Phys.Plasmas 2002,v.9,p.5140;
!         good now for any collisionality (coded by YuP 2019-12)
!         aspect ratios, general eqdsk data. For details,
!         see comments in subr. tdboothi.
!         This is the best jhirsh option. 
c     If jhirsh=0 , use Hinton and Haseltine multi-regime,
c     high-aspect ratio formula. 
c     (default=0, but 99 is best for comparison with cql3d).
c.......................................................................

      if (jhirsh.ne.0) then
         call tdboothi
      else
         call tdbootst
      endif
      
      !===> YuP[2019-12-19] Added calculation of resistivity.
      !Normally something similar is done in tdoutput, 
      !but not necessarily at every time step. 
      call starnue_sptz !Get starnue(),tauee(),taueeh(),sptzr(); all l_
      do lll=1,lrz 
          call tdnflxs(lll) ! determine l_,lr_, ls_, etc.
          call restcon ! for given l_, lr_
          !We would not need to call restcon at each time step,
          ! but because such output values as xconn are not saved
          ! for each lr_, we have to call it each time.
          ! resthks uses starnue(lr_) and xconn as input.
          ! starnue() was calculated in call_tdboothi above.
          starnue_save=starnue(lr_) ! save
          !YuP[2020-02-12] Call subr.resthks with starnue>0,
          !and use zressau2 as the value corresponding to starnue>0,
          !while zressau1 corresponds to starnue=0 
          !(so, no need to call this subr. twice)
          call resthks(l_,lr_,lmdpln_,
     &         zreshin(lr_),zreskim(lr_),zressau1,zressau2)
          sig_starnue(lll)=  1.d0/(zressau2*sptzr(l_)) ! sigma [cgs]
          sig_starnue0(lll)= 1.d0/(zressau1*sptzr(l_)) ! sigma [cgs]
!          !Before YuP[2020-02-12] :
!          !-1-> call resthks with collisional starnue:
!          !Neoclassical resistivity, including collisionality:
!          call resthks(l_,lr_,lmdpln_,
!     &         zreshin(lr_),zreskim(lr_),zressau1,zressau2)
!          sig_starnue(lll)= 1.d0/(zreshin(lr_)*sptzr(l_)) ! sigma [cgs]
!          !-2-> call resthks with starnue=0:
!          starnue(lr_)=zero
!          call resthks(l_,lr_,lmdpln_,
!     &         zreshin(lr_),zreskim(lr_),zressau1,zressau2)
!          sig_starnue0(lll)= 1.d0/(zreshin(lr_)*sptzr(l_)) ! sigma [cgs]
!          !-3-> Restore collisional starnue
!          starnue(lr_)=starnue_save ! restored 
        !Save some values:
        !Note: elecfld is found below (after call_dgesv) as 
        ! elecfld(ll)=elecfldn(ll,nn,it)*300.d0 ! V/cm
        !Therefore, here elecfld is from previous iteration 'it'
        elec_cgs=elecfld(lr_)/300.
        currpar_starnue(lll)= elec_cgs*sig_starnue(lll)/3.d9 ![A/cm^2]
        currpar_starnue0(lll)=elec_cgs*sig_starnue0(lll)/3.d9 ![A/cm^2]
        !From printout, currpar_* is same order of magn. 
        !as current based on distr.func.
        !Save values at this time step - to be used at next step:
        sig_starnue_n(lll)= sig_starnue(lll)
        sig_starnue0_n(lll)=sig_starnue0(lll)
        currpar_starnue_n(lll)= currpar_starnue(lll)
        currpar_starnue0_n(lll)=currpar_starnue0(lll)
        bscurm_n(lll)=bscurm(lll,1,2) !YuP[2019-12-18] saved
        k=1 !for 1st gen.species (supposed to be electrons; if not - then 0.0)
        curra_n(lll)=curra(k,lll) !RE current [cgs] !YuP[2020-10-30]
        !bscurm(1:lrz,1,2) is for '1'==electrons, '2'==non-maxwellian
         if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
         write(*,'(a,3e14.6)')'rya(lll), bscurm_n, dj[A/cm2]',
     &   rya(lll), bscurm_n(lll), 
!     1   elec_cgs
     &   currpar_starnue_n(lll)-currpar_starnue0_n(lll)
         endif
      enddo ! lll YuP[2019-12-19] done
      write(*,*)'---tddiag: n=',n
      

      psynct=0.0
      do 1 k=1,ngen
        currza(k)=0.
        rfpwrt(k)=0.
        gkpwrt(k)=0.
        pegyt(k)=0.0
        pplosst(k)=0.0
        wparzt(k)=0.
        wperpzt(k)=0.

c..................................................................
c     Compute the total RF power absorbed.
c     Also compute the integrals of the perpendicular and 
c     parallel energy (units changed from ergs to joules).
c..................................................................

        do 2 ll=1,lrzmax
          currza(k)=currza(k)+darea(ll)*currz(k,ll)
          rfpwrt(k)=rfpwrt(k)+dvol(ll)*rfpwrz(k,ll)
          gkpwrt(k)=gkpwrt(k)+dvol(ll)*gkpwrz(k,ll)
          pegyt(k)=pegyt(k)+dvol(ll)*pegyz(k,ll)
          pplosst(k)=pplosst(k)+dvol(ll)*pplossz(k,ll)
          wparzt(k)=wparzt(k)+dvol(ll)*wpar(k,ll)*1.e-7
          wperpzt(k)=wperpzt(k)+dvol(ll)*wperp(k,ll)*1.e-7
          if(k.eq.kelecg)  psynct=psynct+dvol(ll)*psyncz(ll)
 2      continue
 1    continue

c..................................................................
c     Compute the total current (Amps) - currtza
c     Compute the total current with electron compensating effects
c     Compute the current integral up to rz(l); currtzi(l)
c     Compute the compensated current up to rz(l); currtpzi(l)
c
c     ccurtor(lr_) is the cumulative toroidal current, integrating
c                   curtor in poloidal cross-section (Amps),
c                   accounting for pol variation of tor current.
c     ccurpol(lr_) is the cumulative poloidal current, integrating
c                   currpol over area in toroidal cross-section
c                   at the outer equatorial plane (Amps).
c..................................................................

c     Presently using electron component of bootstrap in
c     total current.  Need to adjust this (BobH, 990821).

      do k=1,2
      do kk=1,2
         bscurmi(0,k,kk)=0.
      enddo
      enddo
      currtzi(0)=0.
      currtpzi(0)=0.
      totcurzi(0)=0.
      ccurtor(0)=0.
      ccurpol(0)=0.
      do 4 ll=1,lrzmax
        currtzi(ll)=currtzi(ll-1)+darea(ll)*currtz(ll)
        currtpzi(ll)=currtpzi(ll-1)+darea(ll)*currtpz(ll)
        do k=1,2
        do kk=1,2
           bscurmi(ll,k,kk)=bscurmi(ll-1,k,kk)+darea(ll)*bscurm(ll,k,kk)
        enddo
        enddo
        totcurz(ll)=bscurm(ll,1,1)+currtpz(ll)
        totcurzi(ll)=totcurzi(ll-1)+darea(ll)*totcurz(ll)
        ccurtor(ll)=ccurtor(ll-1)+darea(ll)*curtor(ll)*
     +       rpcon(ll)*onovrp(2,ll)/onovrp(1,ll)
        ccurpol(ll)=ccurpol(ll-1)+twopi*rpconz(ll)*
     +       (rpmconz(ll)-rpmconz(ll-1))*curpol(ll)
 4    continue
      do k=1,2
      do kk=1,2
         bscurma(k,kk)=bscurmi(lrzmax,k,kk)
      enddo
      enddo
      currtza=currtzi(lrzmax)
      currtpza=currtpzi(lrzmax)
      totcurza=currtpza+bscurma(1,1)
c
c..................................................................
c     Compute the total source power integrated over space and summed
c     over all beam species
c     Compute the inductance, li. (Needs work for eqmod.ne."enabled":
c        expressions for bpolsqaz and bpolsqlm).
c..................................................................
      
      do k=1,ngen
        sorpw_rfi(k,0)=0.0
        sorpw_nbii(k,0)=0.0
        do ll=1,lrzmax
           sorpw_rfi(k,ll)=sorpw_rfi(k,ll-1)+sorpw_rf(k,ll)*dvol(ll)
           sorpw_nbii(k,ll)=sorpw_nbii(k,ll-1)+sorpw_nbi(k,ll)*dvol(ll)
        enddo
      enddo
      
      sorpwti(0)=0.0
      do 11 ll=1,lrzmax
        sorpwti(ll)=sorpwti(ll-1)+sorpwt(ll)*dvol(ll)
 11   continue
 
      sorpwtza=sorpwti(lrzmax)

      volume=0.
      do 14 ll=1,lrzmax
         volume=volume+dvol(ll)
 14   continue
 
      li=0.     
      if (eqmod.eq."enabled") then
         do 15 ll=1,lrzmax
            li=li+bpolsqaz(ll)*dvol(ll)
 15      continue
         li=li/volume/bpolsqlm
      endif

c..................................................................
c     Compute beam current drive figure of merit, eta, for both
c     cases (with compensating electrons and without).
c     =N(10**14)*radmaj(10**2)*currtza/sorpwtza
c
c     edenlavg=line average density.
c..................................................................

      eden=0.
      edenlavg=0.0
      etemp=0.
      ethtemp=0.
      edntmp=0.
      pden=0.
      pdntmp=0.
      do 30 k=1,ntotal
        if ((k.eq.kelecg.and.kelecm.eq.0) .or. k.eq.kelecm) then
          if (k.eq.kelecg.and.(colmodl.eq.1.or.colmodl.eq.3)) go to 21
          do 20 ll=1,lrzmax
            eden=eden+reden(k,ll)*dvol(ll)
            etemp=etemp+energy(k,ll)*dvol(ll)
            ethtemp=ethtemp+temp(k,ll)*dvol(ll)
            edntmp=edntmp+reden(k,ll)*dvol(ll)*energy(k,ll)
 20       continue
          edenlavg=(rpcon(1)-rmcon(1))*reden(k,0)
          do 22 ll=2,lrzmax
            edenlavg=edenlavg+(rpcon(ll)-rpcon(ll-1)+rmcon(ll-1)
     1        -rmcon(ll))*0.5*(reden(k,ll)+reden(k,ll-1))
 22       continue
 21       continue
        elseif (k.gt.ngen) then
          do 40 ll=1,lrzmax
            pden=pden+reden(k,ll)*dvol(ll)
            pdntmp=pdntmp+reden(k,ll)*dvol(ll)*energy(k,ll)
 40       continue
        endif
 30   continue
      edntmp=edntmp*2./3./eden
      if (pden.ne.zero) then
        pdntmp=pdntmp*2./3./pden
        pden=pden/volume
      endif
      eden=eden/volume
      edenlavg=edenlavg/(rpcon(lrzmax)-rmcon(lrzmax))
      etemp=(2./3.)*etemp/volume
      ethtemp=ethtemp/volume
      coef=eden/1.e+14*radmaj*.01/(sorpwtza+em90)

c..................................................................
c     Compute the figures of merit for c.d. efficiency
c..................................................................

      fom=coef*currtza
      fomp=coef*currtpza
      fompla=fomp*edenlavg/eden
      fomtot=coef*(currtpza+bscurma(1,1))

c..................................................................
c     Compute total plasma energy (joules) in each species
c..................................................................

      do 50  k=1,ntotal
        energyt(k)=0.0
        do 51  ll=1,lrzmax
          energyt(k)=energyt(k)+energy(k,ll)*reden(k,ll)*dvol(ll)
     +      *1.6e-16
 51     continue
 50   continue

      return
      end
