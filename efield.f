c
c
      subroutine efield
      implicit integer (i-n), real*8 (a-h,o-z)
      save
      include 'param.h'
      include 'comm.h'

      character*8 elecset

      real*8 elecfld_t0(lrza),sigma_t0(lrza),curr_t0(lrza) !YuP[2020-04-13] To save values at t=0 
c..................................................................
c     CQL3D mode:
c     At the n.ge.0 time-step:
c        Calculates sptzr, tauee, taueeh, starnue=nue_eff/nue_bounce
c     At n.ge.0:
c        Calculates elecfld(lr_) according to various methods
c        specified by efswtch and efswtchn.
c        sptzr is recalculated with Z.ne.1 corrections.
c..................................................................

      elecset="eoved"
      if (eoved.eq.zero) elecset="elecfld"
c      if (lrzmax.gt.1)   elecset="elecfld"
      call cfpgamma

c.......................................................................
c     Sets Z=1 sptzr resistivity value according to time step n=0 
c     parameters, to 1/sigma_parallel in Book's NRL plasma formulary.
c.......................................................................

      if (n .eq. 0) then
        if (cqlpmod .ne. "enabled") then
c     formula should be 0.5064*N(Z)*me/taueeh/ne/e**2, with N(Z=1)=1
c     Thus one should use game(e,e) as in taueeh, but as it is not
c     clear which to use we take the average between game(e,e) and game(e,i)
          sptzr(l_)=0.5064*4.*sqrt(2.*pi)/3.*charge**2
     *      *0.5*(gama(kelec,kelec)+gama(kelec,kionn))
     *      *sqrt(fmass(kelec))/(energy(kelec,lr_)/1.5*ergtkev)**1.5
        else ! (cqlpmod.eq."enabled")
          sptzr(l_)=0.5064*4.*sqrt(2.*pi)/3.*charge**2
     *      *0.5*(gama(kelec,kelec)+gama(kelec,kionn))
     *      *sqrt(fmass(kelec))/(enrgypa(kelec,ls_)/1.5*ergtkev)**1.5
c
c     divide by ne*Zeff = ni*Zeff**2
c     Hinton-Hazeltine p.270 and 297 (same as ONETWO 4.2-20, 4.2-30)
          taueeh(ls_)=vthpar(kelec,ls_)**3*fmass(kelec)**2
     1      /(4.*sqrt(2.*pi)*denpar(kelec,ls_)*charge**4*
     1      gama(kelec,kelec))*3./zeff(ls_)
          starnue(ls_)=rgeom(lr_)*bmod0(lr_)/bthr0(lr_)/
     1      vthpar(kelec,ls_)/taueeh(ls_)/eps(lr_)**1.5
        endif

      endif

c     The remainder of the subroutine computes only radial quantities
c     Thus done only when l_=lmdpln_ (as is the case in cql3d operation).

      if (l_ .ne. lmdpln_) return

c..................................................................
c     Use Connor formula to compute resistivity - should
c     be accurate for most values of invers aspect ratio.
c..................................................................

      call restcon

c..................................................................
c     Reset Z=1 Spitzer resistivity, vth,tauee,taueeh,elecr for n.ne.0
c..................................................................


c..................................................................
c     vth is the thermal velocity = sqrt(T/m) (at t=0 defined in ainpla)
c     But, T==temp(k,lr) can be changed in profiles.f, 
c     in case of iprote (or iproti) equal to "prbola-t" or "spline-t"
c..................................................................

      if (cqlpmod .ne. "enabled") then
        do k=1,ngen
         vth(k,lr_)=(temp(k,lr_)*ergtkev/fmass(k))**.5
        enddo
      else ! (cqlpmod.eq."enabled") ![2021-02-26]added
        do k=1,ngen
         vthpar(k,ls_)=(temppar(k,ls_)*ergtkev/fmass(k))**.5
        enddo
      endif
c..................................................................
c     tauee(lr_) is the electron-electron slowing down time at midplane
c     tauee = 0.532/nu^{ee}_s, where nu^{ee}_s is the low velocity
c     expression for ee-slowing in Book's NRL Plasma Formulary p.32,
c     0.532=2**(3/2)/(3*pi**0.5). As nu(ee)_s = 2/taueeh/Zeff, we
c     have tauee = 2**(3/2)/3/sqrt(pi)/2*taueeh*Zeff, thus
c     taueeh = (3*sqrt(pi/2)/Zeff)*tauee. 
c     Note taueeh = taue(NRL p.33)=2.9E-06*...
c..................................................................

      if (cqlpmod .ne. "enabled") then
        tauee(lr_)=vth(kelec,lr_)**3*fmass(kelec)**2
     1  /(4.*pi*reden(kelec,lr_)*charge**4*gama(kelec,kelec))
c     Hinton-Hazeltine definition, Eq.(5.4), for e-i collisions.
c     1./taueeh = 4/3 sqrt(2*pi) ne*zeff*charge**4*gamma/(sqrt(me)*Te**1.5)
c     Thus: taueeh=3 sqrt(pi/2) / Zeff * tauee
c     Note: taueeh and nuestar same as ONETWO 4.2-20 and 4.2-30
c     (Eq. 4.2-28 and 4.2-39, in GA-A16178, Pfeiffer, Davidson, Miller, Waltz)
        taueeh(lr_)=tauee(lr_)*3.*sqrt(pi/2.)/zeff(lr_)
        starnue(lr_)=rgeom(lr_)*bmod0(lr_)/bthr0(lr_)/
     1    vth(kelec,lr_)/taueeh(lr_)/eps(lr_)**1.5
c     formula should be 0.5064*N(Z)*me/taueeh/ne/e**2, with N(Z=1)=1
c     Thus one should use game(e,e) as in taueeh, but as it is not
c     clear which to use we take the average between game(e,e) and game(e,i)
c     This expression for sptzr is given by Eq. 4.2-77 in ONETWO manual,
c     GA-A16178, the same as Eq. 5.66 of Hinton-Hazeltine, Rev. Mod. Phys.
          sptzr(l_)=zeff(lr_)*(0.29+0.46/(1.08+zeff(lr_)))
     *      *4.*sqrt(2.*pi)/3.*charge**2
     *      *0.5*(gama(kelec,kelec)+gama(kelec,kionn))
     *      *sqrt(fmass(kelec))/(temp(kelec,lr_)*ergtkev)**1.5
      else ! (cqlpmod.eq."enabled") ![2021-02-26]added
        tauee(ls_)=vthpar(kelec,ls_)**3*fmass(kelec)**2
     1  /(4.*pi*denpar(kelec,ls_)*charge**4*gama(kelec,kelec))
        taueeh(ls_)=tauee(ls_)*3.*sqrt(pi/2.)/zeff(ls_)
        starnue(ls_)=rgeom(lr_)*bmod0(lr_)/bthr0(lr_)/
     1    vthpar(kelec,ls_)/taueeh(ls_)/eps(lr_)**1.5
          sptzr(l_)=zeff(ls_)*(0.29+0.46/(1.08+zeff(ls_)))
     *      *4.*sqrt(2.*pi)/3.*charge**2
     *      *0.5*(gama(kelec,kelec)+gama(kelec,kionn))
     *      *sqrt(fmass(kelec))/(temppar(kelec,ls_)*ergtkev)**1.5
      endif

 2    continue ! Not used

c..................................................................
c     elecr is the Dreicer electric field (as in Kulsrud et al.),
c     converted to volts/cm.
c..................................................................

      if (cqlpmod .ne. "enabled") then
       elecr(lr_)=300.*fmass(kelec)*vth(kelec,lr_)
     &           /(2.*charge*tauee(lr_))
      else ! (cqlpmod.eq."enabled") ![2021-02-26]added
       elecr(ls_)=300.*fmass(kelec)*vthpar(kelec,ls_)
     &           /(2.*charge*tauee(ls_))
      endif
c..................................................................
c     rovsf and rovscf are two small-epsilon approximations to
c     rovs(lr_) (see below)
c..................................................................

      rovsf=1./(1.-1.95*eps(lr_)**.5+.95*eps(lr_))
      rovscf=1./(1.-2.09*eps(lr_)**.5)

c..................................................................
c     Toroidal electric field:
c     In the following,currxj(lr_) is a target current density (A/cm^2).
c                      currtp(lr_) is flux surface area average parallel
c                                   current density, calc'd in diaggnde.
c                                   This quantity is depracated as a
c                                   nonphysical quantity, and should be
c                                   replaced (BH, 010225).
c                      curra(kelec,l_) is flux surface area average 
c                                   parallel current density due to
c                                   runaway electrons (momenturm .gt.
c                                   minimum (ucrit, 3.*clight) 
c                                   (see subs diaggnde and soucrit).
c..................................................................

c     Calculation of plasma current for use in elecfld determination
c     (efswtch.ne."method1").

      !YuP[2018-09-13] But further below, elecfld() is re-defined.
      ! Yup argued to move this section AFTER if(efswtch.eq....) block.
      ! BH is not so sure.  Need to consider meaning of efswtchn, per
      ! cqlinput_help.   On the other hand, there is perhaps a problem with
      ! setting elecfld() at the first time step, which shows up as an
      ! anomalous starting electric field in the wh80.ps RE case.
      ! BH180917:  Could use more checking.
      if (efswtchn.eq."neo_hh") then
         call restcon
         call resthks(l_,lr_,lmdpln_,
     +        zreshin(lr_),zreskim(lr_),zressau1,zressau2)
         !Original version that uses zreshin [Hinton and Hazeltine]:
!YuP         currpar(lr_)=(curra(kelec,l_)+
!YuP     +        elecfld(lr_)/300./(zreshin(lr_)*sptzr(l_)))/3.e9
         !YuP[2020-04-02] Version with zressau1(starnue=0) or zressau2(starnue>0) 
         !Based on Sauter, Angioni and Lin-Liu, Phys.Plasmas 6, 1834 (1999) :
         currpar(lr_)=(curra(kelec,l_)+
     +        elecfld(lr_)/300./(zressau1*sptzr(l_)))/3.e9
         !This currpar() is further used for efswtch=method2,3,4
         !(also for method5, but it is set separately in eflditer.f)
      else ! efswtchn.eq."disabled" (default value) Use total Ip (Ohmic+RE)
         currpar(lr_)=currtp(lr_)/3.e9  
      endif

      if (efswtch.eq."method1") then

c        if input variable eoved (E over E-Dreicer) is .ne. 0.
c        then compute the value of elecfld(lr_)
c        (For lbdry(kelec).eq."fixed", only do it for n=0).
         !write(*,*)'efield_/method1: n,lr,elecfld',n,lr_,elecfld(lr_)
         !write(*,*)'elecset,eoved,elecr(lr_)',elecset,eoved,elecr(lr_)

         if (kelecg.ne.0) then !=> kelec=kelecg (see ainspec.f)
            if (lbdry(kelecg).eq."fixed" .and. n.gt.0)  go to 31
         endif
         if (elecset.eq."eoved") elecfld(lr_)=eoved*elecr(lr_)
 31      continue
         !write(*,*)'efield=/method1: n,lr,elecfld',n,lr_,elecfld(lr_)

      elseif (efswtch.eq."method2") then

c        If time step .lt. noncntrl, use specified electric
c          field. Then relaxation towards specified current.
         if (n.lt.noncntrl) then

            if (kelecg.ne.0) then !=> kelec=kelecg (see ainspec.f)
               if (lbdry(kelecg).eq."fixed" .and. n.gt.0) go to 32
            endif
            if (elecset.eq."eoved") elecfld(lr_)=eoved*elecr(lr_)
 32         continue
            
         else
            
c           Relaxation towards target parallel currxj:
            
            currxj0(lr_)=currxj(lr_)
            elecfld(lr_)=elecfld(lr_)*(1.-efrelax*
c990131     +         (currpar(lr_) -currxj(lr_))/(sign(1.,currxj(lr_))
     +           (currpar(lr_) -currxj(lr_))/(sign(one,currxj(lr_))
c990131     +           *amax1(abs(currpar(lr_)),abs(currxj(lr_)))))
     +           *max(abs(currpar(lr_)),abs(currxj(lr_)))))
            !YuP[2019-10-29] Is currpar FSA current, and currxj - also FSA?
            !Then we don't need currpar(lr_)/psifct1 as in method5
            
         endif
         
      elseif (efswtch.eq."method3") then
         

c        If time step .lt. noncntrl, use specified electric
c          field. Then relaxation towards current obtained
c          up to time noncntrl (which is saved in currxj0).
 
         if (n.lt.noncntrl) then

            currxj0(lr_)=currtp(lr_)/3.e9  
            if (kelecg.ne.0) then !=> kelec=kelecg (see ainspec.f)
               if (lbdry(kelecg).eq."fixed" .and. n.gt.0) go to 33
            endif
            if (elecset.eq."eoved") elecfld(lr_)=eoved*elecr(lr_)
 33         continue
         elseif (n.eq.noncntrl) then

            currxj0(lr_)=currpar(lr_)
            elecfld(lr_)=elecfld(lr_)
            
         else
            
c           Relaxation towards saved target parallel currxj0:

            elecfld(lr_)=elecfld(lr_)*(1.-efrelax*
     +           (currpar(lr_) -currxj0(lr_))/(sign(one,currxj0(lr_))
c990131     +           *amax1(abs(currpar(lr_)),abs(currxj0(lr_)))))
     +           *max(abs(currpar(lr_)),abs(currxj0(lr_)))))

         endif


      elseif (efswtch.eq."method4") then  !-YuP: method4 only.
c        If initial time step, use spitzer+neoclassical to get
c          electric field near what is required for specifed 
c          target currxj (convert to volts/cm from cgs)

         if (n.eq.0) then

            currxj0(lr_)=currxj(lr_) 
            call restcon
            call resthks(l_,lr_,lmdpln_,
     +           zreshin(lr_),zreskim(lr_),zressau1,zressau2)
            !Original version that uses zreshin [Hinton and Hazeltine]:
!YuP            elecfld(lr_)=currxj(lr_)*(zreshin(lr_)*sptzr(l_))*300.*3.e9 !V/cm
            !YuP[2020-04-02] Version with zressau1 (small change in results from this):
            !Based on Sauter, Angioni and Lin-Liu, Phys.Plasmas 6, 1834 (1999) :
            elecfld(lr_)=currxj(lr_)*(zressau1*sptzr(l_))*300.*3.e9 !V/cm
!            write(*,'(a,i4,3e12.3)')
!     &          'efield.f [n=0]: lr,elecfld,totcurtt,currxj',
!     &           lr_,elecfld(lr_),totcurtt,currxj(lr_)
            
         else ! n>0

c           Relaxation towards target parallel currxj:

c           Presently, the calculated parallel current
c           in diaggnde is averaged over the area cross-section
c           (at cnst toroidal angle) between a flux surface.
c           This quantity is depracated, and should be replaced
c           in the future.  Here, for method5(YuP:4?), we adjust currpar 
c           back to parallel current at the minimum B position.
c           See comments in eflditer.f, on psifct.

            psifct1=1. ! Here: efswtch.eq."method4"
            !if(efswtch.eq."method5")then !YuP[2019-10-29] method5 ???
            !   psifct1=psiovr(lr_)/onovrp(1,lr_)
            !endif

!            write(*,*)'efield:elecfld,currpar/psifct,currxj',
!     +            elecfld(lr_),currpar(lr_)/psifct1,currxj(lr_)

            currxj0(lr_)=currxj(lr_) !for efswtch.eq."method4"
            
             !Original procedure: a linear type of response:
!             elecfld(lr_)=elecfld(lr_)*(1.d0-efrelax*
!     +           (currpar(lr_)/psifct1 -currxj(lr_))/
!     +           (sign(one,currxj(lr_))
!     +           *max(abs(currpar(lr_)/psifct1),abs(currxj(lr_)))))
     
            !YuP[2020-04-03] generalized the above to
            !(works better in some cases)
             djrel= (currpar(lr_)-currxj(lr_)) / 
     &              max( abs(currpar(lr_)) , abs(currxj(lr_)) )
             sign_dj=  sign(one,djrel*currxj(lr_))
             if(djrel.eq.0.d0) sign_dj=0.d0 !to give elecfld_new=elecfld_old
             elecfld(lr_)=elecfld(lr_)*
     &                   (1.d0 -efrelax*sign_dj*abs(djrel)**efrelax_exp)
             !Note that when efrelax_exp=1.0, it becomes
             !elecfld(lr_)= elecfld(lr_)*(1.-efrelax*sign_dj*abs(djrel))
             !which is same as the original procedure.
             ! Recommended alternative value: efrelax_exp=0.5
             ! which gives a faster convergence to the target current,
             ! although sometimes produces "jiggles" in E(t).

            !YuP[2019-10-29] Is currpar FSA current, and currxj - at the midplane?
            !Then we need currpar(lr_)/psifct1 -currxj(lr_), indeed.
            !If currxj - at the midplane, then xjc() or xjin_t() are at the midplane.
            !But help file says xjc is FSA, then currxj is FSA, and then
            ! we have to use currpar(lr_) -currxj(lr_)  in the above.
            !But then, why do we need 1/psifct1 in method5 ?

            !YuP[2020-04-02] Try direct definition; at n>0 similar to n=0 above:
!            call restcon
!            call resthks(l_,lr_,lmdpln_,
!     &           zreshin(lr_),zreskim(lr_),zressau1,zressau2)
!            elecfld(lr_)= (currxj(lr_)-curra(kelec,l_))
!     &                     *zressau1*sptzr(l_)*300.*3.e9 !V/cm
!            !Results: initially, current evolves in same way as in "efrelax" version,
!            !but then becomes unstable (when RE appear). 

            !write(*,*)'efield: n,lr,elecfld',n,lr_,elecfld(lr_)

         endif ! n


      elseif (efswtch.eq."method5"  .and.  n.eq.0) then  
          ! calculate 'seed' electric field before starting iterations
      
      
c-YuP: For method5, n>0, elecfld and currxj are calculated in eflditer.f;
c-YuP: currxj0 is set in tdrmshst (at n=0).
c          If initial time step, use spitzer+neoclassical to get
c          electric field near what is required for specifed 
c          target currxj (convert to volts/cm from cgs)
            call restcon
            call resthks(l_,lr_,lmdpln_,
     +           zreshin(lr_),zreskim(lr_),zressau1,zressau2)
            !Original version that uses zreshin [Hinton and Hazeltine]:
!YuP            elecfld(lr_)=currxj(lr_)*(zreshin(lr_)*sptzr(l_))*300.*3.e9
            !YuP[2020-04-02] Version with zressau1
            !Based on Sauter, Angioni and Lin-Liu, Phys.Plasmas 6, 1834 (1999) :
            elecfld(lr_)=currxj(lr_)*(zressau1*sptzr(l_))*300.*3.e9
            
      elseif (efswtch.eq."method6") then !YuP[2020-04-13] Added method6
            ! At t=0, save the value of elecfld and conductivity.
            ! At later time steps, use spitzer-neoclassical conductivity
            ! to evolve electric field so that (updated [2022-03-11])
            !  E(t)= [E(t=0)*sigma(t=0)+J_bs(0) - J_RE(t)-J_bs(t)]/sigma(t)
            !Aimed to yield J(t)=const, where J= Jcode+E*dsigma+J_bs
            !For method6, need to have iproelec.eq.'parabola' or 'spline'
            call restcon
            call resthks(l_,lr_,lmdpln_,
     +           zreshin(lr_),zreskim(lr_),zressau1,zressau2)
            !YuP[2020-04-02] Version with zressau1
            !Based on Sauter, Angioni and Lin-Liu, Phys.Plasmas 6, 1834 (1999) :
            !sigma=1.d0/(zressau1*sptzr(l_)) ! sigma (starnue=0) [cgs]
            ! Consider this option, for fully-collisional sigma:
            sigma=1.d0/(zressau2*sptzr(l_)) ! sigma (starnue>0) [cgs]
            if (n.eq.0) then
              elecfld_t0(lr_)= elecfld(lr_) ! Save, at t=0 
              sigma_t0(lr_)= sigma ! Save , at t=0
              !Assume J_RE(t=0)=0, but add a model-fit BS current:
              bscur= bscurm(lr_,1,2)+bscurm(lr_,2,1) !e_gen+i_maxw [A/cm^2]
              bscur= bscur*3.e9 ! to statA/cm^2
              curr_t0(lr_)= elecfld_t0(lr_)*sigma_t0(lr_)/300. +bscur ![cgs]
     +                     +curra(1,lr_) !YuP[2023-02-08] Added j_RE at t=0
            else ! n>0
              ! Note: In mixed V/cm and cgs units,
              ! E[V/cm] = eta[cgs]*curra[cgs]*300
              ! where 300 is to convert statVolt/cm to V/cm
              bscur= bscurm(lr_,1,2)+bscurm(lr_,2,1) !e_gen+i_maxw [A/cm^2]
              bscur= bscur*3.e9 ! to statA/cm^2
              fr1=1.0 !0.95! fraction of J_RE+J_bs to be subtracted
              djra= fr1*(curra(1,lr_)+bscur)
              sign_e= sign(one,elecfld_t0(lr_))
              if(sign_e.ge.0.d0)then !case E(t=0)>0
                djra=max(djra,0.d0) !Apply it only when it is positive.
                djra=min(djra,0.98*curr_t0(lr_)) !Not to exceed 0.98*curr_t0
              else !case E(t=0)<0 (and so is curr_t0)
                djra=min(djra,0.d0) !Apply it only when it is negative.
                djra=max(djra,0.98*curr_t0(lr_)) !Not to exceed 0.98*|curr_t0|
              endif
              elecfld(lr_)= 300.*(curr_t0(lr_)-djra)/sigma
              !updated [2022-03-11] - Now includes J_RE and J_bs,
              !and sigma is collisional
            endif ! n

         
      endif ! selection of efswtch=
      

      return
      end
