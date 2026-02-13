c     
c     
      subroutine profiles
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c     
c.......................................................................
c     Obtain time-dependent
c     "parabolic" and "spline" profiles, for time-varying quantities.
c.......................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      real*8:: tmpt(njene)  !Temporary array
      dimension ztr(lrza)   !For tdoutput-like printout
      dimension rban_vth(lrzmax) ! for print-out
      character*8 ztext
      
      real*8 l_autocorr,lambda_mfp !local, for Drr from NIMROD coupling


      if (nbctime.gt.0) then
        if (tmdmeth.eq."method1") then
         itme=0
         do jtm=1,nbctime
            if (timet.ge.bctime(jtm)) itme=jtm
         enddo
         itme1=itme+1
        endif
        ! and continue
      elseif(iprote.eq.'prb-expt' .or. iproti.eq.'prb-expt'
     &   .or.iprote.eq.'spl-expt' .or. iproti.eq.'spl-expt'
     &                                                   )then
        !Here: nbctime= 0 which means none of profiles 
        ! is set as 'spline-t' or 'prbola-t',
        ! but Te profile is in decay mode:
        ! T(t)= Tend +(T(tstart)-Tend)*exp(-(t-tstart)/tau) for electrons or ions.
        ! where exp(-(t-tstart)/tau)  is applied only at t.ge.tstart.
        ! See H.M.Smith and E.Verwichte, PoP vol.15, p.072502, (2008),
        ! Eqn.(7).
        ! Update Te profile, then update vth() (further below in this subr)
        continue
      else
        return
      endif

      itab(1)=1
      itab(2)=0
      itab(3)=0

c     Densities of Charge states of impurity !YuP[2021-01-22] 
      !Update  dens_imp(0:nstates,1:lrz) 
      ! from dens_imp_t(0:nstates,1:njene,1:nbctime)
      if (read_data.eq.'nimrod') then

         !YuP[2021-08-20] First, form ryain array from t-dependent ryain_t
         if (itme.eq.0) then
             do l=1,njene
                tmpt(l)=ryain_t(l,1)
             enddo
         elseif (itme.lt.nbctime) then
             do l=1,njene
                tmpt(l)= ryain_t(l,itme)
     +                  +(ryain_t(l,itme1)-ryain_t(l,itme))
     +            /(bctime(itme1)-bctime(itme))*(timet-bctime(itme))
             enddo
         else
             do l=1,njene
                tmpt(l)= ryain_t(l,nbctime)
             enddo
         endif
         ryain(1:njene)=tmpt(1:njene) !Case read_data: ryain is formed at each t
         !This array is further used for spline of different *_t arrays.
      
         !Note: In case read_data='nimrod', nbctime>0 is enforced in ainsetva
         do kstate=0,nstates ! including neutral state (0)
           if (itme.eq.0) then
             do l=1,njene
                tmpt(l)=dens_imp_t(kstate,l,1)
             enddo
           elseif (itme.lt.nbctime) then
             do l=1,njene
                tmpt(l)= dens_imp_t(kstate,l,itme)
     +           +(dens_imp_t(kstate,l,itme1)-dens_imp_t(kstate,l,itme))
     +            /(bctime(itme1)-bctime(itme))*(timet-bctime(itme))
             enddo
           else
             do l=1,njene
                tmpt(l)= dens_imp_t(kstate,l,nbctime)
             enddo
           endif
           call tdinterp("zero","linear",ryain,tmpt,njene,rya(1),
     +                   tr(1),lrzmax)
           do ll=1,lrz
              dens_imp(kstate,ll)=tr(ll) ! dens_imp is updated
           enddo ! ll
         enddo ! kstate
         
         if(transp.eq."enabled")then !could add this condition later.
         if(n.gt.0)then ! skip it at n=0 (x(j) not defined yet)
            ![2021-08] Added, For NIMROD coupling, using data on dB/B. 
            !See subr.read_data_files: dbb2in_t(njene,nbctime)==(dB/B)^2
            !Below, Drr is formed at given time step, based on
            ![REFS]: Rechester, Rosenbluth, Phys.Rev.Lett. 40,40 (1978);
            !A pitch-angle dependence is added as in 
            !Harvey, McCoy, Hsu, Mirin, PRL 47, p.102 (1981).
            !First, interpolate in time and map from njene grid to lrz grid
            if (itme.eq.0) then
              do l=1,njene
                tmpt(l)= dbb2in_t(l,1)
              enddo
            elseif (itme.lt.nbctime) then
              do l=1,njene
                tmpt(l)= dbb2in_t(l,itme)
     +                 +(dbb2in_t(l,itme1)-dbb2in_t(l,itme))
     +               /(bctime(itme1)-bctime(itme))*(timet-bctime(itme))
              enddo
            else
              do l=1,njene
                tmpt(l)= dbb2in_t(l,nbctime)
              enddo
            endif
            call tdinterp("zero","linear",ryain,tmpt,njene,rya(1),
     +                    tr(1),lrzmax) !-> get tr(ll)
            !Now form Drr, based on subr. tdtrvshape,
            !but here it is re-arranged and simplified
            !to have a complete expression
            !for d_rr array at given time step, 
            !for the case of dB/B type of radial diffusion.
            k=kelecg ! Valid for electrons only
            do ll=1,lrz
              dbb2(ll)=tr(ll) !(dB/B)^2 is updated, at given time step
              l_autocorr=pi*qsafety(ll)*rmag !"Lc" or "L_A" in the above REFS
              !write(*,*)'ll,pi,qsafety(ll),rmag',ll,pi,qsafety(ll),rmag
              !Note: qsafety is found from eqdsk data [qar() array]
              !--For test only, find Drr at v=Vthermal :
                drr_max=0.d0 !for print only
                j_thermal=1 ! to initialize
                do j=jx,1,-1 ! scan backward
                 if(vth(k,ll) .le. x(j)*vnorm)then
                   j_thermal=j !such j that v(j_thermal) is just below vth()
                 endif
                enddo
              !--For test only/done.
              do j=1,jx
                vel=x(j)*vnorm/gamma(j) !velocity (not momentum/mass)
                coll=coll_freq(vel,k,ll) !nu_ei !Uses temp(),reden(),zeff()
                !Could move this whole section on Drr further down, 
                !after updating temp(),reden(),zeff() profiles,
                !but it is not very essential (just for the low-Vpar correction).
                gamm=gamma(j)**difus_vshape(4) ! [2022-01-06] added here,
                ! for 1/gamma^difus_vshape(4)  dependence 
                !(to account for finite orbital effects) .
                !Default value is difus_vshape(4)=0.0 which means gamm=1.
              do i= 1,itl_(ll)+1 !1,iytr(ll)
                vpar= abs(vel*coss(i,ll)) !Is using idx() correct here?
                lambda_mfp= max(em100,vpar/coll) !Mean-free-path for electrons
                coll_cutoff= (1.d0 + l_autocorr/lambda_mfp)
                if(j.eq.j_thermal)then !test/print
                 drr_max= max( drr_max, 
     &                  drr_scale*l_autocorr*vpar*dbb2(ll)/coll_cutoff )
                endif !test/print/done
                d_rr(i,j,k,ll)= drr_scale*
     &                                  l_autocorr*vpar*dbb2(ll)/
     &                                  (coll_cutoff*gamm) !== Drr
                !Note that at lambda_mfp >> l_autocorr, we get
                ! Drr= pi*qs*R*|Vpar|*(dB/B)^2
              enddo !i
              do i= itu_(ll)-1,iy_(ll) !iytr(ll)
                vpar= abs(vel*coss(i,ll)) !Is using idx() correct here?
                lambda_mfp= max(em100,vpar/coll) !Mean-free-path for electrons
                coll_cutoff= (1.d0 + l_autocorr/lambda_mfp)
                if(j.eq.j_thermal)then !test/print
                 drr_max= max( drr_max, 
     &                  drr_scale*l_autocorr*vpar*dbb2(ll)/coll_cutoff )
                endif !test/print/done
                d_rr(i,j,k,ll)= drr_scale*
     &                                  l_autocorr*vpar*dbb2(ll)/
     &                                  (coll_cutoff*gamm) !== Drr
                !Note that at lambda_mfp >> l_autocorr, we get
                ! Drr= pi*qs*R*|Vpar|*(dB/B)^2
              enddo !i              
              enddo !j  
              write(*,'(a,3i5,1p4e10.3)')
     &        'n,ll,iytr(ll),dbb2,l_autocorr,lambda_mfp, Drr_max(j_th)',
     &         n,ll,iytr(ll),dbb2(ll),l_autocorr,lambda_mfp, drr_max
            enddo ! ll
         endif !n>0
         endif !transp
         
      endif ! read_data.eq.'nimrod'


c     Temperatures


      !YuP[2019-09-18] added [[
      if (iprote.eq.'prb-expt' .or. iprote.eq.'spl-expt') then 
         do k=1,ntotal
         if (k.eq.kelecg .or. k.eq.kelecm) then
            ! or if(bnumb(k).eq.-1.)then
          !Note: parabolic Te(rho) profiles, 
          ! in cases of iprote= "parabola" or 'prb-expt' ,
          ! were setup in tdxinit
          ! based on values of temp(k,lr=0) and temp(k,1) in cqlinput,
          ! and using:
          ! call tdxin13d(temp,rya,lrzmax,ntotala,k,npwr(k),mpwr(k))
          !Here, we simply multiply them by time-dependent exponent.
          ! T(t)= Tend +(T(tstart)-Tend)*exp(-(t-tstart)/tau) for electrons.
          ! where exp(-(t-tstart)/tau)  is applied only at t.ge.tstart.
          ! See H.M.Smith and E.Verwichte, PoP vol.15, p.072502, (2008),
          ! Eqn.(7).
          ! temp_expt_Tend=0.010d0 ![keV] final(ending) Tend after cooling.
          ! temp_expt_tau0= 3.0d-3  ![sec]slow decay time of Te(t) 
          ! temp_expt_tau1= 0.1d-3  ![sec]fast decay time of Te(t)(for Thermal Quench)
          ! temp_expt_tstart(1:lrz)=0.d0  ![sec] tstart in the above Eqn.
          ! In case of pellet='enabled', it will be calculated during run
          ! to match the pellet position.
          do ll=1,lrz 
            if( timet.ge.temp_expt_tstart(ll)  
!     &          .and. dMpellet_dvol_sum(ll).gt.em100      
     &         )then
              !Pellet reached the surface ll. 
              !Fast drop of T with tau=temp_expt_tau1
              expt= exp(-(timet-temp_expt_tstart(ll))/temp_expt_tau1)
              temp(k,ll)= temp_expt_Tend +
     &                   (temp_expt_T1(k,ll)-temp_expt_Tend)*expt
              !scatfrac=0.d0 !YuP[2020-05-01] Just to try: 
              !disable pitch-angle scattering when T starts dropping
            elseif(imp_depos_method.eq.'pellet' 
     &       .and. timet.gt.pellet_tstart ) then
              !Pellet did not reach surface ll. But pellet is launched.
              !Start a slow collapse of T profile (with tau=temp_expt_tau0).
              !Keep saving temp() until the "Fast drop" section is started:
              temp_expt_T1(k,ll)=temp(k,ll) !Save T for the start of fast drop
              expt= exp(-(timet-pellet_tstart)/temp_expt_tau0)
              temp(k,ll)= temp_expt_Tend +
     &                   (temp_expt_T0(k,ll)-temp_expt_Tend)*expt
            endif
            !write(*,*)'profiles:temp(k,ll)=',n,temp(k,ll)
          enddo ! ll 
         endif ! k.eq.kelecg .or. k.eq.kelecm
         enddo ! k
      endif ! iprote='prb-expt'  !YuP[2019-09-18] added ]]

      !YuP[2019-09-18] added [[
      if (iproti.eq.'prb-expt' .or. iproti.eq.'spl-expt') then
         do k=1,ntotal
         if((k.ne.kelecg) .and. (k.ne.kelecm))then !ions
           !Or if (bnumb(k).ne.-1.) then
          !Note: parabolic Ti(rho) profiles, 
          ! in cases of iproti= "parabola" or 'prb-expt' ,
          ! were setup in tdxinit
          ! based on values of temp(k,lr=0) and temp(k,1) in cqlinput,
          ! and using:
          ! call tdxin13d(temp,rya,lrzmax,ntotala,k,npwr(k),mpwr(k))
          !Here, we simply multiply them by time-dependent exponent.
          ! T(t)= Tend +(T(tstart)-Tend)*exp(-(t-tstart)/tau) .
          ! where exp(-(t-tstart)/tau)  is applied only at t.ge.tstart.
          ! See H.M.Smith and E.Verwichte, PoP vol.15, p.072502, (2008),
          ! Eqn.(7).
          ! temp_expt_Tend=0.010d0 ![keV] final(ending) Tend after cooling.
          ! temp_expt_tau0= 3.0d-3  ![sec]slow decay time of Te(t) 
          ! temp_expt_tau1= 0.1d-3  ![sec]fast decay time of Te(t)(for Thermal Quench)
          ! temp_expt_tstart(1:lrz)=0.d0  ![sec] tstart in the above Eqn.
          ! In case of pellet='enabled', it will be calculated during run
          ! to match the pellet position.
          do ll=1,lrz 
            if( timet.ge.temp_expt_tstart(ll)  
!     &          .and. dMpellet_dvol_sum(ll).gt.em100      
     &         )then
              !Pellet reached the surface ll. 
              !Fast drop of T with tau=temp_expt_tau1
              expt= exp(-(timet-temp_expt_tstart(ll))/temp_expt_tau1)
              temp(k,ll)= temp_expt_Tend +
     &                   (temp_expt_T1(k,ll)-temp_expt_Tend)*expt
              !scatfrac=0.d0 !YuP[2020-05-01] Just to try: 
              !disable pitch-angle scattering when T starts dropping
            elseif(imp_depos_method.eq.'pellet'
     &        .and. timet.gt.pellet_tstart ) then
              !Pellet did not reach surface ll. But pellet is launched.
              !Start a slow collapse of T profile (with tau=temp_expt_tau0).
              !Keep saving temp() until the "Fast drop" section is started:
              temp_expt_T1(k,ll)=temp(k,ll) !Save T for the start of fast drop
              expt= exp(-(timet-pellet_tstart)/temp_expt_tau0)
              temp(k,ll)= temp_expt_Tend +
     &                   (temp_expt_T0(k,ll)-temp_expt_Tend)*expt
            endif
          enddo ! ll 
         endif ! k.ne.kelecg .and. k.ne.kelecm
         enddo
      endif ! iproti='prb-expt'  !YuP[2019-09-18] added ]]
      
         
      if (nbctime.gt.0 .and. iprote.eq."prbola-t") then
         do 5 k=1,ntotal
            if (tmdmeth.eq."method1") then
               if (itme.eq.0) then
                  temp(k,0)=tempc(1,k)
                  temp(k,1)=tempb(1,k)
               elseif (itme.lt.nbctime) then
                  temp(k,0)=tempc(itme,k)+(tempc(itme1,k)-tempc(itme,k))
     1                 /(bctime(itme1)-bctime(itme))
     1                 *(timet-bctime(itme))
                  temp(k,1)=tempb(itme,k)+(tempb(itme1,k)-tempb(itme,k))
     1                 /(bctime(itme1)-bctime(itme))
     1                 *(timet-bctime(itme))
               else
                  temp(k,0)=tempc(nbctime,k)
                  temp(k,1)=tempb(nbctime,k)
               endif
               !YuP: called later: call tdxin13d(temp,rya,lrzmax,ntotala,k,npwr(k),mpwr(k))
            elseif (tmdmeth.eq."method2")  then
               temp(k,0)=tdprof(timet,tempc(1,k),bctime)
               temp(k,1)=tdprof(timet,tempb(1,k),bctime)
            endif
            call tdxin13d(temp,rya,lrzmax,ntotala,k,npwr(k),mpwr(k))
 5       continue
      endif ! iprote=prbola-t


      if (nbctime.gt.0 .and. iprote.eq."spline-t") then

c        Electron temperature

         if (itme.eq.0) then
            do l=1,njene
               tmpt(l)=tein_t(l,1)
            enddo
         elseif (itme.lt.nbctime) then
            do l=1,njene
               tmpt(l)=tein_t(l,itme)+(tein_t(l,itme1)-tein_t(l,itme))
     +               /(bctime(itme1)-bctime(itme))*(timet-bctime(itme))
            enddo
         else
            do l=1,njene
               tmpt(l)=tein_t(l,nbctime)
            enddo
         endif

         do 16  k=1,ntotal
            if(bnumb(k).eq.-1.)  then
               call tdinterp("zero","linear",ryain,tmpt,njene,rya(1),
     +              tr(1),lrzmax)
               tr(0)=tmpt(1)
               do 19  ll=0,lrzmax
                  temp(k,ll)=tr(ll)
                  if(temp(k,ll).le.0.001)then ! rarely happens
CMPIINSERT_IF_RANK_EQ_0
                   WRITE(*,*)'WARNING: temp<0.001.  tmpt(1:njene)=',tmpt
                   WRITE(*,*)'temp<0.001.  temp(k,ll)=',k,ll,temp(k,ll)
CMPIINSERT_ENDIF_RANK  
                  endif
 19            continue ! ll
            endif
 16      continue ! k
      endif ! iprote=spline-t

      if (nbctime.gt.0 .and. iproti.eq."spline-t") then
c        Ion temperatures
         if (itme.eq.0) then
            do l=1,njene
               tmpt(l)=tiin_t(l,1)
            enddo
         elseif (itme.lt.nbctime) then
         do l=1,njene
            tmpt(l)=tiin_t(l,itme)+(tiin_t(l,itme1)-tiin_t(l,itme))
     +               /(bctime(itme1)-bctime(itme))*(timet-bctime(itme))
         enddo
         else
            do l=1,njene
               tmpt(l)=tiin_t(l,nbctime)
            enddo
         endif

         do 26  k=1,ntotal
            if(bnumb(k).ne.-1.)  then
               call tdinterp("zero","linear",ryain,tmpt,njene,rya(1),
     +              tr(1),lrzmax)
               tr(0)=tmpt(1)
               do 29  ll=0,lrzmax
                  temp(k,ll)=tr(ll)
 29            continue
            endif
 26      continue
      endif  ! (iproti.eq."spline-t")
            
         !YuP[2019-09-18] Moved this check (of T<0) from profiles.
         ! to ainsetva.
         !Need to check just once, before start of simulation
!      do it=1,nbctime      
!      if (tempc(it,kelec).le.zero .and. tein_t(1,it).le.zero) then
!         WRITE(*,*) "Time-dependent Te profile input problem at it=",it
!         STOP
!      endif
!      enddo
!
!c     Assume any time dep in tempc is in first ion species.    
!      do it=1,nbctime
!      if (tempc(it,kionn).le.zero .and. tiin_t(1,it).le.zero) then
!         WRITE(*,*) "Time-dependent Ti profile input problem at it=",it
!         STOP
!      endif
!      enddo

c     Renormalize temperatures using tescal/tiscal
      !Note: in case of 'prb-expt' .or. 'spl-expt' - No need to rescale.
      ! Already done in tdxinitl, at n=0
      if (iprote.eq.'prbola-t' .or. iprote.eq.'spline-t') then
        do k=1,ntotal
         if (bnumb(k).eq.-1.) then ! electrons
            do l=0,lrzmax
               temp(k,l)=tescal*temp(k,l)
            enddo
         endif
        enddo ! k
      endif ! (iprote.eq.'prbola-t' .or. iprote.eq.'spline-t')
      
      if (iproti.eq.'prbola-t' .or. iproti.eq.'spline-t') then
        do k=1,ntotal
         if (bnumb(k).ne.-1.) then ! ions
            do l=0,lrzmax
               temp(k,l)=tiscal*temp(k,l)
            enddo
         endif
        enddo ! k
      endif ! (iproti.eq.'prb-expt' .or. iproti.eq.'spl-expt')

      do k=1,ntotal
         do l=1,lrzmax
            if (temp(k,l).le.zero) then
              WRITE(*,*) "profiles.f: temp(k,l)<0 at k,l=",k,l
              STOP
            endif
         enddo ! l
      enddo ! k

c..................................................................
c     YuP[2018-01-05] Added resetting of vth() array in profiles.f.
c     vth is the thermal velocity =sqrt(T/m) (at t=0 defined in ainpla).
c     But, T==temp(k,lr) was changed above, 
c     in case of iprote (or iproti) equal to "prbola-t" or "spline-t".
c     Then, reset the values of vth() [used for plots, and also 
c     in lossegy, efield, restvty, sourceko, tdnpa, tdtrdfus,
c     tdoutput, and vlf*,vlh* subroutines].
      do k=1,ntotal
         do l=1,lrzmax
           vth(k,l)=(temp(k,l)*ergtkev/fmass(k))**.5
           if (k .eq. kelec) vthe(l)=vth(kelec,l)
         enddo
      enddo
      
      !YuP [2018-09-18] Added warning, similar to micxinit
      !Check that there are sufficient number of v-grid points  
      !over thermal part of the distribution (at least 3 points).
      ! If not, print warning.
      if(n.gt.0)then ! skip it at n=0 (x(j) not defined yet)
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,'(a)')"==============================================="
      WRITE(*,*)'profiles.f: time step n, timet=',n,timet
CMPIINSERT_ENDIF_RANK  
      do k=1,ngen
      do l=1,lrzmax
         j_thermal=1 ! to initialize
         do j=jx,1,-1 ! scan backward
           !write(*,*) j, vth(k,l)/vnorm, x(j)
           if(vth(k,l) .le. x(j)*vnorm)then
            j_thermal=j !it is such j that v(j_thermal) is just below vth()
           endif
         enddo
CMPIINSERT_IF_RANK_EQ_0
         if(l.eq.1 .or. l.eq.lrzmax)then
         WRITE(*,'(a,3i4,3f16.11)') 
     +   "profiles: k,lr, j_thermal, x(j_thermal), vth/vnorm, temp =",
     +         k, l, j_thermal, x(j_thermal), vth(k,l)/vnorm, temp(k,l)
         endif
         if(j_thermal.lt.3)then
           WRITE(*,'(a)')" WARNING(profiles.f): V-grid is too coarse."
           WRITE(*,'(a)')" Thermal part of distrib. is covered by only"
           WRITE(*,'(a,i5,a)')"    j=", j_thermal," points."
           WRITE(*,'(a)')" The solution may become unstable."
           WRITE(*,'(a)')" Consider increasing jx or setting xfac<1."
           !pause !-------
         endif
CMPIINSERT_ENDIF_RANK  
      enddo ! l=1,lrzmax 
      enddo ! k=1,ngen
      endif ! n>0
      
c     The energy() array, initially set as 1.5*temp() [see ainpla.f]
c     is re-defined at each time step in subr. diaggnde
c     as the m*v^2/2 average over distr.function in vel. space.
c     BUT, it is done for k=1:ngen species only!
c     So, below, we re-define the energy() array for the Maxwellian 
c     species only.
      do k=ngen+1,ntotal
         rstmss=fmass(k)*clite2/ergtkev
         do l=1,lrzmax
          thta=rstmss/temp(k,l)
          if (thta.gt.100. .or. relativ.eq."disabled") then
            energy(k,l)=1.5*temp(k,l)
          else
            call cfpmodbe(thta,bk1,bk2)
            energy(k,l)=rstmss*(bk1/bk2-1.+3./thta)
          endif
         enddo
      enddo
c     Another thought: Change the definition of vth() for k=1:ngen,
c     i.e. for the general species,
c     to be based on (2/3)*energy() rather than on temp() array. 
c     [Not implemented for the now]   
c..................................................................

      
c     Densities and Zeff  (cf. subroutine tdxinitl)
      
      if (nbctime.gt.0 .and. iprone.eq."prbola-t") then

      do 11 k=1,ntotal
                    
         if (iprozeff.ne."disabled" .and. 
     +           (k.ne.kelecg .and. k.ne.kelecm)) go to 11 !Ionic species:
                                            !Adjust to get Zeff(rho,time) 
             
         !For e_g and e_M continue, to follow the values from input list:    
         
         if((imp_depos_method.ne.'disabled') .and.
     &      (k.eq.kelecg.or.k.eq.kelecm)           )then
          !YuP[2020-06-24] Changed (gamafac.eq."hesslow") to (imp_depos_method.ne.'disabled')
          !   [a more general logic]
          !YuP[2019-12-06] There are two ways to adjust electron density:
          ! 1. Assume that density of main ion species is not changed; 
          !    then, electron density is set to sum_ni_Zi, see line~746.
          ! 2. Assume that electron density is not changed
          !   (or taken from the input list); 
          !    then, reduce the density of main ions, 1st ionic species.
          if(imp_ne_method.ne.'ne_list')then ! Option 1 in the above.
            !YuP[2019-09-18] Skip electrons in this setup
            !YuP[2019-09-18] Density of electrons and Zeff
            ! will be calculated consistently:
            ! ne will be found from reden() of ions(kion) and from 
            ! dens_imp(kstate,lr) of additional ions (from pellet, etc.)
            goto 11 !-> Option 1 in the above
          endif ! (imp_ne_method.ne.'ne_list')
          !Otherwise, continue with reden() below (iprone.eq.prbola-t)
         endif ! imp_depos_method.ne.'disabled'
         
         if (redenc(1,k).ne.zero) then
               
            if (tmdmeth.eq."method1") then

               if (itme.eq.0) then
                  reden(k,0)=redenc(1,k)
                  reden(k,1)=redenb(1,k)
               elseif (itme.lt.nbctime) then
                  reden(k,0)=redenc(itme,k)+
     1                 (redenc(itme1,k)-redenc(itme,k))
     1                 /(bctime(itme1)-bctime(itme))
     1                 *(timet-bctime(itme))
                  reden(k,1)=redenb(itme,k)+
     1                 (redenb(itme1,k)-redenb(itme,k))
     1                 /(bctime(itme1)-bctime(itme))
     1                 *(timet-bctime(itme))
               else
                  reden(k,0)=redenc(nbctime,k)
                  reden(k,1)=redenb(nbctime,k)
               endif
               !write(*,*)'profiles-2: k,reden(k,0)=',k,reden(k,0)

            elseif (tmdmeth.eq."method2") then

               reden(k,0)=tdprof(timet,redenc(1,k),bctime)
               reden(k,1)=tdprof(timet,redenb(1,k),bctime)

            endif


            call tdxin13d(reden,rya,lrzmax,ntotala,k,npwr(0),mpwr(0))
            
         endif
          
 11   continue

      endif  !On iprone.eq.prbola-t


      if (nbctime.gt.0 .and. iprone.eq."spline-t") then  !endif at l 466

      do  k=1,ntotal
         if (iprozeff.ne."disabled" .and. 
     +           (k.ne.kelecg .and. k.ne.kelecm)) go to 12
     
         if((imp_depos_method.ne.'disabled') .and.
     &      (k.eq.kelecg.or.k.eq.kelecm)           )then
          !YuP[2020-06-24] Changed (gamafac.eq."hesslow") to (imp_depos_method.ne.'disabled')
          !   [a more general logic]
          !YuP[2019-12-06] There are two ways to adjust electron density:
          ! 1. Assume that density of main ion species is not changed; 
          !    then, electron density is set to sum_ni_Zi, see line~746.
          ! 2. Assume that electron density is not changed
          !   (or taken from the input list); 
          !    then, reduce the density of main ions, 1st ionic species.
          if(imp_ne_method.ne.'ne_list')then ! Option 1 in the above.
            !YuP[2019-09-18] Skip electrons in this setup
            !YuP[2019-09-18] Density of electrons and Zeff
            ! will be calculated consistently:
            ! ne will be found from reden() of ions(kion) and from 
            ! dens_imp(kstate,lr) of additional ions (from pellet, etc.)
            goto 12 !-> Option 1 in the above
          endif ! (imp_ne_method.ne.'ne_list')
          !Otherwise, continue with reden() below (iprone.eq.spline-t)
         endif ! (imp_depos_method.ne.'disabled')
     
         if (enein_t(1,k,1).ne.zero) then

            if (itme.eq.0) then
               do l=1,njene
                  tmpt(l)=enein_t(l,k,1)
               enddo
            elseif (itme.lt.nbctime) then
            do l=1,njene
               tmpt(l)=enein_t(l,k,itme)+
     +              (enein_t(l,k,itme1)-enein_t(l,k,itme))
     +              /(bctime(itme1)-bctime(itme))*(timet-bctime(itme))
            enddo
            else
            do l=1,njene
               tmpt(l)=enein_t(l,k,nbctime)
            enddo
            endif
            
            call tdinterp("zero","linear",ryain,tmpt,njene,
     +           rya(1),tr(1),lrzmax)
            tr(0)=tmpt(1)
            do 13 ll=0,lrzmax
               reden(k,ll)=tr(ll)
 13         continue
         else
            do 9  ll=0,lrzmax
               reden(k,ll)=tr(ll)/abs(bnumb(k))
 9          continue
         endif
 12      continue
      enddo  !On k

      endif  !On iprone.eq.spline-t, l 419
      
      
c     Finish up with zeff and ions if iprozeff.ne."disabled"
      if (iprozeff.ne."disabled") then  !endif at line 724

         if(nbctime.gt.0 .and. iprozeff.eq."curr_fit")then ![2019-10-31]  to l 557
           !YuP[2019-10-31] Renormalize zeff() in such a way that 
           !current from FPE solution would match the target current, 
           !which is set in totcrt(1:nbctime) in cqlinput.
           !But first, set initial zeff(), similar to iprozeff.eq.
           !"prbola-t", unless has been restored for a restart case.
           if(n.eq.0.and.nlrestrt.eq."disabled")then ! Set initial zeff(ll).
            if (zeffc(1).ne.zero) then
               if (tmdmeth.eq."method1") then
                  if (itme.eq.0) then
                     zeffin(0)=zeffc(1)
                     zeffin(1)=zeffb(1)
                  elseif (itme.lt.nbctime) then
                     zeffin(0)=zeffc(itme)+(zeffc(itme1)-zeffc(itme))
     +                    /(bctime(itme1)-bctime(itme))
     +                    *(timet-bctime(itme))
                     zeffin(1)=zeffb(itme)+(zeffb(itme1)-zeffb(itme))
     +                    /(bctime(itme1)-bctime(itme))
     +                    *(timet-bctime(itme))
                  else
                     zeffin(0)=zeffc(nbctime)
                     zeffin(1)=zeffb(nbctime)
                  endif
               elseif (tmdmeth.eq."method2") then
                  zeffin(0)=tdprof(timet,zeffc(1),bctime)
                  zeffin(1)=tdprof(timet,zeffb(1),bctime)
               endif  !On tmdmeth
            endif  !On zeffc(1)
            !YuP dratio=zeffin(1)/zeffin(0)
            e0=zeffin(0) !YuP[2019-12-29]
            e1=zeffin(1) !YuP[2019-12-29]
            do ll=1,lrzmax
               !YuP call profaxis(rn,npwrzeff,mpwrzeff,dratio,rya(ll))
               !YuP zeff(ll)=zeffin(0)*rn
               call profaxis1(e_out,npwrzeff,mpwrzeff,e0,e1,rya(ll)) !YuP[2019-12-29]
               zeff(ll)=e_out !YuP[2019-12-29]      
            enddo
           endif !(n.eq.0.and.nlrestrt.eq."disabled") !init zeff(ll) is set.

           !If nbctime.gt.0, and this is a restart case, then for
           !iprozeff.eq."curr_fit, need to restore the zeff(ll)
           !from end of the previous run (from distrfunc.nc).
           !if(n.eq.0.and.nlrestrt.eq."ncdfdist")then
           !   call tdreadf(3)  !kopt=3 restores zeff profile
           !endif  !On n.eq.0.and.nlrestrt.eq."ncdfdist"
           !YuP[2020-02-11] Moved this part to tdinitl
           
           ! For any n step:
           if(totcrt(1).ne.zero)then !determine the target current at this n
            if (itme.eq.0) then
               totcurtt=totcrt(1)
            elseif (itme.lt.nbctime) then
               totcurtt=totcrt(itme)+(totcrt(itme1)-totcrt(itme))
     1              /(bctime(itme1)-bctime(itme))
     1              *(timet-bctime(itme))
            else
               totcurtt=totcrt(nbctime)
            endif
            ! Find total current from recent FPE solution,
            ! add delta_sigma*E, add BS, to be compared to 
            ! total target current, which includes 
            ! all those types of current intrinsically
            currxjtot=0.d0  !for total current from FPE [curr() array]
            bscurr_tot=0.d0 !for total BS current
            curra_tot=0.d0 !RE current, which may include RF_CD 
            do ll=1,lrzmax
              currxj(ll)=currtp(ll)/3.e9 !A/cm2, <jpar>_FSA, calc in diaggnde
              !YuP[2019-12-27] Also add 
              !  currpar_starnue_n(ll)-currpar_starnue0_n(ll)
              !It is "missed" in solution of FPE. See tddiag.f
              curr_add= currpar_starnue_n(ll)-currpar_starnue0_n(ll) !A/cm^2
              !Also consider the bootstrap current:
              bscurr_tot= bscurr_tot+darea(ll)*bscurm_n(ll) !bootstrap [A]
              !Do not include bootstrap current here - 
              ! it is not linearly dependent on Zeff;
              ![see functions like sa31, f32ee, f32ei - 
              ! they have a complicated non-linear dependence on zz].
              !Instead, subtract it from the target current totcurtt, below.
              !The adjustment of Zeff (see below) is based on assumption that
              !the main current depends on Zeff through Ip ~ Te^1.5/Zeff;
              !so basically it is adjusted as 
              ! Zeff_new/Zeff_old= currxjtot/Target_current
              !If the calculated current "currxjtot" is larger than
              !the target current, it is "suppressed" at next time step 
              !by a proportionally larger Zeff_new.
              currxjtot=currxjtot+darea(ll)*(currxj(ll)+curr_add) ! [Amps]
             !Note: at n=0, at very 1st call to profiles.f, 
             !darea is not defined yet [it is set in tdrmshst],
             ! and so currxjtot remains 0.
             !At n=0 sub.profiles is called from sub.tdinitl
             !BEFORE tdrmshst. 
             !But then sub.profiles is called again at n=0, 
             ! from tdchief[Line~575], just before call_achiefn(0).
             !At the latter call, darea is defined.
!--CMPIINSERT_IF_RANK_EQ_0
!--             WRITE(*,*)'profiles: n,ll,currxj(ll),currxjtot',
!--     &        n,ll,currxj(ll),currxjtot
!--CMPIINSERT_ENDIF_RANK
            enddo ! ll
!--CMPIINSERT_IF_RANK_EQ_0
!--            WRITE(*,*)'profiles: n, Target current totcurtt=',n,totcurtt
!--CMPIINSERT_ENDIF_RANK
            ! At this point, currxjtot includes current from FPE and delta_sigma*E.
            ! Add BS current, for consistensy with target current totcurtt :
            currxjtot=currxjtot+bscurr_tot
            ! So now we can compare currxjtot with totcurtt
                        
            if( (n.gt.0) .and. 
     &          (totcurtt .ne. zero) .and. 
     &          (currxjtot.ne. zero)        )then 
              !Target current (reduced by bs current)= totcurtt-bscurr_tot
              !YuP[2019-12-27] 
              !YuP[2020-02-11] Added n>0 condition. At n=0, currtp() is unknown.
              !
              !--[early attempt] for renormalization of Zeff:
![early attempt]  renormz=(currxjtot-bscurr_tot)/(totcurtt-bscurr_tot)
              !-- works ok when curra() component is <80% of total curr()
              !-- ( curra_tot < 0.8*currxjtot, approximately).
              !-- If curra_tot is too large, the procedure becomes unstable:
              !-- zeff "jumps" between small and large values, 
              !-- from one time step to another.
              !-----BH,YuP[2020-11-01] Try something similar to method4 for E-field.
              !-1- Find the "deviation" of currxjtot from target current totcurtt:
              djrel= (totcurtt-currxjtot) / 
     &               max(abs(totcurtt),abs(currxjtot))
              sign_dj=  sign(one,djrel*currxjtot)
              if(djrel.eq.0.d0) sign_dj=0.d0
              !-2- Now setup the relaxation procedure.
              !zrelax=    0.5d0 !(default value); in namelist
              !zrelax_exp=1.d0  !(default value); in namelist
              renormz= (1.d0-zrelax*sign_dj*abs(djrel)**zrelax_exp)
              !Note: If djrel=0, then renormz=1.
              !-2a- Make sure that Zeff_new would not over-shoot [2020-11-03]
              fmaxx=0.d0
              do k=1,nionm
                 fmaxx=max(fmaxx,bnumb(kionm(k))) !presently fmaxx=100 [2020-11]
              enddo
              if(zeff(1)*renormz.ge.(fmaxx-1.d-6))then
                 renormz=(fmaxx-1.d-6)/zeff(1) !Assuming flat profile of zeff
                 ! Then, when adjustment is made below, we will get
                 !zeff(ll)[new value] = zeff(ll)*renormz= (fmaxx-1.d-6)
              endif
              !-2b- Make sure that Zeff_new would not under-shoot [2020-11-03]
              if(zeff(1)*renormz.le.(1.d0+1.d-3))then
                 renormz=(1.d0+1.d-3)/zeff(1) !Assuming flat profile of zeff
                 ! Then, when adjustment is made below, we will get
                 !zeff(ll)[new value] = zeff(ll)*renormz= (1.d0+1.d-3)
              endif
              renorm=1.d0/renormz
              !-3- Renormalize Zeff :
              do ll=1,lrzmax
                zeff(ll)=zeff(ll)*renormz ! Note that Ip ~ Te^1.5/Zeff
                zeff(ll)=max(zeff(ll),1.d0) ! Cannot be lower than 1.0
                currxj(ll)=currxj(ll)*renorm
                currtp(ll)=currtp(ll)*renorm
                currpar_starnue_n(ll)=currpar_starnue_n(ll)*renorm !YuP[2019-12-27] 
                currpar_starnue0_n(ll)=currpar_starnue0_n(ll)*renorm
              !Note: we also renormalize currxj() so that if subr.profiles
              !is called again at the same time step,
              !the value of currxjtot would be totcurtt, and then 
              !renorm=1, so that zeff() would NOT be rescaled again.
              enddo
!CMPIINSERT_IF_RANK_EQ_0
!            do ll=1,lrzmax
!            write(*,'(a,3i4,3e11.3)')
!     &          'profiles: n,itme,ll,currxjtot,zeff,currxj',
!     &           n,itme,ll,currxjtot,zeff(ll),currxj(ll)
!            enddo
!CMPIINSERT_ENDIF_RANK  
            endif ! currxjtot.ne.zero
           endif ! totcrt(1).ne.zero
           ! The question now is - How to deal with densities?
           ! Presently, it is setup to keep n_e unchanged,
           ! and adjust the ion densities (example: reduce D+ density,
           ! but increase density of impurity, to keep n_e unchanged).
           ! Another approach would be to increase density of impurity,
           ! and to increase density of electrons accordingly,
           ! while the density of D+ remains unchanged.
         endif ! iprozeff.eq."curr_fit"  ![2019-10-31]new
         

         if (nbctime.gt.0 .and. iprozeff.eq."prbola-t") then
            if (zeffc(1).ne.zero) then
               if (tmdmeth.eq."method1") then
                  if (itme.eq.0) then
                     zeffin(0)=zeffc(1)
                     zeffin(1)=zeffb(1)
                  elseif (itme.lt.nbctime) then
                     zeffin(0)=zeffc(itme)+(zeffc(itme1)-zeffc(itme))
     +                    /(bctime(itme1)-bctime(itme))
     +                    *(timet-bctime(itme))
                     zeffin(1)=zeffb(itme)+(zeffb(itme1)-zeffb(itme))
     +                    /(bctime(itme1)-bctime(itme))
     +                    *(timet-bctime(itme))
                  else
                     zeffin(0)=zeffc(nbctime)
                     zeffin(1)=zeffb(nbctime)
                  endif
               elseif (tmdmeth.eq."method2") then
                  zeffin(0)=tdprof(timet,zeffc(1),bctime)
                  zeffin(1)=tdprof(timet,zeffb(1),bctime)
               endif  !On tmdmeth
            endif  !On zeffc(1)
            !YuP dratio=zeffin(1)/zeffin(0)
            e0=zeffin(0) !YuP[2019-12-29]
            e1=zeffin(1) !YuP[2019-12-29]
            do ll=1,lrzmax
               !YuP call profaxis(rn,npwrzeff,mpwrzeff,dratio,rya(ll))
               !YuP zeff(ll)=zeffin(0)*rn
               call profaxis1(e_out,npwrzeff,mpwrzeff,e0,e1,rya(ll)) !YuP[2019-12-29]
               zeff(ll)=e_out !YuP[2019-12-29]      
            enddo
         endif ! iprozeff.eq."prbola-t"
         
         if (nbctime.gt.0 .and. iprozeff.eq."spline-t") then
               if (itme.eq.0) then
                  do l=1,njene
                     tmpt(l)=zeffin_t(l,1)
                  enddo
               elseif (itme.lt.nbctime) then
                  do l=1,njene
                     tmpt(l)=zeffin_t(l,itme)
     +                    +(zeffin_t(l,itme1)-zeffin_t(l,itme))
     +                    /(bctime(itme1)-bctime(itme))
     +                    *(timet-bctime(itme))
                  enddo
               else
                  do l=1,njene
                     tmpt(l)=zeffin_t(l,nbctime)
                  enddo
               endif
               call tdinterp("zero","linear",ryain,tmpt,njene,rya(1),
     +              tr(1),lrzmax)
               do  ll=1,lrzmax
                  zeff(ll)=tr(ll)
               enddo
         endif ! iprozeff.eq."spline-t"

c     Scale zeff
         do ll=1,lrzmax
            zeff(ll)=zeffscal*zeff(ll)
         enddo
         
c     Check that range of bnumb for Maxl species brackets zeff
         fmaxx=0.
         fminn=0.
         do k=1,nionm
            fmaxx=max(fmaxx,bnumb(kionm(k)))
            fminn=min(fminn,bnumb(kionm(k)))
         enddo
         do 121 ll=1,lrzmax
            if(zeff(ll).gt.fmaxx .or. zeff(ll).lt.fminn) then
CMPIINSERT_IF_RANK_EQ_0
               WRITE(*,*)'profiles.f: ',
     +        'Adjust bnumb(kion) for compatibility with zeff=',zeff(ll)
CMPIINSERT_ENDIF_RANK  
               !stop !YuP[2020-10-30]
               zeff(ll)=min(zeff(ll),fmaxx) !YuP[2020-10-30] reset to max value
            endif
 121     continue


c     Check number of ion Maxwl species with different bnumb.
c     (Need at least two in order to fit zeff. Check ainsetva.f.)
        if (nionm.lt.2) stop 'profiles: ion species problem'
        ndif_bnumb=1
        do k=2,nionm
           if (abs(bnumb(kionm(k))/bnumb(kionm(1))-1.).gt.0.01) 
     +          ndif_bnumb=ndif_bnumb+1
        enddo

cBH120627:  This appears to be already done above:
cBH120627:c    Interpolate input ion densities onto rya grid
cBH120627:     call tdxin13d(reden,rya,lrzmax,ntotala,k,npwr(0),mpwr(0))


         
c     Save ratios of densities for Maxwl ions with same charge in reden
c     to be used in following ion density calc from zeff.
c     (First ndif_bnumb ions have same change.  These are ion species
c     at the head of the ion species list.)

      nsame_bnumb=nionm-ndif_bnumb+1  !i.e., =1 if all diff bnumb ions
      if (nsame_bnumb.gt.1) then  !i.e., 2 or more ions have same bnumb.
                                  !Densities will contain density ratios.
                                  !Equal-bnumb() species at beginning of
                                  !Maxl ions.
c        Renormalizing the density ratios as fractions for equal-bnumb:
            do ll=1,lrzmax
               dsum=0.
               do k=1,nsame_bnumb
                  dsum=dsum+reden(kionm(k),ll)
               enddo
               do k=1,nsame_bnumb
                  reden(kionm(k),ll)=reden(kionm(k),ll)/dsum
               enddo
               reden(kionm(nionm),ll)=one
            enddo
      endif  ! on nsame_bnumb.gt.1

c     Set rest of ion densities to 1.
      do kk=nsame_bnumb,nionm
         k=kionm(kk)
         do l=0,lrzmax
            reden(k,l)=one
         enddo
      enddo
         
c     NOTE: reden(k, ) on rhs of following reden(,) formulas is
c     density ratio, .le.1. for diff species with same bnumb.
      do 14 k=kionm(1),kionm(nionm)
c     write(*,*)'tdxinitl, do 14 loop, k= ',k
c        For k pointing to equal-bnumb species (1 or more)
         if (k.le.(kionm(1)+nsame_bnumb-1)) then
            k1=k
            k2=kionm(nionm)
c        For k beyond equal-bnumb species
         else
            k1=kionm(nionm)
            k2=k-1
         endif
         reden(k,0)=reden(kelec,0)*reden(k,0)*(zeff(1)
     +         -bnumb(k2))/(bnumb(k1)-bnumb(k2))/bnumb(k1)
         !YuP: From the above, reden can be a small negative value,
         !because of a rounding error. Add lower limit =0.d0
         reden(k,0)=max(reden(k,0),zero) !YuP[2018-09-18] added
         do 142 ll=1,lrzmax
               reden(k,ll)=reden(kelec,ll)*reden(k,ll)*(zeff(ll)
     +              -bnumb(k2))/(bnumb(k1)-bnumb(k2))/bnumb(k1)
            !YuP: From the above, reden can be a small negative value,
            !because of a rounding error. Add lower limit =0.d0
            reden(k,ll)=max(reden(k,ll),zero) !YuP[2018-09-18] added
 142     continue
         !write(*,*)'profiles:k1,k2=',k1,k2
         !write(*,*)'zeff=',zeff
         !write(*,*)'profiles:k,max(reden)=',k,MAXVAL(reden(k,:))
 14   continue
      


c     Copy Maxwellian ion densities to any ion general species
c     in order of the indexes.

         do k=1,niong
            if (nionm.ge.niong) then
               do l=0,lrzmax
                  reden(kiong(k),l)=reden(kionm(k),l)
               enddo
            endif
         enddo
         
         
      endif  ! on iprozeff.ne."disabled", begin at l 470
      

c     Renormalize densities using enescal 
![2022-03-10] moved this rescaling to tdxinitl, 
! to be applied to redenc(nbctimea,ntotala),redenb(nbctimea,ntotala)
! enein_t(njenea,ntotala,nbctimea)
!      do k=1,ntotal
!         do l=0,lrzmax
!            reden(k,l)=enescal*reden(k,l) !ions are needed for next section
!            !For case of imp_depos_method.ne.'disabled',
!            ! the impurities (dens_imp(kstate,l))
!            ! are in cm-3 already, so the reden(kion,*) 
!            ! should also be in cm-3
!         enddo
!      enddo
![2022-03-10]
      
      if((imp_depos_method.ne.'disabled').and.(kelec.ne.0))then
        !YuP[2020-06-24] Changed (gamafac.eq."hesslow") to (imp_depos_method.ne.'disabled')
        !   [a more general logic]
        !YuP[2019-09-18] Density of electrons and Zeff
        ! will be calculated consistently:
        ! ne will be found from reden() of ions(kion) and from 
        ! dens_imp(kstate,lr) of additional ions (from pellet, etc.)
        do l=1,lrz
          sum_ni_Zi=0.d0
          do k=1,ntotal ! Sum up density of ions 'k', with proper Zk
            if(k.eq.kelecg.or.k.eq.kelecm)then
              ! skip electrons
            else ! main ions - sum them up
              sum_ni_Zi= sum_ni_Zi +reden(k,l)*bnumb(k) !from maxwellian ions
            endif
          enddo ! k
          sum_nimp_zstate=0.d0 !To count electrons from impurity, all Zstates
          do kstate=1,nstates !These are additional ions from impur.source.
             sum_nimp_zstate= sum_nimp_zstate
     &                       +dens_imp(kstate,l)*bnumb_imp(kstate)
          enddo ! kstate
          sum_ni_Zi= sum_ni_Zi +sum_nimp_zstate !total number of electrons.
          !YuP[2019-12-06] There are two ways to adjust electron density:
          ! 1. Assume that density of main ion species is not changed; 
          !    then, electron density is set to sum_ni_Zi.
          ! 2. Assume that electron density is not changed
          !   (or taken from the input list); 
          !    then, reduce the density of main ions, 1st ionic species:
          kion1=kionm(1)
          if(imp_ne_method.eq.'ne_list')then ! Option 2 in the above.
          if(kion1.ne.0)then
           !Remove contribution from kion1 species to total elec.count;
           !It will be adjusted and then added back.
           sum_ni_Zi= sum_ni_Zi - reden(kion1,l)*bnumb(kion1)
           !Recalculate density of kion1 species: 
           reden(kion1,l)=(reden(kelecm,l)-sum_nimp_zstate)/bnumb(kion1)
           !It may happen that reden(kion1,l) from the above expression
           ! is negative (too many electrons from impurity ions).
           ! Then, set it to a small value and recalculate:
           reden(kion1,l)= max(reden(kion1,l),1.d5)
           !Add contribution from kion1 species, using the updated density:
           sum_ni_Zi= sum_ni_Zi + reden(kion1,l)*bnumb(kion1)
          endif
          endif ! imp_ne_method.eq.'ne_list'
          ! All ions (including those produced by impurity source like pellet
          ! are summed up. Now we calculate the density of electrons:
          reden(kelecg,l)= sum_ni_Zi
          reden(kelecm,l)= sum_ni_Zi
          ! In diagscal, the distribution function will be adjusted
          ! to match the new value of reden(kelecg,l)
        enddo ! l
      endif !(imp_depos_method.ne.'disabled')
      
      



c--------------------------------------------------------------------
      if (nbctime.gt.0 .and. ipronn.eq."spline-t") then ! time-dep. neutrals or impurities
                  
         ! For NPA:
         do kkk=1,npaproc
            if (npa_process(kkk).ne.'notset'.and.kkk.ne.5) then
               if (itme.eq.0) then
                  do l=1,njene
                  tmpt(l)= ennin_t(l,1,kkk)
                  enddo
               elseif (itme.lt.nbctime) then
                  do l=1,njene
                  tmpt(l)= ennin_t(l,itme,kkk)+
     +            (ennin_t(l,itme1,kkk)-ennin_t(l,itme,kkk))
     +            /(bctime(itme1)-bctime(itme))*(timet-bctime(itme))
                  enddo
               else ! itme >= nbctime
                  do l=1,njene
                  tmpt(l)= ennin_t(l,nbctime,kkk)
                  enddo
               endif
               !------------
               call tdinterp("zero","linear",ryain,tmpt,njene,
     +              rya(1),tr(1),lrzmax)
               do ll=1,lrzmax
               enn(ll,kkk)= ennscal(kkk)*tr(ll) ! for tdnpa
               enddo
            endif ! npa_process(kkk).ne.'notset'.and.kkk.ne.5
         enddo ! kkk=1,npaproc
         ! For NPA, process#5 (recombination with electrons):
         if (npa_process(5).eq.'radrecom') then
             k=max(kelecg,kelecm)  !I.e., using bkgrnd distn, if avail.
             do ll=1,lrzmax
                enn(ll,5)= ennscal(5)*reden(k,ll) 
                ! Note: reden is defined above (can be time-dependent)
             enddo
         endif
         
      endif ! ipronn.eq.'spline-t'
c--------------------------------------------------------------------

      

c     Electric field or target current
c     Skip, if ampfmod.eq.enabled .and. n+1.ge.nonampf
      
      if (ampfmod.eq.'enabled' .and. n+1.ge.nonampf) then
cBH191002      if (ampfmod.eq.'enabled' .and. n.ge.nonampf) then
        continue ! Meaning: skip elecfld(0)=elecc(1),etc, setting.
         ! Here, profiles is called from tdchief when
         ! n is not updated yet. Here n=0,1,2,...,nstop-1.
         ! For example, if nstop=5 and nonampf=5 
         ! (meaning: apply ampf calc. at the last step),
         ! we need to start skipping elecfld(0)=elecc(1),etc
         ! when n+1=nstop=nonampf
      else
      
      if (efswtch.eq."method1") then
      
         if (nbctime.gt.0 .and. iproelec.eq."prbola-t") then
            
            if (tmdmeth.eq."method1") then
               !YuP[2019-12-29] Removed ".and.elecc(1).ne.zero" (now can be 0.0)
               
               if (itme.eq.0) then
                  elecfld(0)=elecc(1)
                  elecfld(1)=elecb(1)
                  elecfldc=elecfld(0)
                  elecfldb=elecfld(1)
               elseif (itme.lt.nbctime) then
                  elecfld(0)=elecc(itme)+(elecc(itme1)-elecc(itme))
     1                 /(bctime(itme1)-bctime(itme))
     1                 *(timet-bctime(itme))
                  elecfld(1)=elecb(itme)+(elecb(itme1)-elecb(itme))
     1                 /(bctime(itme1)-bctime(itme))
     1                 *(timet-bctime(itme))
                  elecfldc=elecfld(0)
                  elecfldb=elecfld(1)
               else
                  elecfld(0)=elecc(nbctime)
                  elecfld(1)=elecb(nbctime)
                  elecfldc=elecfld(0)
                  elecfldb=elecfld(1)
               endif

            elseif (tmdmeth.eq."method2") then
               
               elecfld(0)=tdprof(timet,elecc(1),bctime)
               elecfld(1)=tdprof(timet,elecb(1),bctime)
               elecfldc=elecfld(0)
               elecfldb=elecfld(1)
               
            endif

            !YuP dratio=elecfld(1)/elecfld(0)
            e0=elecfld(0) !YuP[2019-12-29]
            e1=elecfld(1) !YuP[2019-12-29]
            do ll=1,lrzmax
               !YuP call profaxis(rn,npwrelec,mpwrelec,dratio,rya(ll))
               !YuP elecfld(ll)=elecfld(0)*rn
               call profaxis1(e_out,npwrelec,mpwrelec,e0,e1,rya(ll)) !YuP[2019-12-29]
               elecfld(ll)=e_out !YuP[2019-12-29]  
            enddo
         endif  !On iproelec.eq.prbola-t

         if (nbctime.gt.0 .and. iproelec.eq."spline-t") then
            if (tmdmeth.eq."method1") then

              if (itme.eq.0) then
                  do l=1,njene
                     tmpt(l)=elecin_t(l,1)
                  enddo
              elseif (itme.lt.nbctime) then
                  do l=1,njene
                     tmpt(l)=elecin_t(l,itme)
     +                    +(elecin_t(l,itme1)-elecin_t(l,itme))
     +                    /(bctime(itme1)-bctime(itme))
     +                    *(timet-bctime(itme))
                  enddo
       write(*,*)'profiles: n,nch,timet,E',n,nch(1),timet,tmpt(1)
               else
                  do l=1,njene
                     tmpt(l)=elecin_t(l,nbctime) ! V/cm
                  enddo
               endif
               call tdinterp("zero","linear",ryain,tmpt,njene,rya(1),
     +              tr(1),lrzmax)
               do  ll=1,lrzmax
                  elecfld(ll)=tr(ll)
               enddo
c              Assume ryain(1)=0 (or close) to get central elecfld
c              needed, e.g., for ampfsoln
               elecfld(0)=tmpt(1)
            endif  !On tmdmeth
            elecfldc=tmpt(1)
            elecfldb=tmpt(njene)
            if (read_data.eq.'nimrod'.and.(n.gt.0)) then !YuP[2021-02-12]
               ! In case of read_data='nimrod', 
               ! E_estimate field is formed from
               ! J_data and estimate of Spitzer resistivity 
               ! with neoclass. correction, "eta" here: 
               ! E_estimate == elecin_t()= eta*J_data
               ! However, if RE current appears, 
               ! we need to adjust this estimate, 
               ! assuming J_data - J_RE = E/eta
               ! or E = eta*J_data - eta*J_RE, so 
               ! E= elecin_t - eta*J_RE
               ! In mixed V/cm and cgs units:
               ! E[V/cm] = elecfld[V/cm] - eta[cgs]*curra[cgs]*300
               ! where 300 is to convert statVolt/cm to V/cm
               fr1=1.0 !0.95 !Test: fraction of J_RE to be subtracted
               do ll=1,lrzmax
               !To be subtracted:
               ! Either fr1*(curra(1,ll)/sig_starnue0(ll))
               ! or 90-95% of E (to leave 10-5%, otherwise E can become 0),
               !whichever is smaller by magnitude. 
!               dee= min( fr1*(curra(1,ll)/sig_starnue0(ll))*300.,
!     &                   elecfld(ll)*0.95 )
               !----------------------------------------------------------
               !See subr. read_data_files() --
               ![2022-01-06]Better use collisional limit (as in NIMROD):
               !e_sptzr= curr_data*(eta_sptzr*pressau2) ! V/m, A/m^2
               !In this case - in profiles.f, 
               ! change curra(1,ll)/sig_starnue0(ll)
               !     to curra(1,ll)/sig_starnue(ll)
               bscur= bscurm(ll,1,2)+bscurm(ll,2,1) !e_gen+i_maxw [A/cm^2]
               bscur= bscur*3.e9 ! to statA/cm^2
               dee= min( fr1*300.*(curra(1,ll)+bscur)/sig_starnue(ll),
     &                   elecfld(ll)*0.95 )
               !----------------------------------------------------------
                  if(dee.gt.0.d0)then !Only apply to reduce
                  elecfld(ll)= elecfld(ll)-dee
                  endif
                  !Note: sig_starnue0(ll) and sig_starnue(ll) 
                  ! [conductivity in cgs units, 
                  !  zero-collis. and collisional limits]
                  !are evaluated in tddiag at each time step.
               !ATTENTION: If we change sig_starnue0 to sig_starnue here
               !(i.e. to finite collisionality), we also need to change
               ! e_sptzr= curr_data*(eta_sptzr*pressau1)  to 
               ! e_sptzr= curr_data*(eta_sptzr*pressau2)
               ! in subr.read_data_files()
               enddo ! ll
               elecfld(0)= elecfld(1)
               elecfldc= elecfld(1)
               elecfldb= elecfld(lrzmax)
            endif ! read_data.eq.'nimrod' ![2021-02-12]

         endif  !On iproelec.eq.spline-t

c100126  Scale elecfld
         do ll=1,lrzmax
            elecfld(ll)=elecscal*elecfld(ll)
         enddo
         elecfldc=elecscal*elecfldc
         elecfldb=elecscal*elecfldb
         
      elseif (efswtch.eq."method2" .or. efswtch.eq."method3" .or.
     +        efswtch.eq."method4") then
         
         if (nbctime.gt.0 .and. iprocur.eq."prbola-t") then
            if (tmdmeth.eq."method1") then
               if (itme.eq.0) then
                  currxj(0)=xjc(1) !for iprocur.eq."prbola-t"
                  currxj(1)=xjb(1) !for iprocur.eq."prbola-t"
               elseif (itme.lt.nbctime) then
                  currxj(0)=xjc(itme)+(xjc(itme1)-xjc(itme))
     1                 /(bctime(itme1)-bctime(itme))
     1                 *(timet-bctime(itme))
                  currxj(1)=xjb(itme)+(xjb(itme1)-xjb(itme))
     1                 /(bctime(itme1)-bctime(itme))
     1                 *(timet-bctime(itme))
               else
                  currxj(0)=xjc(nbctime)
                  currxj(1)=xjb(nbctime)
               endif
            elseif (tmdmeth.eq."method2") then
               currxj(0)=tdprof(timet,xjc(1),bctime) !for iprocur.eq."prbola-t"
               currxj(1)=tdprof(timet,xjb(1),bctime) !for iprocur.eq."prbola-t"
            endif
            !YuP dratio=currxj(1)/currxj(0)
            e0=currxj(0) !YuP[2019-12-29]
            e1=currxj(1) !YuP[2019-12-29]
            do ll=1,lrzmax
               !YuP call profaxis(rn,npwrxj,mpwrxj,dratio,rya(ll))
               !YuP currxj(ll)=currxj(0)*rn
               call profaxis1(e_out,npwrxj,mpwrxj,e0,e1,rya(ll)) !YuP[2019-12-29]
               currxj(ll)=e_out !YuP[2019-12-29]      
            enddo
         endif ! iprocur.eq."prbola-t"
         
         if (nbctime.gt.0 .and. iprocur.eq."spline-t") then
            if (tmdmeth.eq."method1") then
              if (itme.eq.0) then
                  do l=1,njene
                     tmpt(l)=xjin_t(l,1) ! A/cm^2
                  enddo
              elseif (itme.lt.nbctime) then
                  do l=1,njene
                     tmpt(l)=xjin_t(l,itme)
     +                    +(xjin_t(l,itme1)-xjin_t(l,itme))
     +                    /(bctime(itme1)-bctime(itme))
     +                    *(timet-bctime(itme))
                  enddo
               else
                  do l=1,njene
                     tmpt(l)=xjin_t(l,nbctime)
                  enddo
               endif
               call tdinterp("zero","linear",ryain,tmpt,njene,rya(1),
     +              tr(1),lrzmax)
               do  ll=1,lrzmax
                  currxj(ll)=tr(ll) !for iprocur.eq."spline-t"
               enddo
            endif  !On tmdmeth.eq."method1"
         endif  !On iprocur.eq."spline-t"

      elseif (efswtch.eq."method5") then

c          Parallel current from eqdsk:
c          Decided to set up currxj from eqdsk in
c          subroutine tdrmshst.  Also, have to
c          make this setup AFTER processing the eqdsk.

      endif  !On efswtch
      endif  !On ampfmod.eq.enabled .and. n.ge.nonampf

      if (ampfmod.eq.'enabled' .and. n.le.nonampf) then !Fill in for plots
         do it=0,nampfmax
            do ll=0,lrz
               elecfldn(ll,n,it)=elecfld(ll)/300.d0
            enddo
            elecfldn(lrz+1,n,it)=elecfldb/300.d0
         enddo
      endif
      
      if (totcrt(1).ne.zero) then
         
         if (tmdmeth.eq."method1") then
            
            if (itme.eq.0) then
               totcurtt=totcrt(1)
            elseif (itme.lt.nbctime) then
               totcurtt=totcrt(itme)+(totcrt(itme1)-totcrt(itme))
     1              /(bctime(itme1)-bctime(itme))
     1              *(timet-bctime(itme))
            else
               totcurtt=totcrt(nbctime)
            endif
            
         elseif (tmdmeth.eq."method2") then
            
            totcurrtt=tdprof(timet,totcrt,bctime)
            
         endif
         
c     Renormalize currxj
         currxjtot=0.d0
         do 80 ll=1,lrzmax
            currxjtot=currxjtot+darea(ll)*currxj(ll)
            !YuP[2019-10-29] The problem here is that at n=0
            !darea is not defined yet [it is set in tdrmshst].
            !At n=0 sub.profiles is called from sub.tdinitl
            !BEFORE tdrmshst. 
            !But then sub.profiles is called again at n=0, 
            ! from tdchief[Line~575], just before call_achiefn(0).
            !At the latter call, darea is defined.
 80      continue
 
         if(currxjtot.ne.0.d0)then !YuP[2019-10-29] Added currxjtot condition
           !At n=0, at first call of profiles, when darea=0 and so currxjtot=0.d0, 
           ! do this renormalization in tdrmshst.f(239)
           do 81 ll=1,lrzmax
            currxj(ll)=(totcurtt/currxjtot)*currxj(ll)
 81        continue
         endif !currxjtot.ne.0.d0
         
!           do ll=1,lrzmax
!            write(*,'(a,3i4,3e12.3)')
!     &          'profiles: n,itme,ll,elecfld,totcurtt,currxj',
!     &           n,itme,ll,elecfld(ll),totcurtt,currxj(ll)
!           enddo
!           pause
         
      endif ! totcrt(1).ne.zero
         
c     Toroidal rotation velocity
         
      if (nbctime.gt.0 .and. iprovphi.eq."prbola-t") then
         if (tmdmeth.eq."method1") then
            if (itme.eq.0) then
               vphiplin(0)=vphic(1)
               vphiplin(1)=vphib(1)
            elseif (itme.lt.nbctime) then
               vphiplin(0)=vphic(itme)+(vphic(itme1)-vphic(itme))
     1              /(bctime(itme1)-bctime(itme))
     1              *(timet-bctime(itme))
               vphiplin(1)=vphib(itme)+(vphib(itme1)-vphib(itme))
     1              /(bctime(itme1)-bctime(itme))
     1              *(timet-bctime(itme))
            else
               vphiplin(0)=vphic(nbctime)
               vphiplin(1)=vphib(nbctime)
            endif
         elseif (tmdmeth.eq."method2") then
            vphiplin(0)=tdprof(timet,vphic(1),bctime)
            vphiplin(1)=tdprof(timet,vphib(1),bctime)
         endif
         !YuP dratio=vphiplin(1)/vphiplin(0)
         e0=vphiplin(0) !YuP[2019-12-29]
         e1=vphiplin(1) !YuP[2019-12-29]
         do ll=1,lrzmax
            !YuP call profaxis(rn,npwrvphi,mpwrvphi,dratio,rya(ll))
            !YuP vphipl(ll)=vphiscal*vphiplin(0)*rn
            call profaxis1(e_out,npwrvphi,mpwrvphi,e0,e1,rya(ll)) !YuP[2019-12-29]
            vphipl(ll)=e_out !YuP[2019-12-29]      
         enddo
      endif ! iprovphi.eq."prbola-t"
      
      if (nbctime.gt.0 .and. iprovphi.eq."spline-t") then
         if (tmdmeth.eq."method1") then
            if (itme.eq.0) then
               do l=1,njene
                  tmpt(l)=vphiplin_t(l,1)
               enddo
            elseif (itme.lt.nbctime) then
               do l=1,njene
                  tmpt(l)=vphiplin_t(l,itme)
     +                 +(vphiplin_t(l,itme1)-vphiplin_t(l,itme))
     +                 /(bctime(itme1)-bctime(itme))
     +                 *(timet-bctime(itme))
               enddo
            else
               do l=1,njene
                  tmpt(l)=vphiplin_t(l,nbctime)
               enddo
            endif
            call tdinterp("zero","linear",ryain,tmpt,njene,rya(1),
     +           tr(1),lrzmax)
            do  ll=1,lrzmax
               vphipl(ll)=vphiscal*tr(ll)
            enddo
         endif                  !On tmdmeth
      endif  ! iprovphi.eq."spline-t"


CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)
      WRITE(*,*)'profiles: time step n, timet=',n,timet
CMPIINSERT_ENDIF_RANK  

c     From tdoutput.f:
c.......................................................................
cl    0. initialize "radial" mesh for print out
c.......................................................................
      if (pltvs .ne. "psi") then
        ztext=" rovera "
        do l=1,lrzmax
          ztr(l)=rovera(l)
        enddo
      else
        ztext=pltvs
        do l=1,lrzmax
          ztr(l)=(equilpsi(0)-equilpsi(l))/equilpsi(0)
        enddo
      endif

c.......................................................................
cl    1.2 density,temperature,magnetic field,etc.
c.......................................................................

cdir$ nextscalar
c
CMPIINSERT_IF_RANK_EQ_0
      do 123 jk=1,ntotal
         if(n.gt.0)then
           ! At n=0, profiles() is called before aingeom(),
           ! and so bthr(ll) is not defined yet.
           ! Then, rban_vth=NaN
           do ll=1,lrzmax
             qb_mc=bnumb(jk)*charge*bthr(ll)/(fmass(jk)*clight)
             ! Banana width (for v=vthermal, at t-p bndry):
             ! BH171230: For lrzdiff=enabled, lrz<lrzmax, the default
             ! values of itl_(ll>lrz) can cause out-of-bounds coss.
             if (ll.le.lrz) then
             rban_vth(ll)= abs(vth(jk,ll)*coss(itl_(ll),ll)/qb_mc) ! cm
             else
                rban_vth(ll)= zero
             endif
           enddo
         else ! n=0
           rban_vth=0.d0 ! not defined yet
           ! Not a big problem (just for a print-out, 
           ! and also printed by tdoutput, 
         endif
         if (ipronn.eq."disabled".or.jk.ne.nnspec) then
            if(kspeci(2,jk).eq.'general' .and. 
     +                  iprovphi.ne.'disabled') then
            WRITE(6,9126) jk,kspeci(1,jk),kspeci(2,jk),bnumb(jk),ztext
            WRITE(6,9127) (il,ztr(il),reden(jk,il),temp(jk,il),
     +           vth(jk,il),energy(jk,il),vphipl(il),il=1,lrzmax)
            else
            WRITE(6,9120) jk,kspeci(1,jk),kspeci(2,jk),bnumb(jk),ztext
            WRITE(6,9121) (il,ztr(il),reden(jk,il),temp(jk,il),
     +           vth(jk,il),energy(jk,il),rban_vth(il), il=1,lrzmax)
            endif
         elseif (nnspec.eq.jk) then
            if(kspeci(2,jk).eq.'general' .and. 
     +                  iprovphi.ne.'disabled') then
            WRITE(6,9128) jk,kspeci(1,jk),kspeci(2,jk),bnumb(jk),ztext
            WRITE(6,9129) (il,ztr(il),reden(jk,il),temp(jk,il),
     +           vth(jk,il),energy(jk,il),enn(il,1),vphipl(il),
     +           il=1,lrzmax)
            else
            WRITE(6,9122) jk,kspeci(1,jk),kspeci(2,jk),bnumb(jk),ztext
            WRITE(6,9123) (il,ztr(il),reden(jk,il),temp(jk,il),
     +           vth(jk,il),energy(jk,il),enn(il,1),il=1,lrzmax)
            endif
         endif
 123  continue
CMPIINSERT_ENDIF_RANK

 9120 format(/" species no. ",i3,2x,a,a8,"    charge number: ",f6.2
     +  ,/,1x,15("="),//,"  l",4x,a8,5x,"density",4x,
     +  "temperature",6x,"vth", 9x,"energy", 4x,"rban_vth(cm)")
 9121 format(i3,1p6e13.5)
 9122 format(/" species no. ",i3,2x,a,a8,"    charge number: ",f6.2
     +  ,/,1x,15("="),//,"  l",4x,a8,5x,"density",4x,
     +  "temperature",6x,"vth",9x,"energy",4x,"neutral den")
 9123 format(i3,1p6e13.5)
 9125 format(/," along magnetic field line, at lrindx=",i3,":",/,
     +  9x,"s",/,(i3,1p5e13.5))
 9126 format(/" species no. ",i3,2x,a,a8,"    charge number: ",f6.2
     +  ,/,1x,15("="),//,"  l",4x,a8,5x,"density",4x,
     +  "temperature",6x,"vth",9x,"energy",3x,"tor vel(cm/s)")
 9127 format(i3,1p6e13.5)
 9128 format(/" species no. ",i3,2x,a,a8,"    charge number: ",f6.2
     +  ,/,1x,15("="),//,"  l",4x,a8,5x,"density",4x,
     +  "temperature",6x,"vth",9x,"energy",4x,"neutral den",2x,
     +  "tor vel(cm/s)")
 9129 format(i3,1p7e13.5)

CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)
      WRITE(*,*)'profiles: zeff(1:lrzmax)=',(zeff(il), il=1,lrzmax)
      WRITE(*,*)
CMPIINSERT_ENDIF_RANK  

c
      
      return
      end
c
c
      real*8 function tdprof(timet,y,t)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c
c     computes value of tdprof according to:
c     tdprof=y(1), timet < t(1)
c     tdprof=y(2)+0.5*(y(1)-y(2))*(1.+cos((timet-t(1))/(t(2)-t(1))*pi)),
c        timet in (t(1),t(2).
c     tdprof=y(2), timet > t(2)
c
      dimension y(2),t(2)
      data pi/3.141592653589793/

      if (timet.le.t(1)) then
         tdprof=y(1)
      elseif (timet.gt.t(2)) then
         tdprof=y(2)
      else
         tdprof=y(2)+0.5*(y(1)-y(2))*
     +        (1.+cos((timet-t(1))/(t(2)-t(1))*pi))
      endif

      return
      end
