
      subroutine ampfarl(distn,ll, ampf_curr,ampf_energy,ampf_reden)
      implicit integer (i-n), real*8 (a-h,o-z)

c     Returns result of an integral of the distribution function
c     distn, arising in the Ampere-Faraday eqns.
c     distn = combinations of h and g,  where h and g are similar
c     to the iterative solution introduced in 
c     ``Kinetic Modeling of SOL Plasmas'', K. Kupfer, R.W. Harvey, 
c     O. Sauter, G. Staebler, and M.J. Schaffer,  Physics of Plasmas
c     3,  3644 (1996).

      include 'param.h'
      include 'comm.h'
      real*8 distn(0:iy+1,0:jx+1) ! INPUT
      real*8 ampf_curr,ampf_energy,ampf_reden !OUTPUT Scalars; each ll
      real*8 f_2pi_sincos_dtheta ! local
      integer ll,ksp

c     following diaggnde integration
      tam3=zero !call bcast(tam3,zero,jx)
      tam2=zero !YuP[2019-12-18] Added calculation of density and energy,
      !for f[n], fh and fg input functions.
      
      do j=1,jx
         do i=1,iy
            f_2pi_sincos_dtheta= distn(i,j)*cynt2(i,ll)*coss(i,ll)
            tam3(j)=tam3(j)+f_2pi_sincos_dtheta
            tam2(j)=tam2(j)+abs(f_2pi_sincos_dtheta*tau(i,ll)) !YuP[2019-12]for FSAenergy
         enddo
      enddo
      
      cn=zero
      hn=zero
      sn=zero
      do j=1,jx
         !tam3(j)=tam3(j)*x(j)*gammi(j)*cint2(j)*dxi(j)
         cn=cn+tam3(j)*x(j)*gammi(j)*cint2(j) !we only need cn here; no need to keep tam3
         
         !Flux surface averaged energy (per particle)
         sn=sn+tam2(j)*cint2(j)*tcsgm1(j) !YuP[2019-12] tcsgm1=2*cnorm2*(gamma-1)
         !Also added: field line density (particles/cm**2)
         !Re eqsym: tam2 has tau factor, to be divided below by zmaxpsi.
         !tau is v*tau_B, and it's over 1/2 orbit if eqsym.ne.none
         hn=hn+tam2(j)*cint2(j) !YuP[2019-12]
    
      enddo

!     Adjust poloidal length dlpgpsii for eqsym.ne."none" cases
      !YuP: Already done in aingeom:
!      if (eqsym.eq."none") then
!         symm=one
!      else
!         symm=two
!      endif

      !factor1= clight*dtr/(4.d0*pi*3.e9) ! effectively included below,
      ! except 3.e9 .
      ! If you change definition of ampf_curr, be sure to adjust 
      ! factor1 in sub.ampfsoln
      ampf_curr= (4.d0*pi*(-charge*vnorm))/
cBH131110     +        vnorm*drpmconz(ll)*dlpsii(ll)*cn
cBH131110, BUT think vnorm* is correct, cancels vnorm in impanvc0:dfdvz
     +        (clight*dtr) *cn    ! now [statA/cm^2 /cm]
        !YuP[02-2017] removed drpmconz(ll) factor.
        !YuP[03-2017] Removed symm factor next to dlpgpsii(ll) :
        !Both dlpgpsii(ll) and dlpsii(ll) are calc-ed along same 
        !poloidal arclength, so either both should have symm* in front
        !or both should not have.
        !
        !YuP[03-2017] removed dlpsii(ll)/dlpgpsii(ll) factor here
        ! and added it in ampsoln as dlpsii(llp)/dlpgpsii(ll) 
        ! [note that dlpsii is at llp==l' index now, not ll]

      ksp=1 !=kelecg ! Only need electrons
      !YuP[2019-12] These values are needed for tdboothi.f :
      ! before calling, replace energy(ksp,ll) with ampf_energy, 
      ! and reden(ksp,ll) with ampf_reden
      ampf_energy=(sn/hn)*fions(ksp) !YuP[] FSA energy [keV] ! energy(ksp,ll)
      !Note: fions(k)=.5*fmass(k)*vnorm**2/ergtkev, in comm.h
      ampf_reden= hn/zmaxpsi(ll) !YuP[2019-12] FSA density ! reden(ksp,ll)
      
      return
      end subroutine ampfarl



      subroutine ampfalloc
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     Allocates arrays used in Ampere-Faraday routines.
c     COULD move alloc of some other A-F specific arrays here.
c..............................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE
c.......................................................................


c     NOTE: fh is alternatively allocated (differently) in wpalloc.f
c           when cqlpmod.eq.enabled.
c           fh and fg are allocated here for Kupfer functions.
c           Only needed for electrons.
      if(.NOT.ASSOCIATED(fh)) then ! fh is used in CQLP and A-F
        allocate(fh(0:iy+1,0:jx+1,1,1:lrz),STAT=istat)
        if(istat.ne.0) STOP 'ampfalloc: fh alloc problem'
      endif
      call bcast(fh,zero,SIZE(fh))

      if(.NOT.ASSOCIATED(fg)) then ! fg is used in CQLP and A-F
        allocate(fg(0:iy+1,0:jx+1,1,1:lrz),STAT=istat)
        if(istat.ne.0) STOP 'ampfalloc: fg alloc problem'
      endif
      call bcast(fg,zero,SIZE(fg))

      allocate(ampfln(1:lrz),STAT=istat)
      if(istat.ne.0) STOP 'ampfalloc: ampfln alloc problem'
      call bcast(ampfln,zero,SIZE(ampfln))

      allocate(ampflh(1:lrz),STAT=istat)
      call bcast(ampflh,zero,SIZE(ampflh))

      allocate(ampflg(1:lrz),STAT=istat)
      call bcast(ampflg,zero,SIZE(ampflg))

      allocate(ampfa(1:lrz,1:lrz),STAT=istat)
      call bcast(ampfa,zero,SIZE(ampfa))

      allocate(ampfb(1:lrz,1:lrz),STAT=istat)
      call bcast(ampfb,zero,SIZE(ampfb))

      allocate(ampfaa(1:lrz,1:lrz),STAT=istat)
      call bcast(ampfaa,zero,SIZE(ampfaa))

      allocate(ampfc(1:lrz),STAT=istat)
      call bcast(ampfc,zero,SIZE(ampfc))

      allocate(ampf2ebar(1:lrz),STAT=istat)
      call bcast(ampf2ebar,zero,SIZE(ampf2ebar))

      return
      end subroutine ampfalloc


      subroutine ampfinit 
      ! n is not updated yet, n=0,1,...,nstop-1
      ! Called from tdchief when n+1.eq.nonampf
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     Initialize arrays used in Ampere-Faraday routines.
c..............................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE
c.......................................................................
c     Electric field elecfld(0:lrza) units are the mixed Volts/cm.  
c     (1 cgs statvolt=300 volts).  elecfld is used for FP eqn solutions.
c
c     efflag="toroidal" is ensured for ampfmod, giving elecfld is toroidal.
c     On a flux surface, toroidal electric field varies strictly as 1/R,
c     due to tor symmetry. Therefore, picking a point on a flux surface
c     to evaluate the electric field gives it everywhere on the flux surface.
c
c     For ampf: elecfldn,delecfld0,delecfld0n (cgs:statV/cm) are used,
c         [BH190922: Need to check if delecfld0,delecfld0n really used for
c                  anything, any more.  I suspect not since as of 170328
c                  on need bin-centered electric fields.  Or, maybe rename
c                  delecfld0 to more appropriate delecfld.]
c     evaluated on each flux surface at major radius R=rmag posn on a
c     flux surface (that is, at poloidal angle ~pi/2), 
c     as is the main toroidal electric field variable, elecfld.
c     elecfldn(,,) are the values of elecfld used at each radius,time step,
c     and iteration, but in statV/cm, evaluated at radial bin centers (as
c     is elecfld).
c     elecfldn(0:lrz+1,0:nstop,0:nampfmax)=elecfldc,elecfld(1:lrz),elecfldb
c                                 for each n,iteration (converted to cgs).
c     delecfld0(1:lrz,1:nstop)= elecfld(1:lrz,n+1)-elecfld(1:lrz,n)
c                              i.e., total change in elecfld over time step
c                                    evaluated at bin centers.
c     delecfld0n(1:lrz,1:nstop,1:nampfmax)=change in electric field a each
c        step n=1:nstop, and each iteration, evaluated at bin centers.
c.......................................................................

      do nn=nonampf,nstop
        ! Do not set elecfldn(.,nn,.) to 0.d0 for nn<nonampf
        ! They were saved as the background el.fld at earlier steps.
        ! See tdchief, L~639
         do ll=1,lrz
            psi0bar(ll,nn)=one  !Not presently recalc'd or used.
         enddo  !On ll
      enddo  !On nn
      do nn=1,nstop
         do ll=1,lrz
            !delecfld0(ll,nn)=zero !YuP: not used
            do it=1,nampfmax
               delecfld0n(ll,nn,it)=zero !actually, already done in ainalloc
            enddo
         enddo
      enddo

c     Set initial, n=nonampf profiles
      elecfld(0)=elecfldc !Check elecfldc is set properly

cBH170329
      do ll=0,lrz
         elecfldn(ll,nonampf,0)=elecfld(ll)/300.d0
      enddo

c     Fill in elecfldn for n.lt.nonampf
c     with the n=nonampf values, for plotting purposes.
c YuP[21-08-2017] This is already done in tdchief, line ~700,
c and it is done properly: elecfldn(ll,n+1,niter)=elecfld(ll)/300.d0
c for all niter=0,nampfmax.
c So commenting it here:
c      if (nonampf.gt.0) then
c         do nn=0,nonampf-1
c            do ll=0,lrz
c               do it=0,nampfmax
c                  elecfldn(ll,nn,it)=elecfldn(ll,nonampf,1) ! for all it
c               enddo
c            write(*,*)'nn,ll,elecfldn*300=',nn,ll,elecfldn(ll,nn,1)*300
c            enddo
c         enddo
c      endif

      write(*,*)'-------------- done with ampfinit ------------'

      return
      end subroutine ampfinit




      subroutine ampfefldb(nn,timett)
      ! n is not updated yet, n=0,1,...,nstop-1
      ! Called from tdchief when n+1.eq.nonampf, as ampfefldb(n+1,time+dtr)
      ! So, nn= nonampf,nonampf+1,...,nstop
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     Get specified time-advanced boundary toroidal electric field.
c     Follow methods in subroutines tdxinitl and profiles.
c     BH: Probably useful to add one-toroidal turn voltage as function 
c     of time (or constant) to the namelist input.
c     Another input method, more in keeping with experiments, would 
c     specify total plasma toroidal current as a function of time.
c     Subroutine ampfsol would iterate the boundary voltage to find
c     efld giving the specified  current.
c     nn=time-advanced step
c     timett=time-advanced time
c.......................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

cBH131107: For starters, only use elec field specified by fixed parabola
cBH131107: input, or prbola-t, with efswtch=method1, tmdmeth=method1.

      if (nbctime.le.0) then !Time-indep edge bndry value elecfldb
      
c     Follow subroutine tdxinitl
        if (iproelec.eq."parabola") then
          elecfldn(lrz+1,nn,0)=elecfldb/300.d0 !here: ampfefldb(nn.ge.nonampf)
          ! nn= nonampf,nonampf+1,...,nstop
        elseif(iproelec.eq."spline") then ! YuP[03-2017] added option
          ! Was set in tdxinitl:
          !call tdinterp("zero","free",ryain,elecin,njene,rya(1),
     +    !       elecfld(1),lrzmax)
          !elecfldc=elecin(1)
          !elecfldb=elecin(njene)
          elecfldn(lrz+1,nn,0)=elecfldb/300.d0 !here: ampfefldb(nn.ge.nonampf)
        else
          STOP 
     +   'ampfefldb: only setup for iproelec.eq."parabola" or "spline" '
        endif

      else  !That is, time-dependent, nbctime.gt.0
c     Follow subroutine profiles
c     [But maybe just use eledfldb from sub profiles?]
      itab(1)=1
      itab(2)=0
      itab(3)=0

      if (tmdmeth.eq."method1") then
         itme=0
         do jtm=1,nbctime
            if (timett.ge.bctime(jtm)) itme=jtm
         enddo
         itme1=itme+1
      endif  !On tmdmeth
      
      if (efswtch.eq."method1") then
         if (iproelec.eq."prbola-t") then
            
            if (tmdmeth.eq."method1") then
               !YuP[2019-12-29] Removed ".and.elecc(1).ne.zero" (now can be 0.0)
               
               if (itme.eq.0) then
                  elecfldn(lrz+1,nn,0)=elecb(1)/300.d0
               elseif (itme.lt.nbctime) then
                  elecfldn(lrz+1,nn,0)=(elecb(itme)+(elecb(itme1)
     1                 -elecb(itme))/(bctime(itme1)-bctime(itme))
     1                 *(timett-bctime(itme)))/300.d0
               else
                  elecfldn(lrz+1,nn,0)=elecb(nbctime)/300.d0
               endif
            endif
         else !Not set (i.e., set to 0.) for any other iproelec.ne."prbola-t"
           WRITE(*,*)'ampfefldb: nn,elecfldn(lrz+1,nn,0)=',nn,
     &      elecfldn(lrz+1,nn,0)
           !pause
         endif ! iproelec.eq."prbola-t"
      endif  !On efswtch
      
      endif  !On nbctime

c     Scale boundary electric field
      elecfldn(lrz+1,nn,0)=elecscal*elecfldn(lrz+1,nn,0) !here: ampfefldb(nn.ge.nonampf)
      
      !YuP[21-08-2017] Added: copy the boundary value for iteration=0
      ! to all other iterations:
      do niter=0,nampfmax ! or up to nefitera
         elecfldn(lrz+1,nn,niter)=elecfldn(lrz+1,nn,0) !here: ampfefldb(nn.ge.nonampf)
      enddo
      

c     Set zero iteration of elecfldn equal to previous time
c     step radial profile, or previous iteration.
      !it_prev=0 ! YuP[21-08-2017] was 0
      it_prev=nampfmax ! But logically, should be the last? YuP[21-08-2017]
      do ll=0,lrz
         elecfldn(ll,nn,0)=elecfldn(ll,nn-1,it_prev)
         !YuP: Now: previous time step, but the LAST iteration at that step
      enddo

c      do ll=0,lrz+1
c      write(*,*)'ampfefldb: n,ll,elecfldn(ll,n+1,0)*300=',
c     +                      n,ll,elecfldn(ll,nn,0)*300
c      enddo

c     Zero delecfld0
      do ll=1,lrz
         !delecfld0(ll,nn)=zero !YuP: not used
         delecfld0n(ll,nn,1)=zero
      enddo

     
      return
      end subroutine ampfefldb


      subroutine ampfdiff(iflag)
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

c     No real test here, to start with.
c     Future: based on (elecfldn(ll,nn,it+1)-elecfldn(ll,nn,it+1))/
c             elecfldn(ll,nn,it+1).

      iflag=1  !returning test not satisfied, so all iterations carried out.

      return
      end subroutine ampfdiff



      subroutine ampfsoln(it,nn) ! Called from tdchief 
                                 ! here nn=nonampf,...,nstop
                                 ! it=1,nampfmax
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     Updates the toroidal electric field one iteration, using 
c     Ampere-Faraday eqns.  The Kupfer h and g functions have
c     been stored in fh and fg, for this iteration.
c     it=iteration number, starting at 1
c     nn=(advanced) time step, starting at 1
c.......................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      integer ipiv(lrz) ! no need to allocate or save YuP[2019-09]
      real*8 ampf_energy_n(lrz),ampf_energy_h(lrz),ampf_energy_g(lrz) !YuP[2019-12-18]
      real*8 ampf_reden_n(lrz), ampf_reden_h(lrz), ampf_reden_g(lrz)  !YuP[2019-12-18]
      !YuP[2019-12-18] Changed func.ampfarl to subroutine, 
      !and added computation of energy and density quantities,
      !which are needed as input for call_tdboothi.
      !Example: ampf_energy_n() means - energy based on f_ distr.function.
      !Example: ampf_energy_h() means - energy based on fh distr.function.
      !Example: ampf_energy_g() means - energy based on fg distr.function.
      real*8 dbscurm(lrz) !local  !YuP[2019-12-18] To form jbs[it]-jbs[n]
      real*8 dcurr(lrz) !local, To form dsigma*E[it] - dsigma*E[n] 
      real*8 dsig(lrz) !local, to form dsigma

      do ll=1,lrz
         temp2(0:iy+1,0:jx+1)=f_(0:iy+1,0:jx+1,kelec,ll)
         call ampfarl(temp2,ll, 
     &        ampfln(ll),ampf_energy_n(ll),ampf_reden_n(ll)) !temp2=f_
         temp2(0:iy+1,0:jx+1)=fh(0:iy+1,0:jx+1,kelec,ll)
         call ampfarl(temp2,ll, 
     &        ampflh(ll),ampf_energy_h(ll),ampf_reden_h(ll)) !temp2=fh
         temp2(0:iy+1,0:jx+1)=fg(0:iy+1,0:jx+1,kelec,ll)
         call ampfarl(temp2,ll, 
     &        ampflg(ll),ampf_energy_g(ll),ampf_reden_g(ll)) !temp2=fg
      enddo
      
      dbscurm(1:lrz)=zero !initialize. To form jbs[n+1,it]-jbs[n]
      dcurr(1:lrz)=zero !To form/save dsigma[n+1]*E[it] - dsigma[n]*E[n]
      dsig(1:lrz)=zero  !To form/save dsigma[n+1]
      factor1= clight*dtr/(4.*pi*3.e9) !see sub.ampfarl; usage: see below
      sig_starnue=zero
      currpar_starnue=zero
      sig_starnue0=zero
      currpar_starnue0=zero
      if(it.gt.1)then !YuP[2019-12-18] Form jbs[n+1,it]-jbs[n]
        !Here, we don't know f[n+1] distr.function yet.
        !We will use fh+deltaE*fg  as an approximation for f[n+1].
        !For tdboothi, need temp(k,ll) or energy(k,ll), and reden(k,ll)
        !as input.
        !===> To form jbs[n+1,it]-jbs[n] we call tdboothi
        !with values of energy and reden based on fh+deltaE*fg
        ! at each given iteration it, 
        !and with {energy(); reden()} for step=n
        !Form reden and energy at given iteration, using fh+deltaE*fg.
        !Use ampf_energy_h() == energy based on fh distr.function,
        !and other similar values, found by call_ampfarl above.
        !Note: delecfld0n(ll,nn,it) is computed below, AFTER call_dgesv.
        !Here, we can only have delecfld0n(ll,nn,it-1) at best.
        !Note that at it=1, delecfld0n(lll,nn,0) is not defined,
        !so we only form jbs[n+1]-jbs[n] starting from 2nd iteration.
        reden(kelecg,1:lrz)=  ampf_reden_h(1:lrz)
!YuP:canbe large negative &    +ampf_reden_g(1:lrz)*delecfld0n(1:lrz,nn,it-1)
        !For reden, the 2nd term can be from 10% up to 100% or more
        ! relative to ampf_reden_h;
        ! sometimes reden becomes <0 because of ampf_reden_g*delecfld0n.
        !Not sure what to do with it. Removing it from consideration.
        energy(kelecg,1:lrz)= ampf_energy_h(1:lrz)
     &    +ampf_energy_g(1:lrz)*delecfld0n(1:lrz,nn,it-1) !YuP[2019-12-18]
        !For energy, the contributions from  ampf_energy_g*delecfld0n
        ! is ~1e-6 of the ampf_energy_h value, i.e. very small.
        !Now we can calc. bscurm:
        if(ampfadd.eq."add_bscd" .or. ampfadd.eq."neo+bscd")then
          !YuP[2019-12-26] Added ampfadd
          if(jhirsh.ne.0) call tdboothi !get bscurm(ll,k,kk) for all ll(radial)
          !bscurm(1:lrz,1,2) is for '1'==electrons, '2'==non-maxwellian
          !So now we got jbs[n+1,it]-jbs[n] :
          dbscurm(1:lrz)=bscurm(1:lrz,1,2)-bscurm_n(1:lrz) != jbs[n+1,it]-jbs[n]
          !Note: bscurm_n was saved before starting it=1,nampfmax iterations,
          !in tddiag.f.
          !Note: to have same units as ampfln(ll) "current", we need to divide
          !bscurm (which is in A/cm^2) by factor1 (see below). 
          !So, when we form ampfb()~ampflh()-ampfln() coeff, we need to add 
          !  bscurm_n(ll)/factor1 to ampfln(ll)
          !and add
          !  bscurm(ll,1,2)/factor1 to ampflh(ll)
          !or simply add dbscurm()/factor1 to ampflh()-ampfln()
        endif !YuP[2019-12-26] done ampfadd
        !===> Now resistivity.
        if(ampfadd.eq."neosigma" .or. ampfadd.eq."neo+bscd")then
          !YuP[2019-12-26] Added ampfadd
          call starnue_sptz !Get starnue(),tauee(),taueeh(),sptzr(); all lr
          do lll=1,lrz 
          call tdnflxs(lll) ! determine l_,lr_, etc.
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
          !Save some values for verification:
          !Note: elecfld is found below (after call_dgesv) as 
          ! elecfld(ll)=elecfldn(ll,nn,it)*300.d0 ! V/cm
          !Therefore, here elecfld is from previous iteration 'it'
          elec_cgs=elecfld(lr_)/300. ! statV/cm
          currpar_starnue(lll)= elec_cgs*sig_starnue(lll)/3.d9 ![A/cm^2]
          currpar_starnue0(lll)=elec_cgs*sig_starnue0(lll)/3.d9 ![A/cm^2]
          !From printout, currpar_* is same order of magn. 
          !as current based on distr.func.
          !When we form ampfb()~ampflh()-ampfln() coeff, we need to add 
          !(currpar_starnue_n(ll)-currpar_starnue0_n(ll))/factor1 to ampfln(ll)
          !and add
          !(currpar_starnue(ll)-currpar_starnue0(ll))/factor1 to ampflh(ll)
          !Or simply add dcurr/factor1 to ampflh()-ampfln(), where
          dcurr(lll)=(currpar_starnue(lll)-currpar_starnue0(lll))
     &          -(currpar_starnue_n(lll)-currpar_starnue0_n(lll))
          !Note that dcurr uses the same elec_cgs==E[n] 
          !(actually, from previous iteration), i.e. it is
          ! dcurr = dsigma[n+1]*E[n] - dsigma[n]*E[n]
          !The complete expression is 
          ! dsigma[n+1]*(E[n]+deltaE) - dsigma[n]*E[n]
          !The "missing" part, dsigma[n+1]*deltaE, is included through ampfa()
          !---> So, we need to adjust ampfa()
          !Recall that deltaE== delecfld0n(ll,nn,it) ![statV/cm]
          !and it is not included into ampfa() coeff
          ! [it is to be found, i.e. it is the unknown vector].
          !So, we form 
          dsig(lll)= (sig_starnue(lll)-sig_starnue0(lll))/3.d9
          !When mult-ed by deltaE[statV/cm], we get delta_current[A/cm^2]
          !For ampfa() coeff, we add dsig()/factor1
          enddo ! lll
        endif !YuP[2019-12-26] done ampfadd
        
        !Below, just a printout, for verification purposes.
        !Note: if ampfln() is mult-ed by 
        ! factor1= clight*dtr/(4.*pi*3.e9) ! see inside sub.ampfarl
        ! the units become A/cm^2.
        ! Example of these values:
        ! bscurm(ll,1,2), ampfln(ll)*factor1,  ampflh(ll)*factor1,
        !       ampflg(ll)*delecfld0n(ll,nn,it)*factor1
        ! -0.153E+03 -0.744E+03 -0.747E+03  0.298E+01 !Example of printout
        do lll=1,lrz 
         !-1-> Printout the currents - they are in A/cm^2
!         write(*,'(a,4e11.3)')'bscurm,ampfl[n,h,g]*cdt/4pi3e9',
!     &   bscurm(lll,kelecg,2),
!     &   ampfln(lll)*factor1,  ampflh(lll)*factor1,
!     &   ampflg(lll)*delecfld0n(lll,nn,it-1)*factor1
         ! -0.153E+03 -0.744E+03 -0.747E+03  0.298E+01 !Example of printout
         !Note: delecfld0n(ll,nn,it) is computed below.
         !Here, we can only have delecfld0n(ll,nn,it-1) at best.
         !Note that at it=1, delecfld0n(lll,nn,0) is not defined.
         !-2-> Printout energies [keV]
!         write(*,'(a,4e11.3)')'energy,ampf_energy_n,h,gdE',
!     &   energy(kelecg,lll),
!     &   ampf_energy_n(lll),  ampf_energy_h(lll),
!     &   ampf_energy_g(lll)*delecfld0n(lll,nn,it-1)
         !Example of printout for the above(energy):
         !energy,ampf_energy_n,h,gdE  0.427E+01  0.427E+01  0.424E+01 -0.283E-06
         !energy,ampf_energy_n,h,gdE  0.651E+01  0.651E+01  0.647E+01 -0.269E-06
         !energy,ampf_energy_n,h,gdE  0.930E+01  0.930E+01  0.925E+01 -0.296E-06
         !The values energy() and ampf_energy_n() are same because
         !they are based on actual distr. function f_.
         !The values of energy() and ampf_energy_h[based on f_h] are close.
         !The values of ampf_energy_g*delecfld0n are always small. 
         !-3-> Printout densities [cm^-3]
!         write(*,'(a,4e11.3)')'reden,ampf_reden_n,h,gdE',
!     &   reden(kelecg,lll),
!     &   ampf_reden_n(lll),  ampf_reden_h(lll),
!     &   ampf_reden_g(lll)*delecfld0n(lll,nn,it-1)
         !Example of printout for reden:
         !reden,ampf_reden_n,h,gdE  0.800E+14  0.800E+14  0.800E+14 -0.191E+12
         !reden,ampf_reden_n,h,gdE  0.798E+14  0.798E+14  0.798E+14 -0.420E+12
         !The values of reden(), reden_n[based on f_], reden_h[based on f_h]
         !are same or very close. 
         if(reden(kelec,lll).le.zero)then
            WRITE(*,*)'ampfsoln/WARN: reden<0', ampf_reden_h(lll),
     &        ampf_reden_g(lll)*delecfld0n(lll,nn,it-1) 
              ! the last one may be large negative ==> reden becomes negative
              ! if ampf_reden_g()*delecfld0n() is included
         endif
         !-4-> Print dbscurm and other currents  [A/cm^2]
!         write(*,'(a,4e14.6)')'bscurm_n,dbscurm,Jn,dj[A/cm2]',
!     &   bscurm_n(lll),dbscurm(lll),ampfln(lll)*factor1,
!     &   dsig(lll)*delecfld0n(lll,nn,it-1)
         !example of printout: 
         ! bscurm_n=-0.150702E+03     dbscurm=-0.424516E-01
         ! ampfln(lll)*factor1=-0.775531E+03
         ! currpar_starnue=    -0.317958E+03 
         ! currpar_starnue0=   -0.292161E+03
         !currpar_starnue-currpar_starnue0=     -0.282663E+02 (-0.262256E+02) 
         !currpar_starnue_n-currpar_starnue0_n= -0.593003E+02 (-0.252757E+02)
         !dsig(lll)*delecfld0n(lll,nn,it-1)= -0.672896E+01
        enddo ! lll
!        write(*,*)'---ampfsoln  nn(=n+1), it=',nn,it
      endif !(it.gt.1) !YuP[2019-12-18] Form jbs[n+1]-jbs[n] , etc


      one_cr= one/(clight*rmag) !YuP[2019-12-18] Combined factors
      do ll=1,lrz ! == l index in notes
c        BH140302: adjusting - to +, and back
         ampftmp= -one_cr*drpmconz(ll)*rpconz(ll)/dlpgpsii(ll)
         !YuP[2019-12-18] Added 1/dlpgpsii(ll) in the above, removed in dldl below.
         do llp=1,lrz ! == l' index in notes
            !YuP[03-2017] removed dlpsii(ll)/dlpgpsii(ll) factor in ampfarl 
            !and added it here as dlpsii(llp)/dlpgpsii(ll) 
            ![note that dlpsii is at llp==l' index now, not ll]
            dldl=dlpsii(llp) !YuP added here. YuP[2019-12-18] moved 1/dlpgpsii
            ! For a cylinder, dlpsii/dlpgpsii is approximately r(l')/r(l),
            ! where r(l) is the minor radius of flux surf. #l.
            ampfa(ll,llp)=dldl*ampftmp*drpmconz(llp)*
     &              ( ampflg(llp) + dsig(llp)/factor1 )
            ampfb(ll,llp)=dldl*ampftmp*drpmconz(llp)*
     &              ( ampflh(llp)-ampfln(llp) 
     &                 + (dbscurm(llp) + dcurr(llp))/factor1 
     &               )
            !YuP[2019-12-18] added dbscurm(llp)/factor1 in the above.
            !Also added dcurr(llp)/factor1, which is same as adding
            !(currpar_starnue(llp)-currpar_starnue0(llp))/factor1 to ampflh(llp)
            !and 
            !(currpar_starnue_n(llp)-currpar_starnue0_n(llp))/factor1 to ampfln(llp)
            !Also, added dsig(llp)/factor1 to ampfa().
            !Note: dsig(l)*delecfld0n(l,nn,it) gives [A/cm^2]
         enddo
      enddo

c     Set up matrix ampfaa for the delecfld0n(1:lrz,nn,it)
ctmp      Set up ampfaa (coefficient next to deltaE(llp,n+1)
      call bcast(ampfaa,zero,lrz*lrz)
      do ll=1,lrz     !== l  in notes 
         do llp=1,lrz !== l' in notes
            if (llp.eq.ll) then ! l'=l
               ampfaa(ll,llp)= two -ampfa(ll,llp)
            ! Summing a_{l,l'} over l'  -  Up to l or l-1 ?  
            ! (if - up to l-1, comment ampfa(ll,llp) in the above 
            ! Tests: a very little difference
            ! (summing up to l-1 gives a little bit faster decay of I current)
            elseif (llp.gt.ll) then ! l'>l
               ! This should be only present if(ll.le.lrz-1)
               ! But from condition ll < llp (<= lrz)
               ! it is clear that here we can only find ll.le.lrz-1
               ampfaa(ll,llp)= four*(-one)**(llp-ll)
            else ! llp < ll  (l' < l in notes)
               ampfaa(ll,llp)= -ampfa(ll,llp)
            endif
         enddo
      enddo

!      WRITE(*,'(a,3i4)')'ampfsoln: nn,it-1,mpirank=', nn,it-1,mpirank
!      do ll=0,lrz+1
!        WRITE(*,'(a,i4,2e15.6)')
!     +'ampfsoln before soln: ll,elecfldn(ll,nn,it-1)*300,elecfld(ll)=',
!     +                       ll,elecfldn(ll,nn,it-1)*300,elecfld(ll)
!      enddo

c     Set up the rhs vector
ctmp      Setup ampf2ebar,ampfc  (1:lrz), Check elecfldn has it=0 dimensioning
ctmp      and set up.  elecfldn(,,0) is set up from elecfld at previous
ctmp      time step.
      do ll=1,lrz
         ! This is 2E^bar (l,n) in notes
cBH170329 ampf2ebar(ll)=elecfld0n(ll,nn,it-1)+elecfld0n(ll-1,nn,it-1)![statV/cm]
         ampf2ebar(ll)=two*elecfldn(ll,nn,it-1) ![statV/cm]
         !here nn is from nonampf,...,nstop  range
      enddo

      call bcast(ampfc,zero,lrz)
      do ll=1,lrz
         ! This is -2E^bar(l) -2E(lrz)
         ampfc(ll)= -ampf2ebar(ll) 
     +     +two*(-one)**(lrz-ll)*elecfldn(lrz+1,nn,it-1) ![statV/cm]
         ! YuP: the edge term is corrected
         if(ll.le.lrz-1)then
         do llp=ll+1,lrz ! (l < l' < = lrz in notes)
            ampfc(ll)=ampfc(ll)-two*(-one)**(llp-ll)*ampf2ebar(llp) ![statV/cm]
            ! YuP[02-2017] corrected (-one)**(llp-l) to (-one)**(llp-ll)
         enddo
         endif
         do llp=1,ll !ll-1 !Add the SUM[b(l,l')], with l' from 1 to l
            ! Up to ll or ll-1 ?   Tests: a very little difference
            ! (summing up to ll-1 gives a little bit faster decay of I current)
            ampfc(ll)=ampfc(ll)+ampfb(ll,llp) ![statV/cm]
         enddo
      enddo

c     Solve linear equations for delecfld0n(ll,nn,it) using
c     Gaussian elimination with pivoting.

!      if (.not. allocated(ipiv)) then
!         allocate(ipiv(lrz),STAT=istat)
!         if (istat.ne.0)  STOP 'ampfsoln: ipiv alloc problem'
!      endif
      call dgesv(lrz,1,ampfaa,lrz,ipiv,ampfc,lrz,info)
      !    DGESV( N,NRHS, A,  LDA,IPIV,  B,  LDB,INFO ) 
      !computes the solution to a system of linear equations A * X = B
      !So, the matrix A(N,N) is ampfaa, and the vector B is ampfc.
      !On exit, B contains the solution vector X, which corresponds to
      !the values of delecfld0n(ll,*,*) for all ll=1:lrz,
      !which is the time-step (or iteration) change in electric field 
      !evaluated at bin centers ll=1:lrz.
      
      if (info .ne. 0) then
         PRINT *,' warning after dgesv in ampfsoln: mpirank,info= ',
     &      mpirank,info
         stop 'ampfsoln: info.ne.0'
      endif

c     Update elecfld (tor elec fld) to present iteration
c     The ampfc soln is based on statvolt/cm elecfld.
c     (Alternatively, consider code mod to work entirely in V/cm?)
      do ll=1,lrz
         delecfld0n(ll,nn,it)=ampfc(ll) ! statvolt/cm
         elecfldn(ll,nn,it)=elecfldn(ll,nn,it-1)+delecfld0n(ll,nn,it)
         elecfld(ll)=elecfldn(ll,nn,it)*300.d0 ! V/cm
      enddo
      
      ! YuP[2019-12-27] Added E(0) values 
      !(only used for plots)
      !elecfld(0)=elecfld(1) ! Simple version.
      !Another version, based on parabolic shape of E(rho) and
                                !dE/drho=0 at rho=0, per diffusion eqn:
      ! E(1)= E(0) - a*rho(1)**2
      ! E(2)= E(0) - a*rho(2)**2
      !From the above, we get a= (E(1)-E(2))/(rho22-rho12)
      !and  2*E(0)= (E(1)+E(2)) +a*(rho22+rho12)
      !where
      rho12=rya(1)**2
      rho22=rya(2)**2
      !Then
      elecfld(0)= 0.5*( elecfld(1)+elecfld(2) + 
     &               (elecfld(1)-elecfld(2))*(rho12+rho22)/(rho22-rho12)
     &                )
      
!      do ll=0,lrz+1
!        WRITE(*,'(a,2i4,2e15.6)')
!     +'ampfsoln aft soln:Mpirank,ll,elecfldn(ll,nn,it)*300,elecfld(ll)',
!     +                   mpirank,ll,elecfldn(ll,nn,it)*300,elecfld(ll)
!      enddo
      !here nn is from nonampf,...,nstop  range
      
c     Update total delecfld0 for this time step nn
!      do ll=1,lrz
!YuP: not used  delecfld0(ll,nn)=delecfld0(ll,nn)+delecfld0n(ll,nn,it) ![statV/cm]
!      enddo

      return
      end subroutine ampfsoln
