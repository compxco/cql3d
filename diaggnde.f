c
c
      subroutine diaggnde
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     Main diagnostic routine; computes densities, currents,
c     energies, z=effective, J/P. Also redefies electron density
c     to maintain quasineutrality if necessary.
c..................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE
      
c.......................................................................

      if (oldiag .ne. "enabled") then
c     use Legendre decomposition for computing the integrals
        call diaggnde2
        return
      endif

c..................................................................
c     Compute integrals for various moments
c     tam1 will contain midplane density moment
c     tam2 will contain field line (flux surface averaged) density
c     tam3 will contain parallel current density - at midplane.
c     tam4 related to parallel energy calc
c     tam5 reltaed to perpendicular energy calc
c..................................................................

      currmt(l_)=0.d0
      currmtp(l_)=0.d0

c     For cqlpmod.ne."enabled", default lmdpln_ will take on values
c       lrz:1 and successive calls (for given time step).  l_ also
c       takes on these values, so following if statement is satisfied
c       at each call diaggnde.
c     For cqlpmod.eq."enabled", lmpdpln_ will equal 1 at each call,
c       whereas l_ varies lrors:1, representing distance along B-field.
c       Only a single flux surface is examined, lrz=1, and lr_ is a
c       chosen integer.
      if (l_ .eq. lmdpln_) then
        currt(lr_)=0.d0
        currtp(lr_)=0.d0
        curtor(lr_)=0.d0
        curpol(lr_)=0.d0
      endif
      do 90 k=1,ngen
        call bcast(tam1,zero,jx)
        call bcast(tam2,zero,jx)
        call bcast(tam3,zero,jx)
        call bcast(tam4,zero,jx)
        call bcast(tam5,zero,jx)
        do 10 j=1,jx
        do 20 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
           !Note that for meshy="fixed_mu" iy_(l_) can be less than iy
            !if(gone(i,j,k,lr_).eq.zero)then !YuP[07-14-2014]
               ! Only count particles that are NOT in the loss cone
               ! (gone array is set to -1 in the loss cone).
               ! Another option would be simply to remove particles
               ! in the loss cone, but tests show that the solution
               ! can become unstable in such case.
               ! So, keep f() non-zero in the loss cone, 
               ! but do not add such f() to the integrals/sums over d3v.
              tam1(j)=tam1(j)+f(i,j,k,l_)*cynt2(i,l_)
              tam3(j)=tam3(j)+f(i,j,k,l_)*cynt2(i,l_)*coss(i,l_)
c     include vptb=|cos(th0)| * tau
            tam2(j)=tam2(j)+f(i,j,k,l_)*cynt2(i,l_)*
     1        abs(coss(i,lmdpln_))*tau(i,lr_)
            tam4(j)=tam4(j)+f(i,j,k,l_)*cynt2(i,l_)*
     1        abs(coss(i,lmdpln_))*tau(i,lr_)*(1.-sinn(i,l_)**2*
     1        psiba(i,lr_))
cBH080502            tam5(j)=tam5(j)+f(i,j,k,l_)*cynt2(i,l_)*vptb(i,lr_)*
cBH080502  Error:  only need one vptb= abs(coss)*tau factor
            tam5(j)=tam5(j)+f(i,j,k,l_)*cynt2(i,l_)*
     1        abs(coss(i,lmdpln_))*tau(i,lr_)*(sinn(i,l_)**2*
     1        psiba(i,lr_))
            !endif
 20     continue ! i
 10     continue ! j

c     write(*,*) 'diaggnde:l_,(tau(i,lr_),i=1,iy)', l_,(tau(i,lr_),i=1,iy)

        denra(k,l_)=0.d0
        curra(k,l_)=0.d0
c-----------------------------
c   denra is the flux surface average runaway density.
c   curra is the flux surface area average parallel runaway current 
c   density.
c   fdenra is fraction of density that is runaway.
c   fcurra is fraction of current due to runaway.
c-----------------------------
c
        gn=0.d0
        en=0.d0
        hn=0.d0
        sn=0.d0
        cn=0.d0
        wpar_=0.d0
        wperp_=0.d0

        do 40 j=1,jx

c..................................................................
c     gn= midplane density
c..................................................................

          gn=gn+tam1(j)*cint2(j) !== n0 midplane density 
          !except, may not be properly scaled yet, at t=0, because 
          !distr.func. which is set in finit (for ZOW or HYBRID_FOW)
          !is not yet properly scaled. This is done further in this 
          !subroutine. 

c..................................................................
c     midplane current density 
c     currv is integrand in velocity integral for parallel current,
c       at the midplane.
c     currvs contains partial sums of the currv, up to velocity x.
c     cn is the full integral.
c..................................................................

          currv(j,k,l_)=tam3(j)*x(j)*gammi(j)*cint2(j)*dxi(j)
          if (j.eq.1) then
            currvs(j,k)=dx(1)*currv(j,k,l_)
          else
            currvs(j,k)=currvs(j-1,k)+currv(j,k,l_)*dx(j)
          endif
          cn=cn+currv(j,k,l_)*dx(j)

c..................................................................
c     field line density (particles/cm**2)
c     Re eqsym: tam2 has tau factor, to be divided below by zmaxpsi.
c     tau is v*tau_B, and it's over 1/2 orbit if eqsym.ne.none
c..................................................................

         !Field-line density:
         hn=hn+tam2(j)*cint2(j) !actually 1/2 line-density if eqsym.ne.none

c..................................................................
c     Flux surface averaged energy (per particle)
c..................................................................

          sn=sn+tam2(j)*cint2(j)*tcsgm1(j)

c..................................................................
c     Midplane average energy per particle
c..................................................................

          en=en+tam1(j)*cint2(j)*tcsgm1(j)

c..................................................................
c     flux surface averaged parallel and perpendicular energy density
c     Re eqsym:  wpar_/wperp_ has psiba=int_over_l{dtau/bbpsi} factor
c......................................................................

          wpar_=wpar_+tam4(j)*cint2(j)*tcsgm1(j)
          wperp_=wperp_+tam5(j)*cint2(j)*tcsgm1(j)

 40     continue ! j
c-------------------


c     call routine to determine jxcrit (presently determined as
c       index for amin1(3.*clight,ucrit).
        call soucrit

        if (jxcrit(k,l_).ge.jx) go to 50
        do j=jxcrit(k,l_),jx  !YuP[2021-04] (k,lr_) --> (k,l_)
           denra(k,l_)=denra(k,l_)+tam2(j)*cint2(j)
           curra(k,l_)=curra(k,l_)+currv(j,k,l_)*dx(j)
        enddo
 50     continue
 
c..................................................................
c     If density computed is negative call exit
c     Energies are in Kev or Kev/cm**3 (after multiplication by
c     fions)
c..................................................................
        if ((gn.le.0.d0).or.(hn.eq.0.d0)) then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'diaggnde: k,lr_,gn,hn,sum(f)=', k,lr_,gn,hn,
     &     sum(f(1:iy,1:jx,k,lr_))
CMPIINSERT_ENDIF_RANK
          call diagwrng(10) !stops at all cores
        endif
        fdenra(k,l_)=denra(k,l_)/hn
        if(cn.ne.0.d0) fcurra(k,l_)=curra(k,l_)/cn
        energym(k,l_)=en/gn*fions(k)  ! at midplane
        enrgypa(k,ls_)=en/gn*fions(k) ! at midplane
        if (l_ .eq. lmdpln_) then
          energy(k,lr_)=sn/hn*fions(k) ! FSA for ZOW only
          ! at n>0, this definition will be over-written by FOW/ZOW
          ! universal procedure: through reconstruction of local f(R,Z).
cBH080502          wpar(k,lr_)=wpar_*fions(k)/zmaxpsi(lr_)*ergtkev
cBH080502          wperp(k,lr_)=wperp_*fions(k)/zmaxpsi(lr_)*ergtkev
cBH180531:  Following wpar/wperp (diaggnde2 also) needs clarification..
          wpar(k,lr_)=wpar_*fions(k)/hn
          wperp(k,lr_)=wperp_*fions(k)/hn
        endif
cCMPIINSERT_IF_RANK_EQ_0   
c        write(*,*)
c     + 'diaggnde: k,lr_,l_,en,hn,fions(k),energym(k,l_),energy(k,lr_)',
c     +            k,lr_,l_,en,hn,fions(k),energym(k,l_),energy(k,lr_)
cCMPIINSERT_ENDIF_RANK

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
c          print *,'diaggnde:  l_= ',l_,'  gn,hn= ',gn,hn
c%OS  
          gni=1.d0/gn
          hni=0.d0
          if (l_ .eq. lmdpln_) hni=1.d0/hn
          if (l_ .eq. lmdpln_) hnis(k,lr_)=hni

c..................................................................
c     This assures the midplane density of reden for f
c..................................................................

          zfact=hni*zmaxpsi(lr_)*reden(k,lr_)
          if (cqlpmod.eq."enabled") zfact=gni*denpar(k,ls_)

CMPIINSERT_IF_RANK_EQ_0   
          if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
          if (cqlpmod.eq."enabled") then
            !WRITE(*,*)'----------------------- ls_===', ls_
            WRITE(*,'(a,i4,3e12.4)')
     +      'diaggnde_n=0 ls_, denpar, gn, sum_ij(gone)',
     +                   ls_,denpar(k,ls_),gn,sum(gone(:,1:jx,k,lr_))
             !Is gone() array used in CQLP? It is allocated for lrz=1
             !in case of CQLP
          else
            !WRITE(*,*)'----------------------- lr_===', lr_
            WRITE(*,'(a,2i4,3e12.4)')
     +      'diaggnde_n=0 k,lr_, reden, gn, sum_ij(gone)',
     +                    k,lr_,reden(k,lr_),gn,sum(gone(:,1:jx,k,lr_))
          endif
          endif
CMPIINSERT_ENDIF_RANK
     
          if ( (nlrestrt.ne."disabled")
     +         .or. (fpld(1,1).eq.-1.0)) then
             continue
          else
             call dscal(iyjx2,zfact,f(0,0,k,l_),1) 
             !where zfact=hni*zmaxpsi(lr_)*reden(k,lr_)
          endif
c          write(*,*)'diaggnde: k,lr_,zfact,reden(k,lr_)=',
c     +                         k,lr_,zfact,reden(k,lr_)

c..................................................................
c     xlndn is the field line density (particles/cm**2)
c     xlndn/zmaxpsi(lr_) will equal the flux surface averaged (volume
c     averaged) density.
c..................................................................

          if (l_ .eq. lmdpln_) then
            xlndn(k,lr_)=reden(k,lr_)*zmaxpsi(lr_) !n=0: an approximation
cBH080502            wperp(k,lr_)=wperp(k,lr_)*reden(k,lr_)*hni*zmaxpsi(lr_)
cBH080502            wpar(k,lr_)=wpar(k,lr_)*reden(k,lr_)*hni*zmaxpsi(lr_)
            wpar(k,lr_)=wpar_*fions(k)/hn
            wperp(k,lr_)=wperp_*fions(k)/hn
          endif

c..................................................................
c     currm(k,l_) in CQLP, it is the non-explicitly s-dependent parallel current
c       density, for each general species k. 
c     In CQL3D, this is equal to the current density at the outer
c        midplane due to each general species k.
c     note:
c     current(z)=currm(k,l_)*bbpsi(z,lr_)
c     currm(k,lmdpln_)=parallel current at outer midplane
c
c     curr is the flux surface toroidal-area-averaged current density.
c     To wit: curr*dA=the total number of particles/sec passing through
c     an area element associated with the cross sectional area of the
c     flux surface in question. Once again curr and currm are
c     PARALLEL current.  (In non-tokamak situations, it will make
c     more sense to consider Toroidal current times toroidal area.)
c..................................................................

          faccur=reden(k,lr_)*hni*zmaxpsi(lr_)*vnorm*bnumb(k)*charge
          if (cqlpmod.eq."enabled") faccur=denpar(k,ls_)*vnorm*
     *      bnumb(k)*charge*gni
          currm(k,l_)=faccur*cn
          curra(k,l_)=faccur*curra(k,l_)

c..................................................................
c     psifct will contain the factor needed to modify midplane current
c     to flux surface (toroidal-area-)averaged current.
c     Thus,
c       Avg curr = int{d(tor area) * parallel_curr density}/
c                  int{d(tor area)}
c       parallel_curr=parallel_cur_at_min_B_pt * (B/min_B)
c       gives 
c      Avg curr = <B/(min_B * R)>/<1/R>, where < >= FSAvg.
c              
c..................................................................

c..................................................................
c     old bug next
c     psifct=radmaj/(radmaj**2-rovera(lr_)**2*radmin**2)**.5
c     for eqmod=disabled: psifct=sqrt((1+eps)/(1-eps))
c..................................................................

          psifct=psiovr(lr_)/onovrp(1,lr_)
          if (l_ .eq. lmdpln_) then
             curr(k,lr_)=currm(k,l_)*psifct
             curra(k,l_)=curra(k,l_)*psifct
             denra(k,l_)=one_*denra(k,l_)/zmaxpsi(lr_)
          endif

          fgni=faccur*psifct  !/3.e+9 YuP[2021-03-23] removed 1/3.e9
          !Why is currv converted to A/cm2 here, at n=0, but not at n>0?
          !Leaving this conversion to pltendn and pltends,
          !where currv is plotted 
          if (cqlpmod.eq."enabled") fgni=faccur !/3.e+9 YuP[2021-03-23]
          call dscal(jx,fgni,currv(1,k,l_),1) !Here: at n=0
          call dscal(jx,fgni,currvs(1,k),1)   !Here: at n=0
          !YuP[2021-03-23] :
          !Note that currvs() does not have l_ index, 
          !so it should be recomputed for each given l_, from currv(),
          !in pltendn and pltends (the only two places where it is used)

c..................................................................
c     Standard, non-initialization time step (n.ne.0) logic follows..
c..................................................................

        else                                  ! n.ne.0
          faccur=one_*vnorm*bnumb(k)*charge
          psifct=psiovr(lr_)/onovrp(1,lr_)

c..................................................................
c     xlndn(k,lr_) is the field line density
c     reden(k,lr_) is the flux averaged density
c     denpar(k,ls_) is the local in s density
c
c     currm(k,l_) is the local in s current (s:along the magnetic field)
c       (For CQL3D, l_=lmdpln_, this is current density at midplane.
c        For CQLP, l_=lmdpln_=1, is current density at the midplane.)
c     currmtp(l_) sums currm(k,l_) over general species k, including
c                the general local s=0 electron contribution.
c     currmt(l_) is the local s=0 sum of parallel curr. den. 
c                over the ionic gen. species, 
c                i.e., currmtp(l_) minus any electron general species.
c                contribution .
c     curr(k,lr_) is flux surface (area-)average current 
c                density at l_=lmdpln_.
c     currtp(lr_) is sum over general species k of curr(k,lr_)
c     currt(lr_) is sum or flux surface (toroidal-area-)average 
c                parallel curr. den. over the ionic gen. species, 
c                i.e., currmtp(l_) minus any electron general species.
c                contribution.
c..................................................................

          fgni=faccur*psifct  !Here: n>0
          if (cqlpmod.eq."enabled") fgni=faccur  !Here: n>0
          !YuP[2021-03-23] Corrected fgni factor for CQLP [i.e., no psifct]
          !Now currm(k,l_) and currvs(jx,k) are same.
c         Scaling units of currv to be statamps/cm**2*vnorm,
c                          currvs to be statamps/cm**2
          !Leaving conversion to A/cm^2 to pltendn and pltends,
          !where currv is plotted 
          call dscal(jx,fgni,currv(1,k,l_),1)  !Here: n>0
          call dscal(jx,fgni,currvs(1,k),1)    !Here: n>0
          currm(k,l_)=faccur*cn                !Here: n>0
          curra(k,l_)=faccur*curra(k,l_)
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
            curra(k,l_)=curra(k,l_)*psifct
            currtp(lr_)=currtp(lr_)+curr(k,lr_)
            xlndn(k,lr_)=one_*hn
            reden(k,lr_)=one_*hn/zmaxpsi(lr_)
            denra(k,l_)=one_*denra(k,l_)/zmaxpsi(lr_)
          endif
          
        endif  ! n.eq.0/else


c..................................................................
c     Breaking parallel current densities down into toroidal
c     and poloidal contributions (Amps/cm**2):
c     curtor(lr_) is parallel current density vector dotted into
c                  toroidal direction, evaluated at outer eq. plane.
c     curpol(lr_) is parallel current density vector dotted into
c                  poloidal direction, evaluated at outer eq. plane.
c..................................................................

        curtor(lr_)=curtor(lr_)+(fpsi(lr_)/rpcon(lr_))/
     +       bmidplne(lr_)*currm(k,l_)/3.e9
        curpol(lr_)=curpol(lr_)+bthr0(lr_)/
     +       bmidplne(lr_)*currm(k,l_)/3.e9

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

        sorpwt(lr_)=0.d0
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
            areaovol=1.d0/twopi*onovrp(1,lr_)
          else
            areaovol=1.d0/twopi/radmaj
          endif
        endif

c..................................................................
c     sorpwt(lr_) is the combined nbi and rf power (from urf module)
c..................................................................

        if (n.gt.0) then
          if (sorpwt(lr_).ge.1.e-25) then
            bdre(lr_)=areaovol*currt(lr_)/(sorpwt(lr_)*3.e9)
            bdrep(lr_)=areaovol*currtp(lr_)/(sorpwt(lr_)*3.e9)
          else
            bdre(lr_)=0.d0
            bdrep(lr_)=0.d0
          endif
        endif

c.......................................................................
c     Compute Z-effective
c.......................................................................
      if(cqlpmod.ne."enabled")then !YuP[2021-02]
       zeff(lr_)=0.d0 !Notice: full grid here, same as for reden()
       zeff4(lr_)=0.d0 !Yup[2014-05-27] Initialize to 0.
      else ! (cqlpmod.eq."enabled")
       zeff(ls_)=0.d0 !Notice: full grid here, same as for denpar()
       zeff4(ls_)=0.d0 
      endif 
      zeff1=0.d0
      xq=0.d0
      if (izeff.eq."backgrnd") then ! based on all ions (gen+maxw)
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
        
      elseif (izeff.eq."ion") then ! based on maxw. ions only
         do kk=1,nionm ![2025-10] If no Maxw.ions then zeff-->0
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
         if(zeff1.eq.0.d0)then ![2025-10] If no Maxw.ions then zeff=0
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,'(a,2i4,1pe12.4)')
     +     'diaggnde: n,lr,zeff1=', n, l_, zeff1
CMPIINSERT_ENDIF_RANK
           STOP 'Check/change izeff. Possibly no Maxw.ions'
         endif
         
      else
         WRITE(*,*) 'Wrong izeff specification; izeff=',izeff
         stop
      endif
      
      if(cqlpmod.ne."enabled")then !YuP[2021-02]
       zeff4(lr_)=zeff4(lr_)/xq
       zeff(lr_)=zeff(lr_)/zeff1
      else ! (cqlpmod.eq."enabled")
       zeff4(ls_)=zeff4(ls_)/xq
       zeff(ls_)=zeff(ls_)/zeff1
      endif

CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,'(a,2i4,1pe12.4)')
     +     'diaggnde: n,lr,zeff(lr)=', n, l_, zeff(lr_)
CMPIINSERT_ENDIF_RANK

c.......................................................................
c     end of strictly radial dependent variables
c.......................................................................

      endif  !l_.eq.lmdpln_

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
        bdrep(lr_)=areaovol*currtp(lr_)/(sorpwt(lr_)*3.e9)
      else
        bdrep(lr_)=0.d0
      endif

c..................................................................
c     Redefine electron density to maintain quasineutrality.
c..................................................................

      s=0.d0
      if (kelecm.eq.0 .or. kelecg.ne.0) goto 72 ! skip "redefine"
         !No Maxwellian electrons -> skip,
         !electrons are general species -> skip
         ![YuP: however, what if we have D_general+e_general,
         ! and colmodl=3, which means e_maxwell is present ?]
      if (qsineut .ne. "disabled") then
        if (qsineut.eq."maxwel") goto 71 !Use Maxw.ions only.
        !For  qsineut.eq."enabled"  use ALL ions, including general ions
        do 74 k=1,ngen
          s=s+xlndn(k,lr_)*zmaxpsii(lr_)*bnumb(k)
 74     continue
 71     do 70 k=ngen+1,ntotal !Maxw.ions only
          if (k.eq.kelecm) goto 70 !Skip e_maxw
          s=s+reden(k,lr_)*bnumb(k)
 70     continue
        reden(kelecm,lr_)=s
        locquas="disabled" ! To avoid certain steps in subr.diagdenz
      endif ! qsineut.ne."disabled"
 72   continue

c..................................................................
c     determine density as a function of poloidal angle.
c..................................................................
      call diagdenz

 99   continue ! if (l_ .ne. lmdpln_)  handle
c
c     Save copy of pitch angle integrated f in tem5
c
      call bcast(tem5,zero,jx)
      if(cqlpmod.ne."enabled")then !YuP[2021-02] added option for CQLP
         do j=1,jx
         do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
            !Note that for meshy="fixed_mu" iy_(l_) can be less than iy
            tem5(j)=tem5(j)+f(i,j,1,l_)*cynt2(i,l_)*cint2(j)*
     +           vptb(i,lr_)/zmaxpsi(lr_)
         enddo
         enddo
      else ! (cqlpmod.eq."enabled") !YuP[2021-02] added option for CQLP
         do j=1,jx
         do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
            !Note that for meshy="fixed_mu" iy_(l_) can be less than iy
            tem5(j)=tem5(j)+f(i,j,1,l_)*cynt2(i,l_)*cint2(j)
         enddo
         enddo
      endif      

      return
      end
