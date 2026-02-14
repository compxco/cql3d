c
c
      subroutine urfb0
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c
c     This routine computes the volume averaged B_0(i,j,indxlr_)
c     coefficient for the ray-tracing FW/FW/EC. 
c     indxlr_ is the index of the current flux surface.
c     This model applies to electron or ions.
c
c     YuP-110222: Now includes all lr_ internally.
c                 Modification, to facilitate MPI parallelization.
cc..................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE
            
      allocatable :: urfbwk(:) ! local working array, for MPI
                  
      complex*16 cwz,cwxyp,cwxym,cei
      cei=(0.,1.)
      
      if (.NOT. ALLOCATED(urfbwk)) then
        allocate( urfbwk(iyjx*lrz*4) )  ! used for MPI send/recv
        urfbwk=zero !call bcast(urfbwk,zero,iyjx*lrz*4)
      endif
      
c..................................................................
c     Zero out the coefficients: all flux surfaces 
c..................................................................
      urfb=zero !call bcast(urfb(1,1,1,1),zero,iyjx*lrz*mrfn)
      urfc=zero !call bcast(urfc(1,1,1,1),zero,iyjx*lrz*mrfn)
      !call bcast(urfe(1,1,1,1),zero,iyjx*lrz*mrfn)
      !call bcast(urff(1,1,1,1),zero,iyjx*lrz*mrfn)
!YuP[03/18/2015] urfe,urff are expressed through urfb,urfc
      
c..................................................................
c     Up-down symmetry factor symm= is defined in aingeom
c**bh050820:
c**bh050820:  Trapped-particle bounce-time factor, trapfac, here set =1., 
c**bh050820:  accounts for bounce time being twice the transitting bounce
c**bh091031:  time.  However, bounce-averages are divided by bounce times,
c**bh091031:  so this trapfac factor cancels out.
c**bh091031:  The symm factor though involves allocating half the ray power
c**bh091031:  above the midplane (and half mirrored below), for 
c**bh091031:  up-down-symmetric case.
c.......................................................................
      trapfac=one
      o65535= 1.d0/65535.d0
      vnorm4i= 1.d0/vnorm4
      vnorm2i= one/vnorm2

c..................................................................
c     Loop over excitation modes.
c..................................................................
      do 500 krf=1,mrfn
CMPIINSERT_MPIWORKER_KRF
CMPIINSERT_IF_RANK_EQ_MPIWORKER

c       k is general species to which this krf mode is to be applied:
         k=nrfspecies(krfn(krf))
         qmc= charge*bnumb(k)/(fmass(k)*clight) !store in common/orb0/
         qmca= abs(qmc) 

c**bh930729alf=0.5*charge**2*pi/(fmass(kelecg)*clight)**2/symm
         alf=0.5*pi*qmca**2/symm

c..................................................................
c     Set indicators of electrons or ions
c..................................................................

cBH080920      if (kionn.eq.1) then
         if (kiongg(k).eq.k) then
            signi=1.0
         else
            signi=0.0
         endif

cBH080920      if (kelecg.eq.1) then
         if (kelecg.eq.k) then
            signe=1.0
         else
            signe=0.0
         endif

c..................................................................
c     Loop over rays
c..................................................................

         do 10 iray=1,nray(krf)

c..................................................................
c     Loop of ray elements - jump out if ray element does not contribute
c     to current flux surface.
c     nrayelt0 contains lengths of ray in which significant power
c     remains (determined in urfdamp0).
c..................................................................

            do 20 is=1,nrayelt0(iray,krf)
cccYuP110222   Removing this if statement, all ray elements are treated
ccc            with one call to urfb0, rather than ouside loop over radius.
ccc            if(lr_.ne.lloc(is,iray,krf)) go to 20
c..................................................................
c     Determine the poloidal angle on the flux surface where the ray
c     element contributes. pol(ll,lr_) is the poloidal angle of 
c     interest.
c     rray is the major radius R location where it contributes and
c     zray is the Z location (distance above or below the midplane).
c..................................................................

               rray=wr(is,iray,krf)  ! 
               zray=wz(is,iray,krf)  ! 
               bray=sbtot(is,iray,krf) ! B-field at the ray element
               !-> For FOW (some values at the ray-element coord):
               !if (fow.ne.'disabled') then
                  R_loc= rray   !-> store in common/orb_loc/
                  Z_loc= zray   !-> store in common/orb_loc/
                  B_loc=bray    !-> store in common/orb_loc/
               !endif
               !->
               ll=llray(is,iray,krf)
               lrloc=lloc(is,iray,krf) !local to ray element point.
               ! jmin and jmax are only needed for ZOW (and 1st-order)
               jmin=jminray(is,iray,krf) 
               jmax=jmaxray(is,iray,krf)
               !jmin,jmax, etc., are local to ray element point.
               anpar0=wnpar(is,iray,krf) ! n_par0
               anpar02=anpar0**2
               dnpar=wdnpar(is,iray,krf) ! delta(n_par)
               anper0=wnper(is,iray,krf) ! n_prp0
               ! Determine the length of the ray element, slngth:
               slngth=0.0
               if (is.eq.1) then
                  slngth=ws(1,iray,krf)
               elseif (is.lt.nrayelt(iray,krf)) then
                  slngth=0.5*(ws(is+1,iray,krf)-ws(is-1,iray,krf))
               elseif (is.eq.nrayelt(iray,krf)) then
                  slngth=0.5*(ws(is,iray,krf)-ws(is-1,iray,krf))
               endif
               ![YuP 08-27-2015] Added skipping condition:
               if(slngth.le.0.d0) goto 20 !-> next ray element
               ! From print-out: found that sometimes slngth=0.
               ! Obtain the polarizations:
               ! Compute the (cyclotron frequency)/omega  :
               wc_w=qmca*bray/omega(krf)
               wcn_w= nharm(krf)*wc_w
               ! Define vpar_res (resonance Vpar corresponding to npar0)
               !vpar_res= clight*(1.d0-wcn_w)/anpar0
               ! Check that |vpar_res| is within u-grid 
               ! ( |vpar_res| < vnorm  ):
               if( abs(clight*(1.d0-wcn_w)).ge.abs(vnorm*anpar0)
     +             .or. jmin.gt.jx) then
                     ! Resonance is outside of u-grid.
                     ! The above condition includes the case of Npar=0.
                     jminray(is,iray,krf)=jx+1 !To indicate: out of grid
                     !WRITE(*,*)'iray,is,jmin=', iray,is,jmin
                     goto 20 !-> next ray element
               endif
               vpar_res= clight*(1.d0-wcn_w)/anpar0
               cwz=cwezde(is,iray,krf)
               cwxyp=.5*(cwexde(is,iray,krf)+cei*cweyde(is,iray,krf))
               cwxym=.5*(cwexde(is,iray,krf)-cei*cweyde(is,iray,krf))
               abssl=anper0/(wc_w*clight*bsslstp(krf))
               del_pwr=delpwr(is,iray,krf)
               scal_urf=scalurf(is,iray,krf)
               alf_rayelt=alf*urfmult*del_pwr*scal_urf
     1             *slngth*bray*clight*clight/
     1              abs(fluxn(is,iray,krf)*twopi*omega(krf)*anpar0)
               alf_rayel=alf_rayelt*(anpar02/dnpar)*abs(anpar0/clight)
c..................................................................
c     alf_rayel has in it all the leading factors in the contribution
c     to B coefficient, including the ray power and geometry, see notes.
c     For examination of possible non-quasilinear effects,
c     urfmult is a multiplier for the QL diffusion coefficients and
c     damping, defaulted to 1.0.
c..................................................................

               !--------------------------------------------------------
!               if(urfb_version.eq.2)then ! new version developed by YuP
!               ! if 1, it will continue with the original version
!                  lr_=lrloc !default value for ZOW; to be found for FOW.
!                  !!!! For now, Only setup for ICRH/Non-Relativistic,
!                  !!!! which is the case if(kiongg(k).eq.k)
!                  iurfb=0 !iurfb= 0 for calc.diff.coeffs.
!                  !Also (after 02/09/2015) accumulate prf_rayel,etc. 
!                  call urfb_add(anpar0,vpar_res,
!     +                 abssl,alf_rayelt,
!     +                 bray,wcn_w,cwz,cwxyp,cwxym,krf,
!     +                 iurfb,prf_rayel)
!                  !For linear (powrfl) and coll. (powrfc) damping -
!                  !summation is done below, after "20 continue"
!                  goto 20 !-> next ray-element
!               endif 
               !--------------------------------------------------------

c..................................................................
c     Unpack packed data - see subroutine urfpack.
c..................................................................

               ![2026-01-24] was locatn16=(jjx*(is-1)+jjx*nrayelts*(iray-1))/ibytes16+1
               !Per Grant Rutherford and John Wright, this version 
               !helps to avoid overflow when number of rays 
               !and ray elements is very large:
               !locatn16=(jjx/ibytes16)*((is-1)+nrayelts*(iray-1)) +1
               ! locatn16 specifies the storage location 
               ! in the compressed arrays.
               locatn= (jjx/ibytes16)*(is-1) +1 ![2026-01-24]
               !if(urfb_version.eq.1)then ! 2 is the new version developed by YuP
               ! if 1, it will use the original version
               !call unpack16(ilowp(locatn16,krf),8,ilim1(1:jjx),jjx)
               call unpack16(ilowp(locatn,iray,krf),8,ilim1(1:jjx),jjx) ![2026-01-24]
               call unpack16( iupp(locatn,iray,krf),8,ilim2(1:jjx),jjx) ![2026-01-24]
               !BH,YuP[2020-12-18] Changed unpack-->unpack16(which uses integer*2)
               call unpack16(ifct1_(locatn,iray,krf),8,ifct1(1:jjx),jjx) ![2026-01-24]
               call unpack16(ifct2_(locatn,iray,krf),8,ifct2(1:jjx),jjx) ![2026-01-24]
                !YuP[2020-12-18] Do not use a range in the above arrays
                !in the first arguments of unpack16 subroutines!
                !The 1st argument in subr.unpack is integer*1,
                !and the 1st arg, in unpack16 is integer*2,
                ! while ilowp,iupp,ifct1_,ifct2_ are integer*4 by default.
                !This is done by design of unpack and unpack16 subroutines.
               !endif

               do 40 j=jmin,jmax
                  i2=ilim2(j)
                  i1=min(iy,ilim1(j))
                  ulocg=vnorm*x(j)*gammi(j) ! V (speed, not momentum/mass)
                  ! Smooth out the diffusion coefficient in j to avoid
                  ! dividing by a near zero value in the evaluation of alfaj.
                  !YuP: This "smoothening" is useless in low-T 
                  !     non-relativ. plasma 
                  !     where gamma(j)=1 with good accuracy.
                  !     As a result, we can get a nearly 1/0 from 1/rrr term
                  !     at region of crossing the resonance wcn_w=1.
                  rrr= (1.d0-wcn_w*gammi(j))**2
                  div=1.d0
                  if (j.gt.1) then
                     div=div+1.d0
                     rrr=rrr+(1.d0-wcn_w*gammi(j-1))**2
                  endif
                  if (j.lt.jx) then
                     div=div+1.d0
                     rrr=rrr+(1.d0-wcn_w*gammi(j+1))**2
                  endif
                  rrr=rrr/div

                  !YuP[04-2016] Added averaging of rrr over [i2;i1] points.
                  !The value of sqrt(rrr) is |vpar_res*npar/c|.
                  !Instead of using |vpar_res|, we can use |v*cos(theta_loc)|
                  !and average it over few points in theta_local.
!                  lr_=lrloc
!                  l_=indxlr(lr_)
!                  call tdnflxs(lmdpln(l_)) !Determine itl,itu,indxlr_, etc.
!                  upar_loc_sum=0.d0 ! initialize
!                  theta_loc_sum=0.d0
!                  !Adjust the range in i, to be at least 3 pts.
!                  !(from printout: sometimes i2=i1, and it can be vpar~0)
!                  i22=i2
!                  i11=i1
!                  if(i2-i1 .le. 1) then
!                    ! One or two points only. Add two more points:
!                    if(i2.ge.2 .and. i1.le.iy-1) then
!                      i22=i2-1
!                      i11=i1+1
!                    endif
!                    if(i2.eq.1) then
!                      i22=i2
!                      i11=i1+2
!                    endif
!                    if(i1.eq.iy) then
!                      i22=i2-2
!                      i11=i1
!                    endif
!                  endif
!                  do i=i22,i11 ! loop in theta_local, for a given u-level
!                     im1=max(i-1,1)  ! i-1, but never a 0
!                     ip1=min(i+1,iy) ! i+1, but never iy+1 
!                     dtheta_loc= 0.5*(yz(ip1,ll,lr_)-yz(im1,ll,lr_))
!                     cos_loc= cosz(i,ll,lr_) ! can be 0 ! cos(theta_local)
!                     upar_loc=ulocg*cos_loc !ulocg is speed vnorm*x(j)/gamma
!                     upar_loc_sum= upar_loc_sum+abs(upar_loc)*dtheta_loc
!                     theta_loc_sum= theta_loc_sum+dtheta_loc
!                  enddo ! iloc
!                  upar_loc_avg= upar_loc_sum/theta_loc_sum
!                  !The range theta_loc_sum in theta_local
!                  !is supposed to be small.
!                  !Now define "averaged" rrr:
!                  rrr= (upar_loc_avg*anpar0/clight)**2
                  !YuP[04-2016] Added averaging of |upar_loc| and rrr: done
                  !This part over-writes the original definition of rrr.
                  !Same procedure could be added to urfdamp2 and vlf.

c.................................................................
c     Determine most of the argument of the Bessel functions, also
c     some leading coefficient factors.
c.................................................................
                  arg_bssl= abssl*x(j)*vnorm
c..................................................................
c     Add in the contribution to urfb (B_0,indxlr_,krf) (innermost loop)
c..................................................................

                  !-------------------------------------------ZOW
                     lr_=lrloc
                     l_=indxlr(lr_)
                     call tdnflxs(lmdpln(l_)) !Determine itl,itu,indxlr_, etc.
                     do i=i2,i1
                        temc2(i)=1.
                     enddo
                     !if(i1-i2.le.1)then  
                     !-> Almost identical results to the run when 
                     !-> this condition "if(i1-i2.le.1)"  is NOT imposed.
                     !-> So, these factors for i1 and i2 
                     !-> are only important when there is just
                     !-> one or two i-points within resonance strip.
                     temc2(i2)=dfloat(ifct2(j))*o65535 
                     temc2(i1)=dfloat(ifct1(j))*o65535
                     !endif
                     !if temc2(i2) & temc2(i1) are set to 0 => no RF pwr
                     uloc3=xcu(j)*vnorm3
                     uloc=x(j)*vnorm
                     tem=wcn_w/(gamma(j)*bbpsi(ll,lr_))
                     alfaj=alf_rayel*truncd(j)*gammi(j)*uloc3/rrr
                     do 50 i=i2,i1
                        thtf1i=1.
                        if(i.ge.itl .and. i.le.itu)then
                           thtf1i=0.5/trapfac
                        endif
                        indx=sinz(i,ll,lr_)*arg_bssl+1
                        ! Make certain indx is within table bounds
                        indx=min(indx,nbssltbl)
                        !!! if (indx.gt.nbssltbl) call urfwrong(1)
                        sinn0=sinn(i,l_)
                        coss0=coss(i,l_)
                        delb=alfaj*thtf1i/dpsi(lr_)
     1                     *abs(coss0)
     1                     *abs( 
     1                          cosz(i,ll,lr_)*cwz*jb0(indx,krf)
     1                         +sinz(i,ll,lr_)*
     1                         (
     1                          cwxyp*
     1                     (signe*jbp1(indx,krf)+signi*jbm1(indx,krf))
     1                         +cwxym*
     1                     (signe*jbm1(indx,krf)+signi*jbp1(indx,krf))
     1                           )
     1                           )**2
     1                      *temc2(i)
                        urfb(i,j,indxlr_,krf)=
     1                        urfb(i,j,indxlr_,krf)+delb
                        fact_e=(tem-sinn0**2)/(x(j)*coss0)
                       !urfe(i,j,indxlr_,krf)=urfe(i,j,indxlr_,krf)+fact_e*delb
                       !YuP[03/18/2015] urfe,urff are expressed through urfb,urfc
                       ! No need to save them.
                        if(sinn0 .ne. zero) then
                          fact_c= fact_e/sinn0
                          fact_f= fact_c*fact_e
                          urfc(i,j,indxlr_,krf)=
     1                           urfc(i,j,indxlr_,krf)+fact_c*delb
                          !urff(i,j,indxlr_,krf)=urff(i,j,indxlr_,krf)+fact_f*delb
                        endif
		       
!                        if(krf.eq.3 .and. iray.eq.4 .and. j.eq.60 .and. 
!     +                     i.ge.70 .and. i.le.80 ) then
!                           write(*,'(4i5,2e12.3,2i5,2e12.3)') 
!     +                       iray,is,i,lr_,rray,zray,i2,i1,rrr,rrr1
!                           pause
!                        endif 

 50                  continue  ! i=i2,i1
                  !-------------------------------------------ZOW
 40            continue         ! j=jmin,jmax
 20         continue            ! is=1,nrayelt0(iray,krf)
 
 10     continue ! iray=1,nray(krf)

CMPIINSERT_ENDIF_RANK
CMPIINSERT_RECV_URFB0
CMPIINSERT_SEND_URFB0

 500  continue                  !End loop on krf=1:mrfn
 
CMPIINSERT_BARRIER 
CMPIINSERT_BCAST_URFB0


      ! Renormalize urf* coeffs. 
      do krf=1,mrfn
         k=  nrfspecies(krfn(krf))
         fm= fmass(k)*symm/(vnorm*twopi)
      do ll=1,lrz
         call tdnflxs(lmdpln(ll)) !-> l_ and lr_
         !lr0=ll
         !lp1=min(ll+1,lrfow)
         !lm1=max(ll-1,1)
         !dpsi_lr= dpsi(lr_) ! Original version
         !dpsi_lr= rg(lr_)*drg(lr_)*bthr0(lr_) !=R0*dR0*Bp0 (given lr_)
         !rdrbpb0=zmaxpsii(lr_)*dvol(lr_)/symm
         !rdrbpb0= twopi*dpsi_lr/bmod0(lr_) !=2pi*R0*dR0*(Bp0/B0)=dvol/(symm*zmaxpsi)
                  ! 2pi*R0*dR0*(Bp0/B0)=dvol/(symm*zmaxpsi)
         !zfact=zmaxpsii(lr_)*dvol(lr_)*fmass(k)*vnorm2i ! HYBRID only
         !zfact= rdrbpb0*symm*fmass(k)*vnorm2i
         sum_test=0. ! for print-out of sum(i,j) of urfb
         do j=1,jx
            jp1=min(j+1,jx) ! Not exceeding jx
            jm1=max(j-1,1)  ! Not smaller than 1
            !xdx= x(j)*gammi(j)*dx(j)*1.d-7
            !............................................................
               !Symmetrize about pi/2 in trapped region.
               do 57 i=itl,itu
                  temp1(i,j)=urfb(i,j,indxlr_,krf)
                  temp2(i,j)=urfc(i,j,indxlr_,krf)
                  !temp3(i,j)=urfe(i,j,indxlr_,krf)
                  !temp4(i,j)=urff(i,j,indxlr_,krf)
 57            continue
               do 70 i=itl,itu
                  ! YuP-102011 Factor 0.5 moved into thtf1i above
                  ii=iy+1-i
                  urfb(i,j,indxlr_,krf)=(temp1(i,j)+temp1(ii,j))
                  urfc(i,j,indxlr_,krf)=(temp2(i,j)-temp2(ii,j))
                  !urfe(i,j,indxlr_,krf)=(temp3(i,j)-temp3(ii,j))
                  !urff(i,j,indxlr_,krf)=(temp4(i,j)+temp4(ii,j))
                  !YuP[03/18/2015] urfe,urff are expressed through urfb,urfc
 70            continue
            !............................................................
            do 55  i=1,iy  
               !Skip loss cone: 
               !if(gone(i,j,k,l_).lt.zero) goto 55 !-> next i
               ip1=min(i+1,iy)
               im1=max(i-1,1)
               delbw=  urfb(i,j,indxlr_,krf)
               delcc=  urfc(i,j,indxlr_,krf)
               dele0=delcc*sinn(i,l_) !E=C*sin()
               if(delbw.ne.0.d0)then
                  delf0=(delcc*dele0)/delbw ! this is urff  ! BF-CE=0
               else
                  delf0=0.d0
               endif
               !Renormalize for code (Note: Thru x, factors of vnorm and vnorm2
               !have been introduced in above expressions for urf[c,e,f]).
               urfb(i,j,indxlr_,krf)=  delbw*vnorm4i !vnorm4i=1/vnorm4
               urfc(i,j,indxlr_,krf)=  delcc*vnorm4i
               !urfe(i,j,indxlr_,krf)=dele0*vnorm4i
               !urff(i,j,indxlr_,krf)=delf0*vnorm4i
               !YuP[03/18/2015] urfe,urff are expressed through urfb,urfc
               !No need to save into (large) arrays.
               !..................................................................
               !sum_test= sum_test +urfb(i,j,indxlr_,krf)*sum_test_ren
 55         continue    ! i
         enddo          ! j
CMPIINSERT_IF_RANK_EQ_0   
c      WRITE(*,'(a,2i5,e12.3)')'sum_test for urfb0:', ll,krf,sum_test
CMPIINSERT_ENDIF_RANK
      enddo             ! ll=1,lrz
      enddo             ! krf=1,mrfn
      
CMPIINSERT_IF_RANK_EQ_0   
      WRITE(*,*)
     + 'urfb0/AFTER renorm: sum(urfb)=', n,sum(urfb)
CMPIINSERT_ENDIF_RANK

       
CMG added 11/13/2017
      if(allocated(urfbwk)) deallocate(urfbwk)
CMG end added 11/13/2017

      return
      end



c=======================================================================
c=======================================================================

      subroutine urfb_add(anpar0,vpar_res,
     +           abssl,alf_rayelt,
     +           bray,wcn_w,cwz,cwxyp,cwxym,krf,
     +           iurfb,prf_rayel)

      
      implicit integer (i-n), real*8 (a-h,o-z)
c YuP-2011
c..................................................................
c     Add in the contribution to urfb,urfc  QL coeffs.
c     Called once for each ray/ray_element.
c..................................................................
c     Only valid for ICRH case, for now.
c     Designed for FOW-hybrid but here used for ZOW, as in mimic_zow.
c..................................................................
      ! called with iurfb=0 for calc. of diff.coeffs. (called by urfb0)
      !             iurfb=1 for calc. of prf_rayel (called by urfdamp1)
      ! called for each krf=1:mrfn (all wave types and harmonics)
c..................................................................
      include 'param.h'
      include 'comm.h'
                              
      complex*16 cwz,cwxyp,cwxym,cei
      real*8 sum_dth(iy,lrz)
      real*8 urfb_i(iy,lrz),urfc_i(iy,lrz),prf_rayel_i(iy,lrz)

      cei=(0.,1.)
      vnorm2i= one/vnorm2
      
      k=nrfspecies(krfn(krf))
      if (kiongg(k).ne.k) then
         stop 'urfb_add: Only setup for non-relativistic ions, ICRH'
         ! only ready for ICRH; kiongg(k)=k
      endif
      
      ! Note: qmc=charge*bnumb(k)/(fmass(k)*clight)  is in common/orb0/
      
      if (kiongg(k).eq.k) then
         signi=1.0
      else
         signi=0.0
      endif

      if (kelecg.eq.k) then
         signe=1.0
      else
         signe=0.0
      endif
      
c**bh050820:  Trapped-particle bounce-time factor, trapfac, here set =1., 
c**bh050820:  accounts for bounce time being twice the transitting bounce
c**bh091031:  time.  However, bounce-averages are divided by bounce times,
c**bh091031:  so this trapfac factor cancels out.
c.......................................................................
      !YuP[03/26/2015] symm is set in aingeom now
      trapfac=one
      umx=vnorm  ! u-upper-limit for res.region ! 
      anpar_c= anpar0/clight ! n||/c = k||/omega

      !psi_loc=   psi_rz(R_loc,Z_loc)
      !bphib_loc= bphib_rz(R_loc,Z_loc) ! Bphi/B local
      !bbRloc=    bphib_loc*R_loc  ! (Bphi/B)R local

      sum_dth=0.d0 ! Initialize, for all iy,lrz
      if(iurfb.eq.0) then
         urfb_i=0.d0 ! All (iy,lrz)
         urfc_i=0.d0 ! All (iy,lrz)
      else 
         prf_rayel_i=0.d0 ! All (iy,lrz)
      endif

      !YuP[04-2016] Definition of dveps is moved to urfsetup/comm.h

      ! Resonance boundaries, in local upar (ICRH case):
      ! Define lines in res.region (local u-space)  
      ! corresponding to different vpar 
      ! in resonance region.
      ! Each line will be populated with "particles"/points.
      vparlo= vpar_res-dveps*0.95 !Note: dveps is defined to be positive.
      vparlo= max(-umx,vparlo)        ! Not smaller than -umx;
      vparup= vpar_res+dveps*0.95
      vparup= min(+umx,vparup)        ! Not exceeding umx;
      if(vparlo*vparup.le.0.d0)then !point upar=0 is between boundaries
         ! lower limit of u within res.region 
         umn= x(2)*vnorm   !(and avoiding x(1)=0 point)
         jmn= 2
      else ! res.region is shifted from upar=0 point
         umn= min(abs(vparlo),abs(vparup))
         umn_norm= umn/vnorm
         jmn= luf_bin(umn_norm,x(1:jx),jx) ! x(jmn-1) <= umn_norm <  x(jmn)
         jmn= max(2,jmn)  ! to be sure jmn>1
         jmn= min(jmn,jx) ! to be sure jmn is not exceeding jx
      endif
      
      !if(x(jmn)*vnorm.lt.umn)then ! should never happen
      !write(*,*) 'jmn,x(jmn)*vnorm,umn=',jmn,x(jmn)*vnorm,umn
      !pause
      !endif
      
      !if(x(jmn)*vnorm.gt.max(abs(vparlo),abs(vparup)))then ! should never happen
      !write(*,*)'jmn,x(jmn)vnorm,vlo,up=',jmn,x(jmn)*vnorm,vparlo,vparup
      !pause
      !endif
      
      !Redefine the height of the hat function.
      !Normally, it should be (vparup-vparlo +0.1*dveps)/2 = dveps
      !But if vpar_res~vnorm, the base is smaller => the height should be larger
      !dveps_adj= 0.5*( abs(vparup-vparlo) +0.1*dveps ) ! not quite correct
      dveps_adj=dveps
      o_dveps= 1.d0/dveps_adj

      ! Define the smallest uprp_local for the resonance region:
      thet_eps= 0.1*pi/(iy-1) ! step away from uprp=0 to avoid sin()=0

      prf_rayel=0.d0 !sum-up contribution from a given ray el., per unit delpwr
    
      do j=jmn,jx ! Levels of u in res.region, local u-space
         jp1=min(j+1,jx) !Not exceeding jx (j+/-1 will be used for df/dv)
         jm1=max(j-1,1)  !Not smaller than 1
         uloc=   x(j)*vnorm
         uloc2=  uloc*uloc
         ulocn=  x(j)
         u0= uloc  ! the midplane value: assuming conservation of u
         u02=u0*u0 ! 
         xdx= x(j)*gammi(j)*dx(j)
         wcn_wg= wcn_w*gammi(j) ! omega_c*n/(omega*gamma)
         !Define min/max of theta_local for a given uloc in res.region:
         cos_thet_mn=min(+1.d0, vparup/uloc) !when uloc<vparup, thet_mn=0
         cos_thet_mx=max(-1.d0, vparlo/uloc) !when vparlo<(-uloc), thet_mx=pi
         !Note: vparup>vparlo, always
         theta_loc_mn= acos(cos_thet_mn)
         theta_loc_mn= max(theta_loc_mn,thet_eps) !stay away from theta=0
         theta_loc_mx= acos(cos_thet_mx)
         theta_loc_mx= min(theta_loc_mx,pi-thet_eps) !stay away from pi

         if(theta_loc_mx-theta_loc_mn .gt. 0.d0) then ! Usual case
            !Define how many points in local theta, within res.region:
            iyloc=2*CEILING(iy*(theta_loc_mx-theta_loc_mn)/pi) 
            !Note: CEILING(arg)=smallest integer greater than or equal to arg
            iyloc=max(iyloc,5) ! Use at least 5 points in theta_loc
            !TEST: If doubled- almost same result (0.5% diff); t_CPU~doubles;
            ! plots of urfb0 look cleaner (less noise, less isolated points)
            !TEST: If halved (max(iyloc,2)) - urfb0 is getting a reduction
            ! at uprp/vnorm>0.7, smaller tail in f, smaller rf power (by 2%).
            dtheta_loc= (theta_loc_mx-theta_loc_mn)/(iyloc-1)
         else !(anomaly or rounding error)-> skip
            goto 30 !-> next j
         endif
         
         ! The ranges for i,lr_ where the ray element contributed to urfb
         ! will be found in do loop below.
         ! The ranges i_mn:i_mx,lrmn:lrmx will be used to reset 
         ! arrays(iy,lrz) (instead of the whole ranges, to save cpu time)
         i_mn=iy
         i_mx=1
         lrmn=lrz
         lrmx=1
      
         !YuP[04-2016] Added averaging of |upar_loc|.
         !As can be seen from expression for delb below,
         !it is inversely prop. to |upar_loc|, 
         !and so it can diverge at upar_loc~0.
         !In fact this upar_loc comes from the guiding center velocity
         !Vgcpol= upar_loc*(Bp/B)_loc 
         !and enters the expression for <Bql> from consideration
         !of dt~dw/Vgcpol that it takes a particle to traverse 
         !a ray element.
         !To avoid the divergence, we average |upar_loc| 
         !over the hat-function,
         !so <|upar_loc|>_hat is never zero.
         upar_loc_sum=0.d0 ! initialize
         hatfun_sum=0.d0
         do iloc=1,iyloc ! loop in theta_local, for a given u-level
            ! This is an equivalent of (delta-function);
            ! Wider base (large dveps) => smaller magnitude.
            ! Note: o_dveps= (1/dveps)
            theta_loc= theta_loc_mn +dtheta_loc*(iloc-1) !from min to max
            cos_loc=   cos(theta_loc) ! can be 0
            ! upar_loc with the sign, to define the hat-funcion:
            upar_loc=  uloc*cos_loc 
            if(upar_loc>vparup .or. upar_loc<vparlo)then 
             !write(*,*)'upar_loc,vparlo,vparup',upar_loc,vparlo,vparup
             !-> do nothing, next iloc
            else
             !hatfun= o_dveps*max(one-abs(upar_loc-vpar_res)/dveps, zero)
             !The dvpar integral over hat function is 1.0.
             !Not a big difference from averaging method:
             !with hat-function, or simply with factor 1.0.
             hatfun= 1.0 !
             upar_loc_sum= upar_loc_sum +abs(upar_loc)*hatfun*dtheta_loc
             hatfun_sum=   hatfun_sum  + hatfun*dtheta_loc
            endif
         enddo ! iloc
         upar_loc_avg= upar_loc_sum/hatfun_sum
         !YuP[04-2016] Added averaging of |upar_loc|: done
         
         do iloc=1,iyloc ! loop in theta_local, for a given u-level
            ! This is an equivalent of (delta-function);
            ! Wider base (large dveps) => smaller magnitude.
            ! Note: o_dveps= (1/dveps)
            theta_loc= theta_loc_mn +dtheta_loc*(iloc-1) !from min to max
            !theta_loc= theta_loc_mx -dtheta_loc*(iloc-1) !from max to min
            sin_loc=   sin(theta_loc) ! cannot be 0
            sin_loc2=  sin_loc*sin_loc
            cos_loc=   cos(theta_loc) ! can be 0
            !YuP[04-2016] This check for cos_loc~0 
            !is not needed anymore because now  
            !we use upar_loc_avg that cannot be 0.
            !if(abs(cos_loc).lt. 1.d-6)then ! 
            !   cos_loc= sign(1.d-6,cos_loc)
            !   sin_loc2=(1.d0-cos_loc)*(1.d0+cos_loc)
            !   sin_loc= sqrt(sin_loc2)
            !endif
            upar_loc=  uloc*cos_loc   
            !  upar_loc_avg=upar_loc
            if(upar_loc>vparup .or. upar_loc<vparlo)then 
              !write(*,*)'upar_loc,vparlo,vparup',upar_loc,vparlo,vparup
              goto 20 !-> next iloc
            endif
            uprp_loc=  uloc*sin_loc
            uprp_loc2= uprp_loc**2
            hatfun= o_dveps*max(one-abs(upar_loc-vpar_res)/dveps, zero)
            ! For the given (upar_loc; uprp_loc) find
            ! corresponding (i0)-index 
            ! for the (u0,theta0)-point on the midplane.
            ! First, find lr_, the surface with psi=<psi> 
            ! for a given (upar_loc,uprp_loc)
            !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ZOWtest
            !if(mimic_zow.eq.'enabled')then
            !   goto 10 !mimic ZOW; lr_=lrloc is defined in urfb0 !!
            !endif
            !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ZOWtest
c   10       continue !Handle to skip FOW for lr_, and use ZOW (for test)
            l_=indxlr(lr_) 
            !Determine itl,itu,indxlr_,etc. Depends on flux surf!
            call tdnflxs(lmdpln(l_))  !Determine itl,itu, indxlr_, etc.
            b0db=bmod0(lr_)/bray !B_midplane(lr_surf)/B_local(at ray element)
            bb0=1.d0/b0db
            sqbb0=sqrt(bb0)   ! sqrt[B_local/B_midpl]
            sqb0db=1.d0/sqbb0 ! sqrt[B_midpl/B_local]
            ! Note: in Hybrid-FOW mapping, it may happen that
            ! R0<Rlocal, and therefore B0/Blocal >1;
            ! for example, going along the outer leg of fat banana orbit,
            ! we have Rlocal>R0, where R0 is the position of <psi>_BA
            ! on the midplane (the  "center" of banana).
            ! Furthermore, for a "potato" like banana, it may happen
            ! that at one of such points with Rlocal>R0, we get into
            ! a particular point with Upar=0 (theta_loc=pi/2).
            ! It is clear that we cannot use conservation of mu
            ! to map such local point into the center of banana orbit.
            ! There are two possible ways to deal with such problem:
            ! 1. Skip such point => so there will be no contribution.
            ! It will look like a loss cone, but actually it means
            ! that the Hybrid-type distribution function has no 
            ! representation at such (u,theta)local (such "gaps"
            ! usually occur at near-magn-axis area, 
            ! for large-energy particles).
            ! 2. Map such points into the midplane (u0,theta0=pi/2) point.
            ! But in this case lambda=|Upar*tau_b|~0 anyway.
            ! Find index i corresponding to pitch angle theta0
            ! at the midplane of surface 'lr_'.
            uprp0= uprp_loc*sqb0db ! uperp0 = uperp_loc*sqrt(B0/B)
            ! See note above: (B/B0) can be <1, so if uprp_loc2~u02, 
            ! we may have uprp0 > u0. 
            if(uprp0.ge.u0)then
               goto 20 ! uncomment to skip this point (no contribution)
               ! Or could map it into theta0~pi/2 point on the midplane
               !uprp02= u02-em12 ! but contribution is still ~0
            endif
            uprp02 = uprp0*uprp0
            upar02= (u0-uprp0)*(u0+uprp0)  ! ==upar0^2
            upar0= sqrt(upar02) ! == abs(upar0)
            sinn0= uprp0/u0
            sinn0= min(sinn0,one-em6) !make sure sin(theta0)<1.
            coss0= upar0/u0 ! |cos(theta0)|
            theta0=asin(sinn0) ! <pi/2
            i=luf_bin(theta0,ymid(1:iyh,l_),iyh) ! nearest i in theta0-grid
            ! Note: from the above line, i<iyh, or at most i=iyh
            
            !YuP[09-16-2015] Indicator for 
            !banana tip points, i.e. points near upar_loc=0.
            !The intersection of given ray element
            !with most of orbits (passing or trapped/away-from-tip)
            !includes one resonant point (or no resonance at all).
            !However, a trapped orbit that crosses a ray element
            !at near-tip part of orbit may include two resonance events:
            !upar_loc=+/-|upar_eps| (small pos. and neg. values of upar).
            !Such crossing corresponds to TWO "kicks" obtained from given
            !ray element. 
            !(Of course, the resonant condition at such ray element 
            !should be close to vpar_res~0).
            wtip=1.d0 !Initialize:   weight factor.
            !For banana tip points, apply wtip=2. 
            if(vparlo*vparup.le.0.d0)then 
               !Resonant condition at ray el. is close to vpar_res~0.
               !The region "covered" by the hat-function _/\_
               !with base [vparlo;vparup]
               !may include iloc-points that correspond to banana tips.
               if( abs(vparlo).le.abs(vparup) )then
                  !The region that contains banana tip pts is set as
                  ! -|vparlo| ... +|vparlo|
                  if( abs(upar_loc).le.abs(vparlo) )then
                    wtip=2.d0
                  endif
               else
                  !The region that contains banana tip pts is set as
                  ! -|vparup| ... +|vparup|
                  if( abs(upar_loc).le.abs(vparup) )then
                    wtip=2.d0
                  endif
               endif
            endif

            if(i.lt.itl)then ! passing: 
              !map it to cos(theta0) with same sign as local upar_loc
              if (upar_loc.lt.zero) then
               ! passing ptcl with upar_loc<0. 
               ! Reverse the sign of upar0 to match the sign of upar_loc:
               upar0=-upar0
               coss0=-coss0
               theta0=pi-theta0
               i= iy+1-i ! now i>iyh
              endif
            else ! a trapped orbit 
                if (upar_loc.lt.zero) then
                  ! Reverse the sign of upar0 to match upar_loc:
                  upar0=-upar0
                  coss0=-coss0
                  theta0=pi-theta0
                  i= iy+1-i ! now i>iyh
                endif
                ! The tip points are accounted automatically,
                ! one with positive upar0 (for upar_loc>0)
                ! and another with negative upar0 (for upar_loc<0)
            endif
            ! Done: (i,j,lr_) are found.
            !Check: if in the loss cone, skip this point (un-comment next line):
            !!!if(gone(i,j,k,l_).lt.zero) goto 20 !-> next i
                        
            ! Determine the range of i,lr indices, 
            ! for faster resetting of arrays after the work is done:
            i_mn=min(i_mn,i)
            i_mx=max(i_mx,i)
            lrmn=min(lrmn,lr_)
            lrmx=max(lrmx,lr_)
            ! Try: redefine these values to match the grid points:
            !(TESTS: nearly same results)
            !coss0= coss(i,l_) ! cos(theta0(i))
            !sinn0= sinn(i,l_) ! can be =0 !
            !upar0= u0*coss0 
            !uprp0= u0*sinn0
            ! Transformation coefficients:
            !if(mimic_zow.eq.'enabled')then
            dR0dv=   0.d0
            dR0dth=  0.d0
            dth0dv=  0.d0 
            dth0dth= sqb0db*cos_loc/coss0 ! dtheta0/dtheta ZOW or HYBRID-FOW
            !dth0dth=min(dth0dth, 1.d0) ! not to exceed +1.0
            !dth0dth=max(dth0dth,-1.d0) ! not smaller than -1.0
            ! In ZOW limit, dth0dth -> sqb0db*coscos0, all other =0.
c            if(dth0dth.lt.0.d0)then
c               WRITE(*,*)'dth0/dth<0',dth0dth, sqb0db,cos_loc,coss0
c               pause
c            endif
            !endif
            
            ! Determine index for the argument of the Bessel function:
            indx= uprp_loc*abssl+1
            ! Make certain indx is within table bounds:
            indx=min(indx,nbssltbl)
            !!! if (indx.gt.nbssltbl) call urfwrong(1)
            thtf1i=1.
            thtf2i=1.
            if(i.ge.itl .and. i.le.itu)then
               thtf1i=0.5/trapfac
               thtf2i=2.
            endif
                        
            ! omega_c*n/(omega*gamma) + kpar*upar/(omega*gamma) :
            wcnk= wcn_wg + anpar_c*upar_loc*gammi(j)
            ! 'wcnk' is supposed to be 1.0 because of resonant 
            ! condition, but with hat-function instead of 
            ! actual delta-function, wcnk is not exactly 1.0
            ! but typically within 0.99-1.01 range.
            !wcnk=1.d0 ! Reset it to exact 1.0
       
               dpsi_lr= dpsi(lr_) ! Original version
               !dpsi_lr= rg(lr_)*drg(lr_)*bthr0(lr_) !=R0*dR0*Bp0 (given lr_)
               !TESTS: dpsi_lr=R0*dR0*Bp0 yields ~1% smaller Prf to ions, 
               !       and 1% more to electrons.
               
               del_th0= abs(dtheta_loc*dth0dth) ! dtheta_loc mapped onto midplane
               !del_th0= min(del_th0,dy(i,l_)) ! not to exceed dtheta0 on the midplane
               ! Note del_th0 factor in delb: 
               !after iloc-loop, the accumulated urfb_i,...are norm. by 1/sum_dth
               delb= alf_rayelt*thtf1i*truncd(j)*hatfun*clight*del_th0
     1           *abs(upar0/(dpsi_lr*upar_loc_avg))*gamma(j)
     1           *abs(   upar_loc*cwz*jb0(indx,krf)
     1                  +uprp_loc*
     1         (  cwxyp*(signe*jbp1(indx,krf)+signi*jbm1(indx,krf))
     1           +cwxym*(signe*jbm1(indx,krf)+signi*jbp1(indx,krf))  )
     1                )**2
     
               fact_c= 
     =          (wcn_wg*cos_loc-anpar_c*uloc*gammi(j)*sin_loc*sin_loc)
     /           /(ulocn*sin_loc)
               delbw= delb*wcnk*wcnk   
               delcc= delb*fact_c*wcnk*dth0dth
               !dele0= sinn0*delcc
               !delf0= sinn0*delb*fact_c*fact_c*dth0dth*dth0dth
               
               if(iurfb.eq.0) then ! call from urfb0
                 !-> Add contribution from the given ray element:
                 !Note that during this call
                 !(iurfb=0) alf_rayel DOES include delpwr and scalurf.
                 ! Accumulate QL diff.coeffs 
                 urfb_i(i,l_)= urfb_i(i,l_)+delbw
                 urfc_i(i,l_)= urfc_i(i,l_)+delcc
                 !Accumulate contributions in (i,l_) midplane grid cells:
                 sum_dth(i,l_)=sum_dth(i,l_)+del_th0 !for 1/sum_dth averaging
                 ! No need to save <E> and <F> - 
                 ! they are expressed through <B>,<C> 
c                Trying to weigh contribution onto two nearest i, ilo & iup.
c                Result: almost same as just attributing to one nearest i.               
c                urfb_i(i,l_)=urfb_i(i,l_)+delbw*wlo
c                urfb_i(i,l_)=urfb_i(i,l_)+delbw*wup
c                urfc_i(i,l_)=urfc_i(i,l_)+delcc*wlo
c                urfc_i(i,l_)=urfc_i(i,l_)+delcc*wup
               else !(iurfb=1) !This part is for call from urfdamp1, 
                    !for getting urfpwr(is,iray,krf)=prf_rayel
                 !Note that during this call, 
                 !(iurfb=1) alf_rayel does NOT include delpwr and scalurf.
                 ip1=min(i+1,iy)
                 im1=max(i-1,1)
                 !Check: if at least one of points across differencing 
                 !is in the loss cone, skip this point:
                 if(gone(i,jp1,k,l_)+gone(i,jm1,k,l_).lt.zero) goto 20 !-> next i
                 if(gone(ip1,j,k,l_)+gone(im1,j,k,l_).lt.zero) goto 20 !-> next i
                 ggfij=
     1             +delbw*0.5*dxi(j)*(f(i,jp1,k,l_)-f(i,jm1,k,l_))
     1             +delcc*0.5*dyi(i,l_)*(f(ip1,j,k,l_)-f(im1,j,k,l_))
                 rdrbpb0=zmaxpsii(lr_)*dvol(lr_)/symm
                 !rdrbpb0= twopi*dpsi_lr/bmod0(lr_) !=2pi*R0*dR0*(Bp0/B0)=dvol/(symm*zmaxpsi)
                 zfact= rdrbpb0*symm*fmass(k)*vnorm2i
                  ! 2pi*R0*dR0*(Bp0/B0)=dvol/(symm*zmaxpsi)
                 !sum over i,j  for total power: 
cbug             sum_ij= -4.*thtf1i*ggfij*cynt2(i,l_)*zfact !before 02/03/2015
                 sum_ij= -thtf2i*ggfij*cynt2(i,l_)*zfact !Hybrid-FOW, after 02/03/2015
                 prf_rayel_i(i,lr_)= prf_rayel_i(i,lr_)+ xdx*sum_ij
                 ! This is to be saved into urfpwr(), the fractional power
                 ! (per delpwr) (contribution from a given ray element only)
                 !Accumulate contributions in (i,l_) midplane grid cells:
                 sum_dth(i,l_)=sum_dth(i,l_)+del_th0 !for 1/sum_dth averaging
               endif
                           
  20     continue  ! skip handle -> next iloc
         enddo ! iloc
         
         ! Add-in contributions accumulated from all iloc (for a given j)
         if(iurfb.eq.0) then
            ! Accumulate QL diff.coeffs 
            ! (accumulation from all rays and ray-elements):
            do l_=lrmn,lrmx
            do i= i_mn,i_mx
               if(sum_dth(i,l_).ne.0.d0)then
               owk_i=1.d0/sum_dth(i,l_)
               urfb(i,j,l_,krf)= urfb(i,j,l_,krf) +urfb_i(i,l_)*owk_i
               urfc(i,j,l_,krf)= urfc(i,j,l_,krf) +urfc_i(i,l_)*owk_i
               !-> urfb(), urfc() is the output (stored in comm.h)
               urfb_i(i,l_)=0.  ! reset
               urfc_i(i,l_)=0.  ! reset
               sum_dth(i,l_)=0. ! reset
               endif
            enddo
            enddo
         else
            do l_=lrmn,lrmx
            do i= i_mn,i_mx
               if(sum_dth(i,l_).ne.0.d0)then
               prf_rayel= prf_rayel +prf_rayel_i(i,l_)/sum_dth(i,l_)
               prf_rayel_i(i,l_)=0. ! reset
               sum_dth(i,l_)=0.     ! reset
               endif
            enddo
            enddo
         endif
 
 30   continue ! skip all iloc handle        
      enddo  ! loop in j

      return
      end



c=======================================================================
c=======================================================================
c=======================================================================
c=======================================================================


      integer function luf_bin(px,parray,kn)
      implicit integer (i-n), real*8 (a-h,o-z)
c YuP-2011
c     LUF_BIN() is similar to LUF(), but with a BINARY SEARCH (faster).
c     luf_bin(px,table,kn)  is a function returning the index
c        of the first element in the parray that is greater than px.
c
c     Elements in parray must be strictly increasing !!!
c
      dimension parray(kn)
c
c     YuP added: check that parray(i) is increasing with i
c      do i=2,kn
c      !write(*,*) i,parray(i)-parray(i-1)
c        if(parray(i)-parray(i-1) .lt. 0.d-15) then
c          !write(*,*) 'Function LUF_bin: parray(i)=',  parray(1:kn)
c          write(*,*) 'Func. LUF_bin: array should be increasing.',i
c          goto 5
c          !STOP
c        endif
c      enddo

      if(px.lt.parray(1)) then
         luf_bin=1
         return
      endif

      if(px.ge.parray(kn)) then 
         luf_bin=kn   ! Or should we set to kn+1 ?
         return
      endif

c     find first index such that parray(luf_bin).gt.px

      k1=1
      k3=kn

5     continue
      k2= (k3-k1)/2 + k1   ! middle index between k1 and k3

      if (px.ge.parray(k1) .and. px.lt.parray(k2)) then
         ! point px is within this interval;
         if (k2.eq.k1+1) then 
             ! point px is between two adjacent indices
             luf_bin=k2
             goto 10 ! finish
         endif
         k3=k2 ! new k3  !  k1=k1 remains same
         goto 5 !-> divide this interval, and check again
      elseif(px.ge.parray(k2) .and. px.lt.parray(k3)) then
         ! point px is within this interval;
         if (k3.eq.k2+1) then 
             ! point px is between two adjacent indices
             luf_bin=k3
             goto 10 ! finish
         endif
         k1=k2 ! new k1  !  k3=k3 remains same
         goto 5 !-> divide this interval, and check again
      else  ! Neither of two intervals => no index found. 
         luf_bin=kn  ! Or should we set to kn+1 ?
         goto 10 ! finish
      endif

 10   continue
      return
      end function

