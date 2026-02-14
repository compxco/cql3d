c
c
      subroutine urfdamp2(krf)
      implicit integer (i-n), real*8 (a-h,o-z)
cyup      save

c..................................................................
c     This routine computes the damping for all
c     ray elements on all rays for UNIT power delpwr. This is put into
c     array urfpwr(is,iray,krf). The algorithm follows that of urfb0.
c
c     Power absorption for each ray element is obtained by integration
c     over the resonance region of velocity space of energy*(df/dt)_QL.
c     Local values of the distribution function are obtained from
c     midplane values.  The algorithm follows that of urfb0.
c.......................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      complex*16 cwz,cwxyp,cwxym,cei
ccc      include 'advnce.h'  ! YuP: No need?

      integer*1 ipwr(nrayelts) ! YuP: For MPI:  indicator  (0 or 1) 
                               ! that a ray element contributed any power
      
      dimension gftp(0:jx) !, fppj(0:iy+1,0:jx+1) ! local working arrays

      fppj(i,j)=g_(i,j+1,k,l_)  +
     1         (g_(i,j,k,l_)-g_(i,j+1,k,l_))*dj(i,j,k,l_)

      data nray0 /1/
 
      cei=(0.,1.)
      
c..................................................................
c     fill out increasing psi array of bin boundaries, as in urffflx.
c..................................................................
         do l=1,lrzmax
            tr2(l)=psimag-psivalm(l)
         enddo

c     k is general species which this krf mode damps on:
      k=nrfspecies(krfn(krf))
      qmc= charge*bnumb(k)/(fmass(k)*clight) !store in common/orb0/
      qmca= abs(qmc) 
      o65535= 1.d0/65535.d0
      vnorm4i= 1.d0/vnorm4
c..................................................................
c     Express the Chang-Cooper weighted average g_(i,j+1/2,l_)
c..................................................................
c      call bcast(fppj,zero,iyjx2)
c      do i=1,iy
c      do j=1,jx
c         fppj(i,j)=g_(i,j+1,k,l_)  +
c     1            (g_(i,j,k,l_)-g_(i,j+1,k,l_))*dj(i,j,k,l_)
         !YuP: above is same as below, but only one '*':
c        fppj(i,j)=g_(i,j+1,k,l_)*(1.-dj(i,j,k,l_)) +
c     1            g_(i,j,k,l_)*dj(i,j,k,l_)
c      enddo
c      enddo
      !YuP-101228: Setting fppj as an array rather than a function
      !reduces cpu time - fppj does not need to be calc. along each ray
      
c.......................................................................
c     Up-down symmetry factor symm= is defined in aingeom
c**bh050820:
c**bh050820:  Trapped-particle bounce-time factor, trapfac=1., 
c**bh050820:  accounting for bounce time twice transitting bounce time.
c**bh091031:  Bounce-averages are divided by bounce times, so this
c**bh091031:  trapfrac factor does not enter into the problem.
c**bh091031:  symm factor though involves allocating half the ray power
c**bh091031:  above the midplane (and half mirrored below), for 
c**bh091031:  up-down-symmetric case.
c..................................................................

      trapfac=one

      alf=0.5*pi*qmca**2/symm

c.......................................................................
c     Set indicators of electrons or ions
c.......................................................................

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

      do 10 iray=nray0,nray(krf)
        ipwr=0 ! For MPI: initialize
CMPIINSERT_MPIWORKER_IRAYKRF
CMPIINSERT_IF_RANK_EQ_MPIWORKER

c..................................................................
c     Loop of ray elements - jump out if ray element does not contribute
c     to current flux surface.
c..................................................................

        do 20 is=1,nrayelt(iray,krf)

ccc          if(lr_.ne.lloc(is,iray,krf)) go to 20
          lr_=lloc(is,iray,krf)
          l_=indxlr(lr_)
          call tdnflxs(lmdpln(l_)) !-> Determine itl,itu, etc.

          jmin=jminray(is,iray,krf)
          if(jmin.gt.jx)  go to 20

          jmax=jmaxray(is,iray,krf)
          jmax0=min0(jmax,jx-1)
          jmaxp=min0(jmax+1,jx)
          jminm=max0(jmin-1,1)

          ![2026-01-24] was locatn16=(jjx*(is-1)+jjx*nrayelts*(iray-1))/ibytes16+1
          !Per Grant Rutherford and John Wright, this version 
          !helps to avoid overflow when number of rays 
          !and ray elements is very large:
          !locatn16=(jjx/ibytes16)*((is-1)+nrayelts*(iray-1)) +1
          locatn= (jjx/ibytes16)*(is-1) +1 ![2026-01-24]
          call bcast(db(1,0),zero,iyjxp1)
          call bcast(dc(1,0),zero,iyjxp1)
          call ibcast(ilim1d,0,jx)
          call ibcast(ilim2d,0,jx)

c..................................................................
c     Unpack stored data - see subroutine urfpack
c..................................................................
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
          
c..................................................................
c     rr is the major radius R location of the ray element.
c     ll if the index of the array pol (poloidal angle array).
c..................................................................
          ! rr=wr(is,iray,krf) ! YuP: Not used?
          ll=llray(is,iray,krf)

c..................................................................
c     Determine the length of the ray element, slngth
c..................................................................
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

c..................................................................
c     Obtain the polarizations..
c..................................................................
          cwz=cwezde(is,iray,krf)
          cwxyp=0.5*(cwexde(is,iray,krf)+cei*cweyde(is,iray,krf))
          cwxym=0.5*(cwexde(is,iray,krf)-cei*cweyde(is,iray,krf))

c..................................................................
c     Compute the cyclotron frequency.
c..................................................................
c**bh930729wce=charge*sbtot(is,iray,krf)/fmass(kelecg)/clight
          wce=qmca*sbtot(is,iray,krf)
          wcen_omega= nharm(krf)*wce/omega(krf)
c..................................................................
c     alfa(i) will have in it all the leading factors in the contribution
c     to the B coefficient, including ray power and geometry, see notes.
c     Here we have set what was delpwr(is,iray,krf) in subroutine urfb0
c     to 1, in order to normalize the power absorption. Later in
c     urfdamp0 we will trace the ray and compute a new 
c     delpwr(is,iray,krf) and this will be used to scale what 
c     is computed HERE.
c..................................................................

          alfa_work= urfmult*wnpar(is,iray,krf)**2
     1      *slngth*sbtot(is,iray,krf)/(dpsi(lr_)
     1      *fluxn(is,iray,krf)*twopi*wdnpar(is,iray,krf)*omega(krf))
     +      *clight

c..................................................................
c     Determine most of the argument of the Bessel functions, also
c     some leading coefficient factors.
c..................................................................

          do 40 j=jmin,jmax
            i2=ilim2(j)
            i1=min0(iy,ilim1(j))
            rrr= (1.d0-wcen_omega*gammi(j))**2
            div=1.
            if (j.gt.1) then
              div=div+1.
              rrr=rrr+(1.d0-wcen_omega*gammi(j-1))**2
            endif
            if (j.lt.jx) then
              div=div+1.
              rrr=rrr+(1.d0-wcen_omega*gammi(j+1))**2
            endif
            rrr=rrr/div

            alfaj=alfa_work*truncd(j)*gammi(j)*xcu(j)*vnorm3/rrr
            arg_bssl=wnper(is,iray,krf)*x(j)*
     *               vnorm*omega(krf)/(wce*clight*bsslstp(krf))
c..................................................................
c     Add in the contribution to urfb (B_0,indxlr_)  (innermost loop)
c..................................................................
            do i=i2,i1
              temc2(i)=1.
            enddo
            temc2(i2)=dfloat(ifct2(j))*o65535
            temc2(i1)=dfloat(ifct1(j))*o65535
            tem=wcen_omega/(gamma(j)*bbpsi(ll,lr_))
            do 50 i=i2,i1
              thtf1i=1.
              if (i.ge.itl .and. i.le.itu) then
                thtf1i=.5
              endif
              indx=sinz(i,ll,lr_)*arg_bssl+1
              db(i,j)= alfaj*alf*thtf1i*abs(coss(i,l_))
     1             *abs( cosz(i,ll,lr_)*cwz*jb0(indx,krf)
     1                  +sinz(i,ll,lr_)*
     +       ( cwxyp*(signe*jbp1(indx,krf)+signi*jbm1(indx,krf))
     1        +cwxym*(signe*jbm1(indx,krf)+signi*jbp1(indx,krf)) )
     1                   )**2  
     1              * temc2(i)
              sinn_=sinn(i,l_)
              if (sinn_.ne.zero) then
              dc(i,j)=db(i,j)*(tem-sinn_**2)/(x(j)*coss(i,l_)*sinn_)
              endif
 50         continue
 
 40       continue ! j
 
c..................................................................
c     Need a slightly larger integration range...
c     This is to accomodate the interpolation that is done in
c     the local damping calculation, below.
c..................................................................

c     Account for extension of j-values by -1.
          do 4 j=jminm,jmax0
            ilim2d(j)=min0(ilim2(j),ilim2(j+1))
            ilim1d(j)=max0(ilim1(j),ilim1(j+1))
 4        continue
c     Extend  i-bounds by 1
          do 5 j=jminm,jmax
            ilim2d(j)=max0(ilim2d(j)-1,1)
            ilim1d(j)=min0(ilim1d(j)+1,iy)
 5        continue
          ilim2d(jmaxp)=ilim2d(jmax)
          ilim1d(jmaxp)=ilim1d(jmax)
c     Account for extension of j-values by +1.
          do 6 j=jmin,jmaxp
            ilim2dd(j)=min0(ilim2d(j),ilim2d(j-1))
            ilim1dd(j)=max0(ilim1d(j),ilim1d(j-1))
 6        continue
          ilim2dd(jminm)=ilim2d(jminm)
          ilim1dd(jminm)=ilim1d(jminm)
          do 7 j=jminm,jmaxp
            ilim2dd(j)=max0(ilim2dd(j),1)
            ilim1dd(j)=min0(ilim1dd(j),iy)
 7        continue

          call bcast(temp1(0,0),zero,iyjx2)
          call urfmidv_db(jmin,jmax)
ccc          call bcast(temp1(0,0),zero,iyjx2) ! Not really necessary
          call urfmidv_dc(jmin,jmax)
          
          do j=jminm-1,jmaxp
         !Setting gftp as an array rather than function reduces cpu time
         ! advnce.h:  cl(i,j)=.25*vptb(itl,lr_)/vptb(i,lr_)*dc(i,j)
             cl_itlm1= .25*vptb(itl,lr_)/vptb(itl-1,lr_)*dc(itl-1,j)
             cl_itlp1= .25*vptb(itl,lr_)/vptb(itl+1,lr_)*dc(itl+1,j)
             cl_itup1= .25*vptb(itl,lr_)/vptb(itu+1,lr_)*dc(itu+1,j)
             gftp(j)=
     1           +cl_itlm1*eyp5(itl-1,l_)*(fppj(itl,j)-fppj(itl-1,j))
     1        +2.*cl_itlp1*eyp5(itl,l_) * (fppj(itl+1,j)-fppj(itl,j))
     1           +cl_itup1*eyp5(itu,l_) * (fppj(itu+1,j)-fppj(itu,j))
             ! Note: cl(i,j) is proportional to dc(i,j)
          enddo
     
c..................................................................
c     Express the velocity flux at (i,j+1/2)
c..................................................................
c
          prf_rayel=0.d0 !sum-up contribution from a given ray element. 
          do 80 j=jminm,jmaxp
          sum_i=0.d0
          do 90 i=ilim2dd(j),ilim1dd(j)
              dgfp= db(i,j)*exp5(j)*(g_(i,j+1,k,l_)-g_(i,j,k,l_))
     +           -db(i,j-1)*exp5(j-1)*(g_(i,j,k,l_)-g_(i,j-1,k,l_))
              if(i .ne. itl .and. i .ne. itu) then
                dgfp= dgfp 
     +             +dc(i,j)*0.5*dyi(i,l_)*(fppj(i+1,j)-fppj(i-1,j))
     +           -dc(i,j-1)*0.5*dyi(i,l_)*(fppj(i+1,j-1)-fppj(i-1,j-1))
              else
                dgfp= dgfp + (gftp(j)-gftp(j-1))
              endif
              thtf2i=1.
              if (i.ge.itl .and. i.le.itu) then
                 thtf2i=2.
              endif
              zfact=.5*zmaxpsii(lr_)*dvol(lr_)*fmass(k)/vnorm2
              sum_i=sum_i + thtf2i*dgfp*tcsgm1(j)*cynt2(i,l_)*zfact
 90       continue
          prf_rayel= prf_rayel +sum_i
 80       continue

          urfpwr(is,iray,krf)=prf_rayel

          if(prf_rayel.ne.0.d0) ipwr(is)=1 ! For MPI
                    
c         See scaleurf in cqlinput_help for explanation of following 
c         procedure.
          urfpt=urfpwr(is,iray,krf)+urfpwrc(is,iray,krf)
     +          +urfpwrl(is,iray,krf)
          if (urfpt.gt.one .and. scaleurf.eq."enabled")  then
            scalurf(is,iray,krf)=1.d0/urfpt
            urfp1=urfpwr(is,iray,krf)
            urfp2=urfpwrc(is,iray,krf)
            urfp3=urfpwrl(is,iray,krf)
            urfpwr(is,iray,krf)=urfp1/urfpt
            urfpwrc(is,iray,krf)=urfp2/urfpt
            urfpwrl(is,iray,krf)=urfp3/urfpt
          endif

c.............................................................
c     Overwrite salphac if iurfcoll.eq."damp_out".  This is added
c      for code diagnostic purposes and gives o/p to the 
c      netcdf rf file.
c.............................................................
          if (iurfcoll(krfn(krf)).eq."damp_out") then
             if (slngth.eq.zero) then
                salphac(is,iray,krf)=zero
             else
                salphac(is,iray,krf)=urfpwr(is,iray,krf)/slngth
             endif
          endif
          
 20     continue ! do is=1,nrayelt(iray,krf)

CMPIINSERT_SEND_URFPWR
CMPIINSERT_ENDIF_RANK
CMPIINSERT_RECV_URFPWR
 10   continue ! do iray=nray0,nray(krf)


      return
      end
