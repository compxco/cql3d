      subroutine urfpack
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c.......................................................................
c     This routine determines arrays which contain data
c     distilled from the ray tracing routines. Each time the subroutine
c     urfb0 is called it uses and reuses the information computed
c     and stored in this routine.
c.......................................................................

      include 'param.h'
      include 'comm.h'
      data nray0 /1/

c     Temporary printout, checking ray data is on radial mesh.
c      krf=1
c      do iray=nray0,nray(krf)
c      write(*,*)'urfpack:lloc(*,iray,1):',
c     1        (lloc(is,iray,1),is=1,nrayelt(iray,krf))
c      enddo

c.......................................................................
c     Uses library routines:
c     luf(x,table,n) (MATHLIB) which is a function returning the index
c        of the first element in the table that is greater than x.
c        Elements must be strictly increasing. x.gt.table(n)==>n+1.
c     lug(x,table,n,iguess) (MATHLIB) same as luf, 
c        but with guess index iguess.
c     pack16/unpack16, as below.
c.......................................................................


      wx=.5
c**bh930729bcnst=charge/(fmass(kelecg)*clight)

c..................................................................
c     Loop over modes (or harmonics) and flux surfaces..
c..................................................................

      do 500 krf=1,mrfn
        if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
        write(*,*)'urfpack: krf(1:mrfn) = ', krf
        endif

c       k is general species to which this krf mode is to be applied:
        k=nrfspecies(krfn(krf))
        bcnst=abs(bnumb(k))*charge/(fmass(k)*clight)

        do 9 l=1,lrz
          call tdnflxs(lmdpln(l))

c..................................................................
c     Loop over rays
c..................................................................

          icount_outside_ez=0  ! only for printout
          icount_outside_er=0  ! only for printout
          
          do 10 iray=nray0,nray(krf)

c.......................................................................
c     Loop of ray elements - jump out if ray element does not contribute
c     to current flux surface or if data has been stored previously.
c.......................................................................

            do 20 is=1,nrayelt(iray,krf)

              if(lr_.ne.lloc(is,iray,krf)) go to 20

c..................................................................
c     lrayelt is 0 during the first call to urfpack.  If rays are
c     extended, lrayelt will contain the previous values of nrayelt.
c..................................................................

              if (lrayelt(iray,krf).ge.is) go to 20

c.......................................................................
c     The arrays ilim1 and ilim2 defined in this routine will be stored
c     in packed form in array ilowp and iupp. The number of bits
c     utilized from each number in ilim1 or ilim2 will be the rightmost
c     16 bits of each word. The routine pack16 accomplishes
c     compression. The variable "locatn16" pinpoints the address
c     in the packed array where the first of the jx contributions
c     from a given ray element will be stored.
c     jjx is the first multiple of 8 greater than jx .
c     Similary, ifct1 and ifct2 are stored in packed form in ifct1_
c     and ifct2_, in 16 bit (i.e., 2 byte) form. locatn16 specifies
c     the storage location in the compressed arrays.
c
c     From man pack on Cray C90 and J90 (after one or more "q"):
c      NAME
c           PACK - Compresses stored data
c      SYNOPSIS
c           CALL PACK(p, nbits, u, nw)
c      IMPLEMENTATION
c           Cray PVP systems
c      DESCRIPTION
c           PACK compresses stored data.  The following is a list 
c           of valid arguments for this routine.
c           p         On exit, vector of packed data.
c           nbits     Number of rightmost bits of data in each 
c                     partial word; must be 1, 2, 4, 8, 16, or 32.
c           u         Vector of partial words to be compressed.
c           nw        Number of partial words to be compressed.
c
c           PACK takes the 1, 2, 4, 8, 16, or 32 rightmost bits 
c           of several partial words and concatenates them 
c           into full 64-bit words.
c     The code has been modified to also treat the 32-bit interger 
c        (machinea=2) case.  (BobH, 990308).
c     subroutines pack/unpack are for 8-bit words, pack16/unpack16
c     are for 16-bit words.
c.......................................................................

            ![2026-01-24] was locatn16=(jjx*(is-1)+jjx*nrayelts*(iray-1))/ibytes16 +1
            !Per Grant Rutherford and John Wright, this version 
            !helps to avoid overflow when number of rays 
            !and ray elements is very large:
            !locatn16=(jjx/ibytes16)*((is-1)+nrayelts*(iray-1)) +1
            locatn= (jjx/ibytes16)*(is-1) +1 ![2026-01-24]

c.......................................................................
c     Determine the poloidal angle on the flux surface where the ray
c     element contributes.
c     rray is the major radius R location where it contributes and
c     zray is the Z location (distance above or below the midplane).
c..................................................................

            !------------------------- 
            !YuP[04-2016]: adjust Zray,Rray if got outside of equil grid
            zray=wz(is,iray,krf)
            if(zray.gt.ez(nnz))then 
              !For a mirror machine: ray can get outside of zbox
              !which defines the border of ez() equilibrium grid
              ![so that zbox=ez(nnz)-ez(1)]
              if (icount_outside_ez.eq.0) then
                if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
                write(*,*)'urfpack: Ray elements outside of ez grid'
                endif
              endif
              icount_outside_ez=icount_outside_ez+1 !for a printout
              if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
              write(*,*)'urfpack: zray>ez; iray,is,icount_outside_ez',
     +                                     iray,is,icount_outside_ez
              endif
              ! Make an adjustment:
              zray=ez(nnz)
              !This correction is ok for a tokamak, too,
              !although not likely to happen.
            endif
            if(zray.lt.ez(1))then 
              if (icount_outside_ez.eq.0) then
                if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
                write(*,*)'urfpack: Ray elements outside of ez grid'
                endif
              endif
              icount_outside_ez=icount_outside_ez+1 !for a printout
              if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
              write(*,*)'urfpack: zray<ez; iray,is,icount_outside_ez',
     +                                     iray,is,icount_outside_ez
              endif
              ! Similarly, Make an adjustment:
              zray=ez(1)
            endif
            !-------------------------
            rray=wr(is,iray,krf)
            if(rray.gt.er(nnr))then 
              !For a mirror machine: ray can get outside of 
              !er() equilibrium grid
              ![so that zbox=ez(nnz)-ez(1)]
              ! Make an adjustment:
              rray=er(nnr)
              !This correction is ok for a tokamak, too,
              !although not likely to happen.
            endif
            if(rray.lt.er(1))then ! this cannot happen, but ok to add.
              ! Similarly, Make an adjustment:
              rray=er(1)
            endif
            !-------------------------
    
c..................................................................
c     rmag is the location of the magnetic axis
c..................................................................

cBH091012:              rnew=rray-rmag
cBH091012:  For consistency with other thetapol defns
cBH091012:  (and non-updown symmetry). 
c$$$             thetapol=atan(abs(zray)/abs(rnew))
c$$$              if (rnew.lt.0.) then
c$$$                thetapol=pi-thetapol
c$$$              endif
            if (eqsym.ne."none") then  !up-down symmetrize: abs(zz-zmag)
               thetapol=atan2(abs(zray-zmag),rray-rmag) ! in [0,pi]
               !but can be only [0; +pi/2) range in mirror machine
            else  !i.e., no up-down symmetrization
               thetapol=atan2(zray-zmag,rray-rmag)
               if (thetapol.lt.zero) thetapol=thetapol+twopi !in [0,pi2)
            endif

c..................................................................
c     Determine the relevant index "ll" on the computational poloidal
c     mesh - this will differ slightly from the exact value thetapol.
c..................................................................

              ll=luf(thetapol,pol(1,lr_),lz)
              if (ll.gt.lz) then
c      write(*,*)
c     +'urfpack: thetapol,lr_,lz,pol(lz,lr_),pol(lz,lr_-1)=',
c     +          thetapol,lr_,lz,pol(lz,lr_),pol(lz,lr_-1)
                 if (abs(thetapol-pol(lz,lr_)).lt.em12*thetapol) then
                    ! just a small over-shooting
                    ll=lz-1 ! YuP: or maybe ll=lz ?
                    thetapol= pol(lz,lr_) !YuP: adjusted
                 else
                    if(machine.eq."mirror")then
                      ! Field lines can only go from 0 to pi/2,
                      ! or (normally) less. Ray element can be between 
                      ! this given lr surface and the lower lr-1.
                      ! Try lr-1
                      if(lr_.eq.1)then !ray is inside 1st surface, near Z=Zmax
                         ll=lz
                         thetapol= pol(lz,lr_) !adjusted
                      else !(lr_.ge.2) then
                         !ll=luf(thetapol,pol(1,lr_-1),lz) ! try the lower lr
                         ! Or do same as for lr_=1 :
                         ll=lz
                         thetapol= pol(lz,lr_) !adjusted
                      endif
                      if(ll.gt.lz) then ! still could not find: stop the run
                         WRITE(*,*)'urfpack/urfwrong: lr_,ll,Rray,Zray',
     +                       lr_,ll, rray,zray, 
     +                       solrz(lz,lr_),  solzz(lz,lr_), 
     +                       solrz(lz,lr_-1),solzz(lz,lr_-1)
                         call urfwrong(3)
                      endif
                    else ! tokamak 
                      call urfwrong(3)
                    endif
                 endif
              endif ! ll>lz problem
              
              if (ll.gt.1) then
                if (pol(ll,lr_)-thetapol .gt. thetapol-pol(ll-1,lr_))
     +            ll=ll-1
              endif
              llray(is,iray,krf)=ll

              signn=1.
              if (wnpar(is,iray,krf).lt.0) signn=-1.

              omomn=1.0
              signom=1.0

c..................................................................
cBH080920              if (nharm(krf).eq.0 .or. kiong(k).eq.1)  then
              if (nharm(krf).eq.0 .or. kiongg(k).eq.k)  then
c..................................................................

c..................................................................
c     Lower Hybrid/Fast Wave, or non-relativistic ions
c..................................................................


c..................................................................
c     Determine the local minimum and maximum parallel speed for
c     which we assume a resonance.
c     For nharm(krf)=0, if vparl has opposite sign from vparu, damping
c     will be be omitted.  Generally, this occurs when npar goes
c     through zero and hence the wave resonance is beyond the grid.
c     Sign reversal on vl,vu (and cosmz) is to accomodate luf.
c     Velocity integration limits are worked out using the
c     abs(wnpar).  Limits for negative wnpar can be obtained from
c     the abs(wnpar)-results by reflection about theta=pi/2.
c
c     The ion damping case is treated non-relativistically,
c     including the cyclotron damping.  The NR resonances are similar
c     to the electron case, in that the resonant velocity is
c     independent of perpendicular velocity.
c..................................................................

cBH080920                if (kiong(k).eq.1) then
                if (kiongg(k).eq.k) then
                  omn=nharm(krf)*bcnst*sbtot(is,iray,krf)/omega(krf)
                  omomn=1.0-omn
c990131                  signom=sign(1.,omomn)
                  signom=sign(one,omomn)
                endif

                vparu=clight*signom*omomn/
     1            (signn*wnpar(is,iray,krf)-wdnpar(is,iray,krf)*wx)
                vparl=clight*signom*omomn/
     1            (signn*wnpar(is,iray,krf)+wdnpar(is,iray,krf)*wx)

                if(vparl*vparu .gt. 0.)  then
                  vl=-abs(vparl/vnorm)
                  vu=-abs(vparu/vnorm)
                else
                  vl=-1.1
                  vu=-1.2
                endif

c..................................................................
c     Determine the lowest index in the velocity mesh that experiences
c     a resonance. sx is the normalized speed mesh, whereas x in the
c     momentum mesh.
c     jmin will be .gt.1.  If abs(vl) is off the mesh, jmin=jx+1
c..................................................................

                jmin=luf(-vl,sx,jx)
                jminray(is,iray,krf)=jmin
                jmaxray(is,iray,krf)=jx
                if (jmin.gt.jx) go to 20
                if (jmin.eq.1) call urfwrong(4)
                do 25 j=1,jmin-1
                  ilim2(j)=0
                  ilim1(j)=0
 25             continue

c..................................................................
c     Determine the lowest(highest) theta index for which there is a
c     resonance at speed mesh sx(j)
c..................................................................

                ilim2(jmin)=luf(vu/sx(jmin),cosmz(1,ll,lr_),iyh)
                ilim1(jmin)=luf(vl/sx(jmin),cosmz(1,ll,lr_),iyh)
                
                !YuP[04-2016] adjusted the range in theta around resonance
                !to have at least 3 points
!                i2=ilim2(jmin)
!                i1=ilim1(jmin)  ! i2<=i1
!                if( i2-i1 .eq. 0) then
!                   ! One point only i2=i1. Add two more points:
!                   if(i2.ge.2 .and. i1.le.iy-1) then
!                     i2=i2-1
!                     i1=i1+1
!                   endif
!                   if(i2.eq.1) then
!                     i2=i2
!                     i1=i1+2
!                   endif
!                   if(i1.eq.iy) then
!                     i2=i2-2
!                     i1=i1
!                   endif
!                endif
!                ilim2(jmin)=i2
!                ilim1(jmin)=i1
                !YuP[04-2016] done: adjusted
                call urfedge(ilim1(jmin),ilim2(jmin),vl/sx(jmin),
     +            vu/sx(jmin),ll,lr_,jmin)
     
                ilim2(jx)=luf(vu/sx(jx),cosmz(1,ll,lr_),iyh)
                ilim1(jx)=luf(vl/sx(jx),cosmz(1,ll,lr_),iyh)
                !YuP[04-2016] adjusted the range in theta around resonance
                !to have at least 3 points
!                i2=ilim2(jx)
!                i1=ilim1(jx)  ! i2<=i1
!                if( i2-i1 .eq. 0) then
!                   ! One point only i2=i1. Add two more points:
!                   if(i2.ge.2 .and. i1.le.iy-1) then
!                     i2=i2-1
!                     i1=i1+1
!                   endif
!                   if(i2.eq.1) then
!                     i2=i2
!                     i1=i1+2
!                   endif
!                   if(i1.eq.iy) then
!                     i2=i2-2
!                     i1=i1
!                   endif
!                endif
!                ilim2(jx)=i2
!                ilim1(jx)=i1
                !YuP[04-2016] done: adjusted
                call urfedge(ilim1(jx),ilim2(jx),vl/sx(jx),vu/sx(jx),
     1            ll,lr_,jx)
     
                iupjx=ilim2(jx)
                ilwjx=ilim1(jx)
                do 30 j=jmin+1,jx-1
                  ilim2(j)=luf(vu/sx(j),cosmz(ilim2(j-1),ll,lr_),iupjx-
     1              ilim2(j-1))-1+ilim2(j-1)
                  ilim1(j)=luf(vl/sx(j),cosmz(ilim1(j-1),ll,lr_),ilwjx-
     1              ilim1(j-1))-1+ilim1(j-1)
                    !YuP[04-2016] adjusted the range in theta around resonance
                    !to have at least 3 points
!                    i2=ilim2(j)
!                    i1=ilim1(j)  ! i2<=i1
!                    if( i2-i1 .eq. 0) then
!                       ! One point only i2=i1. Add two more points:
!                       if(i2.ge.2 .and. i1.le.iy-1) then
!                         i2=i2-1
!                         i1=i1+1
!                       endif
!                       if(i2.eq.1) then
!                         i2=i2
!                         i1=i1+2
!                       endif
!                       if(i1.eq.iy) then
!                         i2=i2-2
!                         i1=i1
!                       endif
!                    endif
!                    ilim2(j)=i2
!                    ilim1(j)=i1
                    !YuP[04-2016] done: adjusted
                  call urfedge(ilim1(j),ilim2(j),vl/sx(j),vu/sx(j),
     1              ll,lr_,j)
 30             continue

c..................................................................
              elseif (nharm(krf).gt.0)  then
c..................................................................

c..................................................................
c     Relativistic cyclotron damping case.  rnpar1 is the lower n_parallel
c     and rnpar2 is the upper one.  The associated resonance velocities
c     (not momentum-per-mass) are vpar1 and vpar2.  ilim1(j) and ilim2(j)
c     are the lower and upper limits of the pitch angle index.
c
c     Also, if rnpar1 and rnpar2 have opposite signs, damping is again
c       omitted.  This can give inaccuracies for very close to 
c       perpendicular propagation.  It can be mitigated by choosing
c       small wdnpar.
c..................................................................

                rnpar1=signn*wnpar(is,iray,krf)-wx*wdnpar(is,iray,krf)
                rnpar2=signn*wnpar(is,iray,krf)+wx*wdnpar(is,iray,krf)
                r1mn12=1.-rnpar1*rnpar1
                r1mn22=1.-rnpar2*rnpar2

                omn=nharm(krf)*bcnst*sbtot(is,iray,krf)/omega(krf)
                omn2=omn*omn
                rad1=omn2-r1mn12
                rad2=omn2-r1mn22
c
                if (rnpar1*rnpar2.le.0.) then
                  jminray(is,iray,krf)=jx+1
                  go to 20
                endif

c..................................................................
c     Ray outside of pinch point. No reson. (omega(krf) will be 
c     .gt. omega_ce). Indicate skipping of the ray element by taking
c     jminray(is,iray,krf)=jx+1.
c..................................................................

                if (rad1.le.0.0.and.rad2.le.0.0)  then

c..................................................................
c     set flag t skip this ray element
c..................................................................

                  jminray(is,iray,krf)=jx+1
                  go to 20
                endif

c..................................................................
c     The following case should not be possible:
c..................................................................

                if (rad1.gt.0.0.and.rad2.lt.0.0)  call urfwrong(6)

c..................................................................
c     Note: We let uu01 and uu02 become negative for npar**2 > 1.
c           This helps the logic go through for this case, 
c           with minimum changes from the npar**2<1 case.
c..................................................................

                if (rad2.gt.0.0.and.rad1.le.0.0)  then
                  upar01=0.0
                  uu01=0.0
                else
                  upar01=clight*omn*rnpar1/(r1mn12*vnorm)
                  uu01=clight*sqrt(rad1)/(r1mn12*vnorm)
                endif
                upar02=clight*omn*rnpar2/(r1mn22*vnorm)
                uu02=clight*sqrt(rad2)/(r1mn22*vnorm)

c..................................................................
c     upar11 and upar12 are parallel momentum-per-mass limits of the
c     resonance ellipse associated with rnpar1..., upar21 and upar22,
c     associated with rnpar2.
c..................................................................

                upar11=upar01-uu01
                upar12=upar01+uu01
                upar21=upar02-uu02
                upar22=upar02+uu02

c               Set rnpar1, rnpar2 .gt.1.0, for npar**2>1 case:
                if (rnpar1.gt.1.) upar12=1.1
                if (rnpar2.gt.1.) upar22=1.1

                jmin=luf(abs(upar21),x,jx)
                jminray(is,iray,krf)=jmin
                jmax=luf(upar22,x,jx)-1
                jmaxray(is,iray,krf)=jmax
c***  if(jmin.gt.jx.or.jmin.eq.jmax)  go to 20
c***  Try a less restricted condition, compatible with urfb0,...
                if(jmin.gt.jx)  go to 20

                if(jmin.eq.1)  call urfwrong(4)

                do 41  j=1,jmin-1
                  ilim1(j)=0
                  ilim2(j)=0
 41             continue
                do 42  j=jmax+1,jx
                  ilim1(j)=0
                  ilim2(j)=0
 42             continue

c..................................................................
c     Treat cases of (nharm(krf)*omega_ce/omega(krf)).le.1, 
c     and .gt.1, separately
c..................................................................

                if (omn.le.1.)  then
c*bh*940227if(x(jmin).le.upar11)then
                  if(x(jmin).le.upar11 .or. rad1.le.0.0)  then
                    vres1=clight/vnorm ! c*bh*940227
                    ilim2(jmin)=1
                  else
                    vres1=clight*(1.-omn/gamma(jmin))/(rnpar1*vnorm)
                    ilim2(jmin)=luf(-vres1/sx(jmin),cosmz(1,ll,lr_),iyh)
                  endif
                  vres2=clight*(1.-omn/gamma(jmin))/(rnpar2*vnorm)
                  ilim1(jmin)=luf(-vres2/sx(jmin),cosmz(1,ll,lr_),iyh)
                  ii1=ilim1(jmin)
                  call urfedge(ii1,ilim2(jmin),-vres2/sx(jmin),
     +              -vres1/sx(jmin),ll,lr_,jmin)
                  do 45  j=jmin+1,jmax
                    if(x(j).le.upar11)  then
                      ilim2(j)=1
                      vres2=clight*(1.-omn/gamma(j))/(rnpar2*vnorm)
                      ilim1(j)=luf(-vres2/sx(j),cosmz(1,ll,lr_),iyh)
                    elseif (x(j).lt.upar12)  then
                      vres1=clight*(1.-omn/gamma(j))/(rnpar1*vnorm)
                      vres2=clight*(1.-omn/gamma(j))/(rnpar2*vnorm)
                      ilim2(j)=luf(-vres1/sx(j),cosmz(1,ll,lr_),iyh)
                      ilim1(j)=luf(-vres2/sx(j),cosmz(1,ll,lr_),iyh)
                    else
                      ilim2(j)=1
                      vres2=clight*(1.-omn/gamma(j))/(rnpar2*vnorm)
                      ilim1(j)=luf(-vres2/sx(j),cosmz(1,ll,lr_),iyh)
                    endif
                    ii1=ilim1(j)
                    call urfedge(ii1,ilim2(j),-vres2/sx(j),-vres1/sx(j),
     1                ll,lr_,j)
 45               continue


                else

c...............................................................
c     nharm(krf)*omega_ce/omega(krf) .gt.1 - case:
c...............................................................

                  uperpstr=sqrt(omn*omn-1.)*clight/vnorm
                  if(x(jmin).le.abs(upar11))  then
cbh940227
                    vres1=clight*(1.-omn/gamma(jmin))/(rnpar1*vnorm)
                    ilim1(jmin)=iy
                  else
                    vres1=clight*(1.-omn/gamma(jmin))/(rnpar1*vnorm)
                    ilim1(jmin)=luf(-vres1/sx(jmin),cosmz(1,ll,lr_),iy)
                  endif
                  vres2=clight*(1.-omn/gamma(jmin))/(rnpar2*vnorm)
                  ilim2(jmin)=luf(-vres2/sx(jmin),cosmz(1,ll,lr_),iy)
                  ii1=ilim1(jmin)
                  call urfedge(ii1,ilim2(jmin),-vres1/sx(jmin),
     +              -vres2/sx(jmin),ll,lr_,jmin)
                  do 55  j=jmin+1,jmax
                    vres1=clight*(1.-omn/gamma(j))/(rnpar1*vnorm)
                    vres2=clight*(1.-omn/gamma(j))/(rnpar2*vnorm)
                    if(x(j).le.abs(upar11))  then
                      ilim1(j)=iy
                      ilim2(j)=luf(-vres2/sx(j),cosmz(1,ll,lr_),iy)
                      ii1=ilim1(j)
                      call urfedge(ii1,ilim2(j),-vres1/sx(j),
     +                  -vres2/sx(j),ll,lr_,j)
                    elseif (x(j).lt.uperpstr)  then
                      ilim2(j)=luf(-vres2/sx(j),cosmz(1,ll,lr_),iy)
                      ilim1(j)=luf(-vres1/sx(j),cosmz(1,ll,lr_),iy)
                      ii1=ilim1(j)
                      call urfedge(ii1,ilim2(j),-vres1/sx(j),
     +                  -vres2/sx(j),ll,lr_,j)
                    elseif(x(j).lt.upar12)  then
                      ilim2(j)=luf(-vres1/sx(j),cosmz(1,ll,lr_),iy)
                      ilim1(j)=luf(-vres2/sx(j),cosmz(1,ll,lr_),iy)
                      ii1=ilim1(j)
                      call urfedge(ii1,ilim2(j),-vres2/sx(j),
     +                  -vres1/sx(j),ll,lr_,j)
                    elseif (x(j).lt.upar22)  then
                      ilim2(j)=1
                      ilim1(j)=luf(-vres2/sx(j),cosmz(1,ll,lr_),iy)
                      ii1=ilim1(j)
                      call urfedge(ii1,ilim2(j),-vres2/sx(j),
     +                  -vres1/sx(j),ll,lr_,j)
                    endif
                    

 55               continue

                endif

c..................................................................
c     End of nharm(krf) if construction:
c..................................................................

              endif

c..................................................................
c     Check for errors
c..................................................................

              do 70 j=1,jx
                if(ilim1(j).gt.iy) call urfwrong(7)
                if(ilim2(j).gt.iy) call urfwrong(7)
 70           continue

c..................................................................
c     Reflect about pi/2 for negative k_par
c..................................................................

              if (signn*signom.lt.0.) then
                do 80 j=1,jx
                  iota(j)=ilim2(j)
 80             continue
                do 90 j=1,jx
                  ilim2(j)=iy+1-ilim1(j)
                  ilim1(j)=iy+1-iota(j)
 90             continue
                do 81 j=1,jx
                  iota(j)=ifct1(j)
 81             continue
                do 91 j=1,jx
                  ifct1(j)=ifct2(j)
                  ifct2(j)=iota(j)
 91             continue
              endif

c..................................................................
c     Pack results in 2 byte chunks (ilowp and iupp) and
c     2 byte chunks (ifct1_ and ifct2_) to save space.
c..................................................................
              
              !if(urfb_version.eq.1)then ! 2 is the new version developed by YuP
                ! if 1, it will use the original version
                !call pack16(ilowp(locatn16,krf),8,ilim1,jjx)
                call pack16(ilowp(locatn,iray,krf),8,ilim1,jjx) ![2026-01-24]
                call pack16( iupp(locatn,iray,krf),8,ilim2,jjx) ![2026-01-24]
                !BH,YuP[2020-12-18] Changed pack-->pack16(which uses integer*2)
                call pack16(ifct1_(locatn,iray,krf),8,ifct1,jjx) ![2026-01-24]
                call pack16(ifct2_(locatn,iray,krf),8,ifct2,jjx) ![2026-01-24]
              !endif
              
c     Temporary printout, checking elements fall on vel grid:
c              write(*,*)'urfpack: is,iray,krf,jminray,jmaxray',
c     1             is,iray,krf,jminray(is,iray,krf),jmaxray(is,iray,krf)


 20         continue ! is=1,nrayelt(iray,krf)
 10       continue ! iray=nray0,nray(krf)

c.......................................................................
c     end of loop over flux surfaces
c.......................................................................
 9      continue !  l=1,lrz

        do 50 iray=1,nray(krf)
          lrayelt(iray,krf)=nrayelt(iray,krf)
 50     continue

 500  continue     !end of loop over modes krf=1,mrfn

     
c$$$      do is=1,nrayelt(1,1)
c$$$         jmin=jminray(is,1,1)
c$$$         jmax=jmaxray(is,1,1)
c$$$         write(*,1000) is,lloc(is,1,1),jmin,jmax
c$$$      enddo
c$$$ 1000 format(10i5)
c$$$      write(*,*)'urfpack: STOP'
c$$$      STOP


      !if(urfb_version.eq.1)then ! 2 is the new version developed by YuP
         ! if 1, it will use the original version
      if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
          write(*,*)'URFpacked ifct1 to ifct1_; Sizes=',
     +               size(ifct1),size(ifct1_)
      endif
      !endif
            
      write(*,*)'urfpack:  END'
      return
      end
