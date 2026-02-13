c
c
      subroutine synchrad !called for each lr_ (indxlr_)
      implicit integer (i-n), real*8 (a-h,o-z)
      !save

c..................................................................
c     This routine computes  synchrotron radiation loss coefficients 
c     for electrons.
c     syncrad="disabled","gyro","geom", or "gyrogeom". 
c     The relativistic coefficients have been calculated
c     using the hueristic approach of Bernstein and Baxter.
c     (I.B. Bernstein and D.C. Baxter, Phys. Fluids 24, 
c      p. 108 (1981), Appendix A (not equation given in main text).)
c     Plus
c     synchrotron radiation due to toroidal and poloidal rotation
c     (BH + Paul Parks, March 2000).
c..................................................................

      include 'param.h'
      include 'comm.h'

c     Temporary, for writing/printing:
      real*8 synca_gyro(jx),syncd_gyro(jx)
      real*8 synca_geom(jx),syncd_geom(jx)

      if ((syncrad.eq."disabled") .or. (kelecg.eq.0)) return
      
      !synca=0.d0 !call bcast(synca,zero,iyjx*lrz) !DO NOT wipe data for all lr!
      !syncd=0.d0 !call bcast(syncd,zero,iyjx*lrz) !BUG fixed by YuP[2023-06-30]

      if (syncrad.eq."gyro" .or. syncrad.eq."gyrogeom") then

      coefnt=(2.*(charge*bnumb(kelecg))**4*bmod0(lr_)**2)/
     *  (3.*clight**5*fmass(kelecg)**3)
      do  j=1,jx
         do  i=1,iy
            synca(i,j,indxlr_)=coefnt*vptb(i,lr_)*sinn(i,l_)**2*xcu(j)*
     *           psicu(i,lr_)*gamma(j) ! A ~gamma  (YuP: correct)
            syncd(i,j,indxlr_)=coefnt*vptb(i,lr_)*xsq(j)*gammi(j)*
     *           sinn(i,l_)**2/coss(i,l_)*
     *           (psisq(i,lr_)-sinn(i,l_)**2*psicu(i,lr_)) ! D ~1/gamma  (YuP: correct)
         enddo
      enddo

!      synca_gyro=0.d0 !call bcast(synca_gyro,zero,jx) !For printout
!      syncd_gyro=0.d0 !call bcast(syncd_gyro,zero,jx)
!      do j=1,jx
c         do i=1,iy !For printout only:
!          i=11 !YuP[2018-01-16] Why i=11? Only for printout, anyway.
!            synca_gyro(j)=synca_gyro(j)+ synca(i,j,indxlr_)
!            syncd_gyro(j)=syncd_gyro(j)+ syncd(i,j,indxlr_)
c         enddo
!      enddo

      endif !(syncrad.eq."gyro" .or. syncrad.eq."gyrogeom")
      

      
      if (syncrad.eq."geom" .or. syncrad.eq."gyrogeom") then
      !syncrad="geom" or "gyrogeom" is N/A for a mirror machine.
      synca_geom=0.d0 !call bcast(synca_geom,zero,jx) !For printout
      syncd_geom=0.d0 !call bcast(syncd_geom,zero,jx) !For printout
      coefnt1=2.*(charge*bnumb(kelecg))**2/(3.*clight**3*fmass(kelecg))
      bsq=bthr(lr_)**2+btoru(lr_)**2
      bphiob4=(btoru(lr_)**2/bsq)**2
      bthob4=(bthr(lr_)**2/bsq)**2
      reffm2=onovrp(2,lr_)*bphiob4 + pi/areacon(lr_)*bthob4
      
      !Curvature-B related terms
      do  j=1,jx
         do  i=1,iy
            sync1=  +coefnt1*vptb(i,lr_)*
     *           xcu(j)*xsq(j)*vnorm2*gamma(j)*reffm2*
     *           (1.-2.*sinn(i,l_)**2*psiba(i,lr_)+
     *           sinn(i,l_)**4*psisq(i,lr_)) ! was A ~1/gamma  (YuP: wrong)
                 !YuP[2020-06-18] Corrected error in the above: Should be A ~gamma
                 !See YuP_2020_Synchrotron_derivations.pdf
            synca(i,j,indxlr_)=synca(i,j,indxlr_)+sync1
            sync2=  -coefnt1*vptb(i,lr_)*
     *           xcu(j)*x(j)*vnorm2*gammi(j)*reffm2*
     *           sinn(i,l_)**2/coss(i,l_)*
     *           (1.-2.*sinn(i,l_)**2*psiba(i,lr_)+
     *           sinn(i,l_)**4*psisq(i,lr_))
            syncd(i,j,indxlr_)=syncd(i,j,indxlr_)+sync2 ! D ~1/gamma  (YuP: correct)
            if (i.eq.1) then
            synca_geom(j)=synca_geom(j)+  sync1
            syncd_geom(j)=syncd_geom(j)+  sync2
            endif
        enddo !i
      enddo !j

      endif !(syncrad.eq."geom" .or. syncrad.eq."gyrogeom")

      if (ioutput(1).ge.2) then !YuP[2020] diagnostic printout
       if(syncrad.eq."gyro")then !YuP[2023-06-28] added if()
        !write(*,*) 'synchrad: synca_gyro: ',synca_gyro
        !write(*,*) 'synchrad: syncd_gyro: ',syncd_gyro
        !Shorter printout:
        write(*,'(a,i3,1p2e12.5)')' synchrad:  l_,  synca,  syncd=',
     &    indxlr_, synca(iyh,jx,indxlr_), syncd(iyh,jx,indxlr_)
       endif
       if (syncrad.eq."geom" .or. syncrad.eq."gyrogeom") then
        !syncrad.eq."gyrogeom" is N/A to a mirror machine (getting NaNs)
        write(*,*) 'synchrad: synca_geom: ',synca_geom
        write(*,*) 'synchrad: syncd_geom: ',syncd_geom
       endif
      endif


      return
      end
