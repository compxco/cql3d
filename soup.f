c
c
      subroutine soup(cosi,l,kk,m)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     Given a point z(l,lr_) along the field line z, and given
c     the local cosine of the pitch angle of the particle,cosi,
c     compute a velocity source profile soupp(j,lr_),j=1,jx
c     for source number m of species kk.
c..................................................................

      include 'param.h'
      include 'comm.h'

      dimension ifirst(lrza)
      data ifirst/lrza*0/

c..................................................................
c     For the case that a Gaussian profile in polar coordinates
c     is to be computed determine some exponentials depending on
c     speed alone and store them in sovt(j,kk,m,lr_)
c..................................................................

      if (ifirst(lr_) .eq. 0) call soup0(l) !define sovt(j,kk,m,lr_) array,
         !based on xem1,2(*,lr) input for src localization in energy
         !Presently xem1, xem2 do not depend on l index along field line,
         !but maybe in future - they will, so added l index to soup0(l)
      ifirst(lr_)=1
      zl=z(l,lr_)

c..................................................................
c     ctl is cos(theta(z(l,lr_))) for the zero banana width pinch orbit.
c     In other words the trapped/passing boundary maps to acos(ctl).
c..................................................................
      !This is from sin^2(ctl) = (B/B0)*sin^2(theta0)
      ctl=sqrt(abs(1.-psif(zl)*(1.-coss(itl_(lmdpln_),lmdpln_)**2)))
         !YuP[2022-02-09] changed itl-->itl_(lmdpln_)
      thtl=acos(ctl) ! Local theta
      
      
      if (soucoord.eq."cart") then
        sini=sqrt(abs(1.-cosi*cosi))

c..................................................................
c     Determine the Gaussian profile next
c     tam7,10 contain contribution from cosi and tam8,9 contribution fro
c     -cosi (reflected about pi/2).
c..................................................................

        do 10 j=1,jx
          tam13(j)=x(j)*cosi
          tam12(j)=-tam13(j)
          tam11(j)=x(j)*sini
          tam10(j)=-(tam13(j)-sxllm1(kk,m,lr_))**2/sxllm2(kk,m,lr_)
          tam9(j)=-(tam12(j)-sxllm1(kk,m,lr_))**2/sxllm2(kk,m,lr_)
          tam6(j)=-(tam11(j)-sxppm1(kk,m,lr_))**2/sxppm2(kk,m,lr_)
          tam7(j)=exp(tam10(j)+tam6(j))
          tam8(j)=exp(tam9(j)+tam6(j))
 10     continue

c..................................................................
c     Now for polar coordinate case.
c..................................................................

      else ! soucoord.eq."polar"
        if(cosm2(kk,m,lr_).ne.0.d0)then !YuP[2020-10-20] Added check of denom.
          ! cosm2(kk,m,lr_) can be 0 when scm2(kk,m)=0. 
          !(set in aindflt: scm2(k,m)=0., for k>1 species)
          facc= -( cosi-cosm1(kk,m,lr_))**2/cosm2(kk,m,lr_)
          faccr=-(-cosi-cosm1(kk,m,lr_))**2/cosm2(kk,m,lr_)
          q1=exp(facc)
          q2=exp(faccr)
          do j=1,jx
            tam7(j)=q1*sovt(j,kk,m,lr_)
            tam8(j)=q2*sovt(j,kk,m,lr_)
          enddo
        else ! cosm2=0.d0
          tam7(:)=0.d0
          tam8(:)=0.d0
        endif     
      endif

c..................................................................
c     Symmetrize in the trapped region.
c..................................................................

      if(cqlpmod.ne."enabled")then !YuP[2022-02]
        if (abs(cosi).le.ctl) then !trapped region
          do j=1,jx
            soupp(j,lr_)=(tam7(j)+tam8(j))*.5
          enddo
        else
          call dcopy(jx,tam7,1,soupp(1,lr_),1)
        endif
      else !CQLP
          call dcopy(jx,tam7,1,soupp(1,lr_),1)
      endif

c..................................................................
c     Give the desired z(l,lr_) dependence to the source current
c..................................................................

      if (isounor .ne. 1) then
        !Note: isounor is reset to 0 at n=0, see subr. sounorm
        if(zm2(kk,m,lr_).ne.0.d0)then !YuP[2020-10-20] Added check of denom.
          ! zm2(kk,m,lr_) can be 0 when szm2(k,m)=0. 
          !However, in aindflt, szm2(k,m) is set to 1. or other, so - no danger)
          facz=exp(-(zl-zm1(kk,m,lr_))**2/zm2(kk,m,lr_))
        else
          facz=0.d0 ! ok?
        endif
!        if(facz.gt.1d-6)then
!          write(*,'(a,2i4,1pe11.4)')'soup: n,l,facz=',n,l,facz
!        endif
        call dscal(jx,facz*sounor(kk,m,l,lr_)*asor(kk,m,lr_),
     1               soupp(1,lr_),1)
        do j=1,jx
        if(soupp(j,lr_).lt.0.d0)then
          write(*,*)'soup: l,facz,sounor(kk,m,l,lr_)=',
     &     l,facz,sounor(kk,m,l,lr_)     
        endif
        enddo
        
      endif ! isounor
      
      return
      end
