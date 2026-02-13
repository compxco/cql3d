c
c
      subroutine bavgmax
      implicit integer (i-n), real*8 (a-h,o-z)
c.................................................................
c     This routine computes the bounce average of several bbpsi
c     related quantities.
c.................................................................
      save

      include 'param.h'
      include 'comm.h'

      if (n .ge. 2) return
      call bcast(psiiv(1,lr_),zero,iymax) !YuP[2021-03-11] iy-->iymax
      call bcast(psiba(1,lr_),zero,iymax)
      call bcast(psisq(1,lr_),zero,iymax)
      call bcast(psiir(1,lr_),zero,iymax)
      call bcast(vderb(1,lr_),zero,iymax)
      call bcast(psicu(1,lr_),zero,iymax)
      call bcast(psiqu(1,lr_),zero,iymax)
      call bcast(sincosba(1,1,lr_),zero,iymax*ngen)

      lrange=lz
      if (cqlpmod.eq."enabled" .and. numclas .eq. 1) lrange=lz/2+1 !!! YuP: needs to check ?
      !YuP[2025-12-12] added cqlpmod.eq."enabled" in the above line

      !This subr. is only called when if l_=lmdpln_
      !For CQLP, l_=lmdpln_=1
      do 30 l=1,lrange  ! along pol.dir. (along field line)
        if (cqlpmod.eq."enabled") then !YuP[2021-04-02] added
             iiy=iy_(l) !CQLP: at given poloidal index
        else !CQL3D
             iiy=iy_(l_) !CQL3D: at given radial index
        endif
        do 31 i=1,imax(l,lr_)
          iii=iiy+1-i !YuP[2021-03-11] iy-->iy_(l_) 
          ![For CQLP, this subr. is called with l_=1 only, i.e. at midplane]

          ! 1. All orbits that reach/pass a given poloidal node l:
          ! passing (i<itl), or trapped that could reach/pass this l;
          ! also includes Last Trapped, with tip at l=lz_bmax
          !(for such particle, lmax(itl)=lz_bmax; see micxinil)
          ztt=bbpsi(l,lr_)
          dtau_tau=dtau(i,l,lr_)/tau(i,lr_)
          vddert=sin(pol(l,lr_))*(2.+ztt*sinn(i,l_)**2)
          !psiir(i,lr_)=psiir(i,lr_)+sqrt(ztt)*dtau_tau    !-YuP Not used; ztt or 1/ztt?
          psiiv(i,lr_)=psiiv(i,lr_)+(1./ztt)*dtau_tau
          psiba(i,lr_)=psiba(i,lr_)+ztt*dtau_tau
          psisq(i,lr_)=psisq(i,lr_)+ztt**2*dtau_tau
          psicu(i,lr_)=psicu(i,lr_)+ztt**3*dtau_tau
          psiqu(i,lr_)=psiqu(i,lr_)+ztt**4*dtau_tau
          vderb(i,lr_)=vderb(i,lr_)+vddert*dtau_tau
          !write(*,*)'bavg oo',lr_,l,i,dtau(i,l,lr_)
               
          ! 2a. Trapped, with tips between l and l+1 (ABOVE midplane):
          if (l.eq.lmax(i,lr_) .and. l.lt.lz_bmax(lr_)) then 
            ! Add contribution from orbit tip:
            ztt=psif(zboun(i,lr_))
            dtau_tau=dtau(i,l+1,lr_)/tau(i,lr_)
            vddert=sin(pi*zboun(i,lr_)/z_bmax(lr_)) !-YuP  Was /zmax()
            !psiir(i,lr_)=psiir(i,lr_)+sqrt(ztt)*dtau_tau    !-YuP Not used; ztt or 1/ztt?
            psiiv(i,lr_)=psiiv(i,lr_)+(1./ztt)*dtau_tau
            psiba(i,lr_)=psiba(i,lr_)+ztt*dtau_tau
            psisq(i,lr_)=psisq(i,lr_)+ztt**2*dtau_tau
            psicu(i,lr_)=psicu(i,lr_)+ztt**3*dtau_tau
            psiqu(i,lr_)=psiqu(i,lr_)+ztt**4*dtau_tau
            vderb(i,lr_)=vderb(i,lr_)+vddert*dtau_tau
            do  k=1,ngen
              sincost=0.0
              if(paregy(k).le.0.) sincost=1.
              sincosba(i,k,lr_)=sincosba(i,k,lr_)+sincost*dtau_tau
              sincosba(iii,k,lr_)=sincosba(i,k,lr_)
            enddo
            !write(*,*)'bavg <<',lr_,l,i,dtau(i,l+1,lr_)
          endif
          
          ! 2b. Trapped, with tips between l and l-1 (BELOW midplane):
          if (l.eq.lmax(i+iyh,lr_) .and. l.gt.lz_bmax(lr_)) then 
            ! Add contribution from orbit tip:
            ztt=psif(zboun(i,lr_))
            dtau_tau=dtau(i,l-1,lr_)/tau(i,lr_)
            vddert=sin(pi*zboun(i,lr_)/z_bmax(lr_)) !-YuP  Was /zmax()
            !psiir(i,lr_)=psiir(i,lr_)+sqrt(ztt)*dtau_tau    !-YuP Not used; ztt or 1/ztt?
            psiiv(i,lr_)=psiiv(i,lr_)+(1./ztt)*dtau_tau
            psiba(i,lr_)=psiba(i,lr_)+ztt*dtau_tau
            psisq(i,lr_)=psisq(i,lr_)+ztt**2*dtau_tau
            psicu(i,lr_)=psicu(i,lr_)+ztt**3*dtau_tau
            psiqu(i,lr_)=psiqu(i,lr_)+ztt**4*dtau_tau
            vderb(i,lr_)=vderb(i,lr_)+vddert*dtau_tau
            do  k=1,ngen
              sincost=0.0
              if(paregy(k).le.0.) sincost=1.
              sincosba(i,k,lr_)=sincosba(i,k,lr_)+sincost*dtau_tau
              sincosba(iii,k,lr_)=sincosba(i,k,lr_)
            enddo
            !write(*,*)'bavg <<',lr_,l,i,dtau(i,l-1,lr_)
          endif
            
          ! Symmetrize around pitch=pi/2  
c          psiir(iii,lr_)=psiir(i,lr_)    !-YuP Not used
          psiiv(iii,lr_)=psiiv(i,lr_)
          psiba(iii,lr_)=psiba(i,lr_)
          psisq(iii,lr_)=psisq(i,lr_)
          psicu(iii,lr_)=psicu(i,lr_)
          psiqu(iii,lr_)=psiqu(i,lr_)
          vderb(iii,lr_)=vderb(i,lr_)
          
 31     continue ! i
 30   continue ! l


 
      do 70 i=2,iyh
        iii=iy_(l_)+1-i  !YuP[2021-03-11] iy-->iy_(l_)
        swwswwsw=psiiv(i,lr_)/sinn(i,l_)**2-1.
        batot(i,lr_)=tann(i,l_)**2*swwswwsw
        batot(iii,lr_)=batot(i,lr_)
 70   continue
      batot(1,lr_)=psiiv(1,lr_)
      batot(iy_(l_),lr_)=batot(1,lr_)  !YuP[2021-03-11] iy-->iy_(l_)

c      do i=1,iy
c         write(*,*)lr_,i,tau(i,lr_)
c      enddo
c      pause      
c         write(*,*)'bavgmax psimx=',lr_,psimx(lr_)
c         if(lr_.eq.1) pause
      return
      end
