
      subroutine losstor(k)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     This routine provides a number of phenomenological loss models
c     for toroidal scenarios. In general a loss time as a function of
c     momentum is determined.
c
c     tauloss(1,k)(1,k) .lt. 0.  defaults loss times to infinity 
c                   (ep90, no losses).
c
c     torloss(k)="velocity" gives a simple model dependent on three
c     loss times
c
c     torloss(k) = "flutter" gives a model similar to "velocity", but
c     dividing rather than multiplying the x part, to be able to model
c     Rechester-Rosenbluth type loss (with tauloss(3,k)=1):
c     tau = tauloss(1,k)/(1 + tauloss(2,k)*(x*vnorm)**tauloss(3,k))
c     torloss(k) = "flutter1", then further multiply by gamma**5
c     torloss(k) = "flutter2", magnetic fluctuation loss model 
c                   corresponding to given fluct. level delta-B/B. 
c     torloss(k) = "flutter3", further multiply "flutter2" by gamma**5.
c
c     torloss(k)="relativ" means the loss time goes like 
c                 tauloss(1,k)*gamma**3
c
c     torloss(k)="energy" means all particles below an energy of 
c     enloss(k) have a loss time of tauloss(1,k)
c
c     torloss(k)="shell" is like "energy" with the additional effect
c     that all particles with an energy greater than .9*enorm
c     are also lost is a loss time of tauloss(2,k).
c..................................................................

      include 'param.h'
      include 'comm.h'


c..................................................................
c     Default values ...
c..................................................................

      if (tauloss(1,k) .le. 0.) then
        tauloss(1,k)=ep90
        tauloss(2,k)=0.
        tauloss(3,k)=ep90
      endif


      do 50 j=1,jx
         do i=1,iymax !YuP[2021-03-11] iy-->iymax
            taulos(i,j,indxlr_)=ep90
         enddo
 50   continue

c..................................................................
c     Various options - see subroutine header
c     (Skip to end if n not in on-widow)
c..................................................................

      if (n.lt.nonloss .or. n.gt.noffloss) go to 999



      if (torloss(k) .eq. "velocity") then
        do 10 j=1,jx
          if (j .eq. 1) then
             do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
                !YuP[2019-07-03] Was i=i,iy  BUG?
                taulos(i,j,indxlr_)=tauloss(1,k)
             enddo
          else
             do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
                taulos(i,j,indxlr_)=tauloss(1,k)*(1.+tauloss(2,k)
     1               *(x(j)*vnorm)**tauloss(3,k))
             enddo
          endif
 10     continue
      elseif (torloss(k).eq."flutter" .or. 
     +           torloss(k).eq."flutter1") then
        do 12 j=1,jx
          if (j .eq. 1) then
             do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
                taulos(i,j,indxlr_)=tauloss(1,k)
             enddo
          else
             do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
                taulos(i,j,indxlr_)=tauloss(1,k)/(1.+tauloss(2,k)
     1               *(x(j)*vnorm)**tauloss(3,k))
             enddo
          endif
          if (torloss(k).eq."flutter1") then
             do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
                taulos(i,j,indxlr_)=taulos(i,j,indxlr_)
     +               *gamcub(j)*gamsqr(j)
             enddo
          endif
 12    continue

c     These flutter losses apply only to transiting particles:
         do j=1,jx
            do i=itl,itu
               taulos(i,j,indxlr_)=ep90
            enddo
         enddo


c     Magnetic fluctuation loss model, for given delta-B/B:
      elseif (torloss(k).eq."flutter2" .or.
     +      torloss(k).eq."flutter3") then
         if(k.ne.kelecg) stop 'flutter2-3 only set up for electrons'
         c1=8.*pi*charge**4*zeff(lr_)*gama(kelec,kelec)*
     +        reden(k,lr_)/fmass(k)**2/vnorm4
         c2=rhomax*rhomax*(1.-rya(lr_))**2/
     +        (4.*pi*deltabdb**2*vnorm)
         
         do 13 j=1,jx
            if (j .eq. 1) then
               do i=1,itl-1
                  taulos(i,j,indxlr_)=ep90
               enddo
            else
c              Max nu_perp_ei collision freq occurs at v_th_ion.
               x4=max(x(j),
     +              sqrt(temp(kionn,lr_)*ergtkev/fmass(kionn))/vnorm)
               x4=x4**3*x(j)
               do i=1,itl-1
                  reffi=(1./(pi*2.*rmag)) +c1/(x4*coss(i,l_)
     +                 /gamma(j)**2)
                  taulos(i,j,indxlr_)=min(c2*reffi
     +                 /(x(j)*coss(i,l_)/gamma(j)), ep90)
               enddo
            endif
            if (torloss(k).eq."flutter3") then
               do i=1,itl-1
                  taulos(i,j,indxlr_)=taulos(i,j,indxlr_)
     +                 *gamcub(j)*gamsqr(j)
               enddo
            endif
 13      continue

c     These flutter losses apply only to transiting particles:
         do j=1,jx
            do i=itl,iyh
               taulos(i,j,indxlr_)=ep90
            enddo
            do i=1,iyh
               taulos(iy_(l_)+1-i,j,indxlr_)=taulos(i,j,indxlr_)
               !YuP[2021-03-11] iy-->iy_(l_)
            enddo
         enddo



      elseif (torloss(k) .eq. "relativ") then
        do 20 j=1,jx
           do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
              taulos(i,j,indxlr_)=tauloss(1,k)*gamcub(j)
           enddo
 20     continue
      elseif (torloss(k) .eq. "energy") then
        do 30 j=1,jx
          if (fions(k)*tcsgm1(j) .le. enloss(k)) then
             do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
                taulos(i,j,indxlr_)=tauloss(1,k)
             enddo
          else
             do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
                taulos(i,j,indxlr_)=ep90
             enddo
          endif
 30     continue
      elseif (torloss(k) .eq. "shell") then
        do 40 j=1,jx
           do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
              ej=fions(k)*tcsgm1(j)
              if (ej .le. enloss(k)) then
                 taulos(i,j,indxlr_)=tauloss(1,k)
              elseif (ej .gt. .9*enorm) then
                 taulos(i,j,indxlr_)=tauloss(2,k)
              else
                 taulos(i,j,indxlr_)=ep90
              endif
           enddo
 40     continue
      endif
 999  return
      end

