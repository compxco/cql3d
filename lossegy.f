c
c
      subroutine lossegy
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c
cmnt  this routine computes fp coefficients corresponding to the
cmnt  bounce average of the phenomonological energy loss term,
cmnt  or a Bremsstrahlung energy loss term:

c     Phenomonological loss:
c     df/dt = u**-2*d/du [1./tau(u,theta,lr_)*u**3*f/2.]
c     where   tau is energy loss time specified by
c     tau=rfact*tauegy*gamma**gamegy, u.le.vte
c     rfact*tauegy*gamma**gamegy/
c     (abs(u_par,lr_)/vte)**paregy*(u_per/vte)**pe
c     *(u/vte)**pegy),  u.gt.vte
c     rfact is a radially dependent modifier, as below.
c
c     Bremsstrahlung loss due to electron collisions with (only)
c     the Maxwellian ion species:
c     df/dt = u**-2*d/du [beta*u**3*f],
c     where beta=reden(kion,lr_)*sigmarad*clight**2/(u/gamma),
c           sigmarad=as given in Physics Vade Mecum, H.L.Anderson, Ed.,
c                    AIP, Sect. 16.07.F.
c     Sigmarad may be enhanced beyond energy brfacgm3*16.43/Z**(1/6),
c     using brfac,brfac1 (see cqlinput_help and coding below).
c     This is for purposes of stopping flow of runaway electrons
c     off the grid.



      include 'param.h'
      include 'comm.h'
      data gam1/2.30097589089281/,   gam2/16.4312912649856/

      do 200 k=1,ngen

         call bcast(egylosa(0,0,k,indxlr_),zero,(iymax+2)*(jx+2)) !iy-->iymax

c     Phenomenological:

         if(tauegy(k,lr_).le.0.)  go to 130
         coefnt=1./(2.*tauegy(k,lr_))
c     rfact modifier multiplies tau by (1.-rovera(lr_),lr_):
         if(regy(k).eq."enabled") coefnt=coefnt/(1.-rovera(lr_))
c     
c..................................................................
c     Note: the do loop below uses vth(),
c     vth is the thermal velocity =sqrt(T/m) (at t=0 defined in ainpla).
c     But, T==temp(k,lr) can be changed in profiles.f, 
c     in case of iprote (or iproti) equal to "prbola-t" or "spline-t"
c..................................................................
         do 100  j=1,jx
            vel=x(j)*vnorm
            coefnt1=coefnt*xsq(j)*x(j)/gamma(j)**gamegy(k)
            if(vel.le.vth(k,lr_))  then
               do 110  i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
                  egylosa(i,j,k,indxlr_)=vptb(i,lr_)*coefnt1
 110           continue
            else
               do 120  i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
                  egylosa(i,j,k,indxlr_)=
     +                 vptb(i,lr_)*coefnt1*(vel/vth(k,lr_))**
     +                 (paregy(k)+peregy(k)+pegy(k))*sincosba(i,k,lr_)
 120           continue
            endif
 100     continue
 130     continue

c     Bremsstrahlung:
         if (k.eq.kelecg .and. bremsrad.eq."enabled") then
            if(brfacgm3.lt.1.) stop 'Check brfacgm3, see cqlinput_help'
            c1= fmass(kelecg)*clite2 ! m_e*c^2
            r02_alf= (charge**2/c1)**2/137.
            do kk=1,nionm
               kkk=kionm(kk)
               !c1=fmass(kelecg)*clite2 !YuP: it is set above
               c2= r02_alf*bnumb(kkk)**2
               c3=1./bnumb(kkk)**(1./3.)
               c5=sqrt(c3)
               c4=183.*c3
               !c3=137.*c3 ! YuP: not used below
               gam3=c5*gam2
               gam4=brfacgm3*gam3
               sigmarad=0.
               do j=1,jx
                  if (gamm1(j).lt.gam1) then
                     sigmarad=16./3.*c2
                  elseif (gamm1(j).gt.gam1.and.
     &                   gamm1(j).le.gam3) then
                     sigmarad=8.*c2*(log(gamm1(j))-1./6.)
                  elseif (gamm1(j).gt.gam3.and.
     &                  gamm1(j).le.gam4) then
                     sigmarad=4.*c2*(log(c4)+1./18.)
                  elseif (gamm1(j).gt.gam4) then
                     gam5=gamm1(j)-gam4
                     sigmarad=4.*c2*(log(c4)+1./18.)*(1.+brfac*gam5
     &                    **brfac1)
                  endif
                  betau3=reden(kkk,lr_)*sigmarad*clite2*gamma(j)*
     &                 xsq(j)/vnorm
                  do i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
                     egylosa(i,j,k,indxlr_)=egylosa(i,j,k,indxlr_)+
     &                    vptb(i,lr_)*betau3
                  enddo ! i
               enddo ! j
            enddo ! kk=1,nionm
 
            !YuP[2020-06-22] !Contribution from partially ionized impurities.
            !!goto 200 ! Temporary, to skip this part.
            if(n.gt.0 .and. nstates.gt.0)then 
             ! If no suitable impurity is present 
             ! (example: imp_type is set to 0 in cqlinput),
             ! then nstates remains 0 ==> Effectively, no impurities.
             z_imp=bnumb_imp(nstates) !Atomic number (as in fully ionized state)
             !c1= fmass(kelecg)*clite2 ! m_e*c^2
             !r02_alf= (charge**2/c1)**2/137. !Set above (in Maxw.ions section)
             c3=1./z_imp**(1./3.) !This will go under ln(), as ln(c4).
             c5=sqrt(c3)
             c4=183.*c3
             gam3=c5*gam2
             gam4=brfacgm3*gam3
             ! Add ions from impurities, all charge states:
             do kstate=0,nstates ! 0 is for neutral state.
               dens_kstate=dens_imp(kstate,lr_) !updated at each time step.
               z_state= bnumb_imp(kstate) ! Z0j in paper, for given Z state
               z_bound= z_imp-z_state ! Nej in paper (number of bound electrons)
               !Note that for neutral atom, kstate=0 and bnumb_imp(0)=0
               c2= r02_alf*(z_imp**2 + z_bound)
               !Pigarov suggested to use (Za^2 + Nbound) as a leading factor
               ! for the stopping power. (Nbound=z_bound here, in the code).
               !For a neutral gas, Nbound=Za (Za=z_imp here, the atomic number)
               ! and then the leading factor is Za*(Za+1).
               !For fully ionized state, Nbound=0, 
               ! and then the leading factor is Za^2.
               !It is a small difference; for Neon (Za=10), the difference
               ! is only 10%.
               !The rest of the coding below repeats the coding above
               ! for the Maxwellian (fully ionized) ions,
               ! except reden(kion) is replaced by dens_kstate.
               c2_lnc4= 4.*c2*(log(c4)+1./18.)
               sigmarad=0.
               do j=1,jx
                  if (gamm1(j).lt.gam1) then
                     sigmarad=16./3.*c2
                  elseif (gamm1(j).gt.gam1.and.
     &                    gamm1(j).le.gam3) then
                     sigmarad=8.*c2*(log(gamm1(j))-1./6.)
                  elseif (gamm1(j).gt.gam3.and.
     &                    gamm1(j).le.gam4) then
                     sigmarad=c2_lnc4 ! c2_lnc4= 4.*c2*(log(c4)+1./18.)
                  elseif (gamm1(j).gt.gam4) then
                     gam5=gamm1(j)-gam4
                     sigmarad=c2_lnc4*(1.+brfac*gam5**brfac1)
                  endif
                  betau3=dens_kstate*sigmarad*clite2*gamma(j)*
     &                 xsq(j)/vnorm
                  do i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
                     egylosa(i,j,k,indxlr_)=egylosa(i,j,k,indxlr_)+
     &                    vptb(i,lr_)*betau3 !contribution is added
                  enddo ! i
               enddo ! j
             enddo ! kstate
            endif ! nstates.gt.0

         endif ! (k.eq.kelecg .and. bremsrad.eq."enabled")

 200  continue !k=1,ngen

      return
      end
