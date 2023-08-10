c
c
c
      subroutine ainpla
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..........................................................
c     This routine initialize some plasma parameter profiles
c     on the whole lrzmax radial mesh
c     (some where defined in diaggnde before)
c.............................................................

      include 'param.h'
      include 'comm.h'

c.......................................................................
cl    1. Energy, v-thermal
c.......................................................................

cl    1.1 radial mesh

      do 110 k=1,ntotal
        rstmss=fmass(k)*clite2/ergtkev
        do 111 l=1,lrzmax
          thta=rstmss/temp(k,l)
          if (thta.gt.100. .or. relativ.eq."disabled") then
            energy(k,l)=1.5*temp(k,l)
            !write(*,*)'ainpla.1:  energy(k,l)/1.5=',energy(k,l)/1.5
          else
            call cfpmodbe(thta,bk1,bk2)
            energy(k,l)=rstmss*(bk1/bk2-1.+3./thta)
            !write(*,*)'ainpla.2:  energy(k,l)/1.5=',energy(k,l)/1.5
          endif
c          write(*,*)'ainpla: k,l,energy(k,l):',k,l,energy(k,l)
          vth(k,l)=((temp(k,l)*ergtkev)/fmass(k))**.5
          if (k .eq. kelec) vthe(l)=vth(kelec,l)
 111    continue
 110  continue

c.......................................................................
cl    1.2 parallel mesh
c.......................................................................

      if (cqlpmod .eq. "enabled") then

        do 120 k=1,ntotal
          rstmss=fmass(k)*clite2/ergtkev
          do 121 l=1,lsmax
            thta=rstmss/temppar(k,l)
            if (thta.gt.100. .or. relativ.eq."disabled") then
              enrgypa(k,l)=1.5*temppar(k,l)
            else
              call cfpmodbe(thta,bk1,bk2)
              enrgypa(k,l)=rstmss*(bk1/bk2-1.+3./thta)
            endif
            vthpar(k,l)=((temppar(k,l)*ergtkev)/fmass(k))**.5
 121      continue
          if (sbdry.eq."periodic" .and. transp.eq."enabled") then
            enrgypa(k,0)=enrgypa(k,lsmax)
            enrgypa(k,lsmax+1)=enrgypa(k,1)
            vthpar(k,0)=vthpar(k,lsmax)
            vthpar(k,lsmax+1)=vthpar(k,1)
          endif
 120    continue

      endif
c.......................................................................
c     2. Compute radial Z-effective
c.......................................................................

      if (izeff.eq."ion") then
        k1=ngen+1
      else
        k1=1
      endif
      do 200 l=1,lrzmax
        zeff(l)=0.
        zeff1=0.
        zeff4(l)=0.d0 !Yup[2014-05-27] Initialize to 0.
        xq=0.
        do 210 k=k1,ntotal
          if (k.eq.kelecg .or. k.eq.kelecm) goto 210
cBobH990128          if (k.eq.izeff) goto 210
          xq=xq+1.
          zeff(l)=zeff(l)+bnumb(k)**2*reden(k,l)
          zeff4(l)=bnumb(k)**4*reden(k,l)+zeff4(l)
          zeff1=zeff1+bnumb(k)*reden(k,l)
 210    continue

        !if(nstates.gt.0)then
         !YuP[2020-06-22] Skip this part 
         ![contribution of partially ionized impurities to Zeff],
         !because nstates and bnumb_imp() are not set yet 
         !(ainpla is called too early in tdinitl, before set_impurity_data is called).
         !This is not important, because
         !tdinitl->ainitial->diaggnde calculates zeff() and zeff4() anyway
         !(at each n, including n=0).
         !  do kstate=1,nstates ! Now additional ions from impur.source.
         !   dens_kstate=dens_imp(kstate,lr_)
         !   xq=xq+1
         !   zeff(lr_)= zeff(lr_) +dens_kstate*bnumb_imp(kstate)**2
         !   zeff4(lr_)=zeff4(lr_)+dens_kstate*bnumb_imp(kstate)**4
         !   zeff1=     zeff1     +dens_kstate*bnumb_imp(kstate)            
         !  enddo ! kstate
        !endif ! 
        zeff4(l)=zeff4(l)/xq
        zeff(l)=zeff(l)/zeff1
 200  continue ! l=1,lrzmax

      return
      end


!=======================================================================
!=======================================================================
      subroutine set_impurity_data !(INPUT: imp_type is in namelist)
!------------------------- contribution from partially ionized ions -----
!YuP[2020-07-02] Set general data for selected type of impurity.
! (Before 2020-07, it was part of subr.set_gscreen_hesslow). 
!This subroutine should be called when 
!  (imp_depos_method.ne."disabled")
! At present, it is set for one ion type (imp_type in namelist), 
! but it could be generalized in future.
! Below, excitation energies are from Stephan P.A. Sauer, Jens Oddershede, John R. Sabin
! Advances in Quantum Chemistry, Volume 71; http://dx.doi.org/10.1016/bs.aiq.2015.02.001
      implicit integer (i-n), real*8 (a-h,o-z)
CMPIINSERT_INCLUDE
      include 'param.h'
      include 'comm.h'
      !integer imp_type ! INPUT, from namelist
      integer istat ! local
      !OUTPUT:
      !integer nstates ! save into comm.h, to be used by sub.cfpcoefn
      !real*8 fmass_imp ! scalar; save into comm.h, to be used by sub.cfpcoefn
      !Also these arrays are set in this subroutine: 
      !real*8, dimension(:), pointer :: a_imp   !(0:nstates)
      !real*8, dimension(:), pointer :: bnumb_imp !(0:nstates) !Could be set as integer?
      !real*8, dimension(:), pointer :: excit_enrgy !(0:nstates)
      
      SELECT CASE (imp_type)
      ! imp_type: 1->He,2->Be,3->C,4->N,5->Ne,6->Ar,7->Xe,8->W
      CASE(1) ! He
         nstates=2 ! He[0](neutral), He[1+], He[2+]
         allocate(a_imp(0:nstates),STAT=istat) !== a^bar in paper (Table 1)
         allocate(bnumb_imp(0:nstates),STAT=istat) !Z0j in Hesslow
         allocate(excit_enrgy(0:nstates),STAT=istat) !Ij in Hesslow. [eV] here
         fmass_imp=4.*proton
         a_imp=     (/173,  123,  0/ ) ! a_imp(nstates) can be any value.
         bnumb_imp=  (/0.d0, 1.d0, 2.d0/ ) ! Charge number of each kstate
         excit_enrgy=(/42.68,59.88,1./) !eV! Last number is not important
      CASE(2) ! Be
         nstates=4 ! Be[0](neutral), Be[1+], Be[2+], Be[3+], Be[4+]
         allocate(a_imp(0:nstates),STAT=istat) !== a^bar in paper (Table 1)
         allocate(bnumb_imp(0:nstates),STAT=istat) !Z0j in Hesslow
         allocate(excit_enrgy(0:nstates),STAT=istat) !Ij in Hesslow
         fmass_imp=9.*proton
         a_imp=   (/159, 114, 67,  59,  0/ ) ! a_imp(nstates) can be any value
         bnumb_imp=(/0.d0,1.d0,2.d0,3.d0,4.d0/ ) ! Charge number of each kstate
         !No data on excit energy for Be. Use approximation:
         excit_enrgy(0)=10.*bnumb_imp(nstates) ! 10ev*Zatomic_number for neutral
         !We take excit_enrgy_eV= 10eV *Z_imp (atomic number) 
         ![See Breizman/NF2019, after Eq.(27): mean excit. potential] 
         !This is ok for an atom, but not good for a partially-ionized ion.
         do kstate=1,nstates-1 !All ionized states, except fully-ionized
         excit_enrgy(kstate)= 15.*bnumb_imp(kstate)**2 ! 15ev*Zion^2
         !This is a very rough estimate, so better find data.
         enddo
         excit_enrgy(nstates)=1. ! Fully-ionized: Any number .ne.0
         !For fully-ionized state hbethe()=0 because z_bound=0.
      CASE(3) ! C
         nstates=6 !C[0](neutral),C[1+],C[2+],C[3+],C[4+],C[+5],C[+6]
         allocate(a_imp(0:nstates),STAT=istat) !== a^bar (Table 1)
         allocate(bnumb_imp(0:nstates),STAT=istat) !Z0j in Hesslow
         allocate(excit_enrgy(0:nstates),STAT=istat) !Ij in Hesslow
         fmass_imp=12.*proton
         a_imp=     (/144, 118, 95,   70,   42,   39,   0/ ) 
         bnumb_imp=  (/0.d0,1.d0,2.d0, 3.d0, 4.d0, 5.d0, 6.d0/ ) 
         excit_enrgy=(/65.9,92.6,134.8,214.2,486.2,539.5,1./) !eV! Last number is not important
      CASE(4) ! N
         nstates=7 
         allocate(a_imp(0:nstates),STAT=istat) !== a^bar (Table 1)
         allocate(bnumb_imp(0:nstates),STAT=istat) !Z0j in Hesslow
         allocate(excit_enrgy(0:nstates),STAT=istat) !Ij in Hesslow
         fmass_imp=14.*proton
         a_imp=   (/135, 115, 97,  79,  59,  35,  33,  0/ ) 
         bnumb_imp=(/0.d0,1.d0,2.d0,3.d0,4.d0,5.d0,6.d0,7.d0/ ) 
         !No data on excit energy for N. Use approximation:
         excit_enrgy(0)=10.*bnumb_imp(nstates) ! 10ev*Zatomic_number for neutral
         do kstate=1,nstates-1
         excit_enrgy(kstate)= 15.*bnumb_imp(kstate)**2 ! 15ev*Zion^2
         enddo
         excit_enrgy(nstates)=1. ! Any number .ne.0
      CASE(5) ! Ne
         nstates=10 
         allocate(a_imp(0:nstates),STAT=istat) !== a^bar (Table 1)
         allocate(bnumb_imp(0:nstates),STAT=istat) !Z0j in Hesslow
         allocate(excit_enrgy(0:nstates),STAT=istat) !Ij in Hesslow
         fmass_imp=20.*proton
         a_imp=     (/111,  100,  90,   80,   71,   62,   52,   40,    
     &    24,    23,    0/ ) 
         bnumb_imp=  (/0.d0, 1.d0, 2.d0, 3.d0, 4.d0, 5.d0, 6.d0, 7.d0,
     &    8.d0,  9.d0,  10.d0/ ) 
         excit_enrgy=(/137.2,165.2,196.9,235.2,282.8,352.6,475.0,696.8,
     &    1409.2,1498.2,1.0/) !eV! Last number is not important
      CASE(6) ! Ar
         nstates=18 
         allocate(a_imp(0:nstates),STAT=istat) !== a^bar (Table 1)
         allocate(bnumb_imp(0:nstates),STAT=istat) !Z0j in Hesslow
         allocate(excit_enrgy(0:nstates),STAT=istat) !Ij in Hesslow
         fmass_imp=40.*proton
         a_imp=     (/96,  90,  84,  78,  72,  65,  59,  53,  47,  
     &    44,  41,   38,   35,   32,   27,   21,   13,   13,   0/ ) 
         bnumb_imp=  (/0.d0,1.d0,2.d0,3.d0,4.d0,5.d0,6.d0,7.d0,8.d0,
     &    9.d0,10.d0,11.d0,12.d0,13.d0,14.d0,15.d0,16.d0,17.d0,18.d0/ ) 
         excit_enrgy=(/188.5,219.4,253.8,293.4,339.1,394.5,463.4,568.,
     &    728.,795.9,879.8,989.9,1138.1,1369.5,1791.2,2497.,4677.2,
     &    4838.2, 1.0/) !eV! Last number is not important
      CASE(7) ! Xe (Z=54, but only states 1+,2+,3+ are given in Table 1)
         nstates=54 
         allocate(a_imp(0:nstates),STAT=istat) !== a^bar (Table 1)
         allocate(bnumb_imp(0:nstates),STAT=istat) !Z0j in Hesslow
         allocate(excit_enrgy(0:nstates),STAT=istat) !Ij in Hesslow
         fmass_imp=131.*proton !(131.3)
         a_imp(:)=0.d0
         a_imp=   (/65,  65,  63,  61/ ) ! Set the rest to 0?
         do kstate=0,nstates
         bnumb_imp(kstate)= kstate*1.d0 
         enddo
         !No data on excit energy for Xe. Use approximation:
         excit_enrgy(0)=10.*bnumb_imp(nstates) ! 10ev*Zatomic_number for neutral
         do kstate=1,nstates-1
         excit_enrgy(kstate)= 15.*bnumb_imp(kstate)**2 ! 15ev*Zion^2
         enddo
         excit_enrgy(nstates)=1. ! Any number .ne.0
      CASE(8) ! W (Z=74, but only states 0,30+,40+,50+,60+ are given in Table 1)
         !   Z=    [0   30  40  50  60  ];
         !   a_bar=[59  33  25  18  13  ];
         !The value of a_bar (a_imp here) in Hesslow (JPP-2018) is almost 
         ! a linear function of Z state.
         !We simply fill-in the rest of points.
         !Probably the line is approaching a=10 at Z=74.
         nstates=74 
         allocate(a_imp(0:nstates),STAT=istat) !== a^bar (Table 1)
         allocate(bnumb_imp(0:nstates),STAT=istat) !Z0j in Hesslow
         allocate(excit_enrgy(0:nstates),STAT=istat) !Ij in Hesslow
         fmass_imp=184.*proton !(183.84)
         a_imp(:)=0.d0
         a_imp=(/59.0000,58.1333,57.2667,56.4000,55.5333,
     &           54.6667,53.8000,52.9333,52.0667,51.2000,
     &           50.3333,49.4667,48.6000,47.7333,46.8667,
     &           46.0000,45.1333,44.2667,43.4000,42.5333,
     &           41.6667,40.8000,39.9333,39.0667,38.2000,
     &           37.3333,36.4667,35.6000,34.7333,33.8667,
     &           33.0000,32.2000,31.4000,30.6000,29.8000,
     &           29.0000,28.2000,27.4000,26.6000,25.8000,
     &           25.0000,24.3000,23.6000,22.9000,22.2000,
     &           21.5000,20.8000,20.1000,19.4000,18.7000,
     &           18.0000,17.5000,17.0000,16.5000,16.0000,
     &           15.5000,15.0000,14.5000,14.0000,13.5000,
     &           13.0000,12.7857,12.5714,12.3571,12.1429,
     &           11.9286,11.7143,11.5000,11.2857,11.0714,
     &           10.8571,10.6429,10.4286,10.2143, 0.0    /)
         do kstate=0,nstates
         bnumb_imp(kstate)= kstate*1.d0 
         enddo
         !No data on mean excit energy for W. Use approximation:
         !10.*bnumb_imp(nstates) !10ev*Zatomic_number for neutral
         excit_enrgy(0)=727.0 ! From Table 4.3 in nbsir82-2550.pdf
         do kstate=1,nstates-1
         excit_enrgy(kstate)=15.*(bnumb_imp(kstate)+1.)**2 !15ev*(Z+1)^2
         enddo
         excit_enrgy(nstates)=1. ! Any number .ne.0
      CASE DEFAULT 
         !If none of allowed imp_type was used: 
         nstates=0 ! Effectively, no impurities. Below, gscreen(0,j)-->0
         allocate(a_imp(0:nstates),STAT=istat) 
         allocate(bnumb_imp(0:nstates),STAT=istat) !Z0j in Hesslow
         allocate(excit_enrgy(0:nstates),STAT=istat) !Ij in Hesslow
         a_imp=   0.d0 ! a_imp(nstates) can be any value.
         bnumb_imp=0.d0 ! Charge number of each kstate
         excit_enrgy=1.0 ! Any number>0
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
     &    'set_impurity_data/WARNING: No data on this imp_type=',
     &    imp_type
CMPIINSERT_ENDIF_RANK
      END SELECT
      
      !allocate arrays for density and temperature 
      !of each ionization state:
      allocate(fz(0:nstates),STAT=istat) !Distribution over charge states,
      !to be found at each time step and each lr flux surface.
      allocate(dens_imp(0:nstates,1:lrz),STAT=istat)
      allocate(temp_imp(0:nstates,1:lrz),STAT=istat)
      !allocate(dens_imp_allstates(1:lrz),STAT=istat)
      fz(0:nstates)=0.d0 ! initialize
      dens_imp(0:nstates,1:lrz)=0.d0 ! initialize
      temp_imp(0:nstates,1:lrz)=0.d0 ! initialize
      dens_imp_allstates(:)=0.d0 ! initialize
      !The values are supposed to be defined at each time step,
      !from radiative balance.
      
      return
      end subroutine set_impurity_data

!=======================================================================
!=======================================================================
      subroutine set_gscreen_hesslow !(INPUT: imp_type is in namelist)
!------------------------- contribution from partially ionized ions -----
!YuP[2019-07-29] Set gscreen(p) function (of normalized momentum p) 
!that describes the effect of partially screened ions 
!on enhanced scattering of electrons. Fast electrons can "probe" 
!the inner structure of a partially ionized ion, or a neutral atom.
!Also, set hbethe(p) function that describes the slowing down
!of free electron on bound electrons in partially ionized ion
!or neutral atom.
!See Hesslow et al, JPP-2018,vol.84, Eq.(2.25) and (2.31).
!This subroutine should only be called when gamafac.eq."hesslow" .and. kelecg.eq.1
! At present, it is set for one ion type (imp_type in namelist), 
! but it could be generalized in future.
! Below, excitation energies are from Stephan P.A. Sauer, Jens Oddershede, John R. Sabin
! Advances in Quantum Chemistry, Volume 71; http://dx.doi.org/10.1016/bs.aiq.2015.02.001
      implicit integer (i-n), real*8 (a-h,o-z)
CMPIINSERT_INCLUDE
      include 'param.h'
      include 'comm.h'
      !INPUT: integer imp_type ! (from namelist)
      !INPUT: bnumb_imp(),excit_enrgy(),a_imp() [set in subr.set_impurity_data]
      integer istat ! local
      real*8 z_imp,z_state,zz_state,z_bound,zz_bound,p_n,pa32,two3rd !local
      real*8 beta_v,beta2,hj !local
      !OUTPUT: gscreen(0:nstates,1:jx), hbethe(0:nstates,1:jx)
      
      !--- Now setup gscreen for the given case from above.
      write(*,*)'set_gscreen_hesslow: nstates',nstates
      allocate(gscreen(0:nstates,1:jx),STAT=istat)
      allocate(hbethe(0:nstates,1:jx),STAT=istat)
      z_imp=bnumb_imp(nstates) !Atomic charge number (as in fully ionized state)
      two3rd= 2.d0/3.d0 ! Just a constant
      do kstate=0,nstates
         z_state= bnumb_imp(kstate) ! Z0j in paper
         z_bound=  z_imp-z_state ! Nej in paper (number of bound electrons)
         zz_bound= z_bound*z_bound
         zz_state= z_bound*(z_imp+z_state) !=z_imp**2-z_state**2 = Zj^2-Z0j^2 in paper
         excit_en_n= (excit_enrgy(kstate)*1.d-3)/restmkev ! keV/keV
         !Note: excit_en_n is norm-ed by me*c^2==restmkev=510.998902d0keV
         do j=1,jx
           p_n= x(j)*vnorm/clight ! norm-ed momentum
           !Note: p_n = gamma*v/c where v is speed, 
           !and gamma=sqrt(1+(x*vnorm/c)^2)=sqrt(1+p_n^2)
           beta_v= p_n/gamma(j) ! = v/c
           beta2= beta_v*beta_v
           pa32= (p_n*a_imp(kstate))**1.5
           gscreen(kstate,j)= two3rd*( zz_state*log(pa32+1.d0) 
     &                                -zz_bound*pa32/(pa32+1.d0) )
     &                              *imp_bounde_collscat
           !Note: for kstate=nstates we have z_state=z_imp (so z_bound=0),
           !so that gscreen(nstates,j)=0 for all j.
           !That is why the value of a_imp(nstates) is not important.
           ! [To disable this correction, set imp_bounde_collscat to 0]
           !---> Now set hbethe(). This is Nej*[(1/k)ln(1+hj^k) - beta^2] 
           !in the JPP paper, in Eq.(2.31)
           !Note: excit_en_n is norm-ed by me*c^2==restmkev=510.998902d0keV
           hj= p_n*sqrt(gamma(j)-1.d0)/excit_en_n
           hbethe(kstate,j)= z_bound*(0.2*log(1.d0+hj**5) -beta2)
     &                              *imp_bounde_collslow
           !Note: 0.2 is 1/k=1/5 in the paper 
           !(parameter for a smooth transition to p-->0 range)
           !Note: When p is small, hj-->0 and beta-->0,
           !then hbethe-->0 (no additional slowing down
           !because a free electron cannot penetrate electron shells
           !of the partially ionized ion.
           !If we don't use this transitional form, 
           !but use ln(hj)-beta2, this function may become negative
           !at small p. 
           !YuP: I plotted and verified that 0.2*log(1.d0+hj**5) -beta2
           !is always positive. It becomes~0 at p_n<0.02 for Ar0(atom) 
           !(or at p_n<0.10 for Ar[17+])
           ! [To disable this correction, set imp_bounde_collslow to 0]
         enddo ! j
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'set_gscreen_hesslow: kstate, min/max gscreen=',
     &    kstate, MINVAL(gscreen(kstate,:)),MAXVAL(gscreen(kstate,:))
         WRITE(*,*)'set_gscreen_hesslow: kstate, min/max hbethe=',
     &    kstate, MINVAL(hbethe(kstate,:)),MAXVAL(hbethe(kstate,:))
CMPIINSERT_ENDIF_RANK
      enddo ! kstate
            
      return
      end subroutine set_gscreen_hesslow

!=======================================================================
!=======================================================================
      subroutine set_get_ADPAK(kopt,imp_type, 
     &                         temp_Te,dens_nD0,dens_ne,tau_r,
     &                         z1av,z2av) 
      !(INPUT: kopt,imp_type,temp_Te,dens_nD0,dens_ne,tau_r)
      !YuP[2019-08-15]-[2019-08-20]
      !Read impurity emissivity and charge state from POST93 data files.
      !Available data files: 
      ! [should match impurity types in subr. set_gscreen_hesslow]
      !               imp_type: 1->He,2->Be,3->C,4->N,5->Ne,6->Ar,7->Xe,8-W
      ! Oxygen.ntau,   (--- imp_type not set)
      ! Argon.ntau     (corr to imp_type=6), 
      ! Beryllium.ntau (corr to imp_type=2),
      ! Carbon.ntau    (corr to imp_type=3), 
      ! Iron.ntau,     (--- imp_type not set)
      ! Neon.ntau      (corr to imp_type=5), 
      ! Nitrogen.ntau  (corr to imp_type=4)
      ! Comments from such data files:
      !   ADPAK data by Russell Hulse and Doug Post
      !   Rates as a function of electron temperature, neutral fraction
      !   and residence time. May 6, 1993 further revisions will be done.
      !   radiation rates, watts cm^3, and <Z> for electron density =2.0E+14/cm^3
      ! Columns with data:
      !   Te (eV),   n0/ne,   ntau(s cm^-3), Prad(watts cm^3), <Z>, <Z^2>
      ! where n0 is neutral Deuterium density (we also refer to it as nD0).
      ! First three columns make up a 3D grid 
      ! over which the values of Prad,<Z>,<Z^2> are set.
      ! YuP: Observed that each of those grids is uniform in log10 scale.
      !      We will use this fact to quickly determine the nearest 3D node
      !      {jt,jr,jn) in the 3D grid {log10(Te); log10(n0/ne); log10(ntau)}
      !      for a given input point 
      !      {log10(temp_Te), log10(dens_nD0/dens_ne), log10(dens_ne*tau_r)} 
      ! INPUT: kopt=0 Reading data file and setting arrays (for the six columns)
      !        kopt=1 Determine values of <Z> and <Z^2> for given {Te,nD0,ne,tau}
      !        imp_type= impurity type (see above)
      !        temp_Te=  T_e  [keV] Electron temperature
      !        dens_nD0= n_D0 [cm^-3] Density of neutral D
      !        dens_ne=  n_e  [cm^-3] Density of electrons
      !        tau_r=    tau  [sec] Characteristic time of radial decay of T_e
      !           Note from A. Pigarov:
      !           For disruption case, I would set tau(r)=1.e-3 sec. 
      !           For Smith-like run case tau(r)=TauT, 
      !           where TauT is the temperature decay time.
      !           For quasi-stationary plasma it is likely about 
      !           plasma confinement time ~1 s.
      ! OUTPUT:
      !        z1av= <Z> Average charge state (a value between 0 and atomic number)
      !        z2av= <Z^2> Average of Z^2 for given impurity
      ! The data tables are based on Corona equilibrium.

      implicit none
CMPIINSERT_INCLUDE
      integer kopt     ! input: 0 for setup, 1 for getting <Z> and <Z^2>
      integer imp_type ! input: Impurity type
      real*8 temp_Te,dens_nD0,dens_ne,tau_r !INPUT (when kopt=1) [keV,cm^3,sec]
      real*8 z1av,z2av ! OUTPUT (when kopt=1)
      character*64 fname ! local: file with data
      integer atn, atw ! local (just for printout)
      real*8 dlog10_Te, dlog10_nD0ne, dlog10_netau !local
      real*8 wtl,wtu, wrl,wru, wnl,wnu ! local
      real*8 fmmm,f0mm, fm0m,f00m, fmm0,f0m0, fm00,f000 ! local
      integer nt, nr, nn ! local, obtained from data file
      integer jt, jr, jn, jt1, jr1, jn1, jt2, jr2, jn2 ! local
      integer ios,istat,nget ! local
      real*8 accur ! local, accuracy 
      real*8 dens_tau != dens_ne*tau_r    ! local usage
      real*8 r_nD0_ne != dens_nD0/dens_ne ! local usage
      ! To be filled up with data from file and saved into memory:
      real*8, dimension(:,:,:), allocatable ::  tdatm,  rdatm,  ndatm
      real*8, dimension(:,:,:), allocatable ::  emdatm, z1datm, z2datm
      save tdatm,  rdatm,  ndatm
      save emdatm, z1datm, z2datm
      save nt, nr, nn
      
      real*8 Te_min,Te_max, rn0ne_min,rn0ne_max, taun_min,taun_max
      save   Te_min,Te_max, rn0ne_min,rn0ne_max, taun_min,taun_max
      real*8 dlog10_Te_min,   dlog10_Te_max,   delta_log10_Te
      real*8 dlog10_n0ne_min, dlog10_n0ne_max, delta_log10_n0ne
      real*8 dlog10_ntau_min, dlog10_ntau_max, delta_log10_ntau
      save   dlog10_Te_min,   dlog10_Te_max,   delta_log10_Te
      save   dlog10_n0ne_min, dlog10_n0ne_max, delta_log10_n0ne
      save   dlog10_ntau_min, dlog10_ntau_max, delta_log10_ntau
     
      data nget /55/  ! I/O unit for accessing/reading the file
      
      if(kopt.eq.0)then ! Setup arrays, read data into arrays
      !-------------  kopt=0 ----------------------------
      
      ! Selection of file name corresponding to our imp_type
      if(imp_type.eq.1)then
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'set_get_ADPAK: missing data file Helium.ntau'
CMPIINSERT_ENDIF_RANK
        STOP
      endif
      if(imp_type.eq.7)then
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'set_get_ADPAK: missing data file Xeon.ntau'
CMPIINSERT_ENDIF_RANK
        STOP
      endif
      !These are available:
      if(imp_type.eq.2)then
        fname='./ADPAK_data/Beryllium.ntau'
      endif
      if(imp_type.eq.3)then
        fname='./ADPAK_data/Carbon.ntau'
      endif
      if(imp_type.eq.4)then
        fname='./ADPAK_data/Nitrogen.ntau'
      endif
      if(imp_type.eq.5)then
        fname='./ADPAK_data/Neon.ntau'
      endif
      if(imp_type.eq.6)then
        fname='./ADPAK_data/Argon.ntau'
      endif
      
      !Open this file:
      open(unit=nget,file=fname,form='formatted',iostat=ios,
     & status='old')
      if (ios.ne.0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*) "--- data file ADPAK_data/*.ntau not found ---"
         WRITE(*,*) 'In set_get_ADPAK: fname=', fname
CMPIINSERT_ENDIF_RANK
         STOP 'In set_get_ADPAK: data file not found'
      endif

      ! read array dimensions
      read(nget,9000) !First 4 lines in *.ntau file are just a comment
      read(nget,9000)
      read(nget,9000)
      read(nget,9000)
      ! Next 4 lines contain (example for Argon.ntau):
      ! 18   atomic number  (atn) [YuP: specified as integer in the file]
      ! 36   atomic weight  (atw) [YuP: Should be 40?] [as integer, also]
      ! 41  =nt= number of temperature intervals [eV]
      ! 22  =nr= number of D_neutral_density/electron_density intervals  
      ! 22  =nn= number of ntau intervals (n_e*tau_decay phys.quantity) 
      ! Read actual values:      
      read(nget,9001) atn, atw, nt, nr, nn ! 
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'In set_get_ADPAK: atn, atw, nt, nr, nn',
     &                                  atn, atw, nt, nr, nn
CMPIINSERT_ENDIF_RANK
      !Note: atn and atw are not used in this subroutine.

      !allocate storage for arrays
      if (.NOT. ALLOCATED(tdatm)) then
        allocate(tdatm(1:nt,1:nr,1:nn),STAT=istat)
        allocate(rdatm(1:nt,1:nr,1:nn),STAT=istat)
        allocate(ndatm(1:nt,1:nr,1:nn),STAT=istat)
        allocate(emdatm(1:nt,1:nr,1:nn),STAT=istat)
        allocate(z1datm(1:nt,1:nr,1:nn),STAT=istat)
        allocate(z2datm(1:nt,1:nr,1:nn),STAT=istat)
      else ! already allocated (should not happen)
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'In set_get_ADPAK: Attempt to allocate arrays'
        WRITE(*,*)'In set_get_ADPAK: which are already allocated'
CMPIINSERT_ENDIF_RANK
        STOP 'In set_get_ADPAK: allocation problem'
      endif
      
      ! read data arrays
      do jn=1,nn
         do jr=1,nr
            do jt=1,nt
               read(nget,9010) tdatm(jt,jr,jn), rdatm(jt,jr,jn), 
     &                         ndatm(jt,jr,jn), emdatm(jt,jr,jn), 
     &                         z1datm(jt,jr,jn), z2datm(jt,jr,jn)
            enddo
         enddo
      enddo ! jn
      ! The data contains six columns:
      !   Te (eV),   n0/ne,   ntau(s cm^-3), Prad(watts cm^3), <Z>, <Z^2>
      ! where n0 is neutral Deuterium density (==nD0).

      close(nget) ! Close data file
      
 9000 format()
 9001 format(5(1x,i2/))
 9010 format(6(1x,e12.5))
      
      ! Convert Te to keV:
      tdatm=tdatm*1.d-3 ! keV now (CQL3D uses keV; see temp() array)
      
      ! Setup uniform 1D grids in log10 scale
      ! Te grid
      Te_min=tdatm(1, 1, 1) ! MIN value in the table
      Te_max=tdatm(nt,1, 1) ! MAX value in the table
      dlog10_Te_min=   log10(Te_min)
      dlog10_Te_max=   log10(Te_max)
      delta_log10_Te=   (dlog10_Te_max-dlog10_Te_min)/(nt-1) ! [keV]
      ! n0/ne grid
      rn0ne_min=rdatm(1, 1, 1) ! MIN value in the table
      rn0ne_max=rdatm(1,nr, 1) ! MAX value in the table
      dlog10_n0ne_min= log10(rn0ne_min)
      dlog10_n0ne_max= log10(rn0ne_max)
      delta_log10_n0ne= (dlog10_n0ne_max-dlog10_n0ne_min)/(nr-1) ! [-]
      ! ne*tau grid
      taun_min=ndatm(1, 1, 1) ! MIN value in the table
      taun_max=ndatm(1, 1,nn) ! MAX value in the table
      dlog10_ntau_min= log10(taun_min)
      dlog10_ntau_max= log10(taun_max)
      delta_log10_ntau= (dlog10_ntau_max-dlog10_ntau_min)/(nn-1) ![sec/cm^3]
      
      !Just to check:
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'In set_get_ADPAK: MIN/MAX of tdatm [keV]:',
     & MINVAL(tdatm), MAXVAL(tdatm) 
      WRITE(*,*)'In set_get_ADPAK: MIN/MAX of rdatm [n0/ne]:',
     & MINVAL(rdatm), MAXVAL(rdatm) 
      WRITE(*,*)'In set_get_ADPAK: MIN/MAX of ndatm [ne*tau, sec/cm3]:',
     & MINVAL(ndatm), MAXVAL(ndatm) 
      WRITE(*,*)'In set_get_ADPAK: MIN/MAX of z1datm:',
     & MINVAL(z1datm), MAXVAL(z1datm) 
CMPIINSERT_ENDIF_RANK
     
      else  ! kopt>0
      !------------------------- kopt>0 ----------------------------
      ! For a given  Te(r,t), nD0(r,t), ne(r,t), tau(r)
      ! [temp_Te, dens_nD0, dens_ne, tau_r in argument list] 
      ! find the nearest point 
      ! in the table {tdatm, rdatm, ndatm}, and then find 
      ! corresponding values of <Z> and <Z^2>
      ! [z1av and z2av in argument list]
      
      ! Using uniform grids in {log10(Te); log10(n0/ne); log10(ntau)}

      dens_tau= dens_ne*tau_r    ! Only This combination is used below
      r_nD0_ne= dens_nD0/dens_ne ! Only This combination is used below
      
      dlog10_Te= log10(temp_Te) !Our input value
      ! If temp_Te is below lowest value in the table, use the lowest value:
      if(dlog10_Te.lt.dlog10_Te_min)then
         dlog10_Te=   dlog10_Te_min
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
     &  'set_get_ADPAK/WARNING: input value temp_Te<MIN(tdatm) in table'
CMPIINSERT_ENDIF_RANK
      endif
      ! If temp_Te is above largest value in the table, use the largest value:
      if(dlog10_Te.gt.dlog10_Te_max)then
         dlog10_Te=   dlog10_Te_max
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
     &  'set_get_ADPAK/WARNING: input value temp_Te>MAX(tdatm) in table'
CMPIINSERT_ENDIF_RANK
      endif
      jt1= INT( (dlog10_Te-dlog10_Te_min)/delta_log10_Te ) +1
      jt1=max(1,jt1)  !Not lower than 1
      jt1=min(jt1,nt) !Not exceeding nt
      jt2=jt1+1
      jt2=min(jt2,nt) !Not exceeding nt
      ! temp_Te is between tdatm(jt1,*,*) and tdatm(jt2,*,*)
      wtu= (dlog10_Te-dlog10_Te_min)/delta_log10_Te -(jt1-1) !upper
      wtl= 1.d0-wtu ! lower weight factor, for interpolation
      !----------
      if(r_nD0_ne.ge.rn0ne_min) then 
        dlog10_nD0ne= log10(r_nD0_ne) 
      else ! r_nD0_ne can be 0., or very small
        dlog10_nD0ne= dlog10_n0ne_min !Set to lowest value in the table 
        ! If nD0/ne is lower than lowest value in table, use lowest value:
        !     WRITE(*,*)
        ! &'set_get_ADPAK/WARNING: input dens_nD0/dens_ne < lowest in table'
      ! In ADPAK tables, the values of nD0/ne are in the range 1e-7...1.0,
      ! and the results (<Z> values) are almost same at small
      ! values of nD0/ne. In other words, when nD0~0, 
      ! we simply use the lowest values of nD0/ne that are available. 
      ! No need to print the message above: nD0=0 may happen frequently.    
      endif
      ! If nD0/ne is above largest value in the table, use the largest value:
      if(dlog10_nD0ne.gt.dlog10_n0ne_max)then
         dlog10_nD0ne=   dlog10_n0ne_max
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
     &'set_get_ADPAK/WARNING: input dens_nD0/dens_ne > largest in table'
CMPIINSERT_ENDIF_RANK
      endif
      jr1= INT((dlog10_nD0ne-dlog10_n0ne_min)/delta_log10_n0ne) +1
      jr1=max(1,jr1)  !Not lower than 1
      jr1=min(jr1,nr) !Not exceeding nr
      jr2=jr1+1
      jr2=min(jr2,nr) !Not exceeding nr
      ! dens_nD0/dens_ne is between rdatm(*,jr1,*) and rdatm(*,jr2,*)
      wru= (dlog10_nD0ne-dlog10_n0ne_min)/delta_log10_n0ne -(jr1-1) !upper
      wrl= 1.d0-wru ! lower weight factor
      !----------
      dlog10_netau= log10(dens_tau)
      ! If ne*tau_r is lower than lowest value in table, use lowest value:
      if(dlog10_netau.lt.dlog10_ntau_min)then
         dlog10_netau=   dlog10_ntau_min
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
     &'set_get_ADPAK/WARNING: input dens_ne*tau_r < lowest in table'
CMPIINSERT_ENDIF_RANK
      endif
      ! If ne*tau_r is above largest value in the table, use the largest value:
      if(dlog10_netau.gt.dlog10_ntau_max)then
         dlog10_netau=   dlog10_ntau_max
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
     &'set_get_ADPAK/WARNING: input dens_ne*tau_r > largest in table'
CMPIINSERT_ENDIF_RANK
      endif
      jn1= INT((dlog10_netau-dlog10_ntau_min)/delta_log10_ntau) +1
      jn1=max(1,jn1)  !Not lower than 1
      jn1=min(jn1,nn) !Not exceeding nn
      jn2=jn1+1
      jn2=min(jn2,nn) !Not exceeding nn
      ! dens_ne*tau_r is between ndatm(*,*,jn1) and ndatm(*,*,jn2)
      wnu= (dlog10_netau-dlog10_ntau_min)/delta_log10_ntau -(jn1-1) !upper
      wnl= 1.d0-wnu ! lower weight factor, for interpolation
      
      !Verify that the input point is indeed between corresponding 
      ! grid points jt1 and jt2, 
      ! and print a warning message if not compliant.
      !This verification is simply a safeguard against case when
      ! the Te grid in the data file is not uniform in log10 scale.
      !A minor problem in such verification is that the indexes jt1 and jt2
      ! are found in log10 scale (see above), but the verification below
      ! is done in original scale. Because of numerical accuracy,
      ! the temp_Te point can be within [log10(); log10()] range
      ! corresponding to [jt1;jt2] cell, but it can slightly outside
      ! of such range in the original scale. So, we allow the point 
      ! to be slightly outside of the range by setting an accur value: 
      accur= 1.d-3*temp_Te ! Accuracy (the point could be slightly outside
      ! of range just because of accuracy. Allow this - don't stop).
      !Perform such verification only when temp_Te is 
      ! within [MIN;MAX] range of the data table (grid range).
      ! If it is outside of the table range, use the 1st or last point
      ! in the table for evaluating the output values.
      if(temp_Te.ge.Te_min .and. temp_Te.le.Te_max)then
        ! It may happen that temp_Te is smaller than the lowest value
        ! of Te in the table (in tdatm), or larger than the largest value.
        ! However, let's allow such cases. In this case, when temp_Te>MAX(tdatm),
        ! we use jt1=nt and jt2=nt, i.e. we take the last point in the table
        ! to determine <Z> and <Z^2>.  Similarly - when temp_Te<MIN(tdatm).
        ! So, don't stop the run, just print a warning message (done above).
        ! In the rest of cases ( Te_min<temp_Te<Te_max ) check that 
        ! the temp_Te point is really within the proper grid cell [jt1,jt2].
        ! If not, stop the run: 
      if( (temp_Te-tdatm(jt1,jr1,jn1).lt.-accur) .or.
     &    (temp_Te-tdatm(jt2,jr2,jn2).gt. accur)     ) then
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)
     &  'set_get_ADPAK/WARNING: temp_Te[keV],tdatm(jt1),tdatm(jt2)',
     &  temp_Te,tdatm(jt1,jr1,jn1),tdatm(jt2,jr2,jn2), jt1,jt2
CMPIINSERT_ENDIF_RANK
        STOP 'temp_Te is outside of [jt1;jt2] cell'
      endif
      endif
      
      ! Similar check for nD0/ne value
      accur= 1.d-3*r_nD0_ne ! Accuracy
      if(r_nD0_ne.ge.rn0ne_min .and. 
     &   r_nD0_ne.le.rn0ne_max       )then
      if( (r_nD0_ne-rdatm(jt1,jr1,jn1).lt.-accur) .or.
     &    (r_nD0_ne-rdatm(jt2,jr2,jn2).gt. accur)     ) then
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)
     &  'set_get_ADPAK/WARNING: dens_nD0/dens_ne,rdatm(jr1),rdatm(jr2)',
     &  r_nD0_ne,rdatm(jt1,jr1,jn1),rdatm(jt2,jr2,jn2), jr1,jr2
CMPIINSERT_ENDIF_RANK
        STOP 'dens_nD0/dens_ne is outside of [jr1;jr2] cell'
      endif
      endif

      ! Similar check for ne*tau value
      accur= 1.d-3*dens_tau ! Accuracy
      if(dens_tau.ge.taun_min .and. 
     &   dens_tau.le.taun_max       )then
      if( (dens_tau-ndatm(jt1,jr1,jn1).lt.-accur) .or.
     &    (dens_tau-ndatm(jt2,jr2,jn2).gt. accur)     ) then
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,'(a,3e15.8,3i6)')
     &  'set_get_ADPAK/WARNING: dens_tau,ndatm(jn1),ndatm(jn2)',
     &  dens_tau,ndatm(jt1,jr1,jn1),ndatm(jt2,jr2,jn2), jn1,jn2
CMPIINSERT_ENDIF_RANK
        STOP 'dens_tau is outside of [jn1;jn2] cell'
      endif
      endif

      !Make linear interpolation over each axis in 3D 
      ! For <Z>
      fmmm= z1datm(jt1,jr1,jn1)
      f0mm= z1datm(jt2,jr1,jn1)
      fm0m= z1datm(jt1,jr2,jn1)
      f00m= z1datm(jt2,jr2,jn1)
      fmm0= z1datm(jt1,jr1,jn2)
      f0m0= z1datm(jt2,jr1,jn2)
      fm00= z1datm(jt1,jr2,jn2)
      f000= z1datm(jt2,jr2,jn2)
      z1av=  wnl*
     &    (  wtl*wrl*(fmmm)
     &      +wtu*wrl*(f0mm)
     &      +wtl*wru*(fm0m)
     &      +wtu*wru*(f00m) )
     &      +wnu*
     &    (  wtl*wrl*(fmm0)
     &      +wtu*wrl*(f0m0)
     &      +wtl*wru*(fm00)
     &      +wtu*wru*(f000) )
      
      ! And now for <Z^2>
      fmmm= z2datm(jt1,jr1,jn1)
      f0mm= z2datm(jt2,jr1,jn1)
      fm0m= z2datm(jt1,jr2,jn1)
      f00m= z2datm(jt2,jr2,jn1)
      fmm0= z2datm(jt1,jr1,jn2)
      f0m0= z2datm(jt2,jr1,jn2)
      fm00= z2datm(jt1,jr2,jn2)
      f000= z2datm(jt2,jr2,jn2)
      z2av=  wnl*
     &    (  wtl*wrl*(fmmm)
     &      +wtu*wrl*(f0mm)
     &      +wtl*wru*(fm0m)
     &      +wtu*wru*(f00m) )
     &      +wnu*
     &    (  wtl*wrl*(fmm0)
     &      +wtu*wrl*(f0m0)
     &      +wtl*wru*(fm00)
     &      +wtu*wru*(f000) )
      
      !Could add similarly for Prad (emdatm); maybe later; not used presently.
      
      !Could use a better interpolation scheme, but probably it is not important.
      
      endif ! kopt

      return
      end subroutine set_get_ADPAK
      
      
      
!=======================================================================
!=======================================================================
      subroutine get_distr_charge_states(Zatom,z1av,z2av, fz) 
      !          INPUT: Zatom,z1av,z2av   OUTPUT: fz(0:Zatom)
      !YuP[2019-08-15]-[2019-08-20]
      !To be used immediately after call to sub.set_get_ADPAK(kopt=1,...)
      implicit none
CMPIINSERT_INCLUDE
      integer Zatom ! INPUT: Atomic charge number
      real*8 z1av,z2av !INPUT: <Z>,<Z^2>  -- average charge and charge^2
      real*8 fz(0:Zatom) !OUTPUT : Distribution over charge states 
                         !with properties SUM(fz(z))=1, 
                         ! and also SUM(fz(z)*z)=<Z> [with some accuracy]
      integer iz ! local: index over charge states
      real*8  cn ! local: normalization constant
      real*8  wl,wu ! local: weight factors for lin.interpolation
      real*8 sum_fz_Z1, sum_fz_Z2 ! local, for printout
      
      !is=0 ! array index counter, to be incremented in loop
      ! Form fz(z,<z>) as a function of z (parametrically dep. on z1av==<z>)
      ! Loop over iz=0:Zatom   ! Array of charge states
      do iz=0,Zatom  ! Neutral and Ionized states are included here
         !is=is+1  
         if (iz<=z1av-3) then
           fz(iz)= 0.1d0* 3**(iz-(z1av-3))  !--> 0.1 at iz=z1av-3
         elseif (iz<z1av-2) then ! z1av-3 < iz < z1av-2
           ! Interpolate between two nearest nodes:
           wl= (z1av-2) - iz ! weight factor for left (lower) node
           wu= iz - (z1av-3) ! weight factor for upper node
           fz(iz)= 0.1d0*wl + 0.2d0*wu  
         elseif (iz<z1av-1) then ! z1av-2 < iz < z1av-1
           wl= (z1av-1) - iz ! weight factor for left (lower) node
           wu= iz - (z1av-2) ! weight factor for upper node
           fz(iz)= 0.2d0*wl + 0.5d0*wu  
         elseif (iz<z1av)   then ! z1av-1 < iz < z1av
           wl= (z1av-0) - iz ! weight factor for left (lower) node
           wu= iz - (z1av-1) ! weight factor for upper node
           fz(iz)= 0.5d0*wl + 1.0d0*wu  
!         elseif (iz==z1av)  then !          iz=z1av
!           fz(iz)= 1.d0  
         elseif (iz<z1av+1) then ! z1av   < iz < z1av+1
           wl= (z1av+1) - iz ! weight factor for left (lower) node
           wu= iz - (z1av+0) ! weight factor for upper node
           fz(iz)= 1.0d0*wl + 0.5d0*wu  
         elseif (iz<z1av+2) then ! z1av+1 < iz < z1av+2
           wl= (z1av+2) - iz ! weight factor for left (lower) node
           wu= iz - (z1av+1) ! weight factor for upper node
           fz(iz)= 0.5d0*wl + 0.2d0*wu  
         elseif (iz<z1av+3) then ! z1av+2 < iz < z1av+3
           wl= (z1av+3) - iz ! weight factor for left (lower) node
           wu= iz - (z1av+2) ! weight factor for upper node
           fz(iz)= 0.2d0*wl + 0.1d0*wu  
         elseif (iz>=z1av+3)then
           fz(iz)= 0.1d0* 3**(-(iz-z1av-3)) !--> 0.1 at iz=z1av+3   
         else
CMPIINSERT_IF_RANK_EQ_0
           WRITE(*,*)'get_distr_charge_states: Invalid iz. Stopping'
CMPIINSERT_ENDIF_RANK
           STOP
         endif
      enddo ! iz

      cn= 1.d0/SUM(fz(0:Zatom)) ! normalization factor for fz().
      fz(:)= cn*fz(:) !Distribution over charge states, as a func. of Z
      ! So now we have fz(z,<z>), such that sum(fz)=1.0 ;
      ! The problem is that  we need to satisfy
      !    sum(fz*z) = z1av  (= <Z>)
      ! and also
      !    sum(fz*z*z) = z2av  (= <Z^2>)
      ! This is not guaranteed by the present crude model,
      ! but the tests show that it is close.
      
!      ! For printout/verification:
!      sum_fz_Z1=0.d0
!      sum_fz_Z2=0.d0
!      do iz=0,Zatom
!        write(*,*) iz,fz(iz)
!        sum_fz_Z1= sum_fz_Z1 +fz(iz)*iz
!        sum_fz_Z2= sum_fz_Z2 +fz(iz)*iz*iz
!      enddo
!      write(*,*)'% Above: z, fz(z)  over all charge states z'
!      write(*,*)'% z1av,z2av=',z1av,z2av
!      write(*,*)'% sum(fz*z)=', sum_fz_Z1
!      write(*,*)'% sum(fz*z^2)=', sum_fz_Z2

      return
      end subroutine get_distr_charge_states
      
      
!=======================================================================
!=======================================================================
      subroutine get_dens_nD0_ADPAK(model_dens_nD0,dens_nD0_b,dens_nD0_l
     &              ,rho, radmin,
     &                dens_nD0) 
      !          INPUT:  model_dens_nD0,dens_nD0_b,dens_nD0_l, rho,radmin
      !          OUTPUT: dens_nD0
      !          Units: 1/cm^3, cm
      !YuP[2019-09-11]
      !To be used before call to sub. set_get_ADPAK(kopt=1,...)
      !This subr. calculates neutral density of D0, 
      ! needed as an input for ADPAK tables, 
      ! where it enters through the ratio nD0/ne.
      ! In ADPAK tables, the values of nD0/ne are in the range 1e-7...1.0,
      ! and the results (<Z> values) are almost same at small
      ! values of nD0/ne. In other words, when nD0~0, 
      ! we simply use the lowest values of nD0/ne that are available.
      implicit none
CMPIINSERT_INCLUDE
      integer model_dens_nD0 ! Only one model so far for neutral D0. 
      real*8 dens_nD0_b ! 1/cm^3 ! Edge density of neutral D0
      real*8 dens_nD0_l !cm! Scale length, in  dens_nD0_b*exp[(rho-1)*radmin/dens_nD0_l]
      real*8 rho    ! normalized to [0;1] range
      real*8 radmin ! cm
      real*8 dens_nD0
      
      SELECT CASE (model_dens_nD0)
      CASE(1) ! 
         dens_nD0= dens_nD0_b*exp((rho-1.d0)*radmin/dens_nD0_l)
         ! Could become very small~0 at small rho
      CASE DEFAULT
         dens_nD0=0.d0 ! just zero density of D0
      END SELECT
      ! Other (future) models: make it as a func. of time? 
      
      return
      end subroutine get_dens_nD0_ADPAK

!=======================================================================
!=======================================================================
      subroutine set_get_pellet(kopt, timet,
     &           lrz, Raxis, rpcon, rmcon, dvol, Te, ne,
     &           pellet_V, pellet_M0, pellet_Rstart, pellet_tstart,
     &           pellet_rcloud1, pellet_rcloud2, pellet_rp0,
     &           pellet_pn, pellet_pt, pellet_pm,
     &           ipellet_method, pellet_fract_rem, 
     &           ipellet_iter_max, pellet_iter_accur, pellet_Cablation,
     &                         Gablation,dMpellet_sum,dMdv_sum) 
      !YuP[2019-09-15]
      implicit none
CMPIINSERT_INCLUDE
      integer kopt !INPUT: kopt=0 for setup; kopt=1 for pellet propagation
      real*8 timet ! sec ! INPUT: physical time since start of simulation
      real*8 Raxis ! cm  ! INPUT
      integer lrz  !INPUT: size of grids below
      real*8 rpcon(lrz),rmcon(lrz),dvol(lrz),Te(lrz),ne(lrz) !cm,cm^3,keV !INPUT
      
      integer ipellet_method, ipellet_iter_max !INPUT, set in namelist
      !INPUT scalars, set in namelist:
      real*8     pellet_V, pellet_M0, pellet_Rstart, pellet_tstart,
     &           pellet_rcloud1, pellet_rcloud2, pellet_rp0,
     &           pellet_pn, pellet_pt, pellet_pm,
     &           pellet_fract_rem, 
     &           pellet_iter_accur, pellet_Cablation
      !(pellet_Cablation can also be OUTPUT, if ipellet_method=1)

      integer irold,irnew ! local: range of radial indexes
      real*8 Rold,Rnew !local: position of pellet at last and present step
      real*8 dr ! local: distance in R travelled by pellet since last step
      
      !OUTPUT (when kopt=1): array for storing dM/dvol mass density
      ! as a func. of flux surface index. 
      ! It is recalculated each time when this subr. is called (with kopt=1)
      real*8 dMdv_sum(lrz) !gram/cm^3! To be saved into dMpellet_dvol_sum()
      !To calculate particle density, use  convert= Avogadro/atw 
      ! where atw is the atomic weight (40 g/mol for Argon)
      !OUTPUT (when kopt=1):
      real*8 dMpellet_sum ! accumulated from one call to another
      ! dMpellet_sum ! Total ablated material [gram], scalar.
      ! The remaining mass of pellet is Mpellet_rem= pellet_M0-dMpellet_sum 
      real*8 Gablation !ablation rate along pellet trajectory (for plots)
      
      !Setup at kopt=0 call,  These full radial grids:
      real*8,dimension(:),allocatable :: rfull,drfull,dvol_full !(1:2*lrz)
      real*8,dimension(:),allocatable :: drfull_p, drfull_m !(1:2*lrz)
      !and other arrays over rfull() grid:
      real*8,dimension(:),allocatable :: wk !(1:2*lrz) local (weight factors)
      real*8,dimension(:),allocatable :: dMpellet,dMpellet_distr !local
      real*8 G_ref !local [could save and compare with Gablation]
      
      integer lrfull !local, lrfull= 2*lrz
      integer istat !local
      integer ir,ir0,ir1,ir2, irm,irp,irfull, irr ! local
      integer lr,lr0,lr1,lr2, iter, icloud ! local
      integer istop !local
      real*8 drp_lph(lrz),drp_lmh(lrz),drm_lph(lrz),drm_lmh(lrz) !local
      real*8 drp(lrz),drm(lrz) !local
      real*8 Cablation0, pellet_rem_iter, sum_wk ! local
      real*8 rloc,drii,dM0,rp,Mpellet_rem,dM_p,dM_m,Rpellet ! local
      real*8 den_loc, tem_loc, wght0,wght1
      
      save lrfull, rfull, drfull, drfull_p, drfull_m, dvol_full !(2*lrz)
      save wk, dMpellet, G_ref 
      save dMpellet_distr !(1:2*lrz) Accumulated from one call to another
      save Rold,Rnew   ! local
      save irold,irnew ! local: range of radial indexes in rfull() grid
                       ! corresponding the distance travelled by pellet
                       ! during time interval from timet at previous call
                       ! until this call.
      
! INPUT:
!   kopt=0 is to setup grids, etc., 
!          also find pellet_Cablation, if ipellet_method=1
!   kopt=1 is to calculate the profile of mass density 
!          of ablated material dMdv_sum [gram/cm**3],
!          accumulated during flight of pellet from launch time 
!          'pellet_tstart' up to the instant 'timet' 
!          when the subroutine is called.
!          The pellet is propagating towards magnetic axis.
!          It starts at R=pellet_Rstart at t=pellet_tstart
!          so that the position of pellet at t=timet is
!          Rpellet= pellet_Rstart - pellet_V*(timet-pellet_tstart)
! -----------------------
! INPUT from namelist (typical values are given as an example):
!   pellet_Rstart=230. ![cm] Major radius where pellet is launched.
!          Suggestion: Set it to rpcon(lrz), or R_LCFS radius.
!   pellet_tstart=0.0 ![sec] Instant when pellet is launched.
!          Not necessarily 0.0, but should be .ge.0.
!   pellet_V=30000.0 ![cm/s] Pellet speed. Typically 10000--900000 cm/s.
!          Assumed constant all the way through plasma.
!          Assumed that pellet travels along equatorial
!          plane, going through magnetic axis.
!   pellet_M0=30.d-3 ![gram] Initial mass of pellet (at t=pellet_tstart)
!          If pellet is large, it can make to the inner border of plasma.

! INPUTs from namelist related to pellet size and ablation cloud.
!          For distributing the ablated mass among several flux surfaces,
!          assume that the ablation cloud is 5--8 times
!          larger than the pellet itself.
!          Allow for assymmetry between leading (front) 
!          and trailing (back) side of the cloud.
!   pellet_rp0=    0.5 !cm! Pellet radius at t=pellet_tstart (plasma edge).
!   pellet_rcloud1=4.0 !cm! Radius of ablation cloud, leading (front) side.
!   pellet_rcloud2=4.0 !cm! Radius of ablation cloud, trailing (back) side. 
!          Typically pellet_rp0== rp(0) = 0.2--0.5 cm.
!          Recommended: pellet_rcloud ~(5--8)*pellet_rp0
!          Pellet radius is reduced during propagation 
!          (as the mass is reduced).
!          However, in present model, rcloud is not changed, 
!          so the cloud size remains as described by pellet_rcloud1,2 above.

! INPUTs related to description of ablation rate.
!          Assume that the ablation rate of pellet is proportional to local 
!          ne^pn * Te^pt (electron T and density in some powers),
!          and proportional to remaining_mass/pellet_M0 in some power "pm".
!          So that the local ablation rate is 
!          G[gram/s]= Cablation* ne[cm-3]^pn *Te[keV]^pt *(Mpellet(t)/Mpellet(0))^pm
!          See REFS: "2019-03-15-Friday Science Meeting-Jie Zhang.pdf"
! INPUT values for those powers:
!   pellet_pn=1./3.  power "pn" in the above Eqn. for G.
!                    REFS: should be 1/3
!   pellet_pt=5./3.  power "pt" in the above Eqn. 
!                    REFS: should be 11/6, or 5/3 
!   pellet_pm=4./9.  power "pm" in the above Eqn. 
!                    REFS: should be 4/9 (so that rp^(4/3))
!          where Mpellet(t)==Mpellet_rem is the remaining mass
!          at given radial position R(t).
!          Note that  (Mpellet(t)/Mpellet(0))^pm ~~  (rp(t)/rp(0))^(3*pm)
!          For example, when pm=2/3, we get  G~~ rp^2,
!          which means - proportional to surface area of the pellet
!          (S_pellet= 4*pi*rp^2).
!          Why in REFS they use pm=4/9, and not 2/3 ?
!          The value of Cablation=="pellet_Cablation" is either calculated 
!          during kopt=0 call of this subroutine, 
!          or set as a namelist value, see below.
! 
! INPUTs related to calculation of pellet_Cablation value.
!-- ipellet_method=1  Iterative procedure,
!       to find such value of pellet_Cablation which yields the value
!       of fraction of pellet remained at magnetic axis.
!       [Note: when pm=0, i.e. pellet_pm=0, this method works in one iteration]
!   pellet_fract_rem=0.50 !Fraction of remained mass when pellet reaches magn.axis,
!       i.e. it is   pellet_fract_rem=(pellet_M0-dMpellet_sum(t_axis))/pellet_M0
!       where dMpellet_sum(t_axis) is the total ablated mass during the flight
!       of the pellet from plasma edge to magnetic axis.
!       The value of pellet_Cablation will be found from iterations.
!       For this method, also specify:
!   ipellet_iter_max= 50 ! Max number of iterations
!   pellet_iter_accur= 1e-2  !Relative error (accuracy) achieved in iterations,
!       to be compared with |pellet_fract_rem-pellet_rem_iter|/pellet_fract_rem
!
!-- ipellet_method=2  Use the estimate (see below) for 
!       pellet_Cablation  coefficient, and let the pellet propagate
!       at such value.
!       The remaining mass of pellet when it reaches the magnetic axis
!       depends on the value of pellet_Cablation 
!       and on the initial mass pellet_M0.
!       The estimate is done from
!       pellet_Cablation= 39*(0.5)^pt *(1e-14)^pn *(pellet_rp0/0.2)^(3*pm) 
!       This is to make it consistent with expression given in
!       REFS: See "2019-03-15-Friday Science Meeting-Jie Zhang.pdf"
!       Units for pellet_Cablation : (gram/s) / (cm^-3)^pn / (keV^pt) 
!-- ipellet_method=3  Specify the value of pellet_Cablation in namelist.
!--------------------------------------------------------------------------
! INPUTs related to plasma profiles, from general CQL3D setup:
! ne(1:lrz) and Te(1:lrz) profiles, as a func. of rho (rya array)
!       Use ne(1:lrz)= reden(kelecg,1:lrz) ![cm^-3] , 
!       Te(1:lrz)= temp(kelecg,1:lrz) ![keV]
!       temp() and reden() are updated in profiles.f 
!       (when using "spline-t" or "prbola-t" settings).
! rpcon(1:lrz)= [cm] array of major radii going from near-magn.axis to OUTboard.
! rmcon(1:lrz)= [cm] array of major radii going from near-magn.axis to INboard.
!       Note: In CQL3D, arrays (rpcon(lrza) and rmcon(lrza)) are defined,
!       and they can be non-uniform, and having different spacing 
!       (because of Shafranov shift).
! dvol(1:lrz)= [cm^3] Volume around each flux surface, 
!       positioned at each given rya(lr);  the corresponding flux surface 
!       goes through rpcon(lr) and rmcon(lr) coordinates.
!       Note: in CQL3D, dvol(lr) array is calculated for any shape surfaces.
! Raxis= rmag [cm] Major radius at magnetic axis
!       (rmag is available through comm.h)
! timet= [sec] Physical time (stored in comm.h)
!-----------------------------------------------------------------------

      if(kopt.eq.0)then !-----------------------------------------------

        lrfull=2*lrz  ! size of rfull grid
        
        !allocate storage for arrays
        if (.NOT. ALLOCATED(rfull)) then
          allocate(rfull(1:lrfull),STAT=istat)
          allocate(drfull(1:lrfull),STAT=istat)
          allocate(drfull_p(1:lrfull),STAT=istat)
          allocate(drfull_m(1:lrfull),STAT=istat)
          allocate(dvol_full(1:lrfull),STAT=istat)
          allocate(wk(1:lrfull),STAT=istat)
          allocate(dMpellet(1:lrfull),STAT=istat)
          allocate(dMpellet_distr(1:lrfull),STAT=istat)
        else ! already allocated (should not happen)
          WRITE(*,*)'In set_get_pellet: Attempt to allocate arrays'
          WRITE(*,*)'In set_get_pellet: which are already allocated'
          STOP 'In set_get_pellet: allocation problem'
        endif

        rfull(1:lrfull)=  0.d0  ! Initialize 
        drfull(1:lrfull)= 0.d0  ! Initialize
        drfull_p(1:lrfull)= 0.d0  ! Initialize
        drfull_m(1:lrfull)= 0.d0  ! Initialize
        dvol_full(1:lrfull)= 0.d0  ! Initialize
        wk(1:lrfull)=0.d0 !Initialize working array for weight factors.
        dMpellet(1:lrfull)=0.d0  !Initialize array for dM ablated at each ir
        dMpellet_distr(1:lrfull)=0.d0 !Initialize array for accumulation
                             ! of mass from ablation clouds 
                             !(a cloud at each ir will contribute 
                             ! to several neighboring nodes) 

        Gablation=0.d0 ! Initialize [gram/sec]
        G_ref=0.d0  ! Initialize [gram/sec]

        ! Initialize array for storing dM/dvol mass density
        dMdv_sum(1:lrz)= 0.d0  

        ! ===> Setup some grids, for convenience.
        ! ---> Define array of spacings for outboard R grid:
        do lr=2,lrz
           drp_lmh(lr)= rpcon(lr)-rpcon(lr-1) ! dRp_lmh== dRp(lr-0.5)
        enddo
        drp_lmh(1)=   rpcon(1)-Raxis !first point,  dRp(0.5) 
        do lr=1,lrz-1
           drp_lph(lr)= rpcon(lr+1)-rpcon(lr) ! dRp_lph== dRp(lr+0.5)
        enddo
        drp_lph(lrz)= drp_lph(lrz-1) !last point, dRp(lrz+0.5)
        !Array of spacings for outboard R grid:
        drp(1:lrz)= (drp_lmh(1:lrz)+drp_lph(1:lrz))/2.0  
        ! But for the 1st point, the spacing should cover all region
        ! from Raxis to Rp(1+0.5):
        drp(1)= rpcon(1)+0.5*drp_lph(1) - Raxis  
        ! ---> Define array of spacings for inboard R grid:
        do lr=2,lrz
           drm_lmh(lr)= rmcon(lr)-rmcon(lr-1) ! dRm_lmh== dRm(lr-0.5)
        enddo
        drm_lmh(1)=   rmcon(1)-Raxis !first point,  dRm(0.5) 
        do lr=1,lrz-1
           drm_lph(lr)= rmcon(lr+1)-rmcon(lr) ! dRm_lph== dRm(lr+0.5)
        enddo
        drm_lph(lrz)= drm_lph(lrz-1) !last point, dRm(lrz+0.5)
        !Array of spacings for inboard R grid:
        drm(1:lrz)= (drm_lmh(1:lrz)+drm_lph(1:lrz))/2.0  
        ! But for the 1st point, the spacing should cover all region
        ! from Raxis to Rm(1+0.5):
        drm(1)= rmcon(1)+0.5*drm_lph(1) - Raxis  
        ! Note: those drm are negative !

        ! Also, combine two radial grids,  size is lrfull=2*lrz
        do ir=1,lrz
           ir1= lrz-ir+1   ! index from rpcon grid, running backwards
           rfull(ir)= rpcon(ir1)   ! From rpcon(lrz) towards rpcon(1)
           rfull(ir+lrz)= rmcon(ir)   ! From rmcon(1) to rmcon(lrz)
           ! The combined grid runs continuously from outer edge
           ! to inner plasma edge.
           ! Note that rfull(ir)  and rfull (2*lrz-ir+1) correspond to
           ! same flux surface (outer and inner radial points)
           ! Similarly - combine arrays for grid spacings.
           ! We set two half-grid spacings: 
           ! drfull_p is at ir+0.5 index point,
           ! so that rfull(ir)-drfull_p(ir) < rfull(ir) ;
           ! drfull_m is at ir-0.5 index point,
           ! so that rfull(ir)+drfull_m(ir) > rfull(ir)
           drfull_p(ir)= 0.5*drp_lmh(ir1)
           drfull_m(ir)= 0.5*drp_lph(ir1)
           if(ir.eq.lrz)then
           drfull_p(ir)= rfull(ir)-Raxis
           endif
           drfull(ir)= drfull_p(ir)+drfull_m(ir)
           drfull_p(ir+lrz)= -0.5*drm_lph(ir) ! recall that drm is negative
           drfull_m(ir+lrz)= -0.5*drm_lmh(ir) ! recall that drm is negative
           if(ir.eq.1)then
           drfull_m(ir+lrz)= Raxis-rfull(ir+lrz)
           endif
           drfull(ir+lrz)= drfull_p(ir+lrz)+drfull_m(ir+lrz)
           !Note that drfull_p and drfull_m are positive numbers.
           !Also set Volume array, corr. to rfull coordinate:
           dvol_full(ir) = dvol(ir1)
           dvol_full(ir+lrz)= dvol(lrz-ir1+1)  
        enddo ! ir
        !Verified that rfull(ir) is in [rfull-drp; rfull+drp] interval,
        !and those intervals follow each other without gaps:
        do ir=1,lrfull 
          write(*,*)'rfull-drp,rfull,rfull+drp=',
     &    ir, rfull(ir)-drfull_p(ir), rfull(ir), rfull(ir)+drfull_m(ir)
        enddo
        

        ! Estimate the value of  pellet_Cablation  from
        !  G= pellet_Cablation*nemax[cm-3]^pn*Temax[keV]^pt *(Mpellet(t)/pellet_M0)^pm =
        !     = 7(gram/s)  [This value is given by Paul Parks for DIII-D]
        ! pellet_Cablation = 7.0 / ( max(ne)^pn *max(Te)^pt *(pellet_M0/pellet_M0)^pm )  

        ! Alternatively, use this equation 
        ! G_ref= 39.0 * (Te/2)^(5/3) * (ne/1e14)^(1/3) * (rp/0.2)^(4/3)   ! gram/s
        !               where Te[keV], ne[cm^-3], rp[cm]
        ! In terms of our pt,pn,pm powers:
        ! G_ref= 39.0 * (Te(lrz)/2)^pt * (ne(lrz)/1e14)^pn * (pellet_rp0/0.2)^(3*pm) 
        ! Then 
        ! pellet_Cablation= G_ref / (ne(lrz)^pn * Te(lrz)^pt) =  
        !                   =  39.0 * (1/2)^pt * (1/1e14)^pn * (pellet_rp0/0.2)^(3*pm) 
        ! pellet_Cablation= 39*(0.5)^pellet_pt *(1e-14)^pellet_pn *(pellet_rp0/0.2)^(3*pellet_pm)
        ! Units for pellet_Cablation : (gram/s) / (cm^-3)^pn / (keV^pt) 

        !---  ipellet_method=1 -------------------------------------------------:
        if (ipellet_method.eq.1) then
        
          ! Iterative procedure [works in one iteration when pm=0],
          !   to find such value of pellet_Cablation which provides the target value
          !   of pellet_fract_rem at magnetic axis.
          ! Make initial guess (start with a very small pellet_Cablation - 
          !   it should give a little ablation of pellet) :
          Cablation0= 39.0* 0.5**pellet_pt *
     &                  (1d-14)**pellet_pn *
     &                  (pellet_rp0/0.2)**(3*pellet_pm) 
          pellet_Cablation= 0.00001*Cablation0  ! Initial small value
          ! Tests: the number of iterations has almost no dependence on
          ! initial guess for pellet_Cablation.  But it should not be 0.
          iter=0 ! counter of iterations
          pellet_rem_iter=1e3 ! Just any large number. Will be found
          do while ( (iter.le.ipellet_iter_max) .and. 
     &             (abs(pellet_fract_rem-pellet_rem_iter).gt.
     &                  pellet_iter_accur*pellet_fract_rem    )   )
           iter=iter+1  ! Counter of iterations
           dMpellet_sum=0.d0 !To accumulate the ablated material (integral)
           do ir=1,lrz !Over rfull, from edge(ir=1) to center(ir=lrz)
            lr=lrz-ir+1 !Radial index goes from edge(lr=lrz) towards center.
            ! Assume that the pellet crosses flux surfaces 
            ! at normal direction.
            ! Remaining mass of the pellet:
            Mpellet_rem= pellet_M0-dMpellet_sum !dMpellet_sum from previous step
            Gablation=  pellet_Cablation*ne(lr)**pellet_pn *
     &                                   Te(lr)**pellet_pt *
     &                  (Mpellet_rem/pellet_M0)**pellet_pm
            ! Alternatively, just for plots:
            !G_ref= 39.0 * (Te(lr)/2)^pt * (ne(lr)/1e14)^pn * (rp/0.2)^(pm*3)
            ! Ablated mass of pellet within this radial bin (of width dr)
            dMpellet(ir) = Gablation*drfull(ir) / pellet_V 
            ! Units: (gram/sec * cm) / (cm/sec) = gram
            ! Check that it is not larger than the initial mass:
            if(pellet_M0 -(dMpellet_sum+dMpellet(ir)) .gt.0.d0) then
                ! Accumulated ablated mass:
                dMpellet_sum= dMpellet_sum + dMpellet(ir) 
            else ! dMpellet was unphysically large
                dMpellet(ir)= pellet_M0 -dMpellet_sum ! Remaining mass
                dMpellet_sum= dMpellet_sum + dMpellet(ir)
            endif
           enddo ! do ir
           ! For the above run with pellet_Cablation, we got 
            write(*,*)'iter',iter,'(pellet_M0-dMpellet_sum)/pellet_M0=',
     &      (pellet_M0-dMpellet_sum)/pellet_M0      
           pellet_rem_iter= (pellet_M0-dMpellet_sum)/pellet_M0 !This iteration
           ! But the target value is  pellet_fract_rem
           ! Then, adjust the value of pellet_Cablation :
           pellet_Cablation= pellet_Cablation* 
     &                    (1.d0-pellet_fract_rem)*pellet_M0/dMpellet_sum
           ! and continue with next iteration ...
          enddo ! do while iter
          if (abs(pellet_fract_rem-pellet_rem_iter) .gt.
     &            pellet_iter_accur*pellet_fract_rem       ) then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)'set_get_pellet: WARNING: NO CONVERGENCE'
            WRITE(*,*)'  abs(pellet_fract_rem-pellet_rem_iter)=',
     &                   abs(pellet_fract_rem-pellet_rem_iter)
            WRITE(*,*)'  pellet_iter_accur*pellet_fract_rem=',
     &                   pellet_iter_accur*pellet_fract_rem
CMPIINSERT_ENDIF_RANK
          endif
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'Initial guess:    Cablation0=',Cablation0
          WRITE(*,*)'After iterations: pellet_Cablation=',
     &                                 pellet_Cablation
CMPIINSERT_ENDIF_RANK
          
        elseif (ipellet_method.eq.2) then
          !--- ipellet_method=2 --> skip the above iterations,
          ! and simply use the estimate:
          pellet_Cablation= 39.0* 0.5**pellet_pt *
     &                        (1e-14)**pellet_pn * 
     &                      (pellet_rp0/0.2)**(3*pellet_pm) 

        elseif (ipellet_method.eq.3) then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'ipellet_method=3: pellet_Cablation=',
     &                                 pellet_Cablation
CMPIINSERT_ENDIF_RANK
          if(pellet_Cablation.le.0.d0)then
CMPIINSERT_IF_RANK_EQ_0
            WRITE(*,*)
     &      'ipellet_method=3: pellet_Cablation must be set in namelist'
CMPIINSERT_ENDIF_RANK
            STOP 'pellet_Cablation must be set in namelist'
          endif
        else
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,*)'set_get_pellet: Wrong ipellet_method'
CMPIINSERT_ENDIF_RANK
          STOP 'Wrong ipellet_method'
        endif ! if (ipellet_method) ------------------------------------

        write(*,*)'max(rfull)=', MAXVAL(rfull)
        write(*,*)'min(rfull)=', MINVAL(rfull)
        write(*,*)'rfull(lrz),rfull(lrz+1)=',rfull(lrz),rfull(lrz+1)

        dMpellet_sum=0.d0 !Reset: To accumulate the ablated material (integral)
        dMpellet_distr(1:lrfull)=0.d0 !Initialize array for accumulation
                             ! of mass from ablation clouds 
                             !(a cloud at each ir will contribute 
                             ! to several neighboring nodes) 
        irold=0  ! Initialize (will be found in kopt=1 section)
        Rold=0.d0

      else !(kopt=1)!---------------------------------------------------

        !------  Pellet propagation ------------------------------------
        ! (with pellet_Cablation found in iterations or from estimate)
        ! The pellet is propagating towards magnetic axis.
        ! It starts at R=pellet_Rstart at t=pellet_tstart
        ! so that the position of pellet at t=timet is
        ! Rpellet= pellet_Rstart - pellet_V*(timet-pellet_tstart)
        ! Check that timet=pellet_tstart or larger:
        if(timet-pellet_tstart.lt.0.d0) return ! Not launched yet.
        
        !If all pellet is ablated, no need to continue these calculations
        !write(*,*)'Rpellet,dMpellet_sum=',Rpellet,dMpellet_sum
        if(dMpellet_sum.ge.pellet_M0) return ! All pellet is ablated.
        
        ! When timet.ge.pellet_tstart, 
        ! find the nearest index in R radial grid:
        Rpellet= pellet_Rstart - pellet_V*(timet-pellet_tstart)
        Rnew=Rpellet
        ! INBOARD  |--|--|--|--|--|--|--|--|--|  OUTBOARD (R>Raxis)
        !          |                          |
        ! rfull(lrfull)                     rfull(1)
        if(Rpellet.gt.rfull(1)) return !Has not reached plasma yet.
        ! rfull() starts at OUTER edge for ir=1
        ! and continues to INNER edge of plasma.
        if(Rpellet.lt.rfull(lrfull)) then !Passed out of plasma.
           irnew=lrfull 
           return
        endif

        ! irold,irnew ! range of radial indexes in rfull() grid
                      ! corresponding the distance travelled by pellet
                      ! during time interval from timet at previous call
                      ! until this call.
        ! At the very first call of this subr. the value of irold=0.
        ! Search the node nearest to Rpellet:
        irnew=0 ! to be found
        do ir= irold,lrfull  ! over range in rfull grid
          irr=max(1,ir) !YuP[2020-06-26] To be sure irr>0
          if( (Rpellet.ge.rfull(irr)-drfull_p(irr)).and.
     &        (Rpellet.le.rfull(irr)+drfull_m(irr))      ) then
            ! Pellet is close to node ir, within the radial bin width; 
            ! The width of the bin is drfull(ir)=drfull_p(ir)+drfull_m(ir)
            irnew=irr ! found
            goto 125
          endif
        enddo
  125   continue ! exit handle  
        ! If the above condition abs(Rpellet-rfull(ir)).le.drfull(ir)
        ! is not met, then irnew remains equal to 0, 
        ! and then no additional calculations are done.
        ! It could happen that the pellet is moving so fast
        ! that it reached the inner (opposite from launch) 
        ! edge of plasma in just one step
        ! (from previous call of this subroutine to present call).
        ! In this case the above search would not find the value of ir.
        ! Then, use the value of lrfull :
        if(irnew.eq.0)then
          irnew=lrfull
          Rnew= rfull(lrfull)
        endif
        
        ! Find interpolated values of ne and Te at Rnew pellet position.
        !Find two nearest ir nodes between which the pellet is located:
        ir0=0 ! initialize
        ir1=0 ! initialize
        if(irnew.gt.1)then
        if(Rnew.ge.rfull(irnew) .and. Rnew.le.rfull(irnew-1))then
          ! Pellet is in [rfull(irnew); rfull(irnew-1)] range
          ir0=irnew-1
          ir1=irnew
         !write(*,*)'1',ir1,irnew,ir0,rfull(irnew),Rnew,rfull(irnew-1)
         !pause
        endif
        endif
        if(irnew.lt.lrfull)then
        if(Rnew.ge.rfull(irnew+1) .and. Rnew.le.rfull(irnew))then
          ! Pellet is in [rfull(irnew+1); rfull(irnew)] range
          ir0=irnew
          ir1=irnew+1
         !write(*,*)'2',ir1,irnew,ir0,rfull(irnew+1),Rnew,rfull(irnew)
         !pause
        endif
        endif
        ! So, the pellet is in [ir0;ir1] index range,
        !  or [rfull(ir1);rfull(ir0)] R-range.
        ! Just in case, check that ir0 is found:
        if(ir0.eq.0)then ! not found
          WRITE(*,*)'ir0 not found' ! should never print
          STOP 'ir0 not found'
        endif
        ! Determine corresponding indexes in rya grid:
        if (ir0.le.lrz)then
            lr1= lrz-ir0+1  ! index goes from edge(lr=lrz) 
            !towards plasma center (lr=1).
        else ! if ir0.gt.lrz
            ! Flux surface index: goes from center(lr=1) to inner edge (lr=lrz)
            lr1= ir0-lrz  ! Radial index goes from plasma center to lower R.
        endif
        if (ir1.le.lrz)then
            lr0= lrz-ir1+1  ! index goes from edge(lr=lrz) 
            !towards plasma center (lr=1).
        else ! if ir1.gt.lrz
            ! Flux surface index: goes from center(lr=1) to inner edge (lr=lrz)
            lr0= ir1-lrz  ! Radial index goes from plasma center to lower R.
        endif
        !Interpolate density and T from two adj.surfaces to pellet position:
        wght1= (Rnew-rfull(ir1))/(rfull(ir0)-rfull(ir1))
        wght0= 1.d0-wght1
        den_loc= ne(lr0)*wght0 + ne(lr1)*wght1 
        tem_loc= Te(lr0)*wght0 + Te(lr1)*wght1 
        !write(*,*) rfull(ir1),Rnew,rfull(ir0)
        !write(*,*) te(lr0), tem_loc, te(lr1)
        
        dMpellet(1:lrfull)=0.d0 !Initialize array for dM ablated over range
        !of ir where pellet propagates during this call (at this time step)
        
        ! At the end of remaining calculations below,
        ! the value of irnew is copied to irold,
        ! so that at the next call of this subroutine
        ! the scan will be done starting from irold.
        
        !--1--> It can happen that at present call 
        ! we are still in the old radial bin: irnew=irold.
        ! It may happen for a slowly moving pellet,
        ! or a small time step in the code.
        if(irnew.eq.irold)then  ! Add contribution to irnew node
            ir=irnew
            dr=Rold-Rnew !Note: Rnew<Rold (pellet propagates from outer edge)
            Mpellet_rem= pellet_M0-dMpellet_sum !dMpellet_sum is from prev.call
            Gablation=  pellet_Cablation* den_loc**pellet_pn *
     &                                    tem_loc**pellet_pt *
     &                    (Mpellet_rem/pellet_M0)**pellet_pm 
            ! Alternatively, just for plots, (could save) G_ref:
            rp= pellet_rp0*(Mpellet_rem/pellet_M0)**(1.0/3.0)  
            !rp= radius of pellet,  reduced during propagation.
            G_ref= 39.0 *(tem_loc/2)**pellet_pt * 
     &                (den_loc/1e14)**pellet_pn *(rp/0.2)**(pellet_pm*3)
            ! Ablation of pellet within this radial bin (of width dr)
            dMpellet(ir)= Gablation*dr/pellet_V !Note: dr=Rold-Rnew here
            ! dMpellet starts with index 1 at outer plasma edge
            ! Units: (gram/sec * cm) / (cm/sec) = gram
            ! Check that it is not larger than the initial mass:
            if (pellet_M0 - (dMpellet_sum+dMpellet(ir)) .gt.0.d0) then
              ! Accumulated ablated mass:
              dMpellet_sum= dMpellet_sum + dMpellet(ir) 
              !Mpellet= pellet_M0 -dMpellet_sum  
              ! This is effectively the change of mass in time of flight,
              ! to be plotted as a function of r
            else ! dMpellet was unphysically large
              dMpellet(ir) = pellet_M0 -dMpellet_sum  !Remaining mass
              dMpellet_sum= dMpellet_sum + dMpellet(ir) 
              !Mpellet= pellet_M0 -dMpellet_sum !Should be zero here
            endif
        endif ! irnew.eq.irold

        !--2--> Case of pellet getting to a NEW radial bin,
        ! so that irnew>irold. It may include the case of pellet
        ! "jumping" over several radial bins (so that irnew>irold+1)
        if(irnew.gt.irold)then  
          !--2(a)-- Add "left-over" portion at irold bin:
          if(irold.gt.0)then
            ir=irold
            dr= Rold -(rfull(irold)-drfull_p(irold))
            if(dr.lt.0.d0) then
              write(*,*)'Rold, rfull, rfull(irold)-drfull_p(irold)',
     *        Rold,rfull(irold),(rfull(irold)-drfull_p(irold))
              STOP '2a. dr<0'
            endif
            if (ir.le.lrz)then
              lr= lrz-ir+1  ! index goes from edge(lr=lrz) 
              !towards plasma center (lr=1).
            else ! if ir.gt.lrz
              ! Flux surface index: goes from center(lr=1) to inner edge (lr=lrz)
              lr= ir-lrz  ! Radial index goes from plasma center to lower R.
            endif           
            Mpellet_rem= pellet_M0-dMpellet_sum !dMpellet_sum is from prev.call
            Gablation=  pellet_Cablation* ne(lr)**pellet_pn *
     &                                    te(lr)**pellet_pt *
     &                   (Mpellet_rem/pellet_M0)**pellet_pm 
            ! Alternatively, just for plots, (could save) G_ref:
            rp= pellet_rp0*(Mpellet_rem/pellet_M0)**(1.0/3.0)  
            !rp= radius of pellet,  reduced during propagation.
            G_ref= 39.0 *(te(lr)/2)**pellet_pt * 
     &                (ne(lr)/1e14)**pellet_pn *(rp/0.2)**(pellet_pm*3)
            ! Ablation of pellet within this radial bin (of width dr)
            dMpellet(ir)= Gablation*dr/pellet_V
            ! dMpellet starts with index 1 at outer plasma edge
            ! Units: (gram/sec * cm) / (cm/sec) = gram
            ! Check that it is not larger than the initial mass:
            if (pellet_M0 - (dMpellet_sum+dMpellet(ir)) .gt.0.d0) then
              ! Accumulated ablated mass:
              dMpellet_sum= dMpellet_sum + dMpellet(ir) 
              !Mpellet= pellet_M0 -dMpellet_sum  
              ! This is effectively the change of mass in time of flight,
              ! to be plotted as a function of r
            else ! dMpellet was unphysically large
              dMpellet(ir) = pellet_M0 -dMpellet_sum  !Remaining mass
              dMpellet_sum= dMpellet_sum + dMpellet(ir) 
              !Mpellet= pellet_M0 -dMpellet_sum !Should be zero here
            endif
          endif !(irold.gt.0)
          !--2(b)-- Add contributions from all intermediate bins
          ! which the pellet passed through, if any
          !(it can be none, when irnew=irold+1)
          if(irnew.gt.irold+1)then
          do ir= irold+1,irnew-1 ! over range in rfull grid
            dr= drfull(ir)
            if(dr.lt.0.d0) STOP '2b. dr<0'
            if (ir.le.lrz)then
              lr= lrz-ir+1  ! index goes from edge(lr=lrz) 
              !towards plasma center (lr=1).
            else ! if ir.gt.lrz
              ! Flux surface index: goes from center(lr=1) to inner edge (lr=lrz)
              lr= ir-lrz  ! Radial index goes from plasma center to lower R.
            endif
            ! Assume that the pellet crosses flux surfaces at normal direction.
            ! Remaining mass of the pellet:
            Mpellet_rem= pellet_M0-dMpellet_sum !dMpellet_sum is from prev.call
            Gablation=  pellet_Cablation* ne(lr)**pellet_pn *
     &                                    te(lr)**pellet_pt *
     &                   (Mpellet_rem/pellet_M0)**pellet_pm 
            ! Alternatively, just for plots, (could save) G_ref:
            rp= pellet_rp0*(Mpellet_rem/pellet_M0)**(1.0/3.0)  
            !rp= radius of pellet,  reduced during propagation.
            G_ref= 39.0 *(te(lr)/2)**pellet_pt * 
     &                (ne(lr)/1e14)**pellet_pn *(rp/0.2)**(pellet_pm*3)
            ! Ablation of pellet within this radial bin (of width dr)
            dMpellet(ir) = Gablation*dr/pellet_V   
            ! dMpellet starts with index 1 at edge
            ! Units: (gram/sec * cm) / (cm/sec) = gram
            ! Check that it is not larger than the initial mass:
            if (pellet_M0 - (dMpellet_sum+dMpellet(ir)) .gt.0.d0) then
              ! Accumulated ablated mass:
              dMpellet_sum= dMpellet_sum + dMpellet(ir) 
              !Mpellet= pellet_M0 -dMpellet_sum  
              ! This is effectively the change of mass in time of flight,
              ! to be plotted as a function of r
            else ! dMpellet was unphysically large
              dMpellet(ir) = pellet_M0 -dMpellet_sum  !Remaining mass
              dMpellet_sum= dMpellet_sum + dMpellet(ir) 
              !Mpellet= pellet_M0 -dMpellet_sum !Should be zero here
            endif
            ! print when the pellet reached magnetic axis,
            ! just to see the fraction that was ablated so far:
            if(ir.eq.lrz) then
CMPIINSERT_IF_RANK_EQ_0
              WRITE(*,*)'Pellet at timet=',timet
              WRITE(*,*)'Pellet is near magn.axis: Rfull(ir)=',rfull(ir)
              !WRITE(*,*)'dMpellet_sum/pellet_M0=',dMpellet_sum/pellet_M0
              WRITE(*,*)'(pellet_M0-dMpellet_sum)/pellet_M0=',
     &                   (pellet_M0-dMpellet_sum)/pellet_M0
              WRITE(*,*)
     &         'Target value(fract of mass remained) pellet_fract_rem=',
     &          pellet_fract_rem
              WRITE(*,*)'irold,irnew, lrfull=', irold,irnew,lrfull
CMPIINSERT_ENDIF_RANK
            endif
          enddo ! do ir= irold+1 : irnew-1  ! over range in rfull grid
          endif !(irnew.gt.irold+1)
          !--2(c)-- Add a new contribution at irnew bin:
          if(irnew.gt.0)then
            ir=irnew
            dr= (rfull(irnew)+drfull_m(irnew)) -Rnew
            if(dr.lt.0.d0) STOP '2c. dr<0'
            if (ir.le.lrz)then
              lr= lrz-ir+1  ! index goes from edge(lr=lrz) 
              !towards plasma center (lr=1).
            else ! if ir.gt.lrz
              ! Flux surface index: goes from center(lr=1) to inner edge (lr=lrz)
              lr= ir-lrz  ! Radial index goes from plasma center to lower R.
            endif           
            Mpellet_rem= pellet_M0-dMpellet_sum !dMpellet_sum is from prev.call
            Gablation=  pellet_Cablation* den_loc**pellet_pn *
     &                                    tem_loc**pellet_pt *
     &                    (Mpellet_rem/pellet_M0)**pellet_pm 
            ! Alternatively, just for plots, (could save) G_ref:
            rp= pellet_rp0*(Mpellet_rem/pellet_M0)**(1.0/3.0)  
            !rp= radius of pellet,  reduced during propagation.
            G_ref= 39.0 *(tem_loc/2)**pellet_pt * 
     &                (den_loc/1e14)**pellet_pn *(rp/0.2)**(pellet_pm*3)
            ! Ablation of pellet within this radial bin (of width dr)
            dMpellet(ir)= Gablation*dr/pellet_V !Note: dr=Rnew-Rold here
            ! dMpellet starts with index 1 at outer plasma edge
            ! Units: (gram/sec * cm) / (cm/sec) = gram
            ! Check that it is not larger than the initial mass:
            if (pellet_M0 - (dMpellet_sum+dMpellet(ir)) .gt.0.d0) then
              ! Accumulated ablated mass:
              dMpellet_sum= dMpellet_sum + dMpellet(ir) 
              !Mpellet= pellet_M0 -dMpellet_sum  
              ! This is effectively the change of mass in time of flight,
              ! to be plotted as a function of r
            else ! dMpellet was unphysically large
              dMpellet(ir) = pellet_M0 -dMpellet_sum  !Remaining mass
              dMpellet_sum= dMpellet_sum + dMpellet(ir) 
              !Mpellet= pellet_M0 -dMpellet_sum !Should be zero here
            endif
          endif !(irold.gt.0)

        endif !(irnew.gt.irold)
          
          

        ! Distribute the ablated mass among several flux surfaces.
        ! Assume that the ablation cloud is 5-8 times larger than the pellet itself.
        ! Allow for assymmetry between front and trailing(back) side of the cloud.
        ! pellet_rcloud1=! cm ! Effective radius of ablation cloud, leading(front) side. 
        ! pellet_rcloud2=! cm ! Effective radius of ablation cloud, trailing(back) side. 
        ! Recommended: r_cloud ~ (5-8)rp   (of pellet radius)
        do ir=max(1,irold),irnew ! over range in rfull grid,
            !in rfull, index ir goes from 1 at outer R to 2*lrz at inner R
            !!YuP[2020-06-26] At very 1st call [of kopt=1], irold=0.
            rloc= rfull(ir) !Where the center of cloud is at this instant.
            ! Find irp index such that rfull(ir+irp)=rloc-pellet_rcloud1
            ! (leading front of cloud)
            irp=0
            istop=0  
            do while (istop.eq.0)
                irp=irp+1 
                if (ir+irp.gt.lrfull) then
                    irp=irp-1
                    istop=1  
                elseif (rfull(ir+irp) .le. rloc - pellet_rcloud1) then
                    irp=irp-1
                    istop=1 
                endif
            enddo
            ! Find irm index such that rfull(ir-irm)=rloc+pellet_rcloud2
            ! (trailing/backside of cloud)
            irm=0  
            istop=0 
            do while (istop.eq.0)
                irm=irm+1 
                if (ir-irm.lt.1) then
                    irm=irm-1 
                    istop=1  
                elseif (rfull(ir-irm) .ge. rloc + pellet_rcloud2) then
                    irm=irm-1 
                    istop=1 
                endif
            enddo
            !WRITE(*,*)'5. timet,ir,irm,irp=', timet,ir,irm,irp
            ! The ablation cloud is within 
            !    [rloc - pellet_rcloud1 ; rloc + pellet_rcloud2]
            ! It "covers" the radial grid points (in rfull grid) 
            !    [ir+irp ; ir-irm]
            ! Distribute the ablation material within this range,
            ! with a linear drop from R=rloc point (center of cloud)
            ! towards the cloud edges  rloc - r_cloud  and  rloc + r_cloud
            ! (triangular shape of the weight factors)
            if(irm.gt.0)then ! trailing(back) side of the cloud
            do icloud=(ir-irm),ir ! nodes within one half of the cloud.
               drii=abs(rfull(icloud)-rloc) !distance between rloc and given node
               ! Weight factor for each node within this cloud:
               !if pellet_rcloud2>0
               ! Triangular shape:
               wk(icloud)= (pellet_rcloud2 - drii)/pellet_rcloud2 *
     &                  2*drfull(icloud)/(pellet_rcloud1+pellet_rcloud2)
               wk(icloud)= wk(icloud)*dvol_full(icloud) !Volume factor- not sure
               ! Notice that this weight factor is smallest at cloud edge,
               ! where drii = pellet_rcloud2 (with grid-space accuracy)
               ! Another version: square shape:
               ! wk(icloud)=1.0
               !else
               ! wk(icloud)=1.0 
               !end
            enddo
            endif ! if !!! irm>0
            ! If irm=0, wk() remains zero (if irp=0 also, see below)
            
            if(irp.gt.0)then  ! leading(front) side of the cloud
            do icloud=ir,(ir+irp) !nodes within another half of the cloud.
               drii=abs(rfull(icloud)-rloc) !distance between rloc and given node
               ! Weight factor for each node within this cloud:
               !if pellet_rcloud1>0
               ! Triangular shape:
               wk(icloud)= (pellet_rcloud1 - drii)/pellet_rcloud1 *
     &                  2*drfull(icloud)/(pellet_rcloud1+pellet_rcloud2)
               wk(icloud)= wk(icloud)*dvol_full(icloud) !Volume factor- not sure
               ! Notice that this weight factor is smallest at cloud edge,
               ! where drii = pellet_rcloud1 (with grid-space accuracy)
               ! Another version: square shape:
               ! wk(icloud)=1.0    
               !else
               ! wk(icloud)=1.0 
               !end
            enddo
            endif ! if !!! irp>0
            
            if((irp+irm).eq.0)then !both are 0
                !(the case of point-like ablation cloud)
                wk(ir)=1.d0 
                !So, in this case there is no cloud; just a point-like source
            endif
            ! Adjust weight factors so that their sum is exact 1.0
            sum_wk= sum(wk((ir-irm):(ir+irp)))
            wk((ir-irm):(ir+irp))= wk((ir-irm):(ir+irp))/sum_wk 
            !--- Now we can actually distribute dM mass to all these nodes,
            ! with proper weight factors.
            dM0=dMpellet(ir) !Before distributing: 
            !how much mass was removed from pellet.
            do icloud= (ir-irm),(ir+irp) ! All range within cloud
            dMpellet_distr(icloud)=dMpellet_distr(icloud)+dM0*wk(icloud)
            enddo
            ! Note that sum(dMpellet_distr(icloud)) within the cloud is dM0
        enddo ! ir=irold:irnew  ! over range in rfull grid
        ! Note that [ir-irm ; ir+irp] can get outside of [irold;irnew] range.
        ! This is normal - the cloud is larger than the pellet.


        ! Calculate mass density in each radial bin [g/cm^3] .
        ! Combine contributions (ablated mass) from outboard propagation 
        ! and inboard propagation.
        ! The combined grid rfull() runs continuously from outer edge
        ! to inner plasma edge.
        ! Note that rfull(ir)  and rfull(2*lrz-ir+1) correspond to
        ! same flux surface (outer and inner radial points)
        do lr=1,lrz ! Loop over flux surfaces; 
           ! lr=1 is the innermost flux surface
           irfull=lrz-lr+1 ! index that is 1 at OUTER R 
                           !(lr=lrz corr to irfull=1) and 2*lrz at INNER
           dM_p= dMpellet_distr(irfull) !contribution from OUTboard propagation
           !contribution from INboard propagation:
           dM_m= dMpellet_distr(lrfull-irfull+1)  
           !Sum them up; the material was deposited to same radial bin "lr"
           dMdv_sum(lr)=(dM_p+dM_m)/dvol(lr) !mass density of ablated 
           !material, assumed to be spread over all volume 
           !associated with given flux surface.
           ! Could print separate contributions:
           !dMdv_lr= dM_p/dvol(lr) 
           !dMdv_lrplrz= dM_m/dvol(lr) 
        enddo
        !Note: To calculate particle density, use  convert= Avogadro/atw 
        ! where atw is the atomic weight (40 g/mol for Argon)

        !Sanity check: Compare sum(dMdv_sum*dvol) 
        ! with  pellet_M0-Mpellet(t) == dMpellet_sum
CMPIINSERT_IF_RANK_EQ_0
        !WRITE(*,*)'irold,irnew, lrfull=', irold,irnew,lrfull
        WRITE(*,*)'sum(dMdv_sum*dvol)=',DOT_PRODUCT(dMdv_sum,dvol)
        WRITE(*,*)'timet, dMpellet_sum=',timet, dMpellet_sum
CMPIINSERT_ENDIF_RANK
        ! They should be same.
        
        irold=irnew !Save this value for the next call of this subroutine
        Rold=Rnew  !Save this value for the next call of this subroutine

      endif !(kopt) !---------------------------------------------------

      return
      end subroutine set_get_pellet

!=======================================================================
!=======================================================================

      subroutine impurity_update
      !YuP[2020-07-02] Collected these lines that were in tdchief,
      !made them into this subroutine.
      ! ONLY VALID for cqlpmod.eq.'disabled'
      implicit integer (i-n), real*8 (a-h,o-z)
CMPIINSERT_INCLUDE
      include 'param.h'
      include 'comm.h'
      external adcdo
      save dMpellet_sum
      real*8 ZDISTR(100) ! local, for subr. ADCDO()

         if((imp_depos_method.ne.'disabled') .and. (kelecg.eq.1))then
           !YuP[2020-06-24] Changed (gamafac.eq."hesslow") to (imp_depos_method.ne.'disabled')
           !   [a more general logic]
           !YuP[2019-09-16]
           ! After temp() and reden() profiles are updated, 
           ! find profile of impurity produced by pellet:
           if(imp_depos_method.eq.'pellet' 
     &       .and. timet.ge.pellet_tstart) then
           if(dMpellet_sum.lt.pellet_M0)then ! Not all pellet yet gone
             kopt=1 !propagate pellet; calc. dMpellet_dvol_sum(1:lrz)
             Rpellet= pellet_Rstart -pellet_V*(timet-pellet_tstart)
             if(Rpellet.lt.rmag)then 
              ! ablation rate is determined by ne and Te 
              ! which are already affected by pellet deposition
              temp_wk(1:lrz)=temp(kelecg,1:lrz)
              reden_wk(1:lrz)=reden(kelecg,1:lrz)
             else ! Rpellet>rmag
              !use values before pellet affected the temp. and dens.
              temp_wk(1:lrz)=temp_t0(kelecg,1:lrz)
              reden_wk(1:lrz)=reden_t0(kelecg,1:lrz)
             endif
             call set_get_pellet(kopt, timet, lrz, rmag, 
     &           rpcon(1:lrz), rmcon(1:lrz), dvol(1:lrz), 
     &           temp_wk(1:lrz), reden_wk(1:lrz),
     &           pellet_V, pellet_M0, pellet_Rstart, pellet_tstart,
     &           pellet_rcloud1, pellet_rcloud2, pellet_rp0,
     &           pellet_pn, pellet_pt, pellet_pm,
     &           ipellet_method, pellet_fract_rem, 
     &           ipellet_iter_max, pellet_iter_accur, pellet_Cablation,
     &              Gablation,dMpellet_sum, dMpellet_dvol_sum(1:lrz) ) 
!CMPIINSERT_IF_RANK_EQ_0
!         do ll=1,lrz
!          WRITE(*,'(a,i4,2e14.7)')
!     +   'tdchief/set_get_pellet: n,dMpellet_sum,dMpellet_dvol_sum(ll)',
!     +    n,dMpellet_sum,dMpellet_dvol_sum(ll)
!         enddo
!CMPIINSERT_ENDIF_RANK
     
             ! Find and save local rho at pellet position (for plots)
             rloc= Rpellet  ! Local R at pellet position
             zloc= zmag     ! Assumed - pellet travels along Z=Zaxis
             ppsi=terp2(rloc,zloc,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     &          epsirz,nnra,0,0)
             ! From ppsi, find local rho, based on eqrho() array .
             ! Definition of eqrho is set by specification of radcoord.
             call terp1(nconteqn,-eqpsi,eqrho,d2eqrho,-ppsi,1,tab,itab)
             rho_loc=tab(1)/rhomax  ! rhomax is in comm.h
             pellet_rho= rho_loc
             pellet_Mrem= pellet_M0-dMpellet_sum ! Remaining mass[gram]
             !Next: calculate particle density, use convert=Avogadro/atw
             ! where atw is the atomic weight (40g/mol for Argon), 
             ! or simply (Avogadro*proton)/(atw*proton) = 1.0/fmass_imp
             ! where fmass_imp=atw*proton (in comm.h)
             ! THIS IS WHAT WE NEED FOR cfpcoefn.f (stored in comm.h):
            dens_imp_allstates(1:lrz)=dMpellet_dvol_sum(1:lrz)/fmass_imp
             ! Note: At given instant t=timet, pellet is at 
             ! Rpellet(t)= pellet_Rstart -pellet_V*(timet-pellet_tstart)
             ! It means that a given flux surface at R=rpcon(lr) 
             ! is reached by pellet at instant
             ! t= pellet_tstart +(pellet_Rstart-rpcon(lr))/pellet_V
             ! We use this t as the onset time for the 
             ! exp(-(t-tstart)/tau) temperature drop, 
             ! i.e. as the value for 
             ! temp_expt_tstart(lr) in case of iprote='prb-expt' option.
             ! Similarly in case of iproti='prb-expt',
             ! and we use same values of temp_expt_tstart, etc.
           endif !(dMpellet_sum.lt.pellet_M0) Not all pellet yet gone
           ! But what to do when ALL pellet is ablated?
           ! For now, we assume, all that pellet material stays within
           ! flux surfaces (no radial transport). Only ionization
           ! states may change, because of changing Te(time). 
           endif !(imp_depos_method.eq.'pellet')
           
           if(imp_depos_method.eq.'instant' 
     &       .and. timet.ge.tstart_imp) then  !YuP[2019-12-05]
              !Instant deposition of impurity at all surfaces.
              ! dens0_imp(0:lrz) profile must be set in namelist
              dens_imp_allstates(1:lrz)=dens0_imp(1:lrz) !YuP[2019-12-05]
           endif !(imp_depos_method.eq.'instant') !YuP[2019-12-05]
           
           !--------- NOW Find distribution over charge states.
           ! Based on Corona model, and using ADPAK data (from *.ntau files).
           ! general INPUT arguments for sub.set_get_ADPAK() :
           kopt=1 ! option flag: '1' is to get <Z> and <Z^2> 
           tau_r=adpak_tau_r ![sec] Characteristic time of radial decay of T_e
           do ll=1,lrz !ilend
             call tdnflxs(ll) !-> get l_,lr_,...
             dens_ne=reden(kelecg,lr_) ![cm^-3]
             temp_Te= temp(kelecg,lr_) ! Te [keV] at this flux surface
             !Get neutral density of D0, needed as an input for ADPAK tables, 
             ! where it enters through the ratio nD0/ne.
             rho=rya(lr_)
             !YuP[2020-11-15] Moved call to get_dens_nD0_ADPAK() outside of if().
             ! It is not actually specific to ADPAK. 
             ! It simply sets dens_nD0= dens_nD0_b*exp((rho-1.d0)*radmin/dens_nD0_l)
             call get_dens_nD0_ADPAK(model_dens_nD0, dens_nD0_b,
     &            dens_nD0_l, rho, radmin, dens_nD0) !-> dens_nD0
             ![2020-11-15] This change affects results, but not strongly.
             ! Before this change the value of dens_nD0 was not defined
             ! (initialized to a neg. value on PC, or small pos. value on Cray)
             if(adpak.eq.'enabled')then
               call set_get_ADPAK(kopt,imp_type, 
     &            temp_Te,dens_nD0,dens_ne,tau_r,
     &            z1av,z2av) !-> OUT: <Z> and <Z^2> for given imp_type
               !Now, from knowledge of <Z> and <Z^2>, find distribution fz(Z)
               ! over charge states:
               iZatom=INT(bnumb_imp(nstates)) !should be integer to this subr.
               !Could simply use iZatom=nstates
               call get_distr_charge_states(iZatom,z1av,z2av, fz) 
               !          INPUT: iZatom,z1av,z2av   OUTPUT: fz(0:iZatom)
               !Note: fz has normalization SUM(fz(0:iZatom))=1, 
               !      and SUM(fz*Z)= <Z> (with some accuracy).
             else ! adpak.ne.'enabled'
               ! Alternatively (use it for W; which is imp_type=8): 
               INUCZ=INT(bnumb_imp(nstates)) !atom nuclear charge
               ZTE= temp_Te*1d3 ! ZTE is Te in eV
               ZNE= dens_ne  ! electron density in cm-3
               ZNA= dens_nD0 ! density of hydrogen atoms in cm-3
               kion1=kionm(1) !assume all ions have same T, so pick the 1st
               ZTA=temp(kion1,lr_)*1d3/2. 
               !ZTA= Timp/Mimp +Ta/Ma ~ Ta/Ma. For D: Ti/2 (Ma=2)
               !ZTA is relative temperature for charge exchange of ions with
               !hydrogen atoms which is Tz/Mz+Th/Mh
               CALL ADCDO(INUCZ,ZTE,ZNE,ZTA,ZNA,
     &           ZS, ZS2, EZRAD, EZLOSS, ZDISTR)
               !output:
               ! ZS is <z>  ;  ZS2 is <z^2>
               ! EZRAD is the radiation power W*cm3
               ! EZLOSS is the ionization energy losses, W*cm3
               ! ZDISTR is array for distribution function of excited states with
               ! sum(ZDISTR)=1
               ! The ADCDO subroutine is in /ADC/ folder. Need to compile and link
               ! all 16 files in that folder.
               fz(0:nstates)=ZDISTR(1:nstates+1)
             endif ! adpak= enabled or not.
             !For a given total density of impurity dens_imp_allstates(lr_)
             !we now can find the density for each charge state:
             do kstate=0,nstates
               dens_imp(kstate,lr_)=dens_imp_allstates(lr_)*fz(kstate) !cm^-3
             enddo
             !dens_imp_allstates(lr_) is the total 
             !(over all charge states, incl. Z=0)
             !density of impurity at given lr_, at given time step.
             ! This is just the ablated material from pellet,
             ! before ionization process occured.
             !Note: sum(dens_imp(0:nstates,lr_))= dens_imp_allstates(lr_)
           enddo ! do ll
           call profiles ! update reden(kelecg) based on dens_imp()
           !(very small effect from this update of reden)
         endif !((imp_depos_method.ne.'disabled') .and. kelecg.eq.1)
         
      return
      end subroutine impurity_update
