c
c
      subroutine aindflt
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     Set namelist input defaults for all namelist sections
c     except setup0/fsetup and frsetup.
c     Warning: should not set variables read in setup0/fsetup namelist
c     as aindflt is called AFTER the first read(2,setup0).
c     BH070305:  Some other constants derived from namelist
c                variables have been moved to new subroutine aindlft1.
c..................................................................

      include 'param.h'
      include 'name_decl.h'

c     Set a few local constants, same as in subroutine ainsetpa.
      ep100=1.d+100
      zero=0.d0
      one=1.d0
c      pi=3.141592653589793d0
      pi=atan2(zero,-one)

      iy=200 ! default value; will be over-written by cqlinput value
      jx=300 ! default value; will be over-written by cqlinput value
      mx=3   ! default value; will be over-written by cqlinput value

      jfl=151  !!!min(201,jx)
      if (mod(jfl,2).eq.0) jfl=jfl-1  
      ! jfl needed to be odd because of jpxyh=(jfl+1)/2 in pltprppr.f

      nmods=nmodsa ! YuP-101220: should be mrfn, but not known yet
            
      nso=0
      lz=40 !YuP[2021-04-14] was lz=lza .  lza is not used anymore

      ampfmod="disabled"
      ampfadd="neo+bscd" !YuP[2019-12-26] Added ampfadd; 
                         !other values: "disabled","neosigma","add_bscd"
      nampfmax=2
      ampferr=1.d-3
      nonampf=0

      bootst="disabled"   ! analytic (Hinton and Haseltine) 
                          ! bootstrap current
      bootcalc="disabled" !computational bootstrap current off.
      bootupdt="disabled" !updating 0th order distn for bs radial derv.
      bootsign=+1.0
      nonboot=2           !turn on computational bootstrap at n=nonboot.
      bremsrad="disabled"
      brfac=0.
      brfac1=0.
      brfacgm3=1.0
      isoucof=0
      faccof=1.d0
      bth=1000.
      btor=10000.
      chang="enabled"
      constr=1.d-3
      contrmin=1.d-12
      curr_edge=0.
      currerr=0.1 !0.1
      deltabdb=0.
      do k=1,ngena
         difus_type(k)="specify"
         difus_io(k)="disabled"
      enddo
      difus_io_file="drrin.nc"
      ndifus_io_t=0
      do ii=1,nbctimea
         difus_io_t(ii)=zero
         do k=1,ngena
            difus_io_drrscale(ii,k)=one
            difus_io_drscale(ii,k)=one
         enddo
      enddo
      difusr=1.d4
      difus_rshape(1)=1.d0
      difus_rshape(2)=3.d0
      difus_rshape(3)=3.d0
      difus_rshape(4)=1.d0
      difus_rshape(5)=-1.d0
      difus_rshape(6)=0.d0
      difus_rshape(7)=0.d0
      difus_rshape(8)=0.d0
      difus_vshape(1)=0.d0
      difus_vshape(2)=0.d0
      difus_vshape(3)=0.d0
      difus_vshape(4)=0.d0  !==c4, For 1/gamma^c4 dependence
      droptol=0.001d0
      dtr=5.d0
      dtr0=dtr
      do 3 i=1,ndtr1a
         dtr1(i)=0.
 3    continue
      esink=0.
      efflag='toroidal'
      efiter="enabled"
      efswtch="method1"
      efswtchn="disabled"
      efrelax=0.5
      efrelax1=0.5 !0.8
      efrelax_exp=1.d0 !YuP[2020-04-03] For generalized procedure in efswtch="method4"
      elpar0=0.d0
      eoved=-.01
      enorm=200.
      enorme=enorm
      enormi=enorm
      epsthet=0.1
      eqmodel="power"
      eseswtch="disabled"
      f4d_out="disabled"
      f3d_out="disabled"
      f3d_format="ascii"
      gamaset=0.d0
      gamafac="disabled"
      !--------------------------------------------------------------
      !YuP[2019-07-31]-[2019-09] Added new namelist var:
      !Impurity type, which can be in many ionization states,
      ! depending on plasma electron temperature, etc.
      adpak='enabled' ! To use ADPAK tables (alternatively, use ADCDO)
      imp_type=6 !for gamafac="hesslow". 1-He,2-Be,3-C,4-N,5-Ne,6-Ar,7-Xe,8-W,9->B
      ! For ADPAK subroutines/tables, need values of neutral D0.
      model_dens_nD0=1  ! Only one model so far for neutral D0. 
      dens_nD0_b=1.0d10 ! 1/cm^3 ! Edge density of neutral D0
      dens_nD0_l=10.d0  !cm! Scale length of exp-decay, as in  
      !                      dens_nD0_b*exp[(rho-1)*radmin/dens_nD0_l]
      ! Also for ADPAK subroutines/tables, need this:
      adpak_tau_r=1.d-3 !sec! Characteristic time of radial decay of T_e
      !           Note from A. Pigarov:
      !           For disruption case, I would set tau(r)=1.e-3 sec. 
      !           For Smith-like run case tau(r)=TauT, 
      !           where TauT is the temperature decay time.
      !           For quasi-stationary plasma it is likely about 
      !           plasma confinement time ~1 s.
      !           Or should we rather get this time from actual change
      !           of temp() ?  Values of temp() are updated 
      !           in subr.profiles, in case of nbctime.ne.0.
      ! For test purposes (set to 0 to disable):
      imp_bounde_collscat=1 !=1 to enable effects of scattering 
        !of electrons on partially-ionized impurity ions 
        !(Hesslow corrections)
      imp_bounde_collslow=1 !=1 to enable effects of slowing down 
        !of electrons on partially-ionized impurity ions
      !--------------------------------------------------------------
      ! Method of deposition of impurity:
      imp_depos_method='disabled' !or "pellet" or "instant" !YuP[2019-12-05]
      imp_ne_method='ne_list' !YuP[2019-12-06] How ne is calculated:
      !YuP[2019-12-06] There are two ways to adjust electron density:
      ! 1. Assume that density of main ion species (not impurity ions) is
      !    taken from input namelist (could be time-dependent); 
      !    then, electron density is set to 
      !      sum(n(k)*Z(k))[all k=kionm] +
      !    + sum(nimp(kstate)*Zimp(kstate))[all kstates].
      !    This is imp_ne_method.eq.'ni_list' option.
      !    See profiles.f, line~760.
      ! 2. Assume that electron density is 
      !    taken from the input namelist; 
      !    then, reduce the density of main ions, 1st ionic species.
      !    With increase of impurity ions, the density of main ions
      !    will go down, to maintain the value of ne from list.  
      !    This is imp_ne_method.eq.'ne_list' option.
      !    See profiles.f, line~760.
      !--------------------------------------------------------------
      !---> For (imp_depos_method="pellet") pellet propagation/ablation model:
      pellet='disabled' !enable to use a pellet as a source of impurities
      pellet_Rstart=230. ![cm] Major radius where pellet is launched.
      ! Suggestion: Set it to rpcon(lrz), or R_LCFS radius.
      pellet_tstart=0.d0 ![sec] Instant when pellet is launched.
      ! Not necessarily 0.0, but should be .ge.0.
      pellet_V=30000.d0 ![cm/s] Pellet speed. Typically 10000-900000cm/s
      ! Assumed constant all the way through plasma.
      ! Assumed that pellet travels along equatorial
      ! plane, going through magnetic axis.
      pellet_M0=30.d-3 ![gram] Initial mass of pellet(at t=pellet_tstart)
      ! If pellet is large, it can make to the inner border of plasma.
      !.......
      ! Related to pellet size and ablation cloud:
      ! For distributing the ablated mass among several flux surfaces,
      ! assume that the ablation cloud is 5--8 times
      ! larger than the pellet itself.
      ! Allow for assymmetry between leading (front) 
      ! and trailing (back) side of the cloud.
      pellet_rp0=0.5d0 !cm! Pellet radius at t=pellet_tstart (plasma edge).
      pellet_rcloud1=4.d0 !cm! Radius of ablation cloud, leading (front) side.
      pellet_rcloud2=4.d0 !cm! Radius of ablation cloud, trailing (back) side. 
      ! Typically pellet_rp0== rp(0) = 0.2--0.5 cm.
      ! Recommended: pellet_rcloud ~(5--8)*pellet_rp0
      ! Pellet radius is reduced during propagation 
      ! (as the mass is reduced).
      ! However, in present model, rcloud is not changed, 
      ! so the cloud size remains as described by pellet_rcloud1,2 above.
      !.......
      ! Related to description of ablation rate:
      ! Assume that the ablation rate of pellet is proportional to local 
      ! ne^pn * Te^pt (electron T and density in some powers),
      ! and proportional to remaining_mass/pellet_M0 in some power "pm".
      ! So that the local ablation rate is 
      ! G[gram/s]= 
      !  =Cablation* ne[cm-3]^pn *Te[keV]^pt *(Mpellet(t)/Mpellet(0))^pm
      ! See REFS: "2019-03-15-Friday Science Meeting-Jie Zhang.pdf"
      ! Values for those powers:
      pellet_pn=1.d0/3.d0 ! power "pn" in the above Eqn. for G.
      !             REFS: should be 1/3
      pellet_pt=5.d0/3.d0 ! power "pt" in the above Eqn. 
      !             REFS: should be 11/6, or 5/3 
      pellet_pm=4.d0/9.d0 ! power "pm" in the above Eqn. 
      !             REFS: should be 4/9 (so that rp^(4/3))
      ! where Mpellet(t)==Mpellet_rem is the remaining mass
      ! at given radial position R(t).
      ! Note that  (Mpellet(t)/Mpellet(0))^pm ~~  (rp(t)/rp(0))^(3*pm)
      ! For example, when pm=2/3, we get  G~~ rp^2,
      ! which means - proportional to surface area of the pellet
      ! (S_pellet= 4*pi*rp^2).
      ! Why in REFS they use pm=4/9, and not 2/3 ?
      ! The value of Cablation=="pellet_Cablation" is either calculated
      ! during kopt=0 call of this subroutine, 
      ! or set as a namelist value, see below.
      !.......
      ! Related to calculation of pellet_Cablation value.
      ipellet_method=1  !Iterative procedure,
      ! to find such value of pellet_Cablation which yields the value
      ! of fraction of pellet remained at magnetic axis.
      pellet_fract_rem=0.5d0 !Fraction of remaining mass when pellet 
      ! reaches magn.axis, i.e. it is   
      ! pellet_fract_rem= (pellet_M0-dMpellet_sum(t_axis))/pellet_M0
      ! where dMpellet_sum(t_axis) is the total ablated mass 
      ! during the flight of the pellet from plasma edge to magn.axis.
      ! The value of pellet_Cablation will be found from iterations.
      ! For this method, also specify these two values:
      ipellet_iter_max=50 ! Max number of iterations
      pellet_iter_accur=1.d-2 !Relative error (accuracy) 
      ! achieved in iterations, to be compared with 
      ! |pellet_fract_rem-pellet_rem_iter|/pellet_fract_rem
      pellet_Cablation=0.001d0 ! Only needed for ipellet_method=3 :
      ! Instead of value found from iterations, use the value from input
      !-----------------------------------------------------------------
      !YuP[2019-09-18]
      !---> For new option iprote='prb-expt' or 'spl-expt',
      ! to set the temper. decay.
      ! T(t)= Tend +(T(tstart)-Tend)*exp(-(t-tstart)/tau) for electrons.
      ! where exp(-(t-tstart)/tau)  is applied only at t.ge.tstart.
      ! See H.M.Smith and E.Verwichte, PoP vol.15, p.072502, (2008),
      ! Eqn.(7).
      temp_expt_Tend=0.010d0 ![keV] final(ending) Tend after cooling.
      temp_expt_tau0=3.0d-3 ![sec]slow decay time of Te(t) 
      temp_expt_tau1=0.1d-3 ![sec]fast decay time of Te(t)(for Thermal Quench)
      do ll=1,lrza
         temp_expt_tstart(ll)=0.d0  ![sec] tstart in the above Eqn.
      ! In case of pellet='enabled', it will be calculated during run
      ! to match the pellet position. 
      enddo
      ! Similarly in case of iproti='prb-expt' or 'spl-expt',
      ! and we use same values of temp_expt_Tend, temp_expt_tau,
      ! temp_expt_tstart.
      
      !YuP[2020-03-18] Not ready yet
      !---> For new option iprone='prb-expt' or 'spl-expt',
      ! to set the density growth (or decay, depending on ne_end).
      ! n(t)= nend +(n(tstart)-nend)*exp(-(t-tstart)/tau) for electrons.
      ! where exp(-(t-tstart)/tau)  is applied only at t.ge.tstart.
      ! See H.M.Smith and E.Verwichte, PoP vol.15, p.072502, (2008),
      ! Eqn.(7).
!      dens_expt_nend=2.d14  ![cm^-3] final(ending) ne_end after cooling.
!      dens_expt_tau0=3.0d-3 ![sec]slow decay time of ne(t) 
!      dens_expt_tau1=0.1d-3 ![sec]fast decay time of ne(t)(for Thermal Quench)
!      do ll=1,lrza
!         dens_expt_tstart(ll)=0.d0  ![sec] tstart in the above Eqn.
!      ! In case of pellet='enabled', it will be calculated during run
!      ! to match the pellet position. 
!      enddo

      dens0_imp(0:lrza)=0.d0 !YuP[2019-12-05], for imp_depos_method="instant"
      ! dens0_imp = Density profile of deposited impurity [1/cm^3];
      ! This is just the ablated material (e.g. from pellet, flake, etc),
      ! before ionization process occured.
      !Note: For imp_depos_method="pellet", this profile is calculated 
      !during run, based on parameters of pellet (mass, speed, ...).
      tstart_imp=0.d0 ![sec] Instant when impurity is deposited.
      !For imp_depos_method="pellet", tstart_imp=pellet_tstart (launch time)
      !-----------------------------------------------------------------
      
      !-------------------------
      !YuP[2020-07-02] Added, for usage in ainalloc,tdchief,cfpcoefn
      cfp_integrals="disabled" !means: Use the original method for calc. of
      ! integrals for Maxwellian distribution (slow method),
      ! in subr. cfpcoefn.
      ! Alternatively, set to 'enabled', which means that the table 
      ! will be produced in subr. cfp_integrals_maxw.
      ! These integrals describe a contribution to BA coll.coefs
      ! from local collisions of general species with the background 
      ! Maxwellian species (search "kbm=ngen+1,ntotal").
      ! These integrals only depend on mass (fmass)
      ! and local temperature of these (Maxwellian) species.
      ! So, instead of calculating them over and over again,
      ! calculate them once as a table over temperature grid 
      ! and then reuse them by matching a local T
      ! along orbit with the nearest values in the T-grid.
      !-------------------------

      !-----------------------------------------------------------------   
      !BH,YuP[2021-01-21] namelist variables to read data files.
      !(Initial purpose - coupling with NIMROD. 
      ! Can be extended to coupling with other codes.)
      read_data="disabled" !Other possible values: 'nimrod', for now.
      ! Set default names for data files. They are declared as
      !character*128, dimension(1000) :: read_data_filenames !list of files
      ! Max number of files is 1000, for now. 
      ! For coupling with NIMROD, each file contains data at one time slice.
      ! Therefore, it is recommended to match the max number of files
      ! with value of nbctimea [set in param.h]
      do i=1,size(read_data_filenames)
         read_data_filenames(i)="notset"
      enddo
      read_data_filenames_prefix="notset" !YuP[2021-04-06]
      read_data_filenames_suffix="notset" !YuP[2021-04-06]
      temp_min_data=5.d-3 ![keV] Lower limit, to adjust Te and Ti data
      elecfld_data= "jspitzer" ![2022-07]'jspitzer' or 'readdata'
      !-----------------------------------------------------------------    
      
      gsla=270.
      ephicc=0.
      gslb=35.
      jhirsh=0
      do ib=1,kb  ! kb is max number of nbeams (parameter in param.h)
         kfrsou(ib)=0
      enddo
      lbdry0="enabled"
      sbdry="bounded"
      scheck="enabled"
      iactst="disabled"
      idrop="No-Op"
      idskf="disabled"
      idskrf="disabled"
      ichkpnt="disabled"
      izeff="backgrnd"
      implct="enabled"
      ineg="disabled"
      isigtst=1
      do 6 i=1,6
        isigmas(i)=0
 6    continue
      isigsgv1=0
      isigsgv2=0
      kenorm=1
      lfil=30
      lmidpln=1
      lmidvel=0
      laddbnd=1
      colmodl=1
      locquas="disabled"
      machine="toroidal"
      manymat="disabled"
      meshy="free"
      nummods=1
c     if numixts= 1=>forw/back; -1=>back/forw for numindx=2
      numixts=1
      nchec=1
c     if nchgdy=1 adapt dy(ith)
      nchgdy=0
      ndeltarho="disabled"
      negyrg=0
      netcdfnm="disabled"
      netcdfshort="disabled"
      netcdfvecal="disabled"
      netcdfvecc="disabled"
      netcdfvece="disabled"
      netcdfvecrf="disabled"
      netcdfvecs="all"
      ncoef=1
      ncont=25
      nrstrt=1
c     if ngauss.ge.1 => analegco=disabled
c     good numbers are nlagran=4 and ngauss=4 or 6
c     max. nlagran allowed: 15
      nfpld=0
      ngauss=0
      nlagran=4
      nrf=0
      ngen=ngena
      nmax=nmaxa
      nonavgf=5
      nofavgf=10
      noncntrl=0
      nonel=0
      noffel=10000
      nonloss=0
      noffloss=10000
      nonvphi=10000
      noffvphi=10000
      nontran=0
      nofftran=10000
      nonelpr=10000
      noffelpr=0
cBH080305      do k=1,nmodsa
      do k=1,ngena
         nonrf(k)=0
         noffrf(k)=10000
      enddo
      nrskip=10
      numby=20
      do i=1,nplota
         nplot(i)=-10000
         nplt3d(i)=-10000
      enddo
      do i=1,nsavea
         nsave(i)=-10000
      enddo
      do 4 i=1,ndtr1a
         nondtr1(i)=-1
 4    continue
      npa_diag="disabled"
      atten_npa="enabled"
      nstop=5
      nstps=100
c     old way of integrating dens,cur in diaggnde
      oldiag="enabled"
      partner="disabled"
      profpsi="disabled"
      plt3d="enabled"
      pltd="enabled"
       !YuP[2018-02-07] New: pltd='color' and 'df_color' 
       !for color contour plots
      pltdn="disabled"
      pltend="enabled"
      pltfvs="disabled"
      pltinput="enabled"
      pltlim="disabled"
      pltlimm=1.
      pltlos="disabled"
      pltso="disabled" 
       !YuP[2018-02-07] New: pltso='color' and 'first_cl' 
       !for color contour plots
      pltmag=1.  !YuP: Not used?
      pltsig="enabled"
      pltpowe="disabled"
      pltprpp="disabled"
      pltfofv="disabled"
      pltrst="enabled"
      pltstrm="disabled"
      pltflux="disabled"
      do 7 i=1,6
         pltflux1(i)=1.
 7    continue
      pltflux1(7)=0.
      pltvecal="disabled"
      pltvecc="disabled"
      pltvece="disabled"
      pltvecrf="disabled"
      plturfb="enabled" !YuP[2018-02-07] New: 'color' for color contour plots
      pltvflu="disabled"
      pltvs="rho"
      pltra="disabled"
      psimodel="axitorus"
      radcoord="sqtorflx"
      radmaj=100.
      !----- For Miller Equilibrium (eqsource.eq."miller"):   -----------
c**   REF: R.L. Miller et al., "Noncircular, finite aspect ratio, local
c**   equilibrium model", Phys. Plasmas, Vol. 5, No. 4, April 1998.
c**   Setup is done similar to COGENT version (MillerBlockCoordSysF.ChF)  
c**   The difference is in units: COGENT uses [Tesla, meters],
c**   while CQL3D uses [Gauss, cm]. Need to specify input values:
      ! Some of values below are set to unlikely numbers.
      ! It will trigger warning/stop, with suggestion 
      ! to specify them in cqlinput. 
      eq_miller_rmag=1.d99 ! Magnetic axis: major radius coord [cm]
      eq_miller_zmag=0.d0  ! Magnetic axis: vertical coord [cm]
      eq_miller_btor=1.d99 ! Tor field at Geom. center of LCFS [Gauss]
      eq_miller_radmin=1.d99   ! Plasma minor radius [cm]
      eq_miller_cursign=+1.d0  ! Sign of Plasma Current [+1. or -1.]
      eq_miller_psimag=1.d99 ! Pol.flux at magn.axis [cgs] Set as positive
      eq_miller_psilim=1.d99 ! Pol.flux at LCFS: Set smaller than psimag
      eq_miller_psi_n=2.0 ! n and m powers for PSI(r) profile as in 
      eq_miller_psi_m=1.0 !for PSI(r)= psilim+(psimag-psilim)*(1-(r/a)^n)^m
      eq_miller_deltaedge=0.d0 ! Triangularity of LCFS (at r=radmin)
      eq_miller_kappa=1.d0  ! Vertical elongation (const for all surfaces)
      eq_miller_drr0= 0.d0  ! dR0/dr  we assume Shafr.shift=-drr0*r
      ! See subr. eq_miller() for definition of surfaces and fields.
      !------------------------------------------------------------------
      
      eleccomp="enabled"
      radmin=50.
      relativ="enabled"
      relaxden=1.
      relaxtsp="disabled"
      rfacz=.7
      roveram=1.e-6
      rmirror=2.
      rzset="disabled"
      sigmamod="disabled"
      sigvcx=0.
      sigvi=0.
      simpbfac=1.d0
      symtrap="enabled"
      syncrad="disabled"
      softxry="disabled"
      soln_method="direct"
      tandem="disabled"
      taunew="disabled"
      tavg="disabled"
      do i=1,ntavga
         tavg1(i)=zero
         tavg2(i)=zero
      enddo
      tbnd(1)=.002
      do ll=2,lrorsa
         tbnd(ll)=0.
      enddo
      temp_den=0.d0
      tfac=1.
      tfacz=1.
      thetd=0.0
      transp="disabled"
      pinch="disabled"
      advectr=1.0
      adimeth="disabled"
      nonadi= 5
      drr_scale=1.d0 !YuP[2021-08] added (used in case of read_data='nimrod')
      pltdrr="first" !Added/YuP[2021-08] Plots of Drr(i,j) at n=nontran
      
      updown="symmetry"

      rdcmod="disabled"
      rdc_clipping="disabled"
      rdc_upar_sign=+1.
      nrdc=1
      rdcfile(1)="du0u0_input"
      nrdcspecies(1)=1
      rdcscale(1)=1.d0
      do i=2,nrdca
         rdcfile(i)="notset"
         nrdcspecies(i)=0
         rdcscale(i)=1.d0
      enddo

      urfmod="disabled"
      urfmult=1.0
      veclnth=1.0
      vdalp=.03
      vlfmod="disabled"
      vlfmodes=1.
      vlfnpvar="1/r"
      vlfbes="enabled"
      do k=1,nmodsa
         vlfharms(k)=1.
         dlndau(k)=1.
         vlfdnorm(k)=10.
         vlffreq(k)=.8e9
         vlfnp(k)=5.
         vlfdnp(k)=.2
         vlfddnp(k)=.1
         vlfnperp(k)=5.
         vlfharm1(k)=0.
         vlfeplus(k)=(0.,0.)
         vlfemin(k)=(0.,0.)
         vlfpol(k)=0.
         vlfdpol(k)=360.
         vlfddpol(k)=20.
         vlfparmn(k)=-ep100
         vlfparmx(k)=+ep100
         vlfprpmn(k)=0.0
         vlfprpmx(k)=+ep100
         vparmin(k)=-1.
         vparmax(k)=1.
         vprpmin(k)=0.
         vprpmax(k)=ep100
         vlh_karney=0.
         vlhpolmn(k)=0.
         vlhpolmx(k)=180.
         vlhprprp(k)="parallel"
      enddo
      vlhplse="disabled"
      vlhmod="disabled"
      vlhmodes=1.
      vlhpon=.1
      vlhpoff=.11
      vnorm=4.e10   !  Usually set through enorm
      vprprop="disabled"
      xfac=1.
      xlfac=1.
      xlpctlwr=.1
      xlpctmdl=.4
      xllwr=1./43.
      xlmdl=.25
      xpctlwr=.1
      xpctmdl=.4
      xlwr=1./43.
      xsinkm=1.
      xmdl=.25
      xprpmax=1.
      ylower=1.22
      yupper=1.275
      yreset="disabled"
      npwrzeff=1.
      mpwrzeff=1.
      npwrvphi=2.
      mpwrvphi=0.
      npwrelec=1.
      mpwrelec=1.
      npwrxj=1.
      mpwrxj=1.
      urfrstrt="disabled"
      urfwrray="disabled"
      xsink=0.

c.......................................................................
c     lrza arrays
c.......................................................................
      drya=1.d0/dfloat(lrza)
      do 100 ll=1,lrza
        rovera(ll)=.1
        rya(ll)=(ll-0.5)*drya
        elecfld(ll)=0.0
        zmax(ll)=1000.
 100  continue
      elecfld(0)=0.0
      zmax(0)=1000.
      rya(0)=0.
      rya(lrza+1)=1.

c.......................................................................
c     lrorsa arrays
c.......................................................................
      do 105 ll=1,lrorsa
cBH080122         irzplt(ll)=ll
         irzplt(ll)=0
 105  continue

c.......................................................................
c     nva arrays
c.......................................................................
      do nn=1,nva
         thet1(nn)=90.
         thet2(nn)=180.
         thet1_npa(nn)=90.
         thet2_npa(nn)=180.
         rd(nn)=zero
         thetd(nn)=zero
         x_sxr(nn)=zero
         z_sxr(nn)=zero
         rd_npa(nn)=100.d0
         thetd_npa(nn)=zero
         x_npa(nn)=zero
         z_npa(nn)=zero
      enddo
         rd(1)=100.d0


CDIR$ NEXTSCALAR
      mpwrsou(0)=1.
      npwrsou(0)=2.
      do 11 k=1,ngena
         torloss(k)="disabled"
         lbdry(k)="conserv"
         lossfile(k)="./prompt_loss.txt"
         lossmode(k)="disabled"
         enloss(k)=200.
         tauegy(k,0)=-1.
         negy(k)=0.0
         megy(k)=0.0
         regy(k)="disabled"
         gamegy(k)=0.0
         eparc(k,0)=-1.
         eperc(k,0)=-1.
         mpwrsou(k)=3.
         npwrsou(k)=2.
         ntorloss(k)=0.0
         mtorloss(k)=0.0
         paregy(k)=0.0
         peregy(k)=0.0
         pegy(k)=0.0
         tauloss(1,k)=0.
         tauloss(2,k)=0.
         tauloss(3,k)=0.
         do 8 i=1,6
            fpld(i,k)=0.
 8       continue
         fpld(7,k)=0.
         fpld(8,k)=1.d10  !YuP[2022-12] was 1.e10
         fpld(9,k)=0.
         fpld(10,k)=pi
         do ll=1,lrza
            tauegy(k,ll)=-1.
            eparc(k,ll)=-1.
            eperc(k,ll)=-1.
         enddo
 11   continue
      do i1=1,negyrga
         do i2=1,2
            do k=1,ngena
               do ll=1,lrza
                  eegy(i1,i2,k,ll)=zero
                  jegy(i1,i2,k,ll)=0
               enddo
            enddo
         enddo
      enddo
      mpwr(0)=1.
      npwr(0)=2.
      do 10 k=1,ngen+nmax
        mpwr(k)=3.
        npwr(k)=2.
 10   continue
      do 9 k=1,ngen+nmax
        kpress(k)="enabled"
        kfield(k)="enabled"
 9    continue
      qsineut="disabled"
      trapmod="disabled"
      trapredc=0.
      scatmod="disabled"
      scatfrac=1.
      do 21 k=1,ngen+nmax
        reden(k,0)=1.
        denpar(k,0)=1.
        temp(k,0)=1.
        temppar(k,0)=1.
        reden(k,1)=1.
        denpar(k,1)=1.
        temp(k,1)=1.
        temppar(k,1)=1.
        bnumb(k)=1.
        fmass(k)=1.e-29
        nkconro(k)=0
        kspeci(1,k)=" "
        kspeci(2,k)=" "
        do ll=2,lrza
           reden(k,ll)=1.
           temp(k,ll)=1.
        enddo
        do ll=2,lsa+1
          if (cqlpmod .ne. "enabled") then
            denpar(k,ll)=1.  !Only used for CQLP
            temppar(k,ll)=1. !Only used for CQLP
          endif
        enddo
 21   continue
      nkconro(1)=1
      nkconro(2)=2
      nnspec=1
c..................................................................
c     Profile options are "parabola", "splines", and "asdex":
c..................................................................

      iprone="parabola"
      iprote="parabola"
      iproti="parabola"
      iprozeff="disabled"
      iprovphi="disabled"
      iproelec="parabola"
      ipronn="disabled"
      iprocur="parabola"
      tmdmeth="method1"
      njene=0
      njte=njene !Not used
      njti=njene !Not used
      enescal=1.
      tescal=1.
      tiscal=1.
      zeffscal=1.
      !Used for iprozeff='curr_fit' only:
      zrelax=    0.5d0 ![2020-11-01] !For iprozeff='curr_fit' only
      zrelax_exp=1.d0  ![2020-11-01] !For iprozeff='curr_fit' only
      elecscal=1.
      vphiscal=1.
      bctimescal=1.d0
      bctshift=0.d0

c..................................................................
c     acoef's specify ASDEX exponentail profiles
c     (ti profiles given by te profile, for "asdex" option).
c..................................................................

      acoefne(1)=-1.87
      acoefne(2)=-0.57
      acoefne(3)=-24.78
      acoefne(4)=-181.38
      acoefte(1)=7.51
      acoefte(2)=-13.45
      acoefte(3)=6.21
      acoefte(4)=-125.64

      zeffin(0)=1.0d0
      vphiplin(0)=0.0d0
      do 12  i=1,njenea
        ryain(i)=0.0d0
        elecin(i)=0.0d0
        tein(i)=0.0d0
        tiin(i)=0.0d0
        zeffin(i)=1.0d0
        vphiplin(i)=0.0d0
        difin(i)=0.0d0
 12   continue

      do k=1,npaproca
         npa_process(k)="notset"
         ennl(k)=5.0
         ennb(k)=1.d10  !YuP[2022-12] was 1.e10
         do  i=1,njenea
            ennin(i,k)=0.0d0
         enddo
         ennscal(k)=1.0
      enddo
      npa_process(1)="cxh"

      do 13 k=1,ntotala
        do 14 i=1,njenea
          enein(i,k)=0.0d0
 14     continue
 13   continue

c..................................................................
c     Time-dependent profile quantities
c..................................................................

      nbctime=0
      do 30 i=1,nbctimea
         do 31 k=1,ntotala
            redenc(i,k)=0.d0
            redenb(i,k)=0.d0
            tempc(i,k)=0.d0
            tempb(i,k)=0.d0
 31      continue
         bctime(i)=dfloat(i-1)
         zeffc(i)=0.d0
         zeffb(i)=0.d0
         elecc(i)=0.d0
         elecb(i)=0.d0
         vphic(i)=0.d0
         vphib(i)=0.d0
         xjc(i)=0.d0 !for iprocur.eq."prbola-t" (not used in "spline-t")
         xjb(i)=0.d0 !for iprocur.eq."prbola-t" (not used in "spline-t")
         totcrt(i)=0.d0 !target current, set to ne.0. for efswtch=method2,3,4
 30   continue

      do 32 i=1,nbctimea
         do 33 l=1,njenea
            do 34 k=1,ntotala
               enein_t(l,k,i)=0.0d0
 34         continue
            tein_t(l,i)=0.0d0
            tiin_t(l,i)=0.0d0
            zeffin_t(l,i)=0.0d0
            elecin_t(l,i)=0.0d0
            xjin_t(l,i)=0.0d0 !for iprocur.eq."spline-t"
            vphiplin_t(l,i)=0.0d0
 33      continue
 32   continue

      ! For calc. of 1st-order orbit shift
      nr_delta=65
      nz_delta=65
      nt_delta=80 !Needs to be even
      ! For saving f4d== f(R,Z,u,theta) distribution:
      nr_f4d=20
      nz_f4d=21
      nv_f4d=20
      nt_f4d=20
      
      ! For saving f3d== f(pol.angle,vpar,mu) at fixed rho==f3d_rho :
      npol_f3d=20
      nvpar_f3d=200 ! 
      nmu_f3d=100 ! number of grid points in mu=0.5*m*Vperp^2/B
      f3d_rho=0.5d0 ! f3d is saved at this rho value
      nen=nena
      nen_npa=nena
      mmsv=mx
      msxr=mx
      npaproc=1
      nv=1
      nv_npa=1
      enmin=5.
      enmax=50.
      fds=0.2
      enmin_npa=5.
      enmax_npa=50.
      fds_npa=0.2
      soucoord="disabled"
      nsou=1
      pltso="enabled"
      flemodel="fsa"
      knockon="disabled"
      komodel="mr"
      nkorfn=1
      nonko=10000
      noffko=10000
      soffvte=3.
      soffpr=0.5
      do 19 k=1,ngena
        do 20 m=1,nsoa
          nonso(k,m)=100000
          noffso(k,m)=100000
          do ll=1,lrza
             asor(k,m,ll)=0.
          enddo
          sellm1(k,m)=1.
          sellm2(k,m)=1.
          seppm1(k,m)=0.
          seppm2(k,m)=1.
          sem1(k,m)=0.
          sem2(k,m)=0.
          sthm1(k,m)=0.
          scm2(k,m)=0.
          szm1(k,m)=0.
          szm2(k,m)=1.
 20     continue
 19   continue

c     Some specific settings from cqlinput_help


cBH080125  DON'T reset this, as it conflicts with past
cBH080125  usage of asor.
cBH080125      do ll=1,lrza
cBH080125         asor(1,1,ll)=.25e+13
cBH080125         asor(1,2,ll)=3.25e+13
cBH080125      enddo

      scm2(1,1)=.001
      scm2(1,2)=10000.
      sellm1(1,1)=1.
      sellm1(1,2)=1.
      sellm2(1,1)=1.
      sellm2(1,2)=1.
      sem1(1,1)=1600.
      sem1(1,2)=0.
      sem2(1,1)=.5
      sem2(1,2)=25.
      seppm1(1,1)=1.
      seppm1(1,2)=1.
      seppm2(1,1)=1.
      seppm2(1,2)=1.
      sthm1(1,1)=5.
      sthm1(1,2)=0.
      szm1(1,1)=0.
      szm1(1,2)=0.
      szm2(1,1)=1.e+5
      szm2(1,2)=1.e+5

      do k=1,ngena
         do m=1,nsoa
            do ll=0,lrza
               sellm1z(k,m,ll)=sellm1(k,m)
               sellm2z(k,m,ll)=sellm2(k,m)
               seppm1z(k,m,ll)=seppm1(k,m)
               sem1z(k,m,ll)=sem1(k,m)
               sem2z(k,m,ll)=sem2(k,m)
               sthm1z(k,m,ll)=sthm1(k,m)
               scm2z(k,m,ll)=scm2(k,m)
               szm1z(k,m,ll)=szm1(k,m)
               seppm2z(k,m,ll)=seppm2(k,m)
               szm2z(k,m,ll)=szm2(k,m)
            enddo
            asorz(k,m,0)=0.d0 !YuP[2022-12-13] was 0.
            do ll=1,lrza
               asorz(k,m,ll)=asor(k,m,ll)
            enddo
         enddo
      enddo

      

c.......................................................................
cl    4. Output option arrays
c.......................................................................

      do 400 i=1,noutpta
        nlotp1(i)=.false.
        nlotp2(i)=.false.
        nlotp3(i)=.false.
        nlotp4(i)=.false.
 400  continue
      nlotp1(4)=.true.
cBH070414      nlotp1(4)=.true.

c.......................................................................
c     5. Others, sometimes initialized later, but better do it before
c     reading namelist
c.......................................................................


c$$$c.......................................................................
c$$$c     6. Default values for finite orbit width (FOW) calculations
c$$$c.......................................................................
      fow="disabled" ! "disabled" is to use ZOW model 
                     ! as the main model in CQL3D
c$$$      outorb="Not-detailed" ! outorb='detailed' or 'Not-detailed'
c$$$                ! (saving/not-saving data to a file for plotting)  
c$$$      nmu =100  ! grid sizes for ad.ivariant mu 
c$$$      npfi=100  ! and canonical momentum Pfi; 
c$$$                ! to setup COM->R lookup table.
c$$$      nsteps_orb=50000 ! Max.number of time steps for orbit integration.
c$$$                ! Also used to trace Pfi=const levels for COM->R table
c$$$                ! in order to find intersections with mu=const levels.
c$$$      nptsorb=1 ! Number of points on a complete orbit 
c$$$                ! (ityp=0 "main" orbit)
c$$$                ! from which ityp=1 "secondary" orbits are launched.
c$$$                ! ityp=1 orbit is stopped when it reaches the midplane.
c$$$                ! (Note: secondary orbits are not traced usually, 
c$$$                ! see below, iorb2=0)
c$$$      i_orb_width=1 ! 1 -> Normal finite-orbit-width calculations. 
c$$$                    ! 0 -> V_drift_perp is set to 0 
c$$$                    ! (to mimic ZOW approximation)
      return
      end
