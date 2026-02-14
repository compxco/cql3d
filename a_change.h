c
c     a_change.h
c
c
c***********************************************************************
c
c     This file documents changes in the code 
c
c***********************************************************************


------------------------------
version="git_cql3d_251215.2"
------------------------------
c[364] Changed dimensions of arrays ilowp,iupp,ifct1_,ifct2_
  to be ilowp(ipack16_,nrayn,mrfn), etc.
  where ipack16_=(jjx/ibytes16)*nrayelts +1,
  a much smaller size than ipack16=(jjx/ibytes16)*nrayelts*nrayn +1
  previously used. This change is important for runs where 
  many rays and many ray elements are used, and then ipack16 may overflow.
  (This issue was partially addressed in [363], but was not sufficient)
  See [2026-01-24]

c[363] Modification of locatn16 (related to pack16/unpack16)
  which originally was set as 
  locatn16=(jjx*(is-1)+jjx*nrayelts*(iray-1))/ibytes16 +1
  Per Grant Rutherford and John Wright, this version 
  helps to avoid overflow when number of rays 
  and ray elements is very large:
  locatn16=(jjx/ibytes16)*((is-1)+nrayelts*(iray-1)) +1
  (i.e., perform (jjx/ibytes16) division first to reduce the factor)
  See [2026-01-24]

c[362] A fix of issue in ainsetva related to NFREYA setup.
  Added kopt, so that ainsetva(kopt) is called twice:
  kopt=1 - verify all vars NOT related to FREYA,
  kopt=2 - verify vars ONLY related to FREYA.
  Reason: originally, ainsetva was called once, 
  BEFORE frset and frnfreya
  where the values of frmodp, etc., are defined.
  This could result in wrong outcome.
  Now it is called again, after frnfreya, with kopt=2.
  See YuP[2024-02-27]

c[361] Fixed an issue related to kfrsou.
  For each beam 'ib' there should be an associated species 'k',
  set with values of kfrsou(ib) in cqlinput.
  In old version of CQL3D the namelist variable kfrsou was a scalar,
  and its value was applied for all beams.
  Now kfrsou() is a vector, 
  and default values are set to kfrsou(:)=0.
  Therefore, old cqlinput where several beams are used 
  and kfrsou (as a scalar) is set to 2 
  (which means - 1st and 2nd beams are pointing to species k=2)
  would fail now.
  The issue is fixed by resetting kfrsou(ib)=0
  to kfrsou(ib)=kfrsou(ib-1) [value of 'k' from previous beam].
  See frinitz.f, around "[2025-12-16]" 

------------------------------
version="git_cql3d_251215.0"
------------------------------

c[360] Few adjustments, most important is related to cases 
  when impurity species is added (to get the specified Zeff).
  Maxwellian electrons are moved by one index;
  it was done for central/edge values but not for powers 
  in parabolic profiles. Added: 
     !YuP[2025-12-12] Also need to move up powers npwr and mpwr:
     mpwr(kelecm)=mpwr(kelecm-1) !For T_e_maxw
     npwr(kelecm)=npwr(kelecm-1) !For T_e_maxw
  Track changes by "[2025-12-12]"

------------------------------
version="git_cql3d_230113.4"
------------------------------

c[359] Corrected plot for E(rho) field at selected time steps -
 the integration step and the time frame for this step are
 properly identified and printed. The E field is set at the beginning
 of each step. See notes in tdpltmne.f, around "YuP[2024-08-08]"

c[358] Revised appearance of some plots in tdpltmne,
 to have radial profiles as a func of rho and func of R in one page,
 to have both the midplane and FSA profiles, etc.
 YuP[2024-08-01]
 
c[357] For NIMROD coupling, added radcoord="polflx" 
 and "sqpolflx" (through psi_data). 
 In subroutine read_data_files -
 added detection of irow_lcfs,
 which is found as the last irow where psi_data>0
 (psi_data<0 is outside of LCFS) 
 See YuP[2024-07-25] 
 
c[356] The crash reported by S.G. Baek (MIT) -
 when using soln_method='it3drv' and transp='enabled'.
 The code hangs in MPI run at the last time step. 
 Tracked the issue to "do 30" loop in tdchief,
 particularly because of these two calls 
 (they occur at n=nstop only):
 call tdtrfcop(1) !copy frn-->f, and then call diaggnde
 If soln_method='it3drv", frn (and fvn) is not used,
 and then gn becomes 0.0 in diaggnde.
 Once gn=0.0, the "stop" is triggered at this mpi core,
 or the code may hang in diaggnde.
 Added if(soln_method.ne.'it3drv')
 condition around calls to tdtrfcop(*)
 and diaggnde in the "do 30" loop.
 See YuP[2024-07-25] 


------------------------------
version="cql3d_git_230113.2"
------------------------------

c[355] Enabled 'f4d_out' functionality, added 'f3d_out',
 both adapted from CQL3D-FOW version. 
 [2023-09-27]

c[354] Correction in tdnpa0, similar to YuP[2021-11-05] in tdsxr0,
 related to stepping along sightlines outside of plasma.
 YuP[2023-08-10]

c[353] Adjusted subroutine getsgxn which is used for NBI deposition
 in rotating plasma. See comments around [2023-07-11].

c[352] Bug in subr.synchrad - should NOT set arrays synca and syncd to 0
 [as in call_bcast(synca,zero,iyjx*lrz)]
 This subroutine is called for each lr, and bcast(...zero...)
 was wiping data for all other lr !
 YuP[2023-06-30]
 
c[351] Added an option to read elecfld() array from previous run,
  in case of a restart run. See tdreadf.f.
  This option was already present for ampfmod='enabled',
  but it is needed for efswtch.eq."method6", too.
  YuP[2023-02-02]

c[350] Changes/bug fixes in netcdfrw2.f, subr.netcdfvec,   
  related to saving of flux_vectors in velocity space.
  Made by Seung-Gyou Baek [2023-02-02],
  see comments marked with "sgbaek".

------------------------------
version="cql3d_git_230113.0"
------------------------------

c[349] Another transfer of changes made in CQL3D-m version 
  to this mainstream version.
  Revised how certain arrays for FREYA are populated -
  zti(), zni(), zzi(). Previously, the primary ion species kk=1:nprim 
  for FREYA were selected among Maxwellian_ion list in cqlinput,
  if available (in case of colmodl=3, these species are also present in 
  general_ion list); if not available - they are taken 
  from the general_ion list.
  Now they are selected from general_ion list,
  among first nprim ions. See cqlinput_help around 'nprim' and 'nimp',
  also see subr.frstup for details.  
  This change can make a difference for runs with colmodl=3,
  where general ions are repeated in the Maxwellian_ion list.
  If the density of general ions is allowed to grow (as a result of 
  NBI source), and the NBI deposition is updated at each time step, 
  it is important that the density of primary ions zni(kk=1:nprim)
  is also updated. If those ions are referred to the Maxwellian_ion
  list (as was done previously), their density is not updated. 
  YuP[2023-01-10]

c[348] YuP[2023-01-03] Made entrintr(:,:) array a pointer. 
  Previously - static array, it was causing floating-point exception
  and other issues (as in subr. diagentr_vol, see comments cBH120807,
  cBH050616).

c[347] Transfer of changes made in CQL3D-m version 
  to this mainstream version. Mostly related to FREYA modules.
  Enabling multiple beams capability (now kfrsou(1:nbeams) 
  should be set for each beam), enabling capability to update 
  NBI deposition at every time step.
  YuP[2022-12-13..30] As a result of debugging - 
  Many changes in Freya-related subroutines 
  (reason: runs in gfortran fail with NaNs, when NBI deposition
  is updated at every time step).
  See frinitl.f, freya.f, frnfreya.f and zfreya.f.
  It is found that the problem is mostly from
  missing initializations (or wrong index range 
  during initialization - search "BUG?" for details);
  also, because of not-saved arrays that are defined during
  initial call to freya/zfreya and then reused.
  Most important changes are in subr. adasqh6.
  Also see [2022-11-28] for resetting dummy=RANDOM_my(ranseed)
  at each new call (each time step).  

c[346] For the read_data coupling functionality, added an option
  for setting the value of E field:
  When elecfld_data='readdata', get E field directly from data file.
  When elecfld_data='jspitzer', get E as j_data/resistivity
  (the latter version was the only option until this change).
  !YuP[2022-07-13]

c[345] Adjusted subr. tdpltjop - 
  The printout of table with powers and plots of p_RF profiles
  were skipped in some cases,
  because of if(sorpwtza.le.1d-25) check, 
  and because sometimes sorpwtza is negative 
  (from instability in distr func).
  However, powurf is positive, at least for most of krf modes.
  Changed to checking abs(sorpwtza)
  instead of sorpwtza itself, and also
  added maxval(powurf) for the if() condition.
  YuP[2022-05-10] 


c[344] version="cql3d_git_210125.3"
   [Includes all changes related to CQLP 
    and also changes related to NIMROD coupling] 


c[344] Added 'fl' into netcdfrw2.f, which saves
c[344] the value of reduced parallel distr. function F_par(v_par),
c[344] and saves 'xl' = the normalized parallel momentum.
c[344] It can be saved at each time step (use netcdfshort="long_jp"),
c[344] otherwise it is saved at last time step only.
c[344] The logic for saving is same as for 'currv' array,
c[344] and the size of 'fl' is (1:jfl,1:lrors),
c[344] so it would not take much space in mnemonic.nc file.
c[344] See [2022-03-19] 


c[343] version="cql3d_git_210125.2"
       Moved rescaling of time-dependent density
       from profiles.f to tdxinitl.f, where it is done like this:
         redenc(1:nbctimea,1:ntotala)= enescal*redenc(:,:)
         redenb(1:nbctimea,1:ntotala)= enescal*redenb(:,:)
         enein_t(1:njenea,1:ntotala,1:nbctimea)=enescal*enein_t(:,:,:)
       Now the above namelist variables are rescaled at once,
       for all nbctime steps,
       and there is no need to rescale "renden" in profiles.f
       ![2022-03-10] 


c[342] Small corrections in tdsxr*, tdnpa* diagnostics,
   related to sightlines. Search "YuP[2021-11"

c[341] Extensive changes related to CQLP branch 
   of the code (cqlpmod="enabled"). It works now,
   and passes conductivity tests quite well.
   Not tested for MPI runs, needs checking.
   Some of changes may affect CQL3D, e.g., related to conversion of 
   many fixed-size arrays to pointers, 
   changing iy to iy_(l_) in many places, etc.
   In param.h, the parameter lza is not needed anymore
   (just set lz in cqlinput; default is set to lz=40).
   Also, lsa can be smaller than lrza now.  
   Track changes by "YuP[2021-03", "YuP[2021-04", "YuP[2022-02", "YuP[2022-03" 

c[340] Added automatic search of data files (from NIMROD).
    In cqlinput, it is only necessary to specify:
      read_data_filenames_prefix= 'rycq0'  (example; default is "notset")
      read_data_filenames_suffix= '00.dat' (example; default is "notset")
    These two variables are used by the code to form
    a template for file names. 
    Example: Consider data file 'rycq019800.dat', 
    it is made of prefix 'rycq0',
    followed by 3-digit number and suffix '00.dat'
    The code will try to find all files from  rycq000000.dat 
    to rycq099900.dat (max 1000 files, for now); 
    (the sequence may have gaps in numbering range 000:999).   
    YuP[2021-04-06]  also search "YuP[2021-08", related to Drr definition.
    Also YuP[2022-01-06]

c[339] version="cql3d_git_210125.1"
    New option for reading data files with plasma profiles
    computed with other codes;
    presently setup for coupling with NIMROD only.
    See the new namelist var. read_data (presently can be 
    'disabled' or 'nimrod'; other options can be added).
    Related namelist variables:
    read_data_filenames() is the list of data files from NIMROD,
    with FSA values of E field, Te, Ti, density of D+,
    densities of all ionization states of impurity (e.g., Neon), etc.
    Also see temp_min_data= Lower limit, to adjust Te and Ti data.
    Main work is done by new subr. read_data_files().
    BH,YuP[2021-01-21],[2021-01-22]
    

c[338] Corrected rftype(mrf)="ech" to "ec" in urfinitl.f
    and urfsetup.f, to match the logic in urfread.f and cqlinput_help.
    Also adjusted settings for jfl (should be odd number)
    in ainsetva, and logic related to fl() array in pltprppr.f.
    See YuP[2020-12-22]

c[337] For ilowp() and iupp() arrays -
    changed pack-->pack16(which uses integer*2),
    also  unpack-->unpack16, 
    and allocation of those arrays - accordingly. 
    Also removed a range in indexes in the above arrays
    when using as the first arguments of unpack16 subroutines.
    Note: The 1st argument in subr.unpack is integer*1,
    and the 1st arg, in unpack16 is integer*2,
    while ilowp,iupp,ifct1_,ifct2_ are integer*4 by default.
    This is done by design of unpack and unpack16 subroutines.  
    BH,YuP[2020-12-18]

c[336] Added  lbdry(k)="conscalm" to add Maxwellian particles keeping
    the density constant, as an alternative to scaling f, for the 
    time-indep background case, nbctime=0 . Useful, for example, in
    Dreicer runaway rate calcs. BH[201209]. 
    

c[335] version="cql3d_git_201207.0"
c[335] Fixed several bugs in ADC** subroutines.
    Most important - Cannot use 45. in ADCEUND(-ZXN,-45.)
    The argument should be real*8, ADCEUND(-ZXN,-45.d0).
    It messed up the results in NERSC(Cray/Intel) runs.
    Search for YuP[2020-11-15] and YuP[2020-11-16]
    for other changes.
    Also adjusted achiefn.f and ainpla.f
    for calls of some subroutines. See YuP[2020-11-15] in there.


c[334] Adjusted plots related to SXR diagnostics.
    Some of subroutines are used for both NPA and SXR plots,
    such as subr. tdsxrplt and tdsxrvw. The labels in plots
    should depend on (softxry.ne."disabled")
    or (npa_diag.ne."disabled"). Note that softxry and npa_diag
    may have many values, not just 'enabled'.
    [2020-11-02]

c[333] Modified iprozeff="curr_fit" procedure, which changes Zeff
    in such a way as to maintain the target current.
    Two new namelist variables are added: zrelax (default is 0.5d0)
    and zrelax_exp (default is 1.d0). The procedure is similar to 
    that used in efswtch.eq."method4". See cqlinput_help for description.
    BH,YuP[2020-11-01]

c[333.1] Changed the default impurity ion species from charge/mass=50./50.*m_D
c[333.1] to 100./100*m_D in ainsetva.f on 201030.
c[333.1] This is a non-backwards compatible change in the code
c[333.1] and creates somewhat modified results in cases
c[333.1] where an ion species is added to attain a target Zeff.
c[333.1] That is, with iprozeff.ne."disabled" and only 
c[333.1] one ion species bnumb() is given.  See cqlinput_help. [BH: 210609]

c[332] version="cql3d_git_200101.3"


c[332] Added control of printout to the screen, 
    using the namelist variable ioutput(1) [it was present but not used].
    After 2020-10-21, the usage of ioutput(1):
    ioutput(1)=0  means the printout to screen is reduced to a minimum,
                  leaving namelist printout, warning messages,
                  and physics-related values used or computed by code.
                  (default value ioutput(1)=0)
    ioutput(1)=1  The above, and more printout - 
                  values of arrays for diagnostic purpose.
    ioutput(1)=2  (or .ge.2) The above, and even more diagnostic printout.
    ioutput(2)=0 ! No usage at present [2020]
    BH,YuP[2020-10-21]


c[331] Corrected usage of enerkev(1:jx,k) (explicit index k)
    in subr. tdtrdfus and netcdfrw2. 
    Removed referencing (lossmode(1).eq.'simplbn1') in if-endif 
    statements.    !YuP[2020-10-20]

c[330] Added checking of denominator=0 in subroutines soup and sourc0.
    For example, cosm2(kk,m,lr_) can be 0 when scm2(kk,m)=0. 
    (scm2 is set in aindflt: scm2(k,m)=0., for k>1 species).
    It resulted in soupp(j,lr) array getting NaN in runs with ngen=2.
    YuP[2020-10-20]

c[329] Adjusted subr.tdplteq(krf) and tdchief (around call of tdplteq)
    Now it is called for each wave type (was: each mode). YuP[2020-10-19]

c[328] An adjustment in urfbplt.f.  When subr.ufrbplt is called, 
    the time step is not advanced yet,
    so if nstop=2, then n in (n.eq.nplt3d(i)) can only get to 1.
    It means that if nplt3d(i) is set to nstop,
    the condition n.eq.nplt3d(i) is never satisfied.
    Changed (n.eq.nplt3d(i)) to (n+1.eq.nplt3d(i)).
    YuP[2020-10-02]

c[327] Corrected error in subr.netcdfrf(kopt,krf)
  related to recording of data for each wave type 'krf'.
  Such arrays as freqcy, delpwr, cwexde, etc, -
  they are not functions of krf (wave type; Example: krf= 1:mrf= 1:2),
  but rather they are functions of krfn (wave modes; Example: mrfn=3+3).
  Initially, they are read for each krf, 
  but then duplicated into each krfn index.
  This is done in subr.urfread, at the end.
  But subr.netcdfrf(kopt,krf) is called with krf,
  for recording of data (when kopt=0 or 1).
  Then, we need to record them like cwexde(is,iray, irfn(krf)) 
  [before this correction, it was recorded as cwexde(is,iray,krf)],
  where irfn(krf=1) =1 and irfn(krf=2) =4 in our example.
  Index irfn(krf) is pointing to beginning harmonic of each wave type;
  It is the index (in 1:mrfn) of the lowest harmonic for each wave type.
  !YuP[2020-09-23]  

c[326] Small changes in plotting subroutines to improve limits  
    in vertical axis. Also adjusted a warning printout 
    from function lug(), during check of
    requirement for input array to be strictly increasing; allow 
    a certain accuracy (1.d-8*abs(parray(i)) as a margin of error.
     [YuP 2020-08] 


c[325] version="cql3d_git_200101.2"

c[325] New option eqsource="miller"   - 
    Analytically defined equilibrium, D-shape. 
    All related namelist variables start with "eq_miller_" prefix.
    REF: R.L. Miller et al., "Noncircular, finite aspect ratio, local
    equilibrium model", Phys. Plasmas, Vol. 5, No. 4, April 1998,
    See Eqs.36-37.  
    YuP[2020-07-13] [mostly, it was imported from CQL3D-FOW version,
    but then upgraded to D-shape; in FOW version
    it was valid for ellipse only, i.e. for eq_miller_deltaedge=0.
    Also added a more general polynomial profile of PSI(r).]

c[324] YuP[2020-07-02],[2020-07-16]
    Added subroutines cfp_integrals_maxw and cfp_integrals_get.
    Subr. cfp_integrals_maxw calculates certain integrals 
    needed for subr. cfpcoefn.
    These integrals describe a contribution to BA coll.coefs
    from local collisions of general species with the background 
    Maxwellian species (search for sections after "kbm=ngen+1,ntotal").
    These integrals only depend on mass (fmass) 
    and local temperature of these (Maxwellian) species.
    So, instead of calculating them over and over again,
    calculate them once as a table over temperature grid 
    (search "T-grid"), and then reuse them by matching a local T
    with the nearest values in the T-grid.
    These integrals are updated if the temp() of 
    the Maxwellian species is varied in time (in a prescribed form). 
    This subr. is called from tdchief, just after call_profiles.
    It is adapted from cql3d-fow version, then extended
    for impurity ions with all ionization states.
    Search for "call cfp_integrals_get".
    This option is accessed by setting cfp_integrals='enabled'
    in cqlinput (default is 'disabled').
    This option can give some speedup in case of high-Z impurity.


c[323] Added a new option for method of searching 
       for the flux surface in subr.eqfndpsi.
       The new method makes iterations in major radius position
       for the start of flux surface. The old method is based on
       iterations in values of poloidal flux, and it is
       not accurate in area near magnetic axis where the profile
       of PSI(r) is nearly flat. See a long comment in the 
       beginning of subr.eqfndpsi.  Search for "YuP[2020-06-30]"
       to find all changes related to this option.
       [Not a namelist option yet; need to change line 119 
       in subr.eqfndpsi(), use kopt=2 to access this new option.]

c[322] Added new option for gamaset (version of Coulomb log):
       gamaset=-1.  means: use NRL definitions for gama(kk,k). 
       YuP[2020-06-23]

c[321] Added contribution from partially ionized impurities to 
       Bremsstrahlung energy loss term. See the end of lossegy.f,
       with comments on "c2= r02_alf*(z_imp**2 + z_bound)" definition.
       Also added the call to subr.lossegy from subr.achiefn 
       (was only from ainitial, at n=0); 
       the loss term (egylosa() array) must be recomputed
       at each time step, - because Maxw.ion density could change;
       and impurity content could change.
       YuP[2020-06-22]
       
c[320] Corrected error in subr.synchrad, in B-curvature related terms.
       Coefficient A should be proportional to gamma (was ~1/gamma).
       See YuP_2020_Synchrotron_derivations.pdf,
       YuP[2020-06-18] 

c[319] Added efswtch="method6" --
       At t=0, save the value of elecfld and conductivity.
       At later time step, use spitzer+neoclassical conductivity
       to evolve electric field so that 
       E(t) = E(t=0)* sigma(t=0)/sigma(t)
       [it is aimed to yield j(t)=const, if no j_RE]
       !YuP[2020-04-13]

c[318] Added 'favr_thet0' into netcdfrw2.f, which saves
c[318] the value of INTEGR{f*2pi*sin(theta0)dtheta0} /4pi,
c[318] 'Distr function at midplane averaged over pitch angle'.
c[318] It can be saved at each time step (use netcdfshort="long_jp"),
c[318] otherwise it is saved at last time step only.
c[318] The logic for saving is same as for 'currv' array,
c[318] and the size of 'favr_thet0' is same as for 'currv',
c[318] so it would not take much space in mnemonic.nc file.
c[318] See YuP[2020-04-07] 

c[317b] For efswtch="method4", added a generalized procedure for 
       relaxation of E-field:
         elecfld==> elecfld*(1.-efrelax*sign(djrel*currxj)*|djrel|**efrelax_exp)
       where djrel= (curr-currxj) / max(|curr|,|currxj|)
       Presently this generalized procedure is only available for 
       efswtch="method4", but could be extended to other "method#".
       Default value is efrelax_exp=1.d0, which gives
       the original procedure.
       YuP[2020-04-03]
       
c[317a] Modified how current density is calculated in subr.efield
       for efswtch.eq."method4" and "method5" (at n=0). 
       The original version was using resistivity 
       zreshin [Hinton and Hazeltine] from subr.resthks.
       The new version uses resistivity zressau1
       based on Sauter, Angioni and Lin-Liu, 
       Phys.Plasmas 6,1834(1999). Same change is made in 
       section if(efswtchn.eq."neo_hh") in subr.efield.
       This part is more important for the evaluation of E field,
       since it uses a correction of E field based on 
       analytical values of resistivity. This "neo_hh" 
       correction is applied to efswtch= "method2"--"method5".
       Expect changes in runs that use efswtchn="neo_hh",
       such as wh80 test with runaway electrons. YuP[2020-04-02]

c[317] Adjusted sub.profiles, tdinitl and tdchief for the proper
c[317] functionality of iprozeff='curr_fit' option.
c[317] Search "YuP[2020-02-11]" for all related changes and comments.

c[316] Corrected bugs related to saving currv() and pwrrf() 
c[316] into *.nc file at every time step 
c[316] (i.e. with netcdfshort='long_jp' option).
c[316] YuP[2020-02-06]

c[315] Added arrays to be sent/recv in MPI runs. See mpins_par.f
c[315] and tdchief.f.
c[315] YuP[2020-02-05]

c[314] Added more arrays needed for eqmod='disabled'
c[314] (analytical magnetic equilibrium), and adjusted subr. tdplteq
c[314] so it can plot FP surfaces in case of eqmod='disabled'
c[314] (if psimodel='spline'). See aingeom.f, tdplteq.f. 
c[314] YuP[2020-01-31]


c[313] version="cql3d_git_200101.1"
c[313] Corrected sourceko.f. There was procedure for reversing 
c[313] the direction of source if the electric field is positive,
c[313] and also removing the portion of the source at upar>0
c[313] in this case. There is no need to do this.
c[313] The primary RA electrons (that produce the secondary RA electrons, 
c[313] i.e., the KO source) can have both upar>0 and upar<0,
c[313] even for passing electrons. They can originate from hot-tail seed,
c[313] which produces primary RE at all pitch angles.
c[313] YuP[2020-01-30] 

c[312] Added a setup for some arrays needed for eqmod='disabled'
c[312] (analytical magnetic equilibrium). See aingeom.f, micxiniz.f,
c[312] tdrmshst.f, tdtoarad.f. Search for "YuP[2020-01-29]"
c[312] Also, in ainsetva.f, added the printout of warning message for
c[312] "Strongly recommended to use psimodel="spline" "
c[312] in case of eqmod='disabled'
c[312] Particularly, some arrays that are used in ampfmod calculations
c[312] in case of eqmod="disabled" are only setup when psimodel="spline".
c[312] YuP[2020-01-29]

c[311] In achiefn.f, for the case of kopt.eq.3 (solving Amp-Far eqns) 
c[311] added (it_ampf.eq.1) in front of "call sourcee"
c[311] which means  reuse sources from 1st iteration.
c[311] Similarly, in front of "call cfpcoefc" 
c[311] which means reuse coll.coeffs from 1st iteration.
c[311] This is just to save cpu time.
c[311] YuP[2020-01] 
        
c[310] Corrected error in sourcee.f, related to ampfmod.
c[310] Initialization of source(), xlncur() should be done only at 
c[310] it_ampf=1 (for ampfmod.eq."enabled" case).
c[310] That is, when ampfmod.eq."enabled", 
c[310] initialize the two arrays only at 1st iteration, 
c[310] and then reuse them at higher iterations, i.e., it_ampf>1.
c[310] Note: there is if(ampfmod.eq."enabled" .and. it_ampf.gt.1)return
c[310] clause in sourceko.f. 
c[310] So, if we don't use a corresponding clause in sourcee.f,
c[310] then at it_ampf>1 the arrays source(), xlncur(1,lr_)  
c[310] will be set to 0.0 but not computed in sourceko.f.
c[310] YuP[2020-01]

c[309] Added subroutine profaxis1(e_out,expn1,expm1,e0,e1,rova)
c[309] where instead of dratio=e(1)/e(0) value, we use 
c[309] both of these values (e(1)==e1 is for the plasma edge, 
c[309] and e(0)==e0 is for the plasma center).
c[309] This modification allows cases of e0=0.0.
c[309] On output, use, e.g.,  a(ll)=e_out  
c[309] [so, no need to have a(ll)=a(0)*rn ]
c[309] Many changes in the code, where the old subroutine was called.
c[309] Search YuP[2019-12-29] for all changes.  

c[308] Added E(0) field values (only used for plots) for the solution
c[308] of Amp-Far eqns.

c[307] For the recently introduced option iprozeff.eq."curr_fit",
c[307] added corrections for the currpar_starnue_n()-currpar_starnue0_n()
c[307] and bootstrap current. [see the two previous entries].
c[307] See changes around YuP[2019-12-27].

c[306] Added new namelist variable, ampfadd, to control corrections 
c[306] introduced into Amp-Far eqn., see the previous entrance.
c[306] Available values are  "neo+bscd" [default value],
c[306] "add_bscd", "neosigma", "disabled".

c[305] version="cql3d_git_200101.0"
c[305] BH+YuP. Added bootstrap current and delta_sigma*E current into
c[305] Amp-Far equations. See subr. ampfsoln. The coefficient ampfb()
c[305] is modified to include dbscurm(llp)+dcurr(llp), 
c[305] and coefficient ampfa() now includes dsig(llp),
c[305] where dsig= sigma_coll_neo - sigma_banana.
c[305] Also added arrays into netcdfrw2.f:
c[305] 'consn','bscurr_e_gen','bscurr_e_maxw',
c[305] 'bscurr_i_gen','bscurr_i_maxw',
c[305] 'currpar_starnue0'(Current based on sigma_banana (starnue=0)),
c[305] 'currpar_starnue'(Current based on sigma_coll-neo (starnue>0)).
c[305] Also added plots of currpar_starnue_n()-currpar_starnue0_n()
c[305] [current addition from (sigma_coll-neo -sigma_banana)*E_phi],
c[305] as radial profiles of current density, and partial integrals. 
c[305] See YuP[2019-12-18] in ampfar.f, tddiag.f,
c[305] YuP[2019-12-19] YuP[2019-12-20] in tdpltmne, 
c[305] YuP[2019-12-23] in netcdfrw2.

c[304] Modifications of bootstrap current calculations,
c[304] in tdboothi.f (Sauter fit model), tdbootst.f (Hinton-Hazeltine), 
c[304] and in bsl/bsu model (bootcalc='method1' or 'method2').
c[304] In particular, tdboothi.f now includes all coefficients
c[304] from Phys. of Plasmas 6, 2834 (1999), with Erratum, 
c[304] covering all-collisionality cases.
c[304] These corrections are migrated from CQL3D-FOW version. 
c[304] YuP[2019-12-12]
c[304] Besides, added more modifications:
c[304] Added cursign in front of bscurm() current.
c[304] Now the calculated bootstrap current bscurm() is always 
c[304] in the same direction as the Ohmic current 
c[304] (I_Ohm>0 corr. to cursign>0, which is in positive phi direction).
c[304] Positive phi direction is counter-clockwise viewed from above.
c[304] Suggestion: We could also add bootsign factor in front,
c[304] for a more general control, and to match bscurm() with current
c[304] calculated with bootcalc='method1' or 'method2' option.
c[304] See notes near YuP[2019-12-17] in tdboothi.f.
c[304] Also added calculation of starnue() directly into tdboothi.f.
c[304] The problem is that usually starnue is calc-ed
c[304] in sub.efields, but efields is only called when ampfmod.ne."enabled"
c[304] or at n=0.  
c[304] As a by-product, also added calc./update of tauee(l),taueeh(l),sptzr(l).
c[304] Tests: Almost no increase in cpu time from this addition.
c[304] See YuP[2019-12-18] in tdboothi.f.

c[303] Added new namelist variable imp_ne_method="ni_list" or "ne_list".
c[303] Method of adjusting the electron or main-ion densities
c[303] when an impurity is deposited, and densities of ionization states 
c[303] are changed in time. Default is 'ne_list'.
c[303] YuP[2019-12-06]

c[302] Added new namelist variable for the method of deposition of impurity:
c[302] imp_depos_method='disabled'[default] or "pellet" or "instant".
c[302] For imp_depos_method="instant", need to specify 
c[302] dens0_imp(0:lrz) = Density profile of deposited impurity [1/cm^3]
c[302] and tstart_imp= Instant when impurity is deposited [sec].
c[302] YuP[2019-12-05]



c[301] version=cql3d_git_190923.3
c[301] For restart nlrestrt="ncdfdist" and iprozeff.eq."curr_fit",
c[301] then added restore of the zeff profile from the prior run.
c[301] Also, a second (minor) mod: For knockon="enabled", ampfmod="enabled"
c[301] and nampfmax.gt.1 interations, then the source() from the first
c[301] call to sourceko is reused. (sub. sourceko takes significant
c[301] cpu time, and does not change significantly for the iteration
c[301] time steps. (BH191128, 191201)

c[300] version=cql3d_git_190923.2 (with update6_for_cql3d_190923.2)
c[300] Added new option iprozeff.eq."curr_fit":
c[300] Renormalize zeff() in such a way that the current from
c[300] FPE solution would be "pushed" to match the target current,
c[300] which is set in totcrt(1:nbctime) in cqlinput.
c[300] Presently zeff() is simply renormalized in profiles.f by 
c[300]   renorm= totcurtt/currxjtot,   
c[300] where totcurtt is the target current at given time step
c[300] [from totcrt(1:nbctime) values]
c[300] and currxjtot is the total current from solution of FPE.
c[300] The new value of zeff is
c[300]   zeff(ll)=max(1.d0,zeff(ll)/renorm) ! Note that Ip ~ Te^1.5/Zeff
c[300] The tests show that the current from FPE approaches the target value
c[300] in several time steps, typically 20 steps.
c[300]            ! The question now is - How to deal with densities?
c[300]            ! Presently, cql3d is setup to keep n_e unchanged,
c[300]            ! and adjust the ion densities (example: reduce D+ density,
c[300]            ! but increase density of impurity, to keep n_e unchanged).
c[300]            ! Another approach would be to increase density of impurity,
c[300]            ! and to increase density of electrons accordingly,
c[300]            ! while the density of D+ remains unchanged.
c[300] Also note that the profile of zeff(ll) is changed by same factor
c[300] at all rho, so that the shape zeff(rho) is not changed;
c[300] Not very physical.           
c[300] The changes are in profiles.f, ainsetva.f, tdxinitl.f.
c[300] Tested for ampfmod='enabled', but could probably work with other options,
c[300] like efswtch="method1".
c[300] !YuP[2019-10-29]-YuP[2019-10-31]


c[299] Adjusted ainsetva for efswtch="method2","method3" or "method4" 
c[299] cases, particularly for iprocur.eq."prbola-t" and iprocur.eq."spline-t".
c[299] Also, for efswtch.eq."method4", check and force efiter="disabled".
c[299] Also, for these cases [iprocur.eq."prbola-t" and iprocur.eq."spline-t"] 
c[299] corrected the issue with currxj(ll) getting INF values at n=0.
c[299] See this note in profiles.f:
c[299]             !YuP[2019-10-29] The problem here is that at n=0
c[299]             !darea is not defined yet [it is set in tdrmshst].
c[299]             !At n=0 sub.profiles is called from sub.tdinitl
c[299]             !BEFORE tdrmshst. 
c[299]             !But then sub.profiles is called again at n=0, 
c[299]             ! from tdchief[Line~575], just before call_achiefn(0).
c[299]             !At the latter call, darea is defined.
c[299] To avoid this issue, added if(currxjtot.ne.0.d0) condition
c[299] At n=0, at first call of profiles, when darea=0 and so currxjtot=0.d0, 
c[299] do this renormalization in tdrmshst.f(239)        
c[299] !YuP[2019-10-29]


c[298] Made adjustment in sub.achiefn, 
c[298] for (eseswtch.ne."enabled") section: No need to call sub.efield
c[298] when ampfmod='enabled', so added  if(ampfmod.ne."enabled") 
c[298] in front of 'call efield'.
c[298] Before this adjustment, sub.efield was called 
c[298] in case of ampfmod='enabled', but (luckily) 
c[298] this had no effect on value of elecfld()
c[298] because of eoved=0.0 setting (see efield.f, Line~22 and 180).
c[298] See "YuP[2019-10-29]"

c[297] Made adjustments/comments in sub.efield related to efswtch.eq."method4"
c[297] [basically, we need to use psifct1=1.; needs more checking 
c[297] for method5]. See 'YuP[2019-10-29]. 

c[296] Corrected pgplot-related subroutines. Some of them were called 
c[296] with explicit values, like call GSVP2D(.2,.8,.6,.9).
c[296] Need to use declared variables, like PGSVP(R4P2,R4P8,R4P6,R4P9).
c[296] Most of changes are in subs. pltends, pltfluxs, pltfofvv, 
c[296] pltprppr, pltstrml, tdpltjop. Cleaned up other plt* subroutines
c[296] for unused calls, e.g., in pltmain.  YuP[2019-10-28]


c[295] version="git_cql3d_190923.1"
c[295] Much added related to pellet ablation model, In progress,
c[295] and TBAdded YuP191001]

c[294] For ampfmod=enabled, fixed bug which prevented update of edge
c[294] through elecb [BH191012].

c[293] CQL3D begins each simulation at time t=0.  For restart, it would
c[293] it would be preferable for most namelist purposes and plotting
c[293] that time would restart at the restart time.  But this would
c[293] require substantial reworking on the code logic.  As a temporary
c[293] approach to adjusting time dependent namelist data, t_shift
c[293] namelist variable is added, which shifts bctime backwards 
c[293] (subtracts t_shift) so that it the restart runs begins at 
c[293] code time t=0.  Future work will adjust code time so that it
c[293] continues continuously for restart cases.  [BH190920].

c[292] version="cql3d_git_190309.3" (Maybe last pre-f90 version)
c[292] Changes to include enhanced collisions due to particially
c[292] ionized impurity ions per Hesslow. To be further documented
c[292] by YuP   [YuP, Pigarov, BH Sept 2019].

c[292] version="cql3d_git_190309.2" (Last pre-f90 version)
c[292] Explicitly typed REAL*4 variables which are in the arguments
c[292] of the PGPLOT plotting subroutines.  This enables compiling with
c[292] gfortran with compiler setting "-fdefault-real-8", which it is
c[292] henceforth recommended to be used. Without explicitly typing
c[292] REAL*4 literal constants used by PGPLOT, default-real-8 would
c[292] make them REAL*8 constants, and thus not work with PGPLOT subs.
c[292] fdefault-real-8 does not promote variables with explicit kind 
c[292] declarations.
c[292] There is a similar setting for the Intel compiler.
c[292] With this setting, the compiler:
c[292] Sets the default REAL type to an 8 byte wide type. Does nothing 
c[292] if this is already the default. This option also affects the 
c[292] kind of non-double real constants like 1.0, and also does promote 
c[292] the default width of "DOUBLE PRECISION" to 16 bytes if possible,
c[292] unless "-fdefault-double-8" is given, too.
c[292] In any case, for cql3d, all DOUBLE PRECISION type settings 
c[292] were changed to REAL*8.
c[292] [BH, 190729].
c[292] 

c[291] version="cql3d_git_190309.1" (Last pre-f90 version)
c[291] Corrected some bugs found during work on f90 version of the code.
c[291] The changes are mostly in impavnc0.f, tdchiefn.f, netcdfrw2.f, zfreya.f.
c[291] For details, search "YuP[2019-04", "YuP[2019-05", "YuP[2019-06".
c[291] The bugs are not critical - test runs give same results.
c[291] Also adjusted settings related to restart runs - 
c[291] they cannot be accurately performed 
c[291] (meaning a bit-to-bit match with no-restart runs)
c[291] for chang='noneg' and urfdmp='secondd' settings. 
c[291] It is recommended to use chang='enabled' and urfdmp='firstd'
c[291] for restart runs. 
c[291] The sub.ainsetva will check those 
c[291] variables and issue a warning. 
c[291] In general, urfdmp='secondd' should be avoided - 
c[291] It uses sub.urfavg, which has a questionable averaging method
c[291] that depends on nstop variable. See notes after "YuP[2019-07-15]"
c[291] in that subroutine.
c[291] Also, in sub.coefmidt and coefmidv - removed imposing a lower limit
c[291] for |df(i,j)| and |db(i,j)| coefficients, 
c[291] and adjusted subs.coefwti and coefwtj
c[291] to avoid division by 0, related to those lower limits.
c[291] With this change, the no-RF runs now give exact zero RF power;
c[291] Before this change such runs were giving a small "ghost" RF power.
c[291] Also, corrected a bug in losstor (in torloss(k) .eq. "velocity" branch).
c[291] For details, see "YuP[2019-07"


c[290] Added namelist variable bctimescal, which scales the bctime(),
c[290] for background plasma time-variation.  [BH190102, YuP190309].

c[289] version="cql3d_cswim_180101.3"
c[289] For nlrestrt="ncdfdist" or "ncregrid", added resetting
c[289] of elecfld(0:lrz) and elecfldb from distrfunc.nc.
c[289] Also, disallowed reading of distrfunc.nc with multiple f(,,,)
c[289] since there are improper netcdfshort settings. [BH181025].

c[288] Added new option for saving distr. function 
c[288] into mnemonic.nc file, netcdfshort="lngshrtf",
c[288] which saves f at selected time steps only;
c[288] The steps are set by specifying nplot()
c[288] values in cqlinput.
c[288] The value of time is recorded into 
c[288] 'time_select' variable in the nc file.
c[288] YuP[2018-09-28], BH.

c[287] Corrected netcdfrw2.f for storage of arrays
c[287] denra, ucrit, eoe0.  YuP[2018-09-24]

c[286] Noticed that sometimes Te is getting to 
c[286] negative values at plasma edge.
c[286] It is related to interpolation by subroutine tdinterp().
c[286] When tdinterp() is called by profiles.f, 
c[286] at the last (upper) point in ryain-coordinate, 
c[286] it uses iup="free" boundary condition, which means 
c[286] the first derivative is calculated 
c[286] by fitting a cubic to 4 last points
c[286] in original arrays. 
c[286] This procedure occasionally gives an interpolated value
c[286] of Te() at the last point in rya() which is zero or negative value.
c[286] To avoid such condition, an additional option is added 
c[286] for calling tdinterp() with iup="linear", 
c[286] which means that the derivative 
c[286] is set from the last two (outer radius) points in original array,
c[286]   cd2(nold)= (y(nold)-y(nold-1))/(x(nold)-x(nold-1)), 
c[286] where x() corresponds to ryain() array, nold==njene.
c[286] Now this subr. is called with tdinterp("zero","linear",...)
c[286] for all the spline plasma profiles (in subr. profiles and tdinitl).
c[286] YuP[2018-09-19]


c[285] Improved the definition of reden() through zeff and 
c[285] density of other species. In original definition, 
c[285] reden could get a small negative value,
c[285] because of a rounding error. Added lower limit =0.d0,
c[285] reden(k,ll)=max(reden(k,ll),zero) .
c[285] YuP[2018-09-18]

c[284] Fixed bug in diagscal pertaining to scaling density of species k
c[284] to specified time-dependent values, for spline-t and 
c[284] lbdry(k)=scale or consscal. [YuP,BH180918]

c[283] Added enerkev and uoc, corresponding to x(:) array to
c[283] .nc output file (for convenience), in runaway plots.
c[283] Bug fix: jfl reset to jx for knockon="enabled" affecting fle,
c[183] but with no noticed effect except proper termination of run.
c[183] [BH180706]

c[282] version=cql3d_cswim_180101.2
c[282] For rdcmod="format1"/"aorsa", increased the number of input
c[282] RF diffusion coefficient files that can be read, 
c[282] and applied to the same or separate general species.  This
c[282] modification also enables rdcmod with multiple general species.
c[282] Also adjusted code for multiple general species restart, 
c[282] using nlrestrt='ncdfdist',nlwritf='ncdfdist'.
c[282] The rdcmod is not (yet) introduced for rdcmod="format2", but
c[282] is readily extended to this case as need arises.
c[282] Namelist nrdc.ge.1 is the number of rdcmod files.  See
c[282] cqlinput_help for additional new namelist variables,
c[282] rdcfile(1:nrdc),nrdcspecies(1:nrdc),rdcscale(1:nrdc),
c[282] rdc_plot,rdc_netcdf.   [BH180608].

c[281] version=cql3d_cswim_180101.1
c[281] Added capability to make plots in color, particularly 
c[281] contour plots for distr.function, change of distr.func.
c[281] over time step, urfb coeffs, and source.
c[281] Main changes are in pltcont.f.
c[281] The color can be added by setting
c[281] pltd='color', or pltd='df_color', or pltso='color',
c[281] pltso='first_cl', or plturfb='color'. 
c[281] BH,YuP[2018-02-08].

c[280] YuP[2018-01-08] Revised some of plt*.f files, also souplt.f,
c[280] to correct the functionality of pltlim plotting options.

c[279] version=cql3d_cswim_180101.0
c[279] YuP[2018-01-05] Added resetting of vth() array in profiles.f.
c[279] vth is the thermal velocity =sqrt(T/m) (at t=0 defined in ainpla).
c[279] But, T==temp(k,lr) was changed above, 
c[279] in case of iprote (or iproti) equal to "prbola-t" or "spline-t".
c[279] Then, reset the values of vth() [used for plots, and also 
c[279] in lossegy, efield, restvty, sourceko, tdnpa, tdtrdfus,
c[279] tdoutput, and vlf*,vlh* subroutines].
c[279] Similarly - for the energy() array.
c[279] The energy() array, initially set as 1.5*temp() [see ainpla.f]
c[279] is re-defined at each time step in subr. diaggnde
c[279] as the m*v^2/2 average over distr.function in vel. space.
c[279] BUT, it is done for k=1:ngen species only!
c[279] So, in profiles.f, we re-define the energy() array  
c[279] for the Maxwellian species only.

c[278] YuP[2018-01-04] Adjusted the printout of variables 
c[278] that are specified in cqlinput into *.ps file.
c[278] Now the long text lines are wrapped around 
c[278] to additional lines, instead of just cutting them.
c[278] See ainplt.f, lines ~~180-240.

c[278] YuP[2018-01-04] Adjusted subr.pltstrml, 
c[278] to make plots of streamlines of phase flow.
c[278] Also - small corrections in pltcont.f, for the plots 
c[278] of v=vnorm and v=vth lines in different units (v/vnorm or v/c).

c[278] YuP[2018-01-02] Corrected plots of pitch angle avg source vs u.
c[278] In subr.pltsofvv (file souplt.f), added (similar to pltcont):
c[278] Setup different types of horizontal axis, depending on 
c[278] namelist settings of pltlim = 'u/c' or 'energy' or 'x'.

c[277] BH,YuP[2018-01-02] Added resetting of ipro*** values
c[277] to "prbola-t", in case when the run
c[277] is done with nbctime>0 (time-dependent profiles),
c[277] but ipro*** values are set to "parabola".
c[277] See ainsetva.f.
  
c[276] version=171124.1
c[276] Adding time-dependent scale factor, difus_io_drrscale(,k), 
c[276] difus_io_drscale(,k) at times  difus_io_t(1:ndifus_io_t). 
c[276] This is applied only in difus_io(k)="drrin" and "drrdrin" cases.
c[276] Fixed several out-of-bounds bugs affecting the lrz=1 cases.
c[276] [BH171231].

c[275] version=171124.0
c[275] Added capablility to output and input the radial diffusion 
c[275] coeffs d_rr and pinch arrays as function of theta,u,species,
c[275] rho to/from a netCDF file.  This enables outputing a
c[275] template file, and inputing numerical d_rr and d_r.
c[275] Additional namelists in the trsetup namelist are difus_io(1:ngen) 
c[275] and difus_io_file. Added for MST simulations. [BH171122].
c[275] Installed neutral t-dependent splines (ennin_t) from FOW cql3d.
c[275] [YuP171127].
c[275] Also, added some modifications added to facilitate TRANSP
c[275] compile system, labelled CMG (Marina Gorelenkova), from YuP work.
c[275] Added improvement to eqfndpsi.f according to cql3d-fow, which
c[275] fixed a problem in setting up a psi(rho) grid for some eqdsk
c[275] cases with small psi gradient near the magnetic axis.[BH171211]
c[275] Changed the default of nconteq="psigrid" to nconteq="disabled",
c[275] and set nconteqn=50.  See cqlinput_help.   [BH171211]

c[274] YuP[2017-11-17] Migrated eqrhopsi from the CQL3D-FOW version. 
c[274] It has corrections for eqpsi(1:nconteqn) definition, see do_11 loop.
c[274] In cql3d-fow (or in "mirror" version this adjustment 
c[274] is done on [03-2016; with correction on 2017-12-05]. 
c[274] This correction gives an effect on 
c[274] NBI deposition points (very little, but visible - 
c[274] in coords of points in R,Z plane).

c[274] YuP[2017-11-21]: In definition of sorpw_nbi() [Neutral Beam power]
c[274] added if(gone()) check: ONLY COUNT not-lost particles.
c[274] It is done similar to the CQL3D-FOW version (added on 03/23/2015).
c[274] The array sorpw_nbi is only used 
c[274] for plots and NetCDF file. 
c[274] It has no effect on the source operator or solution.
c[274] The consequence of adding if(gone..) condition is
c[274] the drop of NBI power profile at plasma edge 
c[274] (if lossmode is not disabled) because of banana+Larm.rad losses;
c[274] see tdpltjop, plot 'FSA SOURCE POWER DEN: ...';
c[274] Also, 'Power from NBI', and 'Power integrated over radius'
c[274] become smaller (values are printed in *.ps file). 

c[273] version="cql3d_cswim_170101.6"
c[273] Adjusted cql3d to accomodate inclusion in TRANSP.
c[273] Routine to detect if &fsetup namelist being used instead of
c[273] the standard %setup0, and provide for reading it.
c[273] (&fsetup was an accomodation required by the pathscale
c[273] compiler, since only was sensitive to 5 characters in a
c[273] namelist section name.)
c[273] Explicit deallocation of arrays before end of cql3d,
c[273] rather than just depending on the operating system to deallocate
c[273] at the end of a run, adding subroutine de_alloc in it3daloc.f.
c[273] Additional zero initializations.   [Petrov, Nov/2017].

c[272] Migrated some of recent modifications from FOW version:
c[272] 1. Updated freya subroutines in cql3d to calculate NBI
c[272] stopping according to ADAS data in GA freya code[Kinsey, 130821].
c[272] 2. Capability to read a list of neutral beam injected
c[272] ion starting points from NUBEAM.  List is generated by Deyong
c[272] from an NSTX TRANSP run for a modulated beam case.  
c[272] New frsetup namelist: read_birth_pts, nbirth_pts_files,
c[272] birth_pts_files, nbirth_pts [BH130304].
c[272] 3. NBI pulse square-wave modulation.  New namelist variables:
c[272] beamplse,beampon,beampoff.  Output of time-averaged 4D (f4d)
c[272] distribution function.  New namelists: tavg, tavg1, and setting
c[272] f4d_out=tavg.  First use is NSTX NBI+HHFW [BH120608].  
c[272] 4. Correction for B-field signs by introducing cursign and bsign
c[272] factors  in freyasou.f, tdnpa0.f, and tdsxr0.f.  It may not
c[272] have a large effect for cases where the toroidal field is dominant,
c[272] and not effect if cursign/bsign are positive. [YuP and BH, 110815].
c[272] 5. Ampere-Faraday solution option (see 'ampfmod').


c[271] version="cql3d_cswim_170101.3.1". Young-soon Bae, KSTAR, found
c[271] problem with updated ray files generated by cql3d, when using
c[271] multiple ray input files.  Second file overwritten with first 
c[271] file data, for a urfmod, netcdfshort='longer_f' case.  
c[271] Fixed by including a krf dependency in output of the ray data 
c[271] in netcdfr3d.f.  [BH, 170920].

c[270.5] Found that the unphysical looking ledge in the distribution
c[270.5] in wh80.ps knockon/runaway electron case was due to a bug fix in 
c[270.5] sourceko.f on 1/23/2000. (Shown by reverting the fix.) Need to check
c[270.5] out further the reason for the "ledge".  [BH170927].

c[270] version="cql3d_cswim_170101.3".  Restoring original 
c[270] lossmode(k)="simplban" functionality.  lossmode(1)= 'simplbn1'
c[270] needs work.  lossmode(k)='simplbn2' gives same results as 'simplban',
c[270] but plan to update it to use subroutine deltar first order calc
c[270] of radial orbit shifts.  deltar only works with ngen=1 at this time.
c[270] Fixed bug in tdrmshst.f which was giving incorrect area/vol
c[270] mesh increments, resulting in incorrect total currents and powers.
c[270] Also, removed fow_orbits.f from ZOW code.  [BH and YuP, 170717].

c[269] version="cql3d_cswim_170101.3"
c[269] Re-setting namelist lbdry(k)="scale" to lbdry(k)="consscal" in
c[269] in ainsetva.f. The "consscal" model for density rescaling
c[269] is more straight-forward to interpret.
c[269] This adjustment fixes a problem in lbdry(k)="scale" which arose
c[269] due a fix of a "nipple" forming on the distn at v=0.  The "nipple"
c[269] let to constant density rise in a lbdry(k)="scale" modeling
c[269] of LHCD in C-Mod.  Problem identified by Syun\'ichi Shiraiwa.
c[269] [BH and YuP, 170703].

c[268] version="cql3d_cswim_170101.2"
c[268] YuP[04-2017] Adjusted files in /mpi/ subfolder
c[268] to match the recent changes in urf* files.

c[267] Following the message from John Wright[04-2017, 04/03/2017]:
c[267] Changed index 'n' in statement functions in subr.cfpcoefr
c[267] to 'ng'.  Index 'n' is also in comm.h, which causes 
c[267] a compilation error (bug) in new Intel compiler (2017).

c[266] version="cql3d_cswim_170101.1"
c[266] Regression tests of a ngen=2 (2 ion general species) test with
c[266] and NBI heating and multiharmonic FW heating of both general
c[266] species, showed that results had significantly deviated in an
c[266] unphysical way from prior 130124 results.  The changes were tracked
c[266] down to problematic changes in losscone.f and a change in the
c[266] treatment of a resonance contribution (rrr= in the code) in the
c[266] QL diffusion (urfb0.f) and damping (urfdamp1.f).  Coding was
c[266] reverted to the long-standing approaches, pending further 
c[266] studies of the changes.  [YuP and BH, 170105].

c[265] version="cql3d_cswim_161220.1"
c[265] YuP provides cql3d_cswim_160605_update1.zip, updating
c[265] cql3d_cswim_160602 version  from cql3d_160720_mirror 
c[265] which incorporates changes for the cql3d-FOW version in
c[265] urfb0.f etc.  This is a fix for a problem found by Young-Soon Bae
c[265] with damping on 3rd harmonic EC, for top launch of mainly
c[265] 2nd harmonic EC in Kstar.  Updated sigsetup.f according to 
c[265] YuP changes in Aug/2016 [YuP, with BH, 161220]
c[265] More details on the updates:
c[265] -- Removed storage arrays for <E> and <F> QL coeffs, 
c[265] as they are expressed through <B> and <C> coeffs.
c[265] (For urf modules only).
c[265] -- Upgraded vlf and vlh modules: now they are valid for multi-surface
c[265] runs (arrays cqlb and other cql* now include additional radial index).
c[265] -- Value of symm is defined in aingeom.f now, and then used in other
c[265]  files (originally - defined in many places/files). 
c[265] -- Value of truncd is defined in micxinit.f now, and then used in 
c[265] other files (originally - defined in many places/files). 
c[265] -- Plots of QL coeffs in urfbplt/pltcont: the all-modes plot 
c[265] is modified to sum-up all modes and then plot contours of the 
c[265] total array (originally - overlapping contours of separate modes in one plot).
c[265] -- In mixcinit: Check that there are sufficient number of v-grid points  
c[265] over thermal part of the distribution (at least 3 points).
c[265] If not, print warning [see if(j_thermal.lt.3) ...].



c[263] version="cql3d_cswim_160602"
c[263] Upgraded lossmode(1)='simplban' so banana (+gyro) losses are based
c[263] on the first order deltarho orbit shift.  Kept the old pre-160403
c[263] model (='simplbn2'), and an intermediate model (='simplbn1') for
c[263] backward compatibility and cross-checking.  Main changes are in 
c[263] losscome.f.  [See file log.1_simplban_extended.gz for comparisons.]
c[263] Should use lossmode(1)='simplban' in combination with ndeltarho=
c[263] ='freya', to prevent unphysical scrapeoff of beams. [BH, 160602].

c[262] Fixed inadaquecy in tdnpa0.f which was causing NaN for
c[262] some cases for efluxt.  Similar fix for tdsxr0.f [BH160508].

c[212] The total integrated powers - same as with original version.
c[212] The older expression for pwrrf is still present, commented out.


c[211] version="cql3d_cswim_160312.0"
c[211] Bug fixed in tdrmsst.f which was causing EC absorption to
c[211] to be (impossibly) radially inboard of the closest approach
c[211] of the input EC rays to the magnetic axis, for a eqsym='none'
c[211] case [YuP and BH 160312].

c[210] Attemping to get cqlinput_mirror_ions_IC.10 case going:
c[210] Uses Gary Smith b-field model, single flux surface, added couple 
c[210] of logic lines (See cBH151202) but then realized more variables
c[210] need setting for ICRF vlfmod to work in this environment.
c[210] Eventually, single flux surface, particle source + RF QL 
c[210] worked cqlinput_mirror_ions_IC.4.8.4) (diffusion(BH151202).

c[209] Enabled various capitalizations of species namep/namei, for 
c[209] operation with Plasma State software [YuP, 150620].

c[208] YuP[2015/05/03] version=cql3d_fullFOW_140425.1
c[208] Several adjustments in subr.eqorbit().
c[208] 1. Set a limitation/resetting for the value of epsicon_
c[208]   which is the input argument for eqorbit().
c[208]   If it is too small (outside of LCFS), reset with warning:
c[208]     epsicon_= psilim + 0.001*(psimag-psilim)
c[208]   The shifting fraction, 0.001, could be changed.
c[208]   The starting point for tracing the surface
c[208]   should not be too close to the LCFS (psi=psilim),
c[208]   otherwise the "orbit" of the surface may diverge 
c[208]   outside of the LCFS.
c[208] 2. Modified the adjustment of 1st/last point:
c[208]   At small rho, the "orbit" usually "drifts" towards magn.axis,
c[208]   so that solr_(nn)<solr_(1).
c[208]   But at outer surfaces, the "orbit" diverges to the outside,
c[208]   so that solr_(nn)>solr_(1).
c[208]   In such a case, the orbit can get outside of LCFS,
c[208]   if the starting point solr_(1) is close to the LCFS.
c[208]   So, setting solr_(1)=max(solr_(1),solr_(nn)) would have
c[208]   a bad result. Simply move the last point to the starting point:
c[208]   solr_(nn)=solr_(1)
c[208] 3. Sometimes, at ending point, in case of eqsym='none',
c[208]   an orbit/surface cannot get to the starting point
c[208]   because of accuracy of the orbit integrator.
c[208]   Orbit can get to a smaller R, where the value of bval/bmidplne_
c[208]   is a bit smaller than at the starting point, say 0.9999
c[208]   instead of 1.d0. Although the ending point is forced 
c[208]   to be the same as the starting point (later in the subr),
c[208]   the previous point (lorbit_-1) may still have the value of
c[208]   bpsi_(lorbit_-1) = 0.9999 or so.
c[208]   In any case, bpsi_(l) cannot be less than 1.d0 - 
c[208]   the code cannot handle such equilibria.
c[208]   The adjustment that is made only works for eqsym=none
c[208]   (during 2nd half of surface):
c[208]   Adjust (R,Z) points by setting them in a straight line 
c[208]   connecting the first point where bpsi_<1 is detected 
c[208]   (designated l=lbad) to the end/starting point [to be exact, 
c[208]   connecting the point(lbad-1) to the point(lorbit_)]
c[208] There are few more adjustments for the last two points 
c[208] in surface tracing (mostly for the case of eqsym='none').
c[208] Search for 2015/05/03 in eqorbit.f.

c[207] YuP[2015/05/03] Added plotting of the LCFS in plots
c[207] with flux surfaces, in case of eqsym='none'.

c[206] YuP[2015/05/03] Bug fix in eqrhopsi: 
c[206]   ez2=ez(jmag+1) ! was (imag+1)
c[206] It was affecting the tracing of small flux surfaces
c[206] near magn.axis in some cases of nonsymmetrical 
c[206] up-dn equilibria (eqsym='none'), because of error in 
c[206] determination of (supposedly more accurate) value of zmag.

c[205] YuP[2015/05/02] Added a more accurate def. of volume and area
c[205] near magnetic axis.  This definition 
c[205] is not dependent/sensitive to tracing the surface.
c[205] Gives better convergence in eqfndpsi in cases for 
c[205] radcoord.eq."sqtorflx", when 'rhonew' is proportional to 
c[205] sqrt of 'volum' (near magn.axis).

c[204] YuP got rid of the nipple at region of f(v=0) which was
c[204] appearing in restart runs due to that [YuP believes] f(j=1) is 
c[204] not found as the solution, but taken from the previous time step.
c[204] YuP[04/10/2015].

c[203] Slight correction of simplban loss model: use local bmod0
c[203] rather than edge bmod0, for gyro-losses.
c[203] Enabled use of density rescaling along with lbdry(k)="conserv",
c[203] using new option lbdry(k)="consscal". Previously, density
c[203] rescaling only occured with lbdry(k)="fixed". [YuP+BH 140806].

c[202] version="cql3d_cswim_140804"
c[202] Fixed bug in calc of cfpgamma (Coul log) for diagnostic
c[202] purposes in tdoutput.f (tdnflxs not called in l=1,lrz loop).
c[202] (Also, lr_ improperly used in a do loop variable.) BH140804.

c[201] version="cql3d_cswim_131030"
c[201] Reduced size of  lossfile, netcdfnm, eqdsk, rffile() from 
c[201] charcter*512 to character*256, to avoid possible limitation
c[201] with a character read of variable greater than ~300 with intel
c[201] compiler.
c[201] Fixed bug in input time-dependent parabolic profiles (tdxin13d.f).
c[201] Fixed bug in netcdf file writes for wpar,wper, and longer_f
c[201] save of distributions (netcdfwr2.f)  [BH131026-30].

c[200] version="cql3d_cswim_130928_mpi"
c[200] Added delim='apostrophe' to open (iunwrif,... in tdwritef.f.
c[200] This is to address a concern/problem about long character variable
c[200] when writing/reading them as namelist elements with various
c[200] compilers.  [gfortran has this delim as default, but this 
c[200] convention is not universal].  Checked namelist variables
c[200] lossfile, netcdfnm, eqdsk, rffile(), and all are set to 
c[200] character*512  (enabling long path specs). Also, checked that
c[200]  all open() for writing namelist contain the delim statement. 
c[200] Could reduce variable length per Bonoli, but trying this first.
c[200] [BH130928].

c[199] Restored printout of rfpwr for rdcmod.ne.disabled cases
c[199] (fixing oversight in ~2011 when added printout of more than
c[199] 3 harmonics of rfpwr for urfmod.eq.'enabled') [BH130611].

c[198] version="cql3d_cswim_121120_mpi"
c[198] Fix error exit hang for MPI, discovered by John Wright.
c[198] [YuP, 121117].

c[197] Modified tdsxr0.f and tdnpa0.f to handle a sxr/npa sightline
c[197] which occurs very near to the magnetic axis, when the
c[197] magnetic axis psi value is slightly underestimated [BH121115].

c[196] version="cql3d_cswim_120816_mpi"
c[196] Added powrft (rf power summed over modes) to use by SWIM IPS.
c[196] Some cleanup of netcdfrw2.f. Additional storage in tdsxr0.f.
c[196] Reverted time step counter update to previous use [YuP+BH, 
c[196] an MPI related mod, now not necessary]  [BH120816].

c[195] Added option to specify radial diffusion profile as tein, nein, 
c[195] etc, using a new namelist variable: difin
c[195] If sum(abs(difin))>0, then difin(ryain) is used to define rshape 
c[195] in tdtrdfus.f instead of difus_rshape(1-7)
c[195] This is useful since it has been found that a practical profile
c[195] is D0 * chi_e(r). So the chi_e profile from power balance
c[195] can be given in difin and D0 in difusr. TCV experiments showed 
c[195] that D0~0.2 works for many cases [OS 20120809]

c[194] version="cql3d_cswim_120807_mpi"
c[194] Fixed bug preventing power from more than two neutral beams to be
c[194] to be injected [BH120807].

c[193] version="cql3d_cswim_120315_mpi"
c[193] Reworked and debugged nuclear fusion rates.  The DD-rates remain
c[193] unchanged, but D-T, D-He3 were not working.  <sigma-v> results
c[193] now agree within a few percent with standard Bosch-Hale results,
c[193] for the four nuclear fusion rates in the code.  Logic was adjusted
c[193] in subroutines sigsetup, sigmax, sigv5d.  Namelist variable 
c[193] isigsgv2 was made inoperative, as has no apparent physics use
c[193] [BH120315].

c[192] Add netcdf ouput of FSA current parallel curr.  Fixed minor bug 
c[192] re printout of Fisch normalized RFCD [changing value].  
c[192] Fixed bug in output of rfpwr to netcdf file. Power density
c[192] in individual modes, at total (not sorpw/sorpwi) were
c[192] erroneously divided by dvol() [BH120221-3].

c[191] version="cql3d_cswim_120202_mpi"
c[191] Added required blas and lapack routines to r8subs.f, so
c[191] no further need for loading blas and lapack libraries.
c[191] Updated recent makefiles.  Added some makefiles for PPPL.
c[191] Add powurf and powurfc to netcdf .nc file for all mrfn urf modes.
c[191] Corrected defn of rfpwr in .nc file, and added data to all "modes"
c[191] [BH120202].

c[190] version="cql3d_cswim_120124_mpi"
c[190] For multiple general species, added capability to damp same urf
c[190] wave data sets on multiple species.  Previously, only could do
c[190] this for one general species (ngen=1)  [BH120125].

c[189] version="cql3d_cswim_120122_mpi"
c[192] noplots="enabled1" inhibits all plots.  Previously, 
c[192] noplots="enabled" inhibited all GRAFLIB plots, but over a
c[192] period of ~ 10 years, all such routines had been converted to
c[192] PGPLOT library routines, so it has no effect.  The purpose of
c[192] noplots="enabled1" is to enable compilation without the pgplot 
c[192] library [along with using, e.g.,  the -Wl,-noinhibit-exec gnu 
c[192] loader (ld) option in gfortran which enables loading of the code 
c[192] in the presence of unsatisfied externals] [BH121122].
c[192] Also increased max number of NBs, kb=3 ==> 8 (for D3D+).

c[191] Increased iotalim from 750 to 6400, for use with larger
c[191] values of jx [BH120104].

c[190] Fixed an apparent bug in symmetrization in the calculation of
c[190] collisional contribution to C,D, and F in the trapped region. 
c[190] [See cfpcoefn.] The effect on distributions for a test case
c[190] seemed small,  but more checking is warranted [YuP111202].
c[190] Had no effect in an aorsa-dc-cql3d simulation [BH].

c[189] Fixed write of two distributions to the .nc file, for
c[189] ngen=2 cases [BH111103].

c[189] version="cql3d_cswim_110804_mpi"
c[189] Small mods, bug fixes, improvements:
c[189] -option added for output of freya NBI birth points to ascii file
c[189] -clipping option to smooth external diffusion coeffs from DC
c[189]  or AORSA added in rdc_multi.
c[189] -removed some parameter-setting restrictions in netcdfrw2.f,
c[189]  persuant to dynamic dimensioning modification of cql3d.
c[189] -Removed some redundant "save" statements in several subroutines
c[189]  and added a character*8 ntitle,dat in equilib.f, following
c[189]  Jacob Urban compilation with g95.
c[189] -Added deltarho, first order orbit shift to the netcdf output.
c[189] -diagimpd.f changes to trunc-d option to improved treatment of
c[189]  NBI source into prompt loss region.
c[189] -maxp maximum number of freya NBI source particles increased
c[189]  from 150000 to 1500000   [BH and YuP, 110804].

c[188] Updated NPA recombination rate in tdnpa.f from Path and Wolf
c[188] formula to Hutchinson, Eq. 6.3.5, per Aaron Bader. Not much
c[188] change in C-Mod NPA spectra, since recomb. is small [BH110804].

c[187] Inconsistency in distribution function extrapolation when
c[187] increasing enorm, was fixed.  Had caused neg density of
c[187] regridded density, and code to halt, is some cases [BH110525].

c[186] rdc_clipping functionality added.  Imported rf diffusion coeffs
c[186] such as from the DC code (via rdcmod=enabled), can have have
c[186] unphysical spikes near the trapped passing boundary.  These can
c[186] be removed using a running median filter. See code for refs
c[186] [BH110430].

c[185] Slight addition of ineg='enabled1' option, removing spikes
c[185] in the distribution occuring when source points are in
c[185] prompt loss regions of velocity space [BH 110408].

c[184] Add printout of freya NBI birth point positions and velocities
c[184] [BH110406].

c[183] Code version='cql3d_cswim_110401_mpi'
c[183] Added mpi capabilities for X-ray, neutron, and npa diagnostics
c[183] [YuP110404].

c[182] Enabled that number of Legendre polynomials in the approximation
c[182] of f and sigma for xray calc, msxr, can be set independently
c[182] of number of Legenedre polynomials in the FP collision
c[182] coefficients [previously msxr.le.mx]. mx=3 is probably OK
c[182] for collsion coeffs, but msxr~8 is required for accurate calc
c[182] of XR spectra due to electron runaway distributions. 
c[182] Similarly, the number of Legendre polynomials used in fusion
c[182] rate calcs, mmsv, in no longer limited to .le.mx.  Changing
c[182] from mmsv=3 to 8, though, made little difference in neutron
c[182] rate, for a NSTX NBI+HHFW case.  [BH110401].

c[182] Enabled MPI version of code to work for transp="enabled" and
c[182] the default soln_method='direct'.  (Additional work is required
c[182] for parallelization of the full, soln_method='it3drv' case
c[182] [YuP and BH 110329].

c[181] Added coding to include multi-species data in the output
c[181] netCDF file, for cql3d runs with two or more general species.
c[181] Output now includes multiple distribution functions, and
c[181] time-dependent wpar,wperp, specific power densities and
c[181] and currents [BH110321].

c[180] Adjusted coding for larger number of sxr/NPA view cord
c[180] plot points.  A problem appeared due to dynamic dimensioning
c[180] of some temporary storage, previous set to a larger
c[180] dimension by fixed parameter values [BH110318].

c[179] Code version="cql3d_cswim_110315_mpi". Fixed two
c[179] bugs in the freya NBI modules: (1) For single source
c[179] NBI cases, using a source axis centered aperature, s-xxxx,
c[179] resulted in undefined aperature widowing (iap not defined
c[179] in zfreya.f/rotate). Modified ashape namelist defn slightly
c[179] so in nsourc=1 case, can enter centered aperature dimension
c[179] with ashape(,)=s-xxxx (source centered) or s-xxxx, equivalently;
c[179] (2) namelist smooth not passed properly from frcomm.h to comm.h.
c[179] Changed name to smooth1 in comm.h and elsewhere.
c[179] Affected smoothing of NB source profile.  [BH and YuP, 110314].

c[178] Code version="cql3d_cswim_110308_mpi". Should have same 
c[178] functionality as serial version cql3d_cswim_110308.
c[178] The line  if (noplots.eq."enabled") return 
c[178] in pltprppr.f is commented out, so that
c[178] the parallel distribution function will be plotted 
c[178] unless pltprpp="disabled" [YuP 03-07-2011].

c[177] Achief1.f (used for a single flux surface calculations)
c[177] is not used anymore. Now the solution is found through 
c[177] the call of achiefn(0) in ll-loop in tdchief, same as for
c[177] multi-flux-surfaces calculations. 
c[177] The probable reason for using achief1 was the problem with 
c[177] tdxinitl, called from tdinitl. Now tdxinitl, which "fills in" 
c[177] input data parabolically and computes the normalized radial mesh,
c[177] is only called for lrzmax>1 [YuP 03-07-2011].

c[176] Modification made after [03-2011] to ineg="enabled" or "trunc_d": 
c[176] (See diagimpd.f)
c[176] If distr.function f(i,j) is negative or zero 
c[176] at some (i,j)-point in vel. space,
c[176] it is set to zero for ALL jj.ge.j 
c[176] (at fixed i-index of pitch angle),
c[176] for energies greater than the maximum in the source term
c[176] (for example from NBI).  For lower energies, neg f is set = 0.
c[176] Before this modification, it was set to zero 
c[176] for only this specific (i,j)-point where f(i,j)<0.
c[176] So, no more "islands" in distribution function 
c[176] remaining beyond such (i,j)-points, except when they are 
c[176] caused by sources like NBI.


c================================================
c[175] MPI-enabled version.   February 2011 [YuP].
c[175] The parallelization capabilities of the code are upgraded. 
c[175] The MPI executable is produced by launching
c[175] make -f makefile_mpi.franklin 
c[175] (similarly for Hopper NERSC machines)
c[175] The makefile will run Python script doparallel.py 
c[175] (in /mpi/ subdirectory) that converts source files 
c[175] by inserting MPI commands from mpins_par.f.
c[175] The location of such commands in original source files  
c[175] can be tracked by searching all phrazes that start with CMPI.
c[175] The executable can be launched as a batch job (example):
c[175] qsub -q regular franklin_batchscript_mpi128
c[175] The file franklin_batchscript_mpi128 should contain
c[175] #PBS -N xcql3d_mpi.franklin
c[175] #PBS -q regular
c[175] #PBS -A m876
c[175] #PBS -m e
c[175] #PBS -j oe
c[175] #PBS -l walltime=00:05:00
c[175] #PBS -l mppwidth=128
c[175] cd $PBS_O_WORKDIR
c[175] aprun -n 128  /???/xcql3d_mpi.franklin
c[175] (Specify your working directory instead of /???/)
c[175] 
c[175] The parallelization is done for:
c[175] 1. Impavnc0 solver, together with collisional coefficients 
c[175] generator; see achiefn.f. Parallel run for each flux surface.
c[175] 2. Energy transfer diagnostics; see diagimpd.f, 
c[175] call diagentr(lefct,k). Parallel run for lefct=-1:3,5:8,11,12.
c[175] 3. Calculation of diff. coefficients for the ray-tracing;
c[175] see urfb0.f. Parallel run for the number (mrfn) 
c[175] of excitation wave modes. 
c[175] 4. Calculation of damping of rays; see urfdamp1.f and urfdamp2.f.
c[175] Parallel run for combined (number of rays)x(number of wave modes)
c[175] which is nrayn*mrfn (these two numbers are determined from 
c[175] reading the rays data file).
c[175] From above, the optimal number of cores (ranks, or mppwidth) is
c[175] the largest of: lrz+1, 11+1, or nrayn*mrfn+1 (if rays are used);
c[175] [+1 because rank=0 does not perform calculations, only collects
c[175] data from other ranks].  More cores can be requested, but
c[175] extra cores will be idling.
c[175]
c[175] The parallelization of the impavnc0 solver is only done 
c[175] for soln_method="direct".  In other cases of soln_method, there 
c[175] will be not much speed-up because the solver will be running 
c[175] on rank=0, for all flux surfaces; some speed-up can still occur
c[175] if many rays are used, due to parallelization of urf-routines.
c[175] For soln_method="direct", the speed-up can be ~10 times.

c[175a] Changes made in course of MPI-upgrade:
c[175a] Call_diagscal is moved out of diagimpd.f. 
c[175a] Updating of dtr, dtreff, dttr, n_(l_) is moved out of achiefn.f
c[175a] to tdchief.f. 
c[175a] Call_profiles is moved out of achiefn.f to tdchiefn.f, 
c[175a] before ll-loop.
c[175a] Definition of gfu,gft and gfi functions is moved out of advnce.h
c[175a] to the end of diagentr.f as function subprograms, 
c[175a] to avoid cvmgt-construct.


c[174] Removed reading of restart namelist data from distrfunc file
c[174] for restarts with nlrestrt='ncdfdist' or 'ncregrid'. BH110201.

c[173] Added restart option nlrestrt='ncregrid' enabling change in
c[173] cqlinput of enorm, jx and xfac from values in the restart
c[173] .nc file.  The restart distribution is extrapolated and
c[173] interpolated onto the code momentum grid. BH110201.

c[172] [January 2011]
c[172] Adjusted plotting limits for flux surfaces in tdplteq.f.
c[172] In some cases of up/dn symmetrical surfaces, when only half of a
c[172] surface is plotted, the surface can be in negative-Z area. YuP

c[171] Dynamically allocated local arrays in tdinlegw.f - 
c[171] allows using large iy now.  YuP

c[170] Changed default value nonboot=1 to nonboot=2  in aindflt.f 
c[170] (turn on computational bootstrap at n=nonboot).
c[170] The radial derivative of distr. function 
c[170] for bootstrap current calculations uses f() at neighbouring
c[170] flux surfaces; in general, it is known from previous time step;
c[170] during n=1 time step the distr.function is still zero on many
c[170] flux surfaces until completion of the time step.
c[170] So it\'s better to start bootcalc at 2nd time step. YuP

c[169] Problems in tests with bootcalc problem: need high resolution
c[169] in theta (pitch-angle). This can only be achieved by setting 
c[169] nii=25 in baviorbt (baviorbt is used when taunew='enabled').
c[169] nii is the number of pitch angle subintervals used in the
c[169] calculation of dtau and tau. 
c[169] Using nii=1 was ok for most tests, except bootstrap calculations.
c[169] Since higher nii does not add much of computational time, 
c[169] nii=25 will be the default value.
c[169] With nii set to 25 in baviorbt (and iy=160 theta grid), 
c[169] the bootstrap current profile is smooth.  
c[169] Also restored factor of 2 for the first node of poloidal-grid:
c[169] if(l.eq.1) dtau(i,l,lr_)=dtau(i,l,lr_)*2.0 
c[169] !-YuP Should be *2 because first node is over half-interval.
c[169] Result: Better shape of tau at pitch-angle theta~pi/2. YuP

c[168] Moved z00 function from advnce.h  back to the end of impavnc0.f.
c[168] Added k-index into z00, because 
c[168] di(i,j,k,l_) and dj(i,j,k,l_) have k-index.
c[168] This modification allows to reduce the usage of cvmgt function -
c[168] better optimization. See note [162] below.  Results are the same.
c[168] YuP

c[167] Fixed a bug in bsl(jj,kk,ll) and bsu(jj,kk,ll) functions:
c[167] x(jj) in bsl and bsu is only defined for jj=1:jx,
c[167] while bsl and bsu are called with jj=0:jx+1.
c[167] Made this fix:
c[167]   jjj=max(jj,1)   ! to limit from below, for x(jjj)
c[167]   jjj=min(jjj,jx) ! to limit from above, for x(jjj)
c[167] Alternatively, could set bsl and bsu to zero for jj=0 and jx+1.
c[167] Almost no effect on results, but prevents out-of-bounds error. YuP

c[166] Fixed a bug in reading of Complex*16 array into a Real*8  
c[166] dummy array in urfread_i.  YuP

c[165] Possible bug in reading ray##/text data file.
c[165] The formats in GENRAY (write3d.f) for saving data:
c[165] 1    format (2i5,1pe16.9)
c[165] 2    format (5(1pe16.9))
c[165] 3    format (9i5)
c[165] 4    format (9(' ',i4))
c[165] But in CQL3D (urfread_.f) for reading data:
c[165] 1    format(2i5,1pe16.9)
c[165] 2    format(5(1pe16.9))
c[165] 3    format(9i6)
c[165] 4    format(9i5)
c[165] Should we make the last two formats in CQL3D as in GENRAY?
c[165] No changes for now, just keep in mind.

c[164] Update 110107:
c[164] Corrected error in advnce.h in bsu(j,k,l), function fpj0.
c[164] Should be bsu(j,k,l_).  YuP, BH

c[163] Fixed a bug with denpar and temppar - they are dimensioned as 
c[163] (ntotala,0:lza+1), but in clear.f they are zeroed over lsa+1.
c[163] These two arrays have dual usage: 
c[163] either have radial coord. dependence, 
c[163] or along-field-line dependence (when cqlpmod="enabled").
c[163] The problem is fixed by forcing  
c[163] parameter(lza=lsa)  in param.h.  YuP.

c[162] Modifications to address the problem of optimization on Franklin
c[162] and gfortran compilers. The compilers could not perform 
c[162] full optimization because of functions cvmgt or similar
c[162] (not an intrinsic, but a declared function in r8subs).
c[162] During invoking of such function, both 1st and 2nd arguments 
c[162] are evaluated, although only one is needed.
c[162] This function is heavily used in definition of
c[162] other functions, which are called in nested i,j,k,l loops.
c[162] Tried to use intrinsic function merge() - no improvement.
c[162] Reduced the usage of cvmg# functions to a minimum
c[162] by using if-else statements instead.
c[162] For some cases the code now runs ~7 times faster on Franklin. YuP.

c[161] Update YuP-101230:
c[161] Dynamic dimensioning of sounor(ngen,nsoa,lz,lrz) 
c[161] to reduce memory usage.
c[161] Changes lza->lz in many dynamically allocated arrays.

c[160] YuP-101228: Corrected error in diagimpd.f (do 410 loop)
c[160] related to ineg='enabled'. Now the negative parts of 
c[160] distr.function are really set to zero. 
c[160] The effect from ineg bug fix is very small  -
c[160] in 3rd-4th digit. 

c[159] Added dyi()=1./dy() and dxi()=1./dx(), 
c[159] made changes in advnce.h and diagentr.f to re-arrange terms, 
c[159] to make the code run faster.

c[158] YuP-101224: Corrected error in netcdfrf.f:
c[158] In pack21(y,1,iy,1,lrors,urftmp,iymax,lrors),
c[158] replaced urftmp by tem1.  In some cases the size of urftmp 
c[158] is smaller than size of y.

c[156] YuP-101221: Eliminated parameter noncha.
c[156] Arrays that depended on noncha are allocated now 
c[156] using nonch which is set to nonch=nstop+1 in ainsetpa.

c[155] YuP-101220: 
c[155] Reduced size of many arrays dependent on ngena; 
c[155] now they depend on ngen.
c[155] (Also, possible bug in tdxinitl: changed tauegyz to tauegy). 
c[155] Similarly - for nmodsa. Large arrays that depended on nmodsa are
c[155] cqlb...cqlf and wcqlb...wcqlf. 
c[155] Now nmodsa is replaced by mrfn in these arrays.
c[155] Moved allocation of cqlb...cqlf to vlh.f and vlf.f
c[155] where mrfn is determined. 
c[155] Allocation of wcqlb...wcqlf is moved to vlf.f

c[154] YuP-2010 December 08-17
c[154] Other parameters eliminated: maxp,jxa,iya,mxa,jfla,
c[154] and related to them.  Many changes through the code -  
c[154] new version cql3d_101217.   
c[154] Code runs ~2 times faster; smaller memory footprint.
c[154] urfdamp2.f is re-organized to make it run faster.

c[153] YuP-101208: parameter nharma is not needed anymore.

c[152] YuP-101207: added in tdchief.f, just before call achiefn(1)  :
c[152]   do k=1,ngen  ! Compute density gains and losses, and powers.
c[152]      call coefstup(k) ! To define da...df coeffs, gon(i,j), etc
c[152]      call coefmidv(da,temp1,1,vptb(1,lr_))
c[152]      call coefmidv(db,temp1,2,vptb(1,lr_))
c[152]      call coefmidv(dc,temp1,3,vptb(1,lr_))
c[152]      call coefmidt(dd,temp1,1)
c[152]      call coefmidt(de,temp1,2)
c[152]      call coefmidt(df,temp1,3)
c[152]      call coefwtj(k)
c[152]      call coefwti(k)
c[152]      call diagimpd(k) ! to calculate sgain(1:8,k)
c[152]   enddo ! k
c[152] Needed to compute sgain(1:8,k)

c[151] YuP-101207:  Modified definition of ipack and ipack16:
c[151] ipack16= jjx/ibytes16*nrayelts*nrayn +1 
c[151] ipack= jjx/ibytes*nrayelts*nrayn +1
c[151] No need to multiply by mrfn, 
c[151] because ifct1,2_(ipack16,mrfn) include mrfn,
c[151] and ilowp(ipack,mrfn), iupp(ipack,mrfn) include mrfn.
c[151] Saves considerable amount of memory!

c[151] Reverted following, in favor of implementation in the full FOW 
c[151] version of cql3d [BH+YuP170506].
c[151] YuP:  Added fow_orbits.f to compilation !!!!!!!!!!!!!!
c[151] It contains subroutines for Finite Orbit Width option.
c[151] THE WORK is in PROGRESS. The routines are NOT used by default.
c[151] Called from tdinitl:
c[151] if(fow .eq. 'enabled') then
c[151]   call fow_alloc ! Allocate arrays for orbit tracing and com_map.
c[151]   call fow_setb ! Setup rectangular grid in (R,Z) for storing the
c[151]              ! values of equilibrium m.field and its derivatives.
c[151]              ! Setup Beq*(ir,iz), psieq(ir,iz), and derivatives
c[151]              ! on the req(ir),zeq(iz) grid.
c[151]              ! Store in common/Beq/ and common/BeqGrad/
c[151]              ! Needed for finite-orbit-width calculations.
c[151]   call com_map ! Setup 3D grid for storing a map 
c[151]             ! (U,mu,Pfi)->Rmidplane 
c[151]             ! In other words, setup a lookup table
c[151]             ! which gives the midplane value(s) of R 
c[151]             ! for FOW orbits ("leg's" major radius at midplane)
c[151]             ! as a function of three indices corresponding to COM
c[151] endif 


c[150] YuP: In coefwti:  Added to prevent jchang=0 :
c[150] if(jchang(k,l_).le.0) jchang(k,l_)=1  


c[149] YuP: In ilut:  Re-defined cutlo=dsqrt(EPSILON(one)) 
c[149] (the smallest number - machine dependent)


c[148] YuP: Allocation of some arrays is adjusted to reduce memory load
c[148] (changed from jxa to jx, iya for iy, ngena to ngen):
c[148] cal, cbl, ccl, cdl, cel, cfl, eal, ebl, ...


c[147] YuP-101122: nraya and nrayelta are not used anymore.
c[147] Instead, nrayn and nrayelts are determined 
c[147] in urfsetup by reading ray data files.
c[147] In urfalloc:
c[147] Added if(istat.eq.0) in front of call bcast() or ibcast()
c[147] If istat=41 (not enough memory), cannot call bcast()
c[147] because array is not allocated => results in Seg.Fault.
c[147] The arrays ifct1_, ifct2_ are quite large 
c[147] and may cause memory allocation problem.


c[146] YuP: Corrections in tdchief. "bug" affected diagnostics output:
c[146] Added
c[146]     call coefstup(k) ! To define da...df coeffs, gon(i,j), etc
c[146]     call coefmidv(da,temp1,1,vptb(1,lr_))
c[146]     call coefmidv(db,temp1,2,vptb(1,lr_))
c[146]     call coefmidv(dc,temp1,3,vptb(1,lr_))
c[146]     call coefmidt(dd,temp1,1)
c[146]     call coefmidt(de,temp1,2)
c[146]     call coefmidt(df,temp1,3)
c[146]     call coefwtj(k)
c[146]     call coefwti(k)
c[146] just before
c[146]     call diagimpd(k) 


c[145] Added option (netcdfshort=long_urf) to output all urf or 
c[145] rdc coefficients at the last time step [BH100917].

c[144] Added option to output specific current currv(u,r) and 
c[144] rf power pwrrf(u,r) at each time step, rather than just
c[144] on the last. Enabled by netcdfshort="long_jp"
c[144] [BH100912].

c[143] Fixed bug in urfalloc.f where insufficient space could
c[143] by allocated for ilowp/iupp and ifct1_/ifct2_ for compressed
c[143] urf ray data storage for 32 bit integer machines.   Evidently, 
c[143] this usually did not cause a problem  [BH100903].

c[142] Added some checking of memory allocation status in ainalloc
c[142] and urfalloc.  However, this didnt properly catch a case of
c[142] too large memory request, which led to a Segmentation fault.
c[142] So, added print out of when allocation routines are entered
c[142] and left (in ainalloc,urfalloc,eqalloc,sigalloc,vlfalloc,wpalloc,
c[142] freyasou,impavnc0,urfbes,tdtraloc, and vlfsetup [BH100330].

c[141] version='cql3d_merge_100829'
c[141] Added multiple SXR and NPA synthetic diagnostic sightlines
c[141] and starting postions to the netcdf file.  Further debugged
c[141] NPA.  [BH100829].

c[140] Fixed up NPA plotting in .ps file, and added NPA data to
c[140] the .nc netCDF file. Added ennscal(1:npaproc) scale factors
c[140] for corresponding density profiles [BH100815].

c[138] Added NPA related namelist variables, npaproc, npa_process(),
c[138] giving access to CX with boron+4 and electron recombination.
c[138] Modified ennl/ennb from scalars to arrays (1:npaproc),
c[138] enabling separate density profiles for related CX species.
c[138] ennin modified from 1D to 2D array, ennin(1:njene,1:npaproc).
c[138] rd_npa,thetd_npa,x_npa,z_npa modified from scalar to
c[138] arrays (1:nv_npa), enabling separate specification of
c[138] detector locations [while maintaining backwards compatibility.
c[138] [BH100720].

c[137] version="cql3d_merge_100608".  This is a major modification.
c[137] It combines several separate branches of cql3d, and includes
c[137] fully-implicit 3d radial transport (soln_method=it3drv), 
c[137] URF routines applied to multiple general species(multiURF),
c[137] full non-updown symmetric equilibria (eqsym=non),
c[137] NPA diagnostics, deltar first-order finite-orbit-width
c[137] shift (and output to .nc file).  
c[137] The 1st order orbit shift is not yet integrated into 
c[137] the calculation of RF diffusion coefficients in urf-routines 
c[137] or into diagnostics). [Yup, BH, 100608].

c[136] Modified method for calculation of NPA, removing
c[136] use of Legendre polynomial expansion of distribution
c[136] function (such as used with SXR), since CX cross-section
c[136] is much simpler (assuming CX ions are much faster than
c[136] the background neutrals.  Taking neutrals to have zero temp,
c[136] then CX of FI to neutral occurs without change in energy of dirn.
c[136] Removed m_npa from NPA related namelist.  [BH, 100521]

c[135] nrstrt namelist variable functionality removed.  It was
c[135] generally not used anymore, and can take on only default
c[135] value, nrstrt=1. Reason for removal:  Was causing difficult
c[135] logic in achiefn.f, related to new soln_method=it3dv and it3drv.
c[135] [YuP and BH, 100517].

c[134] Fixed code bomb for lrz=1,nmlstout='trnscrib', in which
c[134] cqlinput was open in ain_transcribe, when trying to open it.
c[134] Changes to achief1.f, frset.f.  [BH100517].

c[133] Fixed an indexing problem with the equation scaling in tdtranspn
c[133] and impavnc0. This problem arose for the new soln_method='it3drv'
c[133] and 'it3dv' functionality.  For lbdry(k)='scale' or 'fixed', 
c[133] the v=0 boundary point (j=1 index) gets a special treatment - 
c[133] it is not included into the sparse matrix. In tdtranspn, 
c[133] for j=1 & lbdry(k).ne."conserv", set radial elements to zero. 
c[133] In impavnc0, the j=1 point  is not updated from solution 
c[133] matrix (rhs);  update rhs=>f is performed from jstart=2.
c[133] Also, the re-scaling of f 
c[133] (call diagimpd -> calls diagscal -> renorm f if lbdry.eq."scale")
c[133] is moved from impavnc0 to tdchief.
c[133] Tdchief now has two loops in radial index.
c[133] The first loop calls achiefn(0)-->impavnc0, 
c[133] which defines matrix coefficients and finds new f. 
c[133] For soln_method=it3drv, the solution is only found
c[133] when the loop reaches the last radial index, 
c[133] so it is important to postpone with diagnostics or
c[133] re-scaling of f until the end of the first loop in radial index.
c[133] The second loop re-scales f if needed, and computes diagnostics.
c[133] These changes resulted in total current to be different by ~15% 
c[133] in a MST radial transport (soln_method='it3drv')test case. 
c[133] The value of current  obtained with lbdry(k)='scale' is now 
c[133] different by that from  lbdry(k)='conserv' by less than 1%  
c[133] (before corrections: 15%). The shapes of f now are also 
c[133] almost same in runs with lbdry(k)='scale' and 'conserv'. 
c[133] No more problem with v=0 point in f. [YuP and BH, 100517].

c[132] De-equivalenced ca(:,:), cb(:,:), etc., storage from da(:,:),
c[132] db(:,:), etc.  There was an error in re-scaling (call dscal) of
c[132] ca, cb, etc. in cfpcoefn.f/cfpcoefr.f; not all of coefficients
c[132] could have been properly re-scaled because of presence of zero
c[132] index in ca...cf. Now the size of ca...cf is set to (1:iya,1:jxa), 
c[132] no zero index. The difference between the corrected and the old
c[132] 100217 version is ~3.3% in value of total current. This might have
c[132] affected only soln_method='it3dv' and 'it3drv'.  Had no effect
c[132] on a rdcmod='enabled', soln_method='direct' case [YuP, 100513].

c[131] Multi-species QL diffusion (ngen.ge.2) capability was added
c[131] (previously only general species 1 could be QL diffused).
c[131] This was work in Sept'08 and Oct-Nov'09 by BH, under GA contract.
c[131] This work merged into mainline cql3d_cswim_svn cql3d version,
c[131] including debugging [Merge mostly YuP, with BH, Apr-May,2010].

c[130] A fully-implicit 3d (2d-in-vel, 1d-in-radius) solution of
c[130] the FP equation was introduced into CQL3D using conjugate
c[130] gradient sparse-matrix solve techniques, including 
c[130] "drop tolerance".  This uses SPARSEKIT2 (Yousef Saad, U Miss.)
c[130] routines. [BH, see c[100]].
c[130] Merged this code into mainline cql3d_cswim_svn cql3d version,
c[130] including debugging [Merge mostly YuP, with BH, Feb-Mar,2010].


c[129] Modified initial posn of soft xray detector from scalar to
c[129] array dimensioned [1:nv].  This enables multiple detector
c[129] positions, as needed by MST.  Mod by Mike Kaufman, UW,
c[129] added to mainline cql3d, 100510.

c[128] Fixed code bomb for lrz=1,nmlstout='trnscrib', in which
c[128] cqlinput was open in ain_transcribe, when trying to open it.
c[128] Changes to achief1.f, frset.f.  [BH100517].

c[127] version='cql3d_100420'.
c[127] Petrov upgraded some netCDF coding from netCDF2 and netCDF3.
c[127] BH modified NPA routines to plot energy spectra.
c[127] BH modified tdpltmne for profile plotting at n=1:
c[127] Incorrect background species energy vs radius were
c[127] being plotted for nstop=1 cases, as occur in CQL3D/AORSA
c[127] coupled runs [BH100420].

c[126] version="cql3d_100126"
c[126] enescal/tescal/tiscal applied to all input profiles, static
c[126] and time-dependent.  Useful for adjusting units, e.g.
c[126] [BH100126].

c[125] version="cql3d_100116"
c[125] Added rdcmod="format2" read capability to rdc_multi.f, to
c[125] read multiple du0u0 files for each radius from DC. [YuP, 100101].

c[124] Located bug in tdtranspn.f which was causing unphysical bulge
c[124] in tail of fe in EC test case, and added 3-radial-point smoothing
c[124] of density profile and distn function to get working 
c[124] soln_method='it3drv' (fully implicit 3D interative solve)
c[124] [YuP and BH, Dec. 9, 2009].

c[124a] Fixed bug in setting density profiles for multiple ion species
c[124a] with same charge (e.g., D-T)  [BH091121].

c[123] version="cql3d_091118".
c[123] Reworked urfread_ read of ray data to accomdate possibility of
c[123] added data (complex cnper from toray) [BH091116].

c[122] Fixed bug to properly reset urfmod='disabled' in ainsetva, 
c[122] in case no urf modules are are setup [BH091116].

c[121] Re-arranged expressions for derivatives of gamma^m * alpha^-n,
c[121] to increase accuracy, etc.  The code is somewhat more stable 
c[121] for relativ=fully.  Can use mx=5, if higher-m coeffs at j<33 
c[121] are zero-ed, see cfpleg.  But not as good as hoped - still a 
c[121] noise at v~0 starts to develop.  The major part of cfpcoefr 
c[121] should be re-written to resolve the problem at v~0, if
c[121] in MeV range need be used.  Elsewise, the quasi-relativistic
c[121] relativ='enabled' approximation is quite sufficient, see
c[121] report CompX-2009-1_Fully-Rel.pdf  [Yuri Petrov, 091016].

c[120] The Intel ifort compiler on viz.pppl.gov differed in compiling 
c[120] a comparison between a real*8 variable and 0.0, so changed all
c[120] .ne.0. and .eq.0. in the code to be comparisons with real*8
c[120] zero=0.d0  (in about 35 source files). [BH090904].

c[119] Found error in velocity theta flux expression in advnce.h,
c[119] a "k" rather than "i" in indexing of distribution function.
c[119] Affected plot of velocity-space flux vectors, e.g., efld vectors
c[119] not purely parallel, and not much (perhaps nothing) else
c[119] [BH090827].

c[118] Found bounds check violation in choose(,) in fully relativistic
c[118] FP collision coeff calculation, and fixed coding in comm.h, 
c[118] micxinil.f and fpcoefr.f to agree with Franz\'s thesis
c[118] [YUP+BH, 090809].

c[117] Added additional conversions of old GRAFLIB plotting to PGPLOT
c[117] library, removing all references to graflib [YUP+BH, 090807].

c[116] Added pwrrf (rf power versus x=u/unorm) to netcdf file, modifying
c[116] storage [BH090611].

c[115] Modified urffflx to assign ray elements which are outside the
c[115] LCFS to the outer radial bin inside the LCFS (urfmod="enabled").  
c[115] This removed on out-of-bounds reference, resulting from the
c[115] new GENRAY capability for ray tracing in the SOL [BH090602].

c[114] Updated fully-implicit 3D interative solve from mainline
c[114] cql3d_f90ptr_090514 version.  This is fully f90 pointered
c[114] version rather than cray pointers, facilitating debugging
c[114] [bobh, 090514].

c[114a] Added option to output soft x-ray fluxes to the netcdf file at 
c[114a] each time step (see nml softxry); previously only at first and 
c[114a] last time step.
c[114a] Similarly, an option was added to output the distribution function 
c[114a] f and the QL coeff urfb (through nml netcdfshort) at each time
c[114a] step.  Updated to version="cql3d_f90ptr_090514" [BH090514].

c[113] Adding namelist rdc_upar_sign to reverse u_par order of rdc
c[113] diffusion coeffs (for rdcmod.eq."enabled"), appropriate
c[113] DC originated coeffs when eqdsk toroidal magnetic field is neg.
c[113] Reverted scaling of rdc diffusion coeffs for cases with
c[113] when coeffs originated on a grid with different normalization
c[113] energy [BH090424].

c[112] Fixing read a time-dependent generalized parabolic density/
c[112] temperature/zeff/elecfld/vphi.  There were errors when an
c[112] impurity was automatically added, showed up for a runaway
c[112] knockon test case [BH090312].

c[111] Fixing/adjusting plotting to remove glitches in 64-bit plotting
c[111] at very low ordinate values (e.g. power 10**-43 watts).
c[111] Added setup0 namelist variable lnwidth, specifying plotting
c[111] line width (useful for publication) [BH090220].

c[110] Add namelist &setup0 variable special_calls, which if set
c[110] to "disabled" will avoid system calls (which are not enabled
c[110] for some systems (e.g., franklin.nersc.gov, jaguar.ornl.gov).
c[110] default="enabled" [BH081216].

c[109] Switch entirely to f90 pointers rather than cray pointers.
c[109] This makes debugging easier for gfortran/gdb (and probably
c[109] debugging systems) BH081216.

c[108] Generalized rdc_multi to input data velocity normalization
c[108] different from vnorm (may be less, or greater), and added
c[108] option to read file specifying prompt loss.  This is for
c[108] coupling to the DC diffusion coeff calculation [BH081201].

c[108] Dimensioned namelist variables iurfcoll and iurfl 1:nmodsa,
c[108] in support of multiple ray type and multiple general species
c[108] application of rf [BH081106].

c[107] Fixed inadaquacy of calc of iprozeff = "parabola" or "spline"
c[107] for multiple ion species with same bnumb plus ions being
c[107] a general species [BH081031].

c[106] Fixed bug (gfortran compiler?) wherein xpts(1) and rx(1), etc.,
c[106] were offset by one real*8 memory position, contrary to equivalence
c[106] statement in freya.f, by adding dummy integer in frcomm.h
c[106] after npts.  The problem was causing freyasou bomb [BH,081029].


c[107] Zeroed vflux before summing in diagimpd [BH081125].

c[106] Modified tdsxray.f/tdsxr0.f so that sightline distance from
c[106] the detector to the plasma is increased.  Otherwise, the
c[106] calculated sightlines may not reaching the plasma [BH081106].

c[105] Checked for possibility that lsa.lt.lrza.  Has not usually been
c[105] the case in the past, but can cause memory overwrite [BH081106].

c[104] Fixed inadaquacy of calc of iprozeff = "parabola" or "spline"
c[104] for multiple ion species with same bnumb plus ions being
c[104] a general species [BH081031].

c[105] version="cql3d_biptr-mpi_080925"
c[105] Brought mainline cql3d version up to date w.r.t. Aug 19-21,2006
c[105] changes below.
c[105] Added changes for multispecies (ngen.ge.2) urf and rdc
c[105] QL diffusion. Add namelist .... [BH080918]

c[104] version="cql3d_biptr-mpi_080909"
c[104] Removed calculation of z00(i,j) [rhs of main equation] from
c[104] advnce.h statement functions, putting the z00 directly
c[104] into impavnc0 (impavnc) as an internal procedure function.
c[104] This aided in debugging a problem for nso=1, when source
c[104] parameters where set to zero.  Several traps agaist divide
c[104] by zero under these conditions were added.
c[104] Removed equivalences of dxp5/dxm5 and exp5/exm5, and
c[104] set the variables in micxinit.f [BH,080909].

c[127] version="cql3d_3d-implicit_f90ptr_080303".
c[127] Combined fully-implict 3D eqn solve and mainline cql3d
c[127] [bobh, 080303]

c[103] Added option to restart variable, nlrestrt="ncdfdist", to
c[103] restart using netcdf saved distribution, as a higher accuracy
c[103] alternative to restart from distfunc text file [bobh, 080203].

c[102] version="cql3d_biptr-mpi_080125".
c[102] cql3d running on 64-bit machines such as Intel core 2
c[102] quad processor, using gfortran64 and Lahey-Fujits lf64.
c[102] Added SAVE statements to subprograms with data statements,
c[102] removing some bugs related to differences in compilers.
c[102] Also fixed bug in asor, giving false turnon
c[102] of anaytic particle source [error introduced
c[102] in [98], below.  Added new D1MACH() function, based on
c[102] f90 calls.  Previous version had hex definitions which
c[102] were probably incorrect for 64 bit machines[bobh, 080125].

 
c[101] Made a few modifications, principally in tdxinitl.f and profiles.f     
c[101] to enable iprozeff='parabola' or 'spline' to work with two ion	       
c[101] species with same bnumb() (e.g., H+,D+, and/or T+).		       
c[101] Did a ngen=2 (D+,H+) test run (with ngena=2) in 		       
c[101] /home/bobh/cql3d/aorsa/D3D_test_case/122080.0/116MHZ/8th.1/tmp_ngen2   
c[101] Density H+ is 10**-4 of D+.					       
c[101] Uses makefile_lf95.  Results differ in 4th sig fig from previous       
c[101] ngen=1 D+ run, as expected.					       
c[101] Adjusted ainsetva.f to account for iprozeff=1 option for
c[101] calc of ion densities from ene/zeff so treat different ions
c[101] with same charge (e.g., H+,D+ and/or T+). [bobh, 060819].
c[101] 								       
c[101] Execution time increased from approx 8 minutes to 20, but	       
c[101] addition of the 2nd general species.  There is no QL diffusion	       
c[101] yet on the 2nd species.	  [bobh, Aug 19-21, 2006]		       

c[100] First results from new soln_method="it3drv" option using
c[100] sparskit2 to iteratively solve full 3d (2V-1R)
c[100] implicit  cql3d equations [bobh, 071108]
c[100] Dimensioned wk ==> wk(*) in zcunix.f: coeffs to prevent
c[100] tripping of array length checker [bobh, 070805].
c[100] New iterative sparse-matrix solution capabilities for solving 
c[100] the basic FP matrix equation are introduced, as specified
c[100] by new namelist var soln_method, invoking SPARSKIT2 routines.  
c[100] Additional new namelist vars are droptol and lfil. 
c[100] The distinction between distributions for the velocity and
c[100] radial steps has been removed, according to setting of the
c[100] internal variable ipacktr=0 in tdtrmuy.f.  The resulting
c[100] transp="enabled" solutions did not change substantially.
c[100] This is prepatory to solution of fully-implicit 2D-1r 3D
c[100] equations (with transport) by iterative sparse-matrix methods
c[100] [bobh, 070424].

c[99] Changed nlwritf and nlrestrt from logical to character*8,
c[99] for more flexibility in specifying restart settings.
c[99] Will need to check use of past cqlinput files. [bobh, 070414].

c[98] Broke setup namelist up into two: setup0 and setup.
c[98] This was necessary for SWIM IPS, involving namelist writes,
c[98] and reads.  Backwards compatibility is maintained by checking
c[98] the cqlinput file for existance of &setup0:  if not, then
c[98] cqlinput is rewritten with first occurance of &setup changed
c[98] to &setup0.  Initial (old) cqlinput is restored at the
c[98] end of the run.  [bobh, 070414]

c[97] Added runaway electron related variables (denra,curra,
c[97] ucrit,knockon,eoe0,srckotot), also wpar,wperp to the 
c[97] netcdf output file. [bobh, 070407]

c[96] The namelist input system was restructured to facilitate
c[96  coupling, and backward compatibility after further cql3d
c[96] upgrades, with the SWIM Integrated Plasma Simulator (IPS):
c[96] namelists, namelist type and dimensions (i.e. declarations),
c[96] and subroutines setting defaults have been separated out from
c[96] other code variables . Consequently, the IPS interface uses
c[96] files from the mainline cql3d distributions:  the name.h,
c[96] name_decl.h,frname.h,frname_decl.h include files, and sets 
c[96] defaults using the subroutines in aindfpa.f,aindlft.f,eqindflt.f,
c[96] urfindfl,frinitl.f (of the same root names) [BH070310].

c[95] A 'if ...write(*,*)' [never executed] was added to zcunix.f
c[95] to get around compiler problem on viz.pppl.gov SGI machine.

c[94] Modified eqtopeol (reads Culham TOPEOL file in lieu of eqdsk)
c[94] to integrate poloidal B-fields from the equatorial plane
c[94] rather than minimum -Z plane.   This gets around a problem
c[94] of inexact B-field/singularities outside last closed flux surface
c[94] [Part of benchmarking effort with Saveliev] [bobh, 070116].

c[93a] Vickie Lynch, ORNL found an ancient bug in freyasou (setting
c[93a] of array with unset index) which could cause an overwrite,
c[93a] but seems to have not usually been a problem. [bobh, 060824].

c[93] Added Vincent Tang (MIT, next LLNL) coding for a passive NPA
c[93] synthetic diagnostic.  This is a modification of the SXR
c[93] synthetic diagnostic.  In future work, it is intended to
c[93] convert active NPA synthetic diagnostic coding developed
c[93] in Matlab by Vincent to a fortran module with cql3d. 
c[93] [bobh, 060622].

c[92] makefile_lf95_ptr is system developed by Nikolay Ershov     
c[92] to produce a Fortran 90 pointered version of the code,      
c[92] as alternate to the standard Cray pointered version obtained
c[92] with makefile_lf95.  Comparison of these two makefiles shows the    
c[92] only differences are in four lines referring to doptr.py    
c[92] and tmpptr.f.					       
c[92] The source code modifications are carried out with the      
c[92] python script doptr.py in ./ptr, and additional files       
c[92] in that subdirectory.
c[92] Future additions to code memory should be added in both the
c[92] mainline cray pointered version, and in the f90 pointer mods in
c[92] ./ptr.
c[92] Please code consistent with this scheme for modification in code
c[92] storage, so that the two pointering systems are carried forward.
c[92] [N. Ershov; bobh, May\'06].     

c[91] Removed limitation in urfmod and vlfmod options of no more 
c[91] than 3 rf wave types, or, 3 harmonics for 1 wave type.  Now
c[91] can have multi-wave types with multi-harmonics.  The limit
c[91] on number of wave-types*harmonics is set by parameter 
c[91] nmodsa [bobh, 060319].

c[90] Changed passing of the distribution function to functions
c[90] bsl and bsu by common block rather than as function argument.
c[90] This GREATLY reduced execution time for f90 pointered version
c[90] cql3d (Lahey-Fujitsu f95 compiler, V6.0)  [Ershov, 060316].

c[89] Introduced parallelization of impavnc/impanv0 solver of the
c[89] 2D bounce-averaged equations on each flux surface.
c[89] See MPIreport.pdf in cql3d_manual directory.  Parallelization
c[89] is obtained with CMPI comments/insertion points in source,
c[89] activated by makefile_mpi, using python scripts in ./mpi
c[89] [Ershov, 060114].

c[88] Fixed bug in bsu.f which may effect bootstrap current calc
c[88] at outer radius, rho(lrzmax) [bobh, 051101].
c
c[87] Added NPA (neutral particle analyzer) diagnostic skeleton code
c[87] based on first general ion species and added neutral profile. 
c[87] See npa_diag and ipronn related namelist [Vincent Tang (MIT), 
c[87] bobh, 051007].
c
c[86] Added additional phenomenological radial diffusion drr from
c[86] radius 0. out to normalized radius difus_rshape(8), to 
c[86] simulate sawteeth effects [bobh, 050921].
c
c[85] Substantial bug fix in radial dependence of velocity independent
c[85] radial diffusion coefficient.  Radial dependence specified by
c[85] difus_rshape() is implemented.  Previously, radial dependence
c[85] was constant with r, for constant in velocity space cases 
c[85] [bobh, 050921].
c
c[84] Added namelist var difus_type with possible values of "specify"
c[84] (default, giving older methods of specifying vel and r variation
c[84] of drr, and new "neo_smpl" giving simple, velocity-independent
c[84] neoclassical, bannana regime drr, and "neo_plus" which adds
c[84] "specify" type drr to the "neo_smpl" drr. [bobh, 050919]
c
c[83] Fixed bug in tdtravct.f. Target density for radial convection
c[83] was not being set for colmodl.ne.0, preventing radial transport
c[83] runs with usual colmodl=3.  Error introduced with colmodl=0,
c[83] radial transport update in March, 2002. [bobh, 050913].
c
c[82] Fixed bug for bsign=-1 cases (neg tor fld) which was preventing
c[82] addition of salphal additional damping for iurfl="enabled".
c[82] Bug existed for about last year [bobh, 050911].
c
c[81] Small k_parallel-width ray elements were giving zero damping
c[81] due to inaccuracy in interpolating the diffusion coefficient
c[81] to neighboring velocity-space grid points.  This was improved
c[81] by increasing storage of ifct1/ifct2 from 8-bit words to 
c[81] 16-bit words [bobh, 050812].
c
c[80] Fix bug: Sauter\'s unicity of f at j=1 (v=0) in impavcn0 was
c[80] being  applied in lbdry(1).ne."conserv" cases ("fixed"
c[80] and "scale"), which already applied unicity.  An out of bounds
c[80] reference was generated in the FP coeff matrix. Effects
c[80] are unknown [bobh, 050804].
c
c[79] Moved collisional and linear damping calc out of urfdamp1
c[79] and urfdamp2, to new subroutine urfdampa.  This fixes a bug in 
c[79] which the additional damping was not calculated for ray elements
c[79] for which the QL damping was not calculated [bobh, 050426].
c
c[78] Changed flag indicating output of diagnostic damping rate
c[78] from iurfl="damp_out" to iurfcoll="damp_out", thus enabling
c[78] simultaneous input of additional damping in salphal and
c[78] output of diagnostic damping in salphac [bobh, 050423].
c
c[77] Included gyro-radius with banana width in lossmode()=simplban
c[77] scrape-off loss model [bobh, 050217].
c
c[76] Fix bug: Added bsign1() in urfread_.f to account for the
c[76] different sign conventions in genray and toray when bsign
c[76] .lt.0. This bug, introduced in [74] could give zero cyclotron
c[76] damping for negative toroidal field in the eqdsk [bobh, 050206].
c
c[76] rdcmod modification to read in externally computed RF
c[76] diffusion coefficients for an array of flux surfaces
c[76] [bobh, 041111].
c
c[75] Fix bug: fixed eqdskin functionality so specification of full
c[75] path for the eqdsk works, not just local eqdsk [bobh, 040902].
c
c[74] Version designated as cvs_cql3d_v2-00_040820
c[74] Added netcdf read of standard ray data input files, as
c[74] alternative to reading text files. Modified netcdfrf.f. 
c[74] Added namelist variables rffile(1:3),rfread
c[74] Names of input netcdf files can be specified through rffile.
c[74] Bug fixed: sdpwri was output to .nc file in netcdfrf.f,
c[74] but was not dimensioned.  Unknown effect.
c[73] [bobh, 040814].
c
c[73] Added namelist variable nmlstout, to control namelist o/p to 
c[73] stdout.  Various slight changes in write(*,*).
c[73] Added vol int of FSA power densities entr ==> entrintr(,) and
c[73] put result in .nc file at each time step.
c[73] Fixed bug in zmincon,zmaxcon determination.
c[73] [bobh, 040326].
c
c[72] Added netcdf output giving fusion/neutron rates and powers,
c[72] and RF damping due to salphal.
c[72] Added time-dependent input power in urf rays (nurftime.gt.0).
c[72] [bobh, 030618].
c
c[71] Version designated cvs_cql3d_v1-12_030115. 
c[71] CVS repository changed /usr/local/cvsroot.
c[71] Added netcdf output giving ratio of theoretical electrical 
c[71] conductivity by Connor and by Kim-Hirshman/Sigmar  [bobh, 021121].
c[71] Added additional output to _"flux_" netcdf files [bobh, 030115].
c
c[70] Added variable vlh_karney to enable Karney and Fisch
c[70] u-variation of LH Dql (PF 28, 116 (1985)).  [bobh, 021118].
c
c[69] Version designated cvs_cql3d_v1-11_021028.
c[69] Added new capability to output vel space fluxes in x,theta-coords
c[69] (see netcdfvec.. variables in cqlinput_help).
c[69] Added netcdf output of specific current dens j(u,species,radius).
c[69] [bobh, 021025]
c
c[68] Version designated cvs_cql3d_v1-10_020914.
c[68] Added (much) more accurate calculation of qsafety.  I presume
c[68] that in a bounce-avg code, this won\'t have significant effects.
c[68] It does affect zmax. [bobh, 020914]
c
c[67] Version designated cvs_cql3d_v1-9_020607.
c[67] Added some symmetrization options (see eqsym namelist variable).
c[67] Added eqdskin namelist.  Added netCDF variables [bobh, 020607].
c
c[66] Version designated cvs_cql3d_v1-8_020101.
c[66] Added most remaining data in screen output  into netcdf file.
c[66] Fixed some dimensioning bugs in netcdf o/p of flux vectors.
c[66] Adjusted vector plots of vel space fluxes for PGPLOT.
c[66] [bobh, 020101]
c
c[65] Added parallel current and resistivities to netcdf file.
c[65] [bobh, 010814]
c
c[64] Added simple banana loss model removing particles with banana
c[64] width greater than distance to the plasma edge [bobh, 011125].
c
c[63] Added parallel current and resistivities to netcdf file.
c[63] [bobh, 010814]
c
c[62] Fixed bug in restvty.f: efswtchn.eq.ne0_hh ==> ne0_hh [bh,010812]
c
c[61] Added X-ray detector specs and calculated fluxes to
c[61] the netcdf output file (when softxry.en."disabled" [bobh,010526]
c
c[60] Added a globally convergent Newton-Raphson iteration scheme
c[60] to find radial transport velocities which maintain a given
c[60] target radial density profile [bobh, 010426].
c
c[59] Added a general radial and velocity-space profiles for
c[59] the radial diffusion coefficient d_rr [bobh, 010420].
c
c[58] Version designated cvs_BH20010411. [bobh, 010411]
c
c[57] Output files now all pre-pended with contents of the
c[57] character*40 variable, mnemonic. [bobh, 010411]
c
c[56] Added netCDF output facility for velocity-space flux vectors,
c[56] through netcdfmain.  This approach could be expanded to
c[56] additional output, similar to pltmain facility. [bobh,010411]
c
c[55] Switched to CVS code maintenance system at bobh.compxco.com.
c[55] [bobh, 010319]
c
c[54] Fixed pitch angle references in vlf.f. Found that there is
c[54] an incompatibility between cosz,sinz,tanz setting and the
c[54] maximum pitch angle given by imax, with taunew="enabled".
c[54] Changed taunew default to "disabled".  This needs more attention.
c[54] [bobh, 010316]
c
c[53] Added target current calculation from eqdsk.  Fixed elecfld
c[53] iteration to achieve target current, for multiflux-surface
c[53] runs. [bobh, 010301].
c
c[52] Added e-e Bremsstrahlung to existing e-i, for Xray calc.
c[52] Debugged Xray calc [bobh, 001200].
c
c[51] Incorporated numerical bootstrap current calc  [bobh, 990731].
c[51] Added numerical calc of Hirshman'88 and Sauter'99 analytic
c[51] bootstrap current, for comparison  [bobh, 990825].
c
c[50] Added netCDF o/p for main plasma parameters and RF data
c[50] [bobh, 990601]
c
c[49] Added quasineutrality calc of E_ll to cqlpmod="enabled" model
c[49] [bobh, 9904]
c
c[48] Added relativistic QL diffusion coeffs (vlfmod="enabled") to 
c[48] cqlpmod="enabled" (1D-in-dist-along-B,2V) model.  [bobh, 990402].
c
c[47] Major modifications to run code on both 64- and 32-bit
c[47] machines (machinea=1 or 2), and replacing GRAFLIB graphics
c[47] with more public domain PGPLOT.  Code compiles
c[47] on PC with Absoft, DEC Alpha with DEC For, and J90[bobh, 990501].
c
c[46] Added RFP upgrades: radcoord for new radial coords, calc and
c[46] print pol and tor current, -(epsi) for psilim.lt.psimax,
c[46] minor mods for RFP equilibria (removing a kluge for
c[46] non-monotonicity of B(s).   [bobh, 980919].
c
c[45] Corrected bounce-average in sourceko.f, to get agreement
c[45] with MNR.  Multi-flux surface enabled. bobh,980501
c
c[44] Corrected cross-section for KO operator to agree with Heitler
c[44] bobh, 980417
c
c[43] Fixed bug of incomplete copy of complex*16 numbers in urfread.f
c[43] bobh,980411
c
c[42] Update vth and sptzr at each time step according to current
c[42] temp.  Option for calc of elecfld using neoclassical resistivity
c[42] plus runaway electron current. Fixed pdenra in aclear[bobh, 970922].
c
c[41] Changed out impavnc.f and impavnc0.f for Olivier Sauter\'s
c[41] new version which uses standard LAPACK routines sgbtrf sgbtrs, 
c[41] rather than zband1 (which might have error under f90)[bobh,970901].
c
c[40] Added time-dep profile option for (1+cos(pi*t))-variation 
c[40] bobh, 970714].
c
c[39] Added calc of ko source using pitch angle average primary
c[39] distribution: fsa [970701], local in pol. angle [bobh, 970706].
c
c[38] Many little changes in plotting, sourceko.f soucrit.f[bobh,970417].
c
c[37] Restored equivalence of temp2 and fpn(xpar,xperp), added
c[37] subregion for ploting distributions and fluxes [bobh,970331].
c
c[36] Added coding to warn of potential overwrite by tamt1,tamt2,
c[36] when mxa is too large [bobh, 970312].
c
c[35] Added improved, fully numerical calc of tau and dtau,giving uniform  
c[35] poloidal density for isotropic distributions [bobh, 970212].
c
c[34] Added pol. angle variation for parallel distribution appearing 
c[34] in knock-on source.  Using Rosenbluth formula [sc,bobh, 970122].
c
c[33] Upgrade in cfpcoefn for high Zeff contrib. to Cee [bobh, 961217]
c
c[32] Added plots of runaway current and density [SC].
c[32] Added write and read of preload distribution [bobh, 960919].
c
c[31] Added new "conservative" treatment of knock-on source, with new
c[31] subroutines for reduced distn fle, and sourceko. (bobh, 960820).
c
c[30] (Temporarily) increased max resolution of calc of reduced distn in 
c[30] pltprppr to 201x10001, as fix for runaway mesh resolution problem 
c[30] (bh 960724 & 0801).  May use unnecessary storage. Added xprpmax.
c{30} Gave fpn storage.  Added jpxy.le.jpxya, ipxy.le.ipxya to namelist.
c[30] Don\'t try anything that uses xtail, xhead, ytail, yhead!
c
c[29] Changed Bremsstrahlung radiation evaluation in lossegy. An enhancement
c[29] factor is multiplied in the highest energy range to force
c[29] containment of runaway electrons. Also changed the limits for
c[29] energy for different formulas so that transition is smooth. (SCC960625)
c
c[28] Introduced array of time step settings. (bobh 960620).
c
c[27] Further changes and additions relevant to electron runaway problem:
c[27]  renamed efield.f to coefefld.f, restinit.f to efield.f and added
c[27]  control algorithm to calc. elec. fld. to give specified target
c[27]  current.  Added bremmstahlung reaction force on electrons, and
c[27]  enegy dependent Coulomb logarithm.  Also additional preloading
c[27]  distributions (in finit.f), time-dependent profiles (new
c[27]  subroutines in  profiles.f), plotting of coefficients.
c[27]  Also, knock-on electron source (march 96). (bobh, circa 960521)
c
c[26] Omitted damping in npar1*npar2.lt.0. case, in urfpack. This situation
c[26]  can occur for near perp. injection, but needs more work on logic.
c[26]  (bobh 960421).
c
c[25] Simple search for accurate magnetic axis rather than Newton
c[25]  iteration, in eqorbit.f (bobh, 960309)
c
c[24] Minor change in tdoutput.f of synchrotron o/p.  Not sure correct
c[24]  yet (bh 960305)
c
c[23] Took into acct. like-like collision factor in neutron rates
c[23]  for D-D particles (i.e., 0.5). Changed fusion cross-sections to 
c[23]  Bosch & Hale, Nucl Fus, \'92. Benchmarks with ONETWO (bh 950307).
c
c[22] Fixed over-write of nrayelt0, for nharms.gt.1, in urfdamp0.
c[22] Extended background interpolation for freya from lrz to lrzmax.
c[22] (bh 950221)
c
c[21] Added toroidal rotation effect for ions into FREYA (bh 950109).
c
c[20] Added calculation and plots of fusion power (bh 950102).
c
c[19] Added zeff profile namelist input, thru zeffin (bh 941222).
c
c[18] Added possibility to taper rf diffusion coefficient over
c[18] the last 10 velocity points of the mesh (ineg=trunc_d), to 
c[18] assist in obtaining solutions in ion/icrf problems where there
c[18] is otherwise an rf-induced ion runaway (bh).

c[17] Added pure perpendicular rf diffusion in vlh module (bh).

c[16] Added plot of the theta-averaged distribution function (scchiu).

c[15] Small change to impavnc0.f: In the case that lbdry=fixed or
c[15] scale, lbdry0=enabled, j=1, i=iy, nvar set =1 rather than inew=1.
c[15] This cleans up the "coefficient" matrix somewhat, possibly
c[15] removes a potential error, but had no effect on test cases
c[15] lrz=1, test case with EC.  (bh 94/08/16).
c
c[14] Added vlf.. routines.  These are single flux surface versions
c[14] of the urf... routines, and permit study of LH, FW, and multiple
c[14] cyclotron harmonic effects on electrons or ions, specifying
c[14] wave parameters and region of flux surface for QL diffusion,
c[14] through simple namelist input.   (bh 94/06/21).
c
c[13] Generalized single flux surface LH QL diffusion model (vlh...)
c[13] to multiple resonance regions.  Also more deeply pointered
c[13] the urfb, urfc,urfe,urff, so save significant memory. 
c[13] Started new file (notes.h) keeping information on variables,
c[13] etc., on the code.   (bh 94/06/21).
c
c[12] Combining Sauter version CQLP_B_940425 and Harvey version to that
c[12] below date. Indenting done of bh code according to [10] below, using
c[12] Sauter-s emacs indenting setup files.  (bh 94/05/07).
c
c[11] Change bcast(ca,6*..) and sscal(ca,..) in cfpcoefn and cfpcoefr
c[11] to bcast(ca,3*iyjxua,..), bcast(cd,3*iyujxa,..) and so on, as
c[11] cd(1,1) is not after cc(iya,jxa+1) in memory => bhos_940425
c[11] This version is copied to CQLP_B_940425 and is the start for
c[11] B versions of CQLP/CQL3D codes. (os 94/04/25)
c
c[10]  Indented whole code according to Emacs-Fortran rules, with
c[10]  2 columns shift after loops, ifs and continuation lines.
c[10]  Had to insert new continue statements, as one should not have
c[10]  2 loops using same label. It is also recommended not to use
c[10]  "1" as do loops label. The default continuation character has been
c[10]  set to "+". (os) => new version 940304
c
c[9]  Fix gftp defn. in urfdamp2.f => 5% change in powurf. bh, 940301
c
c[8]  Changes in urfpack and urfedge correcting treatment of 
c[8]  weighting of edge of resonance region. bh, 940227.
c[8]  Added namelist variable wdscale.
c[8]  Should check nharm=0 and inside of resonance layer cases further.
c
c[7]  Synthesized Sauter and Harvey versions incorporating following
c[7]  code changes [6] into CQL3D version with FREYA, ion-cyclotron
c[7]  damping, and multiple cyclotron damping on electrons or ions.
c[7]  Results stored in CFS /u650/940224a/bh940224.tar  bh, 940224.
c
c[6]  CQLP is built starting from the CQL3D E_930209 version, then
c[6]  several new versions  followed. The last one before O.Sauter
c[6]  left GA is the 930624 version. Then the version used for the
c[6]  4th PET93 workshop  paper is the 930910 version saved by O.Sauter
c[6]  on cfs in CQLP_A931103.taZ.         
c[6]  Now this version has been cleaned up, and dcofleg modified to
c[6]  represents the contribution between i-1/2 and i+1/2, thus f is
c[6]  not interpolated at i+1/2 as before (in particular in cfpcoefn). 
c[6]  This version, 940128, is given to B. Harvey.  Olivier S. 940128.

c[5]  Fixed us some storage problems:  rovsp re-dimensioned, 
c[5]  lndum adjusted accordingly, and eqdumptr put in common.
c[5]  940215, BobH

c[4]  Changed sign of n_par, if partner="lsc". BH 930131.

c[3]  Modified prppr.f slightly to get rid of minor gliches
c[3]  in interpolation to f(xpar,xperp) at emax. BH 940130

c[2]  Fixed ainspec.f so it properly calculates maxwellian
c[2]  species designators kionm(nionm) BH 930130

c[1]  Added the namelist variables urfrstrt and urfwrray and
c[1]  made associated changes in urf routines. Bob H. 940127

c[0]  EQRHOPSI.F :eqsource="tsc" eqpsimin/max set as for eqdsk.
c[0]  Began this record file.   (BobH, 931222)

c[-1] The CQL3D code has been under development since about 1985
c[-1] by Harvey and McCoy, as a 3D 2v-1r (v0,theta0,rho) FP
c[-1] code based on (1) the cql 2D-in-v Kerbel and McCoy 
c[-1] FP collision code at each flux surface, plus (2)
c[-1] an added radial variable enabling accounting for rf
c[-1] quasilinear diffusion of electrons and ions using ray tracing
c[-1] input data, determination of self-consistent distributions
c[-1] and ray energy damping, developed by Harvey and McCoy 
c[-1] [First reported for LH heating and current drive in a
c[-1] combined IAEA paper by Soeldner et al., Washington, D.C., 1990.]
c[-1] Kerbel moved on from the cql effort in ~1988, and McCoy from
c[-1] cql/cql3d in 1990-91.
c[-1] Radial dependent synthetic diagnostics have been added, such
c[-1] as Bremsstrahlung xray emission.  ECE emission is calculated
c[-1] from CQL3D distributions, in the HORACE code.
c[-1] Diffusive radial transport and radial pinch is solved by
c[-1] an implicit, alternating direction differencing scheme.
c[-1] See http://www.compxco.com/cql3d_manual_110218.pdf for code
c[-1] status in 1992.


