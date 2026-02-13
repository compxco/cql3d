c     comm.h
c
c.......................................................................
c     There are a number of arrays intended for temporary storage
c     in a subroutine, or for simple passes of data to
c     associated subroutines:
c
cBH180527:  The equivalences listed for tem[1-6]/temp[1-6] are
cBH180527:  have been removed (at some past time).
c     iyjx2=(iy+2)*(jx+2)
c     tem1(iyjx2) --> tem6(iyjx2)
c     temp1(0:iyp1,0:jxp1) --> temp6(0:iyp1,0:jxp1), 
c      "equivalenced" to tem1(iyjx2) --> tem6(iyjx2).
c     item1(iyjx2) --> item6(iyjx2)
c      "equivalenced" to tem1(iyjx2) --> tem6(iyjx2).
c     iyjx2=(iy+2)*(jx+2)
c     MOREOVER: We assume temp[1-6] are in contiguous storage,
c               so we can reference the whole six arrays through tem1.
cBH180527: Dimension of tem[1-6] modified to iyjx2l=max(iy+2,lrz)*(jx+2),
cBH180527: to handle possible situation in netcdfrw2.
c
c     tam1(jx) --> tam30(jx)
c     temc1(iy) --> temc4(iy)
c     tz1(lz) --> tz2(lz)
c     tr(0:lrza)
c     tr1(0:lrza) --> tr5(0:lrza)
c     itemc1(iy) --> itemc2(iy)
c     urftmp(nrayelts*nrayn*5)
c.......................................................................
c
c
c.......................................................................
c     Add in type,size,common declarations for namelist variables.
c     This needs to precede the rest of comm.h declarations so
c     that name_decl.h can be used by itself in aindflt.f.
c     aindflt.f will also be used for setting defaults in the
c     SWIM project Integrated Plasma Simulation (IPS) modules.
c.......................................................................

      include 'name_decl.h'

c.......................................................................
c     variables that take on the values assigned to parameters.
c.......................................................................

      common /params/
     1  idim,iyjx,iyjxp1,iyp1jx, iyjx2, iyp1,jxp1,
     1  lrors,
     1  mxp1,mbet,
     1  nonch,niong,nionm,ntotal


c**********************************************************************
c     Variables in common block diskx have as their last dimension lrza.
c     Thus they are dimensioned with respect to the radial coordinate.
c     If lrza=1  ==> single flux surface CQL run.
c     Note that input variable lrz can be less than lrza. For some of
c     the larger arrays we allocate space using lrz rather than
c     dimensioning with lrza to save some space.
c
c     VECTORS
c
c**********************************************************************
c
      !YuP[2021-03] Rearranged, to collect lrorsa arrays together
      !(those are used in both CQL3D and CQLP, 
      !but with different meaning of l_ index)
      !For future CQL4D development: need to "split" l_ index into two.
      !Made some of them into pointers (over lrors)
      
      integer, dimension(:), pointer ::
     1  itl_,itu_,
     1  iy_,iyh_,iyjx_,
     1  inew_,inewjx_,
     1  ieq_,   !!!(lrors+1),
     1  n_,nch
      
      real*8,  dimension(:), pointer ::
     1  consn,consn0,
     1  currmtp,currmt,rovsloc,sptzr,sgaint1,
     1  thb,time_,twoint
      
      common /diskx_lrors/
     1  itl_,itu_,
     1  iy_,iyh_,iyjx_,
     1  inew_,inewjx_,ieq_,
     1  n_,nch,
     1  consn,consn0,
     1  currmtp,currmt,rovsloc,
     1  sptzr,sgaint1,
     1  thb,time_,twoint
     
      integer indxlr(0:lrorsa),indxls(0:lrorsa)
      common /diskx_lrorsa/ indxlr,indxls
     
      common /diskx_lrza/
     1  bmod0(lrza),btor0(lrza),bthr(lrza),btoru(lrza),bthr0(lrza),
     1  currt(lrza),currxj0(0:lrza),
     1  currtp(lrza),
     1  currxj(0:lrza),
     1  currpar(lrza),curreq(lrza),
     1  zreshin(lrza),zreskim(lrza),rovsc(lrza),rovsc_hi(lrza),
     1  curtor(lrza),curpol(lrza),ccurtor(0:lrza),ccurpol(0:lrza),
     1  deltapsi(lrza),
     1  eps(lrza),etll0(lrza),
     1  lorbit(lrza),lmdpln(0:lrorsa),  !YuP[2022-02-11] changed lmdpln size to lrorsa
     1  psimx(lrza),pibzmax(lrza),psidz(lrza),
     1  qsafety(lrza),
     1  r0geom(lrza), r0drdz(0:lrza),rgeom(lrza), zgeom(lrza),
     1  rovs(lrza),rovsn(lrza)
     
      integer,dimension(:),pointer :: nefiter_ !(lrors)! counts iterations of el.field for each flux surface 
      !YuP[2021-04-09] Now pointer
      common /e_iterations/ nefiter_ !(lrors)

      common /diskx/
     1  vthe(lrza),
     1  xlbnd(lrza),xlndn0(lrza),
     1  zmaxpsi(0:lrza),zmaxi(0:lrza),
     1  zmaxpsii(0:lrza),
     1  vphipl(lrza),
     1  srckotot(lrza),elecr(lrza),
     1  denfl(lrza),denfl1(lrza),denfl2(lrza),
     1  tauii(lrza),tau_neo(lrza),drr_gs(lrza),
     1  rhol(lrza),rhol_pol(lrza)
      common /diskx/
     1  taubi(lrza),tau_neo_b(lrza),drr_gs_b(lrza),
     1  rhol_b(lrza),rhol_pol_b(lrza)

      !YuP[2021-04-14] Remaining arrays related to lz are pointers now:
      real*8, dimension(:), pointer :: den_of_s,den_of_s1,den_of_s2,
     1  tz1,tz2 !(lz)
      common /den_lz/ den_of_s, den_of_s1, den_of_s2, tz1, tz2 !(lz)
     
      real*8, dimension(:), pointer :: tauee,taueeh,starnue,zeff,zeff4
      common /tauee_starnue/ tauee,taueeh,starnue ! lsmax or lrzmax
      common /zeff_/ zeff,zeff4  ! lsmax or lrzmax
      !YuP[2021-03-17] Collected tauee,taueeh,starnue  here,
      !and made as pointers.
      !It seems the only reason why we need them 
      !over full grid (lrzmax or lsmax) is because of printout in tdoutput.
      !Anyway, they are set over full grid in the CQL3D or CQLP, 
      !see, e.g. starnue(lr_) or starnue(ls_)
        
c..................................................................
c     2-D ARRAYS
c..................................................................

      !YuP[2021-03] Rearranged, to collect lrorsa arrays together
      !(those are used in both CQL3D and CQLP, 
      !but with different meaning of l_ index),
      !and made them into pointers.
      !For future CQL4D development: need to "split" l_ index into two.

      integer, dimension(:,:), pointer :: jchang !(ngen,lrors)
      real*8,  dimension(:,:), pointer :: currm,energym !(ngen,lrors)
      common /diskx_lrors/
     1  jchang,
     1  currm,
     1  energym
      
      common /diskx_lrza/
     1  energy(ntotala,lrza),
     1  vth(ntotala,lrza),
     1  den_fsa(ngena,lrza), 
     1  den_fsa_t0(ngena,lrza), 
     &  reden_t0(ngena,lrza),temp_t0(ngena,lrza),
     &  reden_wk(lrza),temp_wk(lrza),
     1  curr(ngena,lrza),
     1  hnis(ngena,lrza),
     1  ratio(ngena,lrza),
     1  wpar(ngena,lrza),wperp(ngena,lrza),
     1  xlndn00(ngena,lrza),xlncur(ngena,lrza),
     1  xlndn(ngena,lrza),
     1  xlndnr(ntotala,lrza),energyr(ntotala,lrza),
     1  currr(ngena,lrza),xlndnv(ntotala,lrza),
     1  energyv(ntotala,lrza),currv_(ngena,lrza),eoe0(ngena,lrza),
     &  enn(lrza,npaproca)
     
      !YuP[2021-04-09] collected RE-related arrays, made them into pointers
      integer,dimension(:,:),pointer :: jxcrit    !(ngen,lrors)
      real*8,dimension(:,:),pointer :: ucrit,denra,curra,fdenra,fcurra   !(ngen,lrors)
      common / RE_related / jxcrit, ucrit,
     1  denra,curra, fdenra,fcurra
     
     
      common /diskx_0/
     1  alm(0:mbeta,lrza),
     1  betta(0:mbeta,lrza)
     
      pointer entrintr
      real*8 entrintr(:,:)    !!!(ngen,-1:15)
      common /diskx_1/ entrintr  ![2023-01] was entrintr(ngena,-1:15)
      !YuP[2023-01-03] Made it a pointer. 
      !Previously - static array, it was causing floating-point exception
      !and other issues.

c..................................................................
c     Next follow variables and arrays not dimensioned by lrza.
c..................................................................

c..................................................................
c     SCALARS.....
c..................................................................

      character*8
     1  outfile,
     1  prefix

      character*512 t_

      common
     1  clight,charge,restmkev,symm,
     1  cnorm,cnorm2,cnorm3,clite2,cnorm2i,cnormi,
     1  ergtkev,eps0,elect,eratio,eovsz,eovedd,
     1  em6,em8,em10,em12,em14,em37,em40,ep37,em90,ep90,
     1  ep100,em100,em120,em300,four,fourpi,fc,
     1  gszb,gsb,gsem,gsep,gszm,gszp,gseb,gszm2,gszp2,gszb2,gsla2,
     1  gslb2,gsnm,half,
     1  iyh,istate,itl,itu,
     1  impcoef,imprf,impelec,impcah,irstart,impadi,
     1  jxm1,jlwr,jmdl,
     1  kgnande,kelecg,kelecm,kelec,kionn,kiong(ngena),kiongg(ngena),
     1  kionm(nmaxa),
     1  l_,lr_,indxlr_,indxls_,lmdpln_,ls_,
     1  miyjx,ibeampon,ibeamponp
     
      !YuP[2022-05-31] Added, for handling species
      integer, pointer :: km_same(:), kg_same(:) !(ntotal)
      common/arrays_species_int/ km_same,kg_same
      !YuP[2022-05-31]

      common
     1  ipad1,r0geomp, rgeomp, zgeomp, rgeom1, rgeom2,
     1  niyjx,nccoef,ncentr,navnc,ncplt,n,
     1  nflux,nframe,npltmx,
     1  outfile,ipad2,one,one_,
     1  prefix,
     1  pi,pio180,pio2,psir,proton,
     1  rgyro,resist,resistn,rovsf,rmp0,rovscf,rbgn,
     1  stopm,
     1  tei,timet,t_,three,third,sevenhun,two,twopi,
     1  vparl,vpovth,
     1  xconn,xmax,ipxy,jpxy,
     1  zero,zstar,
     1  iplot,nplott,isave,nsavet,nirzplt,
     1  nefiter ! counts iterations for el.field (efswtch="method5")
      common
     1  elecfldc,elecfldb,  !V/cm, Added for ampfmod case
     1  it_ampf   !iteraction counter for ampfmod case
     
c..................................................................
c     VECTORS.....
c..................................................................

      character*8
     1  text

      common
     1  ipad3,text(4),
cBH111124     1  iota(0:750),realiota(0:750),
     1  iota(0:6400),realiota(0:6400),
     1  i1p(2),itab(3),ipad4,tab(3),
     1  imsh(20),
     1  frbuf(1024),
     1  ws2d(2500)

      common
     1  ekev(ngena),
     1  engain(ngena),
     1  tnorm(ngena)
      common
     1  fions(ntotala),
     1  gama0(ntotala)
     
      common
     1  wkbc(3*nbctimea),iopbc(2)

      character*8
     1  mplot

       common
     1  mplot(lrorsa),
     1  tplot(nplota),tsave(nsavea)

c..................................................................
c     TWO-DIMENSIONAL ARRAYS.....
c..................................................................

      common
     1  gamt(ntotala,ntotala),gama(ntotala,ntotala),
     1  satioz2(ntotala,ntotala),satiom(ntotala,ntotala),
     1  sgain(8,ngena)

c..................................................................
c     Allocatable arrays allocated in subroutine ainalloc
c..................................................................

      pointer :: ix1(:),ix2(:),ix3(:),ix4(:),     !!!(0:mx)
     1           ix5(:),ix6(:),ix7(:),ix8(:)      !!!(0:mx)
     
      pointer :: tom1(:),tom2(:),tom3(:),tom4(:)  !!!(0:mxp1)

cBH_YP090809  WAS: choose(0:mx+2,0:mx+2),fctrl(0:mx+2)
cBH_YP090809  First dimension of choose should be larger of 2*mx,mx+2
      pointer :: fctrl(:)     !!!(0:2*mx+2)
      pointer :: choose(:,:)  !!!(0:2*mx+2,0:mx+2)
      pointer :: cog(:,:)     !!!(0:mx,15)
      pointer :: pm(:,:)      !!!(0:mxp1,lrors)
      
      common /dptr95/ ix1,ix2,ix3,ix4,ix5,ix6,ix7,ix8,
     1                tom1,tom2,tom3,tom4, 
     1                fctrl,choose,cog,pm

      pointer f
      dimension f(:,:,:,:)  !f(0:iy+1,0:jx+1,ngen,lrors)
      common /dptr95/ f
      pointer favg
      dimension favg(:,:,:,:)
      common /dptr95/ favg
      pointer fxsp
      dimension fxsp(:,:,:,:)
      common /dptr95/ fxsp
      pointer f_
      dimension f_(:,:,:,:)
      common /dptr95/ f_
      pointer spasou
      dimension spasou(:,:,:,:)
      common /dptr95/ spasou
      pointer velsou
      dimension velsou(:,:,:,:)
      common /dptr95/ velsou
      pointer velsou2
      dimension velsou2(:,:,:,:)
      common /dptr95/ velsou2
      pointer source
      dimension source(:,:,:,:)
      common /dptr95/ source
      pointer gone
      dimension gone(:,:,:,:)
      common /dptr95/ gone
      pointer egylosa
      dimension egylosa(:,:,:,:)
      common /dptr95/ egylosa
!      pointer i0tran !YuP[2021-04] moved to subr.sourceko [used locally only]
!      dimension i0tran(:,:,:)
!      common /dptr95/ i0tran
      pointer cal
      dimension cal(:,:,:,:)
      common /dptr95/ cal
      pointer cbl
      dimension cbl(:,:,:,:)
      common /dptr95/ cbl
      pointer ccl
      dimension ccl(:,:,:,:)
      common /dptr95/ ccl
      pointer cdl
      dimension cdl(:,:,:,:)
      common /dptr95/ cdl
      pointer cel
      dimension cel(:,:,:,:)
      common /dptr95/ cel
      pointer cfl
      dimension cfl(:,:,:,:)
      common /dptr95/ cfl
      pointer eal
      dimension eal(:,:,:,:,:)
      common /dptr95/ eal
      pointer ebl
      dimension ebl(:,:,:,:,:)
      common /dptr95/ ebl

      !YuP[08-2017][2020-07-02] Added, for usage in tdchief, cfpcoefn
      !character*8 cfp_integrals ! "enabled" or "disabled"(by default)
      pointer temp_min(:), temp_max(:) ! (nmaxw_sp) 
      ! min/max of temp() for each Maxw.species, 
      ! to set a uniform T-grid within [temp_min; temp_max] range;
      ! Different for each Maxwellian species!
      pointer cfpm(:,:,:,:) !(3,nmaxw_sp,ntemp,jx) 
      ! cfpm() == storage of three integrals, 
      !           for each Maxw.species (1:nmaxw_sp),
      !           set as a table over 1:ntemp uniform T-grid,
      !           saved as a function of j index (func. of momentum). 
      ! Set ntemp=100 ! 100 points is sufficient to represent 
      ! all possible values of temp(kbm,lr) profile;
      ! so typically 3*6*100*200=360000 data points.
      common /dptr95_cfp/ nmaxw_sp,ntemp,temp_min,temp_max,cfpm
            
	  
      pointer scal
      dimension scal(:,:)
      common /dptr95/ scal
      pointer cet
      dimension cet(:,:,:)
      common /dptr95/ cet
      pointer cex
      dimension cex(:,:,:)
      common /dptr95/ cex
      pointer synca
      dimension synca(:,:,:)
      common /dptr95/ synca
      pointer syncd
      dimension syncd(:,:,:)
      common /dptr95/ syncd
      pointer taulos
      dimension taulos(:,:,:)
      common /dptr95/ taulos
      pointer elecfldn
      dimension elecfldn(:,:,:)
      common /dptr95/ elecfldn
      pointer delecfld0n
      dimension delecfld0n(:,:,:)
      common /dptr95/ delecfld0n
      pointer elecn
      dimension elecn(:,:,:)
      common /dptr95/ elecn
      pointer delecfld0
      dimension delecfld0(:,:)
      common /dptr95/ delecfld0
      pointer psi0bar
      dimension psi0bar(:,:)
      common /dptr95/ psi0bar
      pointer di
      dimension di(:,:,:,:)
      common /dptr95/ di
      pointer dj
      dimension dj(:,:,:,:)
      common /dptr95/ dj
      pointer dym5
      dimension dym5(:,:)
      common /dptr95/ dym5
      pointer dyp5
      dimension dyp5(:,:)
      common /dptr95/ dyp5
      pointer eyp5
      dimension eyp5(:,:)
      common /dptr95/ eyp5
      pointer eym5
      dimension eym5(:,:)
      common /dptr95/ eym5
      pointer y
      dimension y(:,:)
      common /dptr95/ y
      pointer dy,dyi
      dimension dy(:,:),dyi(:,:)
      common /dptr95/ dy,dyi
      pointer yptb
      dimension yptb(:,:)
      common /dptr95/ yptb
      pointer coss
      dimension coss(:,:)
      common /dptr95/ coss
      pointer cynt2
      dimension cynt2(:,:)
      common /dptr95/ cynt2
      pointer batot
      dimension batot(:,:)
      common /dptr95/ batot
      pointer lmax
      dimension lmax(:,:)
      common /dptr95/ lmax
      pointer vpint
      dimension vpint(:,:)
      common /dptr95/ vpint
      pointer psiiv
      dimension psiiv(:,:)
      common /dptr95/ psiiv
      pointer psiba
      dimension psiba(:,:)
      common /dptr95/ psiba
      pointer psisq
      dimension psisq(:,:)
      common /dptr95/ psisq
      pointer psicu
      dimension psicu(:,:)
      common /dptr95/ psicu
      pointer psiqu
      dimension psiqu(:,:)
      common /dptr95/ psiqu
      pointer bavpd
      dimension bavpd(:,:)
      common /dptr95/ bavpd
      pointer bavdn
      dimension bavdn(:,:)
      common /dptr95/ bavdn
      pointer psiir
      dimension psiir(:,:)
      common /dptr95/ psiir
      pointer vderb
      dimension vderb(:,:)
      common /dptr95/ vderb
      pointer sinn
      dimension sinn(:,:)
      common /dptr95/ sinn
      pointer tann
      dimension tann(:,:)
      common /dptr95/ tann
      pointer ymid
      dimension ymid(:,:)
      common /dptr95/ ymid
      pointer tau
      dimension tau(:,:)
      common /dptr95/ tau
      pointer vptb
      dimension vptb(:,:)
      common /dptr95/ vptb
      pointer zboun
      dimension zboun(:,:)
      common /dptr95/ zboun
      pointer idx
      dimension idx(:,:)
      common /dptr95/ idx
      pointer imax
      dimension imax(:,:)
      common /dptr95/ imax
      pointer dz
      dimension dz(:,:)
      common /dptr95/ dz
      pointer pol
      dimension pol(:,:)
      common /dptr95/ pol
      pointer solrz
      dimension solrz(:,:)
      common /dptr95/ solrz
      pointer solzz
      dimension solzz(:,:)
      common /dptr95/ solzz
      pointer bpolz, btorz ! Equil.field at pol.angle grid (lz)
      dimension bpolz(:,:), btorz(:,:)  ! (lz,lrzmax)
      common /dptr95/ bpolz, btorz
      
c YuP: [added Apr/2014] area and volume of a cell associated with each
c                     (R,Z) point on flux surface, (R,Z)==(solrz,solzz)
      pointer ddarea, ddvol
      dimension ddarea(:,:), ddvol(:,:)  !  (lz,lrzmax)
      common /dptr95/ ddarea, ddvol
   
      pointer thtab
      dimension thtab(:,:)
      common /dptr95/ thtab
      pointer z
      dimension z(:,:)
      common /dptr95/ z
      pointer zmid
      dimension zmid(:,:)
      common /dptr95/ zmid
      pointer bbpsi
      dimension bbpsi(:,:)
      common /dptr95/ bbpsi
      pointer consnp
      dimension consnp(:,:)
      common /dptr95/ consnp
      pointer ptime
      dimension ptime(:,:)
      common /dptr95/ ptime
      pointer sptzrp
      dimension sptzrp(:,:)
      common /dptr95/ sptzrp
      pointer pefld
      dimension pefld(:,:)
      common /dptr95/ pefld
      pointer rovsp
      dimension rovsp(:,:)
      common /dptr95/ rovsp
      pointer restp
      dimension restp(:,:)
      common /dptr95/ restp
      pointer restnp
      dimension restnp(:,:)
      common /dptr95/ restnp
      pointer vpov
      dimension vpov(:,:)
      common /dptr95/ vpov
      pointer es
      dimension es(:,:)
      common /dptr95/ es
      pointer bpsi
      dimension bpsi(:,:)
      common /dptr95/ bpsi
      pointer d2bpsi
      dimension d2bpsi(:,:)
      common /dptr95/ d2bpsi
      pointer d2solrz
      dimension d2solrz(:,:)
      common /dptr95/ d2solrz
      pointer d2solzz
      dimension d2solzz(:,:)
      common /dptr95/ d2solzz
      pointer d2bpolz, d2btorz
      dimension d2bpolz(:,:), d2btorz(:,:)
      common /dptr95/ d2bpolz, d2btorz
      pointer d2thtpol
      dimension d2thtpol(:,:)
      common /dptr95/ d2thtpol
      pointer d2es
      dimension d2es(:,:)
      common /dptr95/ d2es
      pointer thtpol
      dimension thtpol(:,:)
      common /dptr95/ thtpol
      pointer esfi
      dimension esfi(:,:)
      common /dptr95/ esfi
      pointer psiesfi
      dimension psiesfi(:,:)
      common /dptr95/ psiesfi
      pointer psifi
      dimension psifi(:,:)
      common /dptr95/ psifi
      pointer espsifi
      dimension espsifi(:,:)
      common /dptr95/ espsifi
      pointer soupp
      dimension soupp(:,:)
      common /dptr95/ soupp
      pointer waa
      dimension waa(:,:,:)
      common /dptr95/ waa
      pointer wbb
      dimension wbb(:,:,:)
      common /dptr95/ wbb
      pointer cosz
      dimension cosz(:,:,:)
      common /dptr95/ cosz
      pointer dtau
      dimension dtau(:,:,:)
      common /dptr95/ dtau
      pointer sinz
      dimension sinz(:,:,:)
      common /dptr95/ sinz
      pointer tanz
      dimension tanz(:,:,:)
      common /dptr95/ tanz
      pointer yz
      dimension yz(:,:,:)
      common /dptr95/ yz
      pointer tot
      dimension tot(:,:,:)
      common /dptr95/ tot
      pointer vflux
      dimension vflux(:,:,:)
      common /dptr95/ vflux
      pointer f_aveth
      dimension f_aveth(:,:,:,:)
      common /dptr95/ f_aveth
      pointer sincosba
      dimension sincosba(:,:,:)
      common /dptr95/ sincosba
      pointer densz
      dimension densz(:,:,:,:)
      common /dptr95/ densz
      pointer ss
      dimension ss(:,:,:,:)
      common /dptr95/ ss
      pointer dcofleg
      dimension dcofleg(:,:,:,:)
      common /dptr95/ dcofleg
      pointer dpcosz
      dimension dpcosz(:,:,:,:)
      common /dptr95/ dpcosz
      pointer ssy
      dimension ssy(:,:,:,:)
      common /dptr95/ ssy
      pointer ssyy
      dimension ssyy(:,:,:,:)
      common /dptr95/ ssyy
      pointer ssyi
      dimension ssyi(:,:,:,:)
      common /dptr95/ ssyi
      pointer ssyyy
      dimension ssyyy(:,:,:,:)
      common /dptr95/ ssyyy
      pointer pcurr, pcurrm
      dimension pcurr(:,:,:), pcurrm(:,:,:)
      common /dptr95/ pcurr, pcurrm
      pointer pdens, pdenm
      dimension pdens(:,:,:), pdenm(:,:,:)
      common /dptr95/ pdens, pdenm
      pointer pengy, pengym
      dimension pengy(:,:,:), pengym(:,:,:)
      common /dptr95/ pengy, pengym
      pointer pdenra
      dimension pdenra(:,:)
      common /dptr95/ pdenra
      pointer pcurra
      dimension pcurra(:,:)
      common /dptr95/ pcurra
      pointer pfdenra
      dimension pfdenra(:,:)
      common /dptr95/ pfdenra
      pointer pfcurra
      dimension pfcurra(:,:)
      common /dptr95/ pfcurra
      pointer pucrit
      dimension pucrit(:,:)
      common /dptr95/ pucrit
      pointer peoe0
      dimension peoe0(:,:)
      common /dptr95/ peoe0
      pointer psrc
      dimension psrc(:,:)
      common /dptr95/ psrc
      pointer peoed
      dimension peoed(:,:)
      common /dptr95/ peoed
      pointer cint2
      dimension cint2(:)
      common /dptr95/ cint2
      pointer dx,dxi
      dimension dx(:),dxi(:)
      common /dptr95/ dx,dxi
      pointer ifp
      dimension ifp(:)
      common /dptr95/ ifp
      pointer sg
      dimension sg(:)
      common /dptr95/ sg
      pointer sgx
      dimension sgx(:)
      common /dptr95/ sgx
      pointer sgxx
      dimension sgxx(:)
      common /dptr95/ sgxx
      pointer sh
      dimension sh(:)
      common /dptr95/ sh
      pointer shx
      dimension shx(:)
      common /dptr95/ shx
      pointer shxx
      dimension shxx(:)
      common /dptr95/ shxx
      pointer shxxx
      dimension shxxx(:)
      common /dptr95/ shxxx
      pointer tam1
      dimension tam1(:)
      common /dptr95/ tam1
      pointer tam2
      dimension tam2(:)
      common /dptr95/ tam2
      pointer tam3
      dimension tam3(:)
      common /dptr95/ tam3
      pointer tam4
      dimension tam4(:)
      common /dptr95/ tam4
      pointer tam5
      dimension tam5(:)
      common /dptr95/ tam5
      pointer tam6
      dimension tam6(:)
      common /dptr95/ tam6
      pointer tam7
      dimension tam7(:)
      common /dptr95/ tam7
      pointer tam8
      dimension tam8(:)
      common /dptr95/ tam8
      pointer tam9
      dimension tam9(:)
      common /dptr95/ tam9
      pointer tam10
      dimension tam10(:)
      common /dptr95/ tam10
      pointer tam11
      dimension tam11(:)
      common /dptr95/ tam11
      pointer tam12
      dimension tam12(:)
      common /dptr95/ tam12
      pointer tam13
      dimension tam13(:)
      common /dptr95/ tam13
      pointer tam14
      dimension tam14(:)
      common /dptr95/ tam14
      pointer tam15
      dimension tam15(:)
      common /dptr95/ tam15
      pointer tam16
      dimension tam16(:)
      common /dptr95/ tam16
      pointer tam17
      dimension tam17(:)
      common /dptr95/ tam17
      pointer tam18
      dimension tam18(:)
      common /dptr95/ tam18
      pointer tam19
      dimension tam19(:)
      common /dptr95/ tam19
      pointer tam20
      dimension tam20(:)
      common /dptr95/ tam20
      pointer tam21
      dimension tam21(:)
      common /dptr95/ tam21
      pointer tam22
      dimension tam22(:)
      common /dptr95/ tam22
      pointer tam23
      dimension tam23(:)
      common /dptr95/ tam23
      pointer tam24
      dimension tam24(:)
      common /dptr95/ tam24
      pointer tam25
      dimension tam25(:)
      common /dptr95/ tam25
      pointer tam26
      dimension tam26(:)
      common /dptr95/ tam26
      pointer tam27
      dimension tam27(:)
      common /dptr95/ tam27
      pointer tam28
      dimension tam28(:)
      common /dptr95/ tam28
      pointer tam29
      dimension tam29(:)
      common /dptr95/ tam29
      pointer tam30
      dimension tam30(:)
      common /dptr95/ tam30
      pointer x
      dimension x(:)
      common /dptr95/ x
      pointer xmidpt
      dimension xmidpt(:)
      common /dptr95/ xmidpt
      pointer xi
      dimension xi(:)
      common /dptr95/ xi
      pointer xsq
      dimension xsq(:)
      common /dptr95/ xsq
      pointer x3i
      dimension x3i(:)
      common /dptr95/ x3i
      pointer x2i
      dimension x2i(:)
      common /dptr95/ x2i
      pointer xcu
      dimension xcu(:)
      common /dptr95/ xcu
      pointer xcenter
      dimension xcenter(:)
      common /dptr95/ xcenter
      pointer xcensq, xcent3
      dimension xcensq(:), xcent3(:)
      common /dptr95/ xcensq, xcent3
      pointer uoc
      dimension uoc(:)
      common /dptr95/ uoc
      pointer enerkev
      dimension enerkev(:,:) !YuP[2018-01-08] added 2nd index (k)
      common /dptr95/ enerkev
      pointer gamma
      dimension gamma(:)
      common /dptr95/ gamma
      pointer gamsqr
      dimension gamsqr(:)
      common /dptr95/ gamsqr
      pointer gamcub
      dimension gamcub(:)
      common /dptr95/ gamcub
      pointer gammi
      dimension gammi(:)
      common /dptr95/ gammi
      pointer gamm2i
      dimension gamm2i(:)
      common /dptr95/ gamm2i
      pointer gamm1
      dimension gamm1(:)
      common /dptr95/ gamm1
      pointer tcsgm1
      dimension tcsgm1(:)
      common /dptr95/ tcsgm1
      pointer gamefac
      dimension gamefac(:,:) !YuP[2019-07-26] k index added, so now gamefac(jx,ntotal)
      common /dptr95/ gamefac
      integer, dimension(:), pointer :: ident
      common /dptr95/ ident
      pointer temc1
      dimension temc1(:)
      common /dptr95/ temc1
      pointer temc2
      dimension temc2(:)
      common /dptr95/ temc2
      pointer temc3
      dimension temc3(:)
      common /dptr95/ temc3
      pointer temc4
      dimension temc4(:)
      common /dptr95/ temc4
      integer, dimension(:), pointer :: itemc1
      common /dptr95/ itemc1
      integer, dimension(:), pointer :: itemc2
      common /dptr95/ itemc2
      integer, dimension(:), pointer :: l_lower
      common /dptr95/ l_lower
      integer, dimension(:), pointer :: lpt
      common /dptr95/ lpt
      pointer mun !real*8
      dimension mun(:)
      common /dptr95/ mun
      pointer fll
      dimension fll(:)
      common /dptr95/ fll
      pointer xpar
      dimension xpar(:)
      common /dptr95/ xpar
      pointer rheads
      dimension rheads(:)
      common /dptr95/ rheads
      pointer dfvlle
      dimension dfvlle(:)
      common /dptr95/ dfvlle
      pointer dfvlli
      dimension dfvlli(:)
      common /dptr95/ dfvlli
      pointer xperp
      dimension xperp(:)
      common /dptr95/ xperp
      pointer xl
      dimension xl(:)
      common /dptr95/ xl
      integer, dimension(:), pointer :: jmaxxl
      common /dptr95/ jmaxxl
      pointer xlm
      dimension xlm(:)
      common /dptr95/ xlm
      pointer dxl
      dimension dxl(:)
      common /dptr95/ dxl
      pointer fl
      dimension fl(:,:) ![2022-03-19] Added 2nd dim: 1:lrors
      common /dptr95/ fl
      pointer fl1
      dimension fl1(:)
      common /dptr95/ fl1
      pointer fl2
      dimension fl2(:)
      common /dptr95/ fl2
      pointer ppars
      dimension ppars(:,:)
      common /dptr95/ ppars
      pointer pprps
      dimension pprps(:,:)
      common /dptr95/ pprps
      pointer faci
      dimension faci(:,:)
      common /dptr95/ faci
      pointer pparea
      dimension pparea(:,:)
      common /dptr95/ pparea
      pointer wtfl0
      dimension wtfl0(:,:,:)
      common /dptr95/ wtfl0
      pointer wtflm
      dimension wtflm(:,:,:)
      common /dptr95/ wtflm
      pointer jflbin
      dimension jflbin(:,:,:)
      common /dptr95/ jflbin
      pointer xm
      dimension xm(:,:)
      common /dptr95/ xm
      pointer dbb
      dimension dbb(:,:)
      common /dptr95/ dbb
      pointer dd
      dimension dd(:,:)
      common /dptr95/ dd
      pointer de
      dimension de(:,:)
      common /dptr95/ de
      pointer df
      dimension df(:,:)
      common /dptr95/ df
      pointer dff
      dimension dff(:,:)
      common /dptr95/ dff
      pointer cah
      dimension cah(:,:)
      common /dptr95/ cah
      pointer cthta
      dimension cthta(:,:)
      common /dptr95/ cthta
      pointer gon
      dimension gon(:,:)
      common /dptr95/ gon
      pointer so
      dimension so(:,:)
      common /dptr95/ so
      pointer currv
      dimension currv(:,:,:)
      common /dptr95/ currv
      pointer currvs
      dimension currvs(:,:)
      common /dptr95/ currvs
      pointer pwrrf
      dimension pwrrf(:,:,:)
      common /dptr95/ pwrrf
      pointer tal
      dimension tal(:,:)
      common /dptr95/ tal
      pointer tbl
      dimension tbl(:,:)
      common /dptr95/ tbl
      pointer tfl
      dimension tfl(:,:)
      common /dptr95/ tfl
      pointer pwrrfs
      dimension pwrrfs(:,:,:)
      common /dptr95/ pwrrfs
      pointer pleg
      dimension pleg(:,:)
      common /dptr95/ pleg
      pointer feta
      dimension feta(:,:)
      common /dptr95/ feta
      pointer fetb
      dimension fetb(:,:)
      common /dptr95/ fetb
      pointer wflux
      dimension wflux(:,:,:)
      common /dptr95/ wflux
c     NB:  rhs set up here for full 3d set of eqns (BH070525)
      pointer rhs
      dimension rhs(:)
      common /dptr95/ rhs
      pointer sovt
      dimension sovt(:,:,:,:)
      common /dptr95/ sovt
      pointer sigsxr
      dimension sigsxr(:,:,:,:)
      common /dptr95/ sigsxr

      pointer pentr
      dimension pentr(:,:,:,:)  !!!(nonch,ngen,-1:15,lrors)
      common /dptr95/ pentr

      pointer constp
      dimension constp(:,:)  !!!(nonch,lrors)
      common /dptr95/ constp
      
      pointer :: sigmtt(:,:),sigftt(:,:)  !!!(nonch,4)
      common /dptr95/ sigmtt,sigftt
      
      pointer sgaint
      dimension sgaint(:,:,:)  !!!(8,ngen,lrors)
      pointer entr
      dimension entr(:,:,:)    !!!(ngen,-1:15,lrors)
      pointer xlndnz
      dimension xlndnz(:,:)    !!!(ngen+1,negyrg)
      pointer sounor
      dimension sounor(:,:,:,:)   !!!(ngen,nso,lz,lrz)
      common /dptr95/ sgaint,entr,xlndnz,sounor

      
c.......................................................................
c*****arrays related to relativ=fully option
c.......................................................................
      pointer gamman
      dimension gamman(:,:)
      common /dptr95/ gamman
      pointer alphan
      dimension alphan(:,:)
      common /dptr95/ alphan

      pointer asnha
      dimension asnha(:)
      common /dptr95/ asnha
      pointer item1
      dimension item1(:)
      common /dptr95/ item1
      pointer item2
      dimension item2(:)
      common /dptr95/ item2
      pointer item3
      dimension item3(:)
      common /dptr95/ item3
      pointer item4
      dimension item4(:)
      common /dptr95/ item4
      pointer item5
      dimension item5(:)
      common /dptr95/ item5
      pointer item6
      dimension item6(:)
      common /dptr95/ item6
      pointer dxm5
      dimension dxm5(:)
      common /dptr95/ dxm5
      pointer exm5
      dimension exm5(:)
      common /dptr95/ exm5
      pointer dxp5
      dimension dxp5(:)
      common /dptr95/ dxp5
      pointer exp5
      dimension exp5(:)
      common /dptr95/ exp5
      pointer tamt1
      dimension tamt1(:,:,:,:)
      common /dptr95/ tamt1
      pointer tamt2
      dimension tamt2(:,:,:,:)
      common /dptr95/ tamt2
      pointer da
      dimension da(:,:)
      common /dptr95/ da
      pointer db
      dimension db(:,:)
      common /dptr95/ db
      pointer dc
      dimension dc(:,:)
      common /dptr95/ dc
      pointer ca
      dimension ca(:,:)
      common /dptr95/ ca
      pointer cb
      dimension cb(:,:)
      common /dptr95/ cb
      pointer cc
      dimension cc(:,:)
      common /dptr95/ cc
      pointer cd
      dimension cd(:,:)
      common /dptr95/ cd
      pointer ce
      dimension ce(:,:)
      common /dptr95/ ce
      pointer cf
      dimension cf(:,:)
      common /dptr95/ cf

      pointer tem1
      dimension tem1(:)
      common /dptr95/ tem1
      pointer tem2
      dimension tem2(:)
      common /dptr95/ tem2
      pointer tem3
      dimension tem3(:)
      common /dptr95/ tem3
      pointer tem4
      dimension tem4(:)
      common /dptr95/ tem4
      pointer tem5
      dimension tem5(:)
      common /dptr95/ tem5
      pointer tem6
      dimension tem6(:)
      common /dptr95/ tem6

      pointer egg
      dimension egg(:,:)
      common /dptr95/ egg
      pointer fgg
      dimension fgg(:,:)
      common /dptr95/ fgg

      pointer xhead
      dimension xhead(:,:)
      common /dptr95/ xhead
      pointer xtail
      dimension xtail(:,:)
      common /dptr95/ xtail
      pointer ytail
      dimension ytail(:,:)
      common /dptr95/ ytail
      pointer yhead
      dimension yhead(:,:)
      common /dptr95/ yhead

      pointer fpn
      dimension fpn(:,:)
      common /dptr95/ fpn

      pointer temp1
      dimension temp1(:,:)
      common /dptr95/ temp1
      pointer temp2
      dimension temp2(:,:)
      common /dptr95/ temp2
      pointer temp3
      dimension temp3(:,:)
      common /dptr95/ temp3
      pointer temp4
      dimension temp4(:,:)
      common /dptr95/ temp4
      pointer temp5
      dimension temp5(:,:)
      common /dptr95/ temp5
      pointer temp6
      dimension temp6(:,:)
      common /dptr95/ temp6

      pointer xllji
      dimension xllji(:,:)
      common /dptr95/ xllji
      pointer xppji
      dimension xppji(:,:)
      common /dptr95/ xppji

c     Arrays used for first order orbit width calculations:
      pointer deltarho
      dimension deltarho(:,:,:)
      common /dptr95/ deltarho
      pointer deltarhop
      dimension deltarhop(:,:,:)
      common /dptr95/ deltarhop
      pointer deltarz
      dimension deltarz(:,:,:)
      common /dptr95/ deltarz
      pointer r_delta
      dimension r_delta(:)
      common /dptr95/ r_delta
      pointer z_delta
      dimension z_delta(:)
      common /dptr95/ z_delta
      pointer t_delta
      dimension t_delta(:)
      common /dptr95/ t_delta
      pointer delta_bdb0
      dimension delta_bdb0(:,:)
      common /dptr95/ delta_bdb0


c*****************************************************************
c     BEGIN arrays for analytic ion source (sou..) routines
c*****************************************************************

      common /diskx/
     1  bdre(lrza),bdrep(lrza),
     1  sorpwt(lrza),sorpwti(0:lrza),
     1  sorpw_nbii(1:ngena,0:lrza),sorpw_rfi(1:ngena,0:lrza),
     1  xlncurt(lrorsa)  !YuP[2022-02-11] now lrorsa (was lrza)

      common /diskx/
     1  sorpw_rf(1:ngena,lrza),sorpw_nbi(1:ngena,lrza)

      common /diskx/
     1  cosm1(ngena,nsoa,lrza),cosm2(ngena,nsoa,lrza),
     1  sxllm1(ngena,nsoa,lrza),sxllm2(ngena,nsoa,lrza),
     1  sxppm1(ngena,nsoa,lrza),
     1  sxppm2(ngena,nsoa,lrza),
     1  xem1(ngena,nsoa,lrza),xem2(ngena,nsoa,lrza),
     1  zm1(ngena,nsoa,lrza),zm2(ngena,nsoa,lrza)


      common /scalar/
     1  isounor



c*****************************************************************
c     BEGIN arrays for rf package..(rf...,vlh[B,...,vlf...) routines
c*****************************************************************

      pointer cqlb,cqlc,cqle,cqlf     ! (iy,jx,lrz,mrfn)
      dimension cqlb(:,:,:,:),cqlc(:,:,:,:),cqle(:,:,:,:),cqlf(:,:,:,:)
      common/qlcoef/cqlb,cqlc,cqle,cqlf 

      pointer bqlm
      dimension bqlm(:,:)  ! (iy,jx)
      common/qlcoef/ bqlm



c****************************************************************
c     BEGIN arrays for 3-d (td..) driver.
c****************************************************************


c..................................................................
c     scalars used in CQL3D
c..................................................................

      real*8 li
      common/sc3d/
     1  dttr,dtreff,currtza,currtpza,conserv,
     1  fom,fomp,fompla,fomtot,flxout,
     1  eden,edenlavg,etemp,ethtemp,edntmp,pden,pdntmp,psynct,
     1  li,
cBH110314cBH070408(Not used except in fr routines):    1  smooth,
cBH110314:  Restored, as needed in subroutine frsmooth, but needed
cBH110314:  to change name, as conflict arises in tdreadf/tdwritef
     1  smooth_,
     1  toteqd,cursign,totcurza,total,total0,totcurtt,curxjtot,
     1  ncount,iplt3d,ipacktp,n_d_rr

c..................................................................
c     all arrays used only in CQL3D
c..................................................................

      real*8 jparb,jparbt,jparbp,mun

      !YuP[2021-03] Rearranged, to collect lrorsa arrays together
      !(those are used in both CQL3D and CQLP, 
      !but with different meaning of l_ index)
      !and converted to pointers.
      !For future CQL4D development: need to "split" l_ index into two.
      integer, dimension(:), pointer :: iytr  !(lrors)
      real*8,  dimension(:), pointer :: currmtz,currmtpz,vfluxz !(lrors)
      common/ar3d_lrors/ 
     1  iytr,
     1  currmtz,
     1  currmtpz,
     1  vfluxz
      
      common/ar3d_lrza/ rrz(0:lrza),
     1  tr(0:lrza),tr1(0:lrza),tr2(0:lrza),
     1  tr3(0:lrza),tr4(0:lrza),tr5(0:lrza),drp5(0:lrza),
     1  dpsi(0:lrza),dpsidrho(lrza),
     1  h_r(0:lrza),
     1  area(0:lrza),equilpsp(0:lrza),equilpsi(0:lrza),
     1  areamid(0:lrza),volmid(0:lrza),
     1  psivalm(lrza),rpconz(lrza),rmconz(lrza),rpmconz(0:lrza),
     1  bpolsqaz(0:lrza),aspin(lrza),trapfrac(lrza),
     1  currz(ngena,lrza),currtpz(lrza),currtz(lrza),
     1  currza(ngena),currtzi(0:lrza),currtpzi(0:lrza),
     1  totcurzi(0:lrza),totcurz(lrza),
     1  fpsiz2(0:lrza),fpsiz(lrza),ffpsiz(lrza),
     1  jparb(lrza),jparbt(lrza),jparbp(lrza),
     1  prestp(lrza),prest(lrza),d2prest(lrza),d2fpsiz(lrza),
     1  d2ffpsiz(lrza),
     1  bmdplne(lrza),d2bmdpl(lrza),
     1  gkpwrz(ngena,lrza),
     1  rfpwrz(ngena,lrza),
     1  drrt(ngena),drt(ngena)
      common/ar3d/
     1  dvol(lrza),darea(lrza),
     1  psyncz(lrza),pegyz(ngena,lrza),pplossz(ngena,lrza),
     1  wparzt(ngena),wperpzt(ngena),pegyt(ngena),pplosst(ngena),
     1  rfpwrt(ngena),
     1  gkpwrt(ngena),energyt(ntotala),
     1  rz(0:lrza),
     1  vol(0:lrza),
     1  onovrpz(lrza,2),
     1  tplt3d(nplota),
     1  bscurma(2,2),bscurm(0:lrza,2,2),bscurmi(0:lrza,2,2)
     
      real*8 bscurm_n(lrza),reden_n(lrza),energy_n(lrza) !YuP[2019-12-18]
      real*8 curra_n(lrza) !YuP[2020-10-31]
      common/ar3d/bscurm_n,reden_n,energy_n,curra_n !YuP[2019-12-18] for save/restore,
                          !when making iterations involving jbs current.
      real*8 currpar_starnue(lrza),  sig_starnue(lrza) !YuP[2019-12-19]
      common/ar3d/currpar_starnue,sig_starnue
      real*8 currpar_starnue0(lrza), sig_starnue0(lrza) !YuP[2019-12-19]
      common/ar3d/currpar_starnue0,sig_starnue0
      real*8 currpar_starnue_n(lrza),  sig_starnue_n(lrza) !YuP[2019-12-19]
      common/ar3d/currpar_starnue_n,sig_starnue_n
      real*8 currpar_starnue0_n(lrza), sig_starnue0_n(lrza) !YuP[2019-12-19]
      common/ar3d/currpar_starnue0_n,sig_starnue0_n

      common/sc3d/
     1  sorpwtza

      common /csxr/ sxry(lrza,4),sang(lrza,4),spol(lrza,4),
     1  ibin(lrza,4),eflux(nena,nva),efluxt(nva),alphad(3),xs_(3),
     1  enk(nena),en_(nena),jval_(nena),
     1  inegsxr(nva),lensxr(nva)

      !YuP[2021-03] Rearranged, to collect lrorsa arrays together
      !(those are used in both CQL3D and CQLP, 
      !but with different meaning of l_ index)
      !and converted to pointers.
      !For future CQL4D development: need to "split" l_ index into two.
      real*8, dimension(:,:), pointer :: sigm,sigf,fuspwrv,fuspwrm  !(4,lrors)
      !YuP[2021-03-18] Need to check usage of these 4 arrays: change lr_ to l_ ?
      
      common /csigma/ mtab,msig,jxis,elmin,delegy,
     1  imaxwln(2,4),igenrl(2,4),
     1  sigmt(4),sigft(4),
     1  fuspwrvt(4),fuspwrmt(4)
      common /csigma_lrors/ sigm,sigf,fuspwrv,fuspwrm

      pointer tamm1
      dimension tamm1(:)
      common /csigma/ tamm1  !(0:mmsv)
      
      pointer iind
      dimension iind(:)
      common /csigma/ iind  !(1:jx)

c..............................................................
c     Set up pointers for sigma-v
c..............................................................

      pointer csv
      dimension csv(:,:,:)
      common /dptr95/ csv
      pointer svtab
      dimension svtab(:)
      common /dptr95/ svtab


c..............................................................
c     Set up pointers to allocatable arrays for transport model.
c     Space allocated in subroutine tdtraloc
c..............................................................


      pointer frn_2
      dimension frn_2(:,:,:,:)
      common /dptr95/ frn_2
      pointer frn_1
      dimension frn_1(:,:,:,:)
      common /dptr95/ frn_1
      pointer frn
      dimension frn(:,:,:,:)
      common /dptr95/ frn
      pointer fvn_1
      dimension fvn_1(:,:,:,:)
      common /dptr95/ fvn_1
      pointer fvn
      dimension fvn(:,:,:,:)
      common /dptr95/ fvn
      pointer dl
      dimension dl(:,:,:,:)
      common /dptr95/ dl
      pointer d_rr
      dimension d_rr(:,:,:,:)
      common /dptr95/ d_rr
      pointer d_r
      dimension d_r(:,:,:,:)
      common /dptr95/ d_r
      
      real*8, pointer :: ryain_t(:,:)  !(njene,nbctime)
      common /data_ryain/ ryain_t      !(njene,nbctime)
      real*8, pointer :: dbb2in_t(:,:) !(njene,nbctime)
      real*8, pointer :: dbb2(:) !lrz
      common /data_deltabb/ dbb2in_t   !(njene,nbctime)  !=(deltaB/B)^2 from data
      common /data_deltabb/ dbb2 !(lrz) !=(deltaB/B)^2 at each time step
      ![2021-08] Added, For NIMROD coupling 
      !(reading deltaB/B data and forming Drr(u,theta,r,t) coeff)
      !See subr. read_data_files()
      
      pointer f_lm
      dimension f_lm(:,:,:)
      common /dptr95/ f_lm
      pointer f_lp
      dimension f_lp(:,:,:)
      common /dptr95/ f_lp
      pointer f_up
      dimension f_up(:,:,:)
      common /dptr95/ f_up
      pointer f_vtor
      dimension f_vtor(:,:,:,:)
      common /dptr95/ f_vtor
      pointer cynt2_
      dimension cynt2_(:,:)
      common /dptr95/ cynt2_
      pointer vpint_
      dimension vpint_(:,:)
      common /dptr95/ vpint_
      pointer vptb_
      dimension vptb_(:,:)
      common /dptr95/ vptb_
      pointer cosovb
      dimension cosovb(:,:)
      common /dptr95/ cosovb
      pointer bovcos
      dimension bovcos(:,:)
      common /dptr95/ bovcos
      pointer adv
      dimension adv(:,:)
      common /dptr95/ adv
      pointer dentarget
      dimension dentarget(:)
      common /dptr95/ dentarget
      pointer eg_
      dimension eg_(:,:,:)
      common /dptr95/ eg_
      pointer fg_
      dimension fg_(:,:,:)
      common /dptr95/ fg_



c******************************************************************
c     BEGIN arrays for EQUILIBRIUM MODEL (eq..) (NON-CIRCULAR CROSS
c     SECTIONS).
c******************************************************************


      common /diskx/
     1  areacon(lrza),
     1  bmidplne(lrza),bpolsqa(lrza),
     1  eqdells(lrza),epsicon(lrza),erhocon(lrza),
     1  fpsi(lrza),flxavgd(lrza),
     1  psiovr(lrza),psiavg(2,lrza),onovrp(2,lrza),onovpsir3(lrza),
     1  rpcon(lrza),rmcon(lrza),zpcon(lrza),zmcon(lrza),
     1  volcon(lrza),fppsi(lrza),pppsi(lrza),es_bmax(lrza),
     1  bpsi_max(lrza),bpsi_min(lrza),lbpsi_max(lrza),lbpsi_min(lrza),
     1  z_bmax(lrza),bpsi_z_bmax(lrza),lz_bmax(lrza),
     1  dlpgpsii(lrza),dlpsii(lrza)

      common/params/ nnz,nnr,nj12


      character*8
     1  eqorb,eqcall

      common
     1  bpolsqlm,
     1  eqorb,eqcall,
     1  imag,jmag,
     1  nmag,nrc,nzc,nfp,nnv,iupdn,
     1  psimag,psilim,
     1  rmaxcon,rmincon,rhomax,
     1  zmag,zmaxcon,zmincon,zshift

      common
     1  ibd(4),
     1  eqpsi(nconteqa),eqvol(nconteqa),eqfopsi(nconteqa),
     1  q_(nconteqa),eqrho(nconteqa),d2eqrho(nconteqa),eqarea(nconteqa),
     1  eqrpcon(nconteqa),eqrmcon(nconteqa),
     1  eqzpcon(nconteqa),eqzmcon(nconteqa),
     1  d2fpsiar(nnra),
     1  ez(nnza),dummyaz(nnza),
     1  fpsiar(nnra),ffpar(nnra),d2ffpar(nnra),qar(nnra),d2qar(nnra),
     1  prar(nnra),d2prar(nnra),ppar(nnra),d2ppar(nnra),psiar(nnra),
     1  er(nnra),dummyar(nnra),
     1  wkepsi(nrz3p1a)

      common
     1  epsi(nnra,nnza),epsirr(nnra,nnza),
     1  epsizz(nnra,nnza),epsirz(nnra,nnza),
     1  dummypsi(nnra,nnza),eqovrp(nconteqa,2)

      common/output/ lorbit_,ialign14,rmcon_,rpcon_,zmcon_,zpcon_,
     1  bthr_,btoru_,eqdells_,fpsi_,fppsi_,zmax_,btor0_,bthr0_,
     1  es_bmax_,bpsi_max_,bpsi_min_,lbpsi_max_,lbpsi_min_,
     1  bmidplne_
     
      !YuP[2021-04] collected arrays related to field line tracing,
      !made them into pointers:
      real*8,dimension(:), pointer :: solr_,solz_,es_,eqbpol_,bpsi_,
     1   thtpol_,eqdell_,tlorb1,tlorb2 ! (lfield)
      real*8,dimension(:), pointer :: work !(was tlfld1a, now 3*lfield+1)
      common/eq_line/ solr_,solz_,es_,
     1  eqbpol_,bpsi_,thtpol_,
     1  eqdell_,
     1  tlorb1,tlorb2,
     1  work !used in micxiniz, eqorbit, as input in coeff1()
!Careful - there are other 'work' arrays defined locally in other subroutines

c..................................................................
c     Allocatable arrays allocated in subroutine eqalloc
c..................................................................

      pointer drpmconz
      dimension drpmconz(:)
      common /dptr95/ drpmconz
      pointer eqdell
      dimension eqdell(:,:)
      common /dptr95/ eqdell
      pointer eqbpol
      dimension eqbpol(:,:)
      common /dptr95/ eqbpol
      pointer solr
      dimension solr(:,:)
      common /dptr95/ solr
      pointer solz
      dimension solz(:,:)
      common /dptr95/ solz



c*********************************************************************
c     BEGIN arrays for LOWER HYBRID FAST WAVE and ECH Module.
c*********************************************************************


      common/params/
     1  jjx

      real*8 jbm1,jb0,jbp1

      pointer jbm1
      dimension jbm1(:,:)
      common jbm1
      pointer jb0
      dimension jb0(:,:)
      common jb0
      pointer jbp1
      dimension jbp1(:,:)
      common jbp1

      character*8 irffile

      common
     1  argmax,
     1  nurf,
c     1  lenj0,
     1  mrf,mrfn, 
     1  nrayn,nrayelts, !-YuP 101122: added
     1  irftype,
     1  powray,
     1  vnorm2,vnorm3,vnorm4,
     1  dveps ! YuP[04-2016] for subr. urfb_add

      complex*16 cosz1,sinz1,sinz2
      common
     1  bsslstp(nmodsa),
     1  ncontrib(lrza),
     1  powrf(lrza,nmodsa),powrfc(lrza,nmodsa),
     1  powrfl(lrza,nmodsa),powrft(lrza),
     1  powurf(0:nmodsa),powurfc(0:nmodsa),powurfl(0:nmodsa),
     1  powurfi(0:lrza,0:nmodsa)

      common
     1  freqcy(nmodsa),omega(nmodsa),nharm(nmodsa),nray(nmodsa),
     1  bsign1(nmodsa),krfn(nmodsa),irfn(nmodsa),irfm(nmodsa),
     1  irffile(nmodsa)


c..................................................................
c     Allocatable arrays allocated in subroutine urfalloc
c..................................................................


      complex*16 cwexde,cweyde,cwezde

      pointer urfb
      dimension urfb(:,:,:,:)
      common /dptr95/ urfb
      pointer urfc
      dimension urfc(:,:,:,:)
      common /dptr95/ urfc
      pointer cosmz
      dimension cosmz(:,:,:)
      common /dptr95/ cosmz
      pointer g_
      dimension g_(:,:,:,:)
      common /dptr95/ g_
      pointer alfag
      dimension alfag(:)
      common /dptr95/ alfag
      pointer argmnt
      dimension argmnt(:)
      common /dptr95/ argmnt
      pointer ilim1d
      dimension ilim1d(:)
      common /dptr95/ ilim1d
      pointer ilim2d
      dimension ilim2d(:)
      common /dptr95/ ilim2d
      pointer ilim1dd
      dimension ilim1dd(:)
      common /dptr95/ ilim1dd
      pointer ilim2dd
      dimension ilim2dd(:)
      common /dptr95/ ilim2dd
      pointer sx
      dimension sx(:)
      common /dptr95/ sx
      pointer xmdx
      dimension xmdx(:)
      common /dptr95/ xmdx
      pointer cosz1
      dimension cosz1(:)
      common /dptr95/ cosz1
      pointer sinz1
      dimension sinz1(:)
      common /dptr95/ sinz1
      pointer sinz2
      dimension sinz2(:)
      common /dptr95/ sinz2
      pointer thtf1
      dimension thtf1(:)
      common /dptr95/ thtf1
      pointer thtf2
      dimension thtf2(:)
      common /dptr95/ thtf2
      pointer alfi
      dimension alfi(:)
      common /dptr95/ alfi
      pointer alfa
      dimension alfa(:)
      common /dptr95/ alfa
      integer, dimension(:), pointer :: ilim1
      common /dptr95/ ilim1
      integer, dimension(:), pointer :: ilim2
      common /dptr95/ ilim2
      integer, dimension(:), pointer :: ifct1
      common /dptr95/ ifct1
      integer, dimension(:), pointer :: ifct2
      common /dptr95/ ifct2
      pointer urftmp
      dimension urftmp(:)
      common /dptr95/ urftmp
      pointer urfpwr
      dimension urfpwr(:,:,:)
      common /dptr95/ urfpwr
      pointer urfpwrc
      dimension urfpwrc(:,:,:)
      common /dptr95/ urfpwrc
      pointer urfpwrl
      dimension urfpwrl(:,:,:)
      common /dptr95/ urfpwrl
      pointer jminray
      dimension jminray(:,:,:)
      common /dptr95/ jminray
      pointer jmaxray
      dimension jmaxray(:,:,:)
      common /dptr95/ jmaxray
      pointer lloc
      dimension lloc(:,:,:)
      common /dptr95/ lloc
      pointer llray
      dimension llray(:,:,:)
      common /dptr95/ llray
      pointer psiloc
      dimension psiloc(:,:,:)
      common /dptr95/ psiloc
      pointer scalurf
      dimension scalurf(:,:,:)
      common /dptr95/ scalurf
      pointer cwexde
      dimension cwexde(:,:,:)
      common /dptr95/ cwexde
      pointer cweyde
      dimension cweyde(:,:,:)
      common /dptr95/ cweyde
      pointer cwezde
      dimension cwezde(:,:,:)
      common /dptr95/ cwezde
      pointer delpwr
      dimension delpwr(:,:,:)
      common /dptr95/ delpwr
      pointer fluxn
      dimension fluxn(:,:,:)
      common /dptr95/ fluxn
      pointer seikon
      dimension seikon(:,:,:)
      common /dptr95/ seikon
      pointer spsi
      dimension spsi(:,:,:)
      common /dptr95/ spsi
      pointer sdpwr
      dimension sdpwr(:,:,:)
      common /dptr95/ sdpwr
      pointer sbtot
      dimension sbtot(:,:,:)
      common /dptr95/ sbtot
      pointer sene
      dimension sene(:,:,:)
      common /dptr95/ sene
      pointer salphac
      dimension salphac(:,:,:)
      common /dptr95/ salphac
      pointer salphal
      dimension salphal(:,:,:)
      common /dptr95/ salphal
      pointer ws
      dimension ws(:,:,:)
      common /dptr95/ ws
      pointer wr
      dimension wr(:,:,:)
      common /dptr95/ wr
      pointer wz
      dimension wz(:,:,:)
      common /dptr95/ wz
      pointer wnpar
      dimension wnpar(:,:,:)
      common /dptr95/ wnpar
      pointer wdnpar
      dimension wdnpar(:,:,:)
      common /dptr95/ wdnpar
      pointer wnper
      dimension wnper(:,:,:)
      common /dptr95/ wnper
      pointer wphi
      dimension wphi(:,:,:)
      common /dptr95/ wphi
      pointer ilowp
      dimension ilowp(:,:)
      common /dptr95/ ilowp
      pointer iupp
      dimension iupp(:,:)
      common /dptr95/ iupp
      pointer ifct1_
      dimension ifct1_(:,:)
      common /dptr95/ ifct1_
      pointer ifct2_
      dimension ifct2_(:,:)
      common /dptr95/ ifct2_
      pointer nrayelt
      dimension nrayelt(:,:)
      common /dptr95/ nrayelt
      pointer jslofas
      dimension jslofas(:,:)
      common /dptr95/ jslofas
      pointer nurefls
      dimension nurefls(:,:)
      common /dptr95/ nurefls
      pointer keiks
      dimension keiks(:,:)
      common /dptr95/ keiks
      pointer jpes
      dimension jpes(:,:)
      common /dptr95/ jpes
      pointer jpis
      dimension jpis(:,:)
      common /dptr95/ jpis
      pointer istarts
      dimension istarts(:,:)
      common /dptr95/ istarts
      pointer iprmt5
      dimension iprmt5(:,:)
      common /dptr95/ iprmt5
      pointer jhlfs
      dimension jhlfs(:,:)
      common /dptr95/ jhlfs
      pointer sxxrt
      dimension sxxrt(:,:)
      common /dptr95/ sxxrt
      pointer skpsi
      dimension skpsi(:,:)
      common /dptr95/ skpsi
      pointer skth
      dimension skth(:,:)
      common /dptr95/ skth
      pointer skphi
      dimension skphi(:,:)
      common /dptr95/ skphi
      pointer lrayelt
      dimension lrayelt(:,:)
      common /dptr95/ lrayelt
      pointer delpwr0
      dimension delpwr0(:,:)
      common /dptr95/ delpwr0
      pointer nrayelt0
      dimension nrayelt0(:,:)
      common /dptr95/ nrayelt0
      pointer truncd
      dimension truncd(:) ! 1:jx
      common /dptr95/ truncd


c..................................................................
c     Allocatable arrays allocated in subroutine rdc_multi,
c     used after subroutine execution.
c     Here, we introduce f90 pointers, as they are easier
c     to allocate.

      pointer rdcb
      dimension rdcb(:,:,:,:)
      common /dptr95/ rdcb
      pointer rdcc
      dimension rdcc(:,:,:,:)
      common /dptr95/ rdcc
      pointer rdce
      dimension rdce(:,:,:,:)
      common /dptr95/ rdce
      pointer rdcf
      dimension rdcf(:,:,:,:)
      common /dptr95/ rdcf


c..................................................................
c     Allocatable arrays allocated in subroutine it3dalloc
c     used after subroutine execution.
c..................................................................

      common /it3d/ lapacki,lapackj,icsrij,icsrip,icsri2,krylov,
     +              icsrikry,iwk_ilu
      common /it3d/ icsrijr,icsrijc

      pointer abd_lapack,a_csr,alu,w_ilu,rhs0,sol,vv
      pointer ja_csr,ia_csr,jlu,ju,jw_ilu
      pointer ar_csr,ac_csr
      pointer jar_csr,iar_csr,ipofi,jac_csr,iac_csr
      dimension abd_lapack(:,:),a_csr(:),
     +     alu(:),w_ilu(:),rhs0(:),sol(:),vv(:)
      dimension ja_csr(:),ia_csr(:),jlu(:),ju(:),jw_ilu(:)
      dimension ar_csr(:),ac_csr(:)
      dimension jar_csr(:),iar_csr(:),ipofi(:,:),jac_csr(:),iac_csr(:)
      common /dptr95/ abd_lapack,a_csr,alu,w_ilu,rhs0,sol,vv
      common /iptr95/ ja_csr,ia_csr,jlu,ju,jw_ilu
      common /dptr95/ ar_csr,ac_csr
      common /iptr95/ jar_csr,iar_csr,ipofi,jac_csr,iac_csr


c..................................................................

      common anecc(nrada),tekev(nrada),tikev(nint1a,nrada),
     1  anicc(nint1a,nrada),
     1  amass(nint1a),achrg(nint1a),elecf(nrada),
     1  rho_(nrada),names(10),psiar_(nrada),
     1  nspc,ialign9,fpsiar_(nrada),pary(nrada),ppary(nrada),
     1  gpary(nrada),ztop_,zbot_,rleft,rright,nx,nz,
     1  npsitm

c-----------------------------------------------------------------------
c     BEGIN variables for WP... modules for CQLP case
c-----------------------------------------------------------------------

      character*8  
     1  analegco

      common /wpscal/
     1  analegco,
     1  iymax,
     1  nsleft,nsrigt,numclas,numindx

      !YuP[2021-02-26] I think lsbptopr,...,sz,...,elparol,elparnw,flux1,flux2 
      !could be dimensioned over 0:ls+1 grid (FPE grid). 
      !See below, now in /wp_ls/
      !The other arrays in /wpvec/ are over full lsmax grid (can be larger than ls)
      common /wpvec/
!YuP     1  cofdfds(0:lsa1,2,ntrmdera),  !cofdfds is not used
     1  enrgypa(ntotala,0:lsa1),
     1  vthpar(ntotala,0:lsa1)
     
      !YuP: other arrays over full grid (over 1:lsmax or over 1:lrzmax):
      !  tauee,taueeh,starnue, zeff, zeff4, etc.
     
      !YuP[2021-03-18] Made these arrays into pointers 
      !(main reason: more chances to detect out-of-bounds errors)
      ! All these arrays are allocated with (0:ls+1) [ls is FPE grid]
      integer, dimension(:), pointer :: lsbtopr,lsprtob
      real*8,  dimension(:), pointer ::
     1  sz,dsz,dszm5,dszp5,
     1  eszm5,eszp5,
     1  psis,psisp,psipols,
     1  solrs,solzs,
     1  elparol,elparnw,
     1  flux1,flux2
      integer, dimension(:,:), pointer :: lpm1eff !(0:ls+1,-1:+1)
      common /wp_ls/ 
     1  lsbtopr,lsprtob,lpm1eff,
     1  sz,dsz,dszm5,dszp5,
     1  eszm5,eszp5,
     1  psis,psisp,psipols,
     1  solrs,solzs,
     1  elparol,elparnw,
     1  flux1,flux2
      
c.......................................................................
c     Arrays allocated in subroutine wpalloc for CQLP
c.......................................................................

      pointer l_upper
      dimension l_upper(:)  !!! (1:iy)
      pointer ilpm1ef
      dimension ilpm1ef(:,:,:)  !!! (0:iy+1,0:ls+1,-1:+1)
      common /dptr95_wp/ l_upper,ilpm1ef      !YuP[2021-03-05] was not added

      pointer fnhalf
      dimension fnhalf(:,:,:,:) !(0:iymax+1,0:jx+1,ngen,0:ls+1)
      common /dptr95_wp/ fnhalf
      pointer fnp0
      dimension fnp0(:,:,:,:)   !(0:iymax+1,0:jx+1,ngen,0:ls+1)
      common /dptr95_wp/ fnp0
      pointer fnp1
      dimension fnp1(:,:,:,:)   !(0:iymax+1,0:jx+1,ngen,0:ls+1)
      common /dptr95_wp/ fnp1
      pointer dls
      dimension dls(:,:,:,:)    !(0:iymax+1,0:jx+1,ngen,0:ls+1)
      common /dptr95_wp/ dls
      pointer fh
      dimension fh(:,:,:,:)     !(0:iymax+1,0:jx+1,ngen,0:ls+1) for CQLP
      common /dptr95/ fh ! fh is used in CQLP and A-F
      pointer fg
      dimension fg(:,:,:,:)     !(0:iymax+1,0:jx+1,ngen,0:ls+1) for CQLP
      common /dptr95/ fg ! fg is used in CQLP and A-F
      pointer fedge
      dimension fedge(:,:,:,:)
      common /dptr95_wp/ fedge
      pointer rhspar
      dimension rhspar(:,:,:)
      common /dptr95_wp/ rhspar
      pointer bndmats
      dimension bndmats(:,:,:,:)
      common /dptr95_wp/ bndmats
      pointer wcqlb
      dimension wcqlb(:,:,:,:)
      common /dptr95/ wcqlb
      pointer wcqlc
      dimension wcqlc(:,:,:,:)
      common /dptr95/ wcqlc
      pointer wcqle
      dimension wcqle(:,:,:,:)
      common /dptr95/ wcqle
      pointer wcqlf
      dimension wcqlf(:,:,:,:)
      common /dptr95/ wcqlf



c.......................................................................
c     Arrays allocated in ampfalloc
c.......................................................................
      
      pointer ampfln
      dimension ampfln(:)
      common /dptr95/ ampfln
      pointer ampflh
      dimension ampflh(:)
      common /dptr95/ ampflh
      pointer ampflg
      dimension ampflg(:)
      common /dptr95/ ampflg
      pointer ampfa
      dimension ampfa(:,:)
      common /dptr95/ ampfa
      pointer ampfb
      dimension ampfb(:,:)
      common /dptr95/ ampfb
      pointer ampfaa
      dimension ampfaa(:,:)
      common /dptr95/ ampfaa
      pointer ampfc
      dimension ampfc(:)
      common /dptr95/ ampfc
      pointer ampf2ebar
      dimension ampf2ebar(:)
      common /dptr95/ ampf2ebar


!--------------------------------------------------------------------
!YuP[2019-07-29] For gscreen(p) function (of normalized momentum p) 
!that describes the effect of partially screened ions 
!on enhanced scattering of electrons (fast electrons can "probe" 
!the inner structure of a partially ionized ion).
!See Hesslow et al, JPP-2018,vol.84, Eq.(2.25).
!Also, for hbethe(p) function that describes the slowing down
!of free electron on bound electrons in partially ionized ion
!or neutral atom.
!The arrays are allocated and set in subr. set_gscreen_hesslow.
! They are only needed when gamafac.eq."hesslow" .and. kelecg.eq.1.
! At present, it is set for one ion type (imp_type), 
! but it could be generalized in future.
      integer nstates ! save into comm.h
      real*8, dimension(:),  pointer :: a_imp       ! (0:nstates)
      real*8, dimension(:),  pointer :: bnumb_imp   ! (0:nstates) !Could be set as integer?
      real*8, dimension(:),  pointer :: excit_enrgy ! (0:nstates)
      real*8, dimension(:,:),pointer :: gscreen,hbethe   ! (0:nstates,1:jx)
      real*8, dimension(:),  pointer :: fz        ! (0:nstates)
      real*8, dimension(:,:),pointer :: temp_imp  ! (0:nstates,1:lrz)
      real*8, dimension(:,:),pointer :: dens_imp  ! (0:nstates,1:lrz)
      !real*8, dimension(:),  pointer :: dens_imp_allstates  ! (1:lrz)
      !real*8, dimension(:),  pointer :: dMpellet_dvol_sum  ! (1:lrz)
      common/impur/nstates,a_imp,bnumb_imp,excit_enrgy,fmass_imp
      common/impur/fz,dens_imp,temp_imp ! found by ADCDO or ADPAK
      common/hesslow_gscreen/gscreen,hbethe
      common/impurities/dens_imp_allstates(lrza),dMpellet_dvol_sum(lrza)
      !------------------
      !BH,YuP[2021-01-21] added a time-dependent variable 
      !to save data from data files.
      real*8, dimension(:,:,:),pointer :: dens_imp_t !(0:nstates,1:njene,1:nbctime)
      common/impur_read_data/dens_imp_t
      !Only needed when read_data='nimrod'.
      !Note that dens_imp_t() is not a namelist var, it is just a storage
      !for density of ionization states at time slices.
!--------------------------------------------------------------------
      
      
c.......................................................................
c     Arrays for finite orbit width (FOW) calculations
c.......................................................................
      
      common/psiaxis/ psi_lim,psi_mag,R_axis,Z_axis ![cgs]
      
      pointer rcontr,zcontr,rlimiter,zlimiter
      dimension rcontr(:),zcontr(:),rlimiter(:),zlimiter(:)
      common/limiter/ ncontr, nlimiter,
     +  rcontr, zcontr,    !-> Last closed flux surface
     +  rlimiter, zlimiter !-> Limiter surface [cm]
                           ! Setup by call equilib()

      common/eqbox/ ermin,ermax,ezmin,ezmax  ! Limits of equilibrium
                                             ! (R,Z)-grid (cm);
                                             ! Setup by call equilib()

c.......................................................................
c     variables transferred from freya
c.......................................................................
      character*8 frmodp, fr_gyrop, beamplsep, src_nbi_ep !YuP[2022-06-30]

      integer mfm1p, nbeamsp
      real*8 beamponp, beampoffp  
      real*8 hibrzp(kz,ke,kb)  !kz=nconteqa+2, from param.h
      common /freycomm/
     1  frmodp, fr_gyrop, src_nbi_ep, beamplsep,beamponp,beampoffp,
     1  hibrzp,mfm1p,nbeamsp
			    !These variables are set to frmod,fr_gyro,
			    !beamplse,beampon,beampoff,nbeams
                            !from frmod namelist.
                            !The namelist is declared in frname.h and
                            !passed to the comm.h related subroutines
                            !as arguments of subroutine frnfreya
			    !Similarly, hibrzp and mfmp1 are from freya 
                            !routines through frnfreya arguments.
                            !Purpose is communication with cql3d.
c

