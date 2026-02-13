c
c
      subroutine micxiniz
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     This routine determines certain constants and the mesh along
c     the magnetic field line z and sz, as well as related quantities.
c     This was part of micxinit in CQL3D.
c     These meshes are tailored to suit requirements specified by
c     the user in the namelist input.
c     Note: ls= FPE grid size; lsmax= profiles' grid size (along 's')
c..................................................................

      include 'param.h'
      include 'comm.h'
      
      !For CQLP only:
      dimension zd2bpol(lfield),zd2solr(lfield),zd2solz(lfield) !local
      !YuP[2021-04] Changed to lfield
      
      real*8 znormsh(0:lz) !YuP[2021-03] local working array
                           !(could use 0:lsmax) 
     
      !For CQL3D only:
      dimension wk_z(lz),wk_r(lz),wk_b(lz) ! local work.arrays
      !Meaning of lz: Grid size along poloidal field line

c%OS  compute psi_prime as df/ds
c%OS  fpsismi(l)=psis(l)*dls(2,2,1,l)+psis(l+1)*(1-dls(2,2,1,l))
c YuP      fpsismi(l)=(0.5-0.5*(1/(l+1))+0.5*(1/(ls+1-l)))*psis(l)+
c YuP     +  (0.5+0.5*(1/(l+1))-0.5*(1/(ls+1-l)))*psis(l+1)
c YuP: fpsismi() is not used

c.......................................................................
cl    1. Determine the field line orbit mesh z(1:lz,lr_). 
c     tfacz is a mesh squeeze (nonuniformity) factor:
c     tfacz=1. means uniform mesh; tfacz < 1. means geometric mesh with
c     points closer near zero. tfacz > 1. means points closer near zmax
c     For eqsym.eq."none", tfacz=1. appears best, since z(lz,)=z(1,).
c......................................................................

      if (eqsym.ne."none") then
         z(1,lr_)=0.
         z(lz,lr_)=zmax(lr_)
         htfg=tfacz*(zmax(lr_)/(lz-1))
         z(2,lr_)=htfg
         call micgetr(lz,zmax(lr_),htfg,ram,ksingul)
         if (ksingul .eq. 1) call diagwrng(8)
         do 100 l=3,lz-1
            z(l,lr_)=z(l-1,lr_)+ram*(z(l-1,lr_)-z(l-2,lr_))
 100     continue
         do l=1,lz-1
            zmid(l,lr_)=0.5*(z(l,lr_)+z(l+1,lr_))
         enddo
         zmid(lz,lr_)=zmax(lr_)
         z_bmax(lr_)=zmax(lr_)
         lz_bmax(lr_)=lz
         bpsi_z_bmax(lr_)=psif(zmax(lr_))
c         do l=1,lz
c         write(*,*) l,z(l,lr_),zmid(l,lr_)
c         enddo
c         write(*,*)'micxiniz: l,z,zmid for lr_=',lr_
c         pause
      endif

c......................................................................
c     If eqsym.eq."none", use equispaced mesh on either side of 
c     the inboard max-|B| point, with step size slightly adjusted
c     so the max-|B| point is a mesh point.

      if (eqsym.eq."none") then
         z_bmax(lr_)=es_bmax(lr_)
         !-YuP Was: lz_bmax(lr_)=nint(z_bmax(lr_)/zmax(lr_)*lz) !old version
         lz_bmax(lr_)= 1 + nint(z_bmax(lr_)/zmax(lr_)*(lz-1)) !-YuP Try (lz+1)/2
         dlz1=z_bmax(lr_)/(lz_bmax(lr_)-1)
         !-YuP Was: dlz2=(zmax(lr_)-z_bmax(lr_))/(lz-lz_bmax(lr_)-1) !old version
         dlz2=(zmax(lr_)-z_bmax(lr_))/(lz-lz_bmax(lr_)) !-YuP
         do l=1,lz_bmax(lr_)
            z(l,lr_)=(l-1)*dlz1                 !range: [0., z_bmax]
         enddo
         do l=lz_bmax(lr_)+1,lz
            z(l,lr_)=z_bmax(lr_)+(l-lz_bmax(lr_))*dlz2 ![z_bmax+dlz2, zmax]
         enddo
         do l=1,lz-1
            zmid(l,lr_)=0.5*(z(l,lr_)+z(l+1,lr_))
         enddo
         zmid(lz,lr_)=zmax(lr_) !-YuP: this is an extra ending point
         !-YuP Note: zmid is defined at lz-1 points; 
         !-YuP The mesh z is defined at lz points, 
         !-YuP  but last point = 1st point in R,Z coords.
c         do l=1,lz
c         write(*,*) l,z(l,lr_),zmid(l,lr_)
c         enddo
c         write(*,*)'micxiniz: l,z,zmid for lr_=',lr_
c         pause
         bpsi_z_bmax(lr_)=bpsi_max(lr_)  !New variable name for z-mesh.
      endif


c..................................................................
cl    2. Determine the bpsi array (ratio of B(z)/B(0)).
c     coeff1, terp1 are spline routines frequently utilized by CQL.
c..................................................................

      if (psimodel.eq."spline") then
        if (eqmod.eq."disabled") then
          !lfield=lfielda !YuP[2021-04] lfielda is not used anymore
          !Now all relevant arrays are pointers of size lfield
          lorbit(lr_)=lfield
          delz=zmax(lr_)/(lfield-1)
          do 24 lu=1,lorbit(lr_)
            es(lu,lr_)=(lu-1)*delz
            thtpol(lu,lr_)=pi*es(lu,lr_)*zmaxi(lr_)
            bpsi(lu,lr_)=(1.+eps(lr_))/(1.+eps(lr_)*cos(thtpol(lu,lr_)))
            eqbpol(lu,lr_)=bthr0(lr_)*bpsi(lu,lr_)
            solr(lu,lr_)=radmaj*(1.+eps(lr_)*cos(thtpol(lu,lr_)))
            !YuP[2020-01-29] Added:
            solz(lu,lr_)=radmaj*eps(lr_)*sin(thtpol(lu,lr_)) !assumed zmag=0.
            !recall that eps=rovera(lr_)*radmin/radmaj
            !YuP[2020-01-29] Added dlp definition for eqmod="disabled":
            eqdell(lu,lr_)=radmaj*eps(lr_)*pi/(lfield-1) !=r*pi/(lfield-1)
            !Here, it is constant along field line.
 24       continue
        endif ! (eqmod.eq."disabled")
        if (eqsym.eq."none") then
           i1p(1)=3
           i1p(2)=3
        else
           i1p(1)=2
           i1p(2)=2
           d2bpsi(lorbit(lr_),lr_)=0.
           d2bpsi(1,lr_)=0.
        endif

c-YuP Print this:        
c         step2=55./nstps
c         pt=0. 
c         do il=1,nstps
c            write(*,*)'micxiniz before coeff1 pt,psif=',pt,psif(pt)
c            pt=pt+step2
c         enddo

c-YuP: Here psif(pt) is ok; monotonically increasing with pt.
        
c-YuP        call coeff1(lorbit(lr_),es(1,lr_),bpsi(1,lr_),d2bpsi(1,lr_)
c-YuP     1    ,i1p,1,work) !-> to get d2bpsi (2nd derivatives)

c-YuP: Here psif(pt) is bad - non-monotonic (after coeff1 interpolation)
        
c-YuP From print out of d2bpsi: The 2nd derivative is quite bad.
c-YuP It results in an error with spline interpolation
c-YuP in psif(pt), near B=Bmin point.
c-YuP However, if these two lines calling coeff1 are commented
c-YuP (which makes all d2bpsi remaining zeroed),
c-YuP the subsequent call of terp1 in psif(pt) is
c-YuP equivalent to a linear interpolation.
c-YuP In this case there is no problem around B=Bmin point.
     
        
        i1p(1)=4
        i1p(2)=4
        call coeff1(lorbit(lr_),es(1,lr_),thtpol(1,lr_),d2thtpol(1,lr_),
     1    i1p,1,work)
        itab(1)=1
        itab(2)=0
        itab(3)=0
        


c..................................................................
c     Determine an evenly spaced psi array.
c..................................................................
c%OS  
c%OS  
c%OS  dels=es(lorbit(lr_),lr_)/float(incza-1)
c%OS  psiesfi(1,lr_)=1.
c%OS  esfi(1,lr_)=0.
c%OS  do 29 iv=2,incza-1
c%OS  esfi(iv,lr_)=(iv-1)*dels
c%OS  call terp1(lorbit(lr_),es(1,lr_),bpsi(1,lr_),
c%OS  1     d2bpsi(1,lr_),esfi(iv,lr_),1,tab,itab)
c%OS  psiesfi(iv,lr_)=tab(1)
c%OS29continue
c%OS  psiesfi(incza,lr_)=bpsi(lorbit(lr_),lr_)
c%OS  esfi(incza,lr_)=es(lorbit(lr_),lr_)
c%OS  
c%OSc..................................................................
c%OSc Determine an evenly spaced (in argument) s array (arg is psi)
c%OSc..................................................................
c%OS  
c%OS  call coeff1(lorbit(lr_),bpsi(1,lr_),es(1,lr_),d2es(1,lr_),
c%OS  1     i1p,1,work)
c%OS  deltapsi(lr_)=(bpsi(lorbit(lr_),lr_)-1.)/(incza-1)
c%OS  psifi(1,lr_)=1.
c%OS  espsifi(1,lr_)=0.
c%OS  do 28 iv=2,incza-1
c%OS  psifi(iv,lr_)=psifi(iv-1,lr_)+deltapsi(lr_)
c%OS  call terp1(lorbit(lr_),bpsi(1,lr_),es(1,lr_),d2es(1,lr_),
c%OS  1      psifi(iv,lr_),1,tab,itab)
c%OS  espsifi(iv,lr_)=tab(1)
c%OS28continue
c%OS  espsifi(incza,lr_)=zmax(lr_)
c%OS  espsifi(inczpa,lr_)=zmax(lr_)
c%OS  psifi(inczpa,lr_)=bpsi(lorbit(lr_),lr_)
c%OS  psifi(incza,lr_)=bpsi(lorbit(lr_),lr_)
        pol(1,lr_)=0. ! the midplane outer point
        if (eqsym.eq."none") then
           pol(lz,lr_)=2*pi
        else
           if(machine.eq."toroidal")then
             pol(lz,lr_)=pi ! midplane inner point
           else  ! mirror (only eqsym.ne."none" case, for now)
             pol(lz,lr_)=thtpol(lorbit(lr_),lr_) ! supposed to be <pi/2
           endif
        endif
        do 22 l=2,lz-1
          call terp1(lorbit(lr_),es(1,lr_),thtpol(1,lr_),d2thtpol(1,lr_)
     1      ,z(l,lr_),1,tab,itab)
          pol(l,lr_)=tab(1)
 22     continue

c.......................................................................
c     solrz(1:lz,lr_),solzz(1:lz,lr_)
c.......................................................................

        if (eqsym.eq."none") then
           i1p(1)=3
           i1p(2)=3
        else
           i1p(1)=2
           i1p(2)=2
           d2solrz(1,lr_)=0.0
           d2solrz(lorbit(lr_),lr_)=0.0
           d2solzz(1,lr_)=0.0
           d2solzz(lorbit(lr_),lr_)=0.0
           d2bpolz(1,lr_)=0.0
           d2bpolz(lorbit(lr_),lr_)=0.0
        endif
        call coeff1(lorbit(lr_),es(1,lr_),solr(1,lr_),d2solrz(1,lr_),
     1       i1p,1,work)
        call coeff1(lorbit(lr_),es(1,lr_),solz(1,lr_),d2solzz(1,lr_),
     1       i1p,1,work)
        call coeff1(lorbit(lr_),es(1,lr_),eqbpol(1,lr_),d2bpolz(1,lr_),
     1       i1p,1,work)
     
        do 221 l=1,lz
          call terp1(lorbit(lr_),es(1,lr_),solr(1,lr_),d2solrz(1,lr_),
     1         z(l,lr_),1,tab,itab)
          solrz(l,lr_)=tab(1)
          call terp1(lorbit(lr_),es(1,lr_),solz(1,lr_),d2solzz(1,lr_),
     1         z(l,lr_),1,tab,itab)
          solzz(l,lr_)=tab(1)
          call terp1(lorbit(lr_),es(1,lr_),eqbpol(1,lr_),d2bpolz(1,lr_),
     1         z(l,lr_),1,tab,itab)
          bpolz(l,lr_)=tab(1)
 221    continue
 
! YuP[2019-12-12] migrated some corrections from CQL3D-FOW version.
        !YuP[2017-11-29] Adjustment of solzz(), solrr(), bpolz() arrays.
        !REASON: Some of surfaces can be traced towards positive Z, 
        !while other - towards negative Z (example: RFP, like MST machine).
        !To avoid an error in calculation of ddarea and ddvol of cells,
        !where a cell is defined by lr+1/lr-1 and l+1/l-1 four points,
        !make all traces of surfaces to go towards positive Z,
        !for all lr indexes.
        if(machine.eq."toroidal")then !(no need for mirror configuration?)
        if(solzz(2,lr_).lt.solzz(1,lr_))then ! Z(2)<Z(1)
          ! solz is traced towards smaller(negative) Z.
          ! Make adjustment, depending on eqsym.
          if(eqsym.eq.'none')then ! whole surface was traced.
            ! Reverse l index, so that effectively
            ! the trace of surface goes backward, from end to beginning
            do l=1,lz
              wk_z(l)=solzz(lz-l+1,lr_) 
              ! solzz(l=1)  is mapped to wk_z(lz)
              ! solzz(l=lz) is mapped to wk_z(1)
              wk_r(l)=solrz(lz-l+1,lr_) 
              wk_b(l)=bpolz(lz-l+1,lr_) 
            enddo
            do l=1,lz
              solzz(l,lr_)=wk_z(l)
              solrz(l,lr_)=wk_r(l)
              bpolz(l,lr_)=wk_b(l)
            enddo ! Done - the line is reversed.
          else ! half-surface (started from Z=0)
            !In this case, simply take abs() value for solzz.
            !No need to adjust solrz or bpolz because of up/dn symmetry.
            do l=1,lz
              solzz(l,lr_)=abs(solzz(l,lr_))
            enddo
          endif ! eqsym
        endif ! Z(2)<Z(1)
        endif ! machine
        !YuP[2017-11-29] DONE Adjustment of solzz(), solrr(), bpolz()



      else if(psimodel.eq."axitorus") then

        do 25 l=1,lz
          pol(l,lr_)=pi*z(l,lr_)*zmaxi(lr_)
 25     continue

cBH000416        if (vlfmod.eq."enabled") then
          do 26 l=1,lz
            solrz(l,lr_)=radmaj*(1.+eps(lr_)*cos(pol(l,lr_)))
           !solzz(l,lr_)=radmaj*(1.+eps(lr_)*sin(pol(l,lr_))) ! YuP[07-2017] BUG
            solzz(l,lr_)=radmaj*( zmag/rmag +eps(lr_)*sin(pol(l,lr_)) ) !correct
            !Note: rmag=radmaj in this model, and zmag=0.
c            write(*,'(a,2i4,3f13.2)')
c     +         'micxiniz: lr_,l, solrz(),solzz(),pol(l,lr_)',
c     +          lr_,l, solrz(l,lr_), solzz(l,lr_), pol(l,lr_)
 26       continue
cBH000416        endif

      endif  ! On psimodel

      bbpsi(1,lr_)=1.+em14
      do 220 l=2,lz
        bbpsi(l,lr_)=psif(z(l,lr_))
 220  continue
      if (eqsym.eq."none") bbpsi(lz,lr_)=bbpsi(1,lr_)

c.......................................................................
cl    3. Construct mesh and profiles in CQLP mode on given flux surface
c     Note: If cqlpmod=enabled, lz=lsmax, full mesh along B
c     ls gives mesh on which FP equations are solved
c     (if periodic: sz(1)=sz(ls))
c     psis=B(s)/B(midplane), psisp=d(psis)/ds
c.......................................................................

c.......................................................................
cl    3.1 s, sz, psis, psisp and related meshes
c     Note: only one flux surface in CQLP so far (lrz=1)
c.......................................................................

      if (cqlpmod.ne."enabled" .or. lr_.ne.lrindx(1)) go to 999  !To END
      
      !-------- The rest is for CQLP only -------------------------------

c     Following, not checked for eqsym.eq."none". Possibly OK. BH090923.

c     Assumes midplane at l=1. If not, bmidplne, z mesh, bpsi etc should be
c     redefine
      do 310 l=1,ls ! over FPE grid. 
        !lsindx(l) selects points from profiles' grid,
        ! example: lsindx(1)=3, lsindx(2)=5, lsindx(3)=10
        sz(l)=z(lsindx(l),lr_) !So, sz() is over FPE grid
        psis(l)=bbpsi(lsindx(l),lr_)/bbpsi(lmdpln(indxlr(lr_)),lr_)
        psisp(l)=psifp(z(lsindx(l),lr_))/bbpsi(lmdpln(indxlr(lr_)),lr_)
 310  continue

c     psipols, solrs, solzs
      i1p(1)=2
      i1p(2)=2
      zd2bpol(1)=0.0
      zd2bpol(lorbit(lr_))=0.0
      call coeff1(lorbit(lr_),es(1,lr_),eqbpol(1,lr_),zd2bpol(1),
     1  i1p,1,work)
      i1p(1)=2
      i1p(2)=2
      zd2solr(1)=0.0
      zd2solr(lorbit(lr_))=0.0
      call coeff1(lorbit(lr_),es(1,lr_),solr(1,lr_),zd2solr(1),
     1  i1p,1,work)
      i1p(1)=4
      i1p(2)=4
      call coeff1(lorbit(lr_),es(1,lr_),solz(1,lr_),zd2solz(1),
     1  i1p,1,work)
      !YuP: Then, zd2bpol,zd2solr,zd2solz are over full (profiles) grid

      itab(1)=1
      itab(2)=0
      itab(3)=0

      do 311 l=1,ls ! over FPE grid
        call terp1(lorbit(lr_),es(1,lr_),eqbpol(1,lr_),zd2bpol(1)
     1    ,sz(l),1,tab,itab)
        psipols(l)=tab(1)/bmidplne(lr_)
        call terp1(lorbit(lr_),es(1,lr_),solr(1,lr_),zd2solr(1)
     1    ,sz(l),1,tab,itab)
        solrs(l)=tab(1)
        call terp1(lorbit(lr_),es(1,lr_),solz(1,lr_),zd2solz(1)
     1    ,sz(l),1,tab,itab)
        solzs(l)=tab(1)
 311  continue
      do 312 l=2,ls-1  ! over FPE grid
        dsz(l)=0.5*(sz(l+1)-sz(l-1))
        dszp5(l)=sz(l+1)-sz(l)
        eszp5(l)=1./dszp5(l)
        dszm5(l)=sz(l)-sz(l-1)
        eszm5(l)=1./dszm5(l)
 312  continue
      dsz(1)=(sz(2)-sz(1))*0.5
      dsz(ls)=(sz(ls)-sz(ls-1))*0.5
      dszm5(1)=0.0
      eszm5(1)=0.0
      dszp5(1)=sz(2)-sz(1)
      eszp5(1)=1./dszp5(1)
      dszm5(ls)=sz(ls)-sz(ls-1)
      eszm5(ls)=1./dszm5(ls)
      dszp5(ls)=0.0
      eszp5(ls)=0.0

c.......................................................................
cl    3.2 Construct parallel profile n(s) and T(s)
c.......................................................................

c     tdxin13d assumes a normalized mesh
      znormsh(0)=0.0
      do 320 l=1,lz ! This is the profiles' grid (better use lsmax here)
        znormsh(l)=z(l,lr_)/z(lz,lr_)
 320  continue
c     only parabola option for n(s), T(s) so far
      do 321 ik=1,ntotal
        call tdxin13d(denpar,znormsh,lsmax,ntotala,ik,npwr(0),mpwr(0))
        call tdxin13d(temppar,znormsh,lsmax,ntotala,ik,npwr(ik),
     +    mpwr(ik))
 321  continue

c.......................................................................
cl    4. Define end points accordingly whether mesh is periodic or not
c.......................................................................

      if (transp.ne."enabled") go to 999

      ! Here: we are still dealing with CQLP, to the end of subroutine.
      !Note: if transp=enabled then ls=lsmax
      if (sbdry.ne."periodic" .and. numclas.ne.1) then !CQLP here
        do 400 ik=1,ntotal
          do 401 il=0,lsmax+1,lsmax+1
            denpar(ik,il)=0.0
            temppar(ik,il)=0.0
 401      continue
 400    continue
        !do 410 il=0,lsmax+1,lsmax+1 !YuP[2021-03] Changed to:
        do 410 il=0,ls+1,ls+1
          !YuP: Error? These arrays below are over FPE grid (ls)
          ![2021-02]Suggest to change lsmax-->ls here
          sz(il)=0.0
          psisp(il)=0.0
          psipols(il)=0.0
          solrs(il)=0.0
          solzs(il)=0.0
          dsz(il)=0.0
          dszm5(il)=0.0
          dszp5(il)=0.0
          eszm5(il)=0.0
          eszp5(il)=0.0
 410    continue
      else
c     0 <=> lsmax and lsmax+1 <=> 1
        do 420 ik=1,ntotal
          do 421 il=0,lsmax+1,lsmax+1
            iequiv=il/(lsmax+1)+lsmax*((lsmax+1-il)/(lsmax+1))
            denpar(ik,il)=denpar(ik,iequiv)
            temppar(ik,il)=temppar(ik,iequiv)
 421      continue
 420    continue
        !do 430 il=0,lsmax+1,lsmax+1 !YuP[2021-03] Changed to
        do 430 il=0,ls+1,ls+1
          !YuP: Error? These arrays below are over FPE grid (ls)
          ![2021-02]Suggest to change lsmax-->ls here
          !iequiv=il/(lsmax+1)+lsmax*((lsmax+1-il)/(lsmax+1)) !YuP[2021-03] Changed to
          iequiv=il/(ls+1)+ls*((ls+1-il)/(ls+1))
          sz(il)=sz(iequiv)
          psisp(il)=psisp(iequiv)
          psipols(il)=psipols(iequiv)
          solrs(il)=solrs(iequiv)
          solzs(il)=solzs(iequiv)
          dsz(il)=dsz(iequiv)
          dszm5(il)=dszm5(iequiv)
          dszp5(il)=dszp5(iequiv)
          eszm5(il)=eszm5(iequiv)
          eszp5(il)=eszp5(iequiv)
 430    continue
      endif

c.......................................................................
cl    5. Define dpsis/ds in the same way as df/ds in transport equation
c.......................................................................

c%OS  
c%OS  if (noffso(1,2) .eq. 999) go to 999
c%OS  
c%OS  do 500 l=1,ls
c%OS  psisp(l)=(fpsismi(l)-fpsismi(l-1))/dsz(l)
c%OS  500  continue
c%OS  psisp(0)=0.0
c%OS  psisp(ls+1)=0.0
c%OS  if (sbdry .eq. "periodic") then
c%OS  psisp(0)=psisp(ls)
c%OS  psisp(ls+1)=psisp(1)
c%OS  endif
c%OS  

c.......................................................................

 999  continue

      return
      end  subroutine micxiniz
