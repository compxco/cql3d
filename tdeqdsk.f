c
c
      subroutine tdeqdsk
      implicit integer (i-n), real*8 (a-h,o-z)
      save
      include 'param.h'
      parameter (niterate=20)
      parameter (ncoila=5,nccoila=50)
      include 'comm.h'

      dimension fpsiareq(nnra),ffpareq(nnra),prareq(nnra),
     1  ppareq(nnra),q(nnra),eqpsi_(nconteqa),d2q_(nconteqa)
      parameter(itlrza=3*lrza+1+3*nrada+3*nconteqa)
      common/coils/ cvac(0:9), ccoil(nccoila,ncoila),ncoil(ncoila)
      dimension workk(itlrza),atw(ntotala),dpsiar(nrada),d2gpary(nrada),
     1  d2pary(nrada),d2ppary(nrada),ene(lrza),te(lrza),
     1  ti(lrza,ntotala),enp(lrza,ntotala),eni(lrza,ntotala),
     1  z1(lrza,ntotala),dumm(400),xcontr(50),ycontr(50),
     1  psiart(nrada),xlimiter(50),ylimiter(50)
      dimension zeqpsir(lrza)
      common /tsc1/ psimago,psilimo
      character*5 blanks
      data mode /0/
      data blanks/"     "/

c.......................................................................
c     Called from tdinitl, if partner.eq."bram",
c        and from tdtloop, if n.ge.nstop.
c.......................................................................


      if (eqmod.ne."enabled") return

c.......................................................................
c     pick equilpsp values according to lrindx mesh
c.......................................................................
      do 20 l=1,lrz
        zeqpsir(l)=equilpsp(lrindx(l))
 20   continue

c..................................................................
c     file (the file used to reinitialize the MHD equilibrium code).
c     The procedure is as follows (1) create the arrays on the flux
c     surface grid (2) interpolate to the MHD equilibrium grid (
c     required for eqdsk) and (3) rescale to get all quantities into
c     MKS.
c     The procedure differs depending on the state of variable
c     partner.
c..................................................................

      !ncontr=0    !-YuP: commented out. Why needed here? defined in equilib
      !nlimiter=0  !-YuP: commented out. Why needed here?

      if (partner.eq."selene") then

c..................................................................
c     Compute the derivative of Pressure  w.r.t. psi
c..................................................................

        ipestg=3
        call firstdrv(equilpsp(1),prest,prestp,lrzmax)

c..................................................................
c     Compute <JparB> (volume, or flux surface average).
c     Compute the bootstrap effect too, <JbsB>. The bootstrap
c     current (bscurm) is assumed to be area averaged.
c
c     Presently, only electron bootstrap contribution included 
c                for jhirsh=88/99. (BobH, 990823).
!     Now (2019) electron and ion contribution is included 
!         into bscurm(l,k,kk) in subr. tdboothi for jhirsh=88/99, 
!         but here only electron contribution is used
!         (in bscurm(ilr,1,1) below, last index "1" is for electrons)  
c
c     Input currents are in Amps/cm**2
c..................................................................

        do 5 ll=1,lrz
          ilr=lrindx(ll)
          jparb(ll)=0.0
          jparbp(ll)=0.0
          if (zmaxpsi(ilr).ne.zero) then
            jparb(ll)=currmtz(lmdpln(ll))*bmdplne(ilr)*psidz(ilr)/
     1        zmaxpsi(ilr)*3.e+9
            jparbp(ll)=currmtpz(lmdpln(ll))*bmdplne(ilr)*psidz(ilr)/
     1        zmaxpsi(ilr)*3.e+9
          endif
          jparbt(ll)=bscurm(ilr,1,1)*btor*3.e+9
 5      continue

c..................................................................
c     Procedure for computing f**2 is iterative. We presume f is
c     known at the magnetic axis (boundary condition) from the
c     previous call to the equilibrium code and integrate out.
c     equilpsp=-equilpsi
c..................................................................

        factr=-8.*pi/clight
        do 101 ll=1,lrz
          ilr=lrindx(ll)
          tr1(ll)=bmdplne(ilr)**2*psidz(ilr)/zmaxpsi(ilr)
 101    continue
        if (mode.ne.0) then
cBH070116          fpsiz2(0)=fpsiar(nnv)**2
          fpsiz2(0)=fpsiar(nfp)**2
          do 100 nit=1,niterate
            do 120 ll=1,lrz
              ilr=lrindx(ll)
              fpsiz2(ll)=fpsiz2(ll-1)+(equilpsp(ilr)-equilpsp(ilr-1))*
     1          factr*(1./tr1(ll))*(fpsiz(ll)*(jparbp(ll)+jparbt(ll))+
     1          clight*fpsiz(ll)**2*prestp(ilr))
 120        continue

c..................................................................
c     f
c..................................................................


            do 130 ll=1,lrz
              fpsiz(ll)=sqrt(fpsiz2(ll))
 130        continue
 100      continue
        endif

c..................................................................
c     ff'
c..................................................................

        do 140 ll=1,lrz
          ilr=lrindx(ll)
          ffpsiz(ll)=(1./tr1(ll))*factr*(fpsiz(ll)*(jparbp(ll)+
     1      jparbt(ll))+clight*fpsiz(ll)**2*prestp(ilr))*.5
 140    continue

c..................................................................
c     Compute the toroidal current, toteqd
c..................................................................

        toteqd=0.
        do 150 ll=1,lrz
          ilr=lrindx(ll)
          toteqd=toteqd-clight/(8.*pi**2)*ffpsiz(ll)*onovrpz(ilr,2)
     +      *dvol(ilr)-clight/twopi*prestp(ilr)*dvol(ilr)
 150    continue

c..................................................................
c     The eqdsk format requires that all the quantities which are
c     functions of psi be written on an evenly spaced grid in psi.
c     This will be done now. We have psilim=0
c     The psi array to which we wish to interpolate was determined in
c     subroutine equilib - psiar
c..................................................................

        i1p(1)=4 !=4 means cubic spline; requires at least 4 points 
        i1p(2)=4

c..................................................................
c     ff' and f
c..................................................................
        if(lrz.lt.4)then !YuP[2021-11-17] added check/stop for coeff1
          STOP ' tdeqdsk: subr.coeff1(lrz,...) requires lrz>3'
          !Actually, it could work for lrz=3; 
          !Use i1p()=2 (first derivative is given),
          !Need values of 1st derivatives at rho=0 and rho=1.
        endif
        call coeff1(lrz,zeqpsir(1),fpsiz,d2fpsiz,i1p,1,workk)
        call coeff1(lrz,zeqpsir(1),ffpsiz,d2ffpsiz,i1p,1,workk)
        itab(1)=1
        itab(2)=0
        itab(3)=0

c..................................................................
c     Pressure and derivative
c     psiar defined in equilib - rescale so that min is at mag axis;
c     max (=0 is at magnetic axis).
c..................................................................

cBH070116        call dcopy(nnv,psiar,1,dummyar,1)
        call dcopy(nfp,psiar,1,dummyar,1)
cBH070116        do 60 ne=1,nnv
        do 60 ne=1,nfp
cBH070116          psiar(ne)=-dummyar(nnv+1-ne)
          psiar(ne)=-dummyar(nfp+1-ne)
 60     continue
        do 65 i=2,nconteqn
          eqpsi_(i)=-eqpsi(i)
 65     continue
        call coeff1(nconteqn-1,eqpsi_(2),q_(2),d2q_(2),i1p,1,workk)
cBH070116        do 50 ne=1,nnv
        do 50 ne=1,nfp
          call terp1(lrzmax,equilpsp(1),prest,d2prest,psiar(ne),1,tab,
     +      itab)
          prar(ne)=tab(1)
          if (prar(ne).lt.0.) prar(ne)=0.
          call terp1(lrz,zeqpsir(1),fpsiz,d2fpsiz,psiar(ne),1,tab,
     +      itab)
          fpsiar(ne)=tab(1)
          call terp1(lrz,zeqpsir(1),ffpsiz,d2ffpsiz,psiar(ne),1,tab,
     +      itab)
          ffpar(ne)=tab(1)
          call terp1(nconteqn-1,eqpsi_(2),q_(2),d2q_(2),psiar(ne),1,tab,
     1      itab)
          q(ne)=tab(1)
 50     continue
        ffpar(1)=ffpsiz(1)
cBH070116        call firstdrv(psiar(1),prar(1),ppar(1),nnv)
        call firstdrv(psiar(1),prar(1),ppar(1),nfp)

      else if (eqsource.eq."eqdsk" .or. eqsource.eq."topeol" ) then

cBH070116        nnv_=nnv
cBH070116        do 201 j=1,nnv
cBH070116          fpsiareq(j)=fpsiar(nnv+1-j)
cBH070116          prareq(j)=prar(nnv+1-j)
cBH070116          ppareq(j)=ppar(nnv+1-j)
       nfp_=nfp
        do 201 j=1,nfp
          fpsiareq(j)=fpsiar(nfp+1-j)
          prareq(j)=prar(nfp+1-j)
          ppareq(j)=ppar(nfp+1-j)
 201    continue

      endif   ! endif on (partner.eq.'selene' .or. eqsource.eq.'eqdsk')

c..................................................................
c     Convert as appropriate to MKS
c..................................................................

cBH070116      do 200 i=1,nnv
      do 200 i=1,nfp
        fpsiareq(i)=fpsiareq(i)/1.e+6
        ffpareq(i)=ffpar(i)*1.e+2   !Previously 1e-4, so check if
                                    !using partner='selene', BH010226.
        prareq(i)=prareq(i)/10.   
        ppareq(i)=ppareq(i)*1.e+7
 200  continue


c..................................................................
c     Now for the case partner="bramb"
c..................................................................

      if (eqsource.eq."tsc") then
cBH070116        nnv_=nnr
        nfp_=nnr
        dpsi_=(psilimo-psimago)/(nnr-1)
        do 300 i=1,nnr
          psiart(i)=psimago+(i-1)*dpsi_
 300    continue
        do 307 i=2,nconteqn
          eqpsi_(i)=-eqpsi(i)*1.e-8+psilimo
 307    continue

c..................................................................
c     The radial arrays from the TSC code are defined on an arbitrary
c     rho mesh. To create a GA style eqdsk file, we have to interpolate
c     onto the psiart psi mesh (defined above). We remain in MKS units.
c     The temperatures are in KeV.
c..................................................................

        i1p(1)=4
        i1p(2)=4
        call coeff1(npsitm,psiar_,fpsiar_,dpsiar,i1p,1,workk)
        call coeff1(npsitm,psiar_,gpary,d2gpary,i1p,1,workk)
        call coeff1(npsitm,psiar_,pary,d2pary,i1p,1,workk)
        call coeff1(npsitm,psiar_,ppary,d2ppary,i1p,1,workk)
        call coeff1(nconteqn-1,eqpsi_(2),q_(2),d2q_(2),i1p,1,workk)
        itab(1)=1
        itab(2)=0
        itab(3)=0
        do 301 i=1,nnr
          call terp1(npsitm,psiar_,fpsiar_,dpsiar,psiart(i),1,tab,itab)
          fpsiareq(i)=tab(1)
          call terp1(npsitm,psiar_,gpary,d2gpary,psiart(i),1,tab,itab)
          ffpareq(i)=tab(1)
          call terp1(npsitm,psiar_,pary,d2pary,psiart(i),1,tab,itab)
          prareq(i)=tab(1)
          call terp1(npsitm,psiar_,ppary,d2ppary,psiart(i),1,tab,itab)
          ppareq(i)=tab(1)
          call terp1(nconteqn-1,eqpsi_(2),q_(2),d2q_(2),psiart(i),1,tab
     1      ,itab)
          q(i)=tab(1)
 301    continue

c..................................................................
c     Now set up the plasma arrays that will also be output
c..................................................................

        nprim=nspc
        write(6,'(/A/)') ' WARNING: nimp not defined in comm.h'
         nion=nprim+nimp
      else if (eqsource.eq."eqdsk" .or. eqsource.eq."topeol") then
        ku=0
        if (kelecm.ne.0) ku=ku+1
        if (kelecg.ne.0) ku=ku+1
        nion=ntotal-ku
        nprim=nion
      endif   ! endif on (partner.eq.'tsc' .or. eqsource.eq.'eqdsk')

      nti=1
      nimp=0
      ku=0
      do 302 k=1,ntotal
        if (k.ne.kelecg .and. k.ne.kelecm) then
          ku=ku+1
          atw(ku)=fmass(k)/1.6724e-24
          do 303 l=1,lrzmax
            z1(l,ku)=bnumb(k)
 303      continue
        endif
 302  continue
      k=kelecm
      if (k.eq.0) k=kelecg
      do 304 l=1,lrzmax
        ene(l)=reden(k,l)*1.e6
        te(l)=temp(k,l)
 304  continue
      ku=0
      do 305 k=1,ntotal
        if (k.ne.kelecg .and. k.ne.kelecm) then
          ku=ku+1
          do 306 l=1,lrzmax
            enp(l,ku)=reden(k,l)*1.e6
            ti(l,ku)=temp(k,l)
 306      continue
        endif
 305  continue

c..................................................................
c     Continue conversion to MKS
c..................................................................

      psilim=-psilim/1.e+8
      do 205 i=1,nnz
        do 206 j=1,nnr
          epsi(j,i)=-epsi(j,i)/1.e+8
 206    continue
 205  continue
      psimag=-psimag/1.e+8
      btor=btor/1.e+4
      rbox=rbox/1.e+2
      zbox=zbox/1.e+2
      rboxdst=rboxdst/1.e+2
      radmaj=radmaj/1.e+2
      raxis=rmag/1.e+2
      zaxis=0.
      toteqd=toteqd/3.e+9

c..................................................................
c     Disk writes
c..................................................................

      do i=1,5     ! for dummy data below
         tem1(i)=0.
      enddo

      if (eqsource.eq."eqdsk" .or. eqsource.eq."topeol") then
        if (eqdskalt.ne."enabled") go to 800
        open(unit=17,file="tdeqdsk",delim='apostrophe',
     +       status='unknown')
      elseif (eqsource.eq."tsc") then
        open(unit=17,file="tdeqdsk",delim='apostrophe',status='unknown')
      else ! could be "miller"
        goto 800 ! skip writing data
      endif
cBH:000906  Maybe problem with format of the following write(17,210)
cBH070116      write(17,210) mnemonic,ipestg,nnr,nnz,nnv_
cBH070116:  Swe ymideqd=0.  BE CAREFUL in future something else needed.
      ymideqd=0.
      write(17,210) mnemonic,ipestg,nnr,nnz,nfp_
      write(17,220) rbox,zbox,radmaj,rboxdst,ymideqd
      write(17,220) raxis,zaxis,psimag,psilim,btor
      write(17,220) toteqd,(tem1(i),i=1,4)
      write(17,220) (tem1(i),i=1,5)
cBH070116      write(17,220) (fpsiareq(i),i=1,nnv_)
cBH070116      write(17,220) (prareq(i),i=1,nnv_)
cBH070116      write(17,220) (ffpareq(i),i=1,nnv_)
cBH070116      write(17,220) (ppareq(i),i=1,nnv_)
      write(17,220) (fpsiareq(i),i=1,nfp_)
      write(17,220) (prareq(i),i=1,nfp_)
      write(17,220) (ffpareq(i),i=1,nfp_)
      write(17,220) (ppareq(i),i=1,nfp_)
      write(17,220) ((epsi(i,j),i=1,nnr),j=1,nnz)
      if (partner.ne."selene" .or. partner.eq."bram") then
cBH070116        write(17,220) (q(i),i=1,nnv_)
        write(17,220) (q(i),i=1,nfp_)
      endif
      if (partner.eq."selene") then
        write(17,221) (ncoil(i),i=1,5)
        do 230 i=1,5
          if(ncoil(i).le.0) go to 230
          write(17,250) (ccoil(nn,i),nn=1,ncoil(i))
 230    continue
        write(17,250) (cvac(i),i=1,9)
      endif
      if (partner.ne."selene") then
         write(*,*) 
     +  'tdeqdsk: work reqd to write xcontr,ycontr,xlimiter,ylimiter'
c        write(17,221) ncontr,nlimiter
c        if (ncontr.ne.0) then
c          write(17,220)(xcontr(i),ycontr(i),i=1,ncontr)
c          write(17,220)(xlimiter(i),ylimiter(i),i=1,nlimiter)
c        endif

c..................................................................
c     The following output is not part of the "official" eqdsk. It is
c     what distinguishes eqdsk created in onetwo from those created by
c     EFIT.
c     we put out enough information so that the eqdsk could be used as
c     a restart file.
c
c     lrzmax  = number of radial mesh points
c     nprim= number of primary ion species
c     nimp = number of impurities
c     nti  = number of species for which ion temperatures given
c     (1, for lumped ion species).
c     (nion=nprim+nimp)
c     atw  = atomic mass numbers of ions
c     z1   = charge state of each ion (can vary as function or radius).
c     zeffz= radial variation of z_effective
c     1.e-2*radmin*rya(j) = radial rho mesh from mag. axis to edge.
c     -equilpsi(j)*1.e-8= corresponding psi values (MKS) from magnetic
c     axis to edge.
c     te   = electron temperature (keV) on radial mesh.
c     ti   = ion temperature (keV) on radial mesh.
c     ene  =  electron density (MKS)
c     enp  =  primary (and impurity if nimp.ne.0)ion species density (MKS)
c..................................................................

        nti=1
        write(17,221) lrzmax,nprim,nimp,nti,lrors,lrz
        write (17,220) (atw(k),k=1,nion)
        do 700  k=1,nion
 700    write(17,220) (z1(j,k),j=1,lrzmax)
        write(17,220) (zeff(j),j=1,lrzmax)
        do 705  j=1,lrzmax
 705    dumm(j)=1.e-2*radmin*rya(j)
        write(17,220) (dumm(j),j=1,lrzmax)
        do 706 j=1,lrzmax
          dumm(j)=-equilpsi(j)*1.e-8
 706    continue
        write(17,220) (dumm(j),j=1,lrzmax)
        write(17,220) (te(j),j=1,lrzmax)
        do 710  k=1,nti
 710    write(17,220) (ti(j,k),j=1,lrzmax)
        write(17,220)  (ene(j),j=1,lrzmax)
        do 730 k=1,nprim
 730    write(17,220) (enp(j,k),j=1,lrzmax)
        if (nimp.ne.0)  then
          do 740 k=1,nimp
 740      write(17,220) (enp(j,k),j=1,lrzmax)
        endif
      endif
      close(unit=17)

 800  continue

c..................................................................
c     Convert back to cgs where necessary..
c..................................................................

      psilim=-psilim*1.e8
      psimag=-psimag*1.e+8
      do 809 j=1,nnz
        do 810 i=1,nnr
          epsi(i,j)=-epsi(i,j)*1.e+8
 810    continue
 809  continue
      btor=btor*1.e+4
      rbox=rbox*1.e+2
      zbox=zbox*1.e+2
      rboxdst=rboxdst*1.e+2
      radmaj=radmaj*1.e+2
      toteqd=toteqd*3.e+9
      raxis=rmag*1.e+2
      zaxis=0.
 250  format( (5(e21.14)))
 210  format(a48,4i4)
 220  format(5e16.9)
 221  format(6i5)
      return
      end
