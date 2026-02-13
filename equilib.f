c
c
      subroutine equilib(Requil,Zequil,index,PSIequil,BReq,BPHIeq,BZeq)
     
      implicit integer (i-n), real*8 (a-h,o-z)
      
c  Call with index=0 for setup,
c            index=1 for PSI,BR,BPHI,BZ return (cgs units)
c                    for a given Requil,Zequil coordinate.
      
      include 'param.h'
      include 'comm.h'
      
      parameter(nworka=3*nnra+1)
      parameter (ncoila=5,nccoila=50)

      character*8 ntitle,dat   ! Added for g95 compiler, Urban 110708
      dimension ntitle(5),workk(nworka)
      common/coils/ cvac(0:9), ccoil(nccoila,ncoila),ncoil(ncoila)
c..................................................................
c     CQL3D subroutine:  Oct. 27, 2004, RWH.
c     This routine will read in the eqdsk file if eqsource="eqdsk".
c     The eqdsk data is up-down symmetrized if so speciefied
c     by the input variable eqsym.
c     The cql3d code has been generalized for non-up-down symmertic
c     equilibria (BH: Oct'09).
c     The routine assumes that the data coming from the equilibrium
c     code is in MKS. Conversion is made to cgs. Finally, the setup
c     for the routine which generates f(psi) (called eqfpsi) is
c     done. This involves a call to a spline setup routine.
c     If eqsource="topeol", eqtopeol is called.
c     If eqsource="tsc", data has already been read in from file
c     tscinp. It is a matter only of changing data to cgs.
c..................................................................


c..................................................................
c   EQDSK Description
c..................................................................
c In the eqdsk, poloidal flux function psi [poloidal flux/(2*pi)] is
c given on a rectagular grid, equispaced in the horizontal (r)
c and vertical direction.  Other parameters of the equilibrium
c are given, as described in the following.  Units are entirely MKSA.
c
c A standard convention of EFIT (a source of eqdsk files) is that
c the psi-funtion is minimum at the magnetic axis, regardless of
c the direction of the toroidal current.  This convention is
c not followed by all eqdsk generators.
c
c nnr,nnz,nnv      are the numbers of points in the horizontal (r)
c                  and vertical (z) directions.  In some cases,
c                  nnv, a number of radial flux surfaces different
c                  from nnr, MAY be given.
c rbox,zbox        are the horizontal and vertical full-widths
c                  of the  rectangle on which poloidal flux values
c                  are given.
c radmaj           is the nominal major radius of the torus.
c rboxdst          is the major radius of the inner edge of the box.
c                  of rectangular grid.
c ymideqd          is the vertical shift of the rectangular box
c                  midplane.
c raxis,zaxis      are the major and vertical height of magnetic axis.
c psimag,psilim    are the poloidal flux function values at the
c                  magnetic axis and the last closed flux surface
c                   (touching the limiter or the separatrix).
c btor             is the vacuum toroidal magnetic field at radmaj.
c toteqd           is the toroidal current.
c psimx1,psixm2,xax1,xax2,zax1,zax2,psisep,xsep,ysep - OBSOLETE.
c   [These quantities have been used in the description of doublet
c    plasmas, with two magnetic axes.]
c  ATTENTION:
c psimx1=psimag
c xax1=xma
c zax1=zma
c psisep=psilim
c
c feqd(nnr),pres(nnr), usually, although dimension is
c                      specified by nnv if it is present.
c fpsiar = r * B_phi  at nnr equispaced points in psi
c          from psimag to psilim.
c prar   = are the pressure values at the same points.
c ffpar  = fpsiar_prime, [i.e., d f/d (psi)] at the same points.
c ppeqd  =  p_prime [i.e., d prar/d (psi)] at the same points.
c epsi(nnr,nnz) are the psi values on the nnr * nnz
c               equispaced grid.
c qar(nnr)  gives safety factor q on the equispaced psi grid.

c The following quantities are given in some eqdsks, but are not
c presently used in cql3d:
c nlimit,nves  are the numbers of point at limiters and
c              vacuum vessel wall.
c rlimit(nlimit),zlimit(nlimit):
c rlimit,zlimit      is the r,z location of the limiters wall.
c       rves(nves),zves(nves):
c rves,zves          is the r,z location of the limiters
c                    vacuum vessel wall.
c
c Additional plasma data and coil data appears after the above
c data is some eqdsks.  In general, eqdsks need to contain
c at least all the above data down through epsi.  Most also
c contain the qar(), although is can be derived from the
c preceding data.
c..................................................................


      if (index .eq. 0) then !==========================================

      if(lr_.ne.lrzmax) return ! equilib is called from aingeom in lr loop
                               ! starting at lr_=lrzmax. Only calc once.
      
      if (eqsource.eq."tsc") then
        do 150 j=1,nnz
          do 160 i=1,nnr
            epsi(i,j)=epsi(i,j)-psilim
 160      continue
 150    continue
        do 170 i=1,nnv
          psiar(i)=-(psiar_(nnv+1-i)-psilim)*1.e+8
          fpsiar(i)=fpsiar_(i)
 170    continue
        psimag=psimag-psilim
        psilim=0.
        rbox=rright-rleft
        zbox=ztop_-zbot_
        rboxdst=rleft
        nfp=nnv
      else if (eqsource.eq."eqdsk") then

cBH020822  Adding nveqd (.le. nxeqd) for different number of
cBH020822  flux surfaces on which p,feqd,p',feqd',q are tabulated.
cBH020822  The standard EQDSK does not incorporate this feature,
cBH020822  although is is available in cql3d.
cBH020822  Bonoli uses it for ACCOME eqdsk output.

        open(unit=10,file=eqdskin,status='old')
        read(10,110,iostat=io1)(ntitle(i),i=1,5),dat,ipestg,nnr,nnz,nnv
        if (io1.ne.0) then
           rewind 30
           read(10,110,iostat=io1)(ntitle(i),i=1,5),dat,ipestg,nnr,nnz
           nnv=0
        endif
        if (nnv.eq.0) nnv=nnr
        if (nnv.gt.nnr) call eqwrng(11)
        if(nnr.gt.nnra .or. nnz.gt.nnza) call eqwrng(12)
        read(10,120) rbox,zbox,radmaj,rboxdst,ymideqd
        read(10,120) raxis,zaxis,psimag,psilim,btor
        read(10,120) toteqd,psimx1,psimx2,xax1,xax2
        read(10,120) zax1,zax2,psisep,xsep,zsep
        read(10,120) (fpsiar(i),i=1,nnv)

        read(10,120) (prar(i),i=1,nnv)
        read(10,120) (ffpar(i),i=1,nnv)
        read(10,120) (ppar(i),i=1,nnv)
        read(10,120) ((epsi(i,j),i=1,nnr),j=1,nnz)
        read(10,120) (qar(i),i=1,nnv)

         read (10, 8210)   ncontr, nlimiter
 8210    format (2i5)
         !write(*,*)'ncontr, nconmax ',ncontr, nlimiter
         
         if (ncontr.gt.0) then  !-> LCFS
           allocate(rcontr(ncontr),STAT=istat) 
           allocate(zcontr(ncontr),STAT=istat) 
           read (10, 8200)  
     .     (rcontr(i), zcontr(i),i = 1,ncontr) ! [m]
         else 
           !-YuP: Set LCFS = outer boundary of equilibrium grid
           ncontr=5
           allocate(rcontr(ncontr),STAT=istat) 
           allocate(zcontr(ncontr),STAT=istat) 
           rcontr(1) = rboxdst
           zcontr(1) = -0.5*zbox
           rcontr(2) = rboxdst
           zcontr(2) = +0.5*zbox
           rcontr(3) = rboxdst+rbox
           zcontr(3) = +0.5*zbox
           rcontr(4) = rboxdst+rbox
           zcontr(4) = -0.5*zbox
           rcontr(5) = rcontr(1)
           zcontr(5) = zcontr(1)
         endif
         
         if (nlimiter.gt.0) then !-> Limiter
           allocate(rlimiter(nlimiter),STAT=istat) 
           allocate(zlimiter(nlimiter),STAT=istat) 
           read (10, 8200)
     .     (rlimiter(i), zlimiter(i), i = 1,nlimiter) ![m]
         else 
           !-YuP: Set limiter = outer boundary of equilibrium grid
           nlimiter=5
           allocate(rlimiter(nlimiter),STAT=istat) 
           allocate(zlimiter(nlimiter),STAT=istat) 
           rlimiter(1) = rboxdst
           zlimiter(1) = -0.5*zbox
           rlimiter(2) = rboxdst
           zlimiter(2) = +0.5*zbox
           rlimiter(3) = rboxdst+rbox
           zlimiter(3) = +0.5*zbox
           rlimiter(4) = rboxdst+rbox
           zlimiter(4) = -0.5*zbox
           rlimiter(5) = rlimiter(1)
           zlimiter(5) = zlimiter(1)
         endif
         
 8200    format(5e16.9)
 110  format(6a8,4i4)
 120  format(5e16.9)
 250  format( (5(e21.14)) )
 251  format(5i5)

      write(*,*)
      write(*,98) raxis,zaxis,psimag,psilim,qar(1),qar(nnv)
      write(*,99) rbox,zbox,rboxdst,ymideqd
 98   format('ORIGINAL Eqdsk values [mks units]: 
     +raxis,zaxis,psimag,psilim,qar(1),qar(nnv)=',  6(1pe12.4))
 99   format('Eqdsk values: rbox,zbox,rboxdst,ymideqd=',4(1pe12.4))
      write(*,*)

c.......................................................................
c     CQL cannot work correctly with B<0 (Lin-Liu)
c     A namelist variable, bsign, has been added to
c     adjust the toroidal field equilibrium data, and
c     to keep track of the sign for RF QL diffusion coeff purposes.
c.......................................................................
        if (btor.ge.0. .and. bsign.lt.0.) then
           stop "BTOR .gt.0, but bsign.lt.0. Check BSIGN"
        endif
        if (btor.lt.0.) then
           if (rdcmod.ne."disabled") then
             write(*,*) " " 
             write(*,*) " " 
             write(*,*) "   *******************************************"
             write(*,*) "   **BTOR .lt.0, Check RDC_UPAR_SIGN,      ***"
             write(*,*) "   **when using rdcmod.  DC coeffs use  -1.***"
             write(*,*) "   *******************************************"
             write(*,*) " " 
             write(*,*) " "
           endif
           if ( bsign.lt.0. ) then
              btor=bsign * btor
              do ii=1,nnv
                 fpsiar(ii)=bsign * fpsiar(ii)
              enddo
	      write(*,*)'equilib: Sign of btor and fpsiar is reversed'
           else
             write(*,*) " " 
             write(*,*) " " 
             write(*,*) "   *******************************************"
             write(*,*) "   ****BTOR .lt.0, Check BSIGN if using urf***"
             write(*,*) "   *******************************************"
             write(*,*) " " 
             write(*,*) " " 
             stop "BTOR .lt.0, Check BSIGN"
           endif
        endif
c
c$$$C%OS  
c$$$C%OS  not used and creates error when ncoil not defined properly
c$$$        read(10,251,END=299) (ncoil(i),i=1,5)
c$$$        do 230 i=1,5
c$$$          if(ncoil(i).le.0) go to 230
c$$$C%OS  if(ncoil(i).gt.nccoila) call eqwrng(100)
c$$$          if(ncoil(i).gt.nccoila) go to 299
c$$$          read(10,250) (ccoil(nn,i),nn=1,ncoil(i))
c$$$ 230    continue
c$$$        read(10,250,END=299) (cvac(i),i=1,9)
c$$$ 299    continue
        close(unit=10)


c.......................................................................
c     The convention we expect for the read in epsi from eqdsk
c     within the plasma
c     is that it will have a minimum at the magnetic axis.
c     This is in conformity with usual eqdsk convention. 
c     If psimag.gt.psilim (from an off-brand eqdsk), 
c         we reverse the sign of epsi,psilim,psimag. 
c     The sign of the current, toteqd, will be used
c     to specify the current  and poloidal B-field directions.
c     (Positive is CCW, viewed from above.)
c     Keep sign of current in cursign.
c
c     HOWEVER, at the end of this subroutine, and throughout
c     cql3d, we use epsi with reversed sign, i.e., epsi is maximum
c     at the magnetic axis.  The bi-cubic splines of epsi use
c     this latter convention.
c
c     Also, absolute value of fpsi and btor is effectively taken,
c     and the sign of Bphi is maintained through bsign.
c
c     We have:
c        vector-B = 
c        bsign*fpsi_cql*grad(phi) - cursign*grad(phi) X grad(epsi_cql)
c
c.......................................................................

        if (psimag.gt.psilim) then ! reverse to make minimum at m.axis
           do j=1,nnz
              do i=1,nnr
                 epsi(i,j)=-epsi(i,j)
              enddo
           enddo
           psimag=-psimag
           psilim=-psilim ! now psimag < psilim

           write(*,1000)
 1000      format(//,1x,'WARNING: Sign of epsi,psilim,psimag reversed')
           
        endif

        cursign=sign(one,toteqd)
         
c..................................................................
c     Create the r,z meshes..
c     Assumes ymideqd=0.
c..................................................................
         
        !if(ymideqd.ne.zero) stop 'equilib: Consider effect ymideqd.ne.0'
        !YuP[04-2017] moved the above stop to few lines below.
        ! In case of eqsym=avg_zmag (or top, or bottom) the value of ymideqd
        ! will be redefined, so the code should not be stopped here.
        
        dzz=zbox/(nnz-1)
        drr=rbox/(nnr-1)
        er(1)=rboxdst
        ez(1)=-zbox*.5
        do nn=2,nnr
           er(nn)=er(nn-1)+drr
        enddo
        do nn=2,nnz
           ez(nn)=ez(nn-1)+dzz
        enddo
        
c.......................................................................
c     Up-down symmetrize the eqdsk about z=0, as specified by  eqsym. 
c     (this has no effect if the equilibrium is initially 
c     up-down symmetric about zmag=0.).
c     New options beyond eqsym.eq."average" added (BobH: 020606).
c.......................................................................
        
        if (eqsym.eq."none") then
           if(ymideqd.ne.zero) 
     +        stop 'equilib/eqsym=none: Consider effect ymideqd.ne.0'
           zshift=0.d0
c          Do nothing.
        else if (eqsym.eq."average") then
           if(ymideqd.ne.zero) 
     +        stop 'equilib/eqsym=average: Consider effect ymideqd.ne.0'
           zaxis=0.d0 ! YuP[2014] added
c          Average psi values above and below z=0. This is nmlst deflt.
           do j=1,nnz/2
              do i=1,nnr
                 epsi(i,j)=0.5*(epsi(i,j)+epsi(i,nnz-(j-1)))
              enddo
           enddo
           do j=nnz-nnz/2+1,nnz
              do i=1,nnr
                 epsi(i,j)=epsi(i,nnz+1-j)
              enddo
           enddo
        else                 ! eqsym.eq.(avg_zmag, top, or bottom)
c          Symmetrizing step:
c
c          Expand vertical height of eqdsk by amount 2*(zaxis-ymideqd),
c          making midplane of computational grid at the magnetic axis.
c    
c          Interpolate psi onto new expanded grid.

           zshift=zaxis-ymideqd
           zaxis=0.d0
           ymideqd=0.d0
           if(ncontr.gt.1) then ! YuP[04-2017] Added: adjust LCFS, if available
             ncontr_= min(ncontr,lfield)
             do ilim=1,ncontr_
                zcontr(ilim)=zcontr(ilim)-zshift
             enddo
           endif
           if(nlimiter.gt.1) then ! YuP[04-2017] Added: adjust plasma border
             nline= min(nlimiter,lfield)
             do ilim=1,nline
                zlimiter(ilim)=zlimiter(ilim)-zshift
             enddo
           endif
           zdim_ex=zbox+2.*abs(zshift)
           dr_ex=drr
           dz_ex=zdim_ex/(nnz-1)
           if (zshift.ge.0.d0) then
              dummyaz(1)=ez(1)
           else
              dummyaz(1)=ez(1)-2.*abs(zshift)
           endif
           do ix=1,nnr
              dummyar(ix)=er(ix)
           enddo
c          Gives a shifted, expanded z-array:
           do iz=2,nnz
              dummyaz(iz)=dummyaz(iz-1)+dz_ex
           enddo
           
c          Interpolating:
           kz1=1
           kz2=2
           do 350  j=1,nnz
              zval=dummyaz(j)
 360          if (zval.le.ez(kz2).or.kz2.eq.nnz) go to 365
              
              kz1=kz1+1
              kz2=kz1+1
              go to 360
c     
 365          continue
              kr1=1
              kr2=2
              do 370  i=1,nnr
                 rval=dummyar(i)
 380             if (rval.le.er(kr2).or.kr2.eq.nnr) go to 385
                 
                 kr1=kr1+1
                 kr2=kr1+1
                 go to 380
c     
 385             continue
                 f1eq=epsi(kr1,kz1)+(rval-er(kr1))*
     1                (epsi(kr2,kz1)-epsi(kr1,kz1))/(er(kr2)-er(kr1))
c     
                 f2eq=epsi(kr1,kz2)+(rval-er(kr1))*
     1                (epsi(kr2,kz2)-epsi(kr1,kz2))/(er(kr2)-er(kr1))
c     
                 val=f1eq+(zval-ez(kz1))*(f2eq-f1eq)/(ez(kz2)-ez(kz1))
                 dummypsi(i,j)=val
 370          continue
 350       continue
           
cBH031010:  Following changes do nothing, since xma_ex,rdim_ex
cBH031010:    not used.
cBH031010           xma_ex=xma
           xma_ex=raxis
cBH031010           rdim_ex=xdimeqd
           rdim_ex=rbox
           zmid_ex=0.d0
           zma_ex=0.d0
           
c     Redefine the dummyaz-grid so midplane at z=0.:
           dummyaz(1)=-zdim_ex/2.
           do i=2,nnz
              dummyaz(i)=dummyaz(i-1)+dz_ex
           enddo
           
c          Redefine the eqdsk quantities:
           zbox=zdim_ex
           zaxis=0.d0
cBH031010           dzz=dz_ez  !No effect, since not used here
cBH031010                      !and it is redefined below.
           do i=1,nnz
              ez(i)=dummyaz(i)
           enddo
           
           if (eqsym.eq."avg_zmag") then
c          Up-down symmetrize
              do i=1,nnr
                 j=nnz/2+1
                 epsi(i,j)=dummypsi(i,j)
              enddo
              do j=1,nnz/2
                 do i=1,nnr
                    epsi(i,j)=0.5*(dummypsi(i,j)+dummypsi(i,nnz-(j-1)))
                 enddo
              enddo
              do j=nnz-nnz/2+1,nnz
                 do i=1,nnr
                    epsi(i,j)=epsi(i,nnz+1-j)
                 enddo
              enddo

           else if (eqsym.eq."top") then
              STOP 'Need to fix eqsym=top'
c          Reflect top to bottom    
              do i=1,nnr
                 j=nnz/2+1
                 epsi(i,j)=dummypsi(i,j)
              enddo
              do j=nnz-nnz/2+1,nnz
                 do i=1,nnr
                    epsi(i,j)=dummypsi(i,j)
                 enddo
              enddo
              do j=1,nnz/2
                 do i=1,nnr
                    epsi(i,j)=dummypsi(i,nnz-(j-1))
                 enddo
              enddo
            
           else if (eqsym.eq."bottom") then
              STOP 'Need to fix eqsym=bottom'
c          Reflect bottom to top
              do i=1,nnr
                 j=nnz/2+1
                 epsi(i,j)=dummypsi(i,j)
              enddo
              do j=1,nnz/2
                 do i=1,nnr
                    epsi(i,j)=dummypsi(i,j)
                 enddo
              enddo
              do j=nnz-nnz/2+1,nnz
                 do i=1,nnr
                    epsi(i,j)=dummypsi(i,nnz-(j-1))
                 enddo
              enddo

           endif

        endif  !On eqsym
        
        
                 
c.......................................................................
c     Determine the equally spaced psi array.
c     Note: psilim, psimag and epsi will have their sign changed below.
c     psiar is ordered from edge to magnetic axis.
c.......................................................................

        nfp=nnv 
        ! Up to now, psimag < psilim;  epsi has min. at m.axis
        delpsi=1.e+8*(psilim-psimag)/(nnv-1)  ! positive
        do 10 ix=1,nnv
c%OS  bug?!
c%OS  psiar(ix)=1.e+8*psilim+(ix-1)*delpsi
          psiar(ix)=-1.e+8*psilim+(ix-1)*delpsi ! reversed, and mks->cgs
 10     continue

      endif  !On eqsource

      if (eqsource.eq."eqdsk" .or. eqsource.eq."tsc") then

c.......................................................................
c    Convert to cgs, re-order fpsiar,prar,ppar,qar from edge to mag axis
c    Reverse sign of epsi,psilim,psimag in the code, so max psi will
c    occur on axis.
c.......................................................................

         do i=1,nnv
            fpsiar(i)=fpsiar(i)*1.e+6
            ffpar(i)=ffpar(i)/1.e+2
            prar(i)=prar(i)*10.
            ppar(i)=ppar(i)/1.e+7
         enddo
         call dcopy(nnv,fpsiar,1,dummyar,1)
         do i=1,nnv
            fpsiar(i)=dummyar(nnv+1-i)
         enddo
         call dcopy(nnv,ffpar,1,dummyar,1)
         do i=1,nnv
            ffpar(i)=dummyar(nnv+1-i)
         enddo 
         call dcopy(nnv,prar,1,dummyar,1)
         do i=1,nnv
            prar(i)=dummyar(nnv+1-i)
         enddo 
         call dcopy(nnv,ppar,1,dummyar,1)
         do i=1,nnv
            ppar(i)=dummyar(nnv+1-i)
         enddo
         call dcopy(nnv,qar,1,dummyar,1)
         do i=1,nnv
            qar(i)=dummyar(nnv+1-i)
         enddo
         psilim=-psilim*1.e8  ! reversed !
         psimag=-psimag*1.e8  ! reversed !
         do 5 j=1,nnz
            do 6 i=1,nnr
               epsi(i,j)=-epsi(i,j)*1.e8  ! reversed !
 6          continue ! now psimag > psilim;  epsi has min. at edge !
 5       continue
         write(*,1000)
         ! 1000      format(//,1x,'WARNING: Sign of epsi,psilim,psimag reversed')
 
         btor=btor*1.e+4
         rbox=rbox*1.e+2
         zbox=zbox*1.e+2
         rboxdst=rboxdst*1.e+2
         radmaj=radmaj*1.e+2
         toteqd=toteqd*3.e9
         zshift=zshift*1.e+2
         
         psi_lim = psilim ! cgs
         psi_mag = psimag ! cgs
         R_axis  = raxis*1.e+2 ! cgs
         Z_axis  = zaxis*1.e+2 ! cgs
         
         do ilim = 1,nlimiter
            rlimiter(ilim)= rlimiter(ilim)*1.d2
            zlimiter(ilim)= zlimiter(ilim)*1.d2
         enddo
         do ilim = 1,ncontr
            rcontr(ilim)= rcontr(ilim)*1.d2
            zcontr(ilim)= zcontr(ilim)*1.d2
         enddo

c..................................................................
c        Set initial values of rmag,zmag 
c        (to be refined in eqrhopsi for bicubic splines of epsi).
c..................................................................

         rmag=raxis*1.e+2
         zmag=zaxis*1.e+2

c..................................................................
c     Re-Create the z,r meshes..
c     Assumes ymideqd=0.
c..................................................................
         
         if(ymideqd.ne.zero) stop 'equilib:Consider effect ymideqd.ne.0'
         dzz=zbox/(nnz-1) ! cgs
         drr=rbox/(nnr-1) ! cgs
         er(1)=rboxdst    ! cgs
         ez(1)=-zbox*.5
         do nn=2,nnr
            er(nn)=er(nn-1)+drr
         enddo
         do nn=2,nnz
            ez(nn)=ez(nn-1)+dzz
         enddo
         
         ezmin=ez(1)   ! cgs
         ezmax=ez(nnz)
         ermin=er(1)
         ermax=er(nnr)
                  
c..................................................................
c     Set up spline array for the f =(R*BTOR) subroutine,etc.
c..................................................................
         
         i1p(1)=4
         i1p(2)=4
         call coeff1(nnv,psiar,fpsiar,d2fpsiar,i1p,1,workk)
         call coeff1(nnv,psiar,ffpar,d2ffpar,i1p,1,workk)
         call coeff1(nnv,psiar,prar,d2prar,i1p,1,workk)
         call coeff1(nnv,psiar,ppar,d2ppar,i1p,1,workk)
         call coeff1(nnv,psiar,qar,d2qar,i1p,1,workk)
         

         ibd(1)=4
         ibd(2)=4
         ibd(3)=4
         ibd(4)=4
         call coeff2(nnr,er,nnz,ez,epsi,epsirr,epsizz,epsirz,nnra,ibd,
     1      wkepsi)
     
c..................................................................
c     Adjust z-height of diagnostics, if have shifted eqdsk array
c..................................................................
       
         if (zshift.ne.zero) then
c           X-ray detector height
            do 500 nn=1,nv
              if (z_sxr(nn).ne.0.) z_sxr(nn)=z_sxr(nn)-zshift
 500        continue
c           Neutral beam pivot point height adjusted in frnfreya.
c           Must also shift URF ray data, after it is read in.
         endif

      elseif (eqsource.eq."topeol") then
        call eqtopeol
      else
         call eqwrng(10)
      endif

      write(*,*)
      write(*,598) rmag,zmag,psimag,psilim,qar(1),qar(nnv)
598   format('ADJUSTED (possible sign change; rescaled) [cgs]: 
     +rmag,zmag,psimag,psilim,qar(1),qar(nnv)=',  6(1pe12.4))
      write(*,'(a,e12.3,2f5.1)')' btor,bsign,cursign',btor,bsign,cursign
      write(*,*)'equilib/setup: done'
      !pause


       
      else                      ! index.ne.0 !==========================
         
c..................................................................
c     Find  BR, BPHI, BZ for a given Requil,Zequil on equil. grid
c..................................................................
         der=er(2)-er(1) ! cgs
         dez=ez(2)-ez(1) ! cgs
         
         PSIequil=t2(der,dez,Requil,Zequil,nnr,er,nnz,ez,epsi,
     1               epsirr,epsizz,epsirz,nnra,0,0)           ! cgs
         dpsidr=  t2(der,dez,Requil,Zequil,nnr,er,nnz,ez,epsi,
     1               epsirr,epsizz,epsirz,nnra,1,0)
         dpsidz=  t2(der,dez,Requil,Zequil,nnr,er,nnz,ez,epsi,
     1               epsirr,epsizz,epsirz,nnra,0,1)
c..................................................................
c     Determine f(psi)
c..................................................................
         
         itab(1)=1
         itab(2)=0
         itab(3)=0
         dpsiar=psiar(2)-psiar(1) ! cgs
         call terp1e(nfp,psiar,fpsiar,d2fpsiar,PSIequil,
     ~               1,tab,itab,dpsiar)
         ! YuP Note: The sign of fpsiar was adjusted
         ! during setup by using bsign (see line ~237 above) 
         fpsieq=tab(1)
         
c..................................................................
c     Determine components of B.
c..................................................................
         
         BReq=  -cursign*dpsidz/Requil     ! cgs
         BPHIeq= bsign*fpsieq/Requil ! change to original sign of Bphi
         BZeq=   cursign*dpsidr/Requil

c         if(abs(Zequil+0.8).le.1.e-2) then
c          write(*,'(a,4e12.4)')'equilib',Requil,Zequil,PSIequil,BZeq
c         endif
         
      endif                     ! endif on index !======================

      return
      end
