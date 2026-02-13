c
c
      subroutine eqrhopsi(generate)
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
      character*8 generate
      character*8 eqmirror

      common/tmp1/ btor00,bthr00,bmod00
      parameter(nworka=3*nconteqa+1)
      dimension workk(nworka)

c..................................................................
c     This routine determines an array eqrho(eqpsi(k)) where
c     k=1,...,nconteqn, eqpsi is a value of psi (equilibrium
c     psi_poloidal) and eqrho is the associated  radial coordinate
c     (sqrt(toroidal flux),sqrt(area), sqrt(volume),
c     radial extent of flux surface, normalized poloidal flux,
c     or normalized sqrt(pol flux), according to the value
c     of radcoord.  See cqlinput_help.) 
c     The eqpsi array can be provided externally
c     (generate="disabled") or it can be generated within
c     this routine (generate="enabled"). [It is called with
c     generate="enabled" from eqcoord when lr_.eq.lrzmax.]
c     Also, determines some approximate geometric quantities
c     of the eqdsk.
c..................................................................

c......................................................................
c     General, non-circular flux-surface psi values, epsi, have been
c     set up from eqdsk, topeol, or elliptical plasma models, depending
c     on eqmod.ne.'disabled' and eqsource.  epsi, at this point is an
c     decreasing function from the magnetic axis to the plasam edge.
c     For eqmod.ne.'disabled', namelist eqsource gives the source of
c     equilibrium flux surface data, giving flux (epsi) on a regular 
c     cylindrical coordinate R,Z-grid.  Also provided is f=R*B_phi.
c     If eqsource="ellipse", then epsi, R, Z etc are determined
c     from a subroutine eqelpse, and f is determined arbitrarily from
c     an input parameter fpsimodl.
c     In order to solve the orbit equations it will be necessary to
c     know all the derivatives of epsi up to second order. A 2-D
c     spline package will be used. Set up the NCAR spline package.
c     Ordinarily the pol. flux psi value associated with the flux 
c     surface of interest will be labeled by common variable epsicon_.
c
c     091016: Setup of 2D splines and the rmaz,zmag re-calculation 
c             below has been moved up from subroutine eqorbit,
c             to facilitate de-updown-symmetrization.
c
c     Calls to the current subroutine eqrhospi are from eqcoord, in
c     descending order lr_=lrzmax,1,-1.
c......................................................................


c      if (eqcall.eq."enabled") then   !Called on first run through.
      if (lr_.eq.lrzmax) then   !Called on first run through.
        ibd(1)=4
        ibd(2)=4
        ibd(3)=4
        ibd(4)=4
        eqmirror="disabled"
        if( eqsource.eq."eqdsk".and.machine.eq."mirror") then
           eqmirror="enabled"
        else
           eqmirror="disabled"
        endif
        if(eqsource.eq."mirror1" .or. eqmirror.eq."enabled") then
           ! in this case er(nnr) grid starts from R=0, and rmag=0.
           ! The pol.flux function has zero derivative at R=0,
           !i.e. dPSI/dR=0 at any Z.
           epsirr(1,1:nnza)=0.d0 ! at ir=1 (R=0) boundary, for all iz
           ibd(1)=2 ! means: 
           !=2 the first derivative of f(x,y)==epsi(R,Z) with respect
           !to x is given at (x(1),y(j)) and placed into fxx==epsirr(1,:)
        endif

c       2D bicubic spline coeffs of poloidal flux. epsi has sign change
c       after reading eqdsk, so psimag is max psi, psilim is less.
        call coeff2(nnr,er,nnz,ez,epsi,epsirr,epsizz,epsirz,nnra,ibd,
     1    wkepsi)
c..................................................................
c     Now determine the exact location of the magnetic axis. From
c     The point of view of the spline package it is somewhere
c     between er(imag-1) and er(imag+1). Use Newton iteration.
c     imag was determined in subroutine eqrhopsi.
c..................................................................

c        rmag=er(imag)
c        zmag=0.
c        do 5 iter=1,8
c          dpsidr=terp2(rmag,zmag,nnr,er,nnz,ez,epsi,epsirr,epsizz,
c     1      epsirz,nnra,1,0)
c          d2psidrr=terp2(rmag,zmag,nnr,er,nnz,ez,epsi,epsirr,epsizz,
c     1      epsirz,nnra,2,0)
c          rmag=rmag-dpsidr/d2psidrr
c 5      continue
c        psimag=terp2(rmag,zmag,nnr,er,nnz,ez,epsi,epsirr,epsizz,
c     1    epsirz,nnra,0,0)
c        eqpsi(1)=psimag
c      endif


c     Simple search for more accurate rmag, based in bi-cubic spline
c     of the equilibrium data.  Above simple Newton-Raphson failed 
c     for high beta (shifted flux surface) equilibria (bobh, 960309).
c     [BH091017: Maybe could be cured by better starting imag derived
c     from the input equilibrium raxis,zaxis, as below?]
c     It is assumed here that the magnetic axis is at a maximum
c     of psi.   Up-down symmetry (zmag=0.) assumed.
c
c     BH091018:  Should touchup following calc of the magnetic
c                axis with a Newton-Raphson iteration.  Inaccurate
c                magnetic axis location probably limits minimum
c                rya() which can be successfully used.
        rmag_old=rmag ! From eqdsk
        zmag_old=zmag
        psimag_old=psimag
        
        if (eqsym.ne."none") then  !i.e., assuming mag axis at 0.
cBH091017           imag=nnr/2  !nnr set, e.g., by read of eqdsk
           drr=er(2)-er(1)
           dzz=ez(2)-ez(1)
           imag=nint((rmag-er(1))/drr)+1  !nint() is nearest integer
           jmag=nint((zmag-ez(1))/dzz)+1
           if(imag.gt.1)then
             er1=er(imag-1)
             er2=er(imag+1)
             rmag=er(imag-1)
           else ! imag=1 (could be for eqsource="mirror1" 
             er1=er(1)
             er2=er(2)
             rmag=er(1)
           endif
           zmag=0.
           psi1=terp2(rmag,zmag,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1          epsirz,nnra,0,0)
           npoints=101
           dr=(er2-er1)/(npoints-1)
           do i=1,npoints
              erlocal=er1+(i-1)*dr
              psilocal=terp2(erlocal,zmag,nnr,er,nnz,ez,epsi,epsirr,
     1             epsizz,epsirz,nnra,0,0)
              if (psilocal.gt.psi1) then
                 psi1=psilocal
                 rmag=erlocal
              endif
           enddo
           psimag=terp2(rmag,zmag,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1          epsirz,nnra,0,0)
           eqpsi(1)=psimag
           
        else  !i.e.  No symmetrization: Uses entire equilibrium
              !      Perform 2D search for magnetic axis
           drr=er(2)-er(1)
           dzz=ez(2)-ez(1)
           imag=nint((rmag-er(1))/drr)+1  !nint() is nearest integer
           jmag=nint((zmag-ez(1))/dzz)+1
!           
!        write(*,*)'eqrhopsi: er(imag+-1)',er(imag-1),er(imag),er(imag+1)
!        write(*,*)'eqrhopsi: ez(jmag+-1)',ez(jmag-1),ez(jmag),ez(jmag+1)
!        write(*,*)'epsi',
!     +    epsi(imag-1,jmag-1),epsi(imag,jmag-1),epsi(imag+1,jmag-1)
!        write(*,*)'epsi',
!     +    epsi(imag-1,jmag),epsi(imag,jmag),epsi(imag+1,jmag)
!        write(*,*)'epsi',
!     +    epsi(imag-1,jmag+1),epsi(imag,jmag+1),epsi(imag+1,jmag+1)
!
           if(imag.gt.1)then
             er1=er(imag-1)
             er2=er(imag+1)
             rmag=er(imag-1)
           else ! imag=1 (could be for eqsource="mirror1" 
             er1=er(1)
             er2=er(2)
             rmag=er(1)
           endif
           zmag=ez(jmag-1)
           ez1=ez(jmag-1)
           ez2=ez(jmag+1)  ! YuP[2015/05/03] Bug fix: was imag+1
           psi1=terp2(rmag,zmag,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1          epsirz,nnra,0,0)
           npoints=101
           drrr=(er2-er1)/(npoints-1)
           dzzz=(ez2-ez1)/(npoints-1)
           do i=1,npoints
           do j=1,npoints
              erlocal=er1+(i-1)*drrr
              ezlocal=ez1+(j-1)*dzzz
              psilocal=terp2(erlocal,ezlocal,nnr,er,nnz,ez,epsi,epsirr,
     1             epsizz,epsirz,nnra,0,0)
              if (psilocal.gt.psi1) then
                 psi1=psilocal
                 rmag=erlocal
                 zmag=ezlocal
              endif
           enddo
           enddo
           psimag=terp2(rmag,zmag,nnr,er,nnz,ez,epsi,epsirr,epsizz,
     1          epsirz,nnra,0,0)
           eqpsi(1)=psimag
        endif  !On eqsym
        
        if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
        write(*,*)'eqrhopsi: psilim=',psilim
        !write(*,*)'eqrhopsi: ez(jmag+-1)',ez(jmag-1),ez(jmag),ez(jmag+1)
        write(*,'(a,i6,2e13.4)')'eqrhopsi: imag, rmag_old/new =', 
     +                                     imag, rmag_old, rmag
        write(*,'(a,i6,2e13.4)')'eqrhopsi: jmag, zmag_old/new =',
     +                                     jmag, zmag_old, zmag
        write(*,'(a,2e13.4)')'eqrhopsi: psimag_old/new',
     +                                  psimag_old, psimag
        write(*,'(a,e13.4)')'eqrhopsi: epsi(imag,jmag)=',epsi(imag,jmag)
        write(*,'(a,e13.4)')'eqrhopsi: er(imag)=',er(imag)
        write(*,'(a,e13.4)')'eqrhopsi: ez(jmag)=',ez(jmag)
        endif

c      endif  !On eqcall
      endif   !On lr_.eq..rzmax



c..................................................................
c     This routine  is called with generate="enabled" from eqcoord 
c     when lr_.eq.lrzmax.  radcoord is namelist input indicating
c     the specific definition of the radial coordinate.
c..................................................................

      write(*,*)'eqrhopsi: generate,radcoord: ',generate,radcoord

      if (generate.eq."enabled") then     ! generate eqpsi
!        nzc=(nnz-1)/2+1                   ! up-down symmetry assumed
        nzc=jmag
        if(eqsource.eq."ellipse") psilim=0.
!        nrc=nnr/2
        nrc=imag
        amax=epsi(nrc,nzc) ! =psimag
        imag=nrc ! =1 in mirror machine
c       Find jpsimnr such that psi~psilim
        do 1 j=nrc,nnr ! going from imag to larger R, along Z=zmag
          if (epsi(j,nzc).lt.psilim .and. amax.gt.psilim) then
            iminval=j-1
            eqpmn1=epsi(iminval,nzc)
            go to 2 ! done, found: epsi(j,nzc) is outside of psilim
          endif     !      while epsi(j-1,nzc) was inside of psilim
          if (epsi(j,nzc).gt.amax) then
            amax=epsi(j,nzc)
            imag=j
          endif
 1      continue
        iminval=nnr
        eqpmn1=epsi(nnr,nzc)
 2      continue
        jpsimnr=iminval + 1 - 1/(nnr+1-iminval)
        if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
        write(*,*)'eqrhopsi: iminval,jpsimnr=',iminval,jpsimnr
        endif
c
c       Find jpsimnl such that psi~psilim (at the inboard)
        jpsimnl= jpsimnr ! will be over-written in  a tokamak machine
        !(but in a mirror machine, the loop is skipped because nrc=1) 
        do 3 j=nrc-1,1,-1 !going from er(imag-1) to inner smaller er
          jpsimnl=j
          if(epsi(j,nzc).lt.psilim) go to 4
          if (epsi(j,nzc).gt.amax) then
            amax=epsi(j,nzc)
            imag=j
          endif
 3      continue
 4      continue
        eqpsimin=eqpmn1
        eqpsimax=amax

c..................................................................
c     The eqpsi array will be chosen if possible so that it is spaced
c     between the value of psi at the magnetic axis and the value of
c     psi for a flux surface that passes through:
c     the point (R=rmag,Z=radmin) if eqsource="ellipse" or "miller",
c     and psilim if eqsource = "topeol" or "eqdsk" or "mirror1".
c..................................................................

        if ( (eqsource.eq."ellipse")  ) then
          if (ez(nnz).lt.radmin) call eqwrng(9)
          do 5 i=nzc,nnz
            if (ez(i).ge.radmin) go to 6
 5        continue
 6        continue
          ival=i
          z1=ez(ival-1)
          z2=ez(ival)
          psi1=epsi(imag,ival-1)
          psi2=epsi(imag,ival)
          eqpmn2=(psi1-psi2)/(z1-z2)*(radmin-z1)+psi1
          eqpsimin=eqpmn2
          if (eqpmn1.gt.eqpmn2) eqpsimin=eqpmn1
        else if (eqsource.eq."miller") then
          eqpsimin=psilim !YuP[2020-07-09]
          eqpsimax=psimag !YuP[2020-07-09] (Was same as for eqsource.eq."ellipse")
        else if ( (eqsource.eq."eqdsk")  .or. !Includes eqmirror enabled
     +            (eqsource.eq."tsc")    .or.
     +            (eqsource.eq."mirror1")     ) then
c..................................................................
c     If an eqdsk equilibrium or a ONETWO equilibrium is being used
c     we already know psi at the limiter.
c..................................................................
          eqpsimin=psilim
          eqpsimax=psimag
        else if (eqsource.eq."topeol") then
c         Prior to moving calc of 2D spline of epsi to beginning of 
c         this subroutine, following call appears to be a bug (BH091016).
c         [There must have been some prior rearrangement.]
          if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
          write(*,*)'eqrhopsi: rmaxcon,zmag,rmag =',rmaxcon,zmag,rmag
          endif
          psilim=terp2  (rmaxcon,zmag,nnr,er,nnz,ez,epsi,epsirr,
     1      epsizz,epsirz,nnra,0,0)
          eqpsimin=psilim
          psimag=terp2(rmag   ,zmag,nnr,er,nnz,ez,epsi,epsirr,
     1      epsizz,epsirz,nnra,0,0)
          eqpsimax=psimag
        endif

c..................................................................
c     Determine the array of psi values..
c     (Positioned on the epsi grid if nconteq.eq."psigrid",
c     equally spaced in psi if nconteq.ne."psigrid" and
c     nconteqn is a positive number)
c..................................................................
        eqpsi=eqpsimin !initialize (not for all indexes eqpsi is set below)
        if (nconteq .ne. "psigrid") then
c         Equispaced from Mag axis to psilim
c         (Could be problem in eqorbit if psilim surface has separatrix.
c          Could add a psifactr here, or improve eqorbit LCFS calc. BH).
c          Alternatively, we pull back eqpsi(nconteqn) slightly, as in
c          the nconteq.eq."psigrid" case.
          delpsi=(eqpsimax-eqpsimin)/(nconteqn-1)
          do 10 j=1,nconteqn
            eqpsi(j)=-(j-1)*delpsi+eqpsimax !from eqpsimax down to eqpsimin
 10       continue
          ! Adjust the last point:
!          eqpsi(nconteqn)=eqpsi(nconteqn-1)+
!     +                    0.9*(eqpsi(nconteqn)-eqpsi(nconteqn-1))
          !YuP: which is same as 
          eqpsi(nconteqn)=eqpsi(nconteqn-1)-0.9*delpsi
          !So, instead of reducing by the whole delpsi, 
          ! we reduce by 0.9*delpsi, to stay away from eqpsimin
        else ! nconteq = "psigrid"
c         This calc of eqpsi values chooses eqpsi(nconteqn) slightly
c         inside the psilim flux surface.
          nconteqn=0
          !for a mirror machine imag=1, so the whole range 1:nnr can be used
          do 11 j=imag,nnr 
            ! as soon as epsi(j,nzc) got outside of psilim, then:
            if (epsi(j,nzc).le. eqpsimin) then 
              !YuP[03-2016]  changed to .le. (was .lt.) see notes below.
              nconteqn=nconteqn+1
              !!!eqpsi(nconteqn)=eqpsimin+(eqpsi(nconteqn-1)-eqpsimin)*.1
              !YuP[03-2016] The above line appears to have a flaw:
              !(Note: eqpsimin=psilim, and epsi(j,nzc) and eqpsi()
              ! are descending with j).
              !It can happen that at previous step, 
              ! eqpsi(nconteqn-1) was exactly equal to eqpsimin.
              ! In this case, at present step, eqpsi(nconteqn)=eqpsimin
              ! and so we got two points with same value.
              !Correction made: changed .lt. to .le.,
              ! and the definition for the last point:
              if(     (eqpsi(nconteqn-1)-eqpsimin)  .gt.
     +            0.5*(eqpsi(nconteqn-2)-eqpsi(nconteqn-1)) )then
                !The previous point was far enough from psilim;
                !Form the new (last) point, slightly inside psilim :
                del_psi= eqpsi(nconteqn-2)-eqpsi(nconteqn-1)
                !Note: del_psi is positive (because eqpsi is descending)
                ![this was corrected on 2017-12-05; 
                !the wrong sign got from 03-2016/cql3d-mirror-version]
                eqpsi(nconteqn)= eqpsimin+del_psi*0.1
              else 
                !The previous point too close from psilim;
                !Do not form the new point; 
                !consider the previous points as the last :
                nconteqn=nconteqn-1 ! un-do the increment done above.
              endif  
c              write(*,*)'eqrhopsi: nconteqn,eqpsi(nconteqn-1),eqpsimin',
c     +                             nconteqn,eqpsi(nconteqn-1),eqpsimin
              go to 12 ! done, quit the loop
            endif
            eqpsi(j-imag+1)=epsi(j,nzc)
            !So, epsi(imag,nzc)   --> eqpsi(1)
            !    epsi(imag+1,nzc) --> eqpsi(2)
            !    epsi(imag+2,nzc) --> eqpsi(3)
            !    ..... until epsi(j,nzc) .le. eqpsimin 
            !         (got onto or outside of LCFS)
            !          Then, eqpsi(nconteqn)= eqpsimin+del_psi*0.1
            nconteqn=j-imag+1
 11       continue
 12       continue
        endif ! nconteq .ne. "psigrid"
        
        ! Check that eqpsi is strictly descending:
        do j=2,nconteqn
          if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
          write(*,'(a,i4,e16.8)')' j,eqpsi(j)-psilim=',j,eqpsi(j)-psilim
          endif
          if(eqpsi(j)-eqpsi(j-1).ge.0.d0)then
          WRITE(*,*)'eqrhopsi: eqpsi is not descending for j,j-1=',j,j-1
             stop
          endif
        enddo
        
      endif  ! On generate
      
c..................................................................
c     Determine the volume,... of the flux surface.
c..................................................................

      eqvol(1)=0.
      eqarea(1)=0.
      do 20 j=2,nconteqn
        if (j.eq.2) then
          if (eqsource.ne."ellipse") then
            eqcall="enabled"
          else if (eqsource.eq."ellipse" .and. n.eq.0) then
            eqcall="enabled"
          else
            eqcall="disabled"
          endif
        else
          eqcall="disabled"
        endif
        epsicon_=eqpsi(j)

        if(machine.eq."toroidal")then
        
        rstart=0.d0 ! OUTPUT, when kopt=1
        zstart=zmag ! OUTPUT, when kopt=1
        call eqorbit(1,epsicon_,rstart,zstart) ! kopt=1
        !YuP[2020-06-30] Added kopt=2, which allows tracing surface
        ! directly from point (rstart,zstart) when it is given in INPUT
        ! (in this case value of epsicon_ is not needed).
        ! For the original design, use kopt=1,
        ! which means: find the starting point from knowledge of epsicon_ 
        ! and zmag coordinate (stored in comm.h).
        elseif(machine.eq."mirror")then
!          write(*,*)'j=', j,
!     +       ' eqrhopsi: calling eqorbit_mirror with epsicon_[cgs]=',
!     +         epsicon_
          STOP 'machine=mirror not available in this version'
          !call eqorbit_mirror(epsicon_)
        else
        !WRITE(*,*)'calling eqorbit: machine="toroidal" or "mirror" only'
        WRITE(*,*)'calling eqorbit: machine="toroidal" '
        stop
        endif
        
        eqfopsi(j)=fpsi_
        eqrpcon(j)=rpcon_
        eqrmcon(j)=rmcon_
        eqzpcon(j)=zpcon_
        eqzmcon(j)=zmcon_
        eqorb="disabled"
        call eqvolpsi(epsicon_,volum,areac)
        eqvol(j)=volum
        eqarea(j)=areac
c..................................................................
c     Compute <1./R**2>, <1./R>
c..................................................................

        call eqonovrp(epsicon_,onovrp1,onovrp2)
        eqovrp(j,1)=onovrp1
        eqovrp(j,2)=onovrp2
c
        zmaxpsi_=0.
        do 50 l=2,lorbit_
          zmaxpsi_=zmaxpsi_+(es_(l)-es_(l-1))
     1      /(.5*bpsi_(l)+.5*bpsi_(l-1))
 50     continue
        if (j.eq.nconteqn) then
          do 60 l=1,lorbit_
            tlorb1(l)=eqbpol_(l)**2
 60       continue
          call eqflxavg(epsicon_,tlorb1,bpolsqa_,flxavgd_)
          bpolsqlm=bpolsqa_/bmidplne_*zmaxpsi_/eqdells_
          bpolsqlm=bpolsqlm**2
        endif
        q_(j)=2.*fpsi_*onovrp2*zmaxpsi_/(twopi*bmidplne_)
c        write(*,*)'eqrhopsi: j,eqvol(j),eqrmcon(j),eqrpcon(j),q_(j) =',
c     +                       j,eqvol(j),eqrmcon(j),eqrpcon(j),q_(j)
 20   continue ! j=2,nconteqn
 
      eqrpcon(1)=rmag
      eqrmcon(1)=rmag
      if(rmag.gt.em8)then ! rmag=0 in a mirror machine
        eqovrp(1,1)=1./rmag
        eqovrp(1,2)=1./rmag**2
      endif
      call eqfpsi(eqpsimax,fpsi_,fppsi_)
      eqfopsi(1)=fpsi_

c..................................................................
c     rmaxcon=point of intersection of "largest" flx surface with
c     midplane at low B field side; rmincon analogous for high field.
c     BH140122:  This calculation could be improved upon, since there
c     BH140122:  there are sometimes problems as the LCFS is
c     BH140122:  approached.  For example, consider ONETWO type calc
c     BH140122:  of the LCFS.
c..................................................................

      rmaxcon=rpcon_
      rmincon=rmcon_
      if(machine.eq."mirror") rmincon=-rmaxcon ! YuP[03-2016] ok? 
      if(machine.eq."mirror") rmincon=0.1*er(2) ! YuP[03-2016] ok? 
      if(machine.eq."mirror") rmincon=0.d0 ! YuP[03-2016] ok? 
c      write(*,*)'eqrhospi: rmincon,rmaxcon',rmincon,rmaxcon

c..................................................................
c     Determine the toroidal flux array, eqrho.
c     Also determine the area associated with the flux surface.
c..................................................................

      eqrho(1)=0.0
      if (radcoord.eq."sqtorflx") then
         do 30 j=2,nconteqn
            dvolum=(eqvol(j)-eqvol(j-1))
            eqrph=(eqovrp(j,2)+eqovrp(j-1,2))*.5
            eqfh=(eqfopsi(j)+eqfopsi(j-1))*.5
            eqrho(j)=eqrho(j-1)+eqrph*dvolum*eqfh/pi*0.5
 30      continue
         do 40 j=2,nconteqn
            eqrho(j)=sqrt(eqrho(j)/pi/btor)
 40      continue

      elseif (radcoord.eq."sqarea") then
         do 31 j=2,nconteqn
            eqrho(j)=sqrt(eqarea(j)/pi)
 31      continue

      elseif (radcoord.eq."sqvol") then
         do 32 j=2,nconteqn
            eqrho(j)=sqrt(eqvol(j)/(2.*pi**2*rmag))
 32      continue

      elseif (radcoord.eq."rminmax") then
         do 33 j=2,nconteqn
            eqrho(j)=0.5*(eqrpcon(j)-eqrmcon(j))
 33      continue

      elseif (radcoord.eq."polflx") then
         do j=2,nconteqn
            eqrho(j)=(eqpsi(j)-psimag)/(psilim-psimag)
         enddo

      elseif (radcoord.eq."sqpolflx") then
         do j=2,nconteqn
            eqrho(j)=(eqpsi(j)-psimag)/(psilim-psimag)
            if (eqrho(j).ne.zero) eqrho(j)=sqrt(eqrho(j))
         enddo

      endif

      write(*,*)'eqrhopsi: psilim,psimag',psilim,psimag
      write(*,*)'eqrhopsi: rmag,zmag',rmag,zmag
      
      if (ioutput(1).ge.1) then !YuP[2020] Useful diagnostic printout
      write(*,*)'eqrhopsi: eqpsi(j),j=1,nconteqn',
     +                                    (eqpsi(j),j=1,nconteqn)
      write(*,*)'eqrhopsi: eqrho(j),j=1,nconteqn',
     +                                    (eqrho(j),j=1,nconteqn)
      endif

      rhomax=exlin(eqrho(nconteqn-1),eqrho(nconteqn),
     1  eqpsi(nconteqn-1),eqpsi(nconteqn),eqpsimin)
      radmin=rhomax
      btor00=fpsi_/rpcon_
      bthr00=eqbpol_(1)
      bmod00=sqrt(btor00**2+bthr00**2)
      write(*,*)'eqrhospi: rhomax = ',rhomax

c.................................................................
c     Setup 1D spline coeffs for eqrho (determined according to
c     radcoord) versus eqpsi(1:nconteqn).
c..................................................................

      i1p(1)=4
      i1p(2)=4
c     Change sign of eqpsi, to get asceding order necessary
c     for coeff1.  (Remember this when using the spline coeffs.)
      do j=1,nconteqn
         eqpsi(j)=-eqpsi(j)
      enddo
      call coeff1(nconteqn,eqpsi,eqrho,d2eqrho,i1p,1,workk)
c     Change back eqpsi sign.
      do j=1,nconteqn
         eqpsi(j)=-eqpsi(j)
      enddo

c.................................................................
c     Determine the maximum axial position for the plasma.
c..................................................................
c      write(*,*)'eqrhopsi:solz_(j),j=1,lorbit_', (solz_(j),j=1,lorbit_)
      call aminmx(solz_,1,lorbit_,1,zmincon,zmaxcon,kmin,kmax)
      if (eqsym.ne."none") then
         zmaxcon=max(abs(zmincon),abs(zmaxcon))
         zmincon=-zmaxcon
      endif
      write(*,*)'eqrhospi: zmincon,zmaxcon',zmincon,zmaxcon

C-----------------------------------------------------------------------
c     Determine some approx geometrical factors needed to calculate
c     the aspect ratio and the elongation [Assumes up-down symm.]
c     Could check for non-up-down symmetric case, BH091023.
C-----------------------------------------------------------------------
c     left and right radius of psi=psilim surface at midplane
c     assume magnetic axis is at the midplane of the eqdsk.
c      izmag=nnz/2 + 1
      izmag=jmag
      if(machine.eq."mirror")then
        rgeom1=rmag
        rgeom2=rpcon_
        rgeomp= rpcon_ ! 
        r0geomp= er(2) ! Not relevant, just any number >0
      else
        rgeom1=exlin(er(jpsimnl),er(jpsimnl+1),
     +    epsi(jpsimnl,izmag),epsi(jpsimnl+1,izmag),eqpsimin)
        rgeom2=exlin(er(jpsimnr),er(jpsimnr-1),
     +    epsi(jpsimnr,izmag),epsi(jpsimnr-1,izmag),eqpsimin)
c       minor radius of plasma surface
        rgeomp=0.5 * (rgeom2 - rgeom1)
c       geometrical center of plasma surface
        r0geomp=0.5 * (rgeom1 + rgeom2)
      endif
c
      irc=nnr/2 + 1
      do 100 jz=izmag,nnz
        jzpsimx=jz
        if (epsi(irc,jz+1) .lt. eqpsimin) go to 101
 100  continue
 101  continue
c
      itestl=0
      itestr=0
      jrpsmxl=irc
      jrpsmxr=irc
      do 102 ir=1,nnr-irc
        if (epsi(irc-ir,jzpsimx).lt.eqpsimin .and. itestl.eq.0) then
          jrpsmxl=irc - ir + 1
          itestl=1
        endif
        if (epsi(irc+ir,jzpsimx).lt.eqpsimin .and. itestr.eq.0) then
          jrpsmxr=irc + ir - 1
          itestr=1
        endif
        if (itestl.eq.1 .and. itestr.eq.1) go to 103
 102  continue
 103  continue
c
      jrmaxp=irc
      jzmaxp=jzpsimx
      do 104 jr=jrpsmxl,jrpsmxr
        do 105 jz=jzpsimx+1,nnz
          if (epsi(jr,jz) .lt. eqpsimin) then
            if (jz .le. jzmaxp) go to 106
            jrmaxp=jr
            jzmaxp=jz
            go to 106
          endif
 105    continue
 106    continue
 104  continue

c     not very precise, but used only for diagnostic of plasma elongation
      zgeomp=exlin(ez(jzmaxp-1),ez(jzmaxp),
     +  epsi(jrmaxp,jzmaxp-1),epsi(jrmaxp,jzmaxp),eqpsimin)
c
      return
      end
