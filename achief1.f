c
c
      subroutine achief1
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     This routine directs the calculation for lrzmax=1
c..................................................................

      include 'param.h'
      include 'comm.h'

      include 'name.h'
c.......................................................................

c..................................................................
c     Set defaults - for main code + "eq" module.
c..................................................................
      call aindflt
      call eqindflt
      call aindflt1

c.....................................................................
c     Read in driver input namelist setup
c.....................................................................
      open(unit=2,file="cqlinput",status="old") 
         read(2,setup)
         read(2,trsetup)
         read(2,sousetup)
         read(2,eqsetup)
         read(2,rfsetup)
      close(2)

c..................................................................
c     Call routine which finds electron and ion species indices.
c..................................................................

      call ainspec

c.......................................................................
c     set variables dependent on input variables
c.......................................................................

      call ainsetva

c..................................................................
c     Allocate arrays , if required
c..................................................................

      call ainalloc

c.......................................................................
c     print namelists
c.......................................................................

      if (nmlstout.eq."enabled") then
         write(6,*)'  In achief1: '
         write(6,setup0)
         write(6,setup)
         write(6,trsetup)
         write(6,sousetup)
         write(6,eqsetup)
         write(6,rfsetup)
      elseif (nmlstout.eq."trnscrib") then
         write(6,*)'  In achief1: '
         call ain_transcribe("cqlinput")
      else
         write(6,*)
         write(6,*) 'mnemonic = ',mnemonic
         write(6,*)
      endif

c.....................................................................
c     Determine mesh normalization constant vnorm.
c.....................................................................

      call ainvnorm

c.....................................................................
c     Call the initialization routines for the appended modules..
c.....................................................................

      call eqinitl
      call frinitl
      
      open(unit=2,file="cqlinput",delim='apostrophe',status="old") 
      call frset(lrz,noplots,nmlstout,ngen,kfrsou(1:nbeams))   ! Uses unit 2
      close(2)

c..................................................................
c     Call an initialization routine which determines flux surface
c     geometry and magnetic field structure.
c..................................................................

      call aingeom

c.......................................................................
c     Initialize mesh along magnetic field line
c.......................................................................

      if (cqlpmod.eq."enabled" .and. numclas.eq.1 .and. ls.eq.lsmax)then
        lz=lz/2+1
        lsmax=lsmax/2+1
        ls=ls/2+1
      endif

      call micxiniz

      if (cqlpmod.eq."enabled" .and. numclas.eq.1 .and. ls.eq.lsmax)then
        lz=2*(lz-1)
        lsmax=2*(lsmax-1)
        ls=2*(ls-1)
        call wploweq
      endif

c.......................................................................
c     Initialize some plasma parameters
c.......................................................................

      call ainpla

c.......................................................................
c     call a routine to determine meshes y, x and related quantities
c.......................................................................

      call micxinit
      
      ieq_tot=inewjx_(1) ! inewjx_() is defined in micxinit
      ieq_(1)=1 ! Eqn no. at beginning of each flux surface
      ieq_(lrors+1)=ieq_tot ! lrors+1 should be 2 here

c............................................................
c     call main initialization routine.
c............................................................

      call ainitial

      if (nstop.eq.0) then
        call pltmain
        if (ioutput(1).ge.2) then !YuP[2020] diagnostic printout
        write(*,*) 'In ACHIEF1, before call pgend'
        endif
        call pgend
        stop 'achief1: nstop=0'
      endif

c..................................................................
c     Initialize main netCDF write, if netcdfnm.ne."disabled"
c..................................................................

      if (netcdfnm.ne."disabled") then
         call netcdfrw2(0)
      endif

c.......................................................................
c     Solve equations on the flux surface
c.......................................................................

      call tdnflxs(1)
c     Copy current distribution f into f_
      call dcopy(iyjx2*ngen*lrors,f(0,0,1,1),1,f_(0,0,1,1),1)
c     bring background profiles up to time step n
      if(nefiter.eq.1) call profiles
      ! Reset time step if (n+1).eq.nondtr1(i). .AND. LRZMAX=1
      do i=1,ndtr1a
           if ((n+1).eq.nondtr1(i)) then
              dtr=dtr1(i)
              dtreff=dtr
              dttr=dtr*nrstrt
           endif
      enddo
      !-------------------------------------------!
        call achiefn(0) ! get solution for new f. !
      !-------------------------------------------!
      ! Start time advancement:
      if(nefiter.eq.1) then
           n=n+1
           n_(1)=n ! new time-step for this flux surface
           ! for 2-d (v_par,v_perp) calculation ntloop controls
           ! end of run or restart.
           ! Also updates time.
           call ntloop
      endif
      
      call tdnflxs(1)
        call cfpgamma ! Re-calc. Coul.Log for the new distr.func.
        do k=1,ngen  ! Compute density gains and losses, and powers.
           ! For lbdry0='disabled',  Redefine f at v=0 so it is unique:
           ! (For lbdry0='enabled', coeff matrix is set up 
           !   to automatically maintain unicity.)
           if (lbdry0.ne."enabled") then !-YuP: moved here from impavnc0
             call dcopy(iyjx2,f(0,0,k,l_),1,fxsp(0,0,k,l_),1)
             s=0.
             t=0.
             do 2100 i=1,iy
               s=s+vptb(i,lr_)*cynt2(i,l_)
               t=t+vptb(i,lr_)*cynt2(i,l_)*f(i,1,k,l_)
 2100        continue
             do 2200 i=1,iy
               f(i,1,k,l_)=t/s
 2200        continue
           endif
           call diagscal(k) !-> renorm f() if lbdry(k)="scale"
           call coefstup(k) ! To define da...df coeffs, gon(i,j), etc
           call coefmidv(da,1)
           call coefmidv(db,2)
           call coefmidv(dc,3)
           call coefmidt(dd,1)
           call coefmidt(de,2)
           call coefmidt(df,3)
           call coefwtj(k)
           call coefwti(k)
           call diagimpd(k) 
        enddo ! k
        call achiefn(1)  !Compute plasma energy, density and energy transfer

      
      return
      end
