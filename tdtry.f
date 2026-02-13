c
c
      subroutine tdtry
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'


c.......................................................................
c     Called by micxinit, which is called repeatedly from tdinit with
c     ll=lrors,1,-1;  tdnflxs(ll).
c
c     This routine computes the y (pitch angle) mesh
c     from the normalized mesh determined in tdtrmuy or wptrmuy.
c     In general the y mesh will vary from flux surface to flux surface.
c.......................................................................

      if (eqsym.eq."none") then  !Only if eqmod.eq."enabled",
                                 !        eqsource="eqdsk"
         ilzhfs=lz_bmax(lr_)
      else
         ilzhfs=lz
      endif
      if (cqlpmod.eq."enabled" .and. numclas .eq. 1) ilzhfs=lz/2+1
      !YuP[2025-12-12] added cqlpmod.eq."enabled" in the above line     

      !if (ioutput(1).ge.2) then !YuP[2020] diagnostic printout
      !write(*,*)'tdtry:  ipacktp =',ipacktp
      !endif
      !write(*,*)'tdtry: lz,lz_bmax(lr_)=',lz,lz_bmax(lr_)

      if (cqlpmod .ne."enabled") then
        zsntrp2=1./bbpsi(ilzhfs,lr_)
cBH070419        ipacktp=3
cBH070419 NOTE:  ipacktp now set in tdtrmuy, to coordinate with iytr.
cBH070419        Now can be 0 or 3 here, depending on tfac sign.
      else ! (cqlpmod.eq."enabled")
        zsntrp2=psis(l_)/bbpsi(ilzhfs,lr_)
cBH070419        ipacktp=0
cBH070419 NOTE:  ipacktp now set in tdtrmuy, to coordinate with iytr.
      endif
      thb(l_)=asin(sqrt(zsntrp2))
      iadd=0
      iu=0
      mark=0

c..............................................................
c     constant mu mesh first.
c..............................................................

      if (meshy.eq."fixed_mu") then
        do 10 i=1,iymax/2
          if (cqlpmod .ne."enabled") then
            sinsq=mun(i)*bmidplne(lr_)/bmidplne(lrzmax)
          else ! (cqlpmod.eq."enabled")
            sinsq=mun(i)*psis(l_)
          endif
          if (sinsq.ge.1.) then
            if(mark.eq.0 .and. cqlpmod.ne."enabled") then
              call tdwrng(8)
            else
              it=i-1
              iyh_(l_)=it+ipacktp
              iyh=it+ipacktp
              go to 11 !-> exit i loop
            endif
          elseif (sinsq.ge.zsntrp2 .and. mark.eq.0) then
            if (cqlpmod .ne. "enabled") then
              if (sinsq.eq.zsntrp2) call tdwrng(9)
              iu=i-1
            else ! (cqlpmod.eq."enabled")
              iu=i-1
            endif
            mark=1
            iadd=ipacktp
          endif
          ix=i+iadd
          idx(i,l_)=ix
          y(ix,l_)=asin(sqrt(sinsq))
 10     continue !  i=1,iymax/2
        if (mark.eq.0) call tdwrng(8)
        it=iymax/2-ipacktp
        iyh_(l_)=iymax/2
        iyh=iyh_(l_)
 11     continue
        iyy=2*iyh   !-YuP-101215: Don't use iy=; it's in common /params/
                    ! Don't let overwrite the cqlinput value!
        iy_(l_)=iyy
        !write(*,*) 'tdtry: iyh,iyy=',iyh,iyy
cBH070419        if (cqlpmod .ne. "enabled") then
cBH070419          itl=iu+2
cBH070419        else
cBH070419          if (iu .eq. 0) iu=iyh
cBH070419          itl=iu
cBH070419        endif
        if (ipacktp.eq.3) then
           itl=iu+2
        elseif (ipacktp.eq.0) then
           if (iu .eq. 0) iu=iyh
           itl=iu
        else
           WRITE(*,*)'tdtry:  ipacktp =',ipacktp
           WRITE(*,*)'STOP in tdtry:  Check ipacktp'
           stop
        endif
        itl_(l_)=itl
        itu=iyy+1-itl
        itu_(l_)=itu

c..............................................................
c     Now for the fixed theta mesh case.
c..............................................................

      elseif (meshy.eq."fixed_y") then
        iyh=iyh_(lrors)
        iyh_(l_)=iyh
        iyy=iymax  !-YuP-101215: Don't use iy=; it's in common /params/
                       ! Don't let overwrite the cqlinput value!
        iy_(l_)=iyy
        do 13 i=1,iytr(lrors)/2
          if (mun(i).gt.thb(l_) .and. mark.eq.0) then
            if (cqlpmod.ne."enabled") iu=i-1
            if (cqlpmod.eq."enabled") iu=i-1
            mark=1
            iadd=ipacktp
          endif
          ix=i+iadd
          y(ix,l_)=mun(i)
          idx(i,l_)=ix
 13     continue
cBH070419        if (cqlpmod .ne. "enabled") itl=iu+2
cBH070419        if (cqlpmod .eq. "enabled") itl=iu
        if (ipacktp.eq.3) then
           itl=iu+2
        elseif (ipacktp.eq.0) then
           if (iu .eq. 0) iu=iyh
           itl=iu
        else
           WRITE(*,*)'tdtry:  ipacktp =',ipacktp
           WRITE(*,*)'STOP in tdtry:  Check ipacktp'
           stop
        endif

        itl_(l_)=itl
        itu=iyy+1-itl
        itu_(l_)=itu
        if (mark.eq.0) call tdwrng(10) !iactst=abort forces stop. error>1.e-8

      elseif(meshy.eq."free")then !Not called with this value; just exit
        return
      endif

      iytr(l_)=iyy-2*ipacktp
      iyjx=iyy*jx
      iyjx_(l_)=iyjx

c..............................................................
c     Make sure the passed/trapping interface is between two meshpoints.
c..............................................................

cBH070419      if (cqlpmod .ne. "enabled") then
      if (ipacktp.eq.3) then
        diff1=thb(l_)-y(iu,l_)
        if (diff1.lt.tbnd(l_)) tbnd(l_)=diff1/3.
        ytop=y(iu+4,l_)
        if (iyh.eq.itl+1) ytop=pi/2.
        diff2=ytop-thb(l_)
        if (diff2.lt.tbnd(l_)) tbnd(l_)=diff2/3.
        y(iu+1,l_)=thb(l_)-tbnd(l_)
        y(iu+2,l_)=thb(l_)
        y(iu+3,l_)=thb(l_)+tbnd(l_)
      endif

cBH070419 for tfac.lt.0., have set ipacktp=0, i.e., pack zero additional
cBH070419 y-mesh points around the t-p boundary.   Will simply adjust
cBH070419 a y-mesh point to be at t-p boundary (not sure this is
cBH070419 essential) and space surrounding mesh points equidistant
cBH070419 away.   This is to give same velocity and radial transport
cBH070419 related y-meshes.
cBH070419 The y-meshes will not be identical at t-p bndry, but
cBH070419 will ignore this for now [Probably have to have y(itl)
cBH070419 exactly at thb(l_), but not certain. Check this out later....]

      if (cqlpmod .ne. "enabled" .and. ipacktp.eq.0) then

         deltay1=abs(thb(l_)-y(iu,l_))
         deltay2=abs(y(iu+1,l_)-thb(l_))
         if (deltay1.le.deltay2) then
            y(itl,l_)=thb(l_)
            y(itl-1,l_)=y(itl,l_)-deltay2  ! i.e., y(itl+/-1) equidistant
         else
            y(itl+1,l_)=thb(l_)
            y(itl+2,l_)=y(itl+1,l_)+deltay1
            itl=itl+1                !reset itl,itu
            itl_(l_)=itl
            itu=iyy+1-itl
            itu_(l_)=itu
         endif
      endif

c..............................................................
c     Define the mesh on the other side of pi/2
c..............................................................

      do 12 i=1,iyh
        ii=iyy+1-i
        y(ii,l_)=pi-y(i,l_)
 12   continue
      do 14 i=1,iytr(l_)/2
        ii=iytr(lrors)+1-i
        idx(ii,l_)=iy_(l_)+1-idx(i,l_)
 14   continue
      if (l_.eq.1) then
        do 15 i=1,iymax
          idx(i,0)=1
 15     continue
      endif

ccc      write(*,*)'tdtry: l_,iytr(l_),idx()=',
ccc     +                  l_,iytr(l_),idx(1:iytr(l_),l_)


      return
      end
