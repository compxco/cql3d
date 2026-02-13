
c
c
      subroutine tdnflxs(ll)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c.......................................................................
c     This routine is used when the code jumps from calculations on
c     one flux surface to another. It sets a few mesh and time variables
c.......................................................................
      include 'param.h'
      include 'comm.h'
c

c..................................................................
c     l_  defines the current spatial variable mesh label 
c     (radial or parallel).
c     lr_ defines the effective index of the current flux surface.
c     ls_ defines the effective index of the current orbit along B.
c     lmdpln_ defines the orbit index of midplane (l_=lmdpln_) 
c     on current lr_.
!     In CQL3D case, lmdpln_=radial_index; in CQLP, lmdpln_=1 for any ll
c..................................................................

c
      l_=ll
      lr_=lrindx(l_)
      indxlr_=indxlr(lr_)
      lmdpln_=lmdpln(indxlr_)
      ls_=lsindx(l_)
      indxls_=indxls(ls_)

      itl=itl_(l_)
      itu=itu_(l_)
      n=n_(l_)
      iyy=iy_(l_) !-YuP-101215: Don't use iy=; it's in common /params/
                  ! Don't let overwrite the cqlinput value!
                ! iy=iy_(l_) !TEST
      iyh=iyh_(l_)
      iyjx=iyjx_(l_)
      timet=time_(l_)

c.......................................................................
c     check iy mesh
c.......................................................................

      if (n .gt. 0) then
        if (iyh .ne. iyy/2) call diagwrng(20)
        if (itu .ne. iyy-itl+1) call diagwrng(21)
      endif

      return
      end
