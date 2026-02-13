c
c
      subroutine sigv(icall)
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     sets up calls to sigma-v calculator
c..................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      character*8 icall
      real*8 buff(16) ! local buffer array for mpi

      if (icall.eq."first") then
        call sigalloc
        call sigsetup
c       return, if no nuclear fusion general species
        if (sigmamod.eq.'disabled') return
      endif

      call bcast(fuspwrvt,zero,4)
      call bcast(fuspwrmt,zero,4)
      call bcast(fuspwrv,zero,4*lrors)
      call bcast(fuspwrm,zero,4*lrors)
      call bcast(sigft,zero,4)
      call bcast(sigmt,zero,4)

      do 100  i=1,lrz           ! keep index 'i' - used by mpiworker_i below
CMPIINSERT_MPIWORKER_I
CMPIINSERT_IF_RANK_EQ_MPIWORKER
         call tdnflxs(i)        !-> determine flux surface index lr_
c        Compute reaction rate of general species with self and others.
         call sigv5d
c        Compute reaction rates for "equivalent" maxwellians.
c YuP[06-2016] Why do we need to call this sub again?
c All cases of isigsgv1 are treated inside of sigv5d*.
c Commenting out:
cYuP            if (isigsgv1.eq.1) call sigv5d
CMPIINSERT_ENDIF_RANK
CMPIINSERT_RECV_FUS
CMPIINSERT_SEND_FUS
 100  continue                  ! on i=1,lrz

CMPIINSERT_BARRIER 
CMPIINSERT_BCAST_FUS

c
c         Four fusion reactions are treated (see cqlinput_help)
c         fuspwrv = fusion power density (from general species)
c         fuspwrm = fusion power density (Maxwellian with same density
c                                         energy as the gen. species)
      do ll=1,lrz
          call tdnflxs(ll)
          do 90 knumb=1,4
            fuspwrvt(knumb)=fuspwrvt(knumb)
     +        +fuspwrv(knumb,lr_)*dvol(lr_)
            fuspwrmt(knumb)=fuspwrmt(knumb)
     +        +fuspwrm(knumb,lr_)*dvol(lr_)
            sigft(knumb)=sigft(knumb)
     +        +sigf(knumb,lr_)*dvol(lr_)
            sigmt(knumb)=sigmt(knumb)
     +        +sigm(knumb,lr_)*dvol(lr_)
 90       continue
      enddo ! ll 


c     Store time dependent data. nch has been set by ntdstore.
        if (n .eq. nchec*(n/nchec)) then
          do 80 knumb=1,4
            sigftt(nch(1),knumb)=sigft(knumb)
            sigmtt(nch(1),knumb)=sigmt(knumb)
 80       continue
        endif

cBH110401      endif  ! on icall

      return
      end
