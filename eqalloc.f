c
c
      subroutine eqalloc
      implicit integer (i-n), real*8 (a-h,o-z)

c...................................................................
c     Allocate allocatable arrays
c...................................................................

      include 'param.h'
      include 'comm.h'
cdir$ nobounds

c..................................................................
c     A check on allocations is sucessful entering then exiting
c     the subroutine.
c..................................................................
      write(*,*)'eqalloc:  Entering eqalloc'

      lnlfield=lfield*lrzmax
      lndumeq=4*lnlfield
      allocate(drpmconz(lrzmax),STAT=istat)
      call bcast(drpmconz,zero,SIZE(drpmconz))
      allocate(eqdell(lfield,lrzmax),STAT=istat) !YuP[2021-04] lfielda-->lfield
      call bcast(eqdell,zero,SIZE(eqdell))
      allocate(eqbpol(lfield,lrzmax),STAT=istat) !YuP[2021-04] lfielda-->lfield
      call bcast(eqbpol,zero,SIZE(eqbpol))
      allocate(solr(lfield,lrzmax),STAT=istat) !YuP[2021-04] lfielda-->lfield
      call bcast(solr,zero,SIZE(solr))
      allocate(solz(lfield,lrzmax),STAT=istat) !YuP[2021-04] lfielda-->lfield
      call bcast(solz,zero,SIZE(solz))

      !YuP[2021-04] collected arrays related to field line tracing,
      !made them into pointers:
      allocate(solr_(lfield),STAT=istat)
      allocate(solz_(lfield),STAT=istat)
      allocate(es_(lfield),STAT=istat)
      allocate(eqbpol_(lfield),STAT=istat)
      allocate(bpsi_(lfield),STAT=istat)
      allocate(thtpol_(lfield),STAT=istat)
      allocate(eqdell_(lfield),STAT=istat)
      allocate(tlorb1(lfield),STAT=istat)
      allocate(tlorb2(lfield),STAT=istat)
      allocate(work(3*lfield+1),STAT=istat) !used in micxiniz, eqorbit,
      ! as input in coeff1().
      !Careful - there are other 'work' arrays defined locally in other subroutines

      write(*,*)'eqalloc:  Leaving eqalloc'

      return
      end
