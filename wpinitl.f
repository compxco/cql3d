c
c
      subroutine wpinitl
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     Initialize some arrays for CQLP case (transport along B).
c     Computes the (Chang-Cooper) weights.
c..............................................................

      include 'param.h'
      include 'comm.h'

c.......................................................................
c     1. Weights for defining f(s+1/2)= f(s+1)*(1-dls(s)) + f(s)*dls(s)
c.......................................................................

      do 100 k=1,ngen
        do 110 j=1,jx
          do 120 l=0,ls+1
            do 130 i=1,iy_(ls)
              dls(i,j,k,l)=.5
 130        continue
 120      continue
 110    continue
 100  continue

c.......................................................................
c     Ensures correct differentiation at end of intervals
c.......................................................................

      if (sbdry .ne. "periodic") then
        do 140 k=1,ngen
          do 141 j=1,jx
            do 142 i=1,iy_(ls)
              dls(i,j,k,0)=0.
              dls(i,j,k,ls)=1.
 142        continue
 141      continue
 140    continue
      endif

c.......................................................................
c     2. If the s mesh is assumed periodic, the numeration of the mesh
c     points s(l) is changed, before solving the matrix problem, in
c     order to keep a band-type matrix. The new points are indexed
c     as follows:
c     1, 2, ls, 3, ls-1, ..., ls/2, ls-(ls/2-2), ls/2+1
c     Thus, the band-width is only twice the natural width. The following
c     arrays define the above transformation between the periodic and the
c     bounded mesh and vice versa (like points anti-clockwise on circle):
c     lsbtopr(1:ls)= 1, 2, 4, 6, .., ls, ls-1, ls-3, .., 5, 3
c     lsprtob(1:ls)= 1, 2, ls, 3, ls-1, ..., ls/2, ls-(ls/2-2), ls/2+1
c
c     If 10<nummods<21, then solve a periodic system with 2(ls-1) points
c.......................................................................

      if (sbdry .eq. "periodic") then
        do 200 l=2,ls/2
          lsbtopr(l)=(l-1)*2
          lsbtopr(ls+2-l)=lsbtopr(l)+1
          lsprtob((l-1)*2)=l
          lsprtob((l-1)*2+1)=ls+2-l
 200    continue
        lsbtopr(1)=1
        lsbtopr(ls/2+1)=ls
        lsprtob(1)=1
        lsprtob(ls)=ls/2+1
c     extra points: s(0)=s(ls) and s(ls+1)=s(1)
        lsbtopr(0)=3
        lsbtopr(ls+1)=1
      else
c     bounded case: one-to-one correspondance
        do 220 l=0,ls+1
          lsbtopr(l)=l
          lsprtob(l)=l
 220    continue
      endif

c.......................................................................
cl    3. Define lower and upper limit for l mesh at each value of theta
c.......................................................................

c.......................................................................
cl    3.1 Define l_lower(i); the lowest index l for which pitch angle
c     index i is defined on the local transport pitch angle mesh.
c.......................................................................

      !call ibcast(l_lower,1,iymax)
      l_lower(:)=1 !YuP[2021]
      if (iy_(1).ne.iy_(ls) .or. iy_(1).ne.iy_(ls/2)) then
c     cannot have periodic mesh and not full l mesh
c%OS  if (sbdry .eq. "periodic") call wpwrng(2)
        illend=lrors
        if (sbdry .eq. "periodic") illend=lrors/2+1
        do 310 l=2,illend
          do 311 i=iyh_(l-1)+1,iyh_(l)
            l_lower(i)=l
            !l_lower(iy_(l)+1-i)=l !  !YuP[2021-03-11] iymax-->iy_(l) ?
            l_lower(iymax+1-i)=l 
 311      continue
 310    continue
      endif

c.......................................................................
cl    3.2 Define l_upper(i); the highest index l for which pitch angle
c     index i is defined on the local transport pitch angle mesh.
c.......................................................................
      call ibcast(l_upper,ls,iymax)
      if (iy_(1).ne.iy_(ls) .or. iy_(1).ne.iy_(ls/2)) then
        illend=lrors
        if (sbdry .eq. "periodic") illend=lrors/2+1
        do 320 l=illend-1,1,-1
          do 321 i=iyh_(l+1)+1,iyh_(l)
            l_upper(i)=l
            !l_upper(iy_(l)+1-i)=l ! !YuP[2021-03-11] iymax-->iy_(l) ?
            l_upper(iymax+1-i)=l
 321      continue
 320    continue
      endif

c.......................................................................
cl    3.3 Check values of l_lower and l_upper
c     So far one assumes that l=1 is where all the particles pass (min. B)
c     => l_lower(i)=1 for all i
c     For l_upper, one should have i=iyh_(l) at l=l_upper(i), i>itl_(1)
c     and l_upper(i)=ls for i<=itl_(1)
c.......................................................................

      if (iymax .ne. iy_(1)) call wpwrng(11)
      do 330 i=1,iymax
        if (l_lower(i) .ne. 1) call wpwrng(12) !all l_lower(i)=1 for all i
 330  continue

      do 335 i=1,itl_(1) ! it is "one" in (1)
        if (l_upper(i) .ne. ls) call wpwrng(13)
        if (l_upper(iy_(1)+1-i) .ne. ls) call wpwrng(13)
           !YuP[2021-03-11] iymax-->iy_(1)
           ![actually, it is verified above that iymax=iy_(1) ] 
 335  continue
      do 336 i=itl_(1)+1,iyh_(1) ! it is "one" in (1)
        if (l_upper(min(i+1,iyh_(1))) .ne. l_upper(i)) then
          if (i .ne. iyh_(l_upper(i))) then
            write(*,*)'wpinitl: i, iyh_(l_upper(i))=',i,iyh_(l_upper(i))
            call wpwrng(14)
          endif
          if (iy_(l_upper(i))+1-i .ne. iyh_(l_upper(iy_(1)+1-i))+1) then
             write(*,*)
     *       'wpinitl iy_(l_upper(i))+1-i, iyh_(l_upper(iy_(1)+1-i))+1',
     *                iy_(l_upper(i))+1-i, iyh_(l_upper(iy_(1)+1-i))+1
             call wpwrng(14)
          endif
           !YuP[2021-03-11] iymax-->iy_(1) 
           ![actually, it is verified above that iymax=iy_(1) ] 
        endif
 336  continue

c.......................................................................
cl    4. Initialize some variables
c.......................................................................
cl    4.1 fnp1 (which may be used before first time-step in wparsou)
c.......................................................................

      call dcopy(iyjx2*ngen*lrors,f(0,0,1,1),1,fnp0(0,0,1,1),1)
      call dcopy(iyjx2*ngen*lrors,f(0,0,1,1),1,fnp1(0,0,1,1),1)
      if (sbdry .eq. "periodic") then
        call dcopy(iyjx2*ngen,fnp0(0,0,1,1),1,fnp0(0,0,1,lrors+1),1)
        call dcopy(iyjx2*ngen,fnp0(0,0,1,lrors),1,fnp0(0,0,1,0),1)
        call dcopy(iyjx2*ngen,fnp1(0,0,1,1),1,fnp1(0,0,1,lrors+1),1)
        call dcopy(iyjx2*ngen,fnp1(0,0,1,lrors),1,fnp1(0,0,1,0),1)
      endif

c.......................................................................
cl    4.2 electrostatic electric field
c.......................................................................

      call bcast(elparnw(0),elpar0,ls+2)

c.......................................................................
cl    4.3 Effective value of l-1,l,l+1 for a given l (ex:ls+1->1 if period.)
c.......................................................................

      do 430 l=2,ls-1
        lpm1eff(l,-1)=l-1
        lpm1eff(l, 0)=l
        lpm1eff(l,+1)=l+1
 430  continue
      if (sbdry .eq. "periodic") then
        lpm1eff(0,-1)=ls-1
        lpm1eff(0, 0)=ls
        lpm1eff(0,+1)=1
        lpm1eff(1,-1)=ls
        lpm1eff(1, 0)=1
        lpm1eff(1,+1)=2
        lpm1eff(ls,-1)=ls-1
        lpm1eff(ls, 0)=ls
        lpm1eff(ls,+1)=1
        lpm1eff(ls+1,-1)=ls
        lpm1eff(ls+1, 0)=1
        lpm1eff(ls+1,+1)=2
      else
c     assumes 0->1 and ls+1->ls
        lpm1eff(0,-1)=1
        lpm1eff(0, 0)=1
        lpm1eff(0,+1)=2
        lpm1eff(1,-1)=1
        lpm1eff(1, 0)=1
        lpm1eff(1,+1)=2
        lpm1eff(ls,-1)=ls-1
        lpm1eff(ls, 0)=ls
        lpm1eff(ls,+1)=ls
        lpm1eff(ls+1,-1)=ls-1
        lpm1eff(ls+1, 0)=ls
        lpm1eff(ls+1,+1)=ls
      endif

c.......................................................................
cl    4.4 Value ieff of theta at l=+-1 such that one stays the same i=cst line.
c     Mainly for fixed_mu case when following the i=cst line for i>iyh.
c     Set value to -999 if mesh point does not exist.
c.......................................................................

      dp999=999 !YuP[2021-04-02] was -999, but below we use '-dp999'
      ! so here the sign should be + .
      !In the original 1997 version, the usage was
      ! ilpm1ef(i,l,-1)=cvmgt(i,-999,i.le.iyh_(lpm1eff(l,-1)))
      do 440 l=0,ls+1
        do 441 i=1,iyh_(lpm1eff(l,0))
          dpi=i 
          ilpm1ef(i,l,-1)=cvmgt(dpi,-dp999,i.le.iyh_(lpm1eff(l,-1)))
          ilpm1ef(i,l, 0)=i
          ilpm1ef(i,l,+1)=cvmgt(dpi,-dp999,i.le.iyh_(lpm1eff(l,+1)))
          ii=iy_(lpm1eff(l,0))+1-i
          dpi=iy_(lpm1eff(l,-1))+1-i
          ilpm1ef(ii,l,-1)=cvmgt(dpi,-dp999,
     1      i.le.iyh_(lpm1eff(l,-1)))
          ilpm1ef(ii,l, 0)=ii
          dpi=iy_(lpm1eff(l,+1))+1-i
          ilpm1ef(ii,l,+1)=cvmgt(dpi,-dp999,
     1      i.le.iyh_(lpm1eff(l,+1)))
 441    continue

c     i=0 and iy_(l)+1: set to i=1 and iy_(l) respectively such that df/dtheta
c     gives derivative at i=1/2 and iy_(l)-1/2 respectively
        ilpm1ef(0,l,-1)=ilpm1ef(1,l,-1)
        ilpm1ef(0,l, 0)=1
        ilpm1ef(0,l,+1)=ilpm1ef(1,l,+1)
        ii=iy_(lpm1eff(l,0))+1
        ilpm1ef(ii,l,-1)=ilpm1ef(iy_(lpm1eff(l,0)),l,-1)
        ilpm1ef(ii,l, 0)=iy_(lpm1eff(l,0))
        ilpm1ef(ii,l,+1)=ilpm1ef(iy_(lpm1eff(l,0)),l,+1)
 440  continue

      return
      end
