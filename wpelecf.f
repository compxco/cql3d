c
c
      subroutine wpelecf(kopt)
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     Computes the parallel component of the poloidal electric field due
c     to charge density along the magnetic field line, assuming toroidal
c     symmetry. The analytical solution of the Poisson equation is:
c     (with E_parallel = E_pol * B_pol/B)
c     E_parallel(s) * B(S)/B_pol(s)**2 = [E_par*B/B_pol**2](s=s_0)
c     + 4*pi* int[s_0,s] (ds'*rho(s')/B(s'))
c     where rho(s')=sum over k of q_k*reden(k,l_) is the local charge density.
c
c     kopt =  1: just compute the new E_parallel
c     2: as kopt=1 and adjust the source term velsou accordingly
c     11: same as 1, but assumes reden is already updated (to f)
c     12: same as 2, but assumes reden is already updated (to f)
c
c     This routine assumes that f is the updated distribution function
c..............................................................

      include 'param.h'
      include 'comm.h'
      dimension z4pirho(lsa)

      include 'advnce.h'
      ghelec(i,j)=qz(j)*(zdaij*fpj(i,j)-zdaijm1*fpj(i,j-1)) +
     +  ry(i,j)*(zddij*fpi(i,j)-zddim1j*fpi(i-1,j))
c.......................................................................

      if (n.lt.nonelpr .or. n.gt.noffelpr) return

c.......................................................................
cl    1. Computes the charge density.
c.......................................................................

      do l=1,ls
        z4pirho(l) = 0.0
      end do

c.......................................................................
cl    1.1 Integrate over distribution function if needed
c.......................................................................

      if (kopt .le. 10) then
        do 110 k=1,ngen
          ztra1=charge*bnumb(k)*4.*pi
          do 111 j=1,jx
            ztra2=ztra1*cint2(j)
            do 112 l=1,ls
              do 113 i=1,iy_(l)
                z4pirho(l)=z4pirho(l)+ztra2*f(i,j,k,l)*cynt2(i,l)
 113          continue
 112        continue
 111      continue
 110    continue
      endif

c.......................................................................
c     1.2 Add other species, whose density is already updated
c     nkconro(i) gives the species indices which contribute to the charge
c     density.
c.......................................................................

      do 120 ik=1,ntotal
        k=nkconro(ik)
        if (k .eq. 0) go to 121
        if (k.le.ngen .and. kopt.le.10) go to 120
        ztra1=bnumb(k)*charge*4.*pi
        do 122 l=1,ls
          z4pirho(l)=z4pirho(l)+ztra1*denpar(k,lsindx(l))
 122    continue
 120  continue
 121  continue

c.......................................................................
cl    2. Compute the new parallel component of the electric field
c.......................................................................

      call dcopy(ls+2,elparnw(0),1,elparol(0),1)
      zsumrho=0.d0
      elparnw(1)=elpar0
      zel0cof=elparnw(1)/psipols(1)**2
      do 200 l=2,ls
        zsumrho=zsumrho+0.5*dszm5(l)*(z4pirho(l-1)/psis(l-1)+
     +    z4pirho(l)  /psis(l))
        elparnw(l)=psipols(l)**2/psis(l)*(zel0cof+zsumrho)
        write(*,*)'wpelec: l, elparnw(l)=', l,elparnw(l)
 200  continue

      if (sbdry .eq. "periodic") then
        elparnw(0)=elparnw(ls)
        elparnw(ls+1)=elparnw(1)
      else
        elparnw(0)=0.0
        elparnw(ls+1)=0.0
      endif

c%OS  if (kopt.eq.1 .or. kopt.eq.11) return
      return
c%OS  

c.......................................................................
cl    3. Adjust velsou according to the new value of E_parallel. This
c     is needed if E_parallel is recomputed at half time-steps, in
c     between the velocity split and the spatial split.
c.......................................................................

      do 300 k=1,ngen
        ztra1=bnumb(k)*charge/fmass(k)/vnorm
        do 310 l=1,ls
          zdepar=ztra1*(elparnw(l)-elparol(l))*0.25
          do 320 j=1,jx
            zdacofp=-zdepar*(x(j)+x(j+1-1/(jx+1-j)))**2
            zdacofm=-zdepar*(x(j-1+1/j)+x(j))**2
            zddcof=zdepar*x(j)**2
CDIR$ NOVECTOR
            do 330 i=1,iy_(l)
              zdaij=zdacofp*coss(i,l)
              zdaijm1=zdacofm*coss(i,l)
              zddij=zddcof*(sinn(i,l)+sinn(i+1-1/(iy_(l)+1-i),l))**2
              zddim1j=zddcof*(sinn(i-1+1/i,l)+sinn(i,l))**2
              velsou(i,j,k,l)=velsou(i,j,k,l)+ghelec(i,j)
 330        continue
 320      continue
 310    continue
 300  continue

      return
      end
