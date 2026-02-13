c
c
c     ONETWO DIVERGENCE
      subroutine frnbdep2(psi,mi,mj,r,z,potsid,mf,rzpat,nrpat,nzpat,me,
     &  mb,sgxn,vbeam,hxfrac,iopt,ibstart) !Not used in cql3d
      implicit integer (i-n), real*8 (a-h,o-z)
c-----------------------------------------------------------------------
c     calculates neutral beam deposition on (r,z) grid.  grid size
c     determined by nrpat and nzpat, the number of equally spaced
c     elements in the r and z axes.  outputs the 3rd excited state
c     component, rzhex, to file 'beamdep' for postprocessing.
c-----------------------------------------------------------------------
c     ONETWO DIVERGENCE
      include 'param.h'
CMPIINSERT_INCLUDE

      parameter (nbdep=29)
      dimension psi(ki,kj),r(ki),z(kj),potsid(kf),
     &  rzpat(kix2,kjx2,ke,kb),sgxn(kz,ke,kb),vbeam(ke,kb),
     &  hxfrac(ke,kb)
      dimension psipat(kix2,kjx2),rr(kix2),zz(kjx2),
     &  rznub(kix2,kjx2,ke,kb),rzhex(kix2,kjx2,ke,kb),pc(kz),
     &  capsig(kz),rzsig(kikj),frac(kz),rzfrac(kikj),
     &  splnwk(kwork),cspln1(kzm1,3),cspln2(kzm1,3),
     &  bpar1(4),bpar2(4),psikp(kikj),isupp(4,ke,kb),inc(kjx2)
c
      zero=0.d0

c     set (r,z) grid modification option
      igrid=iopt-1
c
c     zero out arrays
      do 100 i=1,4
        bpar1(i)=0.
        bpar2(i)=0.
 100  continue
c
c     set up integer array to allow vectorization further on
      do 90 i=1,nzpat
        inc(i)=i-1
 90   continue
c
c     get psi on zone centers
      mfm1=mf-1
      do 110 i=1,mfm1
        pc(i)=0.5*(potsid(i)+potsid(i+1))
 110  continue
c
c     if user defined new (r,z) grid, interpolate psi(i,j) onto it
      if(igrid.eq.1)then
        dr=(r(mi)-r(1))/(nrpat-1)
        rr(1)=r(1)
        do 112 i=2,nrpat
          rr(i)=rr(1)+inc(i)*dr
 112    continue
        dz=(z(mj)-z(1))/(nzpat-1)
        zz(1)=z(1)
        do 114 i=2,nzpat
          zz(i)=zz(1)+inc(i)*dz
 114    continue
        call ibcieu1(psi,ki,r,mi,z,mj,rr,nrpat,zz,nzpat,psipat,kix2,
     &    splnwk,ier)
      endif
c
c     begin loop over beam and beam energy
      do 200 ib=ibstart,mb
        do 201 ie=1,me
c
c     get an estimate of the support of rzpat
          call frsuppor(rzpat,nrpat,nzpat,ie,ib,isupp,ifail)
c
c     get spline fits to macroscopic neutral beam attenuation cross
c     sections as a function of psi
          do 120 i=1,mfm1
            capsig(i)=sgxn(i,ie,ib)
 120      continue
          call icsicu1(pc,capsig,mfm1,bpar1,cspln1,kzm1,ier)
c
c     loop over (r,z) points
          do 130 i=isupp(1,ie,ib),isupp(2,ie,ib)
            do 131 j=isupp(3,ie,ib),isupp(4,ie,ib)
              if(rzpat(i,j,ie,ib).ne.zero)then
                n=n+1
                if(igrid.eq.0)then
                  psikp(n)=psi(i,j)
                else
                  psikp(n)=psipat(i,j)
                endif
              endif
 131        continue
 130      continue
          call icsevu1(pc,capsig,mfm1,cspln1,kzm1,psikp,rzsig,n,ier)
          n=0
          do 140 i=isupp(1,ie,ib),isupp(2,ie,ib)
            do 141 j=isupp(3,ie,ib),isupp(4,ie,ib)
              if(rzpat(i,j,ie,ib).ne.zero)then
                n=n+1
                rznub(i,j,ie,ib)=rzpat(i,j,ie,ib)/(vbeam(ie,ib)
     +            *rzsig(n))
              endif
 141        continue
 140      continue
          do 150 i=isupp(1,ie,ib),isupp(2,ie,ib)
            do 151 j=isupp(3,ie,ib),isupp(4,ie,ib)
              rzhex(i,j,ie,ib)=rznub(i,j,ie,ib)*hxfrac(ie,ib)
 151        continue
 150      continue
 201    continue
 200  continue
c
c     output subset of rzhex to 'beamdep'
      nout=nbdep

CMPIINSERT_IF_RANK_NE_0_RETURN

      open(unit=nout,file='beamdep',status='new')
      write(nout,1000) nrpat,nzpat,me,mb
      if(igrid.eq.0)then
        write(nout,1010) r(1),r(mi),z(1),z(mj)
      else
        write(nout,1010) rr(1),rr(nrpat),zz(1),zz(nzpat)
      endif
      do 300 ib=ibstart,mb
        do 301 ie=1,me
          write(nout,1000) (isupp(i,ie,ib),i=1,4)
          write(nout,1010) hxfrac(ie,ib)
          write(nout,1010) ((rzhex(i,j,ie,ib)
     &      ,i=isupp(1,ie,ib),isupp(2,ie,ib))
     &      ,j=isupp(3,ie,ib),isupp(4,ie,ib))
 301    continue
 300  continue
 1000 format(4(5x,i4))
 1010 format(7(2x,e16.7))
      close(unit=nout)
c
      return
      end
