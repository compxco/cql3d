c
c
      subroutine wptrafx ! for meshy="fixed_y" (constant pitch-mesh)
      implicit integer (i-n), real*8 (a-h,o-z)
c..............................................................
c     Time advancement for parallel transport.
c     Assumes constant total number of y mesh points vs. l: iy_(l)=cst
c..............................................................

      include 'param.h'
      parameter(lmatsa=(lsa+1)*nbanda)
      dimension zmat(lmatsa),zxdumy(lsa+1)
      dimension zmat2(lmatsa),zxdumy2(lsa+1)
      include 'comm.h'

      include 'wpadvnc.h'
      dithta(i,j,l)=0.5-0.5*(1/(i+1))+0.5*(1/(iy_(l)+1-i))
      fpithta(i,j,k,l)=fnhalf(i+1,j,k,l)*(1.-dithta(i,j,l)) + 
     +  fnhalf(i  ,j,k,l)*dithta(i,j,l)
c.......................................................................

      do 1 l=1,lrors
        if (iy_(l) .ne. iy_(1)) call wpwrng(15)
 1    continue

c.......................................................................
c     0. The parameter nummods determines the numerical model used to solve
c     the equation along s. nummods is divided into classes of multiple
c     of 10: numclas=nummods/10 => [0,9]; [10,19]; [20,29] ...
c     numclas = 0: normal case
c     1: assumes up/down symmetric case and periodic condition
c     but compute equil. parameters on top half cross-section
c     on ls/2+1 points and then copy the rest in sub. wploweq
c     Otherwise, same as numclas=0 with sbdry="periodic"
c     In each class, nummods is divided into two groups:
c     mod(nummods,10) < 5: 2-D FP velocity equ. solved on nodal s points
c     > 4: velocity eq. solved at s+1/2
c     Then, in these two groups one has the following options, 
c     with numindx=mod(mod(nummods,10),5)=0,1,2,3 or 4, we have:
c     If numindx = 0,1: use backward diff. scheme + bound. cond at l=1
c     2: "  back/forward depending on sign(cos(theta))
c     and on numixts
c     3: " forward diff. + bound cond at l=ls
c     4: " centered diff. + bound. cond. at l=1
c     Note: if sbdry="periodic" there might not be any extra boundary
c     conditions imposed (see wpbdry)
c.......................................................................

c.......................................................................
cl    1. Set the index shift flags and the bandwidths for solving
c     the equation as a function of s only (distance along the magnetic
c     field B).
c     If periodic boundary conditions are used, the unknowns are
c     numbered differently (see wpinitl), which doubles the
c     effective bandwidths.
c     For coss(theta)>0 : df/ds -> (f(s+ipshft2)-f(s+ipshft1))/ds_eff
c     <0 : df/ds -> (f(s+imshft2)-f(s+imshft1))/ds_eff
c     When sign(cos(theta)) is irrelevant, ip... parameters are used
c.......................................................................

      ileft=1
      iright=1
      if (sbdry .eq. "periodic") then
        ileft=2
        iright=2
      endif
      iband=ileft+iright+1

c     standard case: numindx=0 or 1
      ipshft2=0
      ipshft1=-1
      ilstart=2
      ilend=ls
      imshft2=0
      imshft1=-1
      if (numindx .eq. 2) then
        ipshft2=(numixts+1)/2
        ipshft1=(numixts-1)/2
        imshft2=-(numixts-1)/2
        imshft1=-(numixts+1)/2
        ilstart=1
        ilend=ls
      else if (numindx .eq. 3) then
        ipshft2=+1
        ipshft1=0
        ilstart=1
        ilend=ls-1
        imshft2=+1
        imshft1=0
      else if (numindx .eq. 4) then
        ipshft2=+1
        ipshft1=-1
        ilstart=2
        ilend=ls-1
        imshft2=+1
        imshft1=-1
      endif
      if (sbdry .eq. "periodic") then
        ilstart=1
        ilend=ls
      endif
      zip1p2=ipshft1+ipshft2
      zim1p2=imshft1+imshft2

c.......................................................................
cl    2. Construct the matrix. The equations are solved at a given species k
c     and momentum j, but "simultaneously" for all pitch-angle i. This
c     way, one can have longer inner loops over i than over l, as in
c     general iy > ls. The index i was chosen instead of j, because
c     most of the (i,j) arrays have i as first index.
c.......................................................................

c.......................................................................
cl    2.1 Initialize f_n+1/2
c.......................................................................

      call dcopy(iyjx2*ngen*lrors,f(0,0,1,1),1,fnhalf(0,0,1,1),1)
      if (sbdry .eq. "periodic") then
        call dcopy(iyjx2*ngen,fnhalf(0,0,1,1),1,fnhalf(0,0,1,ls+1),1)
        call dcopy(iyjx2*ngen,fnhalf(0,0,1,ls),1,fnhalf(0,0,1,0),1)
        call dcopy(iyjx2*ngen,velsou(0,0,1,1),1,velsou(0,0,1,ls+1),1)
        call dcopy(iyjx2*ngen,velsou(0,0,1,ls),1,velsou(0,0,1,0),1)
      else
        call dcopy(iyjx2*ngen,fnhalf(0,0,1,1),1,fnhalf(0,0,1,0),1)
        call dcopy(iyjx2*ngen,fnhalf(0,0,1,ls),1,fnhalf(0,0,1,ls+1),1)
        call dcopy(iyjx2*ngen,velsou(0,0,1,1),1,velsou(0,0,1,0),1)
        call dcopy(iyjx2*ngen,velsou(0,0,1,ls),1,velsou(0,0,1,ls+1),1)
      endif

c.......................................................................
cl    2.1.2 Might need value of fnhalf at l=l+1/2. Node value still in f
c     (Already the case if mod(nummods,10)>4, as 2-D momentum eq.
c     solved at l+1/2)
c.......................................................................

      if (mod(nummods,10).le.4 .and. numindx.ne.4) then
        do 2120 k=1,ngen
          do 2121 j=1,jx
            do 2122 l=1,ls
              do 2123 i=1,iy_(min(ls,l+1))
                fnhalf(i,j,k,l)=0.5*(fnhalf(i,j,k,l)+fnhalf(i,j,k,l+1))
 2123         continue
 2122       continue
 2121     continue
 2120   continue
        if (sbdry .eq. "periodic") then
          call dcopy(iyjx2*ngen,fnhalf(0,0,1,1),1,fnhalf(0,0,1,ls+1)
     +      ,1)
          call dcopy(iyjx2*ngen,fnhalf(0,0,1,ls),1,fnhalf(0,0,1,0),1)
        else
          call dcopy(iyjx2*ngen,fnhalf(0,0,1,1),1,fnhalf(0,0,1,0),1)
          call dcopy(iyjx2*ngen,fnhalf(0,0,1,ls),1,fnhalf(0,0,1,ls+1)
     +      ,1)
        endif
      endif
      ivelmid=1
      if (mod(nummods,10) .le. 4) ivelmid=0

c.......................................................................
cl    2.2 Loop over general species k
c.......................................................................

      do 220 k=1,ngen

c     j=1 => zeroth order eq. to be solved, assuming velsou(j=1)=0
c     => f_n+1(j=1)=f_n+1/2(j=1)
        do 221 ll=0,ls+1
          call bcast(fnp1(0,1,k,ll),f(1,1,k,ll),iyp1+1)
 221    continue

c.......................................................................
cl    2.3 Loop over momentum j=2,jx
c.......................................................................

        do 230 j=2,jx

          ztra1=vnorm*x(j)
          ztra2=0.5/dtreff

c.......................................................................
cl    2.3.1 Construct matrices at each theta y(i)
c.......................................................................

          do 231 l=ilstart,ilend

c.......................................................................
cl    2.3.1.1 Determine matrix row and band index for f_s-1, f_s and f_s+1
c.......................................................................
            ism1=1
            isdiag=ileft+1
            isp1=iband
            ilsrow=lsbtopr(l)
            if (sbdry.eq."periodic" .and. (l.le.2.or. l.gt.ls/2)) then
              if (l .eq. 1) then
                ism1=iband
                isp1=iband-1
              else if (l .eq. 2) then
                ism1=ileft
              else if (l .eq. ilend/2+1) then
                isp1=ileft
              else if (l .eq. ilend/2+2) then
                ism1=isdiag+1
                isp1=1
              else
                ism1=iband
                isp1=1
              endif
            endif

c.......................................................................
cl    2.3.1.2 Construct matrix in case of cst y mesh: iy_(l)=cst, all l
c.......................................................................

            if (numindx.eq.2 .and. l.eq.(ipshft2*ls-ipshft1) .and.
     1        sbdry.ne."periodic") go to 23120

c     cos(theta)>0 schemes
            ztrders=ztra1/(ipshft2*dszp5(l)-ipshft1*dszm5(l))
            ztrdert=ztra2*dfloat(ipshft2-ipshft1)
            do 23121 i=1,iyh_(l)
              bndmats(i,ilsrow,ism1,1)=
     +          dfloat(ipshft1)*(coss(i,l)*ztrders+zip1p2*ztrdert)
              bndmats(i,ilsrow,isdiag,1)=ztrdert-zip1p2*coss(i,l)
     +          *ztrders
              bndmats(i,ilsrow,isp1,1)=
     +          dfloat(ipshft2)*(coss(i,l)*ztrders+zip1p2*ztrdert)
              rhspar(ilsrow,i,1)=cvmgt(
     +          wprhsmd(i,j,k,l+ipshft1*(1-ipshft2))
     1          ,wprhs  (i,j,k,l+ipshft1*(1-ipshft2))
     1          ,ivelmid.eq.1 .or. (ipshft1+ipshft2).eq.0)
23121       continue

            if (numindx.ne.2 .and. l.eq.(imshft2*ls-imshft1) .and.
     1        sbdry.ne."periodic") go to 231

c     cos(theta)<0 schemes
23120       continue
            ztrders=ztra1/(imshft2*dszp5(l)-imshft1*dszm5(l))
            ztrdert=ztra2*dfloat(imshft2-imshft1)
            do 23123 i=iyh_(l)+1,iy_(l)
              bndmats(i,ilsrow,ism1,1)=
     +          dfloat(imshft1)*(coss(i,l)*ztrders+zim1p2*ztrdert)
              bndmats(i,ilsrow,isdiag,1)=ztrdert-zim1p2*coss(i,l)
     +          *ztrders
              bndmats(i,ilsrow,isp1,1)=
     +          dfloat(imshft2)*(coss(i,l)*ztrders+zim1p2*ztrdert)
              rhspar(ilsrow,i,1)=cvmgt(
     +          wprhsmd(i,j,k,l+imshft1*(1-imshft2))
     1          ,wprhs(i,j,k,l+imshft1*(1-imshft2))
     1          ,ivelmid.eq.1 .or. (imshft1+imshft2).eq.0)
23123       continue

c.......................................................................
c     end of construction of matrix

 231      continue

c.......................................................................
cl    2.3.2 Apply the boundary conditions
c.......................................................................

          call wpbdry(j,k,ileft,iright,1)

c.......................................................................
cl    2.3.3 Solve the linear systems for each theta point
c.......................................................................

          do 233 i=1,iymax

            ilslen=ls

c.......................................................................
cl    2.3.3.1 Arrange matrix according to band matrix solver used
c     Use the fact that the matrix is tri-diagonal
c.......................................................................

            do 2331 jcolumn=1,iband
              do 2332 l=1,ilslen
                ijmat=(l-1)*iband+jcolumn
                zmat(ijmat)=bndmats(i,l,jcolumn,1)
 2332         continue
 2331       continue

c%OS  
            call dcopy(ilslen*iband,zmat,1,zmat2,1)
            call dcopy(ilslen,rhspar(1,i,1),1,zxdumy2,1)
c%OS  

c%OS  
c%OS  if (ilstart.eq.1 .and. ilspts.eq.ilslen-1 .and. iband.eq.3)
c%OS  1         then
c%OScdir$novector
c%OS  do 2334 l=1,ilslen/2
c%OS  zzz=rhspar(l,i,1)
c%OS  rhspar(l,i,1)=rhspar(ilslen-l+1,i,1)
c%OS  rhspar(ilslen-l+1,i,1)=zzz
c%OS  ijmat=(l-1)*iband+1
c%OS  ijmat2=(ilslen-l)*iband+1
c%OS  zzzbm=zmat(ijmat)
c%OS  zzzb0=zmat(ijmat+1)
c%OS  zzzbp=zmat(ijmat+2)
c%OS  zmat(ijmat)=zmat(ijmat2+2)
c%OS  zmat(ijmat+1)=zmat(ijmat2+1)
c%OS  zmat(ijmat+2)=zmat(ijmat2)
c%OS  zmat(ijmat2)=zzzbp
c%OS  zmat(ijmat2+1)=zzzb0
c%OS  zmat(ijmat2+2)=zzzbm
c%OS  2334         continue
c%OS  if ((ilslen/2)*2 .ne. ilslen) then
c%OS  ijmat=(ilslen/2)*iband+1
c%OS  zzzbm=zmat(ijmat)
c%OS  zmat(ijmat)=zmat(ijmat+2)
c%OS  zmat(ijmat+2)=zzzbm
c%OS  endif
c%OS  endif
c%OS  

c.......................................................................
cl    2.3.3.2 Call solver
c.......................................................................

            do 2333 l=ilslen+1,ilslen+iright
              rhspar(l,i,1)=0.0
 2333       continue
            zerr=1.0e-08
            call nonsym(zmat,zxdumy,rhspar(1,i,1),ilslen,ileft,iright
     +        ,zerr,icond)
            if (icond .ne. 0) write(6,'(/," WARNING: bad condition in",
     +        " nonsym: icond = ",i4)') icond

c%OS  
c%OS  if (ilstart.eq.1 .and. ilspts.eq.ilslen-1 .and. iband.eq.3)
c%OS  1         then
c%OScdir$novector
c%OS  do 2335 l=1,ilslen/2
c%OS  zzz=rhspar(l,i,1)
c%OS  rhspar(l,i,1)=rhspar(ilslen-l+1,i,1)
c%OS  rhspar(ilslen-l+1,i,1)=zzz
c%OS  2335         continue
c%OS  endif
c%OS  

c%OS  check solution
c%OS  zdiff=0.0
c%OS  do 2336 l=1,ilslen
c%OS  zzzl=0.0
c%OS  do 2337 jcol=1,iband
c%OS  zzzl=zzzl+zmat2((l-1)*iband+jcol)*rhspar(l-ileft-1+jcol,i,1)
c%OS  2337       continue
c%OS  zdiff=zdiff + abs((zxdumy2(l)-zzzl)/zxdumy2(l))
c%OS  2336     continue
c%OS  zdiff=zdiff/(1.*ilslen)
c%OS  if (zdiff .gt. 1.0E-08) print *,' zdiff,i,j,n= ',zdiff,i,j,n
c%OS  


 233      continue

c.......................................................................
cl    2.3.4 Copy solution onto fnp1
c.......................................................................

          do 2340 l=1,ilslen
            ileff=lsbtopr(l)
            do 2341 i=1,iy_(l)
              fnp1(i,j,k,l)=rhspar(ileff,i,1)
 2341       continue
 2340     continue

c     end of loop over momentum
 230    continue

c     end of loop over general species
 220  continue

      call dcopy(iyjx2*ngen,fnp1(0,0,1,1),1,fnp1(0,0,1,ls+1),1)
      call dcopy(iyjx2*ngen,fnp1(0,0,1,ls),1,fnp1(0,0,1,0),1)

c.......................................................................
cl    3. Check the solution
c.......................................................................

      if (scheck .eq. "enabled") call wpcheck

c.......................................................................
cl    4. Redefine fnp1 to be unique at v=0
c.......................................................................

      if (nonadi .ne. 3) then

        do 400 k=1,ngen
          do 410 l=1,ls
            zs=0.
            zt=0.
            do 420 i=1,iy_(l)
              zs=zs+cynt2(i,l)
              zt=zt+cynt2(i,l)*fnp1(i,1,k,l)
 420        continue
            do 430 i=1,iy_(l)
              fnp1(i,1,k,l)=zt/zs
 430        continue
 410      continue

          if (sbdry .eq. "periodic") then
c     Assumes ilspts=ls, i.e. solution at 0 and ls+1 not yet calculated
            call dcopy(iyp1+1,fnp1(0,1,k,1),1,fnp1(0,1,k,ls+1),1)
            call dcopy(iyp1+1,fnp1(0,1,k,ls),1,fnp1(0,1,k,0),1)
          endif

 400    continue

      endif

      return
      end
