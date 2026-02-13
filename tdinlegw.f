      subroutine tdinlegw(klpar,klrad,klindxr,klmesh)
      implicit integer (i-n), real*8 (a-h,o-z)
c
c     Prepares weights for computing the Legendre coefficients:
c     V_m=int[f*P_m(cos(th))*sin(th)*dth]
c     such that V_m=sum(i) f(i)*dcofleg(i), i=1,imax and iy+1-imax,iy.
c     That is, dcofleg(i) covers the interval [i-1/2,i+1/2].
c
c     If analegco="enabled", the integral int[P_m(cos(th))*sin(th)*dth]
c     is already analytically computed, in dpcosz, but in [i,i+1]
c
c     If analegco="disabled":
c     The integral is computed using a ngauss-point Gaussian quadrature
c     where f at the Gauss points is obtained using a nlagran-point 
c     Lagrangian interpolation with polynomials (see 25.2.2 p.878, 
c     Abramowitz and Stegun). For the weights, we use the actual 
c     value of P_m(cos(th))*sin(th) at the Gauss points, multiplied 
c     by the Gaussian weights. Thus only f is interpolated.
c     (good parameters seemto be: kgauss=2 or 4, 
c     klagran=3 or 4, not respectively)
c
c     Compute for the theta mesh y(i,klmesh), and introduce weights into
c     array dcofleg(i,klpar,m,klindxr), for i=1,iy_(klmesh) and m=0,mmx,
c       mmx=mx, or if softxry.ne."disabled" mmx=max(mx,msxr).
c
c     CQL3D case: cos(th) -> sqrt(1-bbpsi*sin(th0)**2)=cosz(i,klpar,klrad)
c     sin(th)dth -> bbpsi(s)*sin(th0)*cos(th0)/cos(th)*dth0
c     th0 given in y(i,l_), bbpsi(s) in bbpsi(l,lr_) for i=1,imax(l,lr_)
c     add contribution for th0 and pi-th0. Contribution up to
c     turning-point is added in dcofleg(imax).
c
c     CQLP case:th = th(l) given in y(i,l_) for i=1,iy_(l_), i.e. [0,pi]
c
      include 'param.h'
      include 'comm.h'

      allocatable :: zxg(:),zxgh(:,:),zli(:,:), ilaglim(:,:)
             
      mdxgau=iy*20 

      if (softxry.ne.'disabled') then
         mmx=max(mx,msxr)
      else
         mmx=mx
      endif
      
      if (.NOT. ALLOCATED(zxg)) then
        allocate( zxg(mdxgau) ,STAT=istat)
        call bcast(zxg,zero,mdxgau)
        allocate( zxgh(mdxgau,0:mmx) ,STAT=istat)
        call bcast(zxgh,zero,mdxgau*(mmx+1))
        allocate( zli(mdxgau,20) ,STAT=istat)
        call bcast(zli,zero,mdxgau*20)
        allocate( ilaglim(mdxgau,2) ,STAT=istat)
        call ibcast(ilaglim,0,mdxgau*2)
      endif
      
c.......................................................................

      if (analegco .eq. "enabled") go to 200

c     CQL3D: imax points in two intervals [1,imax] and [iy+1-imax,iy]
      ipoints=imax(klpar,klrad)
      iistart=iy_(klmesh)+1-ipoints
      iigstrt=(iistart-1)*ngauss+1
      iparts=2
c     CQLP: one interval [1,iy]
      if (cqlpmod.eq."enabled") then
        ipoints=iy_(klmesh)
        iistart=0
        iigstrt=0
        iparts=1
      endif

c.......................................................................
cl    1. Gaussian quadrature, analegco="disabled"
c.......................................................................

      ilagran=nlagran
      if (nlagran.gt.min(ipoints,15) .or. nlagran.le.1) 
     +  ilagran=min(ipoints,15)

      inxgau=(ipoints-1)*ngauss
      if (mdxgau .lt. max(inxgau,iigstrt+inxgau-1)) then
        print *,' mdxgau= ',mdxgau,' too small for ',ipoints-1,'*'
     +    ,ngauss,' Gaussian points'
        stop 'tdinlegw'
      endif

c.......................................................................
cl    1.1 construct Gaussian mesh and P_m*sin*dh for each m
c.......................................................................

      call congau(y(1,klmesh),zxg,zxgh(1,0),ipoints,ngauss)
      if (iparts .eq. 2) call congau(y(iistart,klmesh),
     +  zxg(iigstrt),zxgh(iigstrt,0),ipoints,ngauss)
     
      do 110 iside=1,iparts
        ig0=(iside-1)*(iigstrt-1)
        zfact=1.0
        if (iside .eq. 2) zfact=-1.0

c     compute m=1 (i.e. cos(th)) and total weight
        if (cqlpmod .ne. "enabled") then
          do 111 ig=ig0+1,ig0+inxgau
            zxgh(ig,1)=zfact*sqrt(1.-bbpsi(klpar,klrad)*sin(zxg(ig))**2)
            zxgh(ig,0)=zxgh(ig,0)*bbpsi(klpar,klrad)*sin(zxg(ig))*
     +        cos(zxg(ig))/zxgh(ig,1)
 111      continue
        else ! (cqlpmod.eq."enabled")
          do 112 ig=ig0+1,ig0+inxgau
            zxgh(ig,1)=cos(zxg(ig))
            zxgh(ig,0)=zxgh(ig,0)*sin(zxg(ig))
 112      continue
        endif
c     construct Legendre polynomials for m=2 to mmx
c     m=2 and 3: explicit formula
        do 113 ig=ig0+1,ig0+inxgau
          zxgh(ig,2)=0.5*(3.*zxgh(ig,1)**2-1.)
          zxgh(ig,3)=0.5*zxgh(ig,1)*(5.*zxgh(ig,1)**2-3.)
 113    continue
c     m=4,mmx: recurrence formula
        do 114 m=3,mmx-1
          do 115 ig=ig0+1,ig0+inxgau
            zxgh(ig,m+1)=((2.*m+1.)*zxgh(ig,1)*zxgh(ig,m)-
     +        m*zxgh(ig,m-1))/(m+1.)
 115      continue
 114    continue

c     multiply by Jacobian
        do 116 m=1,mmx
          do 117 ig=ig0+1,ig0+inxgau
            zxgh(ig,m)=zxgh(ig,m)*zxgh(ig,0)
 117      continue
 116    continue

 110  continue

c.......................................................................
cl    1.2 Determines which mesh points i will contribute, in the Lagrange
cl    interpolation, to calculate the function at the Gaussian points
cl    in the interval iint.
c.......................................................................

      if (ilagran .eq. ipoints) then
        do 121 iside=1,iparts
          iint0=(iside-1)*(iistart-1)
          do 122 iint=iint0+1,iint0+ipoints-1
            ilaglim(iint,1)=iint0+1
            ilaglim(iint,2)=iint0+ipoints
 122      continue
 121    continue
      else
        do 123 iside=1,iparts
          iint0=(iside-1)*(iistart-1)
          do 124 iint=iint0+1,iint0+ipoints-1
            ifirst=max(iint0+1,iint-(ilagran-1)/2)
            ilast=ifirst+ilagran-1
            if (ilast .gt. iint0+ipoints) then
              ilast=iint0+ipoints
              ifirst=iint0+ipoints-ilagran+1
            endif
            ilaglim(iint,1)=ifirst
            ilaglim(iint,2)=ilast
 124      continue
 123    continue
      endif

c.......................................................................
cl    1.3 Construct Lagrangian interpolation and compute weights
cl    V_m = int[ f*Pm*sin(th)*dth ] = sum(ig) [ f_ig*Pm_ig*sin(th_ig)*dth_ig]
cl    with f_ig = sum(jl) f_jl*L_jl_ig, the Lagrangian interpolation of f_ig
cl    using the nodal points f_jl, we have:
cl    V_m = sum(jl) f_jl * [sum(ig)*L_jl_ig*Pm_ig*sin(th_ig)*dth_ig], where
cl    the term [] is dcofleg
c.......................................................................

      do 130 m=0,mmx
        do 1301 inode=1,iy_(klmesh)
          dcofleg(inode,klpar,m,klindxr)=0.0
 1301   continue
 130  continue

      do 131 iside=1,iparts
        iint0=(iside-1)*(iistart-1)
        do 132 iint=iint0+1,iint0+ipoints-1
          call lagrng(y(1,klmesh),iint0+ipoints,ngauss,zxg,iint,
     +      ilaglim(iint,1),ilaglim(iint,2),zli,mdxgau)
          do 133 ig=1,ngauss
            iig=(iint-1)*ngauss
            do 134 m=0,mmx
              do 135 inode=ilaglim(iint,1),ilaglim(iint,2)
                dcofleg(inode,klpar,m,klindxr)=
     +            dcofleg(inode,klpar,m,klindxr)+
     +            zli(inode,ig)*zxgh(iig+ig,m)
 135          continue
 134        continue
 133      continue
 132    continue
 131  continue

c.......................................................................
cl    1.4 For CQL3D case, add contribution of integral between theta=imax
cl    and the turning-point, computed in dpcosz(imax)
c.......................................................................

      if (cqlpmod.eq."enabled") return

      iimax=iy_(klmesh)+1-imax(klpar,klrad)
      do 140 m=0,mmx
        dcofleg(imax(klpar,klrad),klpar,m,klindxr)=
     =    dcofleg(imax(klpar,klrad),klpar,m,klindxr) + 
     +    dpcosz(imax(klpar,klrad),klpar,m,klrad)
        dcofleg(iimax,klpar,m,klindxr)=
     +    dcofleg(iimax,klpar,m,klindxr)
     +    +dpcosz(iimax,klpar,m,klrad)
 140  continue

      return

c.......................................................................
cl    2. Use analytical integration of P_m(cos(th))*sin(th)*dth between
cl    th(i) and th(i+1) (but th(i-1) and th(i) for i>iyh).
cl    Note: dpcosz changes sign after tdinlegw is called =>-1
cl    dcofleg(i) = 0.5 * (dpcosz(i-1) + dpcosz(i)), such that
cl    V_m = sum(i) f_i * dcofleg(i), instead of using dpcosz(i) and
cl    0.5*(f(i)+f(i+1)) as before.
cl    For i=imax, add also contribution between imax and turning-point
cl    computed in dpcosz(imax)
c.......................................................................

 200  continue

      iend=imax(klpar,klrad)-1
      if (cqlpmod.eq."enabled") iend=iyh_(klmesh)-1

      do 201 m=0,mmx
        do 202 i=2,iend
          ii=iy_(klmesh)+1-i
          dcofleg(i,klpar,m,klindxr)=-0.5*(dpcosz(i-1,klpar,m,klrad)
     +      + dpcosz(i,klpar,m,klrad))
          dcofleg(ii,klpar,m,klindxr)=-0.5*(dpcosz(ii,klpar,m,klrad)
     +      +dpcosz(ii+1,klpar,m,klrad))
 202    continue
        dcofleg(1,klpar,m,klindxr)=-0.5*dpcosz(1,klpar,m,klrad)
        dcofleg(iy_(klmesh),klpar,m,klindxr)=-0.5*
     *    dpcosz(iy_(klmesh),klpar,m,klrad)
        dcofleg(imax(klpar,klrad),klpar,m,klindxr)=
     =    -0.5*dpcosz(imax(klpar,klrad)-1,klpar,m,klrad) - 
     -    dpcosz(imax(klpar,klrad),klpar,m,klrad)
        iimax=iy_(klmesh)+1-imax(klpar,klrad)
        dcofleg(iimax,klpar,m,klindxr)=
     =    -0.5*dpcosz(iimax+1,klpar,m,klrad) - 
     -    dpcosz(iimax,klpar,m,klrad)
 201  continue
CMG added 11/13/2017
      if(allocated(zxg)) deallocate(zxg)
      if(allocated(zxgh)) deallocate(zxgh)
      if(allocated(zli)) deallocate(zli)
      if(allocated(ilaglim)) deallocate(ilaglim)
CMG end added 11/13/2017

      return
      end

C=======================================================================
C=======================================================================
      subroutine lagrng(pxnodes,ktotnod,kgaus,pxg,kint,kistrt,kiend,pli
     1  ,klidim)
      implicit integer (i-n), real*8 (a-h,o-z)
c.......................................................................
c
c     Lagrangian interpolation with kmode points
c     compute l_knode(pxg(ii)),ii=1,kpts, assuming pxnodes(i=1,ktotnod) points
c
c.......................................................................

      dimension pxnodes(ktotnod),pxg(ktotnod*kgaus),pli(klidim,20)
c.......................................................................

      iig0=(kint-1)*kgaus
      do 100 inode=kistrt,kiend
        ifirst1=kistrt
        ilast1=inode-1
        ifirst2=inode+1
        ilast2=kiend
c     compute denominator
        zdenom=1.0
        zxnod=pxnodes(inode)
        do 101 i=ifirst1,ilast1
          zdenom=zdenom*(zxnod-pxnodes(i))
 101    continue
        do 102 i=ifirst2,ilast2
          zdenom=zdenom*(zxnod-pxnodes(i))
 102    continue
        zdenom=1./zdenom

c     initialize pli's
        do 200 ig=1,kgaus
          pli(inode,ig)=zdenom
 200    continue

c     compute l_i(x)
        do 210 i=ifirst1,ilast1
          do 211 ig=1,kgaus
            pli(inode,ig)=pli(inode,ig)*(pxg(iig0+ig)-pxnodes(i))
 211      continue
 210    continue
        do 220 i=ifirst2,ilast2
          do 221 ig=1,kgaus
            pli(inode,ig)=pli(inode,ig)*(pxg(iig0+ig)-pxnodes(i))
 221      continue
 220    continue

 100  continue

      return
      end
      subroutine congau(py,pyg,pygh,kpts,kgau)
      implicit integer (i-n), real*8 (a-h,o-z)
c
      dimension py(kpts),pyg(kpts*kgau),pygh(kpts*kgau)
      dimension zgx(20),zgh(20)
c
      call gauss(kgau,zgx,zgh)
c
      do 100 igx=1,kgau
        zhigx=0.5*zgh(igx)
        do 110 i=1,kpts-1
          ii=(i-1)*kgau+igx
          pyg(ii)=(py(i)+py(i+1)+(py(i+1)-py(i))*zgx(igx)) * 0.5
          pygh(ii)=(py(i+1)-py(i)) * zhigx
 110    continue
 100  continue

      return
      end
      subroutine gauss(kpts,pgausx,pgaush)
      implicit integer (i-n), real*8 (a-h,o-z)
c     ------------------------------------
c
c     Define points and weights for gaussian and other quadratures
c
c     up to 6 points and 20, i.e. integrating exactly polynomial of degree
c     up to 12 and 40.
c     taken out from abramowitz
c
c     kpts = 2-6, 20 : kpts-gaussian quadrature
c     9       : thalia's formula, 9 equidistant points
c     1       : middle point rectangle formula
c     12      : 2 points "rectangle" formula at 1/4 and 3/4
c
      dimension   pgausx(kpts+1),pgaush(kpts+1)
c
      if (kpts .eq. 9) go to 900
      if (kpts .eq. 12) go to 1200
      if (kpts .eq. 20) go to 2000
      go to (100,200,300,400,500,600)  kpts
c
c     1-points formula ==> middle point only (rectangle formula)
c     good for ngxpp, when nx=nxpp
 100  pgausx(1) =  0.0
c
      pgaush(1) =  2.0
c
      return
c%%c  
c%%c  1-points formula ==> trapezoidal rule
c%%   100     pgausx(1) = -1.0
c%%   pgausx(2) =  1.0
c%%c  
c%%   pgaush(1) =  1.0
c%%   pgaush(2) =  1.0
c%%c  
c%%   kpts = 2
c%%c  
c%%   return
c
c     2-points formula
 200  pgausx(1) = -0.577350269189626
      pgausx(2) = +0.577350269189626
c
      pgaush(1) =  1.0
      pgaush(2) =  1.0
c
      return
c
c     3-points formula
 300  pgausx(1) = -0.774596669241483
      pgausx(2) =  0.0
      pgausx(3) = +0.774596669241483
c
      pgaush(1) =  0.555555555555556
      pgaush(2) =  0.888888888888889
      pgaush(3) =  0.555555555555556
c
      return
c
c     4-points formula
 400  pgausx(1) = -0.861136311594053
      pgausx(2) = -0.339981043584856
      pgausx(3) = +0.339981043584856
      pgausx(4) = +0.861136311594053
c
      pgaush(1) =  0.347854845137454
      pgaush(2) =  0.652145154862546
      pgaush(3) =  0.652145154862546
      pgaush(4) =  0.347854845137454
c
      return
c
c     5-points formula
 500  pgausx(1) = -0.906179845938664
      pgausx(2) = -0.538469310105683
      pgausx(3) =  0.0
      pgausx(4) = +0.538469310105683
      pgausx(5) = +0.906179845938664
c
      pgaush(1) =  0.236926885056189
      pgaush(2) =  0.478628670499366
      pgaush(3) =  0.568888888888889
      pgaush(4) =  0.478628670499366
      pgaush(5) =  0.236926885056189
c
      return
c
c     6-points formula
 600  pgausx(1) = -0.932469514203152
      pgausx(2) = -0.661209386466265
      pgausx(3) = -0.238619186083197
      pgausx(4) = +0.238619186083197
      pgausx(5) = +0.661209386466265
      pgausx(6) = +0.932469514203152
c
      pgaush(1) =  0.171324492379170
      pgaush(2) =  0.360761573048139
      pgaush(3) =  0.467913934572691
      pgaush(4) =  0.467913934572691
      pgaush(5) =  0.360761573048139
      pgaush(6) =  0.171324492379170
c
      return
c
c     2-points "rectangle" formula
 1200 pgausx(1) = -0.5
      pgausx(2) = +0.5
c
      pgaush(1) =  1.0
      pgaush(2) =  1.0
c
      kpts = 2
c
      return
c
c     20-points formula
 2000 pgausx(1) = -0.993128599185094
      pgausx(2) = -0.963971927277913
      pgausx(3) = -0.912234428251325
      pgausx(4) = -0.839116971822218
      pgausx(5) = -0.746331906460150
      pgausx(6) = -0.636053680726515
      pgausx(7) = -0.510867001950827
      pgausx(8) = -0.373706088715419
      pgausx(9) = -0.227785851141645
      pgausx(10)= -0.076526521133497
      pgausx(11)=  0.076526521133497
      pgausx(12)=  0.227785851141645
      pgausx(13)=  0.373706088715419
      pgausx(14)=  0.510867001950827
      pgausx(15)=  0.636053680726515
      pgausx(16)=  0.746331906460150
      pgausx(17)=  0.839116971822218
      pgausx(18)=  0.912234428251325
      pgausx(19)=  0.963971927277913
      pgausx(20)=  0.993128599185094
c
      pgaush(1) =  0.017614007139152
      pgaush(2) =  0.040601429800386
      pgaush(3) =  0.062672048334109
      pgaush(4) =  0.083276741576704
      pgaush(5) =  0.101930119817240
      pgaush(6) =  0.118194531961518
      pgaush(7) =  0.131688638449176
      pgaush(8) =  0.142096109318382
      pgaush(9) =  0.149172986472603
      pgaush(10)=  0.152753387130725
      pgaush(11)=  0.152753387130725
      pgaush(12)=  0.149172986472603
      pgaush(13)=  0.142096109318382
      pgaush(14)=  0.131688638449176
      pgaush(15)=  0.118194531961518
      pgaush(16)=  0.101930119817240
      pgaush(17)=  0.083276741576704
      pgaush(18)=  0.062672048334109
      pgaush(19)=  0.040601429800386
      pgaush(20)=  0.017614007139152
c
      return
c
c     9-points thalia formula (equidistant points)
 900  pgausx(1) = -1.00
      pgausx(2) = -0.75
      pgausx(3) = -0.50
      pgausx(4) = -0.25
      pgausx(5) =  0.00
      pgausx(6) =  0.25
      pgausx(7) =  0.50
      pgausx(8) =  0.75
      pgausx(9) =  1.00
c
      pgaush(1) =  2.0 * 989. / 28350.
      pgaush(2) =  2.0 * 5888. / 28350.
      pgaush(3) =  - 2.0 * 928. / 28350.
      pgaush(4) =  2.0 * 10496. / 28350.
      pgaush(5) =  - 2.0 * 4540. / 28350.
      pgaush(6) =  pgaush(4)
      pgaush(7) =  pgaush(3)
      pgaush(8) =  pgaush(2)
      pgaush(9) =  pgaush(1)
c
      return
c
      end
