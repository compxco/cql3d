c     
c
      real*8 function taper(x,x0,x1,x2)
      implicit integer (i-n), real*8 (a-h,o-z)

c     A multiplication factor between 0. and 1., giving a tapered
c     (i.e., smoothed) box function.
c
c     For abs(x-x0) .lt. x1/2., gives 1.
c     For (x1+x2)/2 .gt. abs(x-x0) .gt. x1/2., gives 
c                                  0.5*(1.+cos(abs(x-x1)/x2*pi))
c     For abs(x-x0) .ge. (x1+x2)/2., gives 0.
c     
c     x1 and x2 are presumed .ge.0.

      save pi
      data pi/3.141592653589793/

      xx=abs(x-x0)
      
      if (xx.lt. x1/2.) then
        taper=1.
      elseif (xx.gt.x1/2. .and. xx.lt.(x1+x2)/2.) then
        taper=0.5*(1.+cos(xx/x2*pi))
      else
        taper=0.0
      endif

      return
      end
c 
c     subroutine aminmx(array,ifirst,ilast,istride,amin,amax,
c     *indmin,indmax)
c     real array(1)
c     length = 1+(ilast+1-ifirst)/istride
c     if(ilast.lt.ifirst+(length-1)*istride) length=length-1
cc    k=istride
c     m=ismin(length,array(ifirst),istride)
c     m=ifirst+(m-1)*istride
cc    m=ifirst
cc    do 1 i=ifirst+k,ilast,k
cc1   if(array(i).le.array(m)) m=i
c     amin=array(m)
c     m=ismax(length,array(ifirst),istride)
c     m=ifirst+(m-1)*istride
cc    m=ifirst
cc    do 2 i=ifirst+k,ilast,k
cc2   if(array(i).ge.array(m)) m=i
c     amax=array(m)
c     return
c     end

c*****************************************************************************
c     SPLINE PACKAGE
c*****************************************************************************
c     package cubspl         note--documentation for individual routines
c     follows the general package information
c
c     latest revision        january 1985
c
c     purpose                to perform one and two-dimensional cubic spline
c     interpolation with choice of boundary
c     conditions.  the function and selected
c     derivatives may be evaluated at any point where
c     interpolation is required.  in the
c     two-dimensional case, the given data points
c     must be on a rectangular grid, which need not
c     be equally spaced.  the package cubspl contains
c     seven routines.
c
c     subroutine coeff1
c     computes the coefficients for one-dimensional
c     cubic spline interpolation using one of
c     the following boundary conditions at
c     each end of the range.
c     . second derivative given at boundary.
c     . first derivative given at boundary.
c     . periodic boundary condition.
c     . first derivative determined by fitting a
c     cubic to the four points nearest to the
c     boundary.
c
c     subroutine terp1
c     using the coefficients computed by coeff1,
c     this routine evaluates the function and/or
c     first and second derivatives at any point
c     where interpolation is required.
c
c     subroutine coeff2
c     computes the coefficients for two-dimensional
c     bicubic spline interpolation with the same
c     choice of boundary conditions as for coeff1.
c
c     function terp2
c     using the coefficients produced by coeff2,
c     this subroutine evaluates the function or a
c     selected derivative at any point where
c     two-dimensional interpolation is required.
c
c     subroutine trip
c     a simple periodic, tridiagonal linear
c     equation solver used by coeff1.
c
c     subroutine search
c     performs a binary search in a one-dimensional
c     floating point table arranged in ascending
c     order.  this routine is called by terp1 and
c     terp2.
c
c     subroutine intrp
c     given coefficients provided by coeff1 and the
c     position of the interpolation point in the
c     independent variable table, this routine
c     performs one-dimensional interpolation for
c     the function value, first and second
c     derivative, as desired.  this routine is
c     called by terp1 and terp2.
c
c     usage                  for one-dimensional cubic spline interpolation,
c     the user first calls coeff1 by
c
c     call coeff1 (n,x,f,w,iop,int,wk)
c
c     this subroutine returns the coefficients
c     needed for the subsequent interpolation
c     in the array w.  the user then calls
c     subroutine terp1 by
c
c     call terp1 (n,x,f,w,y,int,tab,itab)
c
c     in order to compute the value of the
c     function and/or its derivatives.  the user
c     specifies y, the value of the independent
c     variable where the interpolation is to be
c     performed.  the interpolated values are
c     returned in tab.  the parameters
c     n,x,f,w, and int must not be changed
c     between the subroutine calls.
c
c     for two-dimensional cubic spline interpolation
c     the user first calls coeff2 by
c
c     call coeff2 (nx,x,ny,y,f,fxx,fyy,fxxyy,
c     idm,ibd,wk)
c
c     this subroutine returns the coefficients
c     needed for the subsequent interpolation in
c     the arrays fxx, fyy, fxxyy.  the user then
c     calls the routine terp2 by
c
c     r = terp2 (xb,yb,nx,x,ny,y,f,fxx,fyy,fxxyy,
c     idm,ixd,iyd)
c
c     in order to perform the interpolation at the
c     point specified by the values of xb and yb.
c     depending on the values input in ixd and iyd,
c     the routine returns an interpolated value
c     for the function or one of its partial
c     derivatives.  the parameters nx,x,ny,y,f,fxx,
c     fyy,fxxyy, and idm must not be changed
c     between the calls to coeff2 and terp2.
c
c     special conditions     tables of independent variables must be
c     arranged in ascending order.  for
c     two-dimensional interpolation, the data points
c     must lie on a rectangular mesh.
c
c     i/o                    none
c
c     precision              single
c
c     files                  cray machines.
c
c     language               fortran
c
c     history                this package is based on the routines
c     la e 102a, spl1d1
c     la e 103a, spl1d2
c     la e 104a, spl2d1
c     la e 105a, spl2d2
c     of the los alamos cubic spline package written
c     by thomas j. jordan and bertha fagan, 1968.
c     the routines have been streamlined and
c     standardized.  the algorithm for
c     two-dimensional interpolation is considerably
c     modified.
c
c     algorithm              for one-dimensional interpolation, the cubic
c     spline is expressed in terms of the function
c     values and second derivatives at the data
c     points.  the second derivatives are
c     determined from a tridiagonal linear system
c     which describes the continuity of the first
c     derivative and incorporates the given
c     boundary conditions.  coeff1  sets up this
c     system and calls subroutine trip to solve it.
c
c     the cubic segment between two adjacent
c     tabular points is constructed from the
c     function values and second derivatives at
c     these points.  these provide the four
c     constants needed to define the cubic
c     uniquely.  from this cubic, values of the
c     function and its first and second
c     derivatives are readily determined at any
c     intermediate point.  one-dimensional
c     interpolation is performed by the routine
c     terp1.  for two-dimensional interpolation,
c     the bicubic spline is described in terms of
c     values of f,fxx,fyy, and fxxyy  at each
c     point on the given two-dimensional
c     rectangular grid of data points.  here f
c     is the function value,
c
c     fxx = (d/dx)**2*f
c
c     and so on.  the coefficients are determined
c     by coeff2, which uses successive
c     applications of coeff1.
c
c     1.  the array fxx is determined by applying
c     coeff1 to f along each line in the
c     x-direction.
c
c     2.  the array fyy is determined by applying
c     coeff1 to f along each line in the
c     y-direction.
c
c     3.  fxxyy is determined on the constant y
c     boundaries by applying coeff1 to fyy
c     along these boundaries.
c
c     4.  the remainder of the array fxxyy is
c     determined by applying coeff1 to fxx
c     along each line in the y-direction.
c
c     the bicubic within any rectangular element
c     of the grid is constructed from the values
c     of f,fxx,fyy,fxxyy at the four corners.
c     these provide the 16 constants necessary
c     to define the bicubic uniquely.  to find
c     the value of f corresponding to a point
c     (xb,yb) within the elementary rectangle,
c     (x(i),y(j)),(x(i+1),y(j)),(x(i),y(j+1)),
c     (x(i+1),y(j+1)), five one dimensional
c     interpolations are performed.
c
c     1.  f at (xb,y(j)) and (xb,y(j+1)) are
c     found by interpolating f in the
c     x-direction using fxx. (two interpolations)
c
c     2.  fyy at (xb,y(j)) and (xb,y(j+1)) are
c     found by interpolating fyy in the
c     x-direction using fxxyy. (two
c     interpolations.)
c
c     3.  finally f at (xb,yb) is found by
c     interpolating between f(xb,y(j)) and
c     f(xb,y(j+1)) in the y-direction using
c     values of fyy(xb,y(j)) and fyy(xb,y(j+1))
c     obtained above. (one interpolation).
c
c     two-dimensional interpolation is performed
c     in terp2.
c
c     references             for greater detail, see j.l.walsh,
c     j.h.ahlberg, e.n.nilsen, best approximation
c     properties of the spline fit, journal of
c     mathematics and mechanics, vol.ii(1962),
c     225-234.
c
c     t.l. jordan, smoothing and multivariable
c     interpolation with splines, los alamos
c     report, la-3137, 1965.
c
c     accuracy               near machine accuracy was obtained when
c     interpolating a cubic in one dimension
c     or a bicubic in two dimensions.
c
c     portability            fully portable with respect to fortran 66.
c***********************************************************************
c
c     subroutine coeff1 (n,x,f,w,iop,int,wk)
c
c     dimension of           x(n),f(int*(n-1)+1),w(int*(n-1)+1),iop(2),
c     arguments              wk(3*n+1)
c
c     purpose               :subroutine coeff1 computes the coefficients
c     for one-dimensional cubic spline
c     interpolation using one of the following
c     boundary conditions at each end of the
c     range
c
c     .  second derivatives given at boundary
c     .  first derivative given at boundary
c     .  periodic boundary conditions
c     .  first derivative calculated by fitting a
c     cubic to the four points nearest to the
c     boundary
c
c     note that terp1 must be called to perform
c     the interpolation.
c
c     usage                  call coeff1 (n,x,f,w,iop,int,wk)
c
c     arguments
c
c     on input               n
c     the number of data points.  n must be at
c     least 4.
c
c     x
c     table of n independent variable values in
c     ascending order.  dimension of x in calling
c     program must be at least n.
c
c     f
c     table of n corresponding dependent variable
c     values.  the values are separated by interval
c     int.  this is usually unity for
c     one-dimensional interpolation.  other values
c     are useful when coeff1 is incorporated in a
c     two-dimensional interpolation scheme (see
c     coeff2).  dimension of f in the calling
c     program must be at least (int*(n-1)+1).
c
c     iop
c     two element integer array defining boundary
c     conditions at x(1) and x(n) according to the
c     following code.
c
c     for iop(1)
c     = 1  second derivative given at x(1).  place
c     value of second derivative in w(1)
c     before call to coeff1.
c     = 2  first derivative given at x(1).  place
c     value of first derivative in w(1) before
c     call.
c     = 3  periodic boundary condition.  x(1) and
c     x(n) are equivalent points.  f(1) and
c     f(int*(n-1)+1) are equal.
c     = 4  the first derivative at x(1) is
c     calculated by fitting a cubic to points
c     x(1) through x(4).
c     similarly, iop(2) defines the boundary
c     condition at x(n).  when iop(2) = 1 (or 2),
c     the value of the second (or first) derivative
c     must be placed in w(int*(n-1)+1).  note that
c     if iop(1) = 3, consistency demands that
c     iop(2) = 3 also.
c
c     int
c     spacing in f and w tables.  for
c     one-dimensional interpolation this will
c     usually be unity.
c
c     wk
c     work area of dimension at least (3*n+1).
c
c     on output              w
c     table of second derivatives corresponding to
c     given x and f values.  the separation of
c     tabular entries is int (see above).
c     dimension of w in the calling program must be
c     at least (int*(n-1)+1).
c
c     the arrays x, f, w are used as input for the
c     routine terp1, which performs interpolation
c     at a given value of the independent variable.
c
c     timing                 the timing is linearly proportional to n, the
c     number of data points.
c***********************************************************************
c
c     subroutine terp1 (n,x,f,w,y,int,tab,itab)
c
c
c     dimension of           x(n),f(int*(n-1)+1),w(int*(n-1)+1),tab(3),
c     arguments              itab(3)
c
c     purpose                using the coefficients computed by coeff1,
c     this routine evaluates the function and/or
c     first and second derivatives at any point.
c
c     usage                  call terp1 (n,x,f,w,y,int,tab,itab)
c
c     arguments
c
c     on input               n
c     the number of data points.  n must be at
c     least 4.
c
c     x
c     table of n independent variable values in
c     ascending order.  dimension of x in the
c     calling program must be at least n.
c
c     f
c     table of n corresponding dependent variable
c     values separated by interval int, usually
c     unity for one-dimensional interpolation.
c     dimension of f in the calling program must be
c     at least (int*(n-1)+1).
c
c     w
c     table of second derivatives computed by
c     coeff1.  the separation of tabular entries is
c     int.  dimension of w in the calling program
c     must be at least (int*(n-1)+1).
c
c     y
c     value of the independent variable at which
c     interpolation is required.  if y lies outside
c     the range of the table, extrapolation takes
c     place.
c
c     int
c     spacing of tabular entries in f and w arrays.
c     this is usually unity for one-dimensional
c     interpolation.
c
c     itab
c     three element integer array defining
c     interpolation to be performed at y.
c     if itab(1) = 1, the function value is
c     returned in tab(1).
c     if itab(2) = 1, the first derivative is
c     returned in tab(2).
c     if itab(3) = 1, the second derivative is
c     returned in tab(3).
c     if itab(i) = 0 for i = 1, 2 or 3, the
c     corresponding function value or derivative is
c     not computed and tab(i) is not referenced.
c
c     on output              tab
c     three element array in which interpolated
c     function value, first and second derivatives
c     are returned as dictated by itab (see above).
c
c     timing                 this procedure is fast.  the maximum time for
c     the binary search is proportional to alog(n).
c     the time for function and derivative evaluation
c     is independent of n.
c***********************************************************************
c
c     subroutine coeff2 (nx,x,ny,y,f,fxx,fyy,fxxyy,idm,ibd,wk)
c
c
c     dimension of           x(nx),y(ny),f(idm,ny),fxx(idm,ny),fyy(idm,ny),
c     arguments              fxxyy(idm,ny),ibd(4),wk(3*max0(nx,ny)+1)
c     (idm must be .ge. nx)
c
c     purpose               :subroutine coeff2 computes the coefficients
c     for two-dimensional bicubic spline
c     interpolation with the same choice of
c     boundary conditions as for coeff1.  terp2
c     is called to perform interpolation.
c
c     usage                  call coeff2 (nx,x,ny,y,f,fxx,fyy,fxxyy,idm,ibd,
c     wk)
c
c     arguments
c
c     on input               nx
c     number of grid points in the x-direction.  nx
c     must be at least 4.
c
c     x
c     table of nx values of the first independent
c     variable arranged in ascending order.
c     dimension of x in the calling program must be
c     at least nx.
c
c     ny
c     number of grid points in the y-direction.  ny
c     must be at least 4.
c
c     y
c     table of ny values of the second independent
c     variable arranged in ascending order.
c     dimension of y in the calling program must be
c     at least ny.
c
c     f
c     two dimensional array of function values at
c     the grid points defined by the arrays x and
c     y.  dimension of f in the calling program is
c     (idm, nyy) where
c     idm .ge. nx
c     nyy .ge. ny
c
c     idm
c     first dimension in the calling program of
c     arrays f, fxx, fyy, fxxyy.  idm must be at
c     least nx.
c
c     ibd
c     four element integer array defining boundary
c     conditions according to the following code.
c     for ibd(1)
c     = 1  the second derivative of f with respect
c     to x is given at (x(1),y(j)) for
c     j = 1,ny,1.  values of this second
c     derivative must be placed in fxx(1,j)
c     for j = 1,ny,1, before calling coeff2.
c     = 2  the first derivative of f with respect
c     to x is given at (x(1),y(j)) for
c     j = 1,ny,1.  values of the derivative
c     must be placed in fxx(1,j) for
c     j = 1,ny,1 before calling coeff2.
c     = 3  periodic boundary condition in the
c     x-direction.  (x(1),y(j)) and
c     and (x(nx),y(j)) are equivalent points
c     for j = 1,ny,1.  f(1,j) and f(nx,j) are
c     equal.
c     = 4  the first derivative of f with respect
c     to x at (x(1),y(j)) is computed by
c     fitting a cubic to f(1,j) through f(4,j)
c     for j = 1,ny,1.
c
c     similarly, ibd(2) defines the boundary
c     condition at (x(nx),y(j)) for j = 1,ny,1.
c     when ibd(2) = 1 (or 2) the values of the
c     second (or first) derivative of f with
c     respect to x are placed in fxx(nx,j) for
c     j = 1,ny,1.
c     note that if ibd(1) = 3, consistency
c     requires that ibd(2) = 3 also.
c     for ibd(3)
c     = 1  the second derivative of f with respect
c     to y is given at (x(i),y(1)).  place
c     values of the derivative in fyy(i,1) for
c     i = 1,nx,1 before calling coeff2.
c     = 2  the first derivative of f with respect
c     to y is given at (x(i),y(1)).  values of
c     this derivative must be placed in
c     fyy(i,1) for i = 1,nx,1 before calling
c     coeff2.
c     = 3  periodic boundary condition in the
c     y-direction.  (x(i),y(1)) and
c     (x(i),y(ny)) are equivalent points.
c     f(i,1) and f(i,ny) are equal.
c     = 4  the first derivative of f with respect
c     to y at (x(i),y(1)) is computed by
c     fitting a cubic to f(i,1) through f(i,4)
c     for i = 1,nx,1.
c
c     similary, ibd(4) defines the boundary
c     condition at (x(i),y(ny)) for i = 1,nx,1 and
c     given derivative values are placed in
c     fyy(i,ny).
c     note that consistency demands that if
c     ibd(3) = 3, then ibd(4) = 3 also.
c
c     wk
c     work area of dimension at least
c     (3*max0(nx,ny)+1)
c
c     on output              fxx
c     array of second derivatives of f with respect
c     to x computed by coeff2.  fxx(i,j) is
c     derivative at (x(i),y(j)).  as for f,
c     dimension of fxx in the calling program is
c     (idm,nyy).
c
c     fyy
c     array of second derivatives of f with respect
c     to y computed by coeff2.  dimension of fyy in
c     the calling program is (idm,nyy).
c
c     fxxyy
c     array of fourth derivatives
c     (d/dx)**2*(d/dy)**2*f, computed by coeff2.
c     dimension of fxxyy in the calling program is
c     (idm,nyy).
c
c     the arrays x, y, f, fxx, fyy, fxxyy are used as
c     input for the routine terp2 which performs
c     interpolation at required values of the two
c     independent variables.
c
c     timing                 the timing is proportional to nx*ny.
c***********************************************************************
c
c     function terp2 (xb,yb,nx,x,ny,y,f,fxx,fyy,fxxyy,idm,ixd,iyd)
c
c
c     dimension of           x(nx),y(ny),f(idm,ny),fxx(idm,ny),fyy(idm,ny),
c     arguments              fxxyy(idm,ny))
c     (idm must be .ge. nx)
c
c     purpose                using the coefficients produced by coeff2,
c     this routine evaluates the function on a
c     selected derivative of any point where
c     two-dimensional interpolation is required.
c
c     usage                  r = terp2 (xb,yb,nx,x,ny,y,f,fxx,fyy,fxxyy,idm,
c     ixd,iyd)
c
c     arguments
c
c     on input               xb, yb
c     values of the independent variables, x and y,
c     at which interpolation is required.
c
c     nx
c     number of grid points in the x-direction.  nx
c     must be at least 4.
c
c     x
c     table of nx values of the independent
c     variable, x, arranged in ascending order.
c     dimension of x in the calling program must be
c     at least nx.
c
c     ny
c     number of grid points in the y-direction.  ny
c     must be at least 4.
c
c     y
c     table of ny values of the independent
c     variable, y, arranged in ascending order.
c     dimension of y in the calling program must be
c     at least ny.
c
c     f
c     two-dimensional array of function values at
c     grid points defined by the arrays x and y.
c     dimension of f in the calling program is
c     (idm,nyy), where
c     idm .ge. nx
c     nyy .ge. ny
c
c     fxx
c     array of second derivatives of f with respect
c     to x computed by coeff2.  dimension of fxx in
c     the calling program is (idm,nyy).  see under
c     f above.
c
c     fyy
c     array of second derivatives of f with respect
c     to y computed by coeff2.  dimension of fyy in
c     the calling program is (idm,nyy).
c
c     fxxyy
c     array of fourth derivatives,
c     (d/dx)**2*(d/dy)**2*f, computed by coeff2.
c     dimension of fxxyy in the calling program is
c     (idm,nyy).
c
c     idm
c     first dimension in the calling program of
c     arrays f, fxx, fyy and fxxyy,
c     idm .ge. nx
c
c     ixd, iyd
c     define derivative to be returned by the
c     function terp2.  ixd, iyd may each take the
c     the values 0, 1, 2.  the derivative returned
c     is (d/dx)**ixd*(d/dy)**iyd*f.
c     note that if ixd = iyd = 0, the function
c     value itself is returned.
c
c     timing                 this procedure is fast.  the maximum
c     time for the binary search is proportional to
c     alog(nx*ny).  the time for function evaluation
c     is independent of n.
c***********************************************************************
      subroutine coeff1 (n,x,f,w,iop,int,wk)
      implicit integer (i-n), real*8 (a-h,o-z)
      dimension       x(n)       ,f(n*int)       ,w(n*int)      ,iop(2),
     1  wk(n,*)
cdir$ nobounds
      logical q8q4
      save q8q4
      data q8q4 /.true./
c
c     arithmetic statement function used to locate entries in f and w arrays
c
!YuP[2022-12] was statement func: ii(index)=(index-1)*int+1
c
c     the following call is for gathering statistics on library use at ncar
      if (q8q4) then
        q8q4 = .false.
      endif
c
c     start to set up tridiagonal system
c
      j0 = 1
      do 101 i=2,n
        jm = j0
        j0 = j0+int
        wk(i,1) = x(i)-x(i-1)
        wk(i,2) = (f(j0)-f(jm))/wk(i,1)
        wk(i,3) = wk(i,1)/6.
        wk(i,1) = wk(i,1)/3.
 101  continue
      nn = n
      mk = iop(1)
      ml = iop(2)
c
c     apply boundary conditions at boundary 1
c
      go to (102,103,104,105),mk
c
c     second derivative given at boundary 1
c
 102  continue
      wk(2,2) = wk(3,2)-wk(2,2)-wk(2,3)*w(1)
      wk(2,3) = 0.
      wk(2,1) = wk(2,1)+wk(3,1)
      i1 = 2
      nn = nn-1
      go to 106
c
c     first derivative given at boundary 1
c
 103  continue
      wk(1,2) = wk(2,2)-w(1)
      wk(2,2) = wk(3,2)-wk(2,2)
      wk(1,3) = 0.
      wk(1,1) = wk(2,1)
      wk(2,1) = wk(2,1)+wk(3,1)
      i1 = 1
      go to 106
c
c     periodic boundary condition
c
 104  continue
      y2 = wk(2,2)
      b2 = wk(2,1)
      wk(2,2) = wk(3,2)-wk(2,2)
      wk(2,1) = wk(3,1)+wk(2,1)
      i1 = 2
      nn = nn-1
      go to 106
c
c     first derivative at boundary 1 from 4 point interpolation.
c
 105  continue
      a12 = x(1)-x(2)
      a13 = x(1)-x(3)
      a14 = x(1)-x(4)
      a23 = x(2)-x(3)
      a24 = x(2)-x(4)
      a34 = x(3)-x(4)
      j1 = 1
      j2 = j1+int
      j3 = j2+int
      j4 = j3+int
      w(1)    = (1./a12+1./a13+1./a14)*f(j1)-
     1  a13*a14/(a12*a23*a24)*f(j2)+a12*a14/(a13*a23*a34)*f(j3)-
     2  a12*a13/(a14*a24*a34)*f(j4)
      go to 103
c     compute tridiagonal arrays
 106  continue
      i2 = n-2
      do 107 i=3,i2
        wk(i,2) = wk(i+1,2)-wk(i,2)
        wk(i,1) = wk(i+1,1)+wk(i,1)
 107  continue
c
c     apply boundary conditions at boundary 2.
c
      in =(n-1)*int+1 !YuP[2022-12] was statement func: ii(n)=(n-1)*int+1
      go to (108,109,110,111),ml
c
c     second derivative given at boundary 2.
c
 108  continue
      wk(n-1,2) = wk(n,2)-wk(n-1,2)-wk(n,3)*w(in)
      wk(n,3) = 0.
      wk(n-1,1) = wk(n-1,1)+wk(n,1)
      nn = nn-1
      go to 112
c
c     first derivative given at boundary 2.
c
 109  continue
      wk(n-1,2) = wk(n,2)-wk(n-1,2)
      wk(n,2) = -wk(n,2)+w(in)
      wk(n-1,1) = wk(n-1,1)+wk(n,1)
      wk(1,4) = 0.
      go to 112
c
c     periodic boundary condition
c
 110  continue
      wk(n-1,2) = wk(n,2)-wk(n-1,2)
      wk(n,2) = y2-wk(n,2)
      wk(n-1,1) = wk(n-1,1)+wk(n,1)
      wk(n,1) = wk(n,1)+b2
      wk(1,4) = wk(2,3)
      go to 112
c
c     first derivative at boundary 2 from 4 point interpolation.
c
 111  continue
      a12 = x(n)-x(n-1)
      a13 = x(n)-x(n-2)
      a14 = x(n)-x(n-3)
      a23 = x(n-1)-x(n-2)
      a24 = x(n-1)-x(n-3)
      a34 = x(n-2)-x(n-3)
      j1 = in
      j2 = j1-int
      j3 = j2-int
      j4 = j3-int
      w(in)   = (1./a12+1./a13+1./a14)*f(j1)-
     1  a13*a14/(a12*a23*a24)*f(j2)+a12*a14/(a13*a23*a34)*f(j3)-
     2  a12*a13/(a14*a24*a34)*f(j4)
      go to 109
 112  continue
      iw1 =(i1-1)*int+1 !YuP[2022-12] was statement func: ii(i1)=(i1-1)*int+1
      call trip (nn,wk(i1,3),wk(i1,1),wk(i1+1,3),wk(i1,2),w(iw1),int)
      go to (114,114,113,114),mk
 113  continue
      w(1) = w(in)
 114  continue
cdir$ bounds
      return
      end subroutine coeff1
      
      
      subroutine coeff2 (nx,x,ny,y,f,fxx,fyy,fxxyy,idm,ibd,wk)
      implicit integer (i-n), real*8 (a-h,o-z)
c
cBH070805      dimension       x(nx)       ,y(ny)       ,f(idm,ny)  ,fxx(idm,ny),
cBH070805     1  fyy(idm,ny) ,fxxyy(idm,ny)           ,ibd(4)     ,
cBH070805     2  iloc(2)    ,jloc(2)
      dimension       x(*)       ,y(*)       ,f(idm,*)  ,fxx(idm,*),
     1  fyy(idm,*) ,fxxyy(idm,*)           ,ibd(4)     ,
     2  iloc(2)    ,jloc(2)      ,wk(*)
      logical q8q4
      save q8q4
      save iloc,jloc
      data q8q4 /.true./
      data iloc(1),iloc(2),jloc(1),jloc(2)/1,1,4,4/
c     the following call is for gathering statistics on library use at ncar
      if (q8q4) then
        q8q4 = .false.
      endif
c
c     compute fxx
c
      do 101 j=1,ny
        call coeff1 (nx,x,f(1,j),fxx(1,j),ibd(1),1,wk)
 101  continue
c
c     compute fyy
c
      do 102 i=1,nx
        call coeff1 (ny,y,f(i,1),fyy(i,1),ibd(3),idm,wk)
 102  continue
c
c     check for periodic boundary condition in both directions
c
      if (ibd(1) .eq. 3) go to 103
      if (ibd(3) .eq. 3) go to 105
c
c     calculate fxxyy along left and right boundaries
c
      call coeff1 (ny,y,fxx(1,1),fxxyy(1,1),jloc,idm,wk)
      call coeff1 (ny,y,fxx(nx,1),fxxyy(nx,1),jloc,idm,wk)
      go to 106
 103  continue
c
c     periodic in x direction . calculate fxxyy along lower and upper
c     boundaries.
c
      call coeff1 (nx,x,fyy(1,1),fxxyy(1,1),ibd(1),1,wk)
      call coeff1 (nx,x,fyy(1,ny),fxxyy(1,ny),ibd(1),1,wk)
c
c     calculate remaining fxxyy
c
      do 104 i=1,nx
        call coeff1 (ny,y,fxx(i,1),fxxyy(i,1),iloc,idm,wk)
 104  continue
      go to 108
 105  continue
c
c     periodic in y direction. calculate fxxyy along left and right
c     boundaries.
c
      call coeff1 (ny,y,fxx(1,1),fxxyy(1,1),ibd(3),idm,wk)
      call coeff1 (ny,y,fxx(nx,1),fxxyy(nx,1),ibd(3),idm,wk)
 106  continue
c
c     calculate remaining fxxyy
c
      do 107 j=1,ny
        call coeff1 (nx,x,fyy(1,j),fxxyy(1,j),iloc,1,wk)
 107  continue
 108  continue
      return
      end subroutine coeff2
      
      subroutine intrp (n,x,f,w,y,i,int,tab,itab)
      implicit none !integer (i-n), real*8 (a-h,o-z)
      integer n,i,int !IN
      real*8 y
      integer i0,ip !local
      real*8 flk,flp,fl0,a,b,c
      real*8       x(i+1)    ,f(i*int+1)    ,w(i*int+1)  ,tab(3) !IN
      integer itab(3) !IN
c
c     arithmetic statement function used to locate entries in f and w arrays
c
!YuP[2022-12] was statement func:  ii(index)=(index-1)*int+1
c
c     perform interpolation or extrapolation
c
      flk = x(i+1)-x(i)
      flp = x(i+1)-y
      fl0 = y-x(i)
      i0 = (i-1)*int+1 !YuP[2022-12] was statement func [used once] ii(i)
      ip = i0+int
      if (itab(1) .ne. 0) go to 101
      go to 102
 101  continue
c
c     calculate f(y)
c
      a = (w(i0)*flp**3+w(ip)*fl0**3)/(6.*flk)
      b = (f(ip)/flk-w(ip)*flk/6.)*fl0
      c = (f(i0)/flk-w(i0)*flk/6.)*flp
      !-YuP  Note: If w==0 (set all 2nd derivatives to zero),
      !-YuP  then a+b+c = f(i) + [y-x(i)]*[f(i+1)-f(i)]/[x(i+1)-x(i)]
      !-YuP  which is just a linear interpolation. 
      tab(1) = a+b+c
 102  continue
      if (itab(2) .ne. 0) go to 103
      go to 104
 103  continue
c
c     calculate first derivative at y
c
      a = (w(ip)*fl0**2-w(i0)*flp**2)/(2.*flk)
      b = (f(ip)-f(i0))/flk
      c = (w(i0)-w(ip))*flk/6.
      tab(2) = a+b+c
 104  continue
      if (itab(3) .ne. 0) go to 105
      go to 106
 105  continue
c
c     calculate second derivative at y
c
      tab(3) = (w(i0)*flp+w(ip)*fl0)/flk
 106  continue
      return
      end subroutine intrp
      
      
      subroutine search (xbar,x,n,i)
      implicit none !integer (i-n), real*8 (a-h,o-z)
      real*8 xbar !IN
      real*8 x(n) !IN
      integer n,i !IN, OUT
      integer k,m,nm1 !local
      real*8 b !local
      save b
      data b/.69314718/
c
c     if xbar is outside range of x table extrapolate
c
      if (xbar .gt. x(2)) go to 101
      i = 1
      return
 101  continue
      if (xbar .lt. x(n-1)) go to 102
      i = n-1
      return
 102  continue
c
c     find maximum power of two less than n
c
      m = int((log(dfloat(n)))/b) !YuP[2022-12-19] was FLOAT, alog()
      i = 2**m
      if (i .ge. n) i = i/2
      k = i
      nm1 = n-1
c
c     conduct binary search.
c
 103  continue
      k = k/2
      if (xbar .ge. x(i)) go to 104
      i = i-k
      go to 103
 104  continue
      if (xbar .le. x(i+1)) return
      i = min0(i+k,nm1)
      go to 103
      end subroutine search
      
      
      subroutine terp1 (n,x,f,w,y,int,tab,itab)
      implicit integer (i-n), real*8 (a-h,o-z)
c
      real*8  x(n)       ,f(n*int)       ,w(n*int)    ,tab(3) !IN
      integer itab(3) !IN
c     the following call is for gathering statistics on library use at ncar
      logical q8q4
      save q8q4
      data q8q4 /.true./
      if (q8q4) then
        q8q4 = .false.
      endif
c
c     perform search
c
      call search (y,x,n,i)
c
c     carry out interpolation (or extrapolation)
c
      call intrp (n,x,f,w,y,i,int,tab,itab)
      return
      end subroutine terp1


      real*8 function terp2 
     +     (xb,yb,nx,x,ny,y,f,fxx,fyy,fxxyy,idm,ixd,iyd)
      implicit integer (i-n), real*8 (a-h,o-z)
c
      dimension       x(nx)      ,y(ny)      ,f(idm,ny)  ,fxx(idm,ny),
     1  fyy(idm,ny) ,fxxyy(idm,ny)           ,ff(2)      ,
     2  ww(2)      ,tab(3)     ,itab(3)
c     the following call is for gathering statistics 
c       on library use at ncar
      logical q8q4
      save q8q4
      data q8q4 /.true./
      if (q8q4) then
        q8q4 = .false.
      endif
c
c     search in x and y arrays.
c
      call search (xb,x,nx,i)
      call search (yb,y,ny,j)
c
c     interpolate in x direction
c
      do 101 i1=1,3
        itab(i1) = 0
 101  continue
      i1 = ixd+1
      itab(i1) = 1
      do 102 j1=1,2
        jj = j+j1-1
        call intrp (n,x,f(1,jj),fxx(1,jj),xb,i,1,tab,itab)
        ff(j1) = tab(i1)
        call intrp (n,x,fyy(1,jj),fxxyy(1,jj),xb,i,1,tab,itab)
        ww(j1) = tab(i1)
 102  continue
c
c     interpolate in y direction
c
      do 103 j1=1,3
        itab(j1) = 0
 103  continue
      j1 = iyd+1
      itab(j1) = 1
      call intrp (2,y(j),ff,ww,yb,1,1,tab,itab)
      terp2 = tab(j1)
      return
c
c     revision history---
c
c     june 1977        replaced non-standard statement functions and
c     subscripts to enhance portability.
c
c     january 1978     deleted references to the  *cosy  cards, moved
c     the revision histories to appear before the
c     final end card, and moved the initial comment
c     cards to appear after the first subroutine card
c     and changed  itab  from logical to integer in
c     subroutine intrp and corrected problem with
c     version numbers in one statistics call
c-----------------------------------------------------------------------
      end
      
      subroutine searche (xbar,x,n,i,dx)
      implicit integer (i-n), real*8 (a-h,o-z)
      dimension       x(n)
      save b
      data b/.69314718/
c
c     if xbar is outside range of x table extrapolate
c
      if (xbar .gt. x(2)) go to 101
      i = 1
      return
 101  continue
      if (xbar .lt. x(n-1)) go to 102
      i = n-1
      return
 102  continue
c..................................................................
c     This version knows data is evenly spaced with spacing dx
c..................................................................

      i=(xbar-x(1))/dx+1

      return
      end
      subroutine terp1e (n,x,f,w,y,int,tab,itab,dx)
      implicit integer (i-n), real*8 (a-h,o-z)
c
      dimension       x(n)       ,f(n*int)   ,w(n*int)    ,tab(3)     ,
     1  itab(3)
c     the following call is for gathering statistics on library use at ncar
      logical q8q4
      save q8q4
      data q8q4 /.true./
      if (q8q4) then
        q8q4 = .false.
      endif
c
c     perform search
c
      call searche (y,x,n,i,dx)
c
c     carry out interpolation (or extrapolation)
c
      call intrp (n,x,f,w,y,i,int,tab,itab)
      return
      end


c..................................................................
c     This version knows data is evenly spaced with spacing dx
c..................................................................
      real*8 function t2
     +     (dx,dy,xb,yb,nx,x,ny,y,f,fxx,fyy,fxxyy,idm,ixd,iyd)
      implicit integer (i-n), real*8 (a-h,o-z)
c
      dimension       x(nx)     ,y(ny)     ,f(idm,ny)  ,fxx(idm,ny) ,
     1  fyy(idm,ny)  ,fxxyy(idm,ny)        ,ff(2)      ,
     2  ww(2)      ,tab(3)     ,itab(3)
c     the following call is for gathering statistics 
c       on library use at ncar
      logical q8q4
      save q8q4
      data q8q4 /.true./
      if (q8q4) then
        q8q4 = .false.
      endif
c
c     search in x and y arrays.
c
      call searche (xb,x,nx,i,dx)
      call searche (yb,y,ny,j,dy)
c
c     interpolate in x direction
c
      do 101 i1=1,3
        itab(i1) = 0
 101  continue
      i1 = ixd+1
      itab(i1) = 1
      do 102 j1=1,2
        jj = j+j1-1
        call intrp (n,x,f(1,jj),fxx(1,jj),xb,i,1,tab,itab)
        ff(j1) = tab(i1)
        call intrp (n,x,fyy(1,jj),fxxyy(1,jj),xb,i,1,tab,itab)
        ww(j1) = tab(i1)
 102  continue
c
c     interpolate in y direction
c
      do 103 j1=1,3
        itab(j1) = 0
 103  continue
      j1 = iyd+1
      itab(j1) = 1
      call intrp (2,y(j),ff,ww,yb,1,1,tab,itab)
      t2 = tab(j1)
      return
c
c     revision history---
c
c     june 1977        replaced non-standard statement functions and
c     subscripts to enhance portability.
c
c     january 1978     deleted references to the  *cosy  cards, moved
c     the revision histories to appear before the
c     final end card, and moved the initial comment
c     cards to appear after the first subroutine card
c     and changed  itab  from logical to integer in
c     subroutine intrp and corrected problem with
c     version numbers in one statistics call
c-----------------------------------------------------------------------
      end
      
      
      subroutine trip (n,a,b,c,y,z,int)
      implicit integer (i-n), real*8 (a-h,o-z)
      dimension       a(n)       ,b(n)       ,c(n)       ,y(n)       ,
     1  z(n*int)
c
c     arithmetic statement function used to locate entries in array z.
c
!YuP[2022-12] was statement func: ii(index)=(index-1)*int+1
c
c     gaussian elimination
c
      bn = b(n)
      yn = y(n)
      v = c(n)
      y(1) = y(1)/b(1)
      a(1) = a(1)/b(1)
      b(1) = c(1)/b(1)
      nm2 = n-2
      do 101 j=2,nm2
        den = b(j)-a(j)*b(j-1)
        b(j) = c(j)/den
        y(j) = (y(j)-a(j)*y(j-1))/den
        a(j) = -a(j)*a(j-1)/den
        bn = bn-v*a(j-1)
        yn = yn-v*y(j-1)
        v = -v*b(j-1)
 101  continue
      den = b(n-1)-a(n-1)*b(n-2)
      b(n-1) = (c(n-1)-a(n-1)*a(n-2))/den
      y(n-1) = (y(n-1)-a(n-1)*y(n-2))/den
      bn = bn-v*a(n-2)
      yn = yn-v*y(n-2)
      v = a(n)-v*b(n-2)
c     back substitution
      iin = (n-1)*int+1 !YuP[2022-12] was statement func: ii(n)=(n-1)*int+1
      z(iin) = (yn-v*y(n-1))/(bn-v*b(n-1))
      iin2 =(n-2)*int+1 !YuP[2022-12] was statement func: ii(n-1)=(n-2)*int+1
      z(iin2) = y(n-1)-b(n-1)*z(iin)
      nm1 = n-1
      in = (n-1)*int+1 !YuP[2022-12] was statement func: ii(n)=(n-1)*int+1
      do 102 j=2,nm1
        k = n-j
        ik = (k-1)*int+1 !YuP[2022-12] was statement func: ii(k)=(k-1)*int+1
        ikt = ik+int
      z(ik) = y(k)-b(k)*z(ikt)-a(k)*z(in)
      if(j.eq.(-n)) write(*,140) b(k), y(k), a(k), z(ikt),z(ik)
 102  continue
 140  format( '         cj_debug: trip', 5e16.8)
      return
      end
c************************************************************************
c     END OF SPLINES
c************************************************************************
c
c
      real*8 function erf(xxx)
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c------------------------------------------------------
c     This routine computes the ERROR FUNCTION.
c------------------------------------------------------
      common /tusq/ tusqpi
      dimension a(5)
      sign=1.
      if (xxx .lt. 0.) sign=-1.
      xcg=sign*xxx
      x2=xcg*xcg
      if (xcg .ge. .6) go to 20
      sum=xcg
      term=xcg
      kmax=6
      do 10 k=1,kmax
        t1=dfloat(k)
        t2=dfloat(2*k+1)/dfloat(2*k-1)
        term=-term*x2/(t1*t2)
        sum=sum+term
 10   continue
      erf=tusqpi*sum
      erf=sign*erf
      return
 20   continue
      p=.3275911
      a(1)=.225836846
      a(2)=-.252128668
      a(3)=1.25969513
      a(4)=-1.287822453
      a(5)=.94064607
      eta=1./(1.+p*xcg)
      phip=tusqpi*exp(-x2)
      term=(((a(5)*eta+a(4))*eta+a(3))*eta+a(2))*eta+a(1)
      erf=1.-term*eta*phip
      erf=sign*erf
      return
      end


c######date01jan1984     copyright ukaea, harwell.
c######aliasim01ad
      real*8 function im01ad(la,a,inc)
      implicit integer (i-n), real*8 (a-h,o-z)
      real*8 a(la*inc)
      zero=0.d0
      kount = 0
      do 100 k=1,la
        ipos = (k-1)*inc + 1
        if (a(ipos).eq.zero) kount = kount + 1
 100  continue
      im01ad = kount
      return
      end

c
c
c**********************************************************
      subroutine zzbeslri(x,nb,ize,b,ncalc)
      implicit integer (i-n), real*8 (a-h,o-z) ! DOUBLE NOW !
cmnt  this routine calculates Bessel functions I and J of real
cmnt  argument and integer order.


cmnt  explanation of variables in the calling sequence

cmnt  x (was single) real*8 real argument for which i,s or j,s
cmnt  are to be calculated.  if i,s are to be calculated,
cmnt  abs(x) must be less than exparg (which see below).

cmnt  nb      integer type.  1+highest order to be calculated.
cmnt  it must be positive.

cmnt  ize     integer type.  zero if j,s are to be calculated,
cmnt  1 if i,s are to be calculated.

cmnt  b (was single) real*8 vector of length nb, need not be
cmnt  initialized by user.  if the routine terminates
cmnt  normally (ncalc=nb), it returns j(or i) -sub-zero
cmnt  through j(or i) -sub-nb-minus-one of x in this
cmnt  vector.

cmnt  ncalc  integer type, need not be initialized by user.
cmnt  before using the results, the user should check that
cmnt  ncalc=nb.  i.e. all orders have been calculated to
cmnt  the desired accuracy.  see error returns below.


cmnt  explanation of machine-dependent constants

cmnt  nsig    decimal significance desired.  should be set to
cmnt  ifix(alog10(2)*nbit+1), where nbit is the number of
cmnt  bits in the mantissa of a real*8 variable.
cmnt  setting nsig lower will result in decreased accuracy
cmnt  while setting nsig higher will increase cpu time
cmnt  without increasing accuracy.  the truncation error
cmnt  is limited to t=.5*10**-nsig for j,s of order less
cmnt  than argument, and to a relative error of t for
cmnt  i,s and the j,s.

cmnt  nten    largest integer k such that 10**k is machine-
cmnt  representable in single precision.

cmnt  largex  upper limit on the magnitude of x.  bear in mind
cmnt  that if abs(x)=n, then at least n iterations of the
cmnt  backward recursion will be executed.

cmnt  exparg  largest single precision argument that the library
cmnt  exp  routine can handle.


cmnt  error returns


cmnt  let g denote either i or j.
cmnt  in case of an error, ncalc.ne.nb, and not all g,s
cmnt  are calculated to the desired accuracy.
cmnt  if ncalc.lt.0, an argument is out of range.  nb.le.0
cmnt  or ize is neither 0 nor 1 or ize=1 and abs(x).ge.exparg.
cmnt  in this case, the b-vector is not calculated, and ncalc
cmnt  is set to min0(nb,0)-1 so ncalc.ne.nb.

cmnt  nb.gt.ncalc.gt.0 will occur if nb.gt.magx and abs(g-
cmnt  sub-nb-of-x/g-sub-magx-of-x).lt.10.**(nten/2), i.e.
cmnt  nb is much greater than magx.  in this case, b(n) is calcu
cmnt  lated to the desired accuracy for n.le.ncalc, but for
cmnt  ncalc.lt.n.le.nb, precision is lost.  if n.gt.ncalc and
cmnt  abs(b(ncalc)/b(n)).eq.10.**-k, then only the first nsig-k
cmnt  significant figures of b(n) may be trusted.  if the user
cmnt  wishes to calculate b(n) to higher accuracy, he should use
cmnt  an asymptotic formula for large order.



      dimension b(nb)
      save nsig,nten,largex,exparg
cBH150620      data nsig,nten,largex,exparg/14,307,100000,7.e2/
cBH150620:  Change error tolerance, for very high harmonic cases.
      data nsig,nten,largex,exparg/14,307,100000,7.e2/
      one=1.d0
      ten=10.d0
      tempa= abs(x)
cmnt  magx=ifix(sngl(tempa))
      magx=tempa
      if (nb.gt.0.and.magx.le.largex.and.(ize.eq.0.or.
     1  (ize.eq.1.and.tempa.le.exparg)))goto 10
cmnt  error return -- x,nb,or ize is out of range.
      ncalc=min0(nb,0)-1
      return
 10   sign=1-2*ize
      ncalc=nb
cmnt  use 2-term ascending series for small x
c990131      if (tempa**4.lt..1e0**nsig) goto 230
      if (tempa**4.lt.(.1e0*one)**nsig) goto 230
cmnt  initialize the calculation of p,s
      nbmx=nb-magx
      n=magx+1
      plast=1.d0
      p=(n+n)/tempa
cmnt  calculate general significance test
c990131      test=2.e0*1.e1**nsig
      test=2.e0*ten**nsig
      if (ize.eq.1.and.2*magx.gt.5*nsig) test= sqrt(test*p)
      one585=1.585
c990131      if (ize.eq.1.and.2*magx.le.5*nsig) test=test/1.585**magx
      if (ize.eq.1.and.2*magx.le.5*nsig) test=test/one585**magx
      m=0
      if (nbmx.lt.3) goto 30
cmnt  calculate p,s until n=nb-1.  check for possible overflow.
c990131      tover=1.e1**(nten-nsig)
      tover=ten**(nten-nsig)
      nstart=magx+2
      nend=nb-1
      do 20 n=nstart,nend
        pold=plast
        plast=p
        p=(n+n)*plast/tempa-sign*pold
        if (p-tover) 20,20,40
 20   continue
      n=nend
cmnt  calculate special significance test for nbmx.gt.2.
c990131      test=amax1(test, sqrt(plast*1.e1**nsig)* sqrt(2.e0*p))
      test=max(test, sqrt(plast*ten**nsig)* sqrt(2.e0*p))
cmnt  calculate p,s until significance test passes.
 30   n=n+1
      pold=plast
      plast=p
      p=(n+n)*plast/tempa-sign*pold
      if (p.lt.test) goto 30
      if (ize.eq.1.or.m.eq.1) goto 90
cmnt  for j*s, a strong variant of the test is necessary.
cmnt  calculate it, and calculate p*s until this test is passed.
      m=1
      tempb=p/plast
      tempc=(n+1)/tempa
      if (tempb+1.e0/tempb.gt.2.e0*tempc) tempb=tempc+ sqrt
     1  (tempc**2-1.e0)
      test=test/ sqrt(tempb-1.e0/tempb)
      if (p-test) 30,90,90
cmnt  to avoid overflow, divide p*s by tover.  calculate p*s
cmnt  until abs(p).gt.1).
 40   tover=ten**nten
      p=p/tover
      plast=plast/tover
      psave=p
      psavel=plast
      nstart=n+1
 50   n=n+1
      pold=plast
      plast=p
      p=(n+n)*plast/tempa-sign*pold
      if (p.le.1.e0) goto 50
      tempb=(n+n)/tempa
      if (ize.eq.1) goto 60
      tempc=.5e0*tempb
      tempb=plast/pold
      if (tempb+1.e0/tempb.gt.2.e0*tempc) tempb=tempc+ sqrt
c990131     1  (tempc**2-1.e0)
     1  (tempc**2-one)
cmnt  calculate backward test, and find ncalc, the highest n
cmnt  such that the test is passed.
c990131 60   test=.5e0*pold*plast*(1.e0-1.e0/tempb**2)/1.e1**nsig
 60   test=.5e0*pold*plast*(1.e0-1.e0/tempb**2)/ten**nsig
      p=plast*tover
      n=n-1
      nend=min0(nb,n)
      do 70 ncalc=nstart,nend
        pold=psavel
        psavel=psave
        psave=(n+n)*psavel/tempa-sign*pold
        if (psave*psavel-test) 70,70,80
 70   continue
      ncalc=nend+1
 80   ncalc=ncalc-1
cmnt  the sum b(1)+2b(3)+2b(5)... is used to normalize.  m, the
cmnt  coefficient of b(n), is initialized to 2 or 0.
 90   n=n+1
      m=2*n-4*(n/2)
cmnt  initialize the backward recursion and the normalization
cmnt  sum.
      tempb=0.e0
      tempa=1.e0/p
      sum=m*tempa
      nend=n-nb
      if (nend) 140,120,100
cmnt  recur backward via difference equation.  calculating (but
cmnt  not storing) b(n), until n=nb.
 100  do 110 l=1,nend
        n=n-1
        tempc=tempb
        tempb=tempa
        tempa=(n+n)*tempb/x-sign*tempc
        m=2-m
 110  sum=sum+m*tempa
cmnt  store b(nb)
 120  b(n)=tempa
      if (nb.gt.1) goto 130
cmnt  nb=1.  since 2*tempa was added to the sum, tempa must be
cmnt  subtracted.
      sum=sum-tempa
      goto 200
cmnt  calculate and store b(nb-1)
 130  n=n-1
      b(n)=(n+n)*tempa/x-sign*tempb
      if (n.eq.1) goto 190
      m=2-m
      sum=sum+m*b(n)
      goto 160
cmnt  n.lt.nb, so store b(n) and set higher orders to zero
 140  b(n)=tempa
      nend=-nend
      do 150 l=1,nend
 150  b(n+l)=0.e0
 160  nend=n-2
      if (nend.eq.0) goto 180
cmnt  calculate via difference equation and store b(n),
cmnt  until n=2.
      do 170 l=1,nend
        n=n-1
        b(n)=(n+n)*b(n+1)/x-sign*b(n+2)
        m=2-m
 170  sum=sum+m*b(n)
cmnt  calculate b(1)
 180  b(1)=2.e0*b(2)/x-sign*b(3)
 190  sum=sum+b(1)
cmnt  normalize--if ize=1, divide sum by cosh(x).  divide all
cmnt  b(n) by sum.
 200  if (ize.eq.0) goto 210
      tempa= exp( abs(x))
      sum=2.e0*sum/(tempa+1.e0/tempa)
 210  do 220 n=1,nb
 220  b(n)=b(n)/sum
      return
cmnt  two-term ascending series for small x
 230  tempa=1.e0
      tempb=-.25e0*x*x*sign
      b(1)=1.e0+tempb
      if (nb.eq.1) goto 250
      do 240 n=2,nb
        tempa=tempa*x/(n+n-2)
 240  b(n)=tempa*(1.e0+tempb/n)
 250  return
      end

c***********************************************************************
c$$$      subroutine zzechk(nchars,narray)
c$$$      implicit integer (i-n), real*8 (a-h,o-z)
c$$$      save
c$$$
c$$$cmnt  abstract
c$$$cmnt  zzechk is a companion routine of zzrchk.  it is called
c$$$cmnt  just like zzrchk, and messages from it may be suppressed
c$$$cmnt  by an appropriate call to zzxset.  it differs from zzrchk
c$$$cmnt  in that each call to zzechk will produce no more than one
c$$$cmnt  printed message, regardless of how many times that call is
c$$$cmnt  executed, and zzechk never terminates execution.
c$$$cmnt  its purpose is to provide one-time-only informative
c$$$cmnt  diagnostics.
c$$$
c$$$cmnt  description of arguments
c$$$cmnt  nchars - number of characters in the message.
c$$$cmnt  if negated, the message will be printed (once) even
c$$$cmnt  if nfatal has been set to 0 (see zzxset).
c$$$cmnt  narray - same as in zzrchk
c$$$
c$$$
c$$$
c$$$cmnt  zzechk uses subroutines zzrget, zzrprt, zzxset, zzstgt
c$$$cmnt  compile decks zzrchk
c$$$
c$$$      dimension narray(14)
c$$$      data nflag/4h.$,*/
c$$$      if (narray(1).eq.nflag) return
c$$$      call zzrget(nf,nt)
c$$$      if ((nf.eq.0).and.(nchars.gt.0)) return
c$$$      call zzrprt (59,59hthe following informative diagnostic is
c$$$     1 printed  only once.)
c$$$      call zzrprt(iabs(nchars),narray)
c$$$      if (nf.gt.0) nf = nf-1
c$$$
c$$$cmnt  calculates psi!!, the second derivative of psi with
c$$$cmnt  respect to arc length, as a functionn of psi:  d(dpsi/dz)/dz
c$$$
c$$$      call zzxset(nf,nt)
c$$$      narray(1) = nflag
c$$$      end

c***********************************************************************
      subroutine zzrchk(nchars,narray)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

cmnt  sandia mathematical program library
cmnt  applied mathematics division 2642
cmnt  sandia laboratories
cmnt  albuquerque, new mexico 87115

cmnt  simplified version for stand-alone use.     april 1977

cmnt  abstract
cmnt  the routines zzrchk, zzxset, and zzrget together provide
cmnt  a uniform method with several options for the processing
cmnt  of diagnostics and warning messages which originate
cmnt  in the mathematical program library routines.
cmnt  zzrchk is the central routine, which actually processes
cmnt  messages.

cmnt  description of arguments
cmnt  nchars - number of characters in hollerith message.
cmnt  if nchars is negated, zzrchk will unconditionally
cmnt  print the message and stop execution.  otherwise,
cmnt  the behavior of zzrchk may be controlled by
cmnt  an appropriate call to zzxset.
cmnt  narray - name of array or variable containing the message,
cmnt  or else a literal hollerith constant containing
cmnt  the message.  by convention, all messages should
cmnt  begin with *in subnam, ...*, where subnam is the
cmnt  name of the routine calling zzrchk.

cmnt  examples
cmnt  1. to allow control by calling zzxset, use
cmnt  call zzrchk(30,30hin quad, invalid value of err.)
cmnt  2. to unconditionally print a message and stop execution, use
cmnt  call zzrchk(-30,30hin quad, invalid value of err.)



cmnt  zzrchk uses subroutines zzrget, zzrprt, zzxset, zzstgt
cmnt  compile decks zzrchk

      dimension narray(14)

      call zzrget(nf,nt)
cmnt  if zzrchk was called with negative character count, set fatal flag
      if (nchars.lt.0) nf = -1
cmnt  if messages are to be suppressed, return
      if (nf.eq.0) return
cmnt  if character count is invalid, stop
      if (nchars.eq.0) print 10010
      if (nchars.eq.0) stop 'zcunix: zzrchk'
cmnt  print message
      call zzrprt(iabs(nchars),narray)
cmnt  if last message, say so
      if (nf.eq.1) print 10020
cmnt  print trace-back if asked to
cmnt  if ((nt.gt.0).or.(nf.lt.0)) call system routine for traceback
cmnt  decrement message count
      if (nf.gt.0) nf = nf-1
      call zzxset(nf,nt)
cmnt  if all is well, return
      if (nf.ge.0) return
cmnt  if this message is suppressable by an zzxset call,
cmnt  then explain zzxset usage.
      if (nchars.gt.0) print 10030
      print 10040
      stop
10010 format(/31h zzrchk was called incorrectly.)
10020 format (30h zzrchk message limit reached.)
10030 format (/13h *** note ***,
     1  /53h to make the error message printed above be nonfatal,
     2  /39h or to suppress the message completely,
     3  /37h insert an appropriate call to zzxset,
     4  30h at the start of your program.
     5  /58h for example, to print up to 10 nonfatal warning messages,
     6  /27h  use     call zzxset(10,0)    )
10040 format (/28h program abort due to error.)
      end

c***********************************************************************
      subroutine zzrget(nfatal,ntrace)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

cmnt  abstract
cmnt  zzrget is a companion routine to subroutine zzrchk.
cmnt  zzrget assigns to nfatal and ntrace respectively the values
cmnt  of nf and nt in common block mlblk0 thereby ascertaining the
cmnt  state of the options which control the execution of zzrchk.

cmnt  description of arguments
cmnt  both arguments are output arguments of data type integer.
cmnt  nfatal - current value of nf (see description of zzxset.)
cmnt  ntrace - current value of nt (see description of zzxset.)

      call zzstgt(1,nfatal,ntrace)
      return
      end

c***********************************************************************
      subroutine zzrprt(nchars,narray)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

cmnt  utility routine to simply print the hollerith message in narray,
cmnt  whose length is nchars characters.

      dimension narray(14)

cmnt  note - nch must be the number of hollerith characters stored
cmnt  per word.  if nch is changed, format 1 must also be
cmnt  changed correspondingly.

      nch = 10
cmnt  for line printers, use
cmnt  for data terminals, use
cmnt  1 format (1x,7a10)
      nwords = (nchars+nch-1)/nch
      print 10010,(narray(i),i=1,nwords)
      return
10010 format (1x,13a10)
      end

c***********************************************************************
      subroutine zzstgt(k,nfatal,ntrace)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

cmnt  this routine is a slave to zzrget and errset which keeps
cmnt  the flags as local variables.

cmnt  *** if local variables are not normally retained between
cmnt  calls on this system, the variables lnf and lnt can be
cmnt  placed in a common block and preset to the following
cmnt  values in the main program.

      data lnf/-1/,lnt/0/
      if (k.le.0) lnf = nfatal
      if (k.le.0) lnt = ntrace
      if (k.gt.0) nfatal = lnf
      if (k.gt.0) ntrace = lnt
      return
      end

c***********************************************************************
      subroutine zzxset(nfatal,ntrace)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

cmnt  abstract
cmnt  zzxset is a companion routine to subroutine zzrchk.
cmnt  zzxset assigns the values of nfatal and ntrace respectively
cmnt  to nf and nt in common block mlblk0 thereby specifying the
cmnt  state of the options which control the execution of zzrchk.

cmnt  description of arguments
cmnt  both arguments are input arguments of data type integer.
cmnt  nfatal - is a fatal-error / message-limit flag. a negative
cmnt  value denotes that detected difficulties are to be
cmnt  treated as fatal errors.  nonnegative means nonfatal.
cmnt  a nonnegative value is the maximum number of nonfatal
cmnt  warning messages which will be printed by zzrchk,
cmnt  after which nonfatal messages will not be printed.
cmnt  (default value is -1.)
cmnt  ntrace - .ge.1 will cause a trace-back to be given,
cmnt  if this feature is implemented on this system.
cmnt  .le.0 will suppress any trace-back, except for
cmnt  cases when execution is terminated.
cmnt  (default value is 0.)

cmnt  *note* -- some calls to zzrchk will cause unconditional
cmnt  termination of execution.  zzxset has no effect on such calls.

cmnt  examples
cmnt  1. to print up to 100 messages as nonfatal warnings use
cmnt  call zzxset(100,0)
cmnt  2. to suppress all mathlib warning messages use
cmnt  call zzxset(0,0)



cmnt  zzxset uses subroutines zzstgt
cmnt  compile decks zzrchk

      call zzstgt(0,nfatal,ntrace)
      return
      end


 
c     Following two subroutines from ONETWO
      subroutine allocate_error(var,myid,istat)
c------------------------------------------------------------------
      character *(*) var
      integer istat,myid
      write(*,1)var,istat,myid
 1    format(2x,"Memory Allocation error encountered",/,
     .       2x,"Unable to allocate ",a,/,
     .       2x,"status =",i5)
      istat =0 !reset for next case
      return
      end



      subroutine deallocate_error(var,myid,istat)
c---------------------------------------------------------------------
      character *(*) var
      integer istat, myid
      write(*,1)var,istat,myid
 1    format(2x,"Memory DE-Allocation error encountered",/,
     .       2x,"Unable to deallocate ",a,/,
     .       2x,"status =",i5, " process rank =",i5)
      istat =0 !reset for next case
      return
      end
