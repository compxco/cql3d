c
c
      subroutine exsweep
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     Time advancement routine - uses splitting scheme
c     (called if implct.ne.enabled)
c..................................................................

      save
      include 'param.h'
      include 'comm.h'

      include 'advnce.h'
c.......................................................................

      navnc=navnc+1

c..................................................................
c     Copy current distribution into f_.
c..................................................................

      call dcopy(iyjx2*ngen,f(0,0,1,l_),1,f_(0,0,1,l_),1)

c..................................................................
c     loop over all time advanced species..
c..................................................................

      do 600 k=1,ngen

c..................................................................
c     normalized time step..
c..................................................................

        rbgn=1./dtr

c..................................................................
c     The routine coefstup(k) does the following:
c     Preset all coefficients to zero. da, db, and dc are the three
c     coefficients asscoiated with the velcoity (v) flux while
c     dd, de, df are associated with the theta flux. The contribution
c     from the Krook operator (cah(i,j)*f(i,j,l_)) is stored in cah.
c     Note cah does not contribute to da through df. These six
c     coefficients incorporate only wave-particle, collisional and
c     d.c. electric field effects. The ion source is represented
c     through so(i,j).
c     The routine coefstup(k) adds into the above arrays all the relevan
c     physics.
c..................................................................

        call coefstup(k)

c..................................................................
c     The coefficients of the equation are currently defined on the
c     same mesh as the distribution function f. The fluxes are best
c     defined (from the point of view of differencing and particle
c     conservation) on mid meshpoints. We undertake here to
c     interpolate the coefficients as needed to either (i,j+1/2)
c     (velocity flux) or to (i+1/2,j) (theta flux).
c     Finally to enforce boundary conditions (zero flux in general
c     except at the pass/trapped boundary) certain coefficients
c     are zeroed out or suitably averaged at specific mesh points.
c     The numbers 1,2,3 appearing in the calls below signify
c     which coefficient is being treated.
c..................................................................

c..................................................................
c     first the velocity flux..
c..................................................................

        call coefmidv(da,1)
        call coefmidv(db,2)
        call coefmidv(dc,3)

c..................................................................
c     the theta flux..
c..................................................................

        call coefmidt(dd,1)
        call coefmidt(de,2)
        call coefmidt(df,3)

c..................................................................
c     The differencing is done ala Chang and Cooper as generalized
c     by Karney to 2-D. This requires defining a couple of arrays
c     di(i,j,k,l_) and dj(i,j,k,l_) which are used to define f at mid mesh
c     points: to wit f(i,j+1/2,k,l_)=f(i,j+1,k,l_)*(1.-dj(i,j,k,l_))+ 
c     f(i,j,k,l_)*dj(i,j,k,l_). Similarly for theta.
c     The routine coefwtj and coefwti are used to
c     determine the dj and di arrays.
c..................................................................

        call coefwtj(k)
        call coefwti(k)

c..................................................................
c     copy the distribution function into temp1
c..................................................................

        call dcopy(iyjx2,f(0,0,k,l_),1,temp1(0,0),1)

c..................................................................
c     Initialize at x=0 for the forward recursion of the velocity
c     sweep. This boundary condition is a conservative 0 flux condition.
c     The quantities egg and fgg defined below in the forward recursion
c     will be used in the backward recursion to compute f (at n+1/2).
c     f(i,j,l_)=egg(i,j)*f(i,j+1,l_)+fgg(i,j)
c..................................................................

cdir$ ivdep
        do 20 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)

c..................................................................
c     da(i,0)=db(i,0)=dc(i,0)=0 forces G(i,0)=0. Thus the equation at
c     x=0 is effectively
c
c     df/dt = G(i,1)*integration coefficient
c
c     Recall the quantity G(i,j) is really defined at G(i,j+1/2).
c..................................................................

          fgg(i,1)=delx(i,1)/betx(i,1)
          egg(i,1)=alpx(i,1)/betx(i,1)
 20     continue

c..................................................................
c     Complete the forward recursion for egg and fgg
c..................................................................

        do 50 j=2,jxm1
cdir$ ivdep
          do 40 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            egg(i,j)=alpx(i,j)/(betx(i,j)-gamx(i,j)*egg(i,j-1))
            fgg(i,j)=(delx(i,j)+gamx(i,j)*fgg(i,j-1))/
     1        (betx(i,j)-gamx(i,j)*egg(i,j-1))
 40       continue
 50     continue

c..................................................................
c     A boundary equation is needed at x=xmax (j=jx). This is
c     effectively a free fall (hyperbolic) condition if characteristics
c     are leaving the system and zero flux with f(i,jx,l_) approximately
c     zero if characteristics are entering the system.
c     Note that we have arbitrarily set db(i,jx)=db(i,jx-1)=dc(i,jx)=
c     dc(i,jx-1)=0 for all i for purposes of time advancement. If
c     advection dominates (da dominates) setting diffusion equal to
c     zero does not pose a problem. If diffusion dominates then xmax
c     has to have been chosen high enough that f is close to zero. Thus
c     setting db and dc equal to zero here should not disturb the
c     solution.
c
c     Determine temc1 and temc2 such that
c
c     f(i,jx,l_)=temc1(i)*f(i,jx-1,l_)+temc2(i)
c..................................................................

cdir$ ivdep
        do 60 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
          temc1(i)=gamx(i,jx)/betx(i,jx)
          temc2(i)=delx(i,jx)/betx(i,jx)
 60     continue

c..................................................................
c     Solve for f(i,jx,l_)=temp2 using the coefficients for the equation
c     developed in do loops 60 and 40.
c..................................................................

        do 70 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
          temp2(i,jx)=(temc1(i)*fgg(i,jx-1)+temc2(i))
     1      /(1.-temc1(i)*egg(i,jx-1))
 70     continue

c..................................................................
c     Backsolve to determine solution at end of velocity split.
c..................................................................

        do 90 jj=1,jxm1
          j=jx-jj
          do 80 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            temp2(i,j)=egg(i,j)*temp2(i,j+1)+fgg(i,j)
 80       continue
 90     continue

c..................................................................
c     if desired call routine to check accuracy of differencing
c..................................................................

        if (iactst.ne. "disabled") call exsweepx(k)

c..................................................................
c     compute density diagnostics and if desired set negative values
c     of distribution function to 0
c..................................................................

        call diagxswx(k)
c..................................................................
c     store distribution function after velocity split in fxsp
c..................................................................

        call dcopy(iyjx2,temp1(0,0),1,fxsp(0,0,k,l_),1)

c..................................................................
c     velocity split done - commence with theta split
c
c
c     There will be three (forward) sweeps all of which tie into each
c     other at the pass/trapped boundary. Begin the forward sweeps at
c     y=0, pi/2 and pi. They terminate at  y(itl,l_), y(itl,l_) or y(itu
c     (the lower or upper p/t boundary. All sweeps utilize 0 flux
c     boundary conditions at the outside edges.
c..................................................................

cdir$ ivdep
        do 110 j=2,jx
          egg(1,j)=alpy(1,j)/bety(1,j)
          egg(iyh,j)=gamy(iyh,j)/bety(iyh,j)
          egg(iy_(l_),j)=gamy(iy_(l_),j)/bety(iy_(l_),j)
          fgg(1,j)=dely(1,j)/bety(1,j)
          fgg(iyh,j)=dely(iyh,j)/bety(iyh,j)
          fgg(iy_(l_),j)=dely(iy_(l_),j)/bety(iy_(l_),j)
 110    continue

c..................................................................
c     complete the first sweep over the three regions of the theta mesh
c..................................................................

        do 130 i=2,itl-1
cdir$ ivdep
          do 120 j=2,jx
            egg(i,j)=alpy(i,j)/(bety(i,j)-gamy(i,j)*egg(i-1,j))
            fgg(i,j)=(dely(i,j)+gamy(i,j)*fgg(i-1,j))/(bety(i,j)
     1        -gamy(i,j)*egg(i-1,j))
 120      continue
 130    continue
        do 150 i=iy_(l_)-1,iy_(l_)+2-itl,-1
          do 140 j=2,jx
            egg(i,j)=gamy(i,j)/(bety(i,j)-alpy(i,j)*egg(i+1,j))
            fgg(i,j)=(dely(i,j)+alpy(i,j)*fgg(i+1,j))/(bety(i,j)
     1        -alpy(i,j)*egg(i+1,j))
 140      continue
 150    continue
        do 170 i=iyh-1,itl+1,-1
          do 180 j=2,jx
            egg(i,j)=gamy(i,j)/(bety(i,j)-alpy(i,j)*egg(i+1,j))
            fgg(i,j)=(dely(i,j)+alpy(i,j)*fgg(i+1,j))/(bety(i,j)
     1        -alpy(i,j)*egg(i+1,j))
 180      continue
 170    continue

c..................................................................
c     At the pass/trapped boundary conservation conditions hold. The
c     r.h.s. of the Fokker-Planck equation will be proportional to:
c
c     -H(itl-1/2) + 2.*H(itl+1/2) + H(itu+1/2).
c
c     This translates to an equation of the form:
c
c     -alpha*f(itl+1,l_) +beta*f(itl,l_) -gamma*f(itl+1,l_) -alpa*f(itu
c
c     = delta(itl)
c
c     and these coefficients are near neighbors to alpy, bety, etc
c     used at the other points of the domain. Compute alpha - delta
c     and store them in tam1-tam5
c..................................................................

        do 185 j=2,jx
          tam1(j)=alpy(itl,j)
          tam2(j)=.5*(bety(itl,j)+bety(itu,j))
          tam3(j)=.5*gamy(itl,j)
          tam4(j)=.5*alpy(itu,j)
          tam5(j)=.5*(dely(itl,j)+dely(itu,j))
 185    continue

c..................................................................
c     The equation whose coefficients are defined in do loop 180
c     involves f at the points itl+1, itl-1, itu+1 as well as itl.
c     The recursive equations developed in do loops 120, 140, 160
c     can be employed to recast this equation so that it has only
c     one unknown: f(itl,j,l_). Solve for f(itl,j,l_)=f(itu,j,l_).
c..................................................................

        do 190 j=2,jx
          tam6(j)=tam2(j)-tam1(j)*egg(itl+1,j)-tam3(j)*egg(itl-1,j)-
     1      tam4(j)*egg(iy_(l_)+2-itl,j)
          tam7(j)=tam5(j)+tam1(j)*fgg(itl+1,j)+tam3(j)*fgg(itl-1,j)+
     1      tam4(j)*fgg(iy_(l_)+2-itl,j)
          temp2(itl,j)=tam7(j)/tam6(j)
          temp2(itu,j)=temp2(itl,j)
 190    continue

c..................................................................
c     backsolve over three regions to obtain distribution
c..................................................................

        do 210 i=itl-1,1,-1
cdir$ ivdep
          do 200 j=2,jx
            temp2(i,j)=egg(i,j)*temp2(i+1,j)+fgg(i,j)
 200      continue
 210    continue
        do 230 i=itl+1,iyh
cdir$ ivdep
          do 220 j=2,jx
            temp2(i,j)=egg(i,j)*temp2(i-1,j)+fgg(i,j)
 220      continue
 230    continue
        do 250 i=iy_(l_)+2-itl,iy_(l_)
cdir$ ivdep
          do 240 j=2,jx
            temp2(i,j)=egg(i,j)*temp2(i-1,j)+fgg(i,j)
 240      continue
 250    continue

c..................................................................
c     Add the contribution of krook operator + source at v=0.
c     At x=0 f is clearly isotropic and will be averaged
c     over theta below.
c..................................................................

        do 260 i=1,iy_(l_)
          temc1(i)=.5*so(i,1)+temp1(i,1)*rbgn
          temc2(i)=rbgn-.5*cah(i,1)
          temp2(i,1)=temc1(i)/temc2(i)
 260    continue

c..................................................................
c     update f in upper half of trapping region using symmetry.
c..................................................................

        do 270 j=1,jx
cdir$ ivdep
          do 271 i=itl+1,iyh
            temp2(iy_(l_)+1-i,j)=temp2(i,j) !!YuP[2021-03-11] iy-->iy_(l_)
 271      continue
 270    continue

c..................................................................
c     difference solution - compare with right side.
c..................................................................

        if (iactst.ne."disabled") call exsweept(k)

c..................................................................
c     compute density diagnostics and set negative values of f to 0.
c..................................................................

        call diagxswt(k)

c..................................................................
c     redefine f at v=0 so it is unique
c..................................................................

        s=0.
        t=0.
        do 280 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
          s=s+vptb(i,lr_)*cynt2(i,l_)
          t=t+vptb(i,lr_)*cynt2(i,l_)*temp2(i,1)
 280    continue
        do 512 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
          temp1(i,1)=t/s
 512    continue

c..................................................................
c     store new distribution in f.
c..................................................................

        call dcopy(iyjx2,temp1(0,0),1,f(0,0,k,l_),1)
 600  continue
      return
      end
