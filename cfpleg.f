c
c
      subroutine cfpleg(m,l,jzval)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c.......................................................................
c     This routine evaluates the Legendre coefficients given that the
c     the midplane or the local distribution is stored in temp3.
c
c     Note:dcofleg is weight for interval [i-1/2,i+1/2], 
c     therefore temp3 should have the value of f on the nodes and 
c     not at i+1/2 as previously.
c
c     This will return in tam1(j) the coefficient evalutated at z(l,lr_).
c     The first non-zero coefficient will be indexed by jzval - the 
c     others presumably not being required.
c     The method for integrating over theta to obtain the Leg. coeff. 
c     depends on the value of ngauss (and subsequently of analegco)
c.......................................................................

      include 'param.h'
      include 'comm.h'

      call bcast(tam1,zero,jx) !-> Reset all Legendre coefficients to zero
      
c     Modification introduced by YuP, for relativ='fully':  There are
c     problems with subtraction of two large terms at low v for
c     nonrelativistic Te.  The following jzmin is introduced for a case
c     with jx=300, Te=5 keV.  A general solution would be to combine
c     the relativ="enabled", quasi-relativistic expressions at lower v
c     with relativ="fully" treatment at higher v  (YuP: Oct'09).
c
      jzmin = jzval  !-YuP-> first non-zero coefficient 
      !-YuP-> Keep high-m Legendre coefficients for small j (v~0) equal zero:

      if(relativ.eq."fully". and. m.ge.1) jzmin=MAX(3,jzval) ! For temp=5keV: 
                                                             ! 33 ok (10% of jx=330); 
                                                             ! 25 still bad

      !-YuP-> For high-m, first non-zero coeff. is j=33 or jzval, if jzval>33

c.......................................................................
c     dcofleg(i) is the weight between i-1/2 and i+1/2 such that the 
c     Legendre coefficients tam1(j) = sum(i) [f(i)*dcofleg(i)], i=1,imax 
c     and iy+1-imax,iy
c.......................................................................

      iend=itl
      if (cqlpmod.eq."enabled") iend=iyh
      
      if (cqlpmod.eq."enabled") then !YuP[2021-04-02] added
        iiy=iy_(l) !CQLP: iy_ at given poloidal index;
        !However, in CQLP run, cfpleg is called from 
        !loop  do 600 l=iorbstr,iorbend == l_,l_ 
        !so that l=l_ anyway.
      else !CQL3D
        iiy=iy_(l_) !CQL3D: at given radial index
      endif

c     passing particles for CQL3D (or all particles for CQLP)
      do 100 i=1,iend
        ia=iiy-i+1 !!YuP[2021-03-12] iy-->iy_(l_)
        do 110 j=jzmin,jx
          tam1(j)=tam1(j)+temp3(i,j)*dcofleg(i,l,m,indxlr_)+
     +      temp3(ia,j)*dcofleg(ia,l,m,indxlr_)
 110    continue
 100  continue

      if (cqlpmod.eq."enabled") go to 200 !skip another loop below.
      if (l.eq.lz .or. imax(l,lr_).eq.itl) go to 200

c     particles in trapped region but passing poloidal position l
c     Note: dcofleg(imax) includes contribution up to turning point

      do 120 i=iend+1,imax(l,lr_)
        !CQL3D: l is index along pol.angle (field line), and l_=radial
        !CQLP:  l is along field line, and l_ is also along field line
        !But this part is skipped by CQLP anyway, because ALL particles
        !are already included in the i=1,iend loop above.
        ia=iiy-i+1 !!YuP[2021-03-12] iy-->iy_(l_)
        do 121 j=jzmin,jx
          tam1(j)=tam1(j)+temp3(i ,j)*dcofleg(i,l,m,indxlr_)+
     +      temp3(ia,j)*dcofleg(ia,l,m,indxlr_)
 121    continue
 120  continue

 200  continue
      sa=dfloat(2*m+1)*0.5  !Per Eq. 3.1.63 in Killeen book for V_m
      !call dscal(jx,sa,tam1,1) !YuP[2022-12] issues in some debug mode
      tam1=tam1*sa !YuP[2022-12-16] same as above dscal

      return
      end
