c
c
      subroutine wparsou
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     This routine computes the spatial source term, due to the parallel 
c     transport operator applied on f_n, needed for the velocity split 
c     with the ADI (Alternating Direction Implicit) method.
c.......................................................................

      include 'param.h'
      include 'comm.h'
      dimension zdns(lrors) !local, changed to lrors (YuP) 
      dimension zdnspa(lrors),zdnvel(lrors),zdnp1(lrors) !local, changed to lrors (YuP)
      dimension zdnspa2(iymax),zdnvel2(iymax)
      dimension zspa2i(iymax/2),zvel2i(iymax/2)
      dimension zdnuspa(jx),zdnuvel(jx)

      include 'wpadvnc.h'
c.......................................................................
c     with new try if sbdry=periodic .and. laddbnd=1 (see wpbdry)
c.......................................................................

c     restarted runs read spasou from restart file
      if (n.eq.0 .and. (nlrestrt.ne."disabled")) return

c%OS  
      if (n .eq. 0) return
c%OS  

      call bcast(spasou(0,0,1,1),zero,iyjx2*ngen*lrors)
      call bcast(zdns,zero,lrors)
      call bcast(zdnspa,zero,lrors)
      call bcast(zdnvel,zero,lrors)
      call bcast(zdnp1,zero,lrors)
      call bcast(zdnspa2,zero,iymax)
      call bcast(zdnvel2,zero,iymax)
      call bcast(zspa2i,zero,iymax/2)
      call bcast(zvel2i,zero,iymax/2)
      call bcast(zdnuspa,zero,jx)
      call bcast(zdnuvel,zero,jx)

c     standard case: numindx=0 or 1
      ipshft2=0
      ipshft1=-1
      imshft2=0
      imshft1=-1
c     l loop between 1+i.strt and l_upper(i)+i.end, with .=p or m for i< or >iyh
      ipstrt=+1
      ipend=0
      imstrt=+1
      imend=0
      if (numindx .eq. 2) then
        ipshft2=(numixts+1)/2
        ipshft1=(numixts-1)/2
        imshft2=-(numixts-1)/2
        imshft1=-(numixts+1)/2
        ipstrt=-(numixts-1)/2
        ipend=-(numixts+1)/2
        imstrt=(numixts+1)/2
        imend=(numixts-1)/2
      else if (numindx .eq. 3) then
        ipshft2=+1
        ipshft1=0
        imshft2=+1
        imshft1=0
        ipstrt=0
        ipend=-1
        imstrt=0
        imend=-1
      else if (numindx .eq. 4) then
        ipshft2=+1
        ipshft1=-1
        imshft2=+1
        imshft1=-1
        ipstrt=+1
        ipend=-1
        imstrt=+1
        imend=-1
      endif
      if (sbdry .eq. "periodic") then
        ipstrt=0
        ipend=0
        imstrt=0
        imend=0
      endif
      ip1t1m2=ipshft1*(1-ipshft2)
      im1t1m2=imshft1*(1-imshft2)
      if (lmidvel .eq. 0) ip1t1m2=0
      if (lmidvel .eq. 0) im1t1m2=0

c.......................................................................
cl    1. Loop over general species
c.......................................................................

      do 100 k=1,ngen

c.......................................................................
cl    2. Loop over theta in [0,pi/2]
c.......................................................................

        do 200 i=1,iymax/2
          if (l_upper(i) .eq. 1) go to 200

c.......................................................................
cl    3. Loop over all possible l for given i and numerical scheme
c.......................................................................

          do 300 l=1+ipstrt*imstrt,l_upper(i)-ipend*imend

c.......................................................................
cl    3.0 new try
c.......................................................................

            if (sbdry.eq."periodic" .and. laddbnd.eq.1 .and.
     !        l.eq.l_upper(i) .and. l_upper(i).ne.ls) then
              do 401 j=2,jx
c     points A_i and A_ii (notes p. Num(13))
                spasou(i,j,k,l)=
     !            (fnp1(i,j,k,l)-fnhalf(i,j,k,l))/dtreff-velsou(i,j,k,l)
                zdns(l)=zdns(l)+spasou(i,j,k,l)*cynt2(i,l)*cint2(j)
                iief=iy_(l)+1-i
                spasou(iief,j,k,l)=
     !            (fnp1(iief,j,k,l)-fnhalf(iief,j,k,l))/dtreff
     +            -velsou(iief,j,k,l)
                zdns(l)=zdns(l)+spasou(iief,j,k,l)*cynt2(iief,l)
     +            *cint2(j)

c     points B_i and B_ii (notes p. Num(13))
                ll=ls+2-l_upper(i)
                spasou(i,j,k,ll)=
     !            (fnp1(i,j,k,ll)-fnhalf(i,j,k,ll))/dtreff
     +            -velsou(i,j,k,ll)
                zdns(ll)=zdns(ll)+spasou(i,j,k,ll)*cynt2(i,ll)*cint2(j)
                iief=iy_(ll)+1-i
                spasou(iief,j,k,ll)=
     !            (fnp1(iief,j,k,ll)-fnhalf(iief,j,k,ll))/dtreff
     +            -velsou(iief,j,k,ll)
                zdns(ll)=zdns(ll)+spasou(iief,j,k,ll)*cynt2(iief,ll)*
     !            cint2(j)
 401          continue
              go to 300
            endif

c.......................................................................
cl    3.1 l in [1,l_upper(i)]
c.......................................................................

            if (l.lt.1+ipstrt .or. l.gt.l_upper(i)+ipend .or.
     1        ilpm1ef(i,l,ipshft1).eq.-999 .or.
     !        ilpm1ef(i,l,ipshft2).eq.-999) go to 312

c.......................................................................
cl    3.1.1 cos(theta)>0
c.......................................................................

            l1=lpm1eff(l,ipshft1)
            l2=lpm1eff(l,ipshft2)
            l12=lpm1eff(l,ipshft1+ipshft2)
            ztrders=vnorm*coss(i,l)/(ipshft2*dszp5(l)-ipshft1*dszm5(l))
            do 411 j=2,jx
              spasou(i,j,k,l+ip1t1m2)=-x(j)*ztrders*
     !          (fnp1(i,j,k,l2)-fnp1(i,j,k,l1))
              zdns(l)=zdns(l)+spasou(i,j,k,l+ip1t1m2)*cynt2(i,l)
     +          *cint2(j)
 411        continue

c.......................................................................
cl    3.1.2 cos(theta) < 0
c.......................................................................

 312        continue

            if (l.lt.1+imstrt .or. l.gt.l_upper(i)+imend .or.
     1        ilpm1ef(i,l,imshft1).eq.-999 .or.
     !        ilpm1ef(i,l,imshft2).eq.-999) go to 320

            iieff=iy_(l)+1-i
            l1=lpm1eff(l,imshft1)
            il1=ilpm1ef(iieff,l,imshft1)
            l2=lpm1eff(l,imshft2)
            il2=ilpm1ef(iieff,l,imshft2)
            l12=lpm1eff(l,imshft1+imshft2)
            il12=ilpm1ef(iieff,l,imshft1+imshft2)
            ztrders=vnorm*coss(iieff,l)/
     !        (imshft2*dszp5(l)-imshft1*dszm5(l))
            do 412 j=2,jx
              spasou(iieff,j,k,l+im1t1m2)=-x(j)*ztrders*
     !          (fnp1(il2,j,k,l2)-fnp1(il1,j,k,l1))
              zdns(l)=zdns(l)+spasou(iieff,j,k,l+im1t1m2)*
     !          cynt2(iieff,l)*cint2(j)
 412        continue

 320        if (sbdry.ne."periodic" .or. l_upper(i).eq.ls .or. l.eq.1)
     1        go to 300

c.......................................................................
cl    3.2 point ll=ls+2-l, i.e. bottom half cross-section. Not yet treated
c     in 3.1 if particle is trapped.
c     Thus: l in [ls+2-l_upper(i),ls]
c.......................................................................

            ll=ls+2-l

            if (ilpm1ef(i,ll,ipshft1).eq.-999 .or. 
     !        ilpm1ef(i,ll,ipshft2).eq.-999) go to 322

c.......................................................................
cl    3.2.1 cos(theta)>0, ll>ls/2+1
c.......................................................................

            l1=lpm1eff(ll,ipshft1)
            l2=lpm1eff(ll,ipshft2)
            l12=lpm1eff(ll,ipshft1+ipshft2)
            ztrders=vnorm*coss(i,ll)
     +        /(ipshft2*dszp5(ll)-ipshft1*dszm5(ll))
            do 421 j=2,jx
              spasou(i,j,k,ll+ip1t1m2)=-x(j)*ztrders*
     !          (fnp1(i,j,k,l2)-fnp1(i,j,k,l1))
              zdns(ll)=zdns(ll)+spasou(i,j,k,ll+ip1t1m2)*cynt2(i,ll)*
     !          cint2(j)
 421        continue

c.......................................................................
cl    3.2.2 cos(theta) < 0, ll>ls/2+1
c.......................................................................

 322        continue

            if (ilpm1ef(i,ll,imshft1).eq.-999 .or.
     !        ilpm1ef(i,ll,imshft2).eq.-999) go to 300

            iieff=iy_(ll)+1-i
            l1=lpm1eff(ll,imshft1)
            il1=ilpm1ef(iieff,ll,imshft1)
            l2=lpm1eff(ll,imshft2)
            il2=ilpm1ef(iieff,ll,imshft2)
            l12=lpm1eff(ll,imshft1+imshft2)
            il12=ilpm1ef(iieff,ll,imshft1+imshft2)
            ztrders=vnorm*coss(iieff,ll)/
     !        (imshft2*dszp5(ll)-imshft1*dszm5(ll))
            do 422 j=2,jx
              spasou(iieff,j,k,ll+im1t1m2)=-x(j)*ztrders*
     !          (fnp1(il2,j,k,l2)-fnp1(il1,j,k,l1))
              zdns(ll)=zdns(ll)+spasou(iieff,j,k,ll+im1t1m2)*
     !          cynt2(iieff,ll)*cint2(j)
 422        continue

 300      continue
 200    continue
 100  continue

c%OSc interpolate on node points
c%OS  if (mod(nummods,10) .le. 4) then
c%OS  do 210 i=1,iy_(ls)
c%OS  spasou(i,j,k,ls)=spasou(ilpm1ef(i,ls,-1),j,k,ls-1)+
c%OS  /             dszm5(ls)/(dszm5(ls)+dszm5(ls-1))*
c%OS  1           (spasou(ilpm1ef(i,ls,-1),j,k,ls-1)-
c%OS  1               spasou(ilpm1ef(ilpm1ef(i,ls,-1),ls-1,-1),j,k,ls-2))
c%OS  210        continue
c%OS  do 211 l=ls-1,2,-1
c%OS  do 211 i=1,iy_(l)
c%OS  spasou(i,j,k,l)=(1.-dls(i,j,k,l))*spasou(i,j,k,l)+
c%OS  +                      dls(i,j,k,l)*spasou(ilpm1ef(i,l,-1),j,k,l-1)
c%OS  211        continue
c%OS  do 212 i=1,iy_(1)
c%OS  spasou(i,j,k,1)=2.*spasou(i,j,k,1)-
c%OS  1                                     spasou(ilpm1ef(i,1,+1),j,k,2)
c%OS  212        continue
c%OS  endif
c%OS  200    continue
c%OS  100  continue

c%OS  
c     check spasou and velsou total densities
      do 500 k=1,ngen
        do 501 j=1,jx
          do 502 l=1,lrors
            do 503 i=1,iy_(l)
              zdnp1(l)=zdnp1(l)+fnp1(i,j,k,l)*cynt2(i,l)*cint2(j)
              zdnspa(l)=zdnspa(l)+spasou(i,j,k,l)*cynt2(i,l)*cint2(j)
              zdnvel(l)=zdnvel(l)+velsou(i,j,k,l)*cynt2(i,l)*cint2(j)
 503        continue
            do 504 i=1,iyh_(l)
              ii=iymax+1-i !YuP[2021-03-11] why not iymax-->iy_(l) ? Not sure
              !ii=iy_(l)+1-i !YuP/test
              ief=iy_(l)+1-i
              zdnspa2(i)=zdnspa2(i)+spasou(i,j,k,l)*cynt2(i,l)*cint2(j)
     !          *dsz(l)/dsz(1)/lrors
              zdnspa2(ii)=zdnspa2(ii)+spasou(ief,j,k,l)*cynt2(ief,l)*
     !          cint2(j)*dsz(l)/dsz(1)/lrors
              zdnvel2(i)=zdnvel2(i)+velsou(i,j,k,l)*cynt2(i,l)*cint2(j)
     !          *dsz(l)/dsz(1)/lrors
              zdnvel2(ii)=zdnvel2(ii)+velsou(ief,j,k,l)*cynt2(ief,l)*
     !          cint2(j)*dsz(l)/dsz(1)/lrors
              zdnuspa(j)=zdnuspa(j)+(spasou(i,j,k,l)*cynt2(i,l)+
     !          spasou(ief,j,k,l)*cynt2(ief,l))*
     !          cint2(j)*dsz(l)/dsz(1)/lrors
              zdnuvel(j)=zdnuvel(j)+(velsou(i,j,k,l)*cynt2(i,l)+
     !          velsou(ief,j,k,l)*cynt2(ief,l))*
     !          cint2(j)*dsz(l)/dsz(1)/lrors
 504        continue
 502      continue
 501    continue
 500  continue
c
      zp1to=0.0
      zspato=0.0
      zvelto=0.0
      do 510 l=1,lrors
        zp1to=zp1to+dsz(l)/dsz(1)*zdnp1(l)/lrors
        zspato=zspato+dsz(l)/dsz(1)*zdnspa(l)/lrors
        zvelto=zvelto+dsz(l)/dsz(1)*zdnvel(l)/lrors
 510  continue

      zspa2to=0.0
      zvel2to=0.0
      do 511 i=1,iymax
        zspa2to=zspa2to+zdnspa2(i)
        zvel2to=zvel2to+zdnvel2(i)
 511  continue
      do 512 i=1,iymax/2
        zspa2i(i)=zdnspa2(i)+zdnspa2(iymax+1-i)
        zvel2i(i)=zdnvel2(i)+zdnvel2(iymax+1-i)
 512  continue
c%OS  

      return
      end
