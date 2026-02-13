c
c
      subroutine wpcheck
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     This routine checks the accuracy of the parallel time advancement
c     split by evaluating the rhs (in fxsp) and the lhs (in f) of the 
c     difference equation and then comparing. Aggregate sums (sumleft and
c     sumright) are also computed. Errors should be near round-off.
c..............................................................

      include 'param.h'
      include 'comm.h'
C%OS  
      dimension zermx(51,lsa),imx(51,lsa),jmx(51,lsa),zsumj3(iymax,lsa)
      dimension zermxii(51,lsa),imxii(51,lsa),jmxii(51,lsa)
      dimension zdiffj(iymax,lsa),zdiffi(jx,lsa),zsumj4(iymax,lsa)
C%OS  
      dimension zdns(lrors),zdns1(lrors)  !local, changed to lrors (YuP)
      dimension zdnsr1(lrors),zdnsr2(lrors) !local, changed to lrors (YuP)
      dimension zdnsb(lrors),zdnsr1b(lrors),zdnsr2b(lrors)
      dimension zdn2th(iymax),zdn2thb(iymax)
      dimension zdnsth(iymax),zdnsthb(iymax)
      !YuP[2021-03-11] Changed iy-->iymax in declarations
      !(just in case if iy is changed by iy=iy_(l_) somewhere)

      include 'wpadvnc.h'
c.......................................................................

      call bcast(fxsp(0,0,1,1),zero,iyjx2*ngen*lrors)
      call bcast(f(0,0,1,1),zero,iyjx2*ngen*lrors)
      call bcast(zdns,zero,lrors)
      call bcast(zdnsb,zero,lrors)
      call bcast(zdns1,zero,lrors)
      call bcast(zdnsr1,zero,lrors)
      call bcast(zdnsr2,zero,lrors)
      call bcast(zdnsr1b,zero,lrors)
      call bcast(zdnsr2b,zero,lrors)
      call bcast(zdn2th,zero,iymax)
      call bcast(zdn2thb,zero,iymax)
      call bcast(zdnsth,zero,iymax)
      call bcast(zdnsthb,zero,iymax)

c.......................................................................
cl    0. Parameters determining numerical scheme
c.......................................................................

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
      ip2t1p1=ipshft2*(1+ipshft1)
      im2t1p1=imshft2*(1+imshft1)

      zvelmid=1.d0
      if (mod(nummods,10).le.4 .and. lmidvel.ne.0) zvelmid=0.5d0
      zfcen=0.5d0
      if (numindx.eq.4 .or. lmidvel.eq.0) zfcen=1.d0

c.......................................................................
cl    1. Loop over general species
c.......................................................................

      do 100 k=1,ngen
        sumleft=0.d0
        sumlftn=0.d0
        sumrigt=0.d0
c     j=1:
        j=1
        do 101 l=1,ls
          do 102 i=1,iyh_(l)
            ii=iy_(l)+1-i
            zdns1(l)=zdns1(l)+cint2(j)*(fnp1(i,j,k,l)*cynt2(i,l)+
     !        fnp1(ii,j,k,l)*cynt2(ii,l))
 102      continue
 101    continue

c.......................................................................
cl    2. Loop over theta in [0,pi/2]
c.......................................................................

        do 200 i=1,iymax/2
          ii=iymax+1-i ! Here l=1, so that iy_(1)=iymax
          if (l_upper(i) .eq. 1) then
            ii=iy_(1)+1-i
            do 201 j=2,jx
              zdns1(1)=zdns1(1)+cint2(j)*(fnp1(i,j,k,1)*cynt2(i,1)+
     !          fnp1(ii,j,k,1)*cynt2(ii,1))
 201        continue
            go to 200
          endif

c.......................................................................
cl    3. Loop over all possible l for given i and numerical scheme
c.......................................................................

          do 300 l=1+ipstrt*imstrt,l_upper(i)-ipend*imend

c.......................................................................
cl    3.0 new try
c.......................................................................

            if (sbdry.eq."periodic" .and. laddbnd.eq.1 .and.
     !        l.eq.l_upper(i) .and. l_upper(i).ne.ls) then
              iief=iy_(l)+1-i
              ii=iymax+1-i !YuP: why iymax?
              ll=ls+2-l_upper(i)
              ztra2=cynt2(i,l)*(ipshft2*dszp5(l)-ipshft1*dszm5(l))/
     !          dtreff
              ztra2l=cynt2(i,ll)*(ipshft2*dszp5(ll)-ipshft1*dszm5(ll))/
     !          dtreff
              !iief=iy_(l)+1-i
              !ii=iymax+1-i !YuP: why iymax?
              !ll=ls+2-l_upper(i) !YuP: repeated from few lines above
              !write(*,*)'l,i,ii,iymax=',l,i,ii,iymax
              do 401 j=2,jx
c     points A_i and A_ii (notes p. Num(13))
                f(i,j,k,l)=(fnp1(i,j,k,l)-fnhalf(i,j,k,l))*ztra2*
     !            cint2(j)
                fxsp(i,j,k,l)=f(i,j,k,l)
                f(ii,j,k,l)=(fnp1(iief,j,k,l)-fnhalf(iief,j,k,l))*ztra2
     !            *cint2(j)
                fxsp(ii,j,k,l)=f(ii,j,k,l)

c     points B_i and B_ii (notes p. Num(13))
                f(i,j,k,ll)=(fnp1(i,j,k,ll)-fnhalf(i,j,k,ll))*ztra2l*
     !            cint2(j)
                fxsp(i,j,k,ll)=f(i,j,k,ll)
                f(ii,j,k,ll)=(fnp1(iief,j,k,ll)-fnhalf(iief,j,k,ll))*
     !            ztra2l*cint2(j)
                fxsp(ii,j,k,ll)=f(ii,j,k,ll)
                sumleft=sumleft+f(i,j,k,l)+f(ii,j,k,l)+f(i,j,k,ll)+
     !            f(ii,j,k,ll)
                sumrigt=sumrigt+fxsp(i,j,k,l)+fxsp(ii,j,k,l)+
     !            fxsp(i,j,k,ll)+fxsp(ii,j,k,ll)
                zdns(l)=zdns(l)+(f(i,j,k,l)+f(ii,j,k,l))/
     !            (ipshft2*dszp5(l)-ipshft1*dszm5(l))
                zdns(ll)=zdns(ll)+(f(i,j,k,ll)+f(ii,j,k,ll))/
     !            (ipshft2*dszp5(ll)-ipshft1*dszm5(ll))
                zdns1(l)=zdns1(l)+(fnp1(i,j,k,l)+fnp1(iief,j,k,l))*
     !            cynt2(i,l)*cint2(j)
                zdns1(ll)=zdns1(ll)+(fnp1(i,j,k,ll)+fnp1(iief,j,k,ll))*
     !            cynt2(i,ll)*cint2(j)

 401          continue
              go to 300
            endif

c.......................................................................
cl    3.1 l in [1,l_upper(i)]
c.......................................................................

            itest1=0
            if (l.lt.1+ipstrt .or. l.gt.l_upper(i)+ipend .or.
     1        ilpm1ef(i,l,ipshft1).eq.-999 .or.
     !        ilpm1ef(i,l,ipshft2).eq.-999) go to 312
            itest1=1

c.......................................................................
cl    3.1.1 cos(theta)>0
c.......................................................................

            ztra1=cynt2(i,l)*(ipshft2*dszp5(l)-ipshft1*dszm5(l))
            ztra2=ztra1/dtreff
            l1=lpm1eff(l,ipshft1)
            l2=lpm1eff(l,ipshft2)
            l12=lpm1eff(l,ipshft1+ipshft2)
            ztrders=vnorm*coss(i,l)/(ipshft2*dszp5(l)-ipshft1*dszm5(l))
            do 411 j=2,jx
              f(i,j,k,l)=
     &         (zfcen*fnp1(i,j,k,l)+(1.d0-zfcen)*fnp1(i,j,k,l12)
     -          -fnhalf(i,j,k,l+ip1t1m2))*ztra2*cint2(j)
              fxsp(i,j,k,l)=ztra1*cint2(j)*
     !          (zvelmid*velsou(i,j,k,l+ip1t1m2)+
     +          (1.d0-zvelmid)*velsou(i,j,k,l+ip2t1p1)
     !          -x(j)*ztrders*(fnp1(i,j,k,l2)-fnp1(i,j,k,l1)))
              sumleft=sumleft+f(i,j,k,l)
              sumrigt=sumrigt+fxsp(i,j,k,l)
              zdns(l)=zdns(l)+f(i,j,k,l)/
     /          (ipshft2*dszp5(l)-ipshft1*dszm5(l))
              zdnsth(i)=zdnsth(i)+f(i,j,k,l)/
     /          (ipshft2*dszp5(l)-ipshft1*dszm5(l))
              sumlftn=sumlftn+(fnp1(i,j,k,l)-f_(i,j,k,l))*ztra2*cint2(j)
              zdns1(l)=zdns1(l)+fnp1(i,j,k,l)*cynt2(i,l)*cint2(j)
              zdnsr1(l)=zdnsr1(l)+(zvelmid*velsou(i,j,k,l+ip1t1m2)+
     +          (1.d0-zvelmid)*velsou(i,j,k,l+ip2t1p1))*cynt2(i,l)
     +          *cint2(j)
              zdnsr2(l)=zdnsr2(l)-x(j)*ztrders*
     !          (fnp1(i,j,k,l2)-fnp1(i,j,k,l1))*cynt2(i,l)*cint2(j)
              zdn2th(i)=zdn2th(i)-x(j)*ztrders*
     !          (fnp1(i,j,k,l2)-fnp1(i,j,k,l1))*cynt2(i,l)*cint2(j)
 411        continue

c.......................................................................
cl    3.1.2 cos(theta) < 0
c.......................................................................

 312        continue

            if (l.lt.1+imstrt .or. l.gt.l_upper(i)+imend .or.
     1        ilpm1ef(i,l,imshft1).eq.-999 .or.
     !        ilpm1ef(i,l,imshft2).eq.-999) go to 320
            itest1=itest1+10

            iieff=iy_(l)+1-i
            ztra1=cynt2(iieff,l)*(imshft2*dszp5(l)-imshft1*dszm5(l))
            ztra2=ztra1/dtreff
            l1=lpm1eff(l,imshft1)
            il1=ilpm1ef(iieff,l,imshft1)
            l2=lpm1eff(l,imshft2)
            il2=ilpm1ef(iieff,l,imshft2)
            l12=lpm1eff(l,imshft1+imshft2)
            il12=ilpm1ef(iieff,l,imshft1+imshft2)
            la=lpm1eff(l,im1t1m2)
            ila=ilpm1ef(iieff,l,im1t1m2)
            lb=lpm1eff(l,im2t1p1)
            ilb=ilpm1ef(iieff,l,im2t1p1)
            ztrders=vnorm*coss(iieff,l)/
     !        (imshft2*dszp5(l)-imshft1*dszm5(l))
            do 412 j=2,jx
              f(ii,j,k,l)=(zfcen*fnp1(iieff,j,k,l)+(1.d0-zfcen)*
     !          fnp1(il12,j,k,l12)-fnhalf(ila,j,k,la))*ztra2*cint2(j)
              fxsp(ii,j,k,l)=ztra1*cint2(j)*
     +          (zvelmid*velsou(ila,j,k,la)+
     +          (1.d0-zvelmid)*velsou(ilb,j,k,lb)
     !          -x(j)*ztrders*(fnp1(il2,j,k,l2)-fnp1(il1,j,k,l1)))
              sumleft=sumleft+f(ii,j,k,l)
              sumrigt=sumrigt+fxsp(ii,j,k,l)
              zdns(l)=zdns(l)+f(ii,j,k,l)/
     /          (ipshft2*dszp5(l)-ipshft1*dszm5(l))
              zdnsth(ii)=zdnsth(ii)+f(ii,j,k,l)/
     /          (ipshft2*dszp5(l)-ipshft1*dszm5(l))
              sumlftn=sumlftn+(fnp1(iieff,j,k,l)-f_(iieff,j,k,l))*ztra2*
     !          cint2(j)
              zdns1(l)=zdns1(l)+fnp1(iieff,j,k,l)*cynt2(iieff,l)
     +          *cint2(j)
              zdnsr1(l)=zdnsr1(l)+(zvelmid*velsou(ila,j,k,la)+
     +        (1.d0-zvelmid)*velsou(ilb,j,k,lb))*cynt2(iieff,l)*cint2(j)
              zdnsr2(l)=zdnsr2(l)-x(j)*ztrders*
     !          (fnp1(il2,j,k,l2)-fnp1(il1,j,k,l1))*cynt2(iieff,l)
     +          *cint2(j)
              zdn2th(ii)=zdn2th(ii)-x(j)*ztrders*
     !          (fnp1(il2,j,k,l2)-fnp1(il1,j,k,l1))*cynt2(iieff,l)
     +          *cint2(j)
 412        continue

 320        if (sbdry.ne."periodic" .or. l_upper(i).eq.ls .or. l.eq.1)
     1        go to 300

c.......................................................................
cl    3.2 point ll=ls+2-l, i.e. bottom half cross-section. Not yet treated
c     in 3.1 if particle is trapped.
c     Thus: l in [ls+2-l_upper(i),ls]
c.......................................................................

            ll=ls+2-l

            itest2=0
            if (ilpm1ef(i,ll,ipshft1).eq.-999 .or. 
     !        ilpm1ef(i,ll,ipshft2).eq.-999) go to 322
            itest2=1

c.......................................................................
cl    3.2.1 cos(theta)>0, ll>ls/2+1
c.......................................................................

            ztra1=cynt2(i,ll)*(ipshft2*dszp5(ll)-ipshft1*dszm5(ll))
            ztra2=ztra1/dtreff
            l1=lpm1eff(ll,ipshft1)
            l2=lpm1eff(ll,ipshft2)
            l12=lpm1eff(ll,ipshft1+ipshft2)
            ztrders=vnorm*coss(i,ll)
     +        /(ipshft2*dszp5(ll)-ipshft1*dszm5(ll))
            do 421 j=2,jx
              f(i,j,k,ll)=(zfcen*fnp1(i,j,k,ll)+
     +          (1.d0-zfcen)*fnp1(i,j,k,l12)
     -          -fnhalf(i,j,k,ll+ip1t1m2))*ztra2*cint2(j)
              fxsp(i,j,k,ll)=ztra1*cint2(j)*
     !          (zvelmid*velsou(i,j,k,ll+ip1t1m2)+
     +          (1.d0-zvelmid)*velsou(i,j,k,ll+ip2t1p1)
     !          -x(j)*ztrders*(fnp1(i,j,k,l2)-fnp1(i,j,k,l1)))
              sumleft=sumleft+f(i,j,k,ll)
              sumrigt=sumrigt+fxsp(i,j,k,ll)
              zdns(ll)=zdns(ll)+f(i,j,k,ll)/
     /          (ipshft2*dszp5(ll)-ipshft1*dszm5(ll))
              zdnsth(i)=zdnsth(i)+f(i,j,k,ll)/
     /          (ipshft2*dszp5(ll)-ipshft1*dszm5(ll))
              sumlftn=sumlftn+(fnp1(i,j,k,ll)-f_(i,j,k,ll))*ztra2
     +          *cint2(j)
              zdns1(ll)=zdns1(ll)+fnp1(i,j,k,ll)*cynt2(i,ll)*cint2(j)
              zdnsr1(ll)=zdnsr1(ll)+(zvelmid*velsou(i,j,k,ll+ip1t1m2)+
     +          (1.d0-zvelmid)*velsou(i,j,k,ll+ip2t1p1))*cynt2(i,ll)
     +          *cint2(j)
              zdnsr2(ll)=zdnsr2(ll)-x(j)*ztrders*
     !          (fnp1(i,j,k,l2)-fnp1(i,j,k,l1))*cynt2(i,ll)*cint2(j)
              zdn2th(i)=zdn2th(i)-x(j)*ztrders*
     !          (fnp1(i,j,k,l2)-fnp1(i,j,k,l1))*cynt2(i,ll)*cint2(j)
 421        continue

c.......................................................................
cl    3.2.2 cos(theta) < 0, ll>ls/2+1
c.......................................................................

 322        continue

            if (ilpm1ef(i,ll,imshft1).eq.-999 .or.
     !        ilpm1ef(i,ll,imshft2).eq.-999) go to 330
            itest2=itest2+10

            iieff=iy_(ll)+1-i
            ztra1=cynt2(iieff,ll)*(imshft2*dszp5(ll)-imshft1*dszm5(ll))
            ztra2=ztra1/dtreff
            l1=lpm1eff(ll,imshft1)
            il1=ilpm1ef(iieff,ll,imshft1)
            l2=lpm1eff(ll,imshft2)
            il2=ilpm1ef(iieff,ll,imshft2)
            l12=lpm1eff(ll,imshft1+imshft2)
            il12=ilpm1ef(iieff,ll,imshft1+imshft2)
            la=lpm1eff(ll,im1t1m2)
            ila=ilpm1ef(iieff,ll,im1t1m2)
            lb=lpm1eff(ll,im2t1p1)
            ilb=ilpm1ef(iieff,ll,im2t1p1)
            ztrders=vnorm*coss(iieff,ll)/
     !        (imshft2*dszp5(ll)-imshft1*dszm5(ll))
            do 422 j=2,jx
              f(ii,j,k,ll)=(zfcen*fnp1(iieff,j,k,ll)+(1.d0-zfcen)*
     !          fnp1(il12,j,k,l12)-fnhalf(ila,j,k,la))*ztra2*cint2(j)
              fxsp(ii,j,k,ll)=ztra1*cint2(j)*
     +          (zvelmid*velsou(ila,j,k,la)+
     +          (1.d0-zvelmid)*velsou(ilb,j,k,lb)
     !          -x(j)*ztrders*(fnp1(il2,j,k,l2)-fnp1(il1,j,k,l1)))
              sumleft=sumleft+f(ii,j,k,ll)
              sumrigt=sumrigt+fxsp(ii,j,k,ll)
              zdns(ll)=zdns(ll)+f(ii,j,k,ll)/
     /          (ipshft2*dszp5(ll)-ipshft1*dszm5(ll))
              zdnsth(ii)=zdnsth(ii)+f(ii,j,k,ll)/
     /          (ipshft2*dszp5(ll)-ipshft1*dszm5(ll))
              sumlftn=sumlftn+(fnp1(iieff,j,k,ll)-f_(iieff,j,k,ll))*
     !          ztra2*cint2(j)
              zdns1(ll)=zdns1(ll)+fnp1(iieff,j,k,ll)*cynt2(iieff,ll)*
     !          cint2(j)
              zdnsr1(ll)=zdnsr1(ll)+(zvelmid*velsou(ila,j,k,la)+
     +          (1.d0-zvelmid)*velsou(ilb,j,k,lb))*cynt2(iieff,ll)
     +          *cint2(j)
              zdnsr2(ll)=zdnsr2(ll)-x(j)*ztrders*
     !          (fnp1(il2,j,k,l2)-fnp1(il1,j,k,l1))*cynt2(iieff,ll)
     +          *cint2(j)
              zdn2th(ii)=zdn2th(ii)-x(j)*ztrders*
     !          (fnp1(il2,j,k,l2)-fnp1(il1,j,k,l1))*cynt2(iieff,ll)
     +          *cint2(j)
 422        continue

 330        continue
            if (itest1 .ne. 11) then
c     missing point theta<pi/2, l
              if (itest1.eq.0 .or. itest1.eq.10) then
                do 431 j=2,jx
                  zdnsb(l)=zdnsb(l)+(zfcen*fnp1(i,j,k,l)+
     !              (1.d0-zfcen)*fnp1(i,j,k,l+ipshft1+ipshft2)
     -              -fnhalf(i,j,k,l+ip1t1m2))*cynt2(i,l)*cint2(j)
                  zdnsthb(i)=zdnsthb(i)+(zfcen*fnp1(i,j,k,l)+
     !              (1.d0-zfcen)*fnp1(i,j,k,l+ipshft1+ipshft2)
     -              -fnhalf(i,j,k,l+ip1t1m2))*cynt2(i,l)*cint2(j)
                  zdns1(l)=zdns1(l)+fnp1(i,j,k,l)*cynt2(i,l)*cint2(j)
                  zdnsr1b(l)=zdnsr1b(l)+(zvelmid*velsou(i,j,k,l+ip1t1m2)
     +              +(1.d0-zvelmid)*velsou(i,j,k,l+ip2t1p1))*cynt2(i,l)
     +              *cint2(j)
                  zdnsr2b(l)=zdnsr2b(l)-x(j)*vnorm*coss(i,l)/dszm5(l)*
     !              (fnp1(i,j,k,l)-fnp1(i,j,k,l-1))*cynt2(i,l)*cint2(j)
                  zdn2thb(i)=zdn2thb(i)-x(j)*vnorm*coss(i,l)/dszm5(l)*
     !              (fnp1(i,j,k,l)-fnp1(i,j,k,l-1))*cynt2(i,l)*cint2(j)
 431            continue
              endif
c     missing point theta>pi/2, l
              if (itest1.eq.0 .or. itest1.eq.1) then
                iieff=iy_(l)+1-i
                iieffm=ilpm1ef(iieff,l,-1)
                iieffc=ilpm1ef(iieff,l,ipshft1+ipshft2)
                iieffd=ilpm1ef(iieff,l,ip1t1m2)
                iieffe=ilpm1ef(iieff,l,ip2t1p1)
                do 432 j=2,jx
                  zdnsb(l)=zdnsb(l)+(zfcen*fnp1(iieff,j,k,l)+
     !              (1.d0-zfcen)*fnp1(iieffc,j,k,l+ipshft1+ipshft2)
     -              -fnhalf(iieffd,j,k,l+ip1t1m2))*cynt2(iieff,l)
     +              *cint2(j)
                  zdnsthb(ii)=zdnsthb(ii)+(zfcen*fnp1(iieff,j,k,l)+
     !              (1.d0-zfcen)*fnp1(iieffc,j,k,l+ipshft1+ipshft2)
     -              -fnhalf(iieffd,j,k,l+ip1t1m2))*cynt2(iieff,l)
     +              *cint2(j)
                  zdns1(l)=zdns1(l)+fnp1(iieff,j,k,l)*cynt2(iieff,l)
     !              *cint2(j)
                  zdnsr1b(l)=zdnsr1b(l)+
     +              (zvelmid*velsou(iieffd,j,k,l+ip1t1m2)+
     +              (1.d0-zvelmid)*velsou(iieffe,j,k,l+ip2t1p1))
     +              *cynt2(iieff,l)*cint2(j)
                  zdnsr2b(l)=zdnsr2b(l)-x(j)*vnorm*coss(iieff,l)
     +              /dszm5(l)*
     !              (fnp1(iieff,j,k,l)-fnp1(iieffm,j,k,l-1))
     +              *cynt2(iieff,l)*cint2(j)
                  zdn2thb(ii)=zdn2thb(ii)-x(j)*vnorm*coss(iieff,l)
     +              /dszm5(l)*
     !              (fnp1(iieff,j,k,l)-fnp1(iieffm,j,k,l-1))
     +              *cynt2(iieff,l)*cint2(j)
 432            continue
              endif
            endif
c     ll=ls+2-l
            if (itest2 .ne. 11) then
c     missing point theta<pi/2, ll
              if (itest2.eq.0 .or. itest2.eq.10) then
                do 433 j=2,jx
                  zdns1(ll)=zdns1(ll)+fnp1(i,j,k,ll)*cynt2(i,ll)
     +              *cint2(j)
                  zdnsb(ll)=zdnsb(ll)+(zfcen*fnp1(i,j,k,ll)+
     !              (1.d0-zfcen)*fnp1(i,j,k,ll+ipshft1+ipshft2)
     -              -fnhalf(i,j,k,ll+ip1t1m2))*cynt2(i,ll)*cint2(j)
                  zdnsthb(i)=zdnsthb(i)+(zfcen*fnp1(i,j,k,ll)+
     !              (1.d0-zfcen)*fnp1(i,j,k,ll+ipshft1+ipshft2)
     -              -fnhalf(i,j,k,ll+ip1t1m2))*cynt2(i,ll)*cint2(j)
                  zdnsr1b(ll)=zdnsr1b(ll)+
     +              (zvelmid*velsou(i,j,k,ll+ip1t1m2)+
     +              (1.d0-zvelmid)*velsou(i,j,k,ll+ip2t1p1))*cynt2(i,ll)
     +              *cint2(j)
                  zdnsr2b(ll)=zdnsr2b(ll)-x(j)*vnorm*coss(i,ll)
     +              /dszm5(ll)*
     !              (fnp1(i,j,k,ll)-fnp1(i,j,k,ll-1))*cynt2(i,ll
     +              )*cint2(j)
                  zdn2thb(i)=zdn2thb(i)-x(j)*vnorm*coss(i,ll)/dszm5(ll)*
     !              (fnp1(i,j,k,ll)-fnp1(i,j,k,ll-1))*cynt2(i,ll)
     +              *cint2(j)
 433            continue
              endif
c     missing point theta>pi/2, ll
              if (itest2.eq.0 .or. itest2.eq.1) then
                iieff=iy_(ll)+1-i
                iieffm=ilpm1ef(iieff,ll,-1)
                iieffc=ilpm1ef(iieff,ll,ipshft1+ipshft2)
                iieffd=ilpm1ef(iieff,ll,ip1t1m2)
                iieffe=ilpm1ef(iieff,ll,ip2t1p1)
                do 434 j=2,jx
                  zdns1(ll)=zdns1(ll)+fnp1(iieff,j,k,ll)*cynt2(iieff,ll)
     !              *cint2(j)
                  zdnsb(ll)=zdnsb(ll)+(zfcen*fnp1(iieff,j,k,ll)+
     !              (1.d0-zfcen)*fnp1(iieffc,j,k,ll+ipshft1+ipshft2)
     -              -fnhalf(iieffd,j,k,ll+ip1t1m2))*cynt2(iieff,ll)
     +              *cint2(j)
                  zdnsthb(ii)=zdnsthb(ii)+(zfcen*fnp1(iieff,j,k,ll)+
     !              (1.d0-zfcen)*fnp1(iieffc,j,k,ll+ipshft1+ipshft2)
     -              -fnhalf(iieffd,j,k,ll+ip1t1m2))*cynt2(iieff,ll)
     +              *cint2(j)
                  zdnsr1b(ll)=zdnsr1b(ll)+
     !              (zvelmid*velsou(iieffd,j,k,ll+ip1t1m2)+
     +              (1.d0-zvelmid)*velsou(iieffe,j,k,ll+ip2t1p1))*
     !              cynt2(iieff,ll)*cint2(j)
                  zdnsr2b(ll)=zdnsr2b(ll)-x(j)*vnorm*coss(iieff,ll)/
     !              dszm5(ll)*(fnp1(iieff,j,k,ll)-fnp1(iieffm,j,k,ll-1))
     +              *cynt2(iieff,ll)*cint2(j)
                  zdn2thb(ii)=zdn2thb(ii)-x(j)*vnorm*coss(iieff,ll)/
     !              dszm5(ll)*(fnp1(iieff,j,k,ll)-fnp1(iieffm,j,k,ll-1))
     +              *cynt2(iieff,ll)*cint2(j)
 434            continue
              endif
            endif

 300      continue ! l

 200    continue

C%OS  
c     compute highest errors
        do 900 l=1,ls
          do 9001 i=1,10
            zermx(i,l) = 0.d0
            zermxii(i,l) = 0.d0
 9001     continue
 900    continue ! l
c
        do 901 ll=1,ls  ! YuP: Notice: ll here, not l
          do 902 i=1,iyh_(ll)
            ii=iy_(ll)+1-i ! YuP[2021-03-12] iymax-->iy_(ll)
            zsumj3(i,ll) = 0.d0
            zsumj4(i,ll) = 0.d0
            zsumj3(ii,ll) = 0.d0
            zsumj4(ii,ll) = 0.d0
            !write(*,*)'ll,i,ii=',ll,i,ii
            do 910 j=2,jx
              zsumj3(i,ll) = zsumj3(i,ll) + f(i,j,k,ll)
              zsumj4(i,ll) = zsumj4(i,ll) + fxsp(i,j,k,ll)
              zsumj3(ii,ll) = zsumj3(ii,ll) + f(ii,j,k,ll)
              zsumj4(ii,ll) = zsumj4(ii,ll) + fxsp(ii,j,k,ll)
              zerabs = abs(f(i,j,k,ll)-fxsp(i,j,k,ll))
              if (zerabs .le. zermx(10,ll)) go to 914
              if (zerabs .gt. zermx(1,ll)) then
                do 911 i2=9,1,-1
                  zermx(i2+1,ll) = zermx(i2,ll)
                  imx(i2+1,ll) = imx(i2,ll)
                  jmx(i2+1,ll) = jmx(i2,ll)
 911            continue
                zermx(1,ll) = zerabs
                imx(1,ll) = i
                jmx(1,ll) = j
                go to 914
              endif
              do 912 int=9,1,-1
                if (zerabs.le.zermx(int,ll) .and. 
     !            zerabs.gt.zermx(int+1,ll)) then
                  do 913 i2=9,int+1,-1
                    zermx(i2+1,ll) = zermx(i2,ll)
                    imx(i2+1,ll) = imx(i2,ll)
                    jmx(i2+1,ll) = jmx(i2,ll)
 913              continue
                  zermx(int+1,ll) = zerabs
                  imx(int+1,ll) = i
                  jmx(int+1,ll) = j
                  go to 914
                endif
 912          continue

 914          zerabs = abs(f(ii,j,k,ll)-fxsp(ii,j,k,ll))
              if (zerabs .le. zermxii(10,ll)) go to 910
              if (zerabs .gt. zermxii(1,ll)) then
                do 915 i2=9,1,-1
                  zermxii(i2+1,ll) = zermxii(i2,ll)
                  imxii(i2+1,ll) = imxii(i2,ll)
                  jmxii(i2+1,ll) = jmxii(i2,ll)
 915            continue
                zermxii(1,ll) = zerabs
                imxii(1,ll) = i
                jmxii(1,ll) = j
                go to 910
              endif
              do 916 int=9,1,-1
                if (zerabs.le.zermxii(int,ll) .and. 
     !            zerabs.gt.zermxii(int+1,ll)) then
                  do 917 i2=9,int+1,-1
                    zermxii(i2+1,ll) = zermxii(i2,ll)
                    imxii(i2+1,ll) = imxii(i2,ll)
                    jmxii(i2+1,ll) = jmxii(i2,ll)
 917              continue
                  zermxii(int+1,ll) = zerabs
                  imxii(int+1,ll) = i
                  jmxii(int+1,ll) = j
                  go to 910
                endif
 916          continue

 910        continue

            zdiffj(i,ll) = zsumj4(i,ll)-zsumj3(i,ll)
            zdiffj(ii,ll) = zsumj4(ii,ll)-zsumj3(ii,ll)
 902      continue

          do 920 j=2,jx
            zsumi3 = 0.d0
            zsumi4 = 0.d0
            do 921 i=1,iyh_(ll) !YuP[2021-03-08] was bug: 1,iyh_(l)
              ii=iy_(ll)+1-i !YuP[2021-03-12] iymax-->iy_(ll)
              zsumi3 = zsumi3 + f(i,j,k,ll) + f(ii,j,k,ll)
              zsumi4 = zsumi4 + fxsp(i,j,k,ll) + fxsp(ii,j,k,ll)
 921        continue
            zdiffi(j,ll) = zsumi4-zsumi3
 920      continue
c
C%OS  if (ll.eq.1 .and. (n/2)*2.eq.n)
C%OS  + write(6,'(1pe10.2,2i5)') (zermx(ii,ll),imx(ii,ll),jmx(ii,ll),ii=1,10)
 901    continue ! ll=1,ls


c     check solution for each species
        zerror=2.*abs(sumleft-sumrigt)/(sumleft+sumrigt)
c%OS  if (abs(zerror) .gt. 1.0E-08) then
        write(6,'(/," error(accuracy) in parallel transport equation =",
     1    1pe11.3," time-step n =",i4,"  dn1/2= ",e11.3,"  dn1= ",
     1    e11.3)') zerror,n,sumleft,sumlftn
        if (iactst .eq. "abort") stop 'wpcheck'
c%OS  endif

 100  continue

c%OS  
      zsums=0.d0
      zsumr1=0.d0
      zsumr1b=0.d0
      zsumr2=0.d0
      zsumr2b=0.d0
      zsum2th=0.d0
      zsum2thb=0.d0
      do 951 l=1,lrors
        zsums=zsums+zdns(l)
        zsumr1=zsumr1+zdnsr1(l)
        zsumr1b=zsumr1b+zdnsr1b(l)
        zsumr2=zsumr2+zdnsr2(l)
        zsumr2b=zsumr2b+zdnsr2b(l)
 951  continue
      do 952 i=1,iymax
        zsum2th=zsum2th+zdn2th(i)
        zsum2thb=zsum2thb+zdn2thb(i)
 952  continue
c%OS  

      return
      end
