c
c
      subroutine wptramu
      implicit integer (i-n), real*8 (a-h,o-z)
c..............................................................
c     Time advancement for parallel transport. (meshy="fixed_mu" case)
c     Assume non-constant total number of y mesh points vs. l: iy_(l).ne.cst
c     Assume l_lower(i)=1, for all i (checked in wpinitl)
c..............................................................

      include 'param.h'
      parameter(lmatsa=(lsa+1)*nbanda)

      include 'comm.h'
      dimension zmat(lmatsa),zxdumy(lsa+1)
      dimension zmat2(lmatsa),zxdumy2(lsa+1)
      dimension ism1(0:lsa+1), isdiag(0:lsa+1), isp1(0:lsa+1)
      dimension imaxper(0:lsa+1), iymxper(0:lsa+1), iyhper(0:lsa+1)

      include 'wpadvnc.h'
      dithta(i,j,l)=0.5-0.5*(1/(i+1))+0.5*(1/(iy_(l)+1-i))
      fpithta(i,j,k,l)=fnhalf(i+1,j,k,l)*(1.-dithta(i,j,l)) + 
     +  fnhalf(i  ,j,k,l)*dithta(i,j,l)

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
      zip1p2=ipshft1+ipshft2
      zim1p2=imshft1+imshft2
      ztrdrt=0.5/dtreff
      ip1t1m2=ipshft1*(1-ipshft2)
      im1t1m2=imshft1*(1-imshft2)
      if (lmidvel .eq. 0) ip1t1m2=0
      if (lmidvel .eq. 0) im1t1m2=0

c.......................................................................
cl    2. Construct the matrix. The equations are solved at a given species k
c     and theta i, but "simultaneously" for all momentum j. This
c     way, one can have longer inner loops over j than over l, as in
c     general jx > ls. The index j was chosen instead of i, because
c     y(i,l) mesh does not depend on j. But note that we consider the
c     equ. for i and ii=iymax+1-i "simultaneously".
c.......................................................................

c.......................................................................
cl    2.1 Initialize f_n+1/2, velsou and imaxper, iymxper, iyhper
c.......................................................................

      do 210 l=1,ls
        imaxper(l)=imax(l,lr_)
        iymxper(l)=iy_(l)
        iyhper(l)=iyh_(l)
 210  continue
      call dcopy(iyjx2*ngen*lrors,f(0,0,1,1),1,fnhalf(0,0,1,1),1)
      if (sbdry .eq. "periodic") then
        call dcopy(iyjx2*ngen,fnhalf(0,0,1,1),1,fnhalf(0,0,1,ls+1),1)
        call dcopy(iyjx2*ngen,fnhalf(0,0,1,ls),1,fnhalf(0,0,1,0),1)
        call dcopy(iyjx2*ngen,velsou(0,0,1,1),1,velsou(0,0,1,ls+1),1)
        call dcopy(iyjx2*ngen,velsou(0,0,1,ls),1,velsou(0,0,1,0),1)
        imaxper(0)=imaxper(ls)
        imaxper(ls+1)=imaxper(1)
        iymxper(0)=iymxper(ls)
        iymxper(ls+1)=iymxper(1)
        iyhper(0)=iyhper(ls)
        iyhper(ls+1)=iyhper(1)
      else
        call dcopy(iyjx2*ngen,fnhalf(0,0,1,1),1,fnhalf(0,0,1,0),1)
        call dcopy(iyjx2*ngen,fnhalf(0,0,1,ls),1,fnhalf(0,0,1,ls+1),1)
        call dcopy(iyjx2*ngen,velsou(0,0,1,1),1,velsou(0,0,1,0),1)
        call dcopy(iyjx2*ngen,velsou(0,0,1,ls),1,velsou(0,0,1,ls+1),1)
        imaxper(0)=imaxper(1)
        imaxper(ls+1)=imaxper(ls)
        iymxper(0)=iymxper(1)
        iymxper(ls+1)=iymxper(ls)
        iyhper(0)=iyhper(1)
        iyhper(ls+1)=iyhper(ls)
      endif

c.......................................................................
cl    2.1.2 Might need value of fnhalf at l=l+1/2. Node value still in f
c     (Already the case if mod(nummods,10)>4, as 2-D momentum eq.
c     solved at l+1/2)
c.......................................................................

      if (mod(nummods,10).le.4 .and. lmidvel.ne.0) then
        do 2120 k=1,ngen
          do 2121 j=1,jx
            do 2122 l=1,ls
              do 2123 i=1,min(iyhper(l+1),iyhper(l))
                fnhalf(i,j,k,l)=0.5*(fnhalf(i,j,k,l)+fnhalf(i,j,k,l+1))
                ii=iymxper(l)+1-i
                fnhalf(ii,j,k,l)=0.5*(fnhalf(ii,j,k,l)+
     1            fnhalf(iymxper(l+1)+1-i,j,k,l+1))
 2123         continue
 2122       continue
 2121     continue
 2120   continue
        if (sbdry .eq. "periodic") then
          do 2124 k=1,ngen
            do 2125 j=1,jx
              do 2126 i=1,iyhper(ls)
                fnhalf(i,j,k,0)=0.5*(fnhalf(i,j,k,ls)+fnhalf(i,j,k,1))
                iils=iymxper(ls)+1-i
                ii1=iymxper(1)+1-i
                fnhalf(iils,j,k,0)=0.5*(fnhalf(iils,j,k,ls)+
     1            fnhalf(ii1,j,k,1))
                fnhalf(i,j,k,ls+1)=fnhalf(i,j,k,0)
                fnhalf(ii1,j,k,ls+1)=fnhalf(iils,j,k,0)
 2126         continue
 2125       continue
 2124     continue
        else
          call dcopy(iyjx2*ngen,fnhalf(0,0,1,1),1,fnhalf(0,0,1,0),1)
          call dcopy(iyjx2*ngen,fnhalf(0,0,1,ls),1,fnhalf(0,0,1,ls+1)
     +      ,1)
        endif
      endif
      ivelmid=1
      if (mod(nummods,10).le.4 .and. lmidvel.ne.0) ivelmid=0

c.......................................................................
cl    2.1.3 Pre-determine matrix row and band index for each l
c.......................................................................
      do 213 l=1,lrors
        ism1(l)=1
        isdiag(l)=ileft+1
        isp1(l)=iband
        if (sbdry.eq."periodic" .and. (l.le.2.or. l.gt.ls/2)) then
          if (l .eq. 1) then
            ism1(l)=iband
            isp1(l)=iband-1
          else if (l .eq. 2) then
            ism1(l)=ileft
          else if (l .eq. ls/2+1) then
            isp1(l)=ileft
          else if (l .eq. ls/2+2) then
            ism1(l)=isdiag(l)+1
            isp1(l)=1
          else
            ism1(l)=iband
            isp1(l)=1
          endif
        endif
 213  continue

c.......................................................................
cl    2.2 Loop over general species k
c.......................................................................

      do 220 k=1,ngen

c     j=1 => zeroth order eq. to be solved, assuming velsou(j=1)=0
c     => f_n+1(j=1)=f_n+1/2(j=1)
        do 221 ll=1,ls
          call bcast(fnp1(0,1,k,ll),f(1,1,k,ll),iyp1+1)
 221    continue

c.......................................................................
cl    2.3 Loop over theta mesh
c.......................................................................

        do 230 i=1,iyhper(1)

c     special case if only one point: l_lower(i)=l_upper(i)=1
          if (l_upper(1) .eq. 1) then
            do 610 j=2,jx
              fnp1(i,j,k,1)=f(i,j,k,1)
              fnp1(iy_(1)+1-i,j,k,1)=f(iy_(1)+1-i,j,k,1)
 610        continue
            go to 230
          endif

          !YuP: Not used in this loop? ii=iymax+1-i
          do 2301 l=1,ls
            do 2302 icol=1,iband
              do 2303 j=2,jx
                bndmats(j,l,icol,1)=0.0
                bndmats(j,l,icol,2)=0.0
 2303         continue
 2302       continue
            do 2304 j=2,jx
              rhspar(l,j,1)=0.0
              rhspar(ls-2+l,j,1)=0.0  !YuP: from this line, the size of rhspar
                                      !should be 2*ls-2 (in first dimension) 
              rhspar(l,j,2)=0.0
 2304       continue
 2301     continue

c.......................................................................
cl    2.3.1 Construct matrices at each j , l and ls+2-l (if periodic)
c.......................................................................

          do 231 l=1+ipstrt*imstrt,l_upper(i)-ipend*imend
            ilsrow=lsbtopr(l)
            !write(*,*)'l,ilsrow=lsbtopr(l)=',l,ilsrow

c.......................................................................
cl    2.3.1.1 cos(theta) > 0
c.......................................................................

            if (l.lt.1+ipstrt .or. l.gt.l_upper(i)+ipend .or.
     1        i.gt.min(imaxper(l+ipshft1),imaxper(l+ipshft2)))
     +        go to 2312

            ztrders=vnorm*coss(i,l)/(ipshft2*dszp5(l)-ipshft1*dszm5(l))
            do 23110 j=2,jx
              bndmats(j,ilsrow,ism1(l),1)=dfloat(ipshft1)*
     +          (x(j)*ztrders+lmidvel*zip1p2*ztrdrt)
              bndmats(j,ilsrow,isdiag(l),1)=(2.-lmidvel)*ztrdrt-
     !          zip1p2*x(j)*ztrders
              bndmats(j,ilsrow,isp1(l),1)=dfloat(ipshft2)*
     +          (x(j)*ztrders+lmidvel*zip1p2*ztrdrt)
              rhspar(ilsrow,j,1)=cvmgt(
     1          wprhsmd(ilpm1ef(i,l,ip1t1m2),j,k,lpm1eff(l,ip1t1m2))
     1          ,wprhs(ilpm1ef(i,l,ip1t1m2),j,k,lpm1eff(l,ip1t1m2))
     1          ,ivelmid.eq.1 .or. (ipshft1+ipshft2).eq.0)
23110       continue

c.......................................................................
cl    2.3.1.2 cos(theta) < 0
c.......................................................................

 2312       continue
            if (l.lt.1+imstrt .or. l.gt.l_upper(i)+imend .or.
     1        i.gt.min(imaxper(l+imshft1),imaxper(l+imshft2)))
     +        go to 2313

            iieff=iymxper(l)+1-i
            iieff12=iymxper(l+im1t1m2)+1-i
            ztrders=vnorm*coss(iieff,l)
     +        /(imshft2*dszp5(l)-imshft1*dszm5(l))
            do 23120 j=2,jx
              bndmats(j,ilsrow,ism1(l),2)=dfloat(imshft1)*
     +          (x(j)*ztrders+lmidvel*zim1p2*ztrdrt)
              bndmats(j,ilsrow,isdiag(l),2)=(2.-lmidvel)*ztrdrt-
     !          zim1p2*x(j)*ztrders
              bndmats(j,ilsrow,isp1(l),2)=dfloat(imshft2)*
     +          (x(j)*ztrders+lmidvel*zim1p2*ztrdrt)
              rhspar(ilsrow,j,2)=cvmgt(wprhsmd(iieff12,j,k,l+im1t1m2)
     1          ,wprhs(iieff12,j,k,l+im1t1m2)
     1          ,ivelmid.eq.1 .or. (imshft1+imshft2).eq.0)
23120       continue

c.......................................................................
cl    2.3.1.3 point l=ls+2-l, i.e. bottom half cross-section, if not
c     yet treated
c.......................................................................

 2313       if (sbdry.ne."periodic" .or. l_upper(i).eq.ls .or. l.eq.1)
     1        go to 231 ! next l

            ll=ls+2-l
            ilsrow=lsbtopr(ll)
            !write(*,*)'ll,ilsrow=lsbtopr(ll)=',ll,ilsrow

c.......................................................................
cl    2.3.1.3.1 cos(theta) > 0, ll=ls+2-l
c.......................................................................

            if (i.gt.min(imaxper(ll+ipshft1),imaxper(ll+ipshft2))) 
     1        go to 2314

            ztrders=vnorm*coss(i,ll)
     +        /(ipshft2*dszp5(ll)-ipshft1*dszm5(ll))
            do 23131 j=2,jx
              bndmats(j,ilsrow,ism1(ll),1)=dfloat(ipshft1)*
     +          (x(j)*ztrders+lmidvel*zip1p2*ztrdrt)
              bndmats(j,ilsrow,isdiag(ll),1)=(2.-lmidvel)*ztrdrt-
     !          zip1p2*x(j)*ztrders
              bndmats(j,ilsrow,isp1(ll),1)=dfloat(ipshft2)*
     +          (x(j)*ztrders+lmidvel*zip1p2*ztrdrt)
              rhspar(ilsrow,j,1)=cvmgt(wprhsmd(i,j,k,ll+ip1t1m2)
     1          ,wprhs  (i,j,k,ll+ip1t1m2)
     1          ,ivelmid.eq.1 .or. (ipshft1+ipshft2).eq.0)
23131       continue

c.......................................................................
cl    2.3.1.3.2 cos(theta) < 0, ll=ls+2-l
c.......................................................................

 2314       continue
            if (i.gt.min(imaxper(ll+imshft1),imaxper(ll+imshft2))) 
     1        go to 231
            iieff=iymxper(ll)+1-i
            iieff12=iymxper(ll+im1t1m2)+1-i
            ztrders=vnorm*coss(iieff,ll)/(imshft2*dszp5(ll)-
     1        imshft1*dszm5(ll))
            do 23132 j=2,jx
              bndmats(j,ilsrow,ism1(ll),2)=dfloat(imshft1)*
     +          (x(j)*ztrders+lmidvel*zim1p2*ztrdrt)
              bndmats(j,ilsrow,isdiag(ll),2)=(2.-lmidvel)*ztrdrt-
     !          zim1p2*x(j)*ztrders
              bndmats(j,ilsrow,isp1(ll),2)=dfloat(imshft2)*
     +          (x(j)*ztrders+lmidvel*zim1p2*ztrdrt)
              rhspar(ilsrow,j,2)=cvmgt(wprhsmd(iieff12,j,k,ll+im1t1m2)
     1          ,wprhs(iieff12,j,k,ll+im1t1m2)
     1          ,ivelmid.eq.1 .or. (imshft1+imshft2).eq.0)
23132       continue

c.......................................................................
c     end of construction of matrix

 231      continue ! l


c.......................................................................
cl    2.3.2 Apply the boundary conditions
c.......................................................................

          if (sbdry .ne. "periodic" .or. laddbnd.ne.0)
     !      call wpbdry(i,k,ileft,iright,2)

c.......................................................................
cl    2.3.3 Solve the linear systems for each theta point
cl    If i>itl, combine equation for i and ii=iy-i+1, assuming
cl    perfect reflection at turning point, i.e. f(iyh)=f(iyh+1)
c.......................................................................

          ilslen=l_upper(i)-l_lower(i)+1
          if (sbdry .eq. "periodic" .and. l_upper(i).ne.ls) 
     1      ilslen=2*ilslen-1
          icombin=1
          if (sbdry.eq."periodic" .and. i.gt.itl_(1) .and. numindx.eq.2
     !      .and. laddbnd.eq.0) icombin=2

          do 233 j=2,jx

            do 2330 iic=1,2,icombin

c.......................................................................
cl    2.3.3.1 Arrange matrix according to band matrix solver used
c     Uses the fact that the matrix is tri-diagonal
c.......................................................................

              do 2331 jcolumn=1,iband
                do 2332 l=1,ilslen
                  ijmat=(l-1)*iband+jcolumn
                  zmat(ijmat)=bndmats(j,l,jcolumn,iic)
 2332           continue
 2331         continue

c     When combining eqs. for i and ii, one assumes that iband=5 with
c     only three non-zero element per row, at jcolumn=1, 3 and 5

              if (icombin .eq. 2) then
c     rhs:
                ilento=2*ilslen-2
                rhspar(ilslen-1,j,1)=rhspar(ilslen-1,j,1)+
     !            rhspar(ilslen-1,j,2)
                rhspar(ilslen,j,1)=rhspar(ilslen,j,1)+rhspar(ilslen,j,2)
                do 2333 l=ilslen+1,ilento-2,2
                  rhspar(l,j,1)=rhspar(ilento-l,j,2)
                  rhspar(l+1,j,1)=rhspar(ilento-l+1,j,2)
 2333           continue
                rhspar(ilento,j,1)=rhspar(1,j,2)
c     matrix:
c     point ilslen-1 (i.e. l_upper(i)):
                ijdiag=(ilslen-2)*iband+3
                zmat(ijdiag)=zmat(ijdiag)+bndmats(j,ilslen-1,3,2)
                zmat(ijdiag+2)=bndmats(j,ilslen-1,1,2)
c     point ilslen (i.e. ls+2-l_upper(i)):
                ijdiag=(ilslen-1)*iband+3
                zmat(ijdiag)=zmat(ijdiag)+bndmats(j,ilslen,3,2)
                zmat(ijdiag+2)=bndmats(j,ilslen,1,2)
c     points ilslen-2 to 2:
                do 2334 jcolumn=1,iband,2
                  ijeff=(ilslen-2)*iband+(iband-jcolumn+1)
                  do 2335 l=ilslen-3,2,-2
                    ijeff=ijeff+2*iband
                    zmat(ijeff)=bndmats(j,l,jcolumn,2)
                    zmat(ijeff+iband)=bndmats(j,l+1,jcolumn,2)
 2335             continue
 2334           continue
c     new total number of unknown
                ilslen=2*ilslen-2
c     correct point l=3 (i.e. l=ls):
                ijdiag=(ilslen-2)*iband+3
                zmat(ijdiag+1)=zmat(ijdiag+2)
                zmat(ijdiag+2)=0.0
c     correct point l=2:
                ijdiag=(ilslen-3)*iband+3
                zmat(ijdiag+2)=bndmats(j,2,2,2)
c     point l=1:
                ijdiag=(ilslen-1)*iband+3
                zmat(ijdiag-2)=bndmats(j,1,4,2)
                zmat(ijdiag-1)=bndmats(j,1,5,2)
                zmat(ijdiag  )=bndmats(j,1,3,2)
              endif ! icombin .eq. 2

c%OS  
              call dcopy(ilslen*iband,zmat,1,zmat2,1)
              call dcopy(ilslen,rhspar(1,j,iic),1,zxdumy2,1)
c%OS  

c.......................................................................
cl    2.3.3.2 Call solver
c.......................................................................

              do 2336 l=ilslen+1,ilslen+iright
                rhspar(l,j,iic)=0.0
 2336         continue
              zerr=1.0e-08
              call nonsym(zmat,zxdumy,rhspar(1,j,iic),ilslen,ileft
     +          ,iright,zerr,icond)
              if (icond .ne. 0) write(6,'(/," WARNING: bad condition in"
     +          ," nonsym: icond = ",i4)') icond

c%OS  check solution
!YuP[2021-03-08] Something is wrong here: l-ileft-1+jcol gets to -1
! but 1st dimension in rhspar starts with 0.  Commenting this part 
              zdiff=0.0
              do 2337 l=1,ilslen
                zzzl=0.0
                do 2338 jcol=1,iband
                !write(*,*)' l-ileft-1+jcol=',l-ileft-1+jcol !YuP: becomes -1
                !when ileft=2 (for sbdry="periodic") , jcol=1.
                !It seems this version (two lines below) is only valid
                !when sbdry.ne."periodic" (in which case ileft=1)
!YuP                  zzzl=zzzl+zmat2((l-1)*iband+jcol)
!YuP     !              *rhspar(l-ileft-1+jcol,j,iic)
 2338           continue
                zdiff=zdiff + abs((zxdumy2(l)-zzzl)/zxdumy2(l))
 2337         continue
              zdiff=zdiff/(1.*ilslen)
c%OS  if (zdiff .gt. 1.0E-08) print *,' zdiff,i,j,n= ',zdiff,i,j,n
c%OS  

c     redistribute solution for ii points onto rhspar(.,j,2) and redefine ilslen
              if (icombin .eq. 2) then
                ilslen=ilslen/2+1
                do 2339 l=ilslen-1,ilento-2,2
                  rhspar(ilento-l,j,2)=rhspar(l,j,1)
                  rhspar(ilento-l+1,j,2)=rhspar(l+1,j,1)
 2339           continue
                rhspar(1,j,2)=rhspar(ilento,j,1)
              endif

 2330       continue
 233      continue

c.......................................................................
cl    2.3.4 Copy solution onto fnp1
c.......................................................................

          do 2340 l=1,ilslen
            ileff=lsprtob(l)
            iieff=iymxper(ileff)+1-i
            do 2341 j=2,jx
              fnp1(i,j,k,ileff)=rhspar(l,j,1)
              fnp1(iieff,j,k,ileff)=rhspar(l,j,2)
 2341       continue
 2340     continue

c     end of loop over theta
 230    continue

c     end of loop over general species
 220  continue

      if (sbdry .eq. "periodic") then
        call dcopy(iyjx2*ngen,fnp1(0,0,1,1),1,fnp1(0,0,1,ls+1),1)
        call dcopy(iyjx2*ngen,fnp1(0,0,1,ls),1,fnp1(0,0,1,0),1)
      else
        call dcopy(iyjx2*ngen,fnp1(0,0,1,1),1,fnp1(0,0,1,0),1)
        call dcopy(iyjx2*ngen,fnp1(0,0,1,ls),1,fnp1(0,0,1,ls+1),1)
      endif

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
            do 420 i=1,iymxper(l)
              zs=zs+cynt2(i,l)
              zt=zt+cynt2(i,l)*fnp1(i,1,k,l)
 420        continue
            do 430 i=1,iymxper(l)
              fnp1(i,1,k,l)=zt/zs
 430        continue
 410      continue

          if (sbdry .eq. "periodic") then
            call dcopy(iyp1+1,fnp1(0,1,k,1),1,fnp1(0,1,k,ls+1),1)
            call dcopy(iyp1+1,fnp1(0,1,k,ls),1,fnp1(0,1,k,0),1)
          else
            call dcopy(iyp1+1,fnp1(0,1,k,1),1,fnp1(0,1,k,0),1)
            call dcopy(iyp1+1,fnp1(0,1,k,ls),1,fnp1(0,1,k,ls+1),1)
          endif

 400    continue

      endif

      return
      end
