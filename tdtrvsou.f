c
c
      subroutine tdtrvsou(k)
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     compute source term, due to velocity operator evaluated on
c     f_n+1/2, needed for the transport equation solved with ADI.
c
c     For up/down symmetry cases, also compute the value of f
c     at l_=1 and ls (if sbdry.ne."periodic")
c..............................................................

      include 'param.h'
      include 'comm.h'
      dimension zdns(lrorsa),zdns1(lrorsa) !local (not really needed)

      include 'advnce.h'
      fpithta(i,j)=f(i+1,j,k,l_)*(1.-dithta(i,j,l_)) + 
     +             f(i  ,j,k,l_)*dithta(i,j,l_)
c.......................................................................

c     include cthta in vsou at n=nontran-1/2
c%OS  if (n .eq. nontran) call wpcthta

      zdns(l_)=0.0
      zdns1(l_)=0.0

c.......................................................................
cl    1. Compute velsou for main mesh points
c.......................................................................

      do 100 i=1,iy_(l_) !YuP[2021-03-08] was iy
        !Note that for meshy="fixed_mu" iy_(l_) can be less than iy
        !But don't use iy=iy_(l_) here! It will overwrite iy,
        !which is the namelist value, and iy is used to set dimensions
        !of many arrays.
        if ((i.eq.itl.or.i.eq.itu) .and. cqlpmod.ne."enabled") go to 100
c%OS  should be j=2,jx-1? may depend on lbdry
c%OS  do 110 j=1,jx
        do 110 j=2,jx-1
          velsou(i,j,k,l_)=qz(j)*(gfi(i,j,k)-gfi(i,j-1,k))+
     +      ry(i,j)*(hfi(i,j)-hfi(i-1,j))+
     +      vptb(i,lr_)*(cah(i,j)*f(i,j,k,l_)+so(i,j))+
     +      cthta(i,j)*(fpithta(i,j)-fpithta(i-1,j))
          velsou2(i,j,k,l_)=cthta(i,j)*(fpithta(i,j)-fpithta(i-1,j))
          zdns(l_)=zdns(l_)+velsou(i,j,k,l_)*cynt2(i,l_)*cint2(j)
          zdns1(l_)=zdns1(l_)+velsou2(i,j,k,l_)*cynt2(i,l_)*cint2(j)
 110    continue
        if (cqlpmod.eq."enabled" .and. updown.eq."symmetry" .and. 
     +    sbdry.ne."periodic") then
          if (l_ .le. 2) then
            do 111 j=2,jx-1
c%OS  fedge(i,j,k,l_)=velsou(i,j,k,l_)*dtreff-
c%OS  -                      (dtreff+0.5*dszp5(1)/vnorm/x(j)/coss(i,l_))*
c%OS  *                          cthta(i,j)*(fpithta(i,j)-fpithta(i-1,j))
              fedge(i,j,k,l_)=-dszp5(1)/vnorm/x(j)/coss(i,l_)*
     *          cthta(i,j)*(fpithta(i,j)-fpithta(i-1,j))
 111        continue
          else if (l_ .ge. lrors-1) then
            do 112 j=2,jx-1
c%OS  fedge(i,j,k,l_-lrors+4)=velsou(i,j,k,l_)*dtreff-
c%OS  -                     (dtreff+0.5*dszm5(ls)/vnorm/x(j)/coss(i,l_))*
c%OS  *                          cthta(i,j)*(fpithta(i,j)-fpithta(i-1,j))
              fedge(i,j,k,l_-lrors+4)=-dszm5(ls)/vnorm/x(j)/
     /          coss(i,l_)*cthta(i,j)*(fpithta(i,j)-fpithta(i-1,j))
 112        continue
          endif
        endif
 100  continue

c..................................................................
cl    2. Compute velsou at pass/trapped boundary
c..................................................................

      if (cqlpmod .ne. "enabled") then
        do 200 j=1,jx
          velsou(itl,j,k,l_)=qz(j)*(gfi(itl,j,k)-gfi(itl,j-1,k))+
     +      r2y(j)*(-hfi(itl-1,j)+2.*hfi(itl,j)+hfi(itu,j))+
     +      vptb(itl,lr_)*(cah(itl,j)*f(itl,j,k,l_)+so(itl,j)) !CQL3D only
 200    continue
      endif

c..................................................................
cl    2.1 Symmetry about pi/2. in trapped region.
c..................................................................

      if (symtrap .eq. "enabled") then
        do 210 i= iyh_(l_)+1,itu
          ii= iy_(l_)+1-i !YuP[2021-03-08]  iy-->iy_(l_), iyh-->iyh_(l_)
          do 215 j=1,jx
            velsou(i,j,k,l_)=velsou(ii,j,k,l_)
 215      continue
 210    continue
      endif

c.......................................................................
cl    3. Interpolate on radial velocity mesh, keeping integral of velsou
cl    over velocity conserved (for radial transport case)
c.......................................................................

      if (cqlpmod .ne. "enabled")
     +  call tdtrvtor3(velsou(0,0,1,1),velsou(0,0,1,1),cynt2,cynt2_,2,k)

      return
      end
