c
c
      subroutine diagdens(xline,xmid,eline)
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     Compute, for distribution stored in temp4:
c     xline=(line density for complete poloidal turn)/2
c           (particles/cm**2)
c     xmid=density of midplane distribution (particles/cm**3).
c     eline=(complete) line energy density (ergs/cm**2)/(.5*m*vnorm**2)
c..................................................................

      save
      include 'param.h'
      include 'comm.h'

      xline=0.
      xmid=0.
      eline=0.
      call bcast(tam2,zero,jx)
      call bcast(tam1,zero,jx)
      do 10 j=1,jx
        do 11 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
          !Note that for meshy="fixed_mu" iy_(l_) can be less than iy
          tam2(j)=tam2(j)+temp4(i,j)*cynt2(i,l_)*vptb(i,lr_)
          tam1(j)=tam1(j)+temp4(i,j)*cynt2(i,l_)
 11     continue
 10   continue
      do 40 j=1,jx
        xline=xline+tam2(j)*cint2(j)
        xmid=xmid+tam1(j)*cint2(j)
        eline=eline+tam2(j)*cint2(j)*tcsgm1(j)
 40   continue
      return
      end
