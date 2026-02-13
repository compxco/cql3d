c
c
      subroutine sourc0
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     define source uniquely at v=0
c..................................................................

      include 'param.h'
      include 'comm.h'

      do 10 k=1,ngen
        s=0.
        u=0.
        do 11 i=1,iy
          !u=u+source(i,1,k,indxlr_)*cynt2(i,l_)*vptb(i,lr_) !before[2022-02-11]
          u=u+source(i,1,k,l_)*cynt2(i,l_)*vptb(i,lr_) !after[2022-02-11]
          s=s+cynt2(i,l_)*vptb(i,lr_)
 11     continue
        if(s.gt.0.d0)then !YuP[2020-10-19] added checking s>0
          do i=1,iy
             !source(i,1,k,indxlr_)=u/s !before[2022-02-11]
             source(i,1,k,l_)=u/s !after[2022-02-11]
          enddo
        endif
 10   continue
      return
      end
