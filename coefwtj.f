c
c
      subroutine coefwtj(k)
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     This routine computes w = dy(i,l_)*dd(i,j)/df(i,j) and then sets
c     g(w)=di(i,j,k,l_)=1./w - (1./(exp(w)-1.))
c     Note g(minus infinity)=1.; g(0)=.5; g(plus infinity)=0.
c     This is the Chang-Cooper algorithm (2-d form ala Karney)
c     This routine actually has modified the above by subtracting
c     off the RF contribution to the diffusion (see the coding).
c     This procedure, while not yet
c     justified theoretically, has kept the distribution from
c     going negative for strongly driven problems.
c..................................................................

      include 'param.h'
      include 'comm.h'

      include 'advnce.h'
      
c-YuP      call bcast(dj(1,0,k,l_),half,iyjxp1) ! could it be error?
      call bcast(dj(0,0,k,l_),half,(iymax+2)*(jx+1))  !YuP[2021-03-11] iy-->iymax
      !Note:  dj(0:iymax+1,0:jx,1:ngen,lrors)
      
      if (chang.ne."disabled") then
c..................................................................
c     RF diffusion alone is in dbb, total in db.
c..................................................................

        op=one+em12
        call coefmidv(dbb,2)

        
        do 10 j=1,jx
        
          do 20 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            db_dbb= op*db(i,j)-dbb(i,j) !YuP[2019-07-09] added check of denom=0
            if(db_dbb.ne.zero)then
            temc1(i)=dx(j)*da(i,j)*op*db(i,j)/db_dbb**2
            else
            !write(*,*)'coefwtj: db_dbb=0.', op*db(i,j),dbb(i,j)
            temc1(i)=zero
            endif
 20       continue

c.................................................................
c     Limit magnitudes so exp or 1/temc1 do not blow up
c.................................................................

          do 81 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            if(abs(temc1(i)).gt.em6) then
              temc3(i)=temc1(i)
            else
              temc3(i)=em6
            endif
 81       continue

          do 82 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            if(temc3(i).lt.sevenhun) then
              temc2(i)=temc3(i)
            else
              temc2(i)=sevenhun
            endif
 82       continue

          do 83 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            if(temc2(i).le.-sevenhun) then
              temc2(i)=-sevenhun
            endif
 83       continue

c.................................................................
c     Evaluate the Chang-Cooper weight 
c.................................................................

          do 84 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            dj(i,j,k,l_)=1.d0/temc2(i)-(1.d0/(exp(temc2(i))-1.d0))
 84       continue

c...............................................................
c     Limit for small temc1 follows...
c...............................................................

          do 21 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            wsub=(3.d0+temc1(i))/(2.d0+temc1(i))/3.d0
            if(temc2(i).eq.em6) then
              dj(i,j,k,l_)=wsub
            endif
 21       continue

c..............................................................
c     Limit for large positive or negative temc1 follows
c..............................................................

          do 22 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            if (temc2(i).eq.sevenhun) then
              dj(i,j,k,l_)=1.d0/temc1(i)
            elseif (temc2(i).eq.-sevenhun) then
              dj(i,j,k,l_)=1.d0+1.d0/temc1(i)
            endif
 22       continue
 
 10     continue ! j=1,jx
 
      endif ! if (chang.ne."disabled")


c.......................................................................
c     Ensures correct differentiation at end of intervals
c.......................................................................

      do 30 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
        dj(i,jx,k,l_)=1.d0
        dj(i,0,k,l_)=0.d0
 30   continue


      return
      end
