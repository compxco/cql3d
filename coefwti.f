c
c
      subroutine coefwti(k)
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     This routine computes w = dy(i,l_)*dd(i,j)/df(i,j) and then sets
c     g(w)=di(i,j,k,l_)=1./w - (1./(exp(w)-1.))
c     Note g(minus infinity)=1.; g(0)=.5; g(plus infinity)=0.
c     This is the Chang-Cooper algorithm (2-d form ala Karney)
c     This routine actually has modified the above by subtracting
c     off the RF contribution to the diffusion (see the coding).
c     This procedure, while not yet
c     justified theoretically, helps to keep the distribution from
c     going negative for strongly driven problems.
c..................................................................

      include 'param.h'
      include 'comm.h'

      include 'advnce.h'
      
c-YuP      call bcast(di(0,1,k,l_),half,(iy+1)*jx) ! could it be error?
      call bcast(di(0,0,k,l_),half,(iymax+1)*(jx+2)) !YuP[2021-03-11] iy-->iymax
      !Note: di(0:iymax,0:jx+1,1:ngen,lrors)

      if (chang .ne. "disabled") then
      
        if (chang.eq."noneg") then   
          if (n.eq.1) jchang(k,l_)=jx !This is the first call of coefwti;
                 !so that jchang from previous time step are not defined.
          do 100 j=1,jx
            do 1001 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
              if (f(i,j,k,l_).le.0.d0) then
                !Note: This subr.coefwti is called from impavnc0
                ! before the solution is obtained. So, the f() values here
                ! are from the previous time step. 
                jval=j !scanning from j=1 up, found the first occurence of f<0 (or f=0)
                go to 101 !Skip further scanning.
              endif
 1001       continue
 100      continue
          jval=jx !When the scan did not find any point with f<0 (or =0)
          !In this case the value of jchang remains same as at previous time step.
 101      continue ! jval can be in range from 1 to jx
          if (jval.ne.jx) then !Check if the value of jchang must be updated.
            if (jchang(k,l_).gt.jval-5) then !Here jchang is from previous time step
              !except n=1, then jchang=jx
              jchang(k,l_)=jval-5  !Re-define/reset to new value.
              !The purpose is to keep jchang close to the negative-f region.
              !If at prev. time step this region was starting at high j index,
              !but now it moved to lower j (so that now jval-5<jchang_old) 
              !then - move jchang_new to lower value.
              if (jchang(k,l_).le.0) jchang(k,l_)=1 !Make sure it is at least 1
            endif
            !If jval is large enough, so that jval-5>jchang_old, then 
            ! keep the old value unchanged, i.e. jchang remains unchanged
            ! from the previous time step. 
            !YuP: Why do we keep jchang from previous step in such a case?
            !Why not updating/setting jchang_new based on given f(), 
            !i.e., dropping any reference to the previous time step?
            !Note: This subr.coefwti is called from impavnc0  
            ! before the new solution for f() is obtained. 
            ! So, the f() values here are from the "old" step,
            ! while the saved values of jchang_old are from f_old_old values.
            ! At n=1(in impavnc0) f_old is just a Maxwellian distribution,
            ! while f_old_old in this sense does not exist (jchang was not set yet).
            ! That's why there is a special "if (n.eq.1) jchang(k,l_)=jx"
            ! clause above.
          endif ! jval.ne.jx
        elseif (chang.eq."enabled") then
          jchang(k,l_)=jx
        endif

       if (jchang(k,l_).le.0) jchang(k,l_)=1  
       !-YuP 101126: added to prevent jchang=0
        
        op=one+em8
        call coefmidt(dff,3)
        do 10 j=1,jchang(k,l_)
          do 2 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            df_dff= op*df(i,j)-dff(i,j) !YuP[2019-07-09] added check of denom=0
            if(df_dff.ne.zero)then
            temc1(i)=dy(i,l_)*dd(i,j)*op*df(i,j)/df_dff**2
            else
            !write(*,*)'coefwti: df_dff=0.', op*df(i,j),dff(i,j)
            temc1(i)=zero
            endif
 2        continue

c...............................................................
c     Keep code from blowing up in the 9 loop below.
c..............................................................

          do 8 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            if(abs(temc1(i)).gt.em6) then
              temc3(i)=temc1(i)
            else
              temc3(i)=em6
            endif
 8        continue

          do 4 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            if(temc3(i).lt.sevenhun) then
              temc2(i)=temc3(i)
            else
              temc2(i)=sevenhun
            endif
 4        continue

          do 7 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            if(temc2(i).le.-sevenhun) then
              temc2(i)=-sevenhun
            endif
 7        continue

c..............................................................
c     Evaluate the ratio (modified)
c..............................................................

          do 9 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            di(i,j,k,l_)=1.d0/temc2(i)-(1.d0/(exp(temc2(i))-1.d0))
 9        continue

c.............................................................
c     Correct for errors in 9 above
c.............................................................

          do 21 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            wsub=(three+temc1(i))/(two+temc1(i))/three
            if(temc3(i).eq.em6) then
              di(i,j,k,l_)=wsub
            endif
 21       continue

c..............................................................
c     Limit for large positive or negative temc1 follows
c..............................................................

          do 22 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
            if (temc2(i).eq.sevenhun) then
              di(i,j,k,l_)=1.d0/temc1(i)
            elseif (temc2(i).eq.-sevenhun) then
              di(i,j,k,l_)=1.d0+1.d0/temc1(i)
            endif
 22       continue
 10     continue ! j=1,jchang(k,l_)

c...............................................................
c     Now force one-sided differencing if desired.
c...............................................................
       if (jchang(k,l_).le.0) jchang(k,l_)=1  
       !-YuP 101126: added to prevent jchang=0

        if (jchang(k,l_).lt.jx) then
          do 60 j=jchang(k,l_),jx !for all j points above first occurence of f<0 (actually -5 below)
            do 50 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
              if(dd(i,j).ge.0.d0) then
                di(i,j,k,l_)=zero
              else
                di(i,j,k,l_)=one
              endif
 50         continue
 60       continue ! j=1,jchang(k,l_)
        endif ! jchang(k,l_)<jx  which means there is region with f<0 (or f=0)
        
      endif ! chang.ne."disabled"

c.......................................................................
c     Ensures correct differentiation at end of intervals
c.......................................................................

      do 3 j=1,jx
        di(0,j,k,l_)=0.d0  ! i=0 only
        di(iy_(l_),j,k,l_)=1.d0 ! i=iy only  !YuP[2021-03-11] iy-->iy_(l_)
 3    continue
 

      return
      end
