c
c
      subroutine tdtrcon
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c..............................................................
c     Compute conservation constant.
c..............................................................
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE
      data sgainr /0.0/
c jakub urban 110708: commented out for g95 compiler
c the above blanket save statement should do the job
c      save sgainr

c..............................................................
c     Compute original number of particles in tokamak.
c..............................................................
      if (n.eq.1) then
        total0=0.
        do 10 l=1,lrz
          ilr=lrindx(l)
          total0=xlndn0(ilr)/zmaxpsi(ilr)*dvol(ilr)+total0
 10     continue
      endif

c..............................................................
c     Compute total number of particles in device now.
c...............................................................

      total_xlnd=0.
      total_reden=0.d0
      do 30 l=1,lrz
        ilr=lrindx(l)
        do 40 k=1,ngen
          total_xlnd=  xlndn(k,ilr)/zmaxpsi(ilr)*dvol(ilr)+total_xlnd ! ZOW
          total_reden= reden(k,ilr)*dvol(ilr) +total_reden ! for test
 40     continue
 30   continue

c................................................................
c     Call routine to compute number of particles lost at limiter
c     this time step.
      if(transp.eq.'enabled') then
      call tdtrflx
      sgainr=sgainr+flxout
      endif
c................................................................
      !conserv=(total_xlnd-total0-sgainr)/(total_xlnd*.5+total0*.5)
      ! Comparing to original Nptcls:
      conserv=(total_xlnd-total0-sgainr)/total0
      return
      end
