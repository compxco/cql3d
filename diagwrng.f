c
c
      subroutine diagwrng(kerr)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c...  
c
cmnt  diagnostic error messages  + chkpnt if ichkpnt.ne."disabled"
c
c...  

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      character cptline*80
      integer getpid
      

CMPIINSERT_IF_RANK_EQ_0
! Print-out from id=0 only:

      if (kerr.eq.1) then
        WRITE(*,10010)
      elseif (kerr.eq.2) then
        WRITE(*,10020)
      elseif (kerr.eq.3) then
        WRITE(*,10030)
      elseif (kerr.eq.4) then
        WRITE(*,10040)
      elseif (kerr.eq.5) then
        WRITE(*,10050)
      elseif (kerr.eq.6) then
        WRITE(*,10060)
      elseif (kerr.eq.7) then
        WRITE(*,10070)
      elseif (kerr.eq.8) then
        WRITE(*,10080)
      elseif (kerr .eq. 9) then
        WRITE(*,10090)
      elseif (kerr .eq. 10) then
        WRITE(*,10100)
      elseif (kerr .eq. 11) then
        WRITE(*,10110)
      elseif (kerr.eq.12) then
        WRITE(*,10120)
      elseif(kerr.eq.13) then
        WRITE(*,10130)
      elseif(kerr.eq.14) then
        WRITE(*,10140)
      elseif(kerr.eq.15) then
        WRITE(*,10150)
      elseif(kerr.eq.16) then
        WRITE(*,10160)
      elseif (kerr.eq.17) then
        WRITE(*,10170)
      elseif (kerr.eq.18) then
        WRITE(*,10180)
      elseif (kerr.eq.19) then
        WRITE(*,10190)
      elseif (kerr.eq.20) then
        WRITE(*,10200) iyh,iy,l_
      elseif (kerr.eq.21) then
        WRITE(*,10210) itu,iy,itl,l_
      elseif (kerr.eq.22) then
        WRITE(*,10220)
      elseif (kerr.eq.99) then
        WRITE(*,10990)
      else
        WRITE(*,10990)
      endif

c..................................................................
c     Check-point this job, if ichkpnt.ne."disabled" into file ichkpnt.
c..................................................................
      if (ichkpnt.ne."disabled") then
cBH080118        i=getpid(0)
        WRITE(cptline,100) i,ichkpnt
 100    format("chkpnt -p ",i5," -f ",a8)
        PRINT *,cptline
c        i=dropfile(ichkpnt)
      endif
      
      WRITE(*,*) 'diagwrng: kerr=',kerr 
      
CMPIINSERT_ENDIF_RANK


      stop   ! can be stopped by any core
c      if (kerr.eq.0) return
      return
      
10010 format("velocity normalization error called from ainitial")
10020 format("iteration non-convergence subroutine psiinv")
10030 format("machine type parameters not properly specified in input")
10040 format("arrays avar, ja, ia, rhs given insufficient space"
     1  ,/," icrease their size in advnce")
10050 format("subroutine diagwrng: lefct out of range")
10060 format("function psifppy not equipped for non-extrema")
10070 format("out of range error, subroutine psiinv")
10080 format("error called from routine micgetr")
10090 format("Code demands at least 1 electron and 1 ion species")
10100 format("Negative density computed in diaggnde")
10110 format("implct must be <enabled> for sweep option")
10120 format("namelist error:ngen(nmax).gt.nmax(nmaxa)")
10130 format("subroutine micxinit: failure in y mesh")
10140 format("namelist error: ylower < y(itl,l_) < yupper")
10150 format("iactst = abort forces stop, error > 1.e-8")
10160 format("njene must be less than njenea")
10170 format("(lrzdiff=enabled .and. radial transport) not allowed")
10180 format("(lrzdiff=enabled .and. frmod=enabled) not allowed")
10190 format("(urfmod=enabled .and. meshy=fixed_mu) not allowed")
10200 format("iyh .ne. iy/2, iyh=",i4,"  iy=",i4,"  l_=",i4)
10210 format("itu .ne. iy-itl+1, itu=",i4,"  iy=",i4,"  itl=",i4,
     1  "  l_=",i4)
10220 format("(urfmod=enabled.and.iy.gt.255)not allowed. urfpack:ilim1")
10990 format("unspecified")

      end
