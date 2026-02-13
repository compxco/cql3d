c
c
      subroutine wpwrng(kerr)
      implicit integer (i-n), real*8 (a-h,o-z)

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

! print error messages - on mpirank.eq.0 only
CMPIINSERT_IF_RANK_EQ_0
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
      elseif (kerr.eq.9) then
        WRITE(*,10090)
      elseif (kerr.eq.10) then
        WRITE(*,10100)
      elseif (kerr.eq.11) then
        WRITE(*,10110)
      elseif (kerr.eq.12) then
        WRITE(*,10120)
      elseif (kerr.eq.13) then
        WRITE(*,10130)
      elseif (kerr.eq.14) then
        WRITE(*,10140)
      elseif (kerr.eq.15) then
        WRITE(*,10150)
      elseif (kerr.eq.16) then
        WRITE(*,10160)
      elseif (kerr.eq.17) then
        WRITE(*,10170)
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
CMPIINSERT_ENDIF_RANK

      stop 'wpwrng:' ! stop at all MPI cores

10010 format(" cqlpmod and colmodl=4 not allowed")
10020 format(" Cannot have periodic mesh along B and non full l mesh "
     +  ,"for some pitch-angle")
10030 format(" lmidpln .ne. 1 not yet valid with CQLP")
10040 format(" Cannot run parallel transport with lsdiff=enabled")
10050 format(" error in micxinil in calculating imax")
10060 format(" error in micxinil in calculating ill and cosz")
10070 format(" updown .ne. symmetry with 9<nummods<20")
10080 format(" sbdry .ne. periodic with 9<nummods<20")
10090 format(" ls odd with 9<nummods<20")
10100 format(" lz odd with 9<nummods<20")
10110 format(" iymax .ne. iy_(1)")
10120 format(" l_lower(i).ne.1")
10130 format(" l_upper(i).ne.ls for i<itl_(1) or i>itu_(1)")
10140 format(" i.ne.iyh_(l_upper(i))")
10150 format(" wptrafx called with a non-constant y mesh")
10160 format(" numixts should equal +1 or -1 with numindx=2 option")
10170 format(" lmidvel should equal 0 with numindx=4 (centered scheme)")
10990 format(" unspecified in wpwrng")

      end
