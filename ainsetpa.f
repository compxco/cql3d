c
c
      subroutine ainsetpa
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     This routine should be called after (first) namelist setup0 is read.
c     Determines values of lrz, lrors, and ls according to model
c     if lrzdiff.eq."enabled":
c     lrindx: index of radial surface on which equations are solved
c     indxlr: index of array lrindx giving lr_=> indxlr(lrindx(l=1,lrz))=l.
c     indxlr used to save space (arrays source, gone, egylosa,..)
c     (indxlr labels the radial surfaces from 1 to lrz)
c     if lsdiff.eq."enabled" (operative for cqlpmod="enabled"):
c     lsindx: index of orbit position s on which equ. are solved
c     indxls: index of array lsindx giving ls_=> indxls(lsindx(l=1,ls))=l. 
c     (indxls labels the orbit position from 1 to ls).
c     lrindx(1)=namelist input
c     lrindx(2:ls)=lrindx(1), indxlr(lrindx(1))=1
c
c     Also sets values of some physical constants,
c     and some real*8 numbers used in subroutine arguments.
c..................................................................


      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

c.......................................................................
cl    1. General specification
c.......................................................................

      if (lrz.eq.0) lrz=lrza
c000720      if (lrza.lt.lrz) lrz=lrza
      if (lrza.lt.lrz) stop 'ainsetpa: lrz.lt.lrza'
c000720      if (lrzmax .gt. lrza) lrzmax=lrza
      if (lrzmax .gt. lrza) stop 'ainsetpa: lrzmax.gt.lrza'
      if (ls.eq.0) ls=lsa
c000720      if (lsa.lt.ls) ls=lsa
      if(lsa.lt.ls)stop 'ainsetpa: ls>lsa; Increase lsa in param.h'
c000720      if (lsmax .gt. lsa) lsmax=lsa
      if(lsmax.gt.lsa) stop 'ainsetpa: lsmax>lsa; Increase lsa'

      if (lrzdiff .ne. "enabled") lrzmax=lrz
      
      if (lrzmax .le. lrz) then
        lrzdiff="disabled"
        lrzmax=lrz
        do 100 l=1,lrz
          lrindx(l)=l
 100    continue
      endif

      if (lsdiff .ne. "enabled") lsmax=ls
      if (lsmax .le. ls) then
        lsdiff="disabled"
        lsmax=ls
        do 101 l=1,ls
          lsindx(l)=l
 101    continue
      endif


      if (lrzdiff .eq. "enabled") then

c     shift lrindx if lrindx(0).ne.0 
c     (assume error in namelist:lrindx=... instead of lrindx(1)=...)
        if (lrindx(0) .ne. 0) then
CMPIINSERT_IF_RANK_EQ_0
          WRITE(*,'(/"  WARNING: lrindx has been shifted by one index"
     +      ," as lrindx(0).ne.0",/)')
CMPIINSERT_ENDIF_RANK
cdir$ novector
          do 102 ll=lrz,1,-1
            lrindx(ll) = lrindx(ll-1)
 102      continue
          lrindx(0)=0
        endif
      endif
      do 110 l=1,lrz
        indxlr(lrindx(l))=l
 110  continue


      if (cqlpmod .ne. "enabled") then

c.......................................................................
cl    2. CQL3D: CQL or CQL3D (radial coordinate as variable)
c.......................................................................

        lrors=lrz ! CQL3D (FPE radial grid)
        ls=1
        lsmax=1
        lsdiff="disabled"
        lsindx(1)=1

      else ! (cqlpmod.eq."enabled")

c.......................................................................
c     3. CQLP (parallel): coordinate s along B is variable 
c        instead of radius
c.......................................................................
 300    continue

        lrors=ls ! CQLP (FPE grid along field line)
        if (lsdiff .ne. "enabled") then
          do 310 ll=1,ls
            lsindx(ll)=ll
 310      continue
        endif
        do 311 l=1,ls
          indxls(lsindx(l))=l
 311    continue

c     so far, CQLP can only treat one flux surface 
c     (need 4D storage arrays)
        lrz=1
        if (lrzdiff .eq. "disabled") then
          lrzmax=lrz
          lrindx(1)=1
          indxlr(1)=1
        endif
        do 315 l=1,ls
          lrindx(l)=lrindx(1)
 315    continue

      endif

      !YuP[2021-03-18] Added more pointers over lrors FPE grid.
      !(Previously, these were statically dimensioned in comm.h). 
      !Note: aclear,aindflt1 are called before ainalloc, so - 
      !need to do some allocation here.
      !(just after lrors is defined, see above)
      if(.NOT.ASSOCIATED(itl_))then
      allocate(itl_(lrors),STAT=istat)
      allocate(itu_(lrors),STAT=istat)
      allocate(iy_(lrors),STAT=istat)
      allocate(iyh_(lrors),STAT=istat)
      allocate(iyjx_(lrors),STAT=istat)
      allocate(n_(lrors),STAT=istat)
      allocate(nch(lrors),STAT=istat)
      allocate(time_(lrors),STAT=istat)
      endif
      itl_(:)=0
      itu_(:)=0
      iy_(:)=0
      iyh_(:)=0
      iyjx_(:)=0
      n_(:)=0
      nch(:)=0
      time_(:)=0.d0

c.......................................................................
cl    Set some physical, numerical and normalization constants.
cl    Numerical constants set to help maintain real*8 arithmetic.
c.......................................................................
      zero=0.d0
      one=1.d0
      two=2.d0
      three=3.d0
      four=4.d0
      sevenhun=700.d0
      one_=1.d0
      em6=1.d-6
      em8=1.d-8
      em10=1.d-10
      em12=1.d-12
      em14=1.d-14
      em37=1.d-37
      em40=1.d-40
      em90=1.d-90
      em100=1.d-100
      em120=1.d-120
      em300=1.d-300
      ep37=1.e+37
      ep90=1.d+90
      ep100=1.d+100
      half=one/two
      third=one/three
 
c      pi=3.141592653589793d0
      pi=atan2(zero,-one)
      pio180=pi/180.d0
      twopi=two*pi
      fourpi=4.0*pi
      stopm=-sqrt(two/pi)
      charge=4.8032d-10
      alp=7.2974d-3
      r0=2.817940285d-13
      pio2=pi*half
      ergtkev=1.6022d-09  !energy(ergs) associated with 1 keV
      clight=2.99792458d10
      clite2=clight**2
      proton=1.67262158d-24
      restmkev=510.998902d0

c.......................................................................
c     Additional variable initialization
c.......................................................................

      do 111 ll=0,lrza
        lmdpln(ll)=ll !radial index in CQL3D; For CQLP, it is reset to 1
 111  continue

      nonch=nstop+1 !!! Before 101220: nonch=noncha (==2000)

      it_ampf=0

      return
      end
