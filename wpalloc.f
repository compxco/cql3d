c
c
      subroutine wpalloc
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     Allocates arrays used by parallel transport routines.
c..............................................................
! ls:  number of s-mesh points for which the FP eq. is solved (FPE grid)
!(ls can be smaller than lsmax grid, where profiles like temppar are set)
! YuP[2021-02-26] Changed lsa-->ls for some of arrays below;
! All these arrays are used at FPE grid 

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE
c.......................................................................

c..................................................................
c     A check on allocations is sucessful entering then exiting
c     the subroutine.
c..................................................................
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'wpalloc:  Entering wpalloc'
CMPIINSERT_ENDIF_RANK

      ! YuP-101220: allocation of wcqlb-wcqlf is moved to vlf.f

      lnyxgs2=(iyp1+1)*(jxp1+1)*ngen*(ls+2)
      lny2gx=iymax*jx*ngen*4                 !YuP[2021-03-12] iy-->iymax
      lnsbn2y=(ls+nbanda+2)*max(iymax,jx)*2
      lnys2bn=max(iymax,jx)*(ls+2)*nbanda*2  !YuP[2021-03-12] iy-->iymax

      lndums=5*lnyxgs2+lny2gx+lnsbn2y+lnys2bn

      if (vlfmod.eq."enabled") then
         lnyxms=iymax*jx*nmodsa*ls  !YuP[2021-03-12] iy-->iymax
         lndums=lndums+4*lnyxms
      endif

      allocate(l_upper(1:iymax),STAT=istat)  !YuP[2021-03-12] iy-->iymax
      allocate(ilpm1ef(0:iymax+1,0:ls+1,-1:+1),STAT=istat) ![2012-02] lsa1-->ls+1
      call ibcast(l_upper,0,SIZE(l_upper))
      call ibcast(ilpm1ef,0,SIZE(ilpm1ef))

      !YuP[2021-03-18] Made these arrays into pointers (note: ls is FPE grid)
      allocate(lpm1eff(0:ls+1,-1:+1),STAT=istat) !(0:ls+1,-1:+1)
      allocate(lsbtopr(0:ls+1),STAT=istat)
      allocate(lsprtob(0:ls+1),STAT=istat)
      allocate(sz(0:ls+1),STAT=istat)
      allocate(dsz(0:ls+1),STAT=istat)
      allocate(dszm5(0:ls+1),STAT=istat)
      allocate(dszp5(0:ls+1),STAT=istat)
      allocate(eszm5(0:ls+1),STAT=istat)
      allocate(eszp5(0:ls+1),STAT=istat)
      allocate(psis(0:ls+1),STAT=istat)
      allocate(psisp(0:ls+1),STAT=istat)
      allocate(psipols(0:ls+1),STAT=istat)
      allocate(solrs(0:ls+1),STAT=istat)
      allocate(solzs(0:ls+1),STAT=istat)
      allocate(elparol(0:ls+1),STAT=istat)
      allocate(elparnw(0:ls+1),STAT=istat)
      allocate(flux1(0:ls+1),STAT=istat)
      allocate(flux2(0:ls+1),STAT=istat)
      lpm1eff(0:ls+1,-1:+1)=0 ! initialize
      lsbtopr(0:ls+1)=0   ! initialize
      lsprtob(0:ls+1)=0   ! initialize
      sz(0:ls+1)=0.d0    ! initialize
      dsz(0:ls+1)=0.d0   ! initialize
      dszm5(0:ls+1)=0.d0   ! initialize
      dszp5(0:ls+1)=0.d0   ! initialize
      eszm5(0:ls+1)=0.d0   ! initialize
      eszp5(0:ls+1)=0.d0   ! initialize
      psis(0:ls+1)=0.d0   ! initialize
      psisp(0:ls+1)=0.d0   ! initialize
      psipols(0:ls+1)=0.d0   ! initialize
      solrs(0:ls+1)=0.d0   ! initialize
      solzs(0:ls+1)=0.d0   ! initialize
      elparol(0:ls+1)=0.d0   ! initialize
      elparnw(0:ls+1)=0.d0   ! initialize
      flux1(0:ls+1)=0.d0   ! initialize
      flux2(0:ls+1)=0.d0   ! initialize


      !YuP[2021-03-12] iy-->iymax
      allocate(fnhalf(0:iymax+1,0:jx+1,ngen,0:ls+1),STAT=istat)
      call bcast(fnhalf,zero,SIZE(fnhalf))
      allocate(fnp0(0:iymax+1,0:jx+1,ngen,0:ls+1),STAT=istat)  
      call bcast(fnp0,zero,SIZE(fnp0))
      allocate(fnp1(0:iymax+1,0:jx+1,ngen,0:ls+1),STAT=istat)
      call bcast(fnp1,zero,SIZE(fnp1))
      allocate(dls(0:iymax+1,0:jx+1,ngen,0:ls+1),STAT=istat)
      call bcast(dls,zero,SIZE(dls))
      
      
      if(.NOT.ASSOCIATED(fh)) then ! fh is used in CQLP and A-F
        !YuP[2021-03-05]
        !allocate(fh(0:iymax+1,0:jx+1,1,1:lrz),STAT=istat) !For A-F
        allocate(fh(0:iymax+1,0:jx+1,ngen,0:ls+1),STAT=istat) !For CQLP
        if(istat.ne.0) STOP 'wpalloc: fh alloc problem'
      endif
      call bcast(fh,zero,SIZE(fh))

      if(.NOT.ASSOCIATED(fg)) then ! fg is used in CQLP and A-F
        !YuP[2021-03-05]
        !allocate(fg(0:iymax+1,0:jx+1,1,1:lrz),STAT=istat) !For A-F
        allocate(fg(0:iymax+1,0:jx+1,ngen,0:ls+1),STAT=istat) !For CQLP
        if(istat.ne.0) STOP 'wpalloc: fg alloc problem'
      endif
      call bcast(fg,zero,SIZE(fg))
     
      allocate(fedge(iymax,jx,ngen,4),STAT=istat)
      call bcast(fedge,zero,SIZE(fedge))
      
      !YuP[2021-03-08] From subr. wptramu it follows that 
      !the first dimension of rhspar should be 2*ls-2
      if(transp.eq."enabled")then !YuP[2021-03-08] extended size of rhspar
        allocate(rhspar(0:2*ls+nbanda+1,max(jx,iymax),2),STAT=istat) 
        !         maybe 0:2*ls enough?
        !YuP[2021-03-12] also changed jx--> max(jx,iymax)
      else ! no parallel transport
        allocate(rhspar(0:ls+nbanda+1,jx,2),STAT=istat)  !lsa-->ls
      endif
      call bcast(rhspar,zero,SIZE(rhspar))
      allocate(bndmats(max(jx,iymax),0:ls+1,nbanda,2),STAT=istat) !lsa-->ls
      !YuP[2021-03-12] also changed jx--> max(jx,iymax)
      call bcast(bndmats,zero,SIZE(bndmats))
      
      
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'wpalloc:  Leaving wpalloc'
CMPIINSERT_ENDIF_RANK

      return
      end
