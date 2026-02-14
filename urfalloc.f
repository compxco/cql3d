c
c
      subroutine urfalloc
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      complex*16 czero
cdir$ nobounds

c..................................................................
c     One measure of whether the following allocations are
c     successful is whether code is entered, then exited without
c     a code stopping fault (e.g. Segmentation fault). See istat_tot
c     comments below.
c..................................................................

CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'urfalloc:  Entering urfalloc'
CMPIINSERT_ENDIF_RANK

c.......................................................................
c     Allocate allocatable arrays for rf modules
c.......................................................................

      lniyjx=iy*jx*lrz*mrfn
      lniylz=iy*lz*lrzmax
      lnyxp2=iyjx2*ngen*lrors
      lnj=jx
      lni=iy
      lnjj=jjx
      nrnrm=nrayelts*nrayn*mrfn
      nrnr2=nrayelts*nrayn*2

c     ibytes16 is number of 16-bit words per integer word.
c     ipack16_ is number of integer words required to store 1 set
c     of ray data (*jjx) in the ifct1_,ifct2_ 16-bit-word arrays.
      !ipack16= (jjx/ibytes16)*nrayelts*nrayn +1 
      !YuP-101207: no need to multiply by mrfn, 
      !  because ifct1,2_(*,*,mrfn) includes mrfn
      ipack16_=(jjx/ibytes16)*nrayelts +1 ![2026-01-24]

      nrm=nrayn*mrfn
      
      !if(urfb_version.eq.1)then ! 2 is the new version developed by YuP
        !subr. pack16() is not used by urfb_version.eq.2
        !so - no need to define pack16 and to print this out.
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'urfalloc: ipack16_,mrfn=',ipack16_,mrfn
CMPIINSERT_ENDIF_RANK
      !endif

c...................................................................
c     Note three complex*16 arrays, cosz1, sinz1, sinz2 
c     must be 2x longer than real*8 arrays.
c     Same for cwexde,cweyde and cwezde.
c     lnurfdum is length of urfdum measured in 8 byte words.
c...................................................................

      lnurfdum=4*lniyjx+lniylz+lnyxp2+8*lnj+(3*2+4)*lni+4*lnjj
     1  +1*nrnr2+(9+3*2+16)*nrnrm+2*ipack16+2*ipack16+16*nrm

      czero = (0.0,0.0)


c..................................................................
c     Check on the allocate by adding up istat.
c     If not zero, problem has occurred.
cBH100830:  But, this didn't work with gfortran when nrayn parameter
cBH100830:  is  made excessively large.  Code gets Segmentation fault
cBH100830:  [presumably in urfalloc.f].  But this indicates that
cBH100830:  checking on istat is not sufficient to catch a problem
cBH100830:  of using too much memory.
c-YuP 101121: Added if(istat.eq.0) in front of call bcast() or ibcast()
c-YuP 101121: If istat=41 (not enough memory), cannot call bcast()
c-YuP 101121: because array is not allocated => results in Seg.Fault.
c..................................................................
      istat_tot=0

      allocate(urfb(iy,jx,lrz,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(urfb,zero,SIZE(urfb))
      allocate(urfc(iy,jx,lrz,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(urfc,zero,SIZE(urfc))
      !allocate(urfe(iy,jx,lrz,mrfn),STAT=istat)
      !istat_tot=istat_tot+istat
      !allocate(urff(iy,jx,lrz,mrfn),STAT=istat)
      !istat_tot=istat_tot+istat
      !if(istat.eq.0) call bcast(urff,zero,SIZE(urff))
      !WRITE(*,*)'urfalloc: urff allocated'
!YuP[03/18/2015] urfe,urff are expressed through urfb,urfc
!No need to save them.
      
      allocate(cosmz(iy,lz,lrzmax),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(cosmz,zero,SIZE(cosmz))
      allocate(g_(0:iyp1,0:jxp1,ngen,lrors),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(g_,zero,SIZE(g_))
      allocate(alfag(jx),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(alfag,zero,SIZE(alfag))
      allocate(argmnt(jx),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(argmnt,zero,SIZE(argmnt))
      allocate(ilim1d(jx),STAT=istat)
      if(istat.eq.0) call ibcast(ilim1d,0,SIZE(ilim1d)) 
      istat_tot=istat_tot+istat
      allocate(ilim2d(jx),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ibcast(ilim2d,0,SIZE(ilim2d)) 
      allocate(ilim1dd(jx),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ibcast(ilim1dd,0,SIZE(ilim1dd)) 
      allocate(ilim2dd(jx),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ibcast(ilim2dd,0,SIZE(ilim2dd))
      allocate(sx(jx),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(sx,zero,SIZE(sx))
      allocate(xmdx(jx),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(xmdx,zero,SIZE(xmdx))
      allocate(cosz1(iy),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ccast(cosz1,czero,SIZE(cosz1))
      allocate(sinz1(iy),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ccast(sinz1,czero,SIZE(sinz1))
      allocate(sinz2(iy),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ccast(sinz2,czero,SIZE(sinz2))
      allocate(thtf1(iy),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(thtf1,zero,SIZE(thtf1))
      allocate(thtf2(iy),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(thtf2,zero,SIZE(thtf2))
      allocate(alfi(iy),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(alfi,zero,SIZE(alfi))
      allocate(alfa(iy),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(alfa,zero,SIZE(alfa))
      allocate(ilim1(jjx),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ibcast(ilim1,0,SIZE(ilim1))
      allocate(ilim2(jjx),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ibcast(ilim2,0,SIZE(ilim2))
      allocate(ifct1(jjx),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ibcast(ifct1,0,SIZE(ifct1))
      allocate(ifct2(jjx),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ibcast(ifct2,0,SIZE(ifct2))
      allocate(urftmp(nrayelts*nrayn*5),STAT=istat)
      !urftmp is also used for MPI, to recv 5 arrays nrayelts*nrayn size
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(urftmp,zero,SIZE(urftmp))
      allocate(urfpwr(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(urfpwr,zero,SIZE(urfpwr))
      allocate(urfpwrc(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(urfpwrc,zero,SIZE(urfpwrc))
      allocate(urfpwrl(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(urfpwrl,zero,SIZE(urfpwrl))
      allocate(jminray(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'urfalloc: urfpwr* allocated'
CMPIINSERT_ENDIF_RANK
      
      if(istat.eq.0) call ibcast(jminray,0,SIZE(jminray))
      allocate(jmaxray(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ibcast(jmaxray,0,SIZE(jmaxray))
      allocate(lloc(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ibcast(lloc,0,SIZE(lloc))
      allocate(llray(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ibcast(llray,0,SIZE(llray))
      allocate(psiloc(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(psiloc,zero,SIZE(psiloc))
      allocate(scalurf(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(scalurf,zero,SIZE(scalurf))
      allocate(cwexde(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ccast(cwexde,czero,SIZE(cwexde))
      allocate(cweyde(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ccast(cweyde,czero,SIZE(cweyde))
      allocate(cwezde(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call ccast(cwezde,czero,SIZE(cwezde))
      allocate(delpwr(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(delpwr,zero,SIZE(delpwr))
      allocate(fluxn(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(fluxn,zero,SIZE(fluxn))
      allocate(seikon(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(seikon,zero,SIZE(seikon))
      allocate(spsi(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(spsi,zero,SIZE(spsi))
      allocate(sdpwr(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(sdpwr,zero,SIZE(sdpwr))
      allocate(sbtot(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(sbtot,zero,SIZE(sbtot))
      allocate(sene(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(sene,zero,SIZE(sene))
      allocate(salphac(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(salphac,zero,SIZE(salphac))
      allocate(salphal(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(salphal,zero,SIZE(salphal))
      allocate(ws(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(ws,zero,SIZE(ws))
      allocate(wr(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(wr,zero,SIZE(wr))
      allocate(wz(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(wz,zero,SIZE(wz))
      allocate(wnpar(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(wnpar,zero,SIZE(wnpar))
      allocate(wdnpar(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(wdnpar,zero,SIZE(wdnpar))
      allocate(wnper(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(wnper,zero,SIZE(wnper))
      allocate(wphi(nrayelts,nrayn,mrfn),STAT=istat)
      istat_tot=istat_tot+istat
      if(istat.eq.0) call bcast(wphi,zero,SIZE(wphi))
                  
      !if(urfb_version.eq.1)then ! 2 is the new version developed by YuP
        ! if 1, it will use the original version
        !-YuP 101121:  These are usually large arrays:
        allocate(ilowp(ipack16_,nrayn,mrfn),STAT=istat) ![2026-01-24]
        !BH,YuP[2020-12-18] Changed pack-->pack16(which uses integer*2),
        !and allocation - accordingly.
        istat_tot=istat_tot+istat
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'urfalloc  ilowp: istat=',istat
CMPIINSERT_ENDIF_RANK
        allocate(iupp(ipack16_,nrayn,mrfn),STAT=istat)
        !BH,YuP[2020-12-18] Changed pack-->pack16(which uses integer*2),
        !and allocation - accordingly.
        istat_tot=istat_tot+istat
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'urfalloc   iupp: istat=',istat
CMPIINSERT_ENDIF_RANK
        allocate(ifct1_(ipack16_,nrayn,mrfn),STAT=istat)
        istat_tot=istat_tot+istat
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'urfalloc  ifct1_: istat=',istat
CMPIINSERT_ENDIF_RANK
        allocate(ifct2_(ipack16_,nrayn,mrfn),STAT=istat)
        istat_tot=istat_tot+istat
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'urfalloc  ifct2_: istat=',istat
CMPIINSERT_ENDIF_RANK
        ! istat=41 means not enough memory for allocation
        call ibcast(ilowp,0,SIZE(ilowp))
        call ibcast(iupp,0,SIZE(iupp))
        call ibcast(ifct1_,0,SIZE(ifct1_)) 
        call ibcast(ifct2_,0,SIZE(ifct2_))
      !endif
           
c     Check that allocations were OK
      if (istat_tot.ne.0) then
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'urfalloc.f:  Problem with allocation'
         WRITE(*,*)'urfalloc.f:  Reduce param.h paramaters?'
         WRITE(*,*)'urfalloc.f:  Stopping'
CMPIINSERT_ENDIF_RANK
         STOP
      endif

c..................................................................
c     Sucessful allocation
c..................................................................
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'urfalloc:  Leaving urfalloc'
CMPIINSERT_ENDIF_RANK

      return
      end
