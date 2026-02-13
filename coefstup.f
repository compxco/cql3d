c
c
      subroutine coefstup(k)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     This routine stores all the relevant coefficients into the
c     proper arrays for time advancement
c..................................................................

      include 'param.h'
      include 'comm.h'

      call bcast(da,zero,iyjxp1)
      call bcast(db,zero,iyjxp1)
      call bcast(dc,zero,iyjxp1)
      call bcast(dd,zero,iyp1jx)
      call bcast(de,zero,iyp1jx)
      call bcast(df,zero,iyp1jx)
      call bcast(dbb,zero,iyjxp1)
      call bcast(dff,zero,iyp1jx)

c..................................................................
c     particle source
c..................................................................
      ifag=0
      do 10 is=1,nso
        if (n.ge.nonso(k,is) .and. n.lt.noffso(k,is)) then
          ifag=1
        endif
 10   continue
      if (ifag.eq.0) then
        call bcast(so,zero,iyjx2)
      else
        !call dcopy(iyjx2,source(0,0,k,indxlr_),1,so,1) !before[2022-02-11]
        if (l_ .eq. lmdpln_) then !after[2022-02-11]
         call dcopy(iyjx2,source(0,0,k,l_),1,so,1) !after[2022-02-11]
          !For CQLP, source is set at l_=1 only
        else
         so=0.d0
        endif
        !for CQL3D, source() is copied to so() at every l_=lr_,
        !for CQLP, source() is copied to so() only at l_=lmdpln_=1
        !  (1st point along field line), and set to 0.0 at other l_
      endif
      xrf=0.

c..................................................................
c     rf coefficients
c..................................................................

      if (n.eq.nonrf(k).and.(nrf.gt.0)) imprf=1
      if (urfmod.eq."enabled" .and. n.eq.nonrf(k)) imprf=1
      if (vlhplse.ne."disabled" .and. (vlhmod.eq."enabled".or.
     +  vlfmod.eq."enabled")) then
        if (vlhplse.eq."tauee") then
          timespn=tauee(lr_)*(vlhpon+vlhpoff)
          timeon=tauee(lr_)*vlhpon
        else
          timespn=vlhpon+vlhpoff
          timeon=vlhpon
        endif
        iperiod=timet/timespn
        tim1=iperiod*timespn
        timedf=timet-tim1
        if (timedf.le.timeon) then
          call coefrfad(k,xrf)
        endif
      else if (n .ge. nonrf(k) .and. n .lt. noffrf(k)) then
        call coefrfad(k,xrf)
      endif

c..................................................................
c     parallel electric field
c..................................................................

      impelec=0
      if (elecfld(lr_) .ne. 0. .and.(nonel.le.n) .and.(noffel.gt.n))then
        call coefefad(k)
      endif
      if (n.eq.nonel .or. n.eq.noffel) impelec=1

c..................................................................
c     Add in the krook operator loss term..
c..................................................................

      call coefload(k)

c..................................................................
c     Add in phenomenological energy loss operator
c..................................................................

      call coefegad(k)

c..................................................................
c     Add in collisional coefficients
c..................................................................

      call coeffpad(k)

c..................................................................
c     add in synchrotron radiation losses
c..................................................................

      call coefsyad(k)

c.......................................................................
c     Add contribution of theta operator due to (mu*grad_parallel B) force
c.......................................................................

c%OS  if (cqlpmod.eq."enabled" .and. transp.eq."enabled" .and.
c%OS  +                                             n.ge.nontran+1)  then
c%OS  call wpcthta
c%OS  else
      call bcast(cthta,zero,iymax*jx) !YuP[2021-03-11] iy-->iymax
c%OS  endif

      return
      end
