c
c
      subroutine coefrfad(k,xrf)
      implicit integer (i-n), real*8 (a-h,o-z)

c...............................................................
c     This routine adds the RF contributions to the Fokker-Planck
c     coefficients for species k.
c...............................................................

      save
      include 'param.h'

c..................................................................
c     rf module contributions first.
c..................................................................

      include 'comm.h'

      call bcast(dbb(1,0),zero,iyjxp1)
      call bcast(dff(0,1),zero,iyp1jx)

      if (cqlpmod.ne."enabled") then
      if (vlhmod.eq."enabled" .or. vlfmod.eq."enabled") then
c     vlhmod and vlfmod are only set up for nrf=1
      if (nrf.gt.1) then
         STOP 'coefrfad/vlh/vlf: STOP, Problem with nrf. Should be 1'
      endif
      do 10 kk=1,nrf
        if (k .eq. kk) then
          do krf=1,mrfn
          do 2 j=1,jx
            do 3 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
              dbb(i,j)=dbb(i,j)+cqlb(i,j,indxlr_,krf)
              dff(i,j)=dff(i,j)+cqlf(i,j,indxlr_,krf)
              db(i,j)=  db(i,j)+cqlb(i,j,indxlr_,krf)
              dc(i,j)=  dc(i,j)+cqlc(i,j,indxlr_,krf)
              de(i,j)=  de(i,j)+cqle(i,j,indxlr_,krf)
              df(i,j)=  df(i,j)+cqlf(i,j,indxlr_,krf)
 3          continue
 2        continue
          enddo  !On krf
          xrf=1.
        endif
 10   continue
      endif   ! On vlhmod.or.vlfmod enabled

      elseif (cqlpmod.eq."enabled" .and. vlfmod.eq."enabled") then

      do 20 kk=1,nrf
ccc YuP-101220: Commenting next line, because wcqlb-wcqlf 
ccc are defined for vlfmod="enabled" (but not for vlhmod="enabled")
ccc      if (vlhmod.eq."enabled" .or. vlfmod.eq."enabled") then
c     vlhmod and vlfmod are only set up for nrf=1
        if (nrf.gt.1) then
           STOP 'coefrfad/vlh/vlf: STOP, Problem with nrf. Should be 1'
        endif
        if (k .eq. kk) then
          do krf=1,mrfn
          do 12 j=1,jx
            do 13 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
              dbb(i,j)=dbb(i,j)+wcqlb(i,j,krf,l_)
              dff(i,j)=dff(i,j)+wcqlf(i,j,krf,l_)
              db(i,j)=db(i,j)+wcqlb(i,j,krf,l_)
              dc(i,j)=dc(i,j)+scatfrac*wcqlc(i,j,krf,l_)
              de(i,j)=de(i,j)+scatfrac*wcqle(i,j,krf,l_)
              df(i,j)=df(i,j)+scatfrac*wcqlf(i,j,krf,l_)
 13         continue
 12       continue
          enddo  !On krf
          xrf=1.
        endif
ccc      endif   ! On vlhmod,vlfmod enabled
 20   continue

      endif   ! On cqlpmod,vlfmod enabled

c..................................................................
c     urf module contribution next 
c..................................................................

c**bh930729if(urfmod.eq."enabled".and.k.eq.kelecg) then
      if (urfmod.eq."enabled") then
        l=indxlr_
        do 30 krf=1,mrfn
c         Check if this rf mode is to applied to the current species k:
          if (nrfspecies(krfn(krf)) .eq. k) then
          do 22 j=1,jx
            do 23 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
             delb0= urfb(i,j,l,krf)
             if(delb0.ne.0.d0)then 
                !YuP[03/18/2015] urfe,urff are expr.through urfb,urfc
                !urfR23,urfR33 are expr.through urfb,urfc,urfR13
                !It can be shown that if the local delB=0, then also
                ! delC=0, delE=0, delF=0, and then all urfR** coeffs are zero.
                delc0=  urfc(i,j,l,krf)
                dele0=  delc0*sinn(i,l_)   ! urfe ! valid for ZOW&FOW
                delf0= (delc0*dele0)/delb0 ! urff ! valid for ZOW&FOW
                dbb(i,j)= dbb(i,j)+delb0
                dff(i,j)= dff(i,j)+delf0
                db(i,j)=  db(i,j) +delb0
                dc(i,j)=  dc(i,j) +scatfrac*delc0
                de(i,j)=  de(i,j) +scatfrac*dele0
                df(i,j)=  df(i,j) +scatfrac*delf0
             endif
 23        continue
 22     continue
          endif  ! on nrfspecies
 30     continue ! krf
      xrf=1.
      endif  ! on urfmod

c.......................................................................

c..................................................................
c     rdc module contribution next 
c..................................................................

      if (rdcmod.ne."disabled") then
c         rdcbmax=0.d0     
c         do j=1,jx
c            do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
c               rdcbmax=max(rdcbmax,rdcb(i,j,indxlr_))
c            enddo
c         enddo
!         if (ioutput(1).ge.2) then
!         !write(*,*)'coefrfad: Before rdcmod add, rdcbmax ,indxlr_=',
c     &        rdcbmax,indxlr_
!         endif
         
c$$$         do j=1,jx
c$$$            do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
c$$$              dbb(i,j)=dbb(i,j)+rdcb(i,j,indxlr_)
c$$$              dff(i,j)=dff(i,j)+rdcf(i,j,indxlr_)
c$$$              db(i,j)=db(i,j)+rdcb(i,j,indxlr_)
c$$$              dc(i,j)=dc(i,j)+scatfrac*rdcc(i,j,indxlr_)
c$$$              de(i,j)=de(i,j)+scatfrac*rdce(i,j,indxlr_)
c$$$              df(i,j)=df(i,j)+scatfrac*rdcf(i,j,indxlr_)
c$$$           enddo
c$$$        enddo
c$$$        xrf=1.

        l=indxlr_
        do 40 krf=1,nrdc
c         Check if this rf mode is to applied to the current species k:
          if (nrdcspecies(krf) .eq. k) then
          do 42 j=1,jx
            do 43 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
             delb0= rdcb(i,j,l,krf)
             if(delb0.ne.0.d0)then 
c$$$                !YuP[03/18/2015] urfe,urff are expr.through urfb,urfc
c$$$                !urfR23,urfR33 are expr.through urfb,urfc,urfR13
c$$$                !It can be shown that if the local delB=0, then also
c$$$                ! delC=0, delE=0, delF=0, and then all urfR** coeffs are zero.
                delc0=  rdcc(i,j,l,krf)
c$$$            BH180511: Could try following, as an alternative to direct calc.
c$$$                dele0=  delc0*sinn(i,l_)   ! urfe ! valid for ZOW&FOW
c$$$                delf0= (delc0*dele0)/delb0 ! urff ! valid for ZOW&FOW
                dele0=rdce(i,j,l,krf)
                delf0=rdcf(i,j,l,krf)
                dbb(i,j)= dbb(i,j)+delb0
                dff(i,j)= dff(i,j)+delf0
                db(i,j)=  db(i,j) +delb0
                dc(i,j)=  dc(i,j) +scatfrac*delc0
                de(i,j)=  de(i,j) +scatfrac*dele0
                df(i,j)=  df(i,j) +scatfrac*delf0
             endif
 43        continue
 42     continue
          endif  ! on nrfspecies
 40     continue ! krf
      xrf=1.

      endif  !On rdcmod

      return
      end
