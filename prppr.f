c
c
      subroutine prppr(target,action,xll,xlu,xpl,xpu)
      implicit integer (i-n), real*8 (a-h,o-z)
      character*(*) target,action
      save

c...............................................................
cmnt  This routine takes data stored in temp3 in (y,x) coordinates
cmnt  and linearly interpolates to a quantity fpn in xpar,xperp
cmnt  coordinates.
c...............................................................

      include 'param.h'
      include 'comm.h'

      logical trnsfm
      ipxjpx=jpxy*ipxy
      call bcast(fpn(1,1),zero,ipxjpx)
      trnsfm=(target.eq."velocity".and. relativ .ne. "disabled")

c     ipxy refers to xperp,
c     jpxy refers to xpar (make it odd).

cbh960801      if (mod(jpxy,2).eq.0) jpxy=jpxy-1
      ipxym=ipxy-1
      jpxym=jpxy-1
      icase=1
      if (target.eq."mainmesh") then
        jminn=1
        jmaxxm1=jx-1
        iminn=1
        imaxxm1=iy_(l_)-1   !YuP[2021-03-11] iy-->iy_(l_)
        goto 30
      elseif (target.eq."ionmesh") then
        jminn=1
        jmaxxm1=jlwr
        iminn=1
        imaxxm1=iy_(l_)-1   !YuP[2021-03-11] iy-->iy_(l_)
        goto 30
      endif
      xll2=xll**2
      xlu2=xlu**2
      xpl2=xpl**2
      xpu2=xpu**2
      xmgll=sqrt(xll2+xpl2)
      xmglu=sqrt(xll2+xpu2)
      xmgul=sqrt(xlu2+xpl2)
      xmguu=sqrt(xlu2+xpu2)
      if (trnsfm) then
        xnm=1./(cnorm*gamma(jx))
        if (xmgll.ge.xnm) then
          icase=5
          call diagwrng(100)
        elseif (xmgul.ge.xnm) then
          icase=4
        elseif (xmglu.ge.xnm) then
          icase=3
        elseif (xmguu.ge.xnm) then
          icase=2
        else
          icase=1
        endif
        xmgrll=xmgll*cnorm*sqrt(1./(1.-xmgll))
        xmgrlu=xmglu*cnorm*sqrt(1./(1.-xmglu))
        xmgrul=xmgul*cnorm*sqrt(1./(1.-xmgul))
        xmgruu=xmguu*cnorm*sqrt(1./(1.-xmguu))
c990131        xminn=amin1(xmgrll,xmgrlu,xmgrul,xmgruu)
c990131        xmaxx=amax1(xmgrll,xmgrlu,xmgrul,xmgruu)
        xminn=min(xmgrll,xmgrlu,xmgrul,xmgruu)
        xmaxx=max(xmgrll,xmgrlu,xmgrul,xmgruu)
        if (icase.gt.1) xmaxx=1.
      else
c990131        xminn=amin1(xmgll,xmglu,xmgul,xmguu)
c990131        xmaxx=amax1(xmgll,xmglu,xmgul,xmguu)
        xminn=min(xmgll,xmglu,xmgul,xmguu)
        xmaxx=max(xmgll,xmglu,xmgul,xmguu)
      endif
      if (icase.eq.5) return
      cossll=xll/xmgll
      cosslu=xll/xmglu
      cossul=xlu/xmgul
      cossuu=xlu/xmguu
      abit=1.e-12
      onemabit=one-abit
      if (abs(cossll).ge.onemabit) cossll=sign(onemabit,cossll)
      if (abs(cosslu).ge.onemabit) cosslu=sign(onemabit,cosslu)
      if (abs(cossul).ge.onemabit) cossul=sign(onemabit,cossul)
      if (abs(cossuu).ge.onemabit) cossuu=sign(onemabit,cossuu)
      thll=acos(cossll)
      thlu=acos(cosslu)
      thul=acos(cossul)
      thuu=acos(cossuu)
c990131      thminn=amin1(thll,thlu,thul,thuu)
c990131      thmaxx=amax1(thll,thlu,thul,thuu)
      thminn=min(thll,thlu,thul,thuu)
      thmaxx=max(thll,thlu,thul,thuu)
      iminn=luf(thminn,y,iy_(l_))-1  !YuP[2021-03-11] iy-->iy_(l_)
      jminn=luf(xminn,x,jx)-1
      if (iminn.eq.0) iminn=1
      if (jminn.eq.0) jminn=1
      imaxxm1=luf(thmaxx,y,iy_(l_))-1 !YuP[2021-03-11] iy-->iy_(l_)
      if (icase.gt.1) then
        jmaxxm1=jx-1
      else
        jmaxxm1=luf(xmaxx,x,jx)-1
      endif
 30   continue
      xperp(1)=xpl
      xperp(ipxy)=xpu
      xpar(1)=xll
      xpar(jpxy)=xlu
      dxpp=(xpu-xpl)/float(ipxym)
      dxll=(xlu-xll)/float(jpxym)
      dvll=dxll*vnorm
      dvlli=1./dvll

      do 40 ip=2,ipxy
        xperp(ip)=xperp(ip-1)+dxpp
 40   continue

      if (xlu.eq.(-xll)) then
c     Keeping xpar exactly antisymmetric, attempting to avoid minor
c     difficulties in unsymmetric interpolation near the v-grid edge..
         do 50 jp=2,jpxym/2
            xpar(jp)=xpar(jp-1)+dxll
 50      continue
         jp0=jpxym/2+1
         xpar(jp0)=0.0
         do 51 jp=1,jpxym/2
            xpar(jp0+jp)=-xpar(jp0-jp)
 51      continue
      else
         do 52 jp=2,jpxym
            xpar(jp)=xpar(jp-1)+dxll
 52      continue
      endif

c      do jp=1,jpxy
c      write(*,*)'prppr: jp, xpar(jp)=',jp, xpar(jp)
c      enddo
c      do ip=1,ipxy
c      write(*,*)'prppr: ip, xperp(ip)=',ip, xperp(ip)
c      enddo

      do 130 ip=1,ipxy-1
        xperp2=xperp(ip)**2
        do 120 jp=2,jpxy-1
          xmg2=xperp2+xpar(jp)**2+1.e-26
          xmg=sqrt(xmg2)
          xmgr=xmg
          xjac=1.
          if (trnsfm) then
            xmgr=xmg*cnorm/sqrt(1.-xmg2)
            gam=sqrt(1.+xmgr*xmgr)
            xjac=cnorm*gam**5
          endif
          if (xmgr.ge.xmax) goto 100
          cosse=xpar(jp)/xmg
c990131          if (abs(cosse).gt.1.)  cosse=sign(1.,cosse)
          if (abs(cosse).gt.1.)  cosse=sign(one,cosse)
          thet=acos(cosse)
          do 60 i=iminn,imaxxm1
            if (thet.lt. y(i,l_) .or. thet.ge.y(i+1,l_)) goto 60
            goto 70
 60       continue
          i=imaxxm1
 70       continue
          i2=i+1
          do 80 j=jminn,jmaxxm1
            if (xmgr.lt.x(j) .or. xmgr.ge.x(j+1)) goto 80
            goto 90
 80       continue
          j=jmaxxm1
 90       continue
          j2=j+1
          ripos=(thet-y(i,l_))*eyp5(i,l_)
          rjpos=(xmgr-x(j))*exp5(j)
          ripos1=1.-ripos
          rjpos1=1.-rjpos
          w00=ripos1*rjpos1
          w01=ripos1*rjpos
          w10=ripos*rjpos1
          w11=ripos*rjpos
          fpnt=w00*temp3(i,j)+w01*temp3(i,j2)+w10*temp3(i2,j)+w11
     1      *temp3(i2,j2)
          fpn(jp,ip)=fpnt*xjac
          goto 120
 100      continue
          fpn(jp,ip)=0.
 120    continue ! jp=2,jpxy-1
 130  continue ! ip=1,ipxy-1

c      do jp=1,jpxy
c      do ip=1,ipxy
c         write(*,*)'prppr: jp,ip,fpn(jp,ip):',jp,ip,fpn(jp,ip)
c      enddo
c      enddo
c-obsolete      wmax=0.
c-obsoletec     (The first half-interval is integrated with average(fpn)*x_perp.
c-obsoletec      Thereafter, trapazoidal integration).
c-obsolete      do 150 jp=1,jpxy
c-obsolete        s=(3.*fpn(jp,1)+fpn(jp,2))*dxpp**2/32.
c-obsolete        do 140 ip=2,ipxym
c-obsolete          s=s+fpn(jp,ip)*xperp(ip)*dxpp
c-obsolete 140    continue
c-obsolete        s=s+fpn(jp,ipxy)*xperp(ipxy)*dxpp*.5
c-obsolete        fll(jp)=s*2.*pi
c-obsolete 150  continue
c-obsoletec     if (target.eq."mainmesh") then
c-obsoletec     do 155 jp=2,jpxy
c-obsoletec     dfvlle(jp)=(fpn(jp)-fpn(jp-1))*dvlli
c-obsoletec     155   continue
c-obsoletec     elseif (target.eq."ionmesh") then
c-obsoletec     do 156 jp=2,jpxy
c-obsoletec     dfvlli(jp)=(fpn(jp)-fpn(jp-1))*dvlli
c-obsoletec     156   continue
c-obsoletec     endif
c-obsolete      if (action.eq."norm") then
c-obsolete        call aminmx(fll(1),1,jpxy,1,fmin,fmax,kmin,kmax)
c-obsolete        do 160 jp=1,jpxy
c-obsolete          fll(jp)=fll(jp)/fmax
c-obsolete 160    continue
c-obsolete      endif
      return
      end
