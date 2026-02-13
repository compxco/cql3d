c******************************************************************
      subroutine diagescl(k)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c...  
cmnt  This routine scales the electron distribution function to maintain
cmnt  charge neutrality.
cmnt  It is used iff (colmodl.eq.4 .and. k.eq.kelecg)
c...  
      include 'param.h'
      include 'comm.h'

      data naccel/100/
      if (k.ne.kelecg .or. colmodl.ne.4) return

c...  
cmnt  Compute the new density..
c...  

      vnorm2=vnorm*vnorm
      if (n.ge.naccel) then
        call dcopy(iyjx2,fxsp(0,0,kelecg,l_),1,temp2(0,0),1)
      else
        call dcopy(iyjx2,f(0,0,kelecg,l_),1,temp2(0,0),1)
      endif
      call bcast(tam4,zero,jx)
      do 20 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
        do 10 j=1,jx
          tam4(j)=tam4(j)+vptb(i,lr_)*temp2(i,j)*cynt2(i,l_)
 10     continue
 20   continue
      xlndneg=0.
      do 30 j=1,jx
        xlndneg=xlndneg+tam4(j)*cint2(j)
 30   continue
      call bcast(tam4,zero,jx)
      do 50 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
        do 40 j=1,jx
          tam4(j)=tam4(j)+x(j)*coss(i,l_)*temp2(i,j)*cynt2(i,l_)
 40     continue
 50   continue
      svll=0.
      do 60 j=1,jx
        svll=svll+tam4(j)*cint2(j)
 60   continue
      c0t=sqrt(1.-1./psimx(lr_))
      pfrac=1./(1.-c0t**3)
      xllaveg=svll*zmaxpsi(lr_)*pfrac/xlndneg
      if (cqlpmod .ne. "enabled") then
        thta=fmass(ngen)*clite2/(temp(ngen,lr_)*ergtkev)
        ovthesq=fmass(ngen)/(temp(ngen,lr_)*ergtkev)
      else ! (cqlpmod.eq."enabled")
        thta=fmass(ngen)*clite2/(temppar(ngen,ls_)*ergtkev)
        ovthesq=fmass(ngen)/(temppar(ngen,ls_)*ergtkev)
      endif
      do 90 j=1,jx
        fmxwl=exp(-gamm1(j)*thta)
        if (relativ.eq."disabled") then
          do 70 i=1,itl
            drift=ovthesq*xllaveg*x(j)*coss(i,l_)*vnorm2
            f(i,j,ngen,l_)=fmxwl*(1.+drift)
            f(iy_(l_)-i+1,j,ngen,l_)=fmxwl*(1.-drift)
 70       continue
          do 75 i=itl+1,itu-1
            f(i,j,ngen,l_)=fmxwl
 75       continue
        else
          do 80 i=1,itl
            drift=2.*thta*(sqrt(1.+cnorm2i*xllaveg*x(j)*coss(i,l_))-1.)
            f(i,j,ngen,l_)=fmxwl*(1.+drift)
            f(iy_(l_)-i+1,j,ngen,l_)=fmxwl*(1.-drift) !YuP[2021-03-11] iy-->iy_(l_)
 80       continue
          do 85 i=itl+1,itu-1
            f(i,j,ngen,l_)=fmxwl
 85       continue
        endif
 90   continue
      call dcopy(iyjx2,f(0,0,ngen,l_),1,temp2(0,0),1)
      call bcast(tam4,zero,jx)
      do 110 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
        do 100 j=1,jx
          tam4(j)=tam4(j)+vptb(i,lr_)*temp2(i,j)*cynt2(i,l_)
 100    continue
 110  continue
      hn=0.
      do 120 j=1,jx
        hn=hn+tam4(j)*cint2(j)
 120  continue
      dratio=1.
      denscl=dratio*xlndneg/hn
      call dscal(iyjx2,denscl,f(0,0,ngen,l_),1)
      return
      end
