c
c
      subroutine diagxswx(k)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     routine computes density gain due to partlcle sources and losses
c     during the during velocity sweep in exsweep.
c     gains are positive - losses are negative.
c     1- density gained due to setting negative values of
c     distribution function to 0 during velocity sweep.
c     3- particle source contribution.
c     4- particle loss due to flux at high velocity end.
c     5- bad orbits (part of krook operator)
c     6- toroidal losses(part of krook operator)
c     Further - the normalized velocity flux at mesh point x(j+.5) is pu
c     into vflux(j,k,l_).
c..................................................................

      include 'param.h'
      include 'comm.h'

      include 'advnce.h'

c..................................................................
c     sgain(i,k) will contain the local gain in density due to
c     process i for species k during this time step.
c..................................................................

      sgain(1,k)=0.
      sgain(2,k)=0.
      sgain(3,k)=0.
      sgain(4,k)=0.
      sgain(5,k)=0.
      sgain(6,k)=0.
      sgain(7,k)=0.
      sgain(8,k)=0.
      call bcast(tam5,zero,jx)
      call bcast(tam6,zero,jx)
      call bcast(tam7,zero,jx)
      call bcast(vflux(1,k,l_),zero,jx)

c..................................................................
c     Collect contributions from various pieces of the Krook operator.
c..................................................................

      do 220 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
        call bcast(tam8,zero,jx)
        do 210 j=1,jx
          tam5(j)=tam5(j)+vptb(i,lr_)*dtr*gon(i,j)*
     1      temp2(i,j)*cynt2(i,l_)
          tam6(j)=tam6(j)-vptb(i,lr_)*dtr/taulos(i,j,indxlr_)
     1      *temp2(i,j)*cynt2(i,l_)
          tam7(j)=tam7(j)-vptb(i,lr_)*dtr*tam8(j)
     1      *temp2(i,j)*cynt2(i,l_)
 210    continue
 220  continue

c..................................................................
c     integrate over velocity.
c..................................................................

      do 230 j=1,jx
        sgain(5,k)=sgain(5,k)+tam5(j)*cint2(j)*one_
        sgain(6,k)=sgain(6,k)+tam6(j)*cint2(j)*one_
        sgain(7,k)=sgain(7,k)+tam7(j)*cint2(j)*one_
 230  continue

c..................................................................
c     Determine the fraction of particles leaving a sphere of
c     radius x(j) in a tauee(lr_) time. Also determine the density loss
c     due to particles leaving the domain at the high velocity end.
c..................................................................

      do 310 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
        sgain(4,k)=sgain(4,k)+one_*gfu(i,jx,k)*cynt2(i,l_)*dtr
        do 311 j=1,jx
          vflux(j,k,l_)=vflux(j,k,l_)+one_*gfu(i,j,k)*cynt2(i,l_)
 311    continue
 310  continue
 
      if(cqlpmod.ne."enabled")then !YuP[2021-03-30] CQL3D/CQLP treatment
        do j=1,jx
        vflux(j,k,l_)=vflux(j,k,l_)/xlndn(k,lr_)*tauee(lr_)
        enddo
      else !cqlpmod.eq."enabled"
        do j=1,jx
        vflux(j,k,l_)=vflux(j,k,l_)/xlndn(k,lr_)*tauee(ls_)
        enddo
      endif

c..................................................................
c     Particle source term.
c..................................................................

      sgain(3,k)=xlncur(k,l_)*dtr*0.5 !YuP[2022-02-11] now l_ (was lr_)

c..................................................................
c     if ineg  .eq. "enabled" set negative values of f to zero.
c..................................................................

      if (n .eq. 0 .or. n/nchec*nchec .ne. n)  go to 90

c..................................................................
c     determine energy transfer diagnostics
c..................................................................

      entr(k,4,l_)=0.
      do 120 lefct=-1,8
        call diagentr(lefct,k)
 120  continue
      call diagentr(11,k)
      call diagentr(12,k)
      call coefstup(k)
      call coefmidv(da,1)
      call coefmidv(db,2)
      call coefmidv(dc,3)
      call coefmidt(dd,1)
      call coefmidt(de,2)
      call coefmidt(df,3)
 90   continue
      call dcopy(iyjx2,temp2(0,0),1,temp1(0,0),1)
      if (ineg .eq. "disabled") go to 400

      do 410 j=1,jx
        do 411 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
          if(temp2(i,j) .lt. 0.) then
            temp1(i,j)=zero
            temp4(i,j)=-temp2(i,j)
          else ! temp2(i,j) .ge. 0.
            temp1(i,j)=temp2(i,j)
            temp4(i,j)=zero
          endif
 411    continue
 410  continue

c..................................................................
c     routine diagdens will compute density gained by setting negative
c     values of distribution to 0.
c..................................................................

      call diagdens(xline,xmidp,eline)
      engain(k)=eline*one_
      sgain(1,k)=xline*one_
 400  continue

c..................................................................
c     if debugging,compute density at end of velocity split
c..................................................................

      if (iactst .eq. "disabled") go to 500
      call dcopy(iyjx2,temp1(0,0),1,temp4(0,0),1)
      call diagdens(yline,ymidd,eline)
      yline=yline*one_
 500  continue
      return
      end
