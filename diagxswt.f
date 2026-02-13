c
c
      subroutine diagxswt(k)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     Routine computes density gain due to particle sources and losses
c     during the theta sweep in exsweep.
c     gains are positive - losses are negative.
c     2- density gained due to setting negative values of
c     distribution function to 0 during theta sweep.
c     3- particle source contribution.
c..................................................................

      include 'param.h'
      include 'comm.h'


c..................................................................
c     Add in the source contribution.
c..................................................................

      sgain(3,k)=xlncur(k,l_)*.5*dtr+sgain(3,k) !YuP[2022-02-11] now l_ (was lr_)
      call dcopy(iyjx2,temp2(0,0),1,temp1(0,0),1)
      s=0.
      if (ineg .eq. "disabled") go to 350

c..................................................................
c     if desired set negative values of distribution function = 0.
c..................................................................

      do 300 j=1,jx
        do 301 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
          if(temp2(i,j) .lt. 0.) then
            temp1(i,j)=zero
            temp4(i,j)=-temp2(i,j)
          else ! temp2(i,j) .ge. 0.
            temp1(i,j)=temp2(i,j)
            temp4(i,j)=zero
          endif
 301    continue
 300  continue
      call diagdens(xline,xmidp,eline)
      engain(k)=engain(k)+eline*one_
      sgain(2,k)=xline*one_
 350  continue
      call dcopy(iyjx2,temp1(0,0),1,temp4(0,0),1)

c..................................................................
c     Compute power from df/dt and from setting neg f to zero.
c..................................................................

      if (n .gt. 0 .and. n/nchec*nchec .eq. n) then
        call diagentr(9,k)
        call diagentr(10,k)
        call dcopy(iyjx2,temp4(0,0),1,temp1(0,0),1)
      endif
      if (iactst.eq."disabled") go to 500

c..................................................................
c     if debugging, compute density at end of theta split
c..................................................................

      call diagdens(yline,ymidd,eline)
      yline=yline*one_
 500  continue
      return
      end
