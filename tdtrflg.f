c
c
      subroutine tdtrflg ! CQL3D only (not CQLP)
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     This routine sets some mesh flag arrays to facilitate
c     coding of the transport routine
c..............................................................

      include 'param.h'
      include 'comm.h'

      if (transp.eq."disabled") return

c..............................................................
c     define lpt(i); the transport mesh point l such that
c     the particle characterized by transport pitch angle index i
c     is passing at l-1 and trapped at l. If this never happens
c     set equal to lrors.
c..............................................................

      do 20 i=1,iytr(lrors)/2
         lpt(i)=lrors
         if (idx(i,lrors).lt.itl_(lrors)) go to 20
         do 30 l=lrors-1,2,-1
c         write(*,*)'tdtrflg:l,i,idx(i,l),itl_(l),idx(i,l-1),itl_(l-1)'
c     +           ,l,i,idx(i,l),itl_(l),idx(i,l-1),itl_(l-1)
         if (ipacktp.eq.3) then !Older system for soln_method=direct,tfac=1.
            if (idx(i,l).gt.itl_(l) .and. idx(i,l-1).lt. itl_(l-1)) then
               if (idx(i,l-1).ne.0) lpt(i)=l
c               write(*,*)'lpt(i),i,l',lpt(i),i,l
            endif
         elseif (ipacktp.eq.0) then
            if (idx(i,l).ge.itl_(l) .and. idx(i,l-1).lt. itl_(l-1)) then
               if (idx(i,l-1).ne.0) lpt(i)=l
c               write(*,*)'lpt(i),i,l',lpt(i),i,l
            endif
         endif
 30   continue
 20   continue
      
c..............................................................
c     Define l_lower(i); the lowest index l for which pitch angle
c     index i is defined on the local transport pitch angle mesh.
c..............................................................

      call ibcast(l_lower,1,iytr(lrors))
      do 50 l=2,lrors
        if (iytr(l-1).lt. iytr(l)) then
          do 80 i=iytr(l-1)/2+1,iytr(lrors)-iytr(l-1)/2
            l_lower(i)=l
            if (lpt(i).eq.l_lower(i)) lpt(i)=lrors
 80       continue
        endif
 50   continue
      do 100 i=1,iytr(lrors)/2
        ii=iytr(lrors)+1-i
        lpt(ii)=lpt(i)
 100  continue

      if (ioutput(1).ge.2) then !YuP[2020] diagnostic printout
      write(*,*)'tdtrflg:  iytr(1:lrors)=',iytr(1:lrors)
      write(*,*)'tdtrflg:  l_lower(1:iytr)=',l_lower(1:iytr(lrors))
      write(*,*)'tdtrflg:  lpt(1:iytr)=',lpt(1:iytr(lrors))
      endif
c.......................................................................
c     initialize coefficients for v to r velocity meshes transformation
c.......................................................................

      call bcast(f_vtor,one,jx*ngen*lrors*18)

      return
      end
