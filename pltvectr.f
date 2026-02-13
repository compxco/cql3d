c
c
      subroutine pltvectr(xt,yt,xh,yh,jpxy,ipxy,veclen,noplots)
      implicit integer (i-n), real*8 (a-h,o-z)
      character*8 noplots
      save
c
c     Plots vector field in x(horizontal),y(vertical) space.
c     Tail of vector is at xt(1:jpxy,1:ipxy),yt(1:jpxy,1:ipxy).
c     Vectors have x,y components xy,yh. rheads(1:jpxy) is used
c       for calculation of vector lengths (for each y).
c       Vectors are scaled to length veclen.
c
      dimension tam1(2),tam2(2)
      dimension xt(jpxy,ipxy),yt(jpxy,ipxy)
      dimension xh(jpxy,ipxy),yh(jpxy,ipxy)
      dimension rheads(jpxy) !YuP[2020-12-17] Why needed in arg list? Made it local
c     
      REAL*4 RPX1,RPY1,RPX2,RPY2
      REAL*4 ANGLE,BARB,SIZE
      REAL*4 RBOUND
cBH011228 Modifications for plotting with PGPLOT,  011228.

      ep90=1.d+90
      rscale=-ep90
      do 100 i=1,ipxy
         do 90 j=1,jpxy
            rheads(j)=sqrt(xh(j,i)**2+yh(j,i)**2)
 90      continue
         call aminmx(rheads,1,jpxy,1,dummy,rscalexx,kmin,kmax)
         rscale=max(rscale,rscalexx)
 100  continue
      if(rscale.eq.0.) rscale=1.
c     
c     SET ARROW HEAD STYLE:
      ANGLE=90.
      BARB=0.7
      CALL PGSAH(1,ANGLE,BARB)
c     SAVE PGPLOT attributes, and reset character/arrowhead size:
      CALL PGSAVE
      SIZE=10./jpxy
      CALL PGSCH(SIZE)
c      write(*,*)''
c      write(*,*)'pltvectr: j,i,x1,y1,x2,y2='
      do 200 i=1,ipxy
         do 201 j=1,jpxy
c            if(xh(j,i).eq.0. .and. yh(j,i).eq.0.) goto 201
            tam1(1)=xt(j,i)
            tam2(1)=yt(j,i)
            tam1(2)=xt(j,i)+veclen*xh(j,i)/rscale
            tam2(2)=yt(j,i)+veclen*yh(j,i)/rscale
            RPX1=RBOUND(tam1(1))
            RPY1=max(RBOUND(tam2(1)),0.)
            RPX2=RBOUND(tam1(2))
            RPY2=max(RBOUND(tam2(2)),0.)
            if(abs(RPX2).lt. 1.e-20) RPX2=0.  !Helps near vpar=0.
c          write(*,5) j,i,RPX1,RPY1,RPX2,RPY2
c 5        format(2i5,4(1pe12.5))
c          if ((RPX1.ne.0.) .and. (RPY1.ne.0.) .and. 
c     +        (RPX2.ne.0.) .and. (RPY2.ne.0.)) then
          if ((RPY1.ne.0.) .and. 
     +        (RPY2.ne.0.)) then
              CALL PGARRO(RPX1,RPY1,RPX2,RPY2)
           endif
 201    continue
 200  continue
c     RESTORE PGPLOT ATTRIBUTES
      CALL PGUNSA
c     
      return
      end
      

      


