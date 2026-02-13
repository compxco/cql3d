c
c
      subroutine pltdf
      implicit integer (i-n), real*8 (a-h,o-z)
      parameter(nconta=100)
      common/contours/cont(nconta),tempcntr(nconta)

c..................................................................
c     if (pltd.eq."enabled" or pltd.eq."color") then
c     subroutine pltdf contour plots the distribution function, f
c     if (pltd.eq."df"  or "df_color") then
c     subroutine pltdf also plots the difference between f
c     at the current time and f at the previous time
c     step.
c..................................................................

      include 'param.h'
      include 'comm.h'

      REAL*4 RILIN
      REAL*4 :: R40=0.,R4MP2=-0.2


      if (noplots.eq."enabled1") return

      if (pltd.eq."disabled") return


      do 10 k=1,ngen
      
        ! This part is plotted for any pltd.ne.'disabled'
        
        call dcopy(iyjx2,f(0,0,k,l_),1,temp1(0,0),1)
        write(t_,550) k
 550    format(1x,"Species ",i2," Distribution Function Contour Plot")
        CALL PGPAGE
        itype=1 ! means: plots are made for distr.func f
        call pltcont(k,1,t_,itype) !for f()  pltcase=1 means log10 scale
        write(t_,560)
 560    format("Contour values:")
        RILIN=10.
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)


        do 11 jcs=1,ncont,4
          write(t_,570) (tempcntr(jc),jc=jcs,min(jcs+3,ncont))
          if ((ncont/4)*4.ne.ncont .and. ncont-jcs.le.2) then
            icend=4 * 16 + 1
            t_(icend:icend)="$"
          endif
          RILIN=RILIN+1.
          CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
 11     continue
        
 
        if (n.eq.0) goto 10

        ! Additionally, plot f(n+1)-f(n)
        if (pltd.eq."df" .or. pltd.eq."df_color")then
        
        do 20 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
          do 21 j=1,jx
            temp1(i,j)=f(i,j,k,l_)-f_(i,j,k,l_)
 21       continue
 20     continue
        write(t_,530) k,n
 530    format(1x,
     +  "Contours of df/dt for species",i2,1x,"during timestep",i5)
        CALL PGPAGE
        itype=2 ! means: plots are made for df
        call pltcont(k,1,t_,itype) ! for df
        RILIN=10.
        write(t_,560)
        CALL PGMTXT('B',RILIN,R4MP2,R40,t_)

        do 12 jcs=1,ncont,4
          write(t_,570) (tempcntr(jc),jc=jcs,min(jcs+3,ncont))
          if ((ncont/4)*4.ne.ncont .and. ncont-jcs.le.2) then
            icend=4 * 16 + 1
            t_(icend:icend)="$"
          endif
          RILIN=RILIN+1.
          CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
 12     continue
 
      endif !  pltd.eq."df" .or. pltd.eq."df_color"
      
 
 10   continue ! k

 570  format(4(1pe16.6))

      return
      end
