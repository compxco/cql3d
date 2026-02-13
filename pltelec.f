c
c
c
      subroutine pltelec
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c
cmnt  this routine plots electron density as a function of poloidal angl
c
      include 'param.h'
      include 'comm.h'

      REAL*4 RTAM1(lz),RTAM2(lz) !YuP[2021-04] lza-->lz
      REAL*4 RPGMIN,RPGMAX
      REAL*4 RILIN
      REAL*4 :: R40=0.,R4P2=.2,R4P8=.8,R4P45=.45,R4P95=.95

      if (noplots.eq."enabled1") return
      call aminmx(densz(1,ngen+1,negyrg,lr_),1,lz,1,fmin,fmax,kmin,kmax)
      if (fmin .eq. fmax) fmin=.9*fmax-1.e-20

      CALL PGPAGE
      CALL PGSVP(R4P2,R4P8,R4P45,R4P95)
        
      DO L=1,LZ
         RTAM1(L)=pol(L,lr_)
      ENDDO

      RPGMIN=fmin
      RPGMAX=fmax

      IF ( RPGMAX-RPGMIN .le. 1.e-16 ) THEN ! YuP [02-23-2016]
           RPGMAX= RPGMIN+1.e-16
      ENDIF
      CALL PGSWIN(RTAM1(1),RTAM1(LZ),RPGMIN,RPGMAX)

      do 3001 l=1,lz
         RTAM2(L)=densz(l,ngen+1,negyrg,lr_)
 3001 continue
      CALL PGLINE(LZ,RTAM1,RTAM2)

 3002 continue

      RILIN=1.
      CALL PGMTXT(B,RILIN,R40,R40,T_)
      write(t_,611) kelec,n,timet
      RILIN=RILIN+1.
      CALL PGMTXT(B,RILIN,R40,R40,T_)
      write(t_,612) xlndnz(ngen+1,negyrg)
      RILIN=RILIN+1.
      CALL PGMTXT(B,RILIN,R40,R40,T_)
 610  format("Density as a function of poloidal angle(=pi*z/zmax)")
 611  format("species ",i3, " (electrons)   n= ",i5,"  time= ",1pe14.4)
 612  format("density (line-integration) =",1pe16.5)
      return
      end
