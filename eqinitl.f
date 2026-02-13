c
c
      subroutine eqinitl
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      REAL*4 RILIN
      REAL*4 :: R40=0.,R41=1.,R4P1=0.1,R4P9=0.9

c..................................................................
c     This routine does some minor initialization for
c     the "eq" module. Called after the namelist read.
c..................................................................

      !YuP if (lfield.gt.lfielda) lfield=lfielda  !YuP[2021-04] lfielda is not used anymore
      !Now all relevant arrays are pointers of size lfield
      nrc=(nnr-1)/2+1
      nzc=(nnz-1)/2+1
      zshift=0.0


CMPIINSERT_IF_RANK_NE_0_RETURN
 ! make plots on mpirank.eq.0 only

      if (noplots.ne."enabled1") then
      CALL PGPAGE
      CALL PGSVP(R4P1,R4P9,R4P1,R4P9) !(XLEFT,XRIGHT,YBOT,YTOP)!YuP[2021-01-18]
      RILIN=0.
      CALL PGMTXT('T',-RILIN,R40,R40,"PARAMETER VALUES")
      
      write(t_,1000) 
 1000 format("EQUILIBRIUM model parameters:")
      RILIN=2.
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      
      write(t_,1001)
 1001 format("nnra,nnza give the Maximum size the eqdsk")
      RILIN=3.
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      
      write(t_,1002) nnra,nnza
 1002 format("====>NNRA = ",i5,"        ====>NNZA = ",i5)
      RILIN=4.
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      
      write(t_,1003) nconteqa
 1003 format("====>NCONTEQA = ",i5)
      RILIN=5.
      CALL PGMTXT('T',-RILIN,R40,R40,t_)

      endif


      return
      end
