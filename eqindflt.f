c
c
      subroutine eqindflt
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'name_decl.h'


c..................................................................
c     This routine set defaults for the "eq" module.
c..................................................................

      atol=1.e-8
      bsign=+1.0
      scalebt=+1.0d0
      scalebp=+1.0d0
      ellptcty=0.
      eqdskalt="disabled"
      eqdskin="eqdsk"
      eqmod="disabled"
      eqmodel="power"
      eqpower=1.
      eqsource="eqdsk"
      eqsym="average"
      fpsimodl="constant"
      lfield=250 !lfielda !YuP[2021-04] lfielda is not used anymore
                 !Now all relevant arrays are pointers of size lfield
      methflag=10
!BH171211      Changing the following defaults.  The nconteq="psigrid"
!BH171211      approach basis a psigrid for various interpolations, 
!BH171211      particularly for freqa, on number of major radius grid points
!BH171211      between the magnetic axis and the plasma edge, in some
!BH171211      cases, this is number is too small for good accuracy.
!BH171211      nconteq="psigrid"
!BH171211      nconteqn=0
      nconteq="disabled"
      nconteqn=50
      povdelp=.2
      rbox=100.
      rboxdst=50.
      rmag=100.
      rtol=1.e-8
      zbox=100.
      return
      end
