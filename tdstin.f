c
c
      subroutine tdstin
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c
c
      include 'param.h'
      include 'comm.h'


c..................................................................
c     This routine initializes flux surface dependent data so the
c     2-d code initialization routine ainitial can be called.
c..................................................................

      do 2 k=1,ngen
        do 3 m=1,nso
          sellm1(k,m)=sellm1z(k,m,lr_)
          seppm1(k,m)=seppm1z(k,m,lr_)
          sellm2(k,m)=sellm2z(k,m,lr_)
          seppm2(k,m)=seppm2z(k,m,lr_)
          sem1(k,m)=sem1z(k,m,lr_)
          sem2(k,m)=sem2z(k,m,lr_)
          sthm1(k,m)=sthm1z(k,m,lr_)
          scm2(k,m)=scm2z(k,m,lr_)
          szm1(k,m)=szm1z(k,m,lr_) 
          !for facz=exp(-(zl-zm1(kk,m,lr_))**2/zm2(kk,m,lr_))
          !where zm1=zmax(lr_)*szm1(k,m)
          !      zm2=(zmax(lr_)*szm2(k,m))**2
          szm2(k,m)=szm2z(k,m,lr_)
          asor(k,m,lr_)=asorz(k,m,lr_)
 3      continue
 2    continue

      return
      end
