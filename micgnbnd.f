c
c
      subroutine micgnbnd
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................

c     This routine computes the "average" bounce time
c     in the region of velocity space (a sliver) about the
c     pass/trapped boundary. Note that the bounce time is infinite
c     on the p/t boundary (y(itl,l_)) but the singularity is
c     integrable.
c     See the user manual for mathematical details.
c..................................................................

      include 'param.h'
      include 'comm.h'

      sp=(y(itl,l_)+y(itl-1,l_))*.5
      sm=(y(itl+1,l_)+y(itl,l_))*.5
      dp=cos(sp)/coss(itl,l_)-1.
      dm=-(cos(sm)/coss(itl,l_)-1.)

      ilzhfs=lz
      if (cqlpmod.eq."enabled" .and. numclas .eq. 1) ilzhfs=lz/2+1
      !YuP[2025-12-12] added cqlpmod.eq."enabled" in the above line     

c..................................................................
c     For standard circular cross section model..
c..................................................................

      if (psimodel.eq."axitorus") then
        betta(0,lr_)=0.
        betta(1,lr_)=1.
        alm(0,lr_)=1.
        alm(1,lr_)=-.5
        do 10 m=2,mbet
          betta(m,lr_)=-((2*m-3)*betta(m-2,lr_)-(4.*m-4.)
     1      *betta(m-1,lr_))/(2*m-1)
          alm(m,lr_)=(2*m-3)/(2.*m)*alm(m-1,lr_)
 10     continue
        ak0=1.38629
        bk0=.5
 20     continue
        s=0.
        do 30 m=0,mbet
          s=s-alm(m,lr_)*coss(itl,l_)**(2*m+1)*betta(m,lr_)*(dm+dp)
 30     continue
        s=s+coss(itl,l_)*sinn(itl,l_)
c990131     1    *(dm*(ak0-bk0*(alog(2.*dm)-1.))+dp*(ak0-bk0*(alog(2.*dp)-1.)))
     1    *(dm*(ak0-bk0*(log(2.*dm)-1.))+dp*(ak0-bk0*(log(2.*dp)-1.)))
        xlbnd(lr_)=s*8.*zmax(lr_)

c..................................................................
c     For non-circular geometries use alternate algorithm.
c............................................................

      else
        xlbnd1=4.*pi*coss(itl,l_)*coss(itl,l_)*tau(itl,lr_)
        call micgmbnd(xlbmd2)
        xlbnd(lr_)=(dp+dm)*(xlbnd1)+xlbmd2
      endif
      xu=xlbnd(lr_)/(twopi*sinn(itl,l_)*dy(itl,l_)*coss(itl,l_)*2.)
      dtau(itl,ilzhfs,lr_)=xu-tau(itl,lr_)
      dtau(itu,ilzhfs,lr_)=dtau(itl,ilzhfs,lr_)
      if (lmax(itl,lr_)+1 .eq. ilzhfs) go to 60
      if (lmax(itl,lr_)+1 .gt.lz) stop 'in micgnbnd: need to check this'
      dtau(itl,lmax(itl,lr_)+1,lr_)=0.
      dtau(itu,lmax(itl,lr_)+1,lr_)=0.
 60   continue
      lmax(itl,lr_)=ilzhfs
      lmax(itu,lr_)=ilzhfs
      tau(itl,lr_)=xu
      tau(itu,lr_)=xu
      do 40 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
        vptb(i,lr_)=abs(coss(i,l_))*tau(i,lr_)
 40   continue
      return
      end
