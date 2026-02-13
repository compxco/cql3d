c
c
      subroutine prpprctr
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c
      include 'param.h'
      include 'comm.h'
c
c     Modified from Graflib to pgplot calls by Yuri Petrov, 090727,
c     using PGPLOT + GRAFLIBtoPGPLOT.f routines (put in pltmain.f).
c     Still need to finish contour modifications of GPCTV3.
c

      REAL*4 RILIN !-> For PGPLOT (text output positioning)
      REAL*4 :: R40=0.,R41=1.
      REAL*4 :: R4P2=.2,R4P8=.8,R4P6=.6,R4P9=.9
c...............................................................
c     This routine performs contour plots of data
c     set in tempcntr by subroutine pltcont
c...............................................................
      parameter(nconta=100)
      common/contours/cont(nconta),tempcntr(nconta)
      ipjpxy=ipxy*jpxy
      dmin=0.
      dmax=0.
      do 10 jp=1,jpxy
        do 15 ip=1,ipxy
          xllji(jp,ip)=xpar(jp)
          xppji(jp,ip)=xperp(ip)
 15     continue
 10   continue
      call aminmx(fpn,1,ipjpxy,1,dmin,dmax,kmin,kmax)
      admin=abs(dmin)
      if (dmin.ge.0. .or. admin.lt.contrmin*dmax) then
        k2=1
c990131        smin=alog(contrmin*dmax)
c990131        if (admin/dmax .gt. contrmin) smin=alog(admin)
c990131        smax=alog(dmax)
        smin=log(contrmin*dmax)
        if (admin/dmax .gt. contrmin) smin=log(admin)
        smax=log(dmax)
        dcont=(smax-smin)/float(ncont)
        cont(1)=smin+.5*dcont
        do 20 kc=2,ncont
          cont(kc)=cont(kc-1)+dcont
 20     continue
        do 30 kc=1,ncont
          cont(kc)=exp(cont(kc))
 30     continue
      else
        if (dmax .gt. 0.) then
          k2=ncont/2+1
          ncontp=ncont-k2+1
          ncontm=k2-1
c990131          smaxp=alog(dmax)
c990131          sminp=alog(contrmin*dmax)
          smaxp=log(dmax)
          sminp=log(contrmin*dmax)
        else
          ncontm=ncont
          ncontp=1
          k2=1
        endif
c990131        sminm=alog(-contrmin*dmin)
c990131        if (dmax/dmin.gt.contrmin) sminm=alog(-dmax)
c990131        smaxm=alog(-dmin)
        sminm=log(-contrmin*dmin)
        if (dmax/dmin.gt.contrmin) sminm=log(-dmax)
        smaxm=log(-dmin)
        dcontp=(smaxp-sminp)/float(ncontp)
        dcontm=(smaxm-sminm)/float(ncontm)
        cont(1)=smaxm-.5*dcontm
        do 40 kc=2,ncontm
          cont(kc)=cont(kc-1)-dcontm
 40     continue
        do 50 kc=1,ncontm
          cont(kc)=-exp(cont(kc))
 50     continue
        if (dmax .gt. 0.) then
          cont(k2)=sminp-.5*dcontp
          do 60 kc=k2+1,ncont
            cont(kc)=cont(kc-1)+dcontp
 60       continue
          do 70 kc=k2,ncont
            cont(kc)=exp(cont(kc))
 70       continue
        endif
      endif
      
      CALL PGPAGE ! new page
      call PGSVP(R4P2,R4P8,R4P6,R4P9)
      CALL PGSCH(R41) ! set character size; default is 1.
      call GSWD2D("linlin$",xpar(1),xpar(jpxy),xperp(1),xperp(ipxy))
      call PGLAB('u/unorm_par','u/unorm_perp',' ')
      if (k2.gt.1) then
        CALL PGSLS(2)
        call GSLNSZ(R4P2)
        call GSLNSZ(R40)
      endif
      CALL PGSLS(1)
      call GSLNSZ(R4P2)
      call GSLNSZ(R40)
      
      CALL PGSLS(1) ! restore: solid line
      CALL PGSLW(lnwidth) ! restore linewidth

      CALL PGSCH(R41) ! restore to default font size      
      return
      end
