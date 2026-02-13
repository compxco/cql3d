c
c
      subroutine tdtraloc
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     Allocates arrays used by transport routines.
c..............................................................

      include 'param.h'
      include 'comm.h'
c.......................................................................

      lnyxgrz=(iyp1+1)*(jxp1+1)*ngen*(lrz+1)
      lnjxng=jx*ngen*lrz
      lnjxg18=18*lnjxng
      lnyrs=iy*lrz
      lnyrmx=iy*lrzmax
      lnyrmxp=iy*(lrzmax+1)
      lnyprsp=(iy+1)*(lrz+1)
      lnng=ngen*lrzmax
      lnrz=lrz
      lnj2=jx*2*lrz

      lndum=8*lnyxgrz+3*lnjxng+lnjxg18+lnyrs+lnyrmx+lnyrmxp+
     +  2*lnyprsp+lnng+lnrz+2*lnj2

c..................................................................
c     A check on allocations is sucessful entering then exiting
c     the subroutine.
c..................................................................
      write(*,*)'tdtraloc:  Entering tdtraloc'

      allocate(frn_2(0:iy+1,0:jx+1,ngen,0:lrz),STAT=istat)
      call bcast(frn_2,zero,SIZE(frn_2))
      allocate(frn_1(0:iy+1,0:jx+1,ngen,0:lrz),STAT=istat)
      call bcast(frn_1,zero,SIZE(frn_1))
      allocate(frn(0:iy+1,0:jx+1,ngen,0:lrz),STAT=istat)
      call bcast(frn,zero,SIZE(frn))
      allocate(fvn_1(0:iy+1,0:jx+1,ngen,0:lrz),STAT=istat)
      call bcast(fvn_1,zero,SIZE(fvn_1))
      allocate(fvn(0:iy+1,0:jx+1,ngen,0:lrz),STAT=istat)
      call bcast(fvn,zero,SIZE(fvn))
      
      
      allocate(dl(0:iyp1,0:jxp1,ngen,0:lrz),STAT=istat)
      call bcast(dl,zero,SIZE(dl))
      
      if(.NOT.ASSOCIATED(d_rr))then 
      allocate(d_rr(0:iyp1,0:jxp1,ngen,0:lrz),STAT=istat)
      call bcast(d_rr,zero,SIZE(d_rr))
      endif
      
      allocate(d_r(0:iyp1,0:jxp1,ngen,0:lrz),STAT=istat)
      call bcast(d_r,zero,SIZE(d_r))
      
      allocate(f_lm(jx,ngen,lrz),STAT=istat)
      call bcast(f_lm,zero,SIZE(f_lm))
      allocate(f_lp(jx,ngen,lrz),STAT=istat)
      call bcast(f_lp,zero,SIZE(f_lp))
      allocate(f_up(jx,ngen,lrz),STAT=istat)
      call bcast(f_up,zero,SIZE(f_up))
      allocate(f_vtor(jx,ngen,18,lrz),STAT=istat)
      call bcast(f_vtor,zero,SIZE(f_vtor))
      allocate(cynt2_(iy,lrz),STAT=istat)
      call bcast(cynt2_,zero,SIZE(cynt2_))
      allocate(vpint_(iy,lrzmax),STAT=istat)
      call bcast(vpint_,zero,SIZE(vpint_))
      allocate(vptb_(iy,0:lrzmax),STAT=istat)
      call bcast(vptb_,zero,SIZE(vptb_))
      allocate(cosovb(0:iy,0:lrz),STAT=istat)
      call bcast(cosovb,zero,SIZE(cosovb))
      allocate(bovcos(0:iy,0:lrz),STAT=istat)
      call bcast(bovcos,zero,SIZE(bovcos))
      allocate(adv(ngen,lrzmax),STAT=istat)
      call bcast(adv,zero,SIZE(adv))
      allocate(dentarget(lrz),STAT=istat)
      call bcast(dentarget,zero,SIZE(dentarget))
      allocate(eg_(jx,2,lrz),STAT=istat)
      call bcast(eg_,zero,SIZE(eg_))
      allocate(fg_(jx,2,lrz),STAT=istat)
      call bcast(fg_,zero,SIZE(fg_))

      write(*,*)'tdtraloc:  Leaving tdtraloc'

      return
      end



!=======================================================================
!=======================================================================
      subroutine drrplt
      implicit integer (i-n), real*8 (a-h,o-z)
      !YuP[2021-08] Added subroutine for plots of Drr.
      !Similar to subr.souplt.
      !From cqlinput_help:
!!  pltdrr="enabled" means plot contours of Drr == d_rr(i,j,k,lr)
!!      in momentum 2D space (v_par,v_perp) [normalized by vnorm or c]
!!      at time steps selected by nplot=*,
!!      and such that n.ge.nontran .and. n.le.nofftran
!!    ="first" plot contours (black lines), only if n=nontran
!!     (or first time the subr.pltmain is called at n.ge.nontran)
!!    Also, options to plot with colors :
!!    ="color" means plot contours of the Drr in color,
!!      at time steps selected by nplot=*,
!!      and such that n.ge.nontran .and. n.le.nofftran
!!    ="first_cl" plot contours in color, only if n=nontran
!!     (or first time the subr.pltmain is called at n.ge.nontran)
!!    default: pltdrr="first"            Added: YuP[2021-08]

c..................................................................
c     contour plots of Drr diff. coeff.
c..................................................................

      include 'param.h'
      include 'comm.h'
      parameter(nconta=100)
      common/contours/cont(nconta),tempcntr(nconta)

      REAL*4 RILIN
      REAL*4 RTAM1(jx),RTAM2(jx)
      REAL*4 REMAX,REMIN
      REAL*4 :: R4MP2=-.2,R40=0.

      if (pltdrr.eq."disabled") return
      
      do 10 k=1,ngen
      
         temp1=0.d0 ! initialize for each k species
         do j=1,jx
         do i=1,iy !iytr(indxlr_)
          !temp1(i,j)=d_rr(idx(i,indxlr_),j,k,indxlr_)
          temp1(i,j)=d_rr(i,j,k,indxlr_)
          !Note allocation: d_rr(0:iyp1,0:jxp1,ngen,0:lrz)
         enddo
         enddo
         !!call dcopy(iyjx2,d_rr(0,0,k,indxlr_),1,temp1(0,0),1)
         drr_max=MAXVAL(temp1(1:iy,1:jx)) !Units: [cm^2/sec]
         if (drr_max.lt.1.e-6) goto 10 !-> Too small, skip; next k 
         
         write(t_,550) k
 550     format(1x,"Species k=",i2,".   Drr (units: cm**2/sec)")
         CALL PGPAGE
         
         itype=9 ! means: plots are made for Drr
         call pltcont(k,2,t_,itype) ! itype=9 for Drr
         ! On second argument:
         !pltcase=1: geometric progression of plot contours,
         !pltcase.ne.1: contours equispaced for max. at temp(k,lr_),

         RILIN=10.
         write(t_,560)
 560     format("Contour values:")
         RILIN=RILIN+2.
         CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
         do  jcs=1,ncont,4
            write(t_,570) (tempcntr(jc),jc=jcs,min(jcs+3,ncont))
            if ((ncont/4)*4.ne.ncont .and. ncont-jcs.le.2) then
               icend=4 * 16 + 1
               t_(icend:icend)="$"
            endif
            RILIN=RILIN+1.
            CALL PGMTXT('B',RILIN,R4MP2,R40,t_)
         enddo
         
 10   continue ! k species

 570  format(4(1pe16.4))
         
   
      return
      end subroutine drrplt
!=======================================================================
!=======================================================================