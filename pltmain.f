c
c
      subroutine pltmain
      implicit integer (i-n), real*8 (a-h,o-z)
      save
c
c     This routine controls driver plots
c
c
c     Modified some Graflib to pgplot calls by Yuri Petrov, 090727,
c     using PGPLOT + GRAFLIBtoPGPLOT.f routines (put in pltmain.f).
c
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      REAL*4 RILIN
      REAL*4 :: R40=0.,R41=1.
      REAL*4 :: R4P2=.2,R4P8=.8,R4P6=.6,R4P9=.9 !.2,.8,.6,.9
      

CMPIINSERT_IF_RANK_NE_0_RETURN
 ! make plots on mpirank.eq.0 only
 
      data isouplt/0/, idrrplt/0/ !for logic control
      
      if (noplots.eq."enabled1") return
      if (n.eq.0 .and. lrzmax.gt.1) return
      if (mplot(l_).eq."disabled") return
      
      !write(*,*)'entering pltmain. n,l_=',n,l_

      rr=rpcon(lr_) !rovera(lr_)*radmin  ! YuP[03-2016] changed to rpcon
      if (pltend.eq."notplts") goto 10
      if (pltend.eq."last" .and. n.lt.nstop) goto 10

      CALL PGPAGE
      CALL PGSVP(R4P2,R4P8,R4P2,R4P8) !(XLEFT,XRIGHT,YBOT,YTOP)!YuP[2021-01-18]
      CALL PGSCH(R41) ! restore to default font size
      !(sometimes font is too big from previous plot)

      RILIN=0.
      if (cqlpmod .ne. "enabled") then
         CALL PGMTXT('T',-RILIN,R40,R40,"LOCAL RADIAL QUANTITIES")
      else ! (cqlpmod.eq."enabled")
         CALL PGMTXT('T',-RILIN,R40,R40,"LOCAL PARALLEL QUANTITIES")
      endif
      RILIN=RILIN+1.

      write(t_,150) n,timet
      RILIN=RILIN+1.
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      write(t_,1501) lr_,lrz
      RILIN=RILIN+1.
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      if(cqlpmod.eq."enabled") then !YuP[2021-03-02] Add line with l_ info
        write(t_,1502) l_,sz(l_)
        RILIN=RILIN+1.
        CALL PGMTXT('T',-RILIN,R40,R40,t_)
      endif
      write(t_,151) rovera(lr_),rr
      RILIN=RILIN+1.
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
      write(t_,153) rya(lr_),rpcon(lr_)
      RILIN=RILIN+1.
      CALL PGMTXT('T',-RILIN,R40,R40,t_)
 150  format("time step n=",i5,","5x,"time=",1pe12.4," secs")
 1501 format("flux surf=",i3,5x,"total flux surfs=",i3)
 1502 format("point on B, l=",i4,4x,"Parallel position s=",1pe14.6,"cm")
 151  format("r/a=",1pe10.3,5x,"radial position (R)=",1pe12.4," cms")
 153  format("rya=",1pe10.3,5x,"R=rpcon=",1pe10.3," cm")
 
      if (cqlpmod.eq."enabled") then
        write(t_,152) l_,sz(l_)
        RILIN=RILIN+1.
        CALL PGMTXT('T',-RILIN,R40,R40,t_)
 152    format("orbit at s(",i5,") = ",1pe10.2,"cm")
      endif

      vnormdc=vnorm/clight
      if(cqlpmod .ne. "enabled")then
        vtedc=vthe(lr_)/clight
        vtdvnorm=vthe(lr_)/vnorm
      else !(cqlpmod.eq."enabled") ! YuP[2021-02-26]
        vtedc=   vthpar(kelec,ls_)/clight
        vtdvnorm=vthpar(kelec,ls_)/vnorm
      endif
      if (tandem.eq."enabled") then ! YuP[08-2017] added, 
        ! for the case of ngen=2 tandem i+e runs:
        !In this case, enorm=enorme
        !and  xlwr=sqrt(enormi*fmass(kelecg)/(enorme*fmass(kionn)))
        write(t_,'(a,2f11.3)') ' enormi, enorme(=enorm) (kev) =',
     +                           enormi, enorme
      else
        write(t_,'(a,f11.3)') ' enorm (kev) =' ,enorm
      endif
        RILIN=RILIN+1.
        CALL PGMTXT('T',-RILIN,R40,R40,t_)
      write(t_,161)  vnormdc
        RILIN=RILIN+1.
        CALL PGMTXT('T',-RILIN,R40,R40,t_)
      write(t_,162)  vtedc
        RILIN=RILIN+1.
        CALL PGMTXT('T',-RILIN,R40,R40,t_)
      write(t_,163)  vtdvnorm
        RILIN=RILIN+1.
        CALL PGMTXT('T',-RILIN,R40,R40,t_)
      do k=1,ntotal
         if(cqlpmod .ne. "enabled")then
          vth_l= vth(k,lr_)
         else !(cqlpmod.eq."enabled") ! YuP[2021-02-26]
          vth_l= vthpar(k,ls_)
         endif
         vtdvnorm= vth_l/vnorm
         !YuP/note: For time-dependent profiles, 
         ! temp() can evolve in time, and so vth(k,*) can evolve, too.
         ! See profiles.f.
         write(t_,'(a,i2,a,f15.7)') "k=",k, "  vth(k)/unorm =", vtdvnorm
         RILIN=RILIN+1.
         CALL PGMTXT('T',-RILIN,R40,R40,t_)
      enddo

 160  format("enorm (kev) = ",f11.3)
 161  format("unorm/c = ",f15.7)
 162  format("vthe (sqrt(Te/me))/c = ",f15.7)
 163  format("vthe/unorm = ",f15.7)
 
      if (cqlpmod.eq."enabled") then
        !YuP: Wrong? zvthes=vth(kelec,l_)/clight  
        !YuP: Wrong? zvtheon=vth(kelec,l_)/vnorm
        !YuP: vth() is a func of lr_, not l_ !!!
        !YuP: We need to use vthpar !!!
        vth_l= vthpar(kelec,ls_) !YuP[2021-03]
        zvthes=  vth_l/clight    !YuP[2021-03]
        zvtheon= vth_l/vnorm     !YuP[2021-03]
        write(t_,164) zvthes
        write(t_,165) zvtheon
        RILIN=RILIN+1.
        CALL PGMTXT('T',-RILIN,R40,R40,t_)
 164    format(";","vthe(s) (sqrt(Te/me))/c = ",f15.7)
 165    format("vthe(s)/unorm = ",f15.7)
      endif

      ncplt=ncplt+1
 10   continue

c     Plot energy, density, toroidal current, conservation diagnostic vs
c     time
c
      !write(*,*)'------1 pltmain. n,l_=',n,l_
      if (pltend.ne."disabled" .and. nch(l_).ge.2) then
        if (cqlpmod.ne."enabled") call pltendn
        if (cqlpmod.eq."enabled") call pltends
      endif
      !write(*,*)'------2 pltmain. n,l_=',n,l_
c
c     Plot ion source if marker is engaged.
c
      !isouplt=0 !YuP[2021-08] changed to data isouplt/0/
      if (pltso.eq."enabled" .or. pltso.eq."color".and.
     +         n.ge.nonso(1,1) .and. n .le. noffso(1,1)) then
        call souplt !At time steps within [nonso;noffso]
                    !selected by nplot=*** setting
      elseif ( (pltso.eq."first" .or. pltso.eq."first_cl") .and.
     +         isouplt.eq.0 .and.
     +         n.ge.nonso(1,1) .and. n .le. noffso(1,1)) then
        call souplt !One time only
        isouplt=1
      endif
      !write(*,*)'------3 pltmain. n,l_=',n,l_
           
c
c     Plot Drr(i,j) (radial diff.coeff from d_rr array) !YuP[2021-08]
c
      if(cqlpmod.ne."enabled")then !plot Drr - For CQL3D only
      if(transp.eq."enabled")then 
        !idrrplt=0 !See data idrrplt/0/ above.
        if (pltdrr.eq."enabled" .or. pltdrr.eq."color" .and.
     +           n.ge.nontran .and. n.le.nofftran) then
          call drrplt !at time steps within [nontran;nofftran]
                      !selected by nplot=*** setting
        elseif ( (pltdrr.eq."first" .or. pltdrr.eq."first_cl") .and.
     +           idrrplt.eq.0 .and.
     +           n.ge.nontran) then
          call drrplt !One time only
          idrrplt=1 !inhibits from further calls
        endif
      endif !transp
      endif
            
c
c     Plot electron resistivity and related quantities.
c
      if (abs(elecfld(lr_)).gt.1.e-9 .and. n.ge.nonel) then
        if (pltrst.eq."enabled" .and. nch(l_).ge.2 .and. kelecg.gt.0)
     +    then
          call pltrstv
        endif
      endif
      !write(*,*)'------4 pltmain. n,l_=',n,l_
c
c     Plot normalized v-flux..
c
      if (abs(elecfld(lr_)).gt.1.e-9 .and. n.ge.nonel) then
cBH        if (pltvflu.eq."enabled" .and. n.gt.1 .and. kelecg.gt.0) then
        if (pltvflu.eq."enabled" .and. n.ge.1 .and. kelecg.gt.0) then
          call pltvflux
        endif
      endif
      !write(*,*)'------5 pltmain. n,l_=',n,l_
c
c     Plot power deposited in plasma by various mechanisms vs.
c     time.
c
      if (pltpowe.eq."enabled" .and. nch(l_).ge.2) then
        call pltpower
      elseif (pltpowe.eq."last" .and. n.eq.nstop) then
        call pltpower
      endif
      !write(*,*)'------6 pltmain. n,l_=',n,l_
c...  
cmnt  Distribution f slices at constant pitch and/or pitch angle 
cmnt    averaged f,  vs velocity or energy coordinate.
c...  
      if (pltfvs.eq."enabled".or.pltfofv.eq."enabled") call pltfvsv
c...  
c...  
cmnt  Distribution fluxes vs velocity for some values of theta
c...  
      if (pltflux.eq."enabled") call pltfluxs


c...  
cmnt  Contour the loss region.
c...  
      if (pltlos.ne."disabled") call pltlosc
c...

      !write(*,*)'------7 pltmain. n,l_=',n,l_
  
cmnt  Plot the reduced parallel distribution (f integrated on vpp)
c...  
      if (pltprpp.eq."enabled") call pltprppr
      !write(*,*)'------7a pltmain. n,l_=',n,l_
c
c     Plot the density as a function of poloidal angle for a
c     set of energy ranges..
c
      if (n.ne.0 .and. pltdn.ne."disabled" .and. cqlpmod.ne."enabled") 
     +  call pltdnz
c
c     plot contours of df/dt next..
c
      if (pltd.ne."disabled") call pltdf
c
c     vector flux plots follow..
c
      if (n.eq.0 .or. pltvecal.eq."disabled") goto 670
      call pltvec(4)
      if (pltvecrf.ne."disabled") call pltvec(3)
      if (pltvece.ne."disabled") call pltvec(2)
      if (pltvecc.ne."disabled") call pltvec(1)
 670  continue
c
c     Plot the stream lines of the steady state flux
c
      if (pltstrm.ne."disabled" .and. n.ge.1) call pltstrml
      !write(*,*)'LEAVING pltmain. n,l_=',n,l_
      
      CALL PGSCH(R41) ! restore to default font size      
      return
      end subroutine pltmain





C==== CONVERT SOME GRAFLIB ROUTINES to PGPLOT ========================
C  Yuri Petrov, 090727
c---------------------------------------------------------------------
!      subroutine gxglfr(n) !YuP[2019-10-28] Not used anymore. Converted to PGPAGE
!      integer n       
!        CALL PGPAGE
!      return 
!      end
c---------------------------------------------------------------------
!      subroutine gsvp2d(xmin,xmax,ymin,ymax) ! called explicitly with real*4 args
!      !YuP[2019-10-28] Not used anymore. All calls are converted to PGSVP
!      ! xmin,...defines where on the frame the data is plotted. 
!      real*4 xmin,xmax,ymin,ymax
!      REAL*4 PGxmin,PGxmax,PGymin,PGymax ! PGPLOT uses REAL*4
!        PGxmin = xmin
!        PGxmax = xmax
!        PGymin = ymin
!        PGymax = ymax
!        CALL PGSVP(PGxmin,PGxmax,PGymin,PGymax)
! PGSVP (XLEFT, XRIGHT, YBOT, YTOP)
! XLEFT  (input)  : x-coordinate of left hand edge of viewport, in NDC.
! XRIGHT (input)  : x-coordinate of right hand edge of viewport,in NDC.
! YBOT   (input)  : y-coordinate of bottom edge of viewport, in NDC.
! YTOP   (input)  : y-coordinate of top  edge of viewport, in NDC.        
!      return 
!      end
c---------------------------------------------------------------------
      subroutine gswd2d(scales,xmin,xmax,ymin,ymax) !xmin,...,ymax are real*8
      ! xmin,... defines the user coordinate system.            
      implicit none !integer (i-n), real*8 (a-h,o-z)
      real*8 xmin,xmax,ymin,ymax
      REAL*4 PGxmin,PGxmax,PGymin,PGymax, RPG1,RPG2 ! PGPLOT uses REAL*4
      REAL*4 RBOUND ! external function
      REAL*4 :: R40=0.,R41=1.
      character*7 scale,scales ! = "linlin$" or "linlog$","loglin$","loglog$"
      common/GRAFLIB_PGPLOT_scale/ scale
        scale  = scales ! To gpcv2d -> PGLINE
        PGxmin = RBOUND(xmin) ! to real*4
        PGxmax = RBOUND(xmax)
        PGymin = RBOUND(ymin)
        PGymax = RBOUND(ymax)
        IF ( PGymax-PGymin .le. 1.e-16 ) THEN ! YuP [02-23-2016]
           PGymax= PGymin+1.e-16
        ENDIF
        CALL PGSCH(R41) ! set character size; default is 1.
        if(scale.eq."linlin$") then
          CALL PGSWIN(PGxmin,PGxmax,PGymin,PGymax)
          CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
        endif
        if(scale.eq."loglin$") then 
          CALL PGSWIN(log10(PGxmin),log10(PGxmax),PGymin,PGymax)
          CALL PGBOX('BCNSTL',R40,0,'BCNST',R40,0)
        endif
        !----------------------------
        PGymin= max(PGymin,1.e-32) ! cannot be negative
        PGymax= max(PGymax,1.e-32) ! cannot be negative
        RPG1= log10(PGymin) 
        !do not use alog10 - arg.type dep. on compiler options
        RPG2= log10(PGymax)
        IF ( RPG2-RPG1 .le. 1.e-16 ) THEN ! YuP [02-23-2016]
           RPG2= RPG1+1.e-16
        ENDIF
        if(scale.eq."linlog$") then 
          CALL PGSWIN(PGxmin,PGxmax,RPG1,RPG2)
          CALL PGBOX('BCNST',R40,0,'BCNSTL',R40,0)
        endif
        if(scale.eq."loglog$") then
          CALL PGSWIN(log10(PGxmin),log10(PGxmax),RPG1,RPG2)
          CALL PGBOX('BCNSTL',R40,0,'BCNSTL',R40,0)
        endif
      return 
      end subroutine gswd2d
      
c---------------------------------------------------------------------
      subroutine gpcv2d(xarray,yarray,length)   
      ! Plot a line (xarray,yarray) of length n.    
      implicit integer (i-n), real*8 (a-h,o-z)
      real*8 xarray(length),yarray(length)
      integer length
      REAL*4 PGxarray(length),PGyarray(length)
      character*7 scale ! = "linlin$" or "linlog$","loglin$","loglog$"
      common/GRAFLIB_PGPLOT_scale/ scale
      small_p = EPSILON(1.0) !a positive number that is almost negligible
        if(scale.eq."linlin$") then
          do n=1,length
             PGxarray(n)= xarray(n) ! Convert to REAL*4 for PGPLOT
             PGyarray(n)= yarray(n) ! Convert to REAL*4 for PGPLOT
          enddo
        endif        
        if(scale.eq."linlog$") then
          do n=1,length
             PGxarray(n)= xarray(n) ! Convert to REAL*4 for PGPLOT
             PGyarray(n)= log10( max(small_p,abs(yarray(n))) ) 
          enddo
        endif
        if(scale.eq."loglin$") then
          do n=1,length
             PGxarray(n)= log10( max(small_p,abs(xarray(n))) )
             PGyarray(n)= yarray(n) ! Convert to REAL*4 for PGPLOT
          enddo
        end if
        if(scale.eq."loglog$") then
          do n=1,length
             PGxarray(n)= log10( max(small_p,abs(xarray(n))) ) 
             PGyarray(n)= log10( max(small_p,abs(yarray(n))) )
          enddo
        endif
        CALL PGLINE(length,PGxarray,PGyarray) 
c Primitive routine to draw a Polyline. A polyline is one or more
c connected straight-line segments.  The polyline is drawn using
c the current setting of attributes color-index, line-style, and
c line-width. The polyline is clipped at the edge of the window.
c  N=length (input): number of points defining the line; the line
c                    consists of (N-1) straight-line segments.
c                    N should be greater than 1 (if it is 1 or less,
c                    nothing will be drawn).
c  X=xarray (input): world x-coordinates of the points.
c  Y=yarray (input): world y-coordinates of the points.
c The dimension of arrays X and Y must be greater than or equal to N.
c The "pen position" is changed to (X(N),Y(N)) in world coordinates
c (if N > 1).
      return 
      end
c---------------------------------------------------------------------
!      subroutine gpln2d(x1, x2, y1, y2) !YuP[2019-10-28] Not used anymore.
!      ! draw line between two points.
!      ! x1,x2,y1,y2 defines the two points to draw a line between. 
!      implicit integer (i-n), real*8 (a-h,o-z)
!      REAL*4 PGx1, PGy1, PGx2, PGy2
!        PGx1 = x1 ! Convert to REAL*4
!        PGx2 = x2 ! Convert to REAL*4
!        PGy1 = y1 ! Convert to REAL*4
!        PGy2 = y2 ! Convert to REAL*4
!        CALL PGMOVE (PGx1, PGy1)
!c Move the "pen" to the point with world
!c coordinates (X,Y). No line is drawn.
!        CALL PGDRAW (PGx2, PGy2)
!c Draw a line from the current pen position to the point
!c with world-coordinates (X,Y). The line is clipped at the edge of the
!c current window. The new pen position is (X,Y) in world coordinates.
!      return 
!      end
c---------------------------------------------------------------------
      subroutine gslnsz(size) ! called explicitly with real*4 args
      ! set line size (width), where 0. is the default.
      real*4 size
      INTEGER  LW
      LW = int(size*10. + 1.) !-YuP: Not sure if this conversion is ok
        CALL PGSLW(LW)
      ! Set the line-width attribute. This attribute affects lines, graph
      ! markers, and text. The line width is specified in units of 1/200 
      ! (0.005) inch (about 0.13 mm) and must be an integer in the range
      ! 1-201. On some devices, thick lines are generated by tracing each
      ! line with multiple strokes offset in the direction perpendicular 
      ! to the line.
      return 
      end
c---------------------------------------------------------------------
!      subroutine gslnst(LS)  !YuP[2019-10-28] Not used anymore. 
!      ! sets line style: 1-solid, 2-dashed, 3-dotted, 4-dash-dotted, etc.
!      implicit integer (i-n), real*8 (a-h,o-z)
!      INTEGER  LS
!        CALL PGSLS(LS)
!c Set the line style attribute for subsequent plotting. This
!c attribute affects line primitives only; it does not affect graph
!c markers, text, or area fill.
!c Five different line styles are available, with the following codes:
!c 1 (full line), 2 (dashed), 3 (dot-dash-dot-dash), 4 (dotted),
!c 5 (dash-dot-dot-dot). The default is 1 (normal full line).
!      return 
!      end
c---------------------------------------------------------------------
!      subroutine  gscpvs(gl_x,gl_y)  !YuP[2019-10-28] Not used anymore.
!      ! set current position for text
!      real*4 gl_x,gl_y
!      REAL*4 X,Y,ANGLE,FJUST
!      common/GRAFLIB_PGPLOT_text/ X,Y,ANGLE,FJUST
!        X = gl_x  ! To PGPTXT(X,Y,ANGLE,FJUST,TEXT)
!        Y = gl_y
!      return 
!      end
c---------------------------------------------------------------------
!      subroutine  gstxan(gl_angle)  !YuP[2019-10-28] Not used anymore.
!      ! set angle to plot text
!      real*4 gl_angle
!      REAL*4 X,Y,ANGLE,FJUST
!      common/GRAFLIB_PGPLOT_text/ X,Y,ANGLE,FJUST
!        ANGLE = gl_angle ! To PGPTXT(X,Y,ANGLE,FJUST,TEXT) 
!      return 
!      end
c---------------------------------------------------------------------
!      subroutine  gstxjf(just1,just2) !YuP[2019-10-28] Not used anymore.
!      ! set justification of string
!      character*(*) just1 !can be 'left', 'right', or 'center'
!      character*(*) just2 !can be 'top', 'bottom', or 'center'
!      REAL*4 X,Y,ANGLE,FJUST
!      common/GRAFLIB_PGPLOT_text/ X,Y,ANGLE,FJUST
!         FJUST = 0.0
!         if(just1.eq."left")   FJUST = 0.0
!         if(just1.eq."center") FJUST = 0.5
!         if(just1.eq."right")  FJUST = 1.0
!         ! To PGPTXT(X,Y,ANGLE,FJUST,TEXT)
!      return 
!      end
c---------------------------------------------------------------------
!      subroutine  gstxno(size) !YuP[2019-10-28] Not used anymore.
!      ! set number of characters per line, i.e., character width.
!      ! Default: size=90.  (not sure)
!      real*4 size
!      REAL*4 PGsize
!      REAL*4 :: R490=90.
!         PGsize = size ! Convert to REAL*4
!         CALL PGSCH(R490/PGsize) ! set character size; default is 1.
!      return 
!      end
c---------------------------------------------------------------------
!      subroutine  gptx2d(text)  !YuP[2019-10-28] Not used anymore.
!      ! Plot Text
!      character*(*) text
!      REAL*4 X,Y,ANGLE,FJUST
!      common/GRAFLIB_PGPLOT_text/ X,Y,ANGLE,FJUST
!        call PGPTXT(X, Y, ANGLE, FJUST, text)
!c Primitive routine for drawing text. The text may be drawn at any
!c angle with the horizontal, and may be centered or left- or right-
!c justified at a specified position.  Routine PGTEXT provides a
!c simple interface to PGPTXT for horizontal strings. Text is drawn
!c using the current values of attributes color-index, line-width,
!c character-height, and character-font.  Text is NOT subject to
!c clipping at the edge of the window.
!c X      (input)  : world x-coordinate.
!c Y      (input)  : world y-coordinate. The string is drawn with the
!c                   baseline of all the characters passing through
!c                   point (X,Y); the positioning of the string along
!c                   this line is controlled by argument FJUST.
!c ANGLE  (input)  : angle, in degrees, that the baseline is to make
!c                   with the horizontal, increasing counter-clockwise
!c                   (0.0 is horizontal).
!c FJUST  (input)  : controls horizontal justification of the string.
!c                   If FJUST = 0.0, the string will be left-justified
!c                   at the point (X,Y); if FJUST = 0.5, it will be
!c                   centered, and if FJUST = 1.0, it will be right
!c                   justified. [Other values of FJUST give other
!c                   justifications.]
!c TEXT   (input)  : the character string to be plotted.
!      return 
!      end
c---------------------------------------------------------------------
      



