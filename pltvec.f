
c
c
      subroutine pltvec(lefct)
      implicit integer (i-n), real*8 (a-h,o-z)

c...................................................................
c     Plots fluxes with arrows..
c...................................................................

      save
      include 'param.h'
      include 'comm.h'

      character*8 target
      REAL*4 RILIN
      REAL*4 RPG1
      REAL*4 RPGX(2),RPGY(2)
      REAL*4 :: R40=0.,R4P2=.2,R4P25=.25,R4P5=.5
      REAL*4 :: R4P8=.8,R4P65=.65,R4P9=.9

      real*8 xh(jpxy,ipxy),yh(jpxy,ipxy) !YuP[2020-12-17] added working array,
         ! to avoid overwriting xtail,ytail arrays

      include 'advnce.h'


c...................................................................
c     vector plot for fluxes (x-par,x-perp) coordinates
c...................................................................

cBH011228 Modifications for plotting with PGPLOT,  011228.

      if (noplots.eq."enabled1") return

      veclnth0=2./jpxy

      do 190 k=1,ngen
         if (tandem.eq."enabled" .and. k.eq.kionn) then
            xll=-xlwr
            xlu=xlwr
            xpl=0.
            xpu=xlwr
            veclen=xlwr*veclnth0*veclnth
            xmaxq=xlwr
            target="ionmesh"
         else
            xll=-xmax
            xlu=xmax
            xpl=0.
            xpu=xmax
            veclen=xmax*veclnth0*veclnth
            xmaxq=xmax
            target="mainmesh"
         endif
         
c     If pltlim.ne."disabled", then plot in xpar,xprp-space up
c     to appropriate limit:
         if (pltlim.ne."disabled") then
            if (pltlim.eq.'x') then
               pltlimmm=pltlimm
            elseif (pltlim.eq.'u/c') then
               pltlimmm=pltlimm*cnorm
            elseif (pltlim.eq.'energy') then
               pltlimmm=sqrt((1.+pltlimm/restmkev)**2-1.)*cnorm
            endif
            xll=-pltlimmm
            xlu=pltlimmm
            xpl=0.
            xpu=pltlimmm
            veclen=pltlimmm*veclnth0*veclnth
            xmaxq=pltlimmm
         endif
         
cBH060411         CALL PGSVP(.2,.8,.25,.55)
         CALL PGSVP(R4P2,R4P8,R4P25,R4P5)
         RILIN=5.
c     
         call bcast(da,zero,iyjxp1)
         call bcast(db,zero,iyjxp1)
         call bcast(dc,zero,iyjxp1)
         call bcast(dd,zero,iyp1jx)
         call bcast(de,zero,iyp1jx)
         call bcast(df,zero,iyp1jx)
         if (lefct .eq. 1) go to 50    !collisions
         if (lefct .eq. 2) go to 60    !electric field
         if (lefct .eq. 3) go to 80    !rf
         if (lefct .eq. 4) go to 100   !total
 50      call coeffpad(k)
         CALL PGPAGE
         write(t_,200) k
         CALL PGMTXT('B',RILIN,R40,R40,t_)
         go to 120
 60      if (abs(elecfld(lr_)) .lt. 1.e-09) go to 190
         call coefefad(k)
         
c$$$         write(*,*)'pltvec:((da(i,j),i=46,50),j=1,20)',
c$$$     +                    ((da(i,j),i=46,50),j=1,20)
c$$$         write(*,*)'pltvec:((db(i,j),i=46,50),j=1,20)',
c$$$     +                    ((db(i,j),i=46,50),j=1,20)
c$$$         write(*,*)'pltvec:((dc(i,j),i=46,50),j=1,20)',
c$$$     +                    ((dc(i,j),i=46,50),j=1,20)
c$$$         write(*,*)'pltvec:((dd(i,j),i=46,50),j=1,20)',
c$$$     +                    ((dd(i,j),i=46,50),j=1,20)
c$$$         write(*,*)'pltvec:((de(i,j),i=46,50),j=1,20)',
c$$$     +                    ((de(i,j),i=46,50),j=1,20)
c$$$         write(*,*)'pltvec:((df(i,j),i=46,50),j=1,20)',
c$$$     +                    ((df(i,j),i=46,50),j=1,20)

         CALL PGPAGE
         write(t_,210) k
         CALL PGMTXT('B',RILIN,R40,R40,t_)
         go to 120
 80      continue ! Flux from RF only
         xrf=0.
         if (n .lt. nonrf(k) .or. n .ge. noffrf(k)) go to 90
         call coefrfad(k,xrf)
 90      continue
         if (xrf.gt.0.) then
            CALL PGPAGE
            write(t_,220) k
            CALL PGMTXT('B',RILIN,R40,R40,t_)
         endif
         go to 120
 100     continue ! Total flux
         call coefstup(k)
         CALL PGPAGE
         write(t_,230) k
         CALL PGMTXT('B',RILIN,R40,R40,t_)
         !write(t_,231) !YuP[2020-10-26] Moved it down, avail. for each type of plot
         !RILIN=RILIN+1.
         !CALL PGMTXT('B',RILIN,R40,R40,t_)
         
 120     continue ! exit/continue handle
         
         !YuP[2020-10-26] Add this text on each type of flux plot:
         write(t_,231) !"(bottom linear, top logrithmic length of flux vector)"
         RILIN=RILIN+1.
         CALL PGMTXT('B',RILIN,R40,R40,t_)

         RILIN=RILIN+2.
         write(t_,398) n,timet,rovera(lr_)
         CALL PGMTXT('B',RILIN,R40,R40,t_)
         
 200     format("species no.",i2,5x,"Flux Due to Coulomb Collisions")
 210     format("species no.",i2,5x,"Flux Due to Electric Field")
 220     format("species no.",i2,5x,"Flux Due to RF Diffusion")
 230     format("species no.",i2,5x,"Total Flux in Velocity Space")
 231     format("(bottom linear, top logrithmic length of flux vector)")
         
 398     format("n=",i4,3x,"time=",1pe13.6," secs","   r/a=",1pe10.4)
         
         
c...................................................................
c     The coefficients of the equation are currently defined on the
c     same mesh as the distribution function f. The fluxes are best
c     defined (from the point of view of differencing and particle
c     conservation) on mid meshpoints. We undertake here to
c     interpolate the coefficients as needed to either (i,j+1/2)
c     (velocity flux) or to (i+1/2,j) (theta flux).
c     Finally to enforce boundary conditions (zero flux in general
c     except at the pass/trapped boundary) certain coefficients
c     are zeroed out or suitably averaged at specific mesh points.
c     The numbers 1,2,3 appearing in the calls below signify
c     which coefficient is being treated.
c     
c     first the velocity flux coefficients for gfi..
c...................................................................
         
         call coefmidv(da,1)
         call coefmidv(db,2)
         call coefmidv(dc,3)
        
c...................................................................
c     the theta flux coefficients for hfi..
c...................................................................
         
         call coefmidt(dd,1)
         call coefmidt(de,2)
         call coefmidt(df,3)

c$$$         if (lefct.eq.2) then
c$$$         write(*,*)'pltvec:((da(i,j),i=46,50),j=1,20)',
c$$$     +                    ((da(i,j),i=46,50),j=1,20)
c$$$         write(*,*)'pltvec:((db(i,j),i=46,50),j=1,20)',
c$$$     +                    ((db(i,j),i=46,50),j=1,20)
c$$$         write(*,*)'pltvec:((dc(i,j),i=46,50),j=1,20)',
c$$$     +                    ((dc(i,j),i=46,50),j=1,20)
c$$$         write(*,*)'pltvec:((dd(i,j),i=46,50),j=1,20)',
c$$$     +                    ((dd(i,j),i=46,50),j=1,20)
c$$$         write(*,*)'pltvec:((de(i,j),i=46,50),j=1,20)',
c$$$     +                    ((de(i,j),i=46,50),j=1,20)
c$$$         write(*,*)'pltvec:((df(i,j),i=46,50),j=1,20)',
c$$$     +                    ((df(i,j),i=46,50),j=1,20)
c$$$         endif


         if (lefct .eq. 3 .and. xrf .eq. 0) go to 190
         call bcast(temp5(0,0),zero,iyjx2)
         call bcast(temp4(0,0),zero,iyjx2)
c        In the following, tam2 and tam3 are the code fluxes Gamma_x and
c          sin(theta)*Gamma_theta interpolated onto the code mesh
c          x,theta.  temp5 and temp4 are the parallel and perpendicular
c          components of Gamma, respectively, on the x,theta-mesh.
         if (implct .eq. "enabled") then
            do 140 i=2,iy_(l_)-1 !YuP[2021-03-12] iy-->iy_(l_) in many places
               do 141 j=2,jxm1
                  tam2(j)=-(gfi(i,j,k)+gfi(i,j-1,k))*.5/xsq(j)
                  tam3(j)=-(hfi(i,j)+hfi(i-1,j))*.5*xi(j)
                  temp5(i,j)=tam2(j)*coss(i,l_)-tam3(j)
                  temp4(i,j)=tam2(j)*sinn(i,l_)+tam3(j)/tann(i,l_)
 141           continue
 140        continue

c$$$         write(*,*)'pltvec:((temp5(i,j),i=46,50),j=1,20)',
c$$$     +                    ((temp5(i,j),i=46,50),j=1,20)
c$$$         write(*,*)'pltvec:((temp4(i,j),i=46,50),j=1,20)',
c$$$     +                    ((temp4(i,j),i=46,50),j=1,20)
            


         else
            temp1(0:iy_(l_)+1,0:jx+1)=f_(0:iy_(l_)+1,0:jx+1,k,l_)
            temp2(0:iy_(l_)+1,0:jx+1)=fxsp(0:iy_(l_)+1,0:jx+1,k,l_)
            do 240 j=2,jxm1
               do 241 i=2,iy_(l_)-1
                  temp6(i,j)=-(gfu(i,j,k)+gfu(i,j-1,k))*.5/xsq(j)
 241           continue
 240        continue
            call dcopy(iyjx2,temp2(0,0),1,temp1(0,0),1)
            call dcopy(iyjx2,f(0,0,k,l_),1,temp2(0,0),1)
            do 242 i=2,iy_(l_)-1
               do 243 j=2,jxm1
                  tam3(j)=-(hfu(i,j)+hfu(i-1,j))*.5*xi(j)
                  temp5(i,j)=temp6(i,j)*coss(i,l_)-tam3(j)
                  temp4(i,j)=temp6(i,j)*sinn(i,l_)+tam3(j)/tann(i,l_)
 243           continue
 242        continue
         endif
         do 244 j=1,jx
            temp5(itl,j)=temp5(itl-1,j)
            temp5(itu,j)=temp5(itu+1,j)
            temp4(itl,j)=temp4(itl-1,j)
            temp4(itu,j)=temp4(itu+1,j)
 244     continue
         
c     
c     Above gives flux flux_par(u,theta) ~ temp5, flux_perp ~ temp4.
c     The following calls to prppr and dcopy put 
c     flux_par into xhead, flux_perp into yhead, on an x,y-grid.

c      write(*,*)'pltvec:  lr_,lefct =',lr_,lefct
c      write(*,*)'pltvec:  temp4, temp5',
c     +     ((temp4(i,j),temp5(i,j),i=1,10),j=1,10)

         call dcopy(iyjx2,temp5(0,0),1,temp3(0,0),1) !temp5-->temp3

         call prppr(target,"norm",xll,xlu,xpl,xpu) !uses temp3; makes fpn

cBH090226         call dcopy(iyjx2,temp2(0,0),1,temp1,1)
         ipxjpx=jpxy*ipxy
         call dcopy(ipxjpx,fpn,1,xhead,1)

         call dcopy(iyjx2,temp4(0,0),1,temp3(0,0),1)

         call prppr(target,"norm",xll,xlu,xpl,xpu) !uses temp3; makes fpn

cBH090226          call dcopy(iyjx2,temp2(0,0),1,temp4,1)
         call dcopy(ipxjpx,fpn,1,yhead,1)
         
         do 150 i=1,ipxy
            do 151 j=1,jpxy
               xtail(j,i)=xpar(j)
               ytail(j,i)=xperp(i)
 151        continue
 150     continue
         xh(:,:)=xhead(:,:) !YuP[2020-12-17] Save. xhead can be adjusted below
         yh(:,:)=yhead(:,:) !YuP[2020-12-17] Save. yhead can be adjusted below
         !write(*,*)'pltvec: jpxy,xpar=',jpxy,(xpar(j),j=1,jpxy)
         !write(*,*)'pltvec: ipxy,xperp=',ipxy,(xperp(i),i=1,ipxy)


         !------- LOWER plot (linear scale for arrows) --------------
         RPG1=xmaxq
         CALL PGSWIN(-RPG1,RPG1,R40,RPG1)
         CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
         if (pltlim.eq.'u/c') then
            write(t_,10184) 
         elseif (pltlim.eq.'energy') then
            write(t_,10185) 
         else
            write(t_,10186) 
         endif
10184    format("u/c")
10185    format("energy (keV)")
10186    format("u/unorm")
         CALL PGLAB('Parallel '//t_,'Perp '//t_,' ')
c     Plot vector field, vector lengths proportional to |flux|:
         
c      write(*,*)'pltvec:  lr_,lefct =',lr_,lefct
c      do jp=1,jpxy
c      do ip=1,ipxy
c         write(*,*)'pltvec: jp,ip,xhead,yhead:',
c     +                 jp,ip,xtail(jp,ip),ytail(jp,ip)
c         write(*,*)'pltvec: jp,ip,xhead,yhead:',
c     +                 jp,ip,xhead(jp,ip),yhead(jp,ip)
c      enddo
c      enddo
c         write(*,*)'pltvectr lin: n,lr_,lefct=',n,lr_,lefct

         !YuP[2020-12-17] Trying some adjustments for arrow length,
         ! for linear-scale plot:
!         do i=1,ipxy !perp index
!         do j=1,jpxy ! par index
!           if(l_.eq.1 .and. i.eq.2)then
!             rheads(j)=sqrt(xhead(j,i)**2+yhead(j,i)**2)
!             write(*,'(a,3e11.3)')'xhead,yhead,rheads(j)',
!     &        xhead(j,i),yhead(j,i),rheads(j)
!           endif
!           !xhead(j,i)=min(xhead(j,i),1.d23)
!           !xhead(j,i)=max(xhead(j,i),-1.d23)
!           !yhead(j,i)=min(yhead(j,i),1.d23)
!           !yhead(j,i)=max(yhead(j,i),-1.d23)
!         enddo
!         enddo
         
         call pltvectr(xtail,ytail,xhead,yhead,jpxy,ipxy,veclen,
     +                noplots)  ! here: LOWER plot - lin.scale
     
         xhead(:,:)=xh(:,:) !YuP[2020-12-17] restore. xhead could be adjusted above
         yhead(:,:)=yh(:,:) !YuP[2020-12-17] restore. yhead could be adjusted above
     
         !---> Plot trap-pass boundary
         t0t=sin(thb(l_))/cos(thb(l_))
         if (t0t .lt. 1.) then
            RPGX(1)=0.
            RPGY(1)=0.
cBH060411            RPGX(2)=xmaxq*t0t
            RPGX(2)=xmaxq
cBH060411            RPGY(2)=xmaxq
            RPGY(2)=xmaxq*t0t
            CALL PGLINE(2,RPGX,RPGY)
cBH060411            RPGX(2)=-xmaxq*t0t
            RPGX(2)=-xmaxq
            CALL PGLINE(2,RPGX,RPGY)
         else
            RPGX(1)=0.
            RPGY(1)=0.
            RPGX(2)=xmaxq/t0t
            RPGY(2)=xmaxq
            CALL PGLINE(2,RPGX,RPGY)
            RPGX(2)=-xmaxq/t0t
            CALL PGLINE(2,RPGX,RPGY)
         endif


         !------- TOP plot (log() scale for arrows) --------------
         rhmin=ep90
         rhmax=-ep90
         do i=1,ipxy
            do j=1,jpxy
               rheads(j)=sqrt(xhead(j,i)**2+yhead(j,i)**2)+em90
            enddo
            call aminmx(rheads,1,jpxy,1,rhminsww,rhmaxsww,kmin,kmax)
            rhmin=min(rhmin,rhminsww)
            rhmax=max(rhmax,rhmaxsww)
         enddo
         if (rhmin.lt.rhmax*contrmin) rhmin=rhmax*contrmin
         rhscale=log(rhmax)-log(rhmin)
         do 170 i=1,ipxy
              do 171 j=1,jpxy
                  rhead=em90+sqrt(xhead(j,i)**2+yhead(j,i)**2)
                  rhlog=log(rhead)+rhscale
                  if (rhlog.lt.0.) rhlog=0.
                 xhead(j,i)=xhead(j,i)*rhlog/rhead
                 yhead(j,i)=yhead(j,i)*rhlog/rhead                 
 171          continue
 170     continue

cBH060411           CALL PGSVP(.2,.8,.6,.9)
           CALL PGSVP(R4P2,R4P8,R4P65,R4P9)
           RPG1=xmaxq
           CALL PGSWIN(-RPG1,RPG1,R40,RPG1)
           CALL PGBOX('BCNST',R40,0,'BCNST',R40,0)
           if (pltlim.eq.'u/c') then
              write(t_,10184) 
           elseif (pltlim.eq.'energy') then
              write(t_,10185) 
           else
              write(t_,10186) 
           endif
           CALL PGLAB(' ','Perp '//t_,' ')
c     Plot vector field, vector lengths proportional to log(abs(flux)),
c     from max(abs(flux)) to contrmin*max(abs(flux)):
           call pltvectr(xtail,ytail,xhead,yhead,jpxy,ipxy,
     +          veclen,noplots) !here: Top plot: log scale for arrows

           xhead(:,:)=xh(:,:) !YuP[2020-12-17] restore. xhead could be adjusted above
           yhead(:,:)=yh(:,:) !YuP[2020-12-17] restore. yhead could be adjusted above
     
           !---> Plot trap-pass boundary
           t0t=sin(thb(l_))/cos(thb(l_))
           if (t0t .lt. 1.) then
              RPGX(1)=0.
              RPGY(1)=0.
cBH170721            RPGX(2)=xmaxq*t0t
              RPGX(2)=xmaxq
cBH170721            RPGY(2)=xmaxq
              RPGY(2)=xmaxq*t0t
              CALL PGLINE(2,RPGX,RPGY)
cBH170721            RPGX(2)=-xmaxq*t0t
              RPGX(2)=-xmaxq
              CALL PGLINE(2,RPGX,RPGY)
           else
              RPGX(1)=0.
              RPGX(2)=xmaxq/t0t
              RPGY(1)=0.
              RPGY(2)=xmaxq
              CALL PGLINE(2,RPGX,RPGY)
              RPGX(2)=-xmaxq/t0t
              CALL PGLINE(2,RPGX,RPGY)
           endif
           
 190  continue ! k=1,ngen
      return
      end subroutine pltvec
      
      
