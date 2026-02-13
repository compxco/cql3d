c
c
      subroutine tdtrdfus ! CQL3D only (not CQLP)
      implicit integer (i-n), real*8 (a-h,o-z)

c..............................................................
c     This routine computes the radial diffusion coefficients as
c     a function of mu or y and of rho (radius) - at a
c     given v and k (species index).
c     Present possibilities:  see cqlinput_help
c     There are three general possibilities:
c        1) Simplified ion neoclassical model taking pitch
c           angle variation uniform (should actually be strongly peaked
c           at tr-pass bndry) and proportional to 1/vel above vth_ion.
c        2) User specified Drr giving phenomenological radius and
c           velocity dependences as specified in cqlinput_help.
c        3) Read in a .nc file (named drr_in) which contains the diffusion
c           coefficient (d_rr) on a compatible v,theta,rho grid.
c    Alternatively, a .nc file is created (drr_out_<mnenomic>) containing 
c    a d_rr computed in this subroutine and the grids. 
c    (Also, d_r advection is subsequently added.)
c..............................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE
c%OS
      dimension drshape(0:lrz), rhotr(0:lrz)

      if (transp.eq."disabled") return

      call bcast(d_rr,zero,iyjx2*ngen*(lrz+1))
      call bcast(drshape(0),zero,lrz+1)

      if (read_data.eq.'nimrod') then !YuP[2021-08] added.
        !In this case the values of d_rr() are set in
        !subr.profiles, at every time step, using time-dependent data on dB/B
        !from NIMROD (in future, other codes for coupling could be added)
        return ! Nothing else to do here?
      endif
      
c      write(*,*)'tdtrdfus: difus_type(),',(difus_type(k),k=1,ngen)

      if (difus_io(1).ne."drrin") then

      do k=1,ngen

      if ( difus_type(k) .eq. "neo_smpl"
     1        .or. difus_type(k) .eq. "neo_plus"
     1        .or. difus_type(k) .eq. "neo_trap"
     1        .or. difus_type(k) .eq. "neo_trpp" ) then
         
      do l=0,lrz-1
         ilr=lrindx(l)+1        !Need to check why index is 0,lrz-1 below.
                                !BH051101: bndry condition is zero flux at
                                !rho=0, and Maxwl distn at rho=a.
                                !Zero flux at rho=0 is implemented with
                                !Drr=0 at l=0.  See McCoy notes, 10/12/90,
                                !pp. 8-10.
         vth_cnst=3.*sqrt(2.)*vth(kionn,ilr) !Matching v-dependence in tdrmshst
         rooteps=eps(ilr)**0.5  !This could cause too large diff as eps==>0.
         call tdnflxs(ilr)
         do j=1,jx

            if (x(j)*vnorm .lt. vth_cnst) then 
               do i=1,iymax
                  d_rr(i,j,k,ilr)=drr_gs(ilr)
               enddo
            else
               do i=1,iymax
                  d_rr(i,j,k,ilr)=drr_gs(ilr)*vth_cnst/(x(j)*vnorm)
               enddo
            endif

            if (difus_type(k) .eq. "neo_trap") then
               do i=1,itl-1
                  d_rr(i,j,k,ilr)=0.
               enddo
               do i=itl,itu
                  d_rr(i,j,k,ilr)=d_rr(i,j,k,ilr)/rooteps
               enddo
               do i=itu+1,iy
                  d_rr(i,j,k,ilr)=0.
               enddo
            endif
              
         enddo ! j
c     c        Checking calc of d_rr vel dependence against seperate 
c     c        tdtrdfus calc of d_rr_b  ==> OK
         eighty=80.             ! getting consistent precision
         jkev=luf(eighty,enerkev(1:jx,k),jx) !YuP[2020-10-20] corrected: index k
         jkev=jkev-1
         
         if (ioutput(1).ge.2) then !YuP[2020] diagnostic printout
           if ( l.eq.0) then
            write(*,*)'tdtrdfus: enerkev',
     +       (enerkev(j,k),j=1,jx) !YuP[2018-01-08] added 2nd index (k)
            write(*,*)'tdtrdfus: d_rr',(d_rr(iyh,j,k,ilr),j=1,jx)
            write(*,*)
     &       'tdtrdfus: jkev,d_rr(iyh,1,k,ilr),d_rr(iyh,jkev,k,ilr) '
     1       ,jkev,d_rr(iyh,1,k,ilr),d_rr(iyh,jkev,k,ilr)
           endif
         endif
              
      enddo  !  on l

      endif  ! on difus_type

      if (difus_type(k) .eq. "specify"
     1        .or. difus_type(k) .eq. "neo_plus"
     1        .or. difus_type(k) .eq. "neo_trpp") then

c     Test for radial diffusion coeff cnst in velocity.
      icnst=0
      do ii=1,4
         if (difus_vshape(ii).ne.zero) icnst=1
      enddo

c   check if difin non-zero to test if should be used
      rsum = 0.
      do ii = 1,njene
        rsum = rsum+abs(difin(ii))
      enddo
c   mesh to compute drshape on from difin(ryain)
      do ii=0,lrz
        rhotr(ii) = rrz(ii)/radmin
      enddo

      rdefr = -1.
      if (rsum.gt.0.) then
        rdefr = 1.
        call ryaintorz(njene,ryain(1),difin(1),lrz,rhotr(0),drshape(0))
      endif

      do l=0,lrz-1
         ilr=lrindx(l)
         if (l.eq.0) then   !lrindx(0)=0
            difusr1=0.
            rshape=0.
            rshape1=0.
         else
            difusr1=difusr
            if (rdefr.lt.0.5) then
              rshape=tdtrrshape(ilr)
            else
              rshape=drshape(ilr)
            endif
            if (difus_rshape(8).ne.zero. and. 
     1                   difus_type(k) .eq. "neo_plus") then
               rshape1=tdtrrshape1(ilr)
            else
               rshape1=0.
            endif
         endif
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,'(a,1p3e12.3)')'tdtrdfus, rshape,rshape1,difusr1:',
     &                                     rshape,rshape1,difusr1
CMPIINSERT_ENDIF_RANK
         if (l.ne.0) call tdtrvshape(k,l) !Returns shape in temp1
         do j=1,jx
            if (l.eq.0 .or. icnst.eq.0) then
               do i=1,iymax
cBH050921 Substantial bug fix, adding  *rshape     d_rr(i,j,k,l)=difusr1
                  d_rr(i,j,k,l)=d_rr(i,j,k,l)
     1                        + difusr1*rshape
     1                        + difusr1*rshape1
               enddo
            else
               do i=1,iytr(l)
                  d_rr(idx(i,l),j,k,l)=d_rr(idx(i,l),j,k,l)
     1                 + difusr1*rshape*temp1(idx(i,l),j)
     1                 + difusr1*rshape1
               enddo
            endif
         enddo  !On j

CMPIINSERT_IF_RANK_EQ_0
        !WRITE(*,*)'tdtrdfus:l,(temp1(1,j),j=1,jx)',l,(temp1(1,j),j=1,jx)
        do j=1,jx,60
        do i=1,iyh,8
        WRITE(*,*)'tdtrdfus: l,j,i, d_rr(i,j,l)', l,j,i, d_rr(i,j,k,l)
        enddo
        enddo
CMPIINSERT_ENDIF_RANK

       enddo  ! on l

      endif   ! on difus_type
      enddo   ! on k

c$$$  This will be done at end of code, after final calc of d_r.
c$$$      elseif (difus_io(1).eq."drrout") then
c$$$
c$$$         call diffus_io(1)  !Writes d_rr for k=1 to .nc file
c$$$
c$$$      elseif (difus_io(1).eq."drrdrout") then
c$$$
c$$$         call diffus_io(2)  !Writes d_rr for k=1 to .nc file
c$$$
c$$$  This will be done a beginning of the code, as alternative to
c$$$  call tdtrdfus.
c$$$      elseif (difus_io(1).eq."drrin") then
c$$$
c$$$         call diffus_io(3)  !Read d_rr for k=1 to .nc file
c$$$
c$$$      elseif (difus_io(1).eq."drrdrin") then
c$$$
c$$$         call diffus_io(4) !Reads d_rr and d_r for k=1 from .nc file
         
      endif  !On difus_io(1).ne."drrin"

      return
      end subroutine tdtrdfus
c
c
      real*8 function tdtrrshape(lr)
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     This routine computes additional radial shape function for  
c     the radial diffusion coefficient.
c     Coefficients in the following expression are input
c     throught the namelist variable difus_rshape(1:7).
c.......................................................................

      include 'param.h'
      include 'comm.h'
c 
      if (lr.eq.0) then

         tdtrrshape=0.

      elseif (lr.le.lrzmax-1) then
         
         if (kelecm.ne.0) then  !kelecm=0 when colmodl=1
            kk=kelecm
         else
            kk=kelecg  
         endif
         
         tdtrrshape=(difus_rshape(1)+difus_rshape(2)*
     +              (rrz(lr)/radmin)**difus_rshape(3))**difus_rshape(4)
         tdtrrshape=tdtrrshape*(0.5*(reden(kk,lr)+reden(kk,lr+1))/
     +              reden(kk,0))**difus_rshape(5)
         tdtrrshape=tdtrrshape*(0.5*(temp(kk,lr)+temp(kk,lr+1))/
     +              temp(kk,0))**difus_rshape(6)
         tdtrrshape=tdtrrshape*(0.5*(zeff(lr)+zeff(lr+1))/
     +              zeff(1))**difus_rshape(7)

      else

         tdtrrshape=(difus_rshape(1)+difus_rshape(2)*
     +              (rrz(lr)/radmin)**difus_rshape(3))**difus_rshape(4)
c        Simply use last (lrzmax) value of density....
         tdtrrshape=tdtrrshape*
     +              (reden(kk,lr)/reden(kk,0))**difus_rshape(5)
         tdtrrshape=tdtrrshape*
     +              (temp(kk,lr)/temp(kk,0))**difus_rshape(6)
         tdtrrshape=tdtrrshape*
     +              (zeff(lr)/zeff(1))**difus_rshape(7)

      endif

      return
      end function tdtrrshape
c
c
      real*8 function tdtrrshape1(lr)
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     This routine computes a radial shape function for  
c     the radial diffusion coefficient.
c     Coefficients in the following expression are input
c     throught the namelist variable difus_rshape(1:7).
c.......................................................................

      include 'param.h'
      include 'comm.h'
c 
      if (lr.eq.0) then

         tdtrrshape1=0.

      elseif (lr.le.lrzmax-1) then

         ravg=0.5*(rya(lr)+rya(lr+1))

         if (ravg.le.difus_rshape(8)) then
            tdtrrshape1=1.
         elseif (ravg.le.(1.1*difus_rshape(8))) then
            tdtrrshape1=0.5*(1.+cos(pi*(ravg-difus_rshape(8))/
     1                                  (0.1*difus_rshape(8))))
         else
            tdtrrshape1=0.
         endif

      else
         
         ravg=rya(lr)

         if (ravg.le.difus_rshape(8)) then
            tdtrrshape1=1.
         elseif (ravg.le.1.1*difus_rshape(8)) then
           tdtrrshape1=0.5*(1.+cos(pi*(ravg-difus_rshape(8))/
     1                                 (0.1*difus_rshape(8))))
         else
            tdtrrshape1=0.
         endif
         
      endif

      return
      end function tdtrrshape1
c
c
      subroutine tdtrvshape(k,l)
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     This routine computes a velocity-space shape function 
c     for the radial diffusion coefficient.
c     Coefficients in the following expression are input
c       throught the namelist variable difus_vshape(1:4).
c     The shape function is stored in temp1.
c     BH180921:
c     Note that apart from the coll_cutoff and gamm factor,
c     that the velocity dependence is in terms of velocity
c     (not momentum-per-mass) normalized to central thermal
c     velocity vth(k,1), with value 1.0 at the thermal vel.
c.......................................................................

      include 'param.h'
      include 'comm.h'

      real*8 l_autocorr,lambda_mfp

      lr=lrindx(l)
      l_autocorr=pi*qsafety(lr)*radmaj

      call bcast(temp1(0,0),zero,iyjx2)
c      write(*,*)'tdtrdfus:difus_vshape',difus_vshape
      do  j=1,jx
         vel=x(j)*vnorm/gamma(j)
         coll=coll_freq(vel,k,lr)
         gamm=gamma(j)**difus_vshape(4)
         do  i=1,iytr(l)
            vpar=abs(vel*coss(idx(i,l),l))
            vprp=vel*sinn(idx(i,l),l)
            lambda_mfp=max(em100,vpar/coll)
            coll_cutoff=(1.+l_autocorr/lambda_mfp)**difus_vshape(2)
            vpar=vpar/vth(k,1)
            vprp=vprp/vth(k,1)
            vpar=vpar**difus_vshape(1)
            vprp=vprp**difus_vshape(3)
c      write(*,*)'tdtrvshape, l_autocorr,lambda_mfp,coll,vpar,vprp',
c     +                       l_autocorr,lambda_mfp,coll,vpar,vprp
            temp1(idx(i,l),j)=vpar/coll_cutoff*vprp/gamm
c      write(*,*)'tdtrvshape,idx(i,l),i,j,l,temp1(idx(i,l),j),coll_cut',
c     +                     idx(i,l),i,j,l,temp1(idx(i,l),j),coll_cutoff
             
         enddo
      enddo

      return
      end subroutine tdtrvshape




c 
c
      real*8 function coll_freq(vel,k,lr)
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     This routine computes 
c     Approx. electron or ion collision freq from NRL 
c     perp. deflection time.
c     Relativistic effects are neglected, since this
c     term is only significant in tdtrvshape for lambda_mfp
c     less than the magnetic fluctuation autocorrelation length,
c     generally for velocites much less that vth.
c.......................................................................

      include 'param.h'
      include 'comm.h'

      real*8 lnlambda

c     Approx. electron or ion collision freq from NRL 
c     perp. deflection time

      if (vel.eq.zero) then
         coll_freq=ep100
         return
      endif


      lnlambda=24.-log(reden(kelec,lr)**0.5/temp(kelec,lr))
      !YuP[2022-02-18] There is a bug in the above line -
      !It almost matches the expression for lambda_ei in NRL
      ! (at Te > 10*Z**2) except the temperature should be in eV.
      !But in CQL3D, temp() array is in keV.

      if (k.eq.kelecg) then

         energyev=0.5*fmass(kelec)*vel**2/ergtkev*1000.
         tempev=temp(kelec,lr)*1000.
         fnu_perp1=5.8e-6/(tempev**0.5*energyev)
         fnu_perp2=7.7e-6/energyev**1.5
         coll_freq1=min(fnu_perp1, fnu_perp2)*reden(k,lr)*lnlambda

         tempev=temp(kionn,lr)*1000.
         fmu=fmass(kionn)/proton
         fnu_perp1=2.5e-4*fmu**0.5/(tempev**0.5*energyev)
         fnu_perp2=7.7e-6/energyev**1.5
         coll_freq2=min(fnu_perp1, fnu_perp2)*
     +             reden(kelec,lr)*zeff(lr)*lnlambda

         coll_freq=coll_freq1+coll_freq2

      else                                           !ion case

         energyev=0.5*fmass(kionn)*vel**2/ergtkev*1000.
         tempev=temp(kionn,lr)*1000.
         fmu=fmass(kionn)/proton
         fnu_perp1=1.4e-7/(fmu**0.5*tempev**0.5*energyev)
         fnu_perp2=1.8e-7/(fmu**0.5*energyev**1.5)
         coll_freq=min(fnu_perp1, fnu_perp2)*
     +             (reden(kelec,lr)*zeff(lr)*bnumb(kionn)**2*lnlambda)

      endif
      
      return
      end function coll_freq
        
      subroutine ryaintorz(npts_in,oldx,oldf,npts,ynewx,ynewf)
      implicit integer (i-n), real*8 (a-h,o-z)

c.......................................................................
c     This routine interpolates difin vector in oldx, given in ryain 
c     in cqlinput file to the transport mesh ynewx.
c     Boundary conditions are: 
c                 1st derivative = 0 at plasma centre
c                 2nd derivative = 0 at plasma edge
c
c     At this stage uses linear interpolation
c.......................................................................

      include 'param.h'
      parameter (nwka=3*lrza+1)
      dimension work(nwka),oldx(1),oldf(1),ynewx(1),ynewf(1),i1p(2),
     1  secondd(lrza),itab(3),tab(3)

      do jj = 1,npts_in-1
        do ll = 0,npts
          if (ynewx(ll).ge.oldx(jj) .and. ynewx(ll).le.oldx(jj+1)) then
            ynewf(ll) = 
     .        oldf(jj)+(oldf(jj+1)-oldf(jj))/(oldx(jj+1)-oldx(jj))*
     .        (ynewx(ll)-oldx(jj))
          endif
        enddo
      enddo
      
      return
      end subroutine ryaintorz


      subroutine diffus_io(kopt)
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c.......................................................................
c     Write or read radial diffusion coeff and advective/pinch term,
c     and relate grids.
c     kopt=0:  Define output file & variables (called before kopt=1 or 2)
c     kopt=1:  write d_rr
c     kopt=2:  write d_rr,d_r
c     kopt=3:  read d_rr
c     kopt=4:  read d_rr,d_r
c.......................................................................

      include 'param.h'
      include 'comm.h'
      include 'netcdf.inc'

c --- some stuff for netCDF file ---
      integer ncid,vid,istatus
      integer xdim,ydim,rdim,r0dim,gdim
      integer char64dim
      character*128 name,ltitle

      integer dims(3),count1(3),start1(3)  !for ngen=1 case
      integer dimsg(4),countg(4),startg(4)  !for ngen.gt.1 case
      integer y_dims(2),y_count(2)

      real*8, allocatable :: wkpack(:) ! local working array for pack21

      data start1/1,1,1/
      data startg/1,1,1,1/

      WRITE(*,*) 'In diffus_io, kopt=',kopt

c     Maximum iy as function of radius:
      iymx=0
      do l=1,lrz
         iymx=max(iymx,iy_(l))
      enddo
      if(iymx.gt.iy) stop 'difus_io: iy_(l) should not exceed iy' 

      if (.NOT.ALLOCATED(wkpack)) then ! allocate working array for pack21
         nwkpack=iyjx2 +10 ! +10 just in case
         allocate(wkpack(nwkpack),STAT=istat)
         call bcast(wkpack,zero,SIZE(wkpack))
      endif

      if (kopt.eq.0) then  !Down to line 682

! ngen=1 case
      count1(1)=iymax
      count1(2)=jx
      count1(3)=1  !radial index

! ngen.gt.1 case
      countg(1)=iymax
      countg(2)=jx
      countg(3)=1  !radial index, 1 at at time
      countg(4)=1  !species index, 1 at at time

      y_count(1)=iymax
      y_count(2)=lrz  ! lrz radii at a time

c.......................................................................
cl    1.1.1 create netCDF filename.
c     CLOBBER old file, if it exists.
c     istatus is 0, if no errors.
c     ncid is created file id.

      if (difus_io_file.eq."mnemonic") then
         write(t_,1000) mnemonic(1:length_char(mnemonic))
 1000    format(a,"_difus_io.nc")
      else
c         write(t_,1001)
c 1001    format(a,difus_io_file)
         t_=difus_io_file
      endif

      istatus = NF_CREATE(t_, NF_CLOBBER, ncid)
      call check_err(istatus)

c     Define dimensions

      istatus= NF_DEF_DIM(ncid, 'xdim',     jx,       xdim)
      istatus= NF_DEF_DIM(ncid, 'ydim',     iymax,    ydim)
      istatus= NF_DEF_DIM(ncid, 'rdim',     lrz,      rdim)
      istatus= NF_DEF_DIM(ncid, 'r0dim', lrzmax, r0dim)
c     Number of general species treated by the _difus_io.nc file
      n_d_rr=0
      do k=1,ngen
         if (difus_io(k).ne."disabled") n_d_rr=n_d_rr+1
      enddo
      istatus= NF_DEF_DIM(ncid, 'drr_gen_species_dim', n_d_rr,  gdim)

      istatus= NF_DEF_DIM(ncid, 'char64dim',   64,    char64dim)

c     Define vectors of dimensions
      dims(1)=ydim
      dims(2)=xdim
      dims(3)=rdim

      dimsg(1)=ydim
      dimsg(2)=xdim
      dimsg(3)=rdim
      dimsg(4)=gdim

      y_dims(1)=ydim
      y_dims(2)=rdim


c     Define variables
c     Note, the variable IDs (denoted below as vid) are
c     not saved here in this subroutine; rather, the IDs
c     are retrieved from the netCDF data file, as needed,
c     by calling the netCDF routine ncvid.

      ltitle='NetCDF output/input of diffusion/pinch coeffs'
      if( length_char(ltitle).gt.128 ) stop 'Adjust ltitle in difus_io'
      call ncaptc2(ncid,NCGLOBAL,'title',NCCHAR,length_char(ltitle),
     +     ltitle,istatus)

      vid=ncvdef2(ncid,'version',NCCHAR,1,char64dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +          'CQL3D version number',istatus)

      vid=ncvdef2(ncid,'mnemonic',NCCHAR,1,char64dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,23,
     +          'Mnemonic run identifier',istatus)

c  Mesh related quantities

      vid=ncvdef2(ncid,'lrzmax',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +            'Number of radial surfaces',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'rya',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,22,
     +           'Normalized radial mesh',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'rpconz',NCDOUBLE,1,r0dim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,39,
     +           'Major radius at outside of flux surface',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'rhomax',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +           'cms',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'lrz',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,29,
     +            'Number of FPd radial surfaces',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'lrindx',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,30,
     +           'Radial indices of FPd surfaces',istatus)

      vid=ncvdef2(ncid,'jx',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +            'momentum-per-mass dimension',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'x',NCDOUBLE,1,xdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,28,
     +           'normalized momentum-per-mass',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'vnorm',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,33,
     +           'velocity (momentum-per-mass) norm',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +                     'cms/sec',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'enorm',NCDOUBLE,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,20,
     +                     'Energy normalization',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,3,
     +                     'keV',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'iy',NCLONG,0,0,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,25,
     +            'Max pitch angle dimension',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'y',NCDOUBLE,2,y_dims,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,11,
     +           'pitch angle',istatus)
      call ncaptc2(ncid,vid,'units',NCCHAR,7,
     +           'radians',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'iy_',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,36,
     +            'Pitch angle dimension at each radius',istatus)
      call check_err(istatus)

      vid=ncvdef2(ncid,'itl',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,26,
     +           'lower trapped-passing bndy',istatus)

      vid=ncvdef2(ncid,'itu',NCLONG,1,rdim,istatus)
      call ncaptc2(ncid,vid,'long_name',NCCHAR,26,
     +           'upper trapped-passing bndy',istatus)
      call check_err(istatus)

c  d_rr diffusion coeff, and d_r pinch (kopt.eq.2)
c  Could use ngen.gt.1 coding for ngen=1 case, if turns out preferable.

c  n_d_rr is the number of diffused general species

      if (n_d_rr.eq.1) then
         vid=ncvdef2(ncid,'d_rr',NCDOUBLE,3,dims,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +        'radial diffusion coefficient',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,8,
     +        'cm**2/sec',istatus)
         !Define d_r pinch velocity if difus_io(1).eq."drrdrout"
         if (difus_io(1).eq."drrdrout") then
            vid=ncvdef2(ncid,'d_r',NCDOUBLE,3,dims,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,38,
     +           'radial vel (pinch) term (pos, outwards)',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,6,
     +           'cm/sec',istatus)
         endif
      endif

      if (n_d_rr.gt.1) then
         vid=ncvdef2(ncid,'d_rr',NCDOUBLE,4,dimsg,istatus)
         call ncaptc2(ncid,vid,'long_name',NCCHAR,27,
     +        'radial diffusion coefficient',istatus)
         call ncaptc2(ncid,vid,'units',NCCHAR,8,
     +        'cm**2/sec',istatus)
         !Define d_r pinch velocity if difus_io(1).eq."drrdrout"
         if (difus_io(1).eq."drrdrout") then
            vid=ncvdef2(ncid,'d_r',NCDOUBLE,4,dimsg,istatus)
            call ncaptc2(ncid,vid,'long_name',NCCHAR,38,
     +           'radial vel (pinch) term (pos, outwards)',istatus)
            call ncaptc2(ncid,vid,'units',NCCHAR,6,
     +           'cm/sec',istatus)
         endif
      endif

c  End define mode
      istatus= NF_ENDDEF(ncid)

      endif  !On kopt.eq.0

c  Close file

      istatus= NF_CLOSE(ncid)



c.......................................................................      
c  Write data  (assuming prior call with kopt=0)
c.......................................................................      

      if (kopt.eq.1 .or. kopt.eq.2) then   !Down to 818

      istatus=NF_OPEN(t_, NF_WRITE, ncid)

c  Write version, mnemonic,and grid related quantities

      istatus= NF_INQ_VARID(ncid,'version',vid)
      ll=length_char(version)
      call ncvptc2(ncid,vid,1,ll,version,ll,istatus)

      istatus= NF_INQ_VARID(ncid,'mnemonic',vid)
      ll=length_char(mnemonic)
      call ncvptc2(ncid,vid,1,ll,mnemonic,ll,istatus)


      istatus= NF_INQ_VARID(ncid,'lrzmax',vid)
      call ncvpt_int2(ncid,vid,1,1,lrzmax,istatus)

      istatus= NF_INQ_VARID(ncid,'rya',vid)
      call ncvpt_doubl2(ncid,vid,1,lrzmax,rya(1),istatus)

      istatus= NF_INQ_VARID(ncid,'rpconz',vid)
      call ncvpt_doubl2(ncid,vid,1,lrzmax,rpconz(1),istatus)

      istatus= NF_INQ_VARID(ncid,'rhomax',vid)
      call ncvpt_doubl2(ncid,vid,1,1,rhomax,istatus)

      istatus= NF_INQ_VARID(ncid,'lrz',vid)
      call ncvpt_int2(ncid,vid,1,1,lrz,istatus)

      istatus= NF_INQ_VARID(ncid,'lrindx',vid)
      call ncvpt_int2(ncid,vid,1,lrz,lrindx(1),istatus)

      istatus= NF_INQ_VARID(ncid,'jx',vid)
      call ncvpt_int2(ncid,vid,1,1,jx,istatus)

      istatus= NF_INQ_VARID(ncid,'x',vid)
      call ncvpt_doubl2(ncid,vid,1,jx,x,istatus)

      istatus= NF_INQ_VARID(ncid,'vnorm',vid)
      call ncvpt_doubl2(ncid,vid,1,1,vnorm,istatus)

      istatus= NF_INQ_VARID(ncid,'enorm',vid)
      call ncvpt_doubl2(ncid,vid,1,1,enorm,istatus)

      istatus= NF_INQ_VARID(ncid,'iy',vid)
      call ncvpt_int2(ncid,vid,1,1,iymax,istatus)

      if (iy*lrors.gt.iyjx2) stop 'netcdfrf:  Need to set jx>lrza'
      call pack21(y,1,iy,1,lrors,tem1,iymax,lrors)
      istatus= NF_INQ_VARID(ncid,'y',vid)
      call ncvpt_doubl2(ncid,vid,start1,y_count,tem1,istatus)

      istatus= NF_INQ_VARID(ncid,'iy_',vid)
      call ncvpt_int2(ncid,vid,1,lrz,iy_,istatus)

      istatus= NF_INQ_VARID(ncid,'itl',vid)
      call ncvpt_int2(ncid,vid,1,lrz,itl_,istatus)

      istatus= NF_INQ_VARID(ncid,'itu',vid)
      call ncvpt_int2(ncid,vid,1,lrz,itu_,istatus)
      
c  n_d_rr is the number of diffused general species
      if (n_d_rr.eq.1) then

      istatus= NF_INQ_VARID(ncid,'d_rr',vid)
         do ll=1,lrz
            do j=1,jx
               do i=1,iy
                  temp1(i,j)=d_rr(i,j,1,lrindx(ll))
               enddo
            enddo
            call pack21(temp1,0,iyp1,0,jxp1,wkpack,iy,jx)
            start1(3)=ll
            call ncvpt_doubl2(ncid,vid,start1,count1,wkpack,istatus)
         enddo
         if (kopt.eq.2) then
         istatus= NF_INQ_VARID(ncid,'d_r',vid)
         do ll=1,lrz
            do j=1,jx
               do i=1,iy
                  temp1(i,j)=d_r(i,j,1,lrindx(ll))
               enddo
            enddo
            call pack21(temp1,0,iyp1,0,jxp1,wkpack,iy,jx)
            start1(3)=ll
            call ncvpt_doubl2(ncid,vid,start1,count1,wkpack,istatus)
         enddo
         endif  !On kopt.eq.2
         
      else                      !n_d_rr.ge.2
         
         istatus= NF_INQ_VARID(ncid,'d_rr',vid)
         do k=1,n_d_rr
            do ll=1,lrz
               do j=1,jx
                  do i=1,iy
                     temp1(i,j)=d_rr(i,j,k,lrindx(ll))
                  enddo
               enddo
               call pack21(temp1,0,iyp1,0,jxp1,wkpack,iy,jx)
               startg(3)=ll
               startg(4)=k
               call ncvpt_doubl2(ncid,vid,startg,countg,wkpack,istatus)
            enddo               !  On ll
         enddo                  !  On k
         if (kopt.eq.2) then
         istatus= NF_INQ_VARID(ncid,'d_r',vid)
         do k=1,n_d_rr
            do ll=1,lrz
               do j=1,jx
                  do i=1,iy
                     temp1(i,j)=d_r(i,j,k,lrindx(ll))
                  enddo
               enddo
               call pack21(temp1,0,iyp1,0,jxp1,wkpack,iy,jx)
               startg(3)=ll
               startg(4)=k
               call ncvpt_doubl2(ncid,vid,startg,countg,wkpack,istatus)
            enddo               !  On ll
         enddo                  !  On k
         endif  !On kopt.eq.2
         
      endif                     ! on n_d_rr

      endif  !On kopt.eq.1 .or. kopt.eq.2

c.......................................................................
c  Read diffusion/pinch terms
c  It is not necessary to have prior call to diffus_io
c.......................................................................

      if (kopt.eq.3 .or. kopt.eq.4) then

      if (difus_io_file.eq."mnemonic") then
         t_=mnemonic//"_difus_io.nc"
      else
         t_=difus_io_file
      endif

      istatus=NF_OPEN(t_,NF_NOWRITE,ncid)
      if (istatus .NE. NF_NOERR) then
         WRITE(*,*)'   ***   Problem opening d_rr .nc data file   ***'
         Stop
      endif

c  Checking dimensions
      istatus= NF_INQ_DIMID(ncid,'xdim',xdim)
      istatus= NF_INQ_DIMID(ncid,'ydim',ydim)
      istatus= NF_INQ_DIMID(ncid,'rdim',rdim)
      istatus= NF_INQ_DIMID(ncid,'drr_gen_species_dim',gdim)
      istatus= NF_INQ_DIM(ncid,xdim,name,jx_file) !name should ='xdim', etc.
      istatus= NF_INQ_DIM(ncid,ydim,name,iy_file)
      istatus= NF_INQ_DIM(ncid,rdim,name,lrz_file)
      istatus= NF_INQ_DIM(ncid,gdim,name,n_d_rr_file)

c     Number of general species treated by the _difus_io.nc file
      n_d_rr=0
      do k=1,ngen
         if (difus_io(k).ne."disabled") n_d_rr=n_d_rr+1
      enddo

c     Check the data dimensions in the input file
      
      if (jx*iy*lrz*n_d_rr.ne.jx_file*iy_file*lrz_file*n_d_rr_file) then
         STOP '  WRONG DIMENSIONS IN _difus_io.nc file: STOP'
      endif

c     Could check rya, etc., in the _difus_io.nc file (but not yet done).

c     Read d_rr and (kopt.eq.4) d_r

      if (n_d_rr.eq.1) then

!     start1=10  !Testing that sets all start1(1:3).  (Yes, 4 gfortran.)
      start1(1)=1
      start1(2)=1
      start1(3)=1
      count1(1)=iy
      count1(2)=jx
      count1(3)=1

      istatus= NF_INQ_VARID(ncid,'d_rr',vid)
      do ll=1,lrz
         start1(3)=ll
         istatus= NF_GET_VARA_DOUBLE(ncid,vid,start1,count1,wkpack)
         call unpack21(temp1,0,iyp1,0,jxp1,wkpack,iy,jx)
         do j=1,jx
            do i=1,iy
               d_rr(i,j,1,lrindx(ll))=temp1(i,j)
            enddo
         enddo
      enddo  !On ll

      if (kopt.eq.4) then
      istatus= NF_INQ_VARID(ncid,'d_r',vid)
      do ll=1,lrz
         start1(3)=ll
         istatus= NF_GET_VARA_DOUBLE(ncid,vid,start1,count1,wkpack)
         call unpack21(temp1,0,iyp1,0,jxp1,wkpack,iy,jx)
         do j=1,jx
            do i=1,iy
               d_r(i,j,1,lrindx(ll))=temp1(i,j)
            enddo
         enddo
      enddo  !On ll
      endif  !On kopt.eq.4

      else  !n_d_rr.gt.1

      startg=1  !should give startg(1:4)=1
      countg(1)=iy
      countg(2)=jx
      countg(3)=1
      countg(4)=1
      
      istatus= NF_INQ_VARID(ncid,'d_rr',vid)
      do k=1,n_d_rr
         startg(4)=k
         do ll=1,lrz
            startg(3)=ll
            istatus= NF_GET_VARA_DOUBLE(ncid,vid,startg,countg,wkpack)
            call unpack21(temp1,0,iyp1,0,jxp1,wkpack,iy,jx)
            do j=1,jx
               do i=1,iy
                  d_rr(i,j,1,lrindx(ll))=temp1(i,j)
               enddo
            enddo
         enddo
      enddo  !On k

      if (kopt.eq.4) then
      istatus= NF_INQ_VARID(ncid,'d_r',vid)
      do k=1,n_d_rr
         startg(4)=k
         do ll=1,lrz
            startg(3)=ll
            istatus= NF_GET_VARA_DOUBLE(ncid,vid,startg,countg,wkpack)
            call unpack21(temp1,0,iyp1,0,jxp1,wkpack,iy,jx)
            do j=1,jx
               do i=1,iy
                  d_r(i,j,1,lrindx(ll))=temp1(i,j)
               enddo
            enddo
         enddo
      enddo  !On k
      endif  !On kopt.eq.4

      endif  !On n_d_rr

      istatus = NF_CLOSE(ncid)
      call check_err(istatus)

      start1(3)=1  !For safety
      startg(3)=1
      startg(4)=1

      endif  !On kopt.eq.3 .or. kopt.eq.4

      return
      end subroutine diffus_io
c
c
      real*8 function difus_io_scale(k,iopt)
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'

c......................................................................
c     Returns t-dependent scale factor for radial diffusion drr 
c     or pinch vel dr, for each species k.
c     iopt=1,   for drr
c         =2,   for dr
c......................................................................
     
      if (iopt.eq.1) then

      if (ndifus_io_t.le.0) then
         difus_io_scale=one
      else
         itme=1
         do jtm=1,ndifus_io_t
            if (timet.ge.difus_io_t(jtm)) itme=jtm
         enddo
         itme1=itme+1
         if (itme.lt.ndifus_io_t) then
            difus_io_scale=difus_io_drrscale(itme,k)
     1           +(difus_io_drrscale(itme1,k)-difus_io_drrscale(itme,k))
     1           /(difus_io_t(itme1)-difus_io_t(itme))
     1           *(timet-difus_io_t(itme))
         else
            difus_io_scale=difus_io_drrscale(ndifus_io_t,k)
         endif
      endif  !  On ndifus_io_t

      elseif (iopt.eq.2) then

      if (ndifus_io_t.le.0) then
         difus_io_scale=one
      else
         itme=1
         do jtm=1,ndifus_io_t
            if (timet.ge.difus_io_t(jtm)) itme=jtm
         enddo
         itme1=itme+1
         if (itme.lt.ndifus_io_t) then
            difus_io_scale=difus_io_drscale(itme,k)
     1           +(difus_io_drscale(itme1,k)-difus_io_drscale(itme,k))
     1           /(difus_io_t(itme1)-difus_io_t(itme))
     1           *(timet-difus_io_t(itme))
         else
            difus_io_scale=difus_io_drscale(ndifus_io_t,k)
         endif
      endif  !  On ndifus_io_t

      endif  !  On iopt

      return

      end function difus_io_scale
