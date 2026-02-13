c
c
      subroutine tdwritef
      implicit integer (i-n), real*8 (a-h,o-z)
c.......................................................................
c     Save current parameters ,distribution function and spatial source
c     to enable complete restart of run
c.......................................................................

      include 'param.h'
      include 'comm.h'
      include 'frname_decl.h'
      include 'name.h'
      include 'frname.h'
CMPIINSERT_INCLUDE
c.......................................................................

CMPIINSERT_IF_RANK_NE_0_RETURN

      iunwrif=19
      open(unit=iunwrif,delim='apostrophe',file='distrfunc')

c.......................................................................
cl    1. Write current time-step and main diagnostics
c     inamlin=line number of first namelist (i.e. number of lines to
c     skip when reading this file). [innamlin NOT USED, BH070508].]
c.......................................................................

      write(iunwrif,9100) lrors+5,n,dtr,lrors,lrz,cqlpmod
      if (cqlpmod .ne. "enabled") then
      
        write(iunwrif,9101)
        do k=1,ngen
        do 101 l=1,lrors
cBH070408 write(iunwrif,9102) l,rovera(l),iy_(l),reden(kelecg,lrindx(l))
          write(iunwrif,9102) l,rovera(l),iy_(l),reden(k,lrindx(l))
     +      ,energy(k,lrindx(l)),totcurz(lrindx(l)),rovs(lrindx(l))
 101    continue
        enddo
CMPIINSERT_IF_RANK_EQ_0
        WRITE(*,*)'tdwritef_43: For checkup SUM(reden),SUM(energy)=', 
     +   SUM(reden),SUM(energy)
        WRITE(*,*)'tdwritef_44: For checkup SUM(totcurz),SUM(rovs)=', 
     +   SUM(totcurz),SUM(rovs)
CMPIINSERT_ENDIF_RANK
        
      else ! (cqlpmod.eq."enabled")
      
        write(iunwrif,9103)
        do k=1,ngen
        do 102 l=1,lrors
cBH070408 write(iunwrif,9102) l,sz(l),iy_(l),denpar(kelecg,lsindx(l)),
          write(iunwrif,9102) l,sz(l),iy_(l),denpar(k,lsindx(l)),
     +      enrgypa(k,lsindx(l)),currmtpz(l),rovsloc(l)
 102    continue
        enddo
        
      endif

c.......................................................................
cl    2. Write namelists and distribution function
c.......................................................................

      write(iunwrif,'(" ")')
      write(iunwrif,setup0)
      write(iunwrif,setup)
      write(iunwrif,trsetup)
      write(iunwrif,sousetup)
      write(iunwrif,eqsetup)
      write(iunwrif,rfsetup)
      write(iunwrif,frsetup)
c.......................................................................
cBH070408:  Have added write of frsetup here.  Probably should
c           also write the seed number for the random number generator,
c           so can achief same NBI results with restart.
c.......................................................................



c.......................................................................
cl    2.2 f(i,j,k,l)
c.......................................................................

c     Don't bother writing f, if nlwritf="ncdfdist", indicating
c     will use the netcdf mnemonic.nc file as source for f for
c     restart (nlrestrt="ncdfdist").

      if (nlwritf.ne."ncdfdist") then
      
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)
     +     'tdwritef[nlwritf.ne."ncdfdist"]:Writing f into distrfunc'
         WRITE(*,*)'tdwritef_83: For checkup SUM(f)=', SUM(f)
CMPIINSERT_ENDIF_RANK
cBH050328:  Problem with reading the given format with index
cBH050328:  numbers .lt.1.e-99
        do 220 k=1,ngen
        do 221 il=1,lrors
          do 222 j=1,jx
             do i=1,iy_(il)
                f(i,j,k,il)=max(f(i,j,k,il),1.d-99)
             enddo
             write(iunwrif,9220) (f(i,j,k,il),i=1,iy_(il))
 222      continue
 221    continue
 220    continue
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'tdwritef_98: For checkup SUM(f)=', SUM(f)
CMPIINSERT_ENDIF_RANK

      endif ! nlwritf.ne."ncdfdist"

      if (cqlpmod.eq."enabled") then
c.......................................................................
cl    2.3 Spatial source term: spasou(i,j,k,l)
c.......................................................................
         do 230 k=1,ngen
            do 231 il=1,lrors
               do 232 j=1,jx
                  write(iunwrif,9220) (spasou(i,j,k,il),i=1,iy_(il))
 232           continue
 231        continue
 230     continue
c.......................................................................
cl    2.4 Velocity source term: velsou(i,j,k,l)
c.......................................................................
         do 240 k=1,ngen
            do 241 il=1,lrors
               do 242 j=1,jx
                  write(iunwrif,9220) (velsou(i,j,k,il),i=1,iy_(il))
 242           continue
 241        continue
 240     continue
      endif ! cqlpmod=enabled
      
c.......................................................................
 9100 format(i3," more lines to skip before reading namelist and f",//,
     !  "time-step n=",i4," dtr=",1pe12.5," lrors=",i3," lrz=",i3,
     !  " cqlpmod=",a)
 9101 format(/,"  l",3x,"rovera",4x," iy ",3x,"density",5x,"energy",4x,
     !  "tot.curr.",2x,"res/spitzr")
 9102 format(i3,1pe12.4,i4,4e12.4)
 9103 format(/,"  l",6x,"s",6x," iy ",3x,"density",5x,"energy",4x,
     !  "tot.curr.",2x,"res/spitzr")
cBH080201 9220 format(1p10e13.6)
 9220 format(1p10e14.6)
c.......................................................................
      close(unit=iunwrif)
      !pause
      return
      end
