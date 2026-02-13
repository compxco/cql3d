
c*****************************************************************

c
c
      subroutine sounorm
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     This routine establishes an array of normalization constants
c     which allows the code to force a particular poloidal source
c     profile. It is called only at initialization.
c..................................................................


      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE     


c..................................................................
c     Set a flag.
c..................................................................

      isounor=1
      do 200 l=1,lz
      
        if (cqlpmod.eq."enabled") then !YuP[2025-09-01] added
          iiy=iy_(l) !CQLP: iy_ at given poloidal index;
        else !CQL3D
          iiy=iy_(l_) !CQL3D: at given radial index
        endif
      
        do 100 k=1,ngen
          call bcast(temp1(0,0),zero,iyjx2)
          do 50 m=1,nso

c..................................................................
c     Call a routine which determines the non-normalized source
c     distribution for a given i and all j.
c..................................................................

            do 40 i=1,iiy !YuP[2022-02-09] iy-->iy_()
              call soup(coss(i,l_),l,k,m) !-> get soupp array,
              !based on sem1z(*,lr) input for src localization in energy,
              !sthm1z(*,lr) localization in pitch angle,
              !szm1z(*,lr) localization in z(l).
              !Presently those inputs do not depend on l index along field line.
              do 30 j=1,jx
                temp1(i,j)=soupp(j,lr_)
 30           continue
 40         continue

c..................................................................
c     Set the array sounor such that when the computed source
c     profile is multiplied by sounor the resulting current
c     density is unity.
c..................................................................

            s=0.
            do 10 i=1,iiy !YuP[2022-02-09] iy-->iy_()
              do 20 j=1,jx
                s=s+temp1(i,j)*cynt2(i,l_)*cint2(j)
 20           continue
 10         continue
            if (s.ne.zero) sounor(k,m,l,lr_)=1./(s*one_)
 50       continue !m
 100    continue !k
 
 200  continue ! l=1,lz

c..................................................................
c     reset the flag (subroutine soup)
c..................................................................

      isounor=0
      return
      end
