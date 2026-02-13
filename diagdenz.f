c
c
      subroutine diagdenz
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c..................................................................
c     Computes density as a function of angle= twopi*z(l,lr_)/zmax(lr_)
c     and as a function of energy range. The results are used as
c     diagnostics in plots, on in the event that locquas="enabled"
c     to establish the background electron density as a function of
c     poloidal angle to maintain LOCAL (in z) quasi-neutrality.
c     negyrg is the number of energy bins over which we will be
c     computing density. For each energy range [eegy(ny,1,k,lr_)
c     ,eegy(ny,2,k,lr_)]
c     the velocity range associated with it must be determined. 
c     The indicator will be jegy(ny,1,k,lr_) (lower) and
c     jegy(ny,2,k,lr_) (upper). Note that the
c     final energy bin or range is forced to be the entire energy range.
c..................................................................

      include 'param.h'
      include 'comm.h'

      if (negyrg.lt.1) return
      do 103 k=1,ngen

c..................................................................
c     Determine the lower and upper momentum indices for the given
c     energy range.
c..................................................................

        do 100 ny=1,negyrg

c..................................................................
c     ny=negyrg forces consideration of complete energy range.
c..................................................................

          if (ny .eq. negyrg) then
            jegy(ny,1,k,lr_)=1
            jegy(ny,2,k,lr_)=jx
            eegy(negyrg,1,k,lr_)=0.
            eegy(negyrg,2,k,lr_)=fions(k)*tcsgm1(jx)
            go to 100
          endif
          do 101 j=2,jx
            if (fions(k)*tcsgm1(j) .gt. eegy(ny,1,k,lr_)) go to 104
 101      continue
          jegy(ny,1,k,lr_)=0
          go to 108
 104      continue
          jegy(ny,1,k,lr_)=j-1
          do 106 j=2,jx
            if (fions(k)*tcsgm1(j) .gt. eegy(ny,2,k,lr_)) go to 107
 106      continue
          jegy(ny,2,k,lr_)=jx
          go to 108
 107      jegy(ny,2,k,lr_)=j
 108      continue
          if (jegy(ny,1,k,lr_) .ge. jegy(ny,2,k,lr_)) jegy(ny,1,k,lr_)=0
 100    continue ! ny=1,negyrg
 103  continue ! k=1,ngen
      call bcast(xlndnz,zero,size(xlndnz)) !shape: (ngen+1,max(negyrg,1))

c..................................................................
c     xlndnz(k,ny) will eventually hold the line integrated density
c
c     densz(l,k,ny,lr_) will contain the local density at z(l,lr_) (partic
c..................................................................

      if (cqlpmod .ne. "enabled") then
        iorbstr=1
        iorbend=lz
      else ! (cqlpmod.eq."enabled")
        iorbstr=l_
        iorbend=l_
      endif
      do 10 k=1,ngen
        call dcopy(iyjx2,f(0,0,k,l_),1,temp3(0,0),1)
        do 11 l=iorbstr,iorbend
          ileff=l
          if (cqlpmod.eq."enabled") ileff=ls_
          call cfpleg(0,ileff,1)
          do 40 ny=1,negyrg
            if (jegy(ny,1,k,lr_).eq.0 .or. eegy(ny,2,k,lr_).lt.1.e-15)
     1        go to 40
            densz(l,k,ny,lr_)=0.
            do 6 j=jegy(ny,1,k,lr_),jegy(ny,2,k,lr_)
 6          densz(l,k,ny,lr_)=densz(l,k,ny,lr_)+cint2(j)
     1          *tam1(j)*one_*twopi*2.
            xlndnz(k,ny)=xlndnz(k,ny)+densz(l,k,ny,lr_)
     1        *dz(ileff,lr_)/bbpsi(ileff,lr_)
 40       continue
 11     continue
 10   continue


c      write(*,*) 'DENSZ(1,1,1:3,LR_),LR_', 
c     +                 (DENSZ(1,1,NY,LR_),NY=1,NEGYRG),LR_

c..................................................................
c     Now determine local electron density for locquas="enabled"
c     calculations
c..................................................................

      if (kelecg.gt.0 .or. locquas .eq. "disabled") return
      nw=negyrg
      xlndnz(ngen+1,nw)=0.
      do 20 l=iorbstr,iorbend
        densz(l,ngen+1,nw,lr_)=0.0
        ileff=l
        if (cqlpmod.eq."enabled") ileff=ls_
        do 30 k=1,ngen
          densz(l,ngen+1,nw,lr_)=densz(l,ngen+1,nw,lr_)
     1      +densz(l,k,nw,lr_)*bnumb(k)
 30     continue
        xlndnz(ngen+1,nw)=xlndnz(ngen+1,nw)
     1    +densz(l,ngen+1,nw,lr_)*dz(ileff,lr_)
     1    /bbpsi(ileff,lr_)
 20   continue
      denn=0.
      do 200 ku=ngen+1,ntotal
        if (ku .eq. kelecm) go to 200
        denn=denn+reden(ku,lr_)
 200  continue
      xlndnz(ngen+1,nw)=xlndnz(ngen+1,nw)+denn*zmaxpsi(lr_)
      do 201 l=iorbstr,iorbend
        densz(l,ngen+1,nw,lr_)=densz(l,ngen+1,nw,lr_)+denn
 201  continue
      return
      end
