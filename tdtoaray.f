c
c
      subroutine tdtoaray
      implicit integer (i-n), real*8 (a-h,o-z)

c     Storing values on limited lrz mesh and time-dependent quantities
c     recall:
c     lr_     : radial index
c     l_      : spatial variable index
c     lmdpln_ : midplane at l_=lmdpln_
c
      include 'param.h'
      include 'comm.h'


      if (kelecg.gt.0) vfluxz(l_)=vflux(jx-1,kelecg,l_) !YuP[2023-06-30] 1-->kelecg
                         !YuP[2023]: kelecg is not always 1
      currmtz(l_)=currmt(l_)/3.e+9
      currmtpz(l_)=currmtp(l_)/3.e+9

c..................................................................
c     Copies flux surface dependent quantities into radial arrays
c..................................................................

      if (l_ .ne. lmdpln_) return

      do 5 k=1,ngen
        pegyz(k,lr_)=entr(k,12,l_)
        pplossz(k,lr_)=entr(k,6,l_)
 5    continue
cBH081202      psyncz(lr_)=entr(kelecg,11,l_)
      if (kelecg.ne.0) then
         psyncz(lr_)=entr(kelecg,11,l_)
         !write(*,*)' tdtoarray: l_,psyncz(lr_)=',l_,psyncz(lr_)
      else
         psyncz(lr_)=zero
      endif

c..................................................................
c     rfpwrz(k,lr_) is the local rf power deposited in species k
c     at flux surface lr_ in Watts/cc (averaged over the orbit)
c     currz(k,lr_) is the current density of species k at lr_ in A/cm**2
c       (averaged over incremental toroidal-area at given flux surface).
c     currtz(lr_) is the sum of currz(k,lr_) over ionic general species.
c     currmtz(lmdpln_) is the sum of the midplane currents over ionic species
c     curr(m)tpz(lr_) is curr(m)tz plus the electron contribution to
c     the current.
c     1 Amp = 3.e9 statAmps
c..................................................................


      do 21 k=1,ngen
        currz(k,lr_)=curr(k,lr_)/3.e+9
        rfpwrz(k,lr_)=entr(k,3,l_)
        if (nso.gt.0) then
          numsrce=nso
        else
          numsrce=1
        endif
        do 23 m=1,numsrce
          asorz(k,m,lr_)=asor(k,m,lr_)
 23     continue
 21   continue

      currtz(lr_)=currt(lr_)/3.e+9
      currtpz(lr_)=currtp(lr_)/3.e+9

c..................................................................
c     Compute the plasma pressure (cgs)
c     if (kpress(k).ne. "enabled") this species is not used
c     in calculation of total pressure.
c..................................................................

      if (eqmod.eq."enabled") then
        phot=0.
        do 10 k=1,ngen
          if (kpress(k).eq."disabled") go to 10
          phot=phot+reden(k,lr_)*energy(k,lr_)*2./3.*ergtkev
 10     continue
        prest(lr_)=phot
        do 20 k=ngen+1,ntotal
          if (kpress(k).eq."disabled") go to 20
          prest(lr_)=prest(lr_)+reden(k,lr_)*energy(k,lr_)*2./3.*
     *      ergtkev
 20     continue
 30     continue
      endif
      return
      end
