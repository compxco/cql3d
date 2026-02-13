c
c
      subroutine resthks(kl_,klr_,klmdpln,preshin,preskim,pressau1,
     +  pressau2)
      implicit integer (i-n), real*8 (a-h,o-z)
c     ------------------------------------------------------------------
c
c     Compute resistivity ratios (to Spitzer Z=1 value), xxxBH,010807
c     BH,010807: Spitzer with Zeff.ne.1 to be used with HH...,below.
c               Olivier Sauter evidently adjusted the other ratios also.
c       
c     including collisionality and Zeff, according to
c     Hinton and Hazeltine (1976), Kim-Hirshman (1988) and Sauter (1993)
c
c     preshin=Hinton and Hazeltine ratio
c     preskim=Kim (Hirshman-Sigmar) ratio
c     pressau1=Sauter1
c     pressau2=Sauter2
c
! YuP[2020-02] Notes: 
! 1. Value of starnue only affects the values of pressau2 and preshin.
!    Values of pressau1 and preskim do not depend on starnue.
! 2. From printout: Value of preshin is smaller when starnue>0 
!    (comparing to its value at starnue=0); Similarly for pressau2.
! 3. From printout: Usually the values of pressau2 are slightly higher
!    than the values of preshin (at starnue=0, and at starnue>0, too).
! 4. It is suggested to call this subroutine with starnue>0,
!    and use pressau2 as the value corresponding to starnue>0,
!    while pressau1 corresponds to starnue=0 
!    (so, no need to call this subr. twice, 
!    when we want to find delta_sigma that gives the correction for
!     collisional-vs-noncollisional neoclassical conductivity)
!
c     Assume restcon called just before, i.e. xconn corresponds to kl_
c
      save
      include 'param.h'
      include 'comm.h'
c.......................................................................
c     N(Zeff) from Spitzer such that N(1,2,4,16,inf)=
c                                               1.,0.85,0.74,0.63,0.58
      zzef=zeff(klr_)
c     gives 1.00045 for Z=1 and less than 1% error up to infinity
      znzeff=0.58+0.74/(0.76+zzef)
c
cl    1. Hinton-Hazeltine (Rev.Mod.Phys. 48, 239 (1976))
c        Also, see ONETWO manual, GA-A16178.
c
      za33=4./3.*(1.-1./zzef)*0.47 + 1./3.*(4./zzef-1.)*0.68
      zb33=4./3.*(1.-1./zzef)*0.20 + 1./3.*(4./zzef-1.)*0.32
      zc33=4./3.*(1.-1./zzef)*0.51 + 1./3.*(4./zzef-1.)*0.66
      zk033=1.46+(1.83-1.46)/zzef
      zfcol=zk033/(1.+za33*sqrt(starnue(klmdpln))+zb33*starnue(klmdpln))
     /  /(1.+zc33*starnue(klmdpln)*eps(klr_)**1.5)
      zftraph=zfcol*sqrt(eps(klr_)) - (zfcol-1.)*eps(klr_)

c%OS  preshin=znzeff/(1.-zftraph)
      preshin=1.0/(1.-zftraph)
c
cl    2. Kim from thesis, which is Hirshman-Sigmar formula
c
c     note: znzeffk is equivalent to: 0.60 + 0.59/(0.43+Zeff)
      zg=(1.-xconn)/xconn
      zftrapk=(zzef**2+1.41*zzef+(2.*zzef**2+2.66*zzef+0.75)*zg+
     +  (zzef**2+1.24*zzef+0.35)*zg**2)/
     /  (3.25*zzef**2+1.41*zzef+(3.25*zzef**2+1.39*zzef)*zg)
      preskim=zftrapk*(3.25*zzef**2+1.41*zzef)/(zzef**2+1.41*zzef)

c
cl    3. Sauter formula
c        O. Sauter, C. Angioni and Y.R. Lin-Liu, Phys. of Plasmas 6, 
c        1834 (1999).
c
c%OS  pressau1=znzeff*(1.+1.24*zg)
      pressau1=1.0*(1.+1.24*zg)
      pressau2=(1.+1.24*zfcol/zk033*zg)

      return
      end
