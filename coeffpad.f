c
c
      subroutine coeffpad(k)
      implicit integer (i-n), real*8 (a-h,o-z)
c      real*8,dimension(iy):: prnt1,prnt2,prnt3
c      real*8,dimension(iy):: prnt4,prnt5,prnt6

c..................................................................
c     This routine adds in the collisional contribution to the 
c     coefficients employed in time advancement..
c     scatfrac=0. disables pitch angle scattering (along with
c       mx=0, see cqlinput_help. scatfrac=1. by default).
c..................................................................

      include 'param.h'
      include 'comm.h'

      do 10 j=1,jx
        scatfrac_j=scatfrac ! Or uncomment these lines:
!        if(j.gt.jxcrit(k,l_))then  !YuP[2020-05-01] Just to try: 
!          ! Apply scatfrac=0., but only at u > ucrit
!          !(i.e., disable scattering for RE, but not for Ohmic electrons)
!          scatfrac_j=scatfrac
!        else
!          scatfrac_j=1.0 !at j.le.jxcrit
!        endif
        do 11 i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
          da(i,j)=da(i,j)+cal(i,j,k,l_)
          db(i,j)=db(i,j)+cbl(i,j,k,l_)
          dc(i,j)=dc(i,j)+scatfrac_j*ccl(i,j,k,l_)
          dd(i,j)=dd(i,j)+scatfrac_j*cdl(i,j,k,l_) !YuP[2019-11-18] Added scatfrac
          de(i,j)=de(i,j)+scatfrac_j*cel(i,j,k,l_)
          df(i,j)=df(i,j)+scatfrac_j*cfl(i,j,k,l_)
 11     continue
 10   continue
c      do i=1,iy_(l_) !YuP[2021-03-11] iy-->iy_(l_)
c         prnt1(i)=da(i,2)
c         prnt2(i)=db(i,2)
c         prnt3(i)=dc(i,2)
c         prnt4(i)=dd(i,2)
c         prnt5(i)=de(i,2)
c         prnt6(i)=df(i,2)
c      enddo

         
      return
      end
