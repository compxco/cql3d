c
c
      subroutine eflditer ! for k=kelecg (=1) only
      implicit integer (i-n), real*8 (a-h,o-z)
      save

c.........................................................................
c     This routine facilitates iteration to obtain a given current density.
c     Current is calculated.
c     If this current is not acceptably close to a target current,
c     the electric field is adjusted, and the distribution is reset
c     to the beginning of the time step, so that another trial can
c     be make.
c.........................................................................

      include 'param.h'
      include 'comm.h'

c     Return, for cases not implemented:
      if (cqlpmod.eq."enabled" ) return
      if (ngen.gt.1 .or. kelecg.ne.1) stop 'check operation of eflditer'
      
c-YuP: moved this control outside of this subroutine:
c     Set (stop iteration) flag and return, if beyond iteration limit:
c-YuP      if (nefiter+1.gt.nefitera) then
c-YuP         nefiter=0
c-YuP         go to 999
c-YuP      endif

c     Set diagnostic variable elecn, on first iteration:
      if (nefiter.eq.1) then
cBH171230         elecfldn(nefiter,n,l_)=elecfld(lr_)
         elecn(l_,n,nefiter)=elecfld(lr_)
         do iter=2,nefitera
            elecn(l_,n,iter)=0.
         enddo
      endif
            
c-YuP: moved this control outside of this subroutine:
c     Set (stop iteration) flag and return, if n.lt.(control turn on step):
c-YuP      if (n.le.noncntrl) then
c-YuP         nefiter=0
c-YuP         go to 999
c-YuP      endif

c     Calculate current (following subroutine diaggnde).
c     This is parallel current at minimum B point on a flux surface.

      currtp(lr_)=0.0
      call bcast(tam3,zero,jx)
      do 20 i=1,iy_(l_)  !YuP[2021-03-11] iy-->iy_(l_)
         do 10 j=1,jx
            tam3(j)=tam3(j)+f(i,j,1,l_)*cynt2(i,l_)*coss(i,l_)
 10      continue
 20   continue

      cn=0.
      do 40 j=1,jx
         currv(j,1,l_)=tam3(j)*x(j)*gammi(j)*cint2(j)*dxi(j)
         cn=cn+currv(j,1,l_)*dx(j)
 40   continue

c     Determine jxcrit (presently determined as
c     index for amin1(3.*clight,ucrit).
      call soucrit
      
      curra(1,l_)=0.0
      if (jxcrit(1,l_).ge.jx) go to 50
      do j=jxcrit(1,l_),jx !YuP[2021-04] (k,lr_) --> (k,l_)
         curra(1,l_)=curra(1,l_)+currv(j,1,l_)*dx(j)
      enddo
 50   continue

      faccur=one_*vnorm*bnumb(1)*charge
      currm(1,l_)=faccur*cn
      psifct=psiovr(lr_)/onovrp(1,lr_)

c     psifct=<(B/B_0)/R>/<1/R>, where B_0 is min mag fld of a flux surf,
c     <...> is flux surface volume average.

c     curr, is obtainted by multiplying parallel current at the 
c     midplane, currm, by psifct.  This gives the average j_parallel 
c     over the fixed-toroidal-angle cross-section area between
c     neighboring flux surfaces.  This approximates the same average of
c     the toroidal current, in the usual tokamak case.
c     curr is not an accurate physically meaningful quantity, 
c     and should be excised from the code. (BobH, 010301).
     
      curr(1,lr_)=currm(1,l_)*psifct ! FSA now
      currtp(lr_)=currtp(lr_)+curr(1,lr_) ! FSA
      curra(1,l_)=faccur*psifct*curra(1,l_)
c     Following statement for consistency with diaggnde
      call dscal(jx,faccur*psifct,currv(1,1,l_),1)
c     (End of following subroutine diaggnde.)

      if (efswtchn.eq."neo_hh") then
         call restcon
         call resthks(l_,lr_,lmdpln_,
     +        zreshin(lr_),zreskim(lr_),zressau1,zressau2)
         !Original version that uses zreshin [Hinton and Hazeltine]:
         currpar(lr_)=(curra(kelec,l_)+
     +        elecfld(lr_)/300./(zreshin(lr_)*sptzr(l_)))/3.e9
         !YuP[2020-04-02] We could change to version with zressau1
         !similar to subr.efield.
         !Based on Sauter, Angioni and Lin-Liu, Phys.Plasmas 6, 1834 (1999) :
!---         currpar(lr_)=(curra(kelec,l_)+
!---     +        elecfld(lr_)/300./(zressau1*sptzr(l_)))/3.e9
     
         !This currpar() is used below, for efswtch=method5
      else ! efswtchn.eq."disabled" (default value) Use total Ip (Ohmic+RE)
         currpar(lr_)=currtp(lr_)/3.e9
      endif


c           Presently, the calculated parallel current
c           above is averaged over the area cross-section
c           (at cnst toroidal angle) between a flux surface.
c           This quantity is depracated, and should be replaced
c           in the future.  Here, for method5, we adjust currpar 
c           back to parallel current at the minimum B position.
c           See comments in eflditer.f, above, on psifct.

            psifct1=1.
            if(efswtch.eq."method5") then
               psifct1=psiovr(lr_)/onovrp(1,lr_)
            endif


         currerrf=(currpar(lr_)/psifct1 -currxj0(lr_))/
     +           ( sign(one,currxj0(lr_))*
     +           max(abs(currpar(lr_)/psifct1),abs(currxj0(lr_))) )

c     Compare fractional current error with allowable.
c     In necessary, set iteration flag, obtain new efield,
c        and adjust contribution of electric field coefficients.

      if (abs(currerrf).lt.currerr) then
         nefiter_(l_)=0  !-YuP: made it a function of flux surface
      else

         elecfld_old=elecfld(lr_) !-YuP: 
c        Subtract old elecfld contribution to FP equation coefficients:
         elecfld(lr_)=-elecfld_old
         call coefefad(1)

c        Find new elecfld:
         corrf=efrelax1*currerrf
         !corrf=MIN(corrf,0.95d0)   !-YuP: limited by +0.95 from above
         !corrf=MAX(-0.95d0,corrf)  !-YuP: limited by -0.95 from below
         elecfld(lr_)=elecfld_old*(1.d0-corrf)
c-YuP         call coefefad(1) !-YuP: don't add el.field here;
                               ! It will be added during next iteration 
                               ! through impavnc0->coefstup->coefefad
         
c        Save elecfld values for diagnostic purposes:
cBH171230         elecfldn(nefiter,n,l_)=elecfld(lr_)
         elecn(l_,n,nefiter)=elecfld(lr_)
      write(*,'(a,i5,4e12.4)')'elecfld,currpar/psifct,currxj0,currerrf',
     +     lr_,elecfld(lr_),currpar(lr_)/psifct1,currxj0(lr_),currerrf

      endif

 
         
      return
      end
