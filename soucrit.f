c
c      
      subroutine soucrit
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
c
c   scchiu, 9609..
c  calculate the critical momentum-per-mass.
c  Since jxcrit is used for calculation of relativistic runaways,
c   it is chosen to be no larger than for particles at 3.*clight.
c   This is to reduce a problem of definition which occurs
c   transiently for abs(E) < abs(E_crit_for_runaway).
c
      call cfpgamma
      do 10 k=1,ngen
        fack=abs(elecfld(lr_))/reden(k,lr_)*1.e16/0.0918
     1              *18./gama(k,k)
        eoe0(k,lr_)=fack
        
            if((fack-1.).eq.zero)then
             write(*,*)'soucrit: (fack-1.)=0=',(fack-1.) 
             !pause
            endif
        
        fack1=1./(fack-1.) ! ~ Ec/E
        if (fack1.le.0.d0) then
          ucrit(k,l_)=1.e20 !YuP[2021-04] (k,lr_) --> (k,l_)
        else
          !YuP[2020-11-30] Try Zeff correction for critical speed :
          !alfa_z= (2.d0+zeff(lr_))**0.25 /1.5d0  !YuP version; 
            !See Smith_model_vs_CQL3D_Zcorrection_10d.pdf
          alfa_z=1.d0 !Un-comment for No correction (as in original definition)
          ucrit(k,l_)= (clight*sqrt(fack1)/vnorm)*alfa_z 
          !YuP[2021-04] (k,lr_) --> (k,l_)
        endif
        !ucrit(k,l_)=clight/vnorm !YuP[2020-05-02] Just to try: Set Boundary for RE as u/c=1
        !Result: A "delay" in growth of denra (and curra), but the final denra is same as before.
c  Take runaway electrons to have momentum per mass beyond the
c  minimum of 3.*clight or ucrit:
c990131        xcrit=amin1(3.*clight/vnorm,ucrit(k,l_))
        xcrit=min(3.*clight/vnorm,ucrit(k,l_)) !YuP[2021-04] (k,lr_) --> (k,l_)
        jxcrit(k,l_)=luf(xcrit,x,jx) !YuP[2021-04] (k,lr_) --> (k,l_)
10    continue
      return
      end
