c
c
      subroutine frstup(mf,mfm1,mi,mj,nion,potsid,codeid,rin,
     1  rmax,zax,zminn,zmaxx,zni,zne,zte,zti,zzi,frpsi,psivol,
     1  xxx,yyy,nprim,nimp,zeffctv,zshift1)
      implicit integer (i-n), real*8 (a-h,o-z)

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      character*8 codeid 
      dimension psivol(*),potsid(*),frpsi(nnra,*),
     1  zne(kz,*),zni(kz,*),zte(*),zzi(kz,*),xxx(*),yyy(*),
     1  zti(*),zeffctv(*)

c..................................................................
c     This routine should be called just before the call to FREYA.
c     It defines a number of time (or iteration) dependent input
c     variables required in the call to FREYA.
c..................................................................


c..................................................................
c     Set the frpsi (epsi) array as required by FREYA
c     Maximum will be at limiter not the axis.
c..................................................................

      do 10 i=1,nnr
        xxx(i)=er(i)
        do 11 j=1,nnz
          frpsi(i,j)=-epsi(i,j)
 11     continue
 10   continue
      do 15 j=1,nnz
        yyy(j)=ez(j)
 15   continue
c.....................................................................
c     eqpsi(1:nconteqn) is set up in cql3d routine eqrhopsi.f.
c     It is used to give the flux zones for freya returned in the
c     izone argument of subroutine inject (now inject_old and inject1),
c     called in subroutine freya.
c     Change the sign of (positive) eqpsi array to get it in ascending 
c     order for splines.  (Changed back at end of subroutine).
c.....................................................................

      do 2 l=1,nconteqn
        eqpsi(l)=-eqpsi(l)
 2    continue
c..................................................................
c     The number of flux zones (mfm1) used in FREYA
c..................................................................

      mfm1=nconteqn-1
      mf=mfm1+1

CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)
      WRITE(*,*)'Number of flux zones (mfm1) used in FREYA=',mfm1
      WRITE(*,*)'Adjust mfm1=nconteqn-1 throught NL input of nconteqn'
      WRITE(*,*)
CMPIINSERT_ENDIF_RANK

c..................................................................
c     Interpolate densities, temperatures etc over to the 
c     (high resolution?) mesh utilized by FREYA.
c
c..................................................................

      psivol(1)=.5*(eqvol(2)+eqvol(1))
      psivol(mfm1)=eqvol(mf)-.5*(eqvol(mfm1)+eqvol(mfm1-1))
      do 20 ll=2,mfm1-1
        psivol(ll)=.5*(eqvol(ll+1)-eqvol(ll-1))
 20   continue

c..................................................................
c     Electron density, energy
c..................................................................
     
      if( (colmodl.eq.1 .or. colmodl.eq.3) .and.(kelecm.ne.0))then 
        !YuP[2022-11-23]added this branch: 
        !For tests: if Maxw.electrons are available,
        !use their density.
        !(We could use general electrons, if available,
        !but it would be inconvenient for the test:
        !make a run with colmodl=3, with steady-state ne_maxwell,
        !update NBI at every time step, ==> confirm that NBI deposition
        !is unchanged. If we use general electrons in such test,
        !their density may change if src_nbi_e= "enabled" or 
        !because of loss cone.)
        !However, for physics, it is still better to use kelecg 
        !(general electrons, if available).
        !For tests: kel=kelecm !then, ne is not updated in time.
        kel=kelec !For physics (=kelecg, so that ne is updated in time)
CMPIINSERT_IF_RANK_EQ_0   
        WRITE(*,*)'frstup: For ne,Te, using k=kel=',kel
CMPIINSERT_ENDIF_RANK  
        do ll=1,lrzmax
          tr1(ll)=reden(kel,ll) 
          tr2(ll)=energy(kel,ll)*2./3. 
CMPIINSERT_IF_RANK_EQ_0   
          WRITE(*,'(a,i4,1p2e11.3)') 
     &     ' frstup: ne[cgs],Te[keV]=',ll,tr1(ll),tr2(ll)
CMPIINSERT_ENDIF_RANK
        enddo
      else !Original part; now used in case of colmodl=0
        !(or colmodl=2, or other colmodl, if kelecm=0)
        !If general electrons are available, use them for ne in FREYA, 
        !in this case kelec==kelecg; example - a tandem run.
        !If e_general are absent, then kelec=kelecm (i.e., e_maxwell)
CMPIINSERT_IF_RANK_EQ_0   
        WRITE(*,*)'frstup: Using kelec=',kelec
CMPIINSERT_ENDIF_RANK  
        do ll=1,lrzmax
          tr1(ll)=reden(kelec,ll)  !From kelecg (general electrons)
          tr2(ll)=energy(kelec,ll)*2./3. !From kelecg
CMPIINSERT_IF_RANK_EQ_0   
          WRITE(*,'(a,i4,1p2e11.3)') 
     &     ' frstup: ne[cgs],Te[keV]=',ll,tr1(ll),tr2(ll)
CMPIINSERT_ENDIF_RANK
        enddo
      endif

c..................................................................
c     Call spline interpolation routine to get values on FREYA 
c     eqspi mesh
c     (which is same as intemediary radial mesh used in CQL3D proper).
c..................................................................

      call frsplft(lrzmax,equilpsp(1),tr1(1),mfm1,eqpsi,zne) !set ne
      call frsplft(lrzmax,equilpsp(1),tr2(1),mfm1,eqpsi,zte) !set Te

      !YuP[2021-01-11] Revision of method how to match ions from cqlinput
      !to ions in zfreya.
CMPIINSERT_IF_RANK_EQ_0
      !These vars are set in cqlinput (nprim could have been adjusted in frset)
      !nprim= Number of primary ion species (such as NBI species);
      !nimp=  Number of impurity ion species (0 or 1, for now)
      WRITE(*,*)'frstup: nprim,nimp=',nprim,nimp
      WRITE(*,*)'frstup: nionm=',nionm
CMPIINSERT_ENDIF_RANK

      !Check that there are ions in Maxwellian_ion list 
      !to represent impurity nimp for FREYA.
      !nimp=  Number of impurity ion species (0 or 1, for now)
      if(nionm.gt.0)then  ![2022-05-26] added nionm condition
        !A maxwellian ion is present in input 
        !(should be present in case for colmodl=3, 
        !but not necessarily in case of colmodl=0)
CMPIINSERT_IF_RANK_EQ_0      
        WRITE(*,*)'frstup: kionm(1:nionm)=',kionm(1:nionm)
CMPIINSERT_ENDIF_RANK
!        if(kionm(nionm)-kionm(1)+1 .lt. nion) then
!          WRITE(*,*)
!     &    'frstup: Need to have nprim+nimp ion Maxwellian species'
!          STOP 'frstup'
!        endif
        !YuP[2023-01-09] Changed nion to nimp
        !(i.e., take impurity ion from Maxwellian_ion list
        ! and primary ion(s) from general_ion list)
        if(kionm(nionm)-kionm(1)+1 .lt. nimp) then
          WRITE(*,*)'frstup: Need to have nimp ion Maxwellian species'
          STOP 'frstup'
        endif
      else !nionm=0
CMPIINSERT_IF_RANK_EQ_0      
        WRITE(*,*)'frstup: nionm=0, No ionic Maxw.species? Using nimp=0'
        !Can be no Maxw.ions (only general) ==> 
        !get primary ion(s) from general_ion list, 
        !and no impurity ion in this case.
CMPIINSERT_ENDIF_RANK
        nimp=0 
      endif

c..................................................................
c     Average ion temperature.. [populate zti()]
c..................................................................
      !First, determine index k in cqlinput list of species
      ! that points to impurity species.
      !OLD version: kimp_=ntotal-nimp+1 !YuP[2021] This is not always correct.
        !Example: Using  D_gen,T_gen; D_Maxw,T_Maxw,C_maxw,e_Maxw
        !Here we have nprim=2, nimp=1, ntotal=6
        !Then, ntotal-nimp+1= 6, pointing to e_Maxw (wrong).
      !YuP[2023-01] Revised version to ID the impurity ion: 
      !Scan the Maxwellian_ion list, 
      !skip those that are present among general_ion list
      !(as in colmodl=3 runs), take the first ion in the remaining list
      !as the impurity ion.
      kk=1 ! To be incremented
      kimp_=0 !initialize ![2022-05-26]
      if(nionm.gt.0)then  ![2022-05-26] added nionm condition
        !Maxwellian ions are present in input 
        !(should be present in case for colmodl=3, 
        !but not necessarily in case of colmodl=0)
        do k=ngen+1,ntotal !YuP: scan all Maxwellian species.
          kg=kg_same(k) !Pointing from k to kg in the general_ion list.
          if(kimp_.eq.0)then !scan until the first kimp_=k is found.
          if(kionm(kk).eq.k) then ! This is an ion Maxw. species.
            if(kg.eq.0)then !means: This ion is not present in general_ion list
              kimp_=k !==kionm(kk) !Impurity ion found.
            endif
            kk=kk+1 !Increment index that counts Maxwellian ions
          endif !(kionm(kk).eq.k)
          endif !(kimp_.eq.0)
        enddo !k=ngen+1,ntotal
      endif !(nionm.gt.0)
      !In the above example, kimp_ will point to 5 (which is C_Maxw)
CMPIINSERT_IF_RANK_EQ_0      
      WRITE(*,*)'frstup: Impurity species index k=kimp_=',kimp_ 
      !Index of species in cqlinput list 
      !that points to impurity ion species.
CMPIINSERT_ENDIF_RANK

      ! Now compute an average ion temperature.
      !YuP[2023-01] Revised version: scan general_ion list
      !within k=1:nprim,
      !then add impurity ion k=kimp_ from Maxw_ion list
      !(only one impurity is allowed, for now, nimp.le.1)
      call bcast(tr1(1),zero,lrzmax)
      do 50 ll=1,lrzmax
        tp=0.d0
        dn=0.d0
        kk=1
        do k=1,ngen !Scan all general species.
          !Consider first 1:nprim ion species among general species:
          if(kiong(kk).eq.k .and. kk.le.nprim) then 
            ! This is an ion general species
            tp=tp+ (energy(k,ll)*2.d0/3.d0)*reden(k,ll)
            !At this point, energy() is evaluated without lost ions, 
            !so it can be lower than temp(k,ll)
            dn=dn+reden(k,ll)
            kk=kk+1 !Increment index that counts general ions
            !write(*,*)'frstup: ll,tp,energy(k,ll)=',ll,tp,energy(k,ll)
          endif
        enddo !k=1:ngen
        !Add impurity ions, if present:
        if(kimp_.gt.0)then
          tp=tp+ (energy(kimp_,ll)*2.d0/3.d0)*reden(kimp_,ll)
          dn=dn+reden(kimp_,ll)
        endif
        tr1(ll)=tp/dn !Average T over nprim&nimp ions.
 50   continue ! ll=1,lrzmax
CMPIINSERT_IF_RANK_EQ_0      
      WRITE(*,'(a,1pe14.5,a)')' frstup: Ion Temp(lr=1)=',tr1(1),'keV'
CMPIINSERT_ENDIF_RANK
      call frsplft(lrzmax,equilpsp(1),tr1(1),mfm1,eqpsi,zti(1)) !->zti()

c..................................................................
c     Ion densities [populate zni() array for kk=1:nprim+nimp ions]
c..................................................................
      !YuP[2023-01] Revised version: scan general_ion list
      !within kk=1:nprim,
      !then add impurity ion k=kimp_ from Maxw_ion list
      !(only one impurity is allowed, for now, nimp.le.1)
      kk=1 !Counter of primary ions, To be incremented
      do k=1,ngen !YuP: scan all general species 
       if(kiong(kk).eq.k .and. kk.le.nprim) then 
          ! This is an ion gen. species
          do ll=1,lrzmax
            tr1(ll)=reden(k,ll)
            !write(*,*) kk,k, ll, tr1(ll), ' ion density'
          enddo ! ll
CMPIINSERT_IF_RANK_EQ_0      
          WRITE(*,'(a,i3,a,1pe14.5)')
     &         ' frstup: Ion dens: use gen.ion k=',k,
     &         ' reden(k,lr=1)=',reden(k,1)
CMPIINSERT_ENDIF_RANK
          call frsplft(lrzmax,equilpsp(1),tr1(1),mfm1,eqpsi,zni(1,kk)) !->zni
          !zni(:,kk) corresponds to zfreya list, starting with kk=1 
          kk=kk+1 !Increment index that counts general ions within nprim
       endif
      enddo ! k
      !Add impurity ions, if present:
CMPIINSERT_IF_RANK_EQ_0      
      if(nimp.gt.0) WRITE(*,*)' Add Impurity (if nimp>0). kk=',kk  
CMPIINSERT_ENDIF_RANK
      if(kimp_.gt.0)then
          do ll=1,lrzmax
            tr1(ll)=reden(kimp_,ll)
          enddo ! ll
CMPIINSERT_IF_RANK_EQ_0    
          WRITE(*,'(a,i3,a,1pe14.5)')
     &         ' frstup: Ion dens: use Maxw.ion kimp_=',kimp_,
     &         ' reden(kimp_,lr=1)=',reden(kimp_,1)
CMPIINSERT_ENDIF_RANK
          call frsplft(lrzmax,equilpsp(1),tr1(1),mfm1,eqpsi,zni(1,kk)) !->zni
      endif


c..................................................................
c     Ionic charge [populate zzi() array for kk=1:nprim+nimp ions]
c..................................................................
      !YuP[2023-01] Revised version: scan general_ion list
      !within kk=1:nprim,
      !then add impurity ion k=kimp_ from Maxw_ion list
      !(only one impurity is allowed, for now, nimp.le.1)
      kk=1 !Counter of primary ions, To be incremented
      do k=1,ngen !YuP: scan all general species 
       if(kiong(kk).eq.k .and. kk.le.nprim) then 
          ! This is an ion gen. species
          do ll=1,lrzmax
            tr1(ll)=bnumb(k)
            !write(*,*) kk,k, ll, tr1(ll), ' ion Charge state'
          enddo ! ll
          call frsplft(lrzmax,equilpsp(1),tr1(1),mfm1,eqpsi,zzi(1,kk))
          !zzi(:,kk) corresponds to zfreya list, starting with kk=1 
CMPIINSERT_IF_RANK_EQ_0      
          WRITE(*,*)'frstup: kk,k,zzi(1,kk)=', kk,k,zzi(1,kk)
CMPIINSERT_ENDIF_RANK
          kk=kk+1 !Increment index that counts general ions within nprim
       endif
      enddo ! k
      !Add impurity ions, if present:
      if(kimp_.gt.0)then
          do ll=1,lrzmax
            tr1(ll)=bnumb(kimp_)
          enddo ! ll
          call frsplft(lrzmax,equilpsp(1),tr1(1),mfm1,eqpsi,zzi(1,kk))
CMPIINSERT_IF_RANK_EQ_0    
          WRITE(*,*)'frstup: kk,k,zzi(1,kk)=', kk,kimp_,zzi(1,kk)
CMPIINSERT_ENDIF_RANK
      endif


      
c..................................................................
c     Compute zeff(lr_) on Freya mesh.
c..................................................................

CMPIINSERT_IF_RANK_EQ_0      
      WRITE(*,*) 'frstup/before frsplft.  zeff(1:lrz)=',zeff(1:lrz)
CMPIINSERT_ENDIF_RANK
      call frsplft(lrzmax,equilpsp(1),zeff(1),mfm1,eqpsi,zeffctv(1))
      !YuP[2022-12] For some reason, zeffctv may become 0.999999999999999

      !YuP[2021-01-11] if(nprim.eq.1 .and. nimp.eq.1) then
       !Removed if(nprim.eq.1 .and. nimp.eq.1) condition,
       !so now it is used for any nprim and nimp (why not?)
CMPIINSERT_IF_RANK_EQ_0      
       WRITE(*,*) 'copying Zeff into zzi...'
CMPIINSERT_ENDIF_RANK
       !YuP[2021-01-11] was zzi(j,nion+2)=zeff(j)
       !YuP[2021-01-11] changed to nion+1, to match zfreya.f.
       !See zfreya.f(6076): Zeff(i=1,...mfm1) array has been stored in zzi(i,nion+1)
       kk=nion+1 ! Note: nion=nprim+nimp
       do j=1,mfm1
         !YuP[2021-01-11]: use interpolated array zeffctv(), not zeff() !!!
         zzi(j,kk)= zeffctv(j) !== Z_eff(rho); saved into kk=nion+1 index
!--CMPIINSERT_IF_RANK_EQ_0      
!         WRITE(*,*) 'j,zzi(j,nion+1) = ',j,zzi(j,nion+1)
!--CMPIINSERT_ENDIF_RANK
       enddo
CMPIINSERT_IF_RANK_EQ_0      
       WRITE(*,*) 'MIN/MAX of zzi(:,nion+1) =',
     +  minval(zeffctv(1:mfm1)),maxval(zeffctv(1:mfm1))
CMPIINSERT_ENDIF_RANK
      !YuP[2021-01-11] endif
c
c..................................................................
c     Change eqpsi back to convention at call to this subroutine.
c..................................................................

      do 101 l=1,nconteqn
        eqpsi(l)=-eqpsi(l)
 101  continue

c..................................................................
c     For running NFREYA with CQL3D, we assume iborb=0; so
c     it is necessary to set only the endpoints of potsid
c..................................................................

      potsid(1)=-psimag
      potsid(mf)=-psilim
      elong=0
      codeid="twodee"
      rin=rmincon
CMPIINSERT_IF_RANK_EQ_0      
      WRITE(*,*)'frsetup: rmincon,rmaxcon=',rmincon,rmaxcon
CMPIINSERT_ENDIF_RANK
      rmax=rmaxcon
      zminn=zmincon
      zmaxx=zmaxcon
      zax=0.
      mi=nnr
      mj=nnz
      zshift1=zshift

      return
      end subroutine frstup
