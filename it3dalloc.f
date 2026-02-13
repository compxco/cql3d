c
c
      subroutine it3dalloc
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

      dimension istat(18)
cdir$ nobounds

c.................................................................
c     Setup CSR storage for iterative, sparse matrix solve,
c     coefficient matrix and rhs/solution matrix.
c     Storage is allocated depending on the settings of soln_method.
c     Storage is provided for coefficient matrix for up to the complete
c     set of equations:
c     soln_method='it3dv'  ==> number of eqns = sum l_=1:lrz {inew*jx}, 
c         where inew=iyh_(l_) + itl_(l_) - 1
c         The number of columns = sum l_=1:lrz {inew*jx*9}
c         (Usually there are 9 coeffs per eqn; the few 12 coeff cases
c          are more than offset the few 2 and 3 coeff cases.)
c     soln_method='it3drv' ==> 
c     
c     
c.................................................................



c     For soln_method='itsol' or 'itsol1', allocate sufficient
c     storage for flux-surface soln with largest number or eqns (l_=1).
c     Specific shape of abd_lapack will be set in impanvc0 for
c     soln_method='itsol1' case.

CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'it3dalloc: entering...  soln_method=',
     +           soln_method
CMPIINSERT_ENDIF_RANK

         

      if (   soln_method.eq.'itsol' .or. soln_method.eq.'it3dv'
     +  .or. soln_method.eq.'it3drv') then

CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'icsrij,icsrip,icsrikry,lfil,iwk_ilu = ',
     +              icsrij,icsrip,icsrikry,lfil,iwk_ilu
CMPIINSERT_ENDIF_RANK

         istat(1:18)=0
         allocate( a_csr(icsrij),stat=istat(1))
         allocate(ja_csr(icsrij),stat=istat(2))
         allocate(ia_csr(icsrip),stat=istat(3))
         allocate(alu(iwk_ilu),stat=istat(4))
         allocate(jlu(iwk_ilu),stat=istat(5))
         allocate(ju(icsrip),stat=istat(6))
         allocate(jw_ilu(icsri2),stat=istat(7))
         allocate(w_ilu(icsrip),stat=istat(8))
         allocate(rhs0(icsrip),stat=istat(9))
         allocate( sol(icsrip),stat=istat(10))
         allocate(vv(icsrikry),stat=istat(11))
         i2=11
         if (soln_method .eq. 'it3drv') then
            write(*,*)'icsrijr,icsrijc = ',icsrijr,icsrijc
            allocate(ipofi(iy,lrz),stat=istat(12))
            allocate( ar_csr(icsrijr),stat=istat(13))
            allocate(jar_csr(icsrijr),stat=istat(14))
            allocate(iar_csr(icsrip),stat=istat(15))
            allocate(iac_csr(icsrip),stat=istat(18))
            allocate( ac_csr(icsrijc),stat=istat(16))
            allocate(jac_csr(icsrijc),stat=istat(17))
            i2=18
         endif
         istat0=0
         do i=1,i2 !!12
            if (istat(i).ne.0) then
               write(*,*)'it3dalloc:  i,istat(i)=',i,istat(i)
               istat0=max(istat0,1)
            endif
         enddo
         if (istat0.ne.0) stop 'Allocation problem in it3dalloc'
c$$$  
         
      elseif ( soln_method.eq.'itsol1' ) then
      
CMPIINSERT_IF_RANK_EQ_0
         WRITE(*,*)'lapacki,lapackj,icsrij,icsrip,iwk_ilu = ',
     +              lapacki,lapackj,icsrij,icsrip,iwk_ilu
CMPIINSERT_ENDIF_RANK

         allocate(abd_lapack(lapacki,lapackj),stat=istat(1))
         allocate(a_csr(icsrij),stat=istat(1))
         allocate(ja_csr(icsrij),stat=istat(2))
         allocate(ia_csr(icsrip),stat=istat(3))
         allocate(alu(iwk_ilu),stat=istat(4))
         allocate(jlu(iwk_ilu),stat=istat(5))
         allocate(ju(icsrip),stat=istat(6))
         allocate(jw_ilu(icsri2),stat=istat(7))
         allocate(w_ilu(icsrip),stat=istat(8))
         allocate(rhs0(icsrip),stat=istat(9))
         allocate(sol(icsrip),stat=istat(10))
         allocate(vv(icsrikry),stat=istat(11))
         istat0=0
         do i=1,11
            if (istat(i).ne.0) then
               write(*,*)'it3dalloc:  i,istat(i)=',i,istat(i)
               istat0=max(istat0,1)
            endif
         enddo
         if (istat0.ne.0) stop 'Allocation problem in it3dalloc'
         
      endif
      
CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'it3dalloc: exiting...'
CMPIINSERT_ENDIF_RANK

      return
      end
      
c=======================================================================      
      
      subroutine it3ddalloc 
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'it3ddalloc: deallocating...'
CMPIINSERT_ENDIF_RANK

      if ( soln_method.eq.'itsol' .or. soln_method.eq.'it3dv' 
     +     .or. soln_method.eq.'it3drv') then
         if(associated(a_csr)) deallocate(a_csr)
         if(associated(ja_csr)) deallocate(ja_csr)
         if(associated(ia_csr)) deallocate(ia_csr)
         if(associated(alu)) deallocate(alu)
         if(associated(jlu)) deallocate(jlu)
         if(associated(ju)) deallocate(ju)
         if(associated(jw_ilu)) deallocate(jw_ilu)
         if(associated(w_ilu)) deallocate(w_ilu)
         if(associated(rhs0)) deallocate(rhs0)
         if(associated(sol)) deallocate(sol)
         if(associated(vv)) deallocate(vv)
         if (soln_method.eq.'it3drv') then
          if(associated(ar_csr)) deallocate(ar_csr)
          if(associated(jar_csr)) deallocate(jar_csr)
          if(associated(iar_csr)) deallocate(iar_csr)
          if(associated(ac_csr)) deallocate(ac_csr)
          if(associated(jac_csr)) deallocate(jac_csr)
          if(associated(iac_csr)) deallocate(iac_csr)
         endif
         
      elseif ( soln_method.eq.'itsol1' ) then
         deallocate (abd_lapack,a_csr,ja_csr,ia_csr,alu,jlu,ju,
     +        jw_ilu,w_ilu,rhs0,sol,vv)
         
      elseif ( soln_method.eq.'it3drv' ) then
         
      endif
      
      return
      end


c=======================================================================      
      
      subroutine de_alloc ! YuP[11-2017] more deallocation
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

c  The purpose of this subroutine is to ensure deallocation of variables
c  at the end of a run.  If running cql3d as a stand-alone code, 
c  deallocation would occur automatically.
c  With present TRANSP coupling, invoking cql3d through
c  a subroutine call, allocated memory would build up with
c  each call to cql3d.

c  Generally, all pointers that are defined in comm.h
c  can be deallocated here.
c  Not all of them are allocated during run.
c  It depends on particular setup/cqlinput.
c  So, always check - add if(ASSOCIATED(array_name)) 
c  in front of deallocate(array_name) statement.
c  Other arrays like abd(),...ampfda() [few lines below]
c  are NOT in comm.h, but rather in local common blocks.
c  So, these common blocks must be added here, too.

!      real*8 abd(:,:)
!      pointer abd 
!      integer ipivot(:) 
!      pointer ipivot
!      common/gauss_/ipivot
!      common/gauss_abd/abd

!      dimension ampfda(:,:),ampfdd(:,:)
!      pointer ampfda,ampfdd
!      common /ampf/ ampfda,ampfdd ! from impavnc0


CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'de_alloc: START deallocating...'
CMPIINSERT_ENDIF_RANK

      if(ASSOCIATED(rhs0)) then
        ! rhs0 and other arrays are allocated => deallocate
        deallocate(rhs0)
      endif
!      if(ASSOCIATED(abd)) then
!        deallocate(abd)
!      endif
!      if(ASSOCIATED(ipivot)) then
!        deallocate(ipivot)
!      endif
!      if(ASSOCIATED(ampfda)) then
!        deallocate(ampfda,ampfdd,ampfaa)
!      endif
      if(ASSOCIATED(fh)) then
        deallocate(fh,fg)
      endif
      if(ASSOCIATED(urfb)) then
        deallocate(ilim1,ilim2,ifct1,ifct2,jminray,jmaxray,lloc,llray)
        deallocate(urfb,urfc,cosmz,urftmp,urfpwr,urfpwrc,urfpwrl)
        deallocate(g_,alfag,argmnt,ilim1d,ilim2d,ilim1dd,ilim2dd)
        deallocate(psiloc,scalurf,cwexde,cweyde,cwezde,delpwr,fluxn)
        deallocate(seikon,spsi,sdpwr,sbtot,sene,salphac,salphal)
        deallocate(ws,wr,wz,wnpar,wdnpar,wnper,wphi)
        deallocate(ilowp,iupp,ifct1_,ifct2_)
      endif

      if(ASSOCIATED(f)) then
        deallocate(f)
      endif
      if(ASSOCIATED(favg)) then
        deallocate(favg)
      endif
      if(ASSOCIATED(fxsp)) then
        deallocate(fxsp)
      endif
      if(ASSOCIATED(f_)) then
        deallocate(f_)
      endif
      if(ASSOCIATED(spasou)) then
        deallocate(spasou)
      endif
      if(ASSOCIATED(velsou)) then
        deallocate(velsou)
      endif
      if(ASSOCIATED(velsou2)) then
        deallocate(velsou2)
      endif
      if(ASSOCIATED(source)) then
        deallocate(source)
      endif
      if(ASSOCIATED(gone)) then
        deallocate(gone)
      endif
      if(ASSOCIATED(egylosa)) then
        deallocate(egylosa)
      endif
!      if(ASSOCIATED(i0tran)) then
!        deallocate(i0tran)
!      endif
      if(ASSOCIATED(cal)) then
        deallocate(cal,cbl,ccl,cdl,cel,cfl,eal,ebl,scal,cet,cex)
        deallocate(synca,syncd,taulos,psi0bar)
        deallocate(delecfld0,elecfldn,delecfld0n,elecn,di,dj,ss,dcofleg)
        deallocate(dpcosz,ssy,ssyy,ssyi,ssyyy,pcurr,pcurrm,pdens,pdenm)
        deallocate(pengy,pengym)
        deallocate(wtfl0,wtflm,jflbin)
        deallocate(currv,pwrrf,pwrrfs,wflux,feta,fetb)
        deallocate(sgaint,entr)
      endif
      if(ASSOCIATED(f_aveth)) then
        deallocate(f_aveth)
      endif
      if(ASSOCIATED(densz)) then
        deallocate(densz,waa,wbb,cosz,dtau,sinz,tanz,yz,tot)
        deallocate(vflux,sincosba)
      endif
      if(ASSOCIATED(sovt)) then
        deallocate(sovt)
      endif
      if(ASSOCIATED(sigsxr)) then
        deallocate(sigsxr)
      endif

      if(ASSOCIATED(pentr)) then
        deallocate(pentr)
      endif
      if(ASSOCIATED(sounor)) then
        deallocate(sounor)
      endif
      if(ASSOCIATED(tamt1)) then
        deallocate(tamt1,tamt2)
      endif
      if(ASSOCIATED(cqlb)) then
        deallocate(cqlb,cqlc,cqle,cqlf)
      endif
      if(ASSOCIATED(frn_2)) then
        deallocate(frn_1,frn_2)
      endif
      if(ASSOCIATED(frn)) then
        deallocate(frn)
      endif
      if(ASSOCIATED(fvn)) then
        deallocate(fvn)
      endif
      if(ASSOCIATED(fvn_1)) then
        deallocate(fvn_1)
      endif
      if(ASSOCIATED(dl)) then
        deallocate(dl)
      endif
      if(ASSOCIATED(d_rr)) then
        deallocate(d_rr)
      endif
      if(ASSOCIATED(d_r)) then
        deallocate(d_r)
      endif
      if(ASSOCIATED(f_vtor)) then
        deallocate(f_vtor)
      endif
      if(ASSOCIATED(fnhalf)) then
        deallocate(fnhalf)
      endif
      if(ASSOCIATED(fnp0)) then
        deallocate(fnp0)
      endif
      if(ASSOCIATED(fnp1)) then
        deallocate(fnp1)
      endif
      if(ASSOCIATED(dls)) then
        deallocate(dls)
      endif
      if(ASSOCIATED(fedge)) then
        deallocate(fedge)
      endif
      if(ASSOCIATED(bndmats)) then
        deallocate(bndmats)
      endif
      if(ASSOCIATED(wcqlb)) then
        deallocate(wcqlb,wcqlc,wcqle,wcqlf)
      endif
      if(ASSOCIATED(rdcb)) then
        deallocate(rdcb,rdcc,rdce,rdcf)
      endif
      if(ASSOCIATED(ilpm1ef)) then
        deallocate(ilpm1ef)
      endif
      if(ASSOCIATED(rhspar)) then
        deallocate(rhspar)
      endif
      if(ASSOCIATED(fg_)) then
        deallocate(f_lm,f_lp,f_up,eg_,fg_)
      endif
      if(ASSOCIATED(csv)) then
        deallocate(csv)
      endif
      if(ASSOCIATED(deltarz)) then
        deallocate(deltarho,deltarhop,deltarz)
      endif
      if(ASSOCIATED(fgg)) then
        deallocate(fgg,egg,temp1,temp2,temp3,temp4,temp5,temp6)
        deallocate(xllji,xppji)
      endif
      if(ASSOCIATED(jbm1)) then
        deallocate(jbm1,jb0,jbp1)
      endif
      if(ASSOCIATED(nrayelt)) then
        deallocate(nrayelt,jslofas,nurefls,keiks,jpes,jpis,istarts)
        deallocate(iprmt5,jhlfs,sxxrt,skpsi,skth,skphi)
        deallocate(lrayelt,delpwr0,nrayelt0)
      endif
      if(ASSOCIATED(eqdell)) then
        deallocate(eqdell,eqbpol,solr,solz,drpmconz)
      endif
      if(ASSOCIATED(cynt2_)) then
        deallocate(cynt2_,vpint_,vptb_,cosovb,bovcos,adv)
      endif

CMG added 11/13/2017
      if(ASSOCIATED(dentarget)) then
         deallocate(dentarget)
      endif
      if(ASSOCIATED(sx)) then
         deallocate(sx)
      endif
      if(ASSOCIATED(xmdx)) then
         deallocate(xmdx)
      endif
      if(ASSOCIATED(cosz1)) then
         deallocate(cosz1)
      endif
      if(ASSOCIATED(sinz1)) then
         deallocate(sinz1)
      endif
      if(ASSOCIATED(sinz2)) then
         deallocate(sinz2)
      endif
      if(ASSOCIATED(thtf1)) then
         deallocate(thtf1)
      endif
      if(ASSOCIATED(thtf2)) then
         deallocate(thtf2)
      endif
      if(ASSOCIATED(alfi)) then
         deallocate(alfi)
      endif
      if(ASSOCIATED(alfa)) then
         deallocate(alfa)
      endif
      if(ASSOCIATED(truncd)) then
         deallocate(truncd)
      endif
      if(ASSOCIATED(w_ilu)) then
         deallocate(w_ilu)
      endif
      if(ASSOCIATED(sol)) then
         deallocate(sol)
      endif
      if(ASSOCIATED(vv)) then
         deallocate(vv)
      endif
      if(ASSOCIATED(jw_ilu)) then
         deallocate(jw_ilu)
      endif
      if(ASSOCIATED(ampfln)) then
         deallocate(ampfln)
      endif
      if(ASSOCIATED(ampflg)) then
         deallocate(ampflg)
      endif
      if(ASSOCIATED(ampfa)) then
         deallocate(ampfa)
      endif
      if(ASSOCIATED(ampfb)) then
         deallocate(ampfb)
      endif
      if(ASSOCIATED(ampfc)) then
         deallocate(ampfc)
      endif
      if(ASSOCIATED(ampf2ebar)) then
         deallocate(ampf2ebar)
      endif
              write(*,*)'it3dalloc-1.0'

      if(ASSOCIATED(dym5)) deallocate(dym5)
      if(ASSOCIATED(dyp5)) deallocate(dyp5)
      if(ASSOCIATED(eym5)) deallocate(eym5)
      if(ASSOCIATED(eyp5)) deallocate(eyp5)
      if(ASSOCIATED(y)) deallocate(y)
      if(ASSOCIATED(dy)) deallocate(dy)
      if(ASSOCIATED(yptb)) deallocate(yptb)
      if(ASSOCIATED(coss)) deallocate(coss)
      if(ASSOCIATED(cynt2)) deallocate(cynt2)
      if(ASSOCIATED(batot)) deallocate(batot)
      if(ASSOCIATED(lmax)) deallocate(lmax)
      if(ASSOCIATED(vpint)) deallocate(vpint)
      if(ASSOCIATED(psiiv)) deallocate(psiiv)
      if(ASSOCIATED(psiba)) deallocate(psiba)
      if(ASSOCIATED(psisq)) deallocate(psisq)
      if(ASSOCIATED(psicu)) deallocate(psicu)
      if(ASSOCIATED(psiqu)) deallocate(psiqu)
      if(ASSOCIATED(bavpd)) deallocate(bavpd)
      if(ASSOCIATED(bavdn)) deallocate(bavdn)
      if(ASSOCIATED(psiir)) deallocate(psiir)
      if(ASSOCIATED(vderb)) deallocate(vderb)
      if(ASSOCIATED(sinn)) deallocate(sinn)
      if(ASSOCIATED(tann)) deallocate(tann)
      if(ASSOCIATED(ymid)) deallocate(ymid)
              write(*,*)'it3dalloc-1.1'
      if(ASSOCIATED(tau)) deallocate(tau)
      if(ASSOCIATED(vptb)) deallocate(vptb)
      if(ASSOCIATED(zboun)) deallocate(zboun)
      if(ASSOCIATED(idx)) deallocate(idx)
      if(ASSOCIATED(imax)) deallocate(imax)
      if(ASSOCIATED(dz)) deallocate(dz)
      if(ASSOCIATED(pol)) deallocate(pol)
      if(ASSOCIATED(solrz)) deallocate(solrz)
      if(ASSOCIATED(solzz)) deallocate(solzz)
      if(ASSOCIATED(thtab)) deallocate(thtab)
      if(ASSOCIATED(z)) deallocate(z)
      if(ASSOCIATED(zmid)) deallocate(zmid)
      if(ASSOCIATED(bbpsi)) deallocate(bbpsi)
      if(ASSOCIATED(bpolz)) deallocate(bpolz)
      if(ASSOCIATED(btorz)) deallocate(btorz)
      if(ASSOCIATED(consnp)) deallocate(consnp)
      if(ASSOCIATED(ptime)) deallocate(ptime)
      if(ASSOCIATED(pefld)) deallocate(pefld)
      if(ASSOCIATED(rovsp)) deallocate(rovsp)
      if(ASSOCIATED(restp)) deallocate(restp)
      if(ASSOCIATED(restnp)) deallocate(restnp)
      if(ASSOCIATED(vpov)) deallocate(vpov)
      if(ASSOCIATED(es)) deallocate(es)
      if(ASSOCIATED(bpsi)) deallocate(bpsi)
      if(ASSOCIATED(d2bpsi)) deallocate(d2bpsi)
              write(*,*)'it3dalloc-1.2'
      if(ASSOCIATED(d2solrz)) deallocate(d2solrz)
      if(ASSOCIATED(d2solzz)) deallocate(d2solzz)
      if(ASSOCIATED(d2bpolz)) deallocate(d2bpolz)
      if(ASSOCIATED(d2btorz)) deallocate(d2btorz)
      if(ASSOCIATED(d2thtpol)) deallocate(d2thtpol)
      if(ASSOCIATED(d2es)) deallocate(d2es)
      if(ASSOCIATED(thtpol)) deallocate(thtpol)
      if(ASSOCIATED(esfi)) deallocate(esfi)
      if(ASSOCIATED(psiesfi)) deallocate(psiesfi)
      if(ASSOCIATED(psifi)) deallocate(psifi)
      if(ASSOCIATED(espsifi)) deallocate(espsifi)
      if(ASSOCIATED(soupp)) deallocate(soupp)
      if(ASSOCIATED(pcurra)) deallocate(pcurra)
      if(ASSOCIATED(pdenra)) deallocate(pdenra)
      if(ASSOCIATED(pfdenra)) deallocate(pfdenra)
      if(ASSOCIATED(pfcurra)) deallocate(pfcurra)
      if(ASSOCIATED(pucrit)) deallocate(pucrit)
      if(ASSOCIATED(peoe0)) deallocate(peoe0)
      if(ASSOCIATED(psrc)) deallocate(psrc)
      if(ASSOCIATED(peoed)) deallocate(peoed)
      if(ASSOCIATED(cint2)) deallocate(cint2)
      if(ASSOCIATED(dx)) deallocate(dx)
      if(ASSOCIATED(dxi)) deallocate(dxi)
      if(ASSOCIATED(ifp)) deallocate(ifp)
      if(ASSOCIATED(sg)) deallocate(sg)
      if(ASSOCIATED(sgx)) deallocate(sgx)
      if(ASSOCIATED(sgxx)) deallocate(sgxx)
      if(ASSOCIATED(sh)) deallocate(sh)
      if(ASSOCIATED(shx)) deallocate(shx)
      if(ASSOCIATED(shxx)) deallocate(shxx)
      if(ASSOCIATED(shxxx)) deallocate(shxxx)
              write(*,*)'it3dalloc-1.3'
      if(ASSOCIATED(tam1)) deallocate(tam1)
      if(ASSOCIATED(tam2)) deallocate(tam2)
      if(ASSOCIATED(tam3)) deallocate(tam3)
      if(ASSOCIATED(tam4)) deallocate(tam4)
      if(ASSOCIATED(tam5)) deallocate(tam5)
      if(ASSOCIATED(tam6)) deallocate(tam6)
      if(ASSOCIATED(tam7)) deallocate(tam7)
      if(ASSOCIATED(tam8)) deallocate(tam8)
      if(ASSOCIATED(tam9)) deallocate(tam9)
      if(ASSOCIATED(tam10)) deallocate(tam10)
      if(ASSOCIATED(tam11)) deallocate(tam11)
      if(ASSOCIATED(tam12)) deallocate(tam12)
      if(ASSOCIATED(tam13)) deallocate(tam13)
      if(ASSOCIATED(tam14)) deallocate(tam14)
      if(ASSOCIATED(tam15)) deallocate(tam15)
      if(ASSOCIATED(tam16)) deallocate(tam16)
      if(ASSOCIATED(tam17)) deallocate(tam17)
      if(ASSOCIATED(tam18)) deallocate(tam18)
      if(ASSOCIATED(tam19)) deallocate(tam19)
      if(ASSOCIATED(tam20)) deallocate(tam20)
      if(ASSOCIATED(tam21)) deallocate(tam21)
      if(ASSOCIATED(tam22)) deallocate(tam22)
      if(ASSOCIATED(tam23)) deallocate(tam23)
      if(ASSOCIATED(tam24)) deallocate(tam24)
      if(ASSOCIATED(tam25)) deallocate(tam25)
      if(ASSOCIATED(tam26)) deallocate(tam26)
      if(ASSOCIATED(tam27)) deallocate(tam27)
      if(ASSOCIATED(tam28)) deallocate(tam28)
      if(ASSOCIATED(tam29)) deallocate(tam29)
      if(ASSOCIATED(tam30)) deallocate(tam30)
      if(ASSOCIATED(x)) deallocate(x)
              write(*,*)'it3dalloc-1.4'
      if(ASSOCIATED(xmidpt)) deallocate(xmidpt)
      if(ASSOCIATED(xi)) deallocate(xi)
      if(ASSOCIATED(xsq)) deallocate(xsq)
      if(ASSOCIATED(x3i)) deallocate(x3i)
      if(ASSOCIATED(x2i)) deallocate(x2i)
      if(ASSOCIATED(xcu)) deallocate(xcu)
      if(ASSOCIATED(xcenter)) deallocate(xcenter)
      if(ASSOCIATED(xcensq)) deallocate(xcensq)
      if(ASSOCIATED(xcent3)) deallocate(xcent3)
      if(ASSOCIATED(uoc)) deallocate(uoc)
      if(ASSOCIATED(enerkev)) deallocate(enerkev)
      if(ASSOCIATED(gamma)) deallocate(gamma)
      if(ASSOCIATED(gamsqr)) deallocate(gamsqr)
      if(ASSOCIATED(gamcub)) deallocate(gamcub)
      if(ASSOCIATED(gammi)) deallocate(gammi)
      if(ASSOCIATED(gamm2i)) deallocate(gamm2i)
      if(ASSOCIATED(gamm1)) deallocate(gamm1)
      if(ASSOCIATED(tcsgm1)) deallocate(tcsgm1)
      if(ASSOCIATED(gamefac)) deallocate(gamefac)
      if(ASSOCIATED(ident)) deallocate(ident)
      if(ASSOCIATED(temc1)) deallocate(temc1)
      if(ASSOCIATED(temc2)) deallocate(temc2)
      if(ASSOCIATED(temc3)) deallocate(temc3)
      if(ASSOCIATED(temc4)) deallocate(temc4)
      if(ASSOCIATED(itemc1)) deallocate(itemc1)
      if(ASSOCIATED(itemc2)) deallocate(itemc2)
              write(*,*)'it3dalloc-1.5'
      if(ASSOCIATED(l_lower)) deallocate(l_lower)
      if(ASSOCIATED(lpt)) deallocate(lpt)
      if(ASSOCIATED(mun)) deallocate(mun)
      if(ASSOCIATED(fll)) deallocate(fll,STAT=istat)
              write(*,*)'it3dalloc fll', istat
      if(ASSOCIATED(xpar)) deallocate(xpar,STAT=istat)
              write(*,*)'it3dalloc xpar', istat
      if(ASSOCIATED(rheads)) deallocate(rheads,STAT=istat)
              write(*,*)'it3dalloc rheads', istat
      if(ASSOCIATED(dfvlle)) deallocate(dfvlle,STAT=istat)
              write(*,*)'it3dalloc dfvlle', istat
      if(ASSOCIATED(dfvlli)) deallocate(dfvlli,STAT=istat)
              write(*,*)'it3dalloc dfvlli', istat
cBH180430: Next statement causing Seg fault. Don't see why??
!      if(ASSOCIATED(xperp)) deallocate(xperp,STAT=istat)
!              write(*,*)'it3dalloc xperp', istat
!              write(*,*)xl,jmaxxl
!      if(ASSOCIATED(xl)) deallocate(xl,STAT=istat)
!              write(*,*)'it3dalloc xl', istat
!!      if(ASSOCIATED(jmaxxl)) deallocate(jmaxxl,STAT=istat)
!!              write(*,*)'it3dalloc jmaxxl', istat

!      if(ASSOCIATED(xlm)) deallocate(xlm,STAT=istat)
!              write(*,*)'it3dalloc xlm', istat
!      if(ASSOCIATED(dxl)) deallocate(dxl,STAT=istat)
!              write(*,*)'it3dalloc dxl', istat
!      if(ASSOCIATED(fl)) deallocate(fl,STAT=istat)
!              write(*,*)'it3dalloc fl', istat
!      if(ASSOCIATED(fl1)) deallocate(fl1,STAT=istat)
!              write(*,*)'it3dalloc fl1', istat
! YuP: A problem with deallocation: sometimes ok, sometimes
! crashes (using the same exe file).
!      if(ASSOCIATED(fl2)) deallocate(fl2,STAT=istat)
!          write(*,*)'it3dalloc fl2', istat

      if(ASSOCIATED(pparea)) deallocate(pparea)
      if(ASSOCIATED(faci)) deallocate(faci)
      if(ASSOCIATED(pprps)) deallocate(pprps)
      if(ASSOCIATED(ppars)) deallocate(ppars)
      if(ASSOCIATED(dff)) deallocate(dff)
      if(ASSOCIATED(cthta)) deallocate(cthta)
      if(ASSOCIATED(cah)) deallocate(cah)
      if(ASSOCIATED(xm)) deallocate(xm)
      if(ASSOCIATED(so)) deallocate(so)
      if(ASSOCIATED(gon)) deallocate(gon)
      if(ASSOCIATED(dbb)) deallocate(dbb)
      if(ASSOCIATED(item1)) deallocate(item1)
      if(ASSOCIATED(item2)) deallocate(item2)
      if(ASSOCIATED(item3)) deallocate(item3)
      if(ASSOCIATED(item4)) deallocate(item4)
      if(ASSOCIATED(item5)) deallocate(item5)
      if(ASSOCIATED(item6)) deallocate(item6)
      if(ASSOCIATED(rhs)) deallocate(rhs)
              write(*,*)'it3dalloc-1.7'

      if(ASSOCIATED(da)) deallocate(da)
      if(ASSOCIATED(db)) deallocate(db)
      if(ASSOCIATED(dc)) deallocate(dc)
      if(ASSOCIATED(dd)) deallocate(dd)
      if(ASSOCIATED(de)) deallocate(de)
      if(ASSOCIATED(df)) deallocate(df)
      if(ASSOCIATED(ca)) deallocate(ca)
      if(ASSOCIATED(cb)) deallocate(cb)
      if(ASSOCIATED(cc)) deallocate(cc)
      if(ASSOCIATED(ce)) deallocate(ce)
      if(ASSOCIATED(cd)) deallocate(cd)
      if(ASSOCIATED(cf)) deallocate(cf)
              write(*,*)'it3dalloc-1.8'

      if(ASSOCIATED(bqlm)) deallocate(bqlm)
      if(ASSOCIATED(tem1)) deallocate(tem1)
      if(ASSOCIATED(tem2)) deallocate(tem2)
      if(ASSOCIATED(tem3)) deallocate(tem3)
      if(ASSOCIATED(tem4)) deallocate(tem4)
      if(ASSOCIATED(tem5)) deallocate(tem5)
      if(ASSOCIATED(tem6)) deallocate(tem6)
      if(ASSOCIATED(xhead)) deallocate(xhead)
      if(ASSOCIATED(xtail)) deallocate(xtail)
      if(ASSOCIATED(ytail)) deallocate(ytail)
      if(ASSOCIATED(yhead)) deallocate(yhead)
      if(ASSOCIATED(fpn)) deallocate(fpn)
        write(*,*)'it3dalloc-1.9'
      if(ASSOCIATED(dyi)) deallocate(dyi)
      if(ASSOCIATED(pleg)) deallocate(pleg)
      if(ASSOCIATED(tfl)) deallocate(tfl)
      if(ASSOCIATED(tbl)) deallocate(tbl)
      if(ASSOCIATED(tal)) deallocate(tal)
      if(ASSOCIATED(currvs)) deallocate(currvs)
      if(ASSOCIATED(dxm5)) deallocate(dxm5)
      if(ASSOCIATED(exm5)) deallocate(exm5)
      if(ASSOCIATED(dxp5)) deallocate(dxp5)
      if(ASSOCIATED(exp5)) deallocate(exp5)
      if(ASSOCIATED(pm)) deallocate(pm)
      if(ASSOCIATED(cog)) deallocate(cog)
      if(ASSOCIATED(choose)) deallocate(choose)
      if(ASSOCIATED(constp)) deallocate(constp)
      if(ASSOCIATED(sigmtt)) deallocate(sigmtt)
      if(ASSOCIATED(sigftt)) deallocate(sigftt)
      if(ASSOCIATED(xlndnz)) deallocate(xlndnz)
      if(ASSOCIATED(ix1)) deallocate(ix1)
      if(ASSOCIATED(ix2)) deallocate(ix2)
      if(ASSOCIATED(ix3)) deallocate(ix3)
      if(ASSOCIATED(ix4)) deallocate(ix4)
      if(ASSOCIATED(ix5)) deallocate(ix5)
      if(ASSOCIATED(ix6)) deallocate(ix6)
      if(ASSOCIATED(ix7)) deallocate(ix7)
      if(ASSOCIATED(ix8)) deallocate(ix8)
      if(ASSOCIATED(tom1)) deallocate(tom1)
      if(ASSOCIATED(tom2)) deallocate(tom2)
      if(ASSOCIATED(tom3)) deallocate(tom3)
      if(ASSOCIATED(tom4)) deallocate(tom4)
      if(ASSOCIATED(fctrl)) deallocate(fctrl)
     
CMG end added by  11/13/2017

      if(ASSOCIATED(delta_bdb0)) then
        deallocate(delta_bdb0)
      endif


CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,*)'de_alloc:  DONE deallocating'
CMPIINSERT_ENDIF_RANK
      
      return
      end
