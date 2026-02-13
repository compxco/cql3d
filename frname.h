c     frname.h
c***********************************************************************
c     Namelist variables for the NFREYA fr.... routines
c***********************************************************************

c................................................................
c     Freya namelist variables.   It is not safe to put these
c     into main cql3d namelist files name.h, as there could
c     be duplication of variable names.  If you want to combine
c     with namelist in name.h, then check all the following variables.
c     Might have to change a couple of variable names.
c     frname.h is used included into frcomm.h, which is used by
c     freya.f,frhexdrv.f,frinitl.f,frnfreya.f
c................................................................
      namelist/frsetup/ timbplt,beamon,btime,nameb,relnub
     &  ,anglev,angleh,ashape,aheigh,awidth,bcur,bptor,blenp,bshape
     &  ,bleni,bheigh,bwidth,bhfoc,bvfoc,bhdiv,bvdiv,ebkev,fbcur,nbeams
     &  ,naptr,alen,bvofset,bhofset,nsourc,sfrac1,mf,npart,npskip
     &  ,rpivot,zpivot,ranseed,fionx,nbinject
     &  ,xdebug
     &  ,a1rf,a2rf,wrfe,wrfi,nrfzon,rfzone,ichmod,betalm
     &  ,beamplse,beampon,beampoff
c     ONETWO DIVERGENCE
     &  ,nimp,nprim,frmod,fr_gyro,smooth,multiply,multiplyn,bmsprd
     &  ,frplt,nfrplt,inubpat,npat
     &  ,ibcur,ibcx,ibslow,iborb,iyoka,ishot,itime
     &  ,itrapfi
     &  ,iexcit,ilorent,mstate,izstrp,kdene
     &  ,kdeni,kdenz,ksvi,ksvz,ksve,krad,ngh,ngl,nouthx
     &  ,hdepsmth
     &  ,birth_pts_files,nbirth_pts_files,nbirth_pts,read_birth_pts
     &  ,ne_tk,ds_tk,fe_tk
c     Remove NBI source at psi outside of psicutoff:     
     +  ,psicutoff
     +  ,src_nbi_e ![2022-06-30] Add source to e_general matching NBI sources

cRemoved namelist, not used, BH070308:
c tfusbb,iddcal,fdbeam,nbinject,rfmode,rfon,rftime,rfpow,
c idamp,isrc,zsrc,fpsrc,phisrc,
c freq,rfrad1,rfrad2,wrfo,wrfx,rnormin,njqin,qine,qini,
c a1rf,a2rf,wrfe,wrfi,nrfzon,rfzone,ichmod,betalm
c relrf,nprf,iside,xkpar,nhigh,ykperp,navg
c nampel,pelrad,vpel,nbgpel,timpel
c ifus,iaslow,wtifus
c itrapech,
c nshell,nraysh,thetsh,phish,praysh
c wdelt,wgam,wohm,nqrad,qradr,qradin,refrad
c rnp,irfcur,ifb,rfcur,lmode,ifbprof,alphaf
