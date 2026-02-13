c
c
      subroutine restvty
      implicit integer (i-n), real*8 (a-h,o-z)
      include 'param.h'
      include 'comm.h'


c..................................................................
c     Routine computes resistivity and related quantities
c     for electron resistivity calculations. A minimum of
c     jx=125 should be used or the electron-electron fokker-planck
c     energy transfer term will be large (should be 0).
c..................................................................

      if (kelec .gt. ngen) return

c..................................................................
c     [ old: resist is the resistivity - the axis toroidal electric
c     field divided by the flux surface averaged parallel current. ]
c
c
c     If efflag="toroidal", then code electric field is
c                           assumed to be purely toroidal,
c                           varying like elecfld*(Rmag/R).
c     If ifflag="parallel", then code electric field is
c                           assumed to be purely parallel,
c                           varying like elecfld*(Rmag/R).
c     Code current density is purely parallel, and varies
c       on a flux surface as 
c       j_par = j_midplane * B(z)/B(0)(=bpsi(l))
c     resist=Resistivity, calc'd from distn fnctn results
c                  =<E_phi/R>/<j_phi/R>, is a "toroidal" resistivity.
c                  Except, if efswtchn.eq."neo_hh" .and. 
c                    cqlpmod.ne."enabled" ==>
c                    resist=(pol cross-section-area avg of E)/currpar
c                    and currpar is sum of Hinton-Hazeltine neoclassical
c                    current + runaway current.

c     resistn=<E_parall*B>/<j_parall*B>
c        
c     rovsn=  <E_parall*B>/<j_parall*B>/sptzr
c       
c
c     rovsloc is the local resistivity E_par/j_par(no flux averaged),
c     meaningful in cqlpmod only (CQLP)
c
c     If efswtchn="neo_hh" and cqlpmod.ne."enabled":
c                           resist= (area avg of E)/currpar
c                           rovs = resist/(H&H neoclassical value).
c.......................................................................

      if (l_ .eq. lmdpln_) then
        resist=0.0
        resistn=0.0
      endif
      resistlo=0.0
      if (abs(currm(kelec,l_)) .gt. 1.e-10) then
         
         if (l_ .eq. lmdpln_) then
C%OS  old:   resist=elecfld(lr_)/(300.*curr(kelec,lr_))
            

            if (efflag.ne."parallel") then

            if (efswtchn.eq."neo_hh" .and. cqlpmod.ne."enabled") then
              if (abs(currpar(lr_)) .gt. 1.e-10)then !YuP[2020-01-02] added if()
                resist=elecfld(lr_) / 300. /(currpar(lr_)*3.e9) *rmag*
     +           fpsi(lr_)  / bmod0(lr_) * onovrp(2,lr_)/ 
     +           psiavg(2,lr_)
              endif !YuP
            else ! For CQLP, only at l_=1 (midplane):
               resist=elecfld(lr_) / 300. / currm(kelec,l_) * rmag * 
     +              bmod0(lr_) / fpsi(lr_)
            endif
            resistn=elecfld(lr_) / 300. / currm(kelec,l_) * rmag *
     +           fpsi(lr_)  / bmod0(lr_) * onovrp(2,lr_)/ 
     +           psiavg(2,lr_)

            else                          !I.E., efflag.eq."parallel"

            if (efswtchn.eq."neo_hh" .and. cqlpmod.ne."enabled") then
              if (abs(currpar(lr_)) .gt. 1.e-10)then !YuP[2020-01-02] added if()
               resist=elecfld(lr_)/300.*rmag*onovrp(2,lr_)/onovrp(1,lr_)
     +              /(currpar(lr_)*3.e9)
              endif !YuP
            else
               resist=elecfld(lr_) / 300. / currm(kelec,l_) * rmag * 
     +              onovpsir3(lr_) / onovrp(2,lr_)
            endif
            resistn=elecfld(lr_) / 300. / currm(kelec,l_) * rmag *
     +           psiovr(lr_)/psiavg(2,lr_)

            endif

         endif

         if (cqlpmod.eq."enabled") then   !Local along fld line resist
c%OS  if (mod(nummods,10).le.4 .or. n.ge.nontran) then
            resistlo=elecfld(lr_)/300.*rmag*fpsi(lr_)/bmidplne(lr_)/
     +           (psis(l_)*solrs(l_)**2) / currm(kelec,l_)
c%OS  else
c%OS  resistlo=elecfld(lr_)/300.*rmag*fpsi(lr_)/bmidplne(lr_)/
c%OS  /           0.125/(psis(l_)+psis(l_+1))/(solrs(l_)+solrs(l_+1))**2
c%OS  /                                                  /currm(kelec,l_)
c%OS  endif
         endif
      endif !if (abs(currm(kelec,l_)) .gt. 1.e-10)


      rovsloc(l_)=resistlo/(sptzr(l_)+em90)  !CQLP only

c..................................................................
c     rovs(lr_)=computed resistivity / Spitzer resistivity 
c     Spitzer resistivity includes Zeff variation.
c..................................................................

      if (l_ .eq. lmdpln_) then
         if (efswtchn.eq."neo_hh" .and. cqlpmod.ne."enabled") then
            rovs(lr_)=resist/(zreshin(lr_)*sptzr(lmdpln_)+em90)
         else !(cqlpmod.eq."enabled") (at midplane only)
            rovs(lr_)=resist/(sptzr(lmdpln_)+em90)
         endif
         rovsn(lr_)=resistn/(sptzr(lmdpln_)+em90)
         elect=clight*btor*300.*resist/(radmaj*2.*pi)
         eratio=elect/(elecr(lr_)+em90)

c..................................................................
c     vparl=average parallel velocity of distribution
c..................................................................

         if(cqlpmod .ne. "enabled")then
         vparl=curr(kelec,lr_)/(charge*reden(kelec,lr_))
         vpovth=vparl/vth(kelec,lr_)    !CQL3D
         else !(cqlpmod.eq."enabled") ! YuP[2021-02-26]
         vparl=currm(kelec,l_)/(charge*denpar(kelec,ls_))
         vpovth=vparl/vthpar(kelec,ls_) !CQLP
         endif

c..................................................................c
c     eovedd is E-toroidal/E-Dreicer at the magnetic field
c..................................................................

         eovedd=elecfld(lr_)/(elecr(lr_)+em90)

      endif
c
      return
      end
