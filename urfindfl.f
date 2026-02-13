c
c
      subroutine urfindfl
      implicit integer (i-n), real*8 (a-h,o-z)

c..................................................................
c     This routine sets defaults for the "urf" module
c..................................................................

      include 'param.h'
      include 'name_decl.h'


c..................................................................
c     Set defaults
c..................................................................

      one=1.d0

      nrayn=1 !-YuP 101122: added here, instead of nraya in param.h;
              ! will be re-defined in urfsetup.
              ! Will be used to allocate arrays, mostly in urfalloc
      nrayelts=1 !-YuP: added here, instead of nrayelta in param.h;
              ! will be re-defined in urfsetup.
              ! will be used to allocate arrays, mostly in urfalloc.
              
c..................................................................
c     ieqbrurf designates the source of equilibrium data to be used by
c     xbram.  Appropriate values are: ieqbrurf=0 to use Brambilla
c     ieqbrurf=0, to use Brambilla analytic "equilibria",
c     =3, to use standard eqdsk input.
c     =4, to use extended eqdsk, including density, and
c     temperature profiles,...[output by GA ONETWO transport code].
c     If eqsource="tsc", ieqbrurf is reset to = 4.
c..................................................................

      ieqbrurf=1

c..................................................................
c     Step interval at which to recalc diffusion coefficients,
c     as a fraction of ncoef.
c..................................................................

      urfncoef=one

c..................................................................
c     number of elements in Bessel table at setup time
c..................................................................

      nbssltbl=10000

c..................................................................
c     damping on at step nondamp
c..................................................................

      nondamp=0

c..................................................................
c     maximum number of ray elements per ray for first call to R.T code
c     [Not presently implemented, BH060314]
c..................................................................
      do i=1,nmodsa
         nrfstep1(i)=100 ![nrfstep1 presently is not used]
      enddo
c..................................................................
c     scaling of power and nparallel- width in rays
c..................................................................

      do 20  i=1,nmodsa
        pwrscale(i)=one
        wdscale(i)=one
 20   continue

c..................................................................
c     nharms.gt.0, then calculate damping for harmonics
c     nharm1 to nharm1+(nharms-1), rather than according to nharm
c     in the rayop file.
c     This option is only viable for one rf mode, i.e., only one
c     of lh, ech, or fw may be enabled.
c     060214: Extended to multi-rf-wave types.
c..................................................................

      do i=1,nmodsa
         nharms(i)=0
         nharm1(i)=0
      enddo

c..................................................................
c     Additional time-dependent scaling of power
c..................................................................

      nurftime=0
      do  i=1,nbctimea
        pwrscale1(i)=one
        urftime(i)=0.0
      enddo

c..................................................................
c     number of steps in power ramp-up
c..................................................................

      nrfpwr=3

c..................................................................
c     number of iterations at full power, for first ray elements
c..................................................................

      nrfitr1=1

c..................................................................
c     number of iterations after additional ray data
c..................................................................

      nrfitr2=1

c..................................................................
c     number of steps adding new ray data
c..................................................................

      nrfitr3=2

c..................................................................
c     number of ray elements added at each step in addition of new data.
c     [Not presently implemented, BH060314]
c..................................................................

      nrfstep2=50

c..................................................................
c     scaleurf="enabled" rescale contribution to B so that a particular
c     ray does not "overdamp" on a given flux surface volume.
c..................................................................

      scaleurf="enabled"

c..................................................................
c     iurfcoll (iurfl) indicate use of collisional (additional linear)
c     absorption coeff. Passed in ray data files.
c..................................................................

      do i=1,nmodsa
         iurfcoll(i)="disabled"
         iurfl(i)="disabled"
      enddo

c..................................................................
c     lh,ech and fw determine which wave modes are utilized
c     Alternatively(060314), this data can be input through rftype()
c..................................................................

      lh="disabled"
      ech="disabled"
      fw="disabled"
      do i=1,nmodsa
         rftype(i)="notset"
         rffile(i)="notset"
         nrfspecies(i)=1
      enddo

c..................................................................
c     Setting nmods=nmodsa, for the time being. Generalize later.
c..................................................................

      nmods=nmodsa ! YuP-101220: should be mrfn, but not known yet

c..................................................................
c     Variables determine which, if any,  ray tracing code is called.
c..................................................................

      call_lh="disabled"
      call_ech="disabled"
      call_fw="disabled"

      rfread="text"

c..................................................................
c     urfdmp="secondd" means utilize the "second derivative" damping
c     diagnostic in computing the damping due to each ray as
c     it moves through the plasma. If urfdmp .ne. "secondd" the
c     diagnostic uses an integration by parts technique to compute
c     the damping. We highly recommend "secondd" because convergence
c     is the agreement between the sum of the damping of all rays
c     and the absorbed power as computed from dF/dt. This latter
c     diagnostic utilizes the "second derivative" approach so
c     consistency demands "secondd" for the rays.
c..................................................................

      urfdmp="secondd"
      return
      end
