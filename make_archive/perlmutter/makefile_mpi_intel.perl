# makefile producing MPI version of cql3d.

# May/06:  makefile_mpi.xxx  is a system setup by Nikolai Ershov
#          for creation of mpi version of cql3d from the serial
#          version.   Therefore, only one version of cql3d needs
#          be maintained.
#          Comments (beginning with CMPI) have been inserted in the source.
#          These are processed using python code from the ./mpi subdirectory
#          to produce an MPI cql3d version.
#          The procedure is described in CompX report 
#          CQL3D_Parallelization_CompX-2006-1.pdf, and has been evolved
#          by Yuri Petrov.
#          Please follow this procedure in future cql3d modifications.

#Make sure have intel environment (Intel is default on Cori):
#module list
#Else switch environment:
#For example: module swap PrgEnv-xxx PrgEnv-intel

#make sure have netcdf libraries and includes:
#module unload darshan; module load cray-netcdf; module load cray-hdf5

#YuP[April 2017] It is strongly advised to add into your .bashrc.ext script:
#   export FORT_BUFFERED=1
# or into your .cshrc.ext script:
#   setenv FORT_BUFFERED 1
# It was the default option for NERSC in 2016,
# but it was disabled by NERSC personnel in 2017
# because of problems with some newer compilers.
# As a result of this change, the MPI runs became slower by ~3x
# when there is I/O of data into one file from all cores.
 
#Works on Perlmutter YuP[2023], mpilib.f is not needed anymore

# Per 2020-01-16 - Johannes Blaschke (NERSC):  -----------------------------------
# 1. Add `-lX11` to the LIBS variable (Note that the LIBRARIES variable is no longer
# used. Instead the make file uses LIBS). So LIBS should read:
# ```
# LIBS = $(LOCATION) -lpgplot -lX11
# ```
# (note the `-lnetcdf` `-lnetcdff` flags are
# also no longer needed, cf. point 2 below)
# 2. You should rely on the compiler wrappers to find all loaded module libraries
# (such as NetCDF). So you don't need to manually include `$(NETCDF_DIR)` and
# `-lnetcdf`, etc. The compiler wrapper already does that if the `cray-netcdf` module
# has been loaded. You can find out what the compiler wrapper already contains by
# running `cc -v` or `ftn -v`.
# 3. The location of your `pgplot` library now needs to be included in `LOCATION`.
# ```
# LOCATION= -L/global/cfs/cdirs/m77/pgplot.perl.intel
# (or whatever the path is)
# ```
# Also add into your cori_batchscript_mpi  file, just before srun line:
#  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/cfs/cdirs/m77/pgplot.perl.intel
# --------------------------------------------------------
# YuP/perNERSCconsulters: Need to use -L/usr/lib64 just before -lX11
# Also add into your cori_batchscript_mpi  file, just before srun line:
#  export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
#  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/cfs/cdirs/m77/pgplot.perl.intel 
#----------------  On this subject (from NERSC consulters; 2020-02-05):
# ... the "-fast" flag is the cause for the final link stage to need -L/usr/lib64 before
#  -lX11.
# And without -fast, the -L/usr/lib64 is not needed. 
# And this "-fast" is also related to  ipo warning messages. 
# "-fast" includes an option to do ipo (interprocedural optimization), and it fails
# to find some X11 libraries in /usr/lib64, which by default the compiler should know
# about this path. I think it is kind of an Intel compiler bug with "-fast". 
# Recommended: use "-O3" instead of "-fast" to avoid the "-ipo" included in the
#  "-fast" option.
#-----------------------------------------------------------------------------------

SHELL     = /bin/sh
NAME      = xcql3d_mpi_intel.perl
COMPILER=   ftn
BUILDER=	$(COMPILER)
INCLUDES  = advnce.h comm.h frcomm.h frname.h frname_decl.h name.h name_decl.h \
	param.h trans.h wpadvnc.h \
         mpilib.h
SOURCES  = a_cqlp.f  abchief.f  achief1.f  achiefn.f  aclear.f  ainalloc.f  \
	aindflt.f  aindflt1.f aindfpa.f  aingeom.f  ainitial.f   \
	ainpla.f  ainplt.f  ainpltpa.f   \
	ainsetpa.f  ainsetva.f  ainspec.f  ainvnorm.f  \
	aminmx.f ampfar.f bavdens.f  bavgmax.f  baviorbt.f   \
	bcast.f  bsl.f  bsu.f  cfpcoefc.f  cfpcoefn.f  \
	cfpcoefr.f  cfpgamma.f  cfpleg.f     \
	cfpmodbe.f  cfpsymt.f  coefefad.f  coefefld.f  coefegad.f  \
	coeffpad.f  coefload.f  coefmidt.f  coefmidv.f   \
	coefrfad.f  coefstup.f  coefsyad.f  coefwti.f  \
	coefwtj.f  diag.f  diagcfac.f   \
	diagdens.f  diagdenz.f  diagentr.f  diagescl.f  diaggnde.f  \
	diaggnde2.f  diagimpd.f  diagscal.f  diagwrng.f  diagxswt.f   \
	diagxswx.f  dsk_gr.f dskout.f  efield.f  eflditer.f eqalloc.f   \
	eqcoord.f  eqelpse.f  eqflxavg.f  eqfn.f  \
	eqfndpsi.f  eqfninv.f  eqfpsi.f   \
	eqindflt.f  eqinitl.f  eqjac.f  eqonovrp.f  \
	eqorbit.f  eqrhopsi.f  eqrhs.f   \
	eqtopeol.f  equilib.f  eqvolpsi.f  eqwrng.f  \
	esefld.f  exlin.f  exsweep.f  exsweept.f   \
	exsweepx.f  finit.f  firstdrv.f  fle.f flxfn.f  \
	freya.f  freyasou.f  frhexdrv.f   \
	frinitl.f  frinitz.f  frnbdep2.f  frnfreya.f  \
	frplteq.f  frset.f frsmooth.f  frsplft.f   \
	frstup.f  frsubs.f  frsuppor.f  frwrong.f  \
	ilut.f impavnc0.f  impchk.f  impnorm.f  it3dalloc.f  \
	lookup.f  losscone.f  lossegy.f  lossorbm.f  \
	losstor.f  micfrplt.f  micgetr.f   \
	micgmbnd.f  micgnbnd.f  micxinil.f  micxinim.f  micxinit.f  \
	micxiniz.f  netcdfrf.f  netcdfrw2.f ntdstore.f   \
	ntloop.f  pack21.f  pltcont.f  pltcycl.f  pltdf.f  \
	pltdnz.f  pltelec.f  pltendn.f   \
	pltends.f  pltfluxs.f pltfofvv.f pltfvsv.f  pltinit.f  pltlosc.f  \
	pltmain.f  pltpower.f  pltprppr.f   \
	pltrstv.f  pltrun.f pltstrml.f  pltvec.f  pltvectr.f  \
	pltvflux.f  profaxis.f  profiles.f prppr.f   \
	prpprctr.f  psif.f  psifp.f  psifppy.f  psifpy.f  psiinv.f  \
	r8lsode.f  r8subs.f  rdc_multi.f rdc_bplt.f restcon.f resthks.f  \
	restvty.f  rf.f  sigalloc.f  siggy.f  sigmax.f  sigsetup.f  \
	sigv5d.f sigfn.f      sigie.f  sigmaxwl.f   sigv.f  \
	soucrit.f sounorm.f  soup.f  soup0.f  souplt.f   \
	sourc0.f  sourcee.f  sourcef.f  sourceko.f sourcpwr.f   \
	synchrad.f  tdbootst.f tdboothi.f  tdchief.f  tddiag.f  \
	tddsig.f  tdeqdsk.f  tdfinterp.f  tdinitl.f  tdinlegw.f  \
	tdinterp.f  tdnflxs.f  tdnpa.f  tdnpadiag.f  tdnpa0.f  \
	tdnpacxcs.f  tdnpalam.f  tdnpabscs.f  tdoutput.f   \
	tdplteq.f  tdpltjop.f  tdpltmne.f  tdpro.f  \
	tdreadf.f  tdrmshst.f  tdsetnpa.f  tdsetsxr.f   \
	tdstin.f  tdsxr.f  tdsxr0.f  tdsxray.f  tdsxrplt.f  \
	tdtloop.f  tdtoarad.f   \
	tdtoaray.f  tdtraloc.f  tdtransp.f  tdtranspn.f  tdtravct.f  \
	tdtrchk.f  tdtrchkd.f  tdtrcon.f   \
	tdtrdfus.f  tdtrfcop.f  tdtrflg.f  tdtrflx.f  \
	tdtrmuy.f  tdtrrsou.f  tdtrrtov.f   \
	tdtrrtov2.f  tdtrsavf.f  tdtrsym.f  tdtrvint.f  \
	tdtrvsou.f  tdtrvtor.f  tdtrvtor2.f   \
	tdtrvtor3.f  tdtrwtl.f  tdtry.f  tdtscinp.f  tdtscout.f  \
	tdwrng.f  tdwritef.f  tdxin13d.f   \
	tdxin23d.f  tdxin33d.f  tdxinitl.f  urfalloc.f  \
	urfavg.f  urfb0.f  urfbes.f   \
	urfbplt.f  urfchief.f  urfdamp0.f  urfdamp1.f  \
	urfdamp2.f  urfdampa.f urfdout.f  urfedge.f   \
	urffflx.f  urfindfl.f  urfinitl.f  urfmidv.f  \
	urfpack.f  urfpackm.f  urfrays.f  urfread.f   \
	urfread_.f  urfsetup.f  urfwrite.f  urfwrite_.f  \
	urfwrong.f  urfwr0.f  urfwr0c.f \
	vlf.f vlfalloc.f vlfbplt.f vlfsetup.f vlh.f  vlhbplt.f vlhd.f  \
	wpalloc.f  wparsou.f  wpavg.f  wpbdry.f  \
	wpcheck.f  wpchgdy.f wpcthta.f  wpelecf.f   \
	wpinitl.f  wploweq.f  wpsavf.f  wptrafx.f  wptramu.f  \
	wptrmuy.f  wpvptb.f  wpwrng.f  wpmshchk.f \
	zblock.f  zcunix.f  zfreya.f    znonsym.f  \
         adcaut.f adcbrm.f adcdo.f adcdrc.f adce.f adcei.f adcerc.f \
         adcexc.f adcfnm.f adcinit.f adcout.f adcrrc.f adcset.f adcstd.f adctip.f adcxxs.f


OBJECTS   = $(SOURCES:.f=.o)

#LOCATION= -L/global/cfs/cdirs/m77/CompX/pgplot.intel 
LOCATION= -L/global/cfs/cdirs/m77/pgplot.perl.intel


LIBS = $(LOCATION) -lpgplot -L/usr/lib64 -lX11

#Location of netcdf.inc:
# INCLUDE= $(NETCDF_DIR)/include  # No longer needed [2020]
# -g is full debug;   -gopt allows optimization
DEBUG     = #-g
OPTIMIZE  = -O3 #-fast
LISTING   = -Mlist/etc/shells
CSPECIAL  = #-O1 #-check bounds -check pointers
#YuP: Flag (-real-size 64) may ruin some pages made by PGPLOT, at Intel/PC and Intel/NERSC 
#YuP: Flags (-check bounds -check pointers ) make runs slower by 2x.
SPECIAL   = -byteswapio -Mbackslash -fallow-argument-mismatch -std=legacy
#LDSPECIAL = -Wl,-noinhibit-exec 
LDSPECIAL = 
#COMPILE   = $(COMPILER) -c $(CSPECIAL) $(INCLUDE) $(DEBUG) # or use $(OPTIMIZE)
COMPILE   =  $(COMPILER) -c $(CSPECIAL) $(INCLUDE) $(OPTIMIZE) # or use $(DEBUG)
BUILD      = $(BUILDER) -o $(NAME) $(LDSPECIAL) # $(OPTIMIZE) # $(DEBUG)
PROTECT   = chmod 755
DELETE    = rm -f

# The following gives suffixes to be used in checking for suffix rules.
# Written without dependencies, it may be useful to turn of such checking?
.SUFFIXES:

$(NAME): $(OBJECTS)
	$(BUILD) $(OBJECTS) $(LIBS)
	$(PROTECT) $(NAME)

# Following use of pattern matching works; 
# it is based on statements in Sect. 10.7 of gmake Manual
# by Stallman and McGrath.
# However, the two rules following this seem clearer to me (BobH).
#%.o:               %.f        $(INCLUDES)
#	$(COMPILE) $< -o $@

#include deps   #Introduced by John Wright (JCW)
$(SOURCES):        $(INCLUDES)

# see Sect. 4.10.1, Static Pattern Rules.
#$(OBJECTS): %.o: %.f    $(INCLUDES)
%.o: %.f  #    $(INCLUDES)
#Changes introduced by JCW so obtain saved copy of mpi modified sources
#	mpi/doparallel.py $< mpitmp.f mpi/mpins_par.f
#	$(COMPILE) mpitmp.f -o $@
	mpi/doparallel.py $< $*_mpitmp.f mpi/mpins_par.f
	$(COMPILE) $*_mpitmp.f -o $@

rebuild:
	$(COMPILE) $(SOURCES)
	$(BUILD) $(OBJECTS) $(LIBS)

clean:
	$(DELETE)  $(OBJECTS) *.lst *_mpitmp.f

