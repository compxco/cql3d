c***********************************************************************
      program a_cql3d
c***********************************************************************


c***********************************************************************
c
c   Copyright R.W. Harvey and Yu. V Petrov
c   CompX company, Del Mar, California, USA
c   1995-2011
c   Below, "this program" refers to the CQL3D, alternatively designated
c   as CQLP when used in cqlpmod='enabled' mode (and alternative 
c   capitalizations), and associated source files and manuals.
c
c   The primary reference for the code is:
c   ``The CQL3D Fokker-Planck Code'',  R.W. Harvey and M.G. McCoy, 
c   Proc. of IAEA Technical Committee Meeting on Advances in Simulation 
c   and Modeling of Thermonuclear Plasmas, Montreal, 1992, p. 489-526, 
c   IAEA, Vienna (1993), available through NTIS/DOC (National Technical 
c   Information Service, U.S. Dept. of Commerce), Order No. DE93002962.
c   See also, http://www.compxco.com/cql3d.html, CQL3D Manual.
c
c
c                GNU Public License Distribution Only
c   This program is free software; you can redistribute it and/or modify
c   it under the terms of the GNU General Public License as published by
c   the Free Software Foundation; either version 3 of the License, or
c   any later version.
c
c   This program is distributed in the hope that it will be useful,
c   but WITHOUT ANY WARRANTY; without even the implied warranty of
c   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
c   GNU General Public License for more details.
c
c   You should have received a copy of the GNU General Public License
c   along with this program; if not, see <http://www.gnu.org/licenses/>.
c
c         --R.W. Harvey, CompX, Del Mar, CA, USA
c
c   E-mail:  rwharvey@compxco.com
c   Address: CompX company
c            P.O. Box 2672
c            Del Mar, CA 92014-5672
c
c   It will be appreciated by CompX if useful additions to CQL3D can be
c   transmitted to the authors, for inclusion in the source distribution.
c
c***********************************************************************
c
c
c..................................................................
c
c     Highlights of code history (see also a_change.h):
c
c     02/01/2011
c     MPI version of code (YuP)
c
c     01/01/2011
c     Completed major dynamic dimensioning of cql3d variables.
c     Adjustment of parameters settings in param.h to fit different
c     cql3d type problems will generally no longer be required (YuP)
c
c     07/20/2010
c     Rewrote NPA diagnostic, obtaining good absolute value agreement
c     with Aaron Bader MIT C-Mod diagnostic (bh)
c
c     06/08/2010
c     Major modification: combines several separate branches of cql3d,
c     and includes fully-implicit 3d radial transport (soln_method=it3drv), 
c     URF routines applied to multiple general species(multiURF),
c     full non-updown symmetric equilibria (eqsym=non),
c     NPA diagnostics, deltar first-order finite-orbit-width
c     shift (and output to .nc file)   (YuP, bh)
c
c     10/16/2009
c     Debugged and rewrote fully relativistic collision operator
c     and compared with mildly relativistic  (YuP)
c
c     05/14/2009
c     Fully f90 pointered version rather than cray pointers
c
c     04/24/2007
c     First results from new soln_method="it3drv" option using
c     sparskit2 to iteratively solve full 3d (2V-1R) implicit  cql3d
c     equations (bh)
c
c     06/22/2006
c     Add NPA synthetic diagnostic (bh and Vincent Tang)
c
c     11/11/2004
c     rdcmod modification to read in externally computed RF
c     diffusion coefficients, such as from AORSA or DC codes (bh)
c
c     04/26/2001
c     General radial and vel-space profiles for D_rr(rho,u,theta).
c     Newton iteration to maintain target density profile (bh)
c
c     03/19/01
c     Switch to CVS maintenance at bobh.compxco.com
c
c     01/01/00
c     Compiling cql with PGF77 Portland Group Compiler (bh).
c
c     07/31/99
c     Numerical calculation of boostrap current added to
c     current version of the code (from 1993 cqlp version) (bh).
c
c     06/01/99
c     Much output directed to a netCDF file.
c
c     05/01/99
c     CQL3D has been converted from Cray C90 environment to
c     a workstation environment (PC/Absoft, Alpha/Dec-fortran)
c     as well as the Cray J90/f90 environment.  Graphic output
c     is converted to freeware PGPLOT package.  NetCDF output
c     files implemented. See notes.h for J90    (bh).
c
c     04/02/99
c     Added vlfmod="enabled" and self-consistent ambipolar elecric
c     to cqlpmod="enabled" parallel FP code. (bh)
c
c     07/06/97
c     Knock-On operator accounting for poloidal variation of
c     electons.(bh+sc)
c
c     05/21/96
c     Knock-on operator, pre-loading, time-dependent background
c     profiles (bh).
c
c
c     01/10/95
c     Added Zeff profile, toroidal rotation (vphi) effect into
c     Freya, and fusion neutron calculation. (bh).
c     06/21/94
c     Added vlf...-routines for single flux surface calculation of
c     quasilinear diffusion effects for given frequency, strength,
c     cyclotron harmonic, npar, nperp, polarizations, and localization
c     of a flux surface. (bh).
c    
c     05/10/94
c     Final mix of BH and OS versions in B_940507, minor changes from
c     B_940425 mainly in urf modules
c
c     04/25/94
c     Mix B.Harvey and O.Sauter versions to give bh940224, reindent it
c     according to Emacs-Fortran rules => bhos_940304. Modify a few 
c     things plus the initialization and scaling of ca,cb,..,cf 
c     in cfpcoefn and cfpcoefr
c     which were wrong as cd doesn't follow cc in memory.
c     This gives new version: bhos_940425, which is renamed to 
c     CQLP_B_940425 and is the start for the B versions of 
c     CQLP/CQL3D codes.
c
c     01/28/94
c     CQLP/CQL3D  is built starting from the CQL3D E_930209 version, 
c     then several new versions followed up. The last one before 
c     O.Sauter left GA is the 930624 version. Then the version used 
c     for the 4th PET93 workshop  paper is the 930910 version save 
c     by O.Sauter on cfs in CQLP_A931103.taZ
c
c     The CQLP part of the code is a Fokker-Planck code which is 2D in
c     momentum-per-mass space and 1D along the magnetic field.   
c     The transport term, v_par*d(f)/ds for
c     distance along the magnetic field s, is incorporated.  
c     The code provides additional functionality to  the CQL3D code. 
c     It was constructed by Olivier Sauter (working with Bob Harvey)  
c     at GA during the period July 1993-July 1994. 
c     cqlpmod="enabled" accesses the new functionality.
c
c
c
c     02/17/93
c     CQL3D code with option of solving the FP equations for only
c     a few flux surfaces in a given radial array (lrzdiff="enabled")
c     (valid only for CQL: transp="disabled", and tested only with
c     electric field and RF contribution).
c
c     10/30/91
c     This is the  Bounce-Averaged Collisional Quasilinear (CQL3D)
c     code. It may be obtained by contacting the authors.
c     The authors are R.W. Harvey, M.G. McCoy, and G.D. Kerbel.
c     R.W. Harvey may be contacted at CompX, P.O. Box 2672, Del Mar, 
c     Calfornia 92014-5692, (858)509-2131, bobh@compxco.com.
c
c     CQL3D posseses a multi-flux surface driver routine, called
c     tdchief, which calls CQL in sequence as a parameter ranges over
c     the flux surfaces in consideration. There is also an option
c     to add (ad-hoc) transport (that is a "radial" derivative) which
c     is achieved via a splitting algorithm. When operating in this
c     multi-flux surface mode, the code is referred to as CQL3D.
c     CQL is the name of the single flux surface precurser of this
c     code, and is not extinct, as setting input variable lrz=1
c     will enable this operating mode.
c
c
c     NOTES ON COMPILATION AND USE OF CODE:

c     This code has been re-written from running under CRAY UNIX 
c     (UNICOS and CSOS) operating systems to a 32-bit UNIX workstation
c     environment (RWH, May, 1999). Makefiles are available for
c     execution with (1) the Portland Group PGF77 compiler under
c     Linux, (2) the Absoft F77/F90 compiler under Linux, (3) the
c     Dec Alpha compiler under OSF.
c
c     Instructions are given in a_notes for adjusting the code
c     to run in the 64-bit Cray environment.
c
c     The freely available and widerly ported netCDF and PGPLOT 
c     fortran libraries are required for data output and plotting.
c     The free LAPACK/BLAS libraries are used for linear algebra.
c
c     For certain operating scenarios, this code requires input
c     disk files, such as an "eqdsk" file if non-circular flux
c     surfaces relevant to some experiment are required. In this
c     case, an equilibrium code must generate the "eqdsk" file.
c     Similarly, CQL3D operates in tandem with the  ray
c     tracing data from several different ray tracing codes
c     (GENRAY, TORAY,...) for EC, EBW, LH and fast wave scenarios.
c     Documentation on these options limited, and to run this code
c     successfully, interaction with one of the authors is advisable.
c
c     INPUT variables are described in the input deck, 'cqlinput_help'.
c
c     While this code uses dynamic dimensioning for most arrays,
c     some statically allocated arrays persist in this version. Thus
c     some parameters must be set. See include file 'param.h'. A few
c     parameters could need setting in 'frcomm.h' if the neutral beam 
c     NFREYA code is being utilized.  Finally zfreya.F holds some 
c     routines used by the neutral beam deposition module 
c     (NFREYA - Oak Ridge). 
c
c
c     Significant contributions to this code have been made by Mark
c     Franz (U.S.A.F.), who wrote the relativistic corrections to the
c     collision operator, and was involved in all aspects of the
c     relativistic work. Steve White (IBM) was responsible for 
c     numerous improvements to speed up the calculation. In particular,
c     he wrote the CRAY-2 optimized Gaussian elimination routines
c     that make this code inexpensive for high-resolution and multi-flux
c     surface simulations. (Since superceded by LAPACK routines.)
c
c     Executables are created with makefiles, and the assortment of
c     them in this distribution indicates the range of machines and
c     which the code has been compiled on.
c
c..................................................................

      real*4 tarray(2)

CMPIINSERT_INCLUDE

      mpirank=0 ! When MPI is used, mpirank is set in init_mpi below

c     Initialize MPI:
CMPIINSERT_START
c     It will insert:
c     ! Initialize MPI and set mpitime0=MPI_WTIME

CMPIINSERT_BARRIER

      call cpu_time(tarray(1))    !This is an f95 intrinsic subroutine
      !------------!
      call abchief !-> calls tdchief (only)
      !------------!
      call cpu_time(tarray(2))

CMPIINSERT_BARRIER

CMPIINSERT_IF_RANK_EQ_0
      WRITE(*,'(a,i5,f10.3)') 
     +' a_cqlp: rank, Exec.time tarray(2)-tarray(1)',
     +      mpirank,            tarray(2)-tarray(1)
c      WRITE(*,'(a)') ' a_cqlp: END of CQL3D, just before MPI_FINISH'
CMPIINSERT_ENDIF_RANK

      call it3ddalloc ! Deallocate it3d related storage
      !YuP[2021] giving problems: call de_alloc   ! Deallocate other arrays
CMPIINSERT_BARRIER
    
c     close MPI (print 'MPI Full time =',MPI_WTIME()-mpitime0
c                then - MPI_FINALIZE )
CMPIINSERT_FINISH


      call exit(0)
      stop 
      end






c***********************************************************************
c***********************************************************************

c                       GNU GENERAL PUBLIC LICENSE
c                          Version 3, 29 June 2007
c   
c    Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
c    Everyone is permitted to copy and distribute verbatim copies
c    of this license document, but changing it is not allowed.
c   
c                               Preamble
c   
c     The GNU General Public License is a free, copyleft license for
c   software and other kinds of works.
c   
c     The licenses for most software and other practical works are designed
c   to take away your freedom to share and change the works.  By contrast,
c   the GNU General Public License is intended to guarantee your freedom to
c   share and change all versions of a program--to make sure it remains free
c   software for all its users.  We, the Free Software Foundation, use the
c   GNU General Public License for most of our software; it applies also to
c   any other work released this way by its authors.  You can apply it to
c   your programs, too.
c   
c     When we speak of free software, we are referring to freedom, not
c   price.  Our General Public Licenses are designed to make sure that you
c   have the freedom to distribute copies of free software (and charge for
c   them if you wish), that you receive source code or can get it if you
c   want it, that you can change the software or use pieces of it in new
c   free programs, and that you know you can do these things.
c   
c     To protect your rights, we need to prevent others from denying you
c   these rights or asking you to surrender the rights.  Therefore, you have
c   certain responsibilities if you distribute copies of the software, or if
c   you modify it: responsibilities to respect the freedom of others.
c   
c     For example, if you distribute copies of such a program, whether
c   gratis or for a fee, you must pass on to the recipients the same
c   freedoms that you received.  You must make sure that they, too, receive
c   or can get the source code.  And you must show them these terms so they
c   know their rights.
c   
c     Developers that use the GNU GPL protect your rights with two steps:
c   (1) assert copyright on the software, and (2) offer you this License
c   giving you legal permission to copy, distribute and/or modify it.
c   
c     For the developers' and authors' protection, the GPL clearly explains
c   that there is no warranty for this free software.  For both users' and
c   authors' sake, the GPL requires that modified versions be marked as
c   changed, so that their problems will not be attributed erroneously to
c   authors of previous versions.
c   
c     Some devices are designed to deny users access to install or run
c   modified versions of the software inside them, although the manufacturer
c   can do so.  This is fundamentally incompatible with the aim of
c   protecting users' freedom to change the software.  The systematic
c   pattern of such abuse occurs in the area of products for individuals to
c   use, which is precisely where it is most unacceptable.  Therefore, we
c   have designed this version of the GPL to prohibit the practice for those
c   products.  If such problems arise substantially in other domains, we
c   stand ready to extend this provision to those domains in future versions
c   of the GPL, as needed to protect the freedom of users.
c   
c     Finally, every program is threatened constantly by software patents.
c   States should not allow patents to restrict development and use of
c   software on general-purpose computers, but in those that do, we wish to
c   avoid the special danger that patents applied to a free program could
c   make it effectively proprietary.  To prevent this, the GPL assures that
c   patents cannot be used to render the program non-free.
c   
c     The precise terms and conditions for copying, distribution and
c   modification follow.
c   
c                          TERMS AND CONDITIONS
c   
c     0. Definitions.
c   
c     "This License" refers to version 3 of the GNU General Public License.
c   
c     "Copyright" also means copyright-like laws that apply to other kinds of
c   works, such as semiconductor masks.
c   
c     "The Program" refers to any copyrightable work licensed under this
c   License.  Each licensee is addressed as "you".  "Licensees" and
c   "recipients" may be individuals or organizations.
c   
c     To "modify" a work means to copy from or adapt all or part of the work
c   in a fashion requiring copyright permission, other than the making of an
c   exact copy.  The resulting work is called a "modified version" of the
c   earlier work or a work "based on" the earlier work.
c   
c     A "covered work" means either the unmodified Program or a work based
c   on the Program.
c   
c     To "propagate" a work means to do anything with it that, without
c   permission, would make you directly or secondarily liable for
c   infringement under applicable copyright law, except executing it on a
c   computer or modifying a private copy.  Propagation includes copying,
c   distribution (with or without modification), making available to the
c   public, and in some countries other activities as well.
c   
c     To "convey" a work means any kind of propagation that enables other
c   parties to make or receive copies.  Mere interaction with a user through
c   a computer network, with no transfer of a copy, is not conveying.
c   
c     An interactive user interface displays "Appropriate Legal Notices"
c   to the extent that it includes a convenient and prominently visible
c   feature that (1) displays an appropriate copyright notice, and (2)
c   tells the user that there is no warranty for the work (except to the
c   extent that warranties are provided), that licensees may convey the
c   work under this License, and how to view a copy of this License.  If
c   the interface presents a list of user commands or options, such as a
c   menu, a prominent item in the list meets this criterion.
c   
c     1. Source Code.
c   
c     The "source code" for a work means the preferred form of the work
c   for making modifications to it.  "Object code" means any non-source
c   form of a work.
c   
c     A "Standard Interface" means an interface that either is an official
c   standard defined by a recognized standards body, or, in the case of
c   interfaces specified for a particular programming language, one that
c   is widely used among developers working in that language.
c   
c     The "System Libraries" of an executable work include anything, other
c   than the work as a whole, that (a) is included in the normal form of
c   packaging a Major Component, but which is not part of that Major
c   Component, and (b) serves only to enable use of the work with that
c   Major Component, or to implement a Standard Interface for which an
c   implementation is available to the public in source code form.  A
c   "Major Component", in this context, means a major essential component
c   (kernel, window system, and so on) of the specific operating system
c   (if any) on which the executable work runs, or a compiler used to
c   produce the work, or an object code interpreter used to run it.
c   
c     The "Corresponding Source" for a work in object code form means all
c   the source code needed to generate, install, and (for an executable
c   work) run the object code and to modify the work, including scripts to
c   control those activities.  However, it does not include the work's
c   System Libraries, or general-purpose tools or generally available free
c   programs which are used unmodified in performing those activities but
c   which are not part of the work.  For example, Corresponding Source
c   includes interface definition files associated with source files for
c   the work, and the source code for shared libraries and dynamically
c   linked subprograms that the work is specifically designed to require,
c   such as by intimate data communication or control flow between those
c   subprograms and other parts of the work.
c   
c     The Corresponding Source need not include anything that users
c   can regenerate automatically from other parts of the Corresponding
c   Source.
c   
c     The Corresponding Source for a work in source code form is that
c   same work.
c   
c     2. Basic Permissions.
c   
c     All rights granted under this License are granted for the term of
c   copyright on the Program, and are irrevocable provided the stated
c   conditions are met.  This License explicitly affirms your unlimited
c   permission to run the unmodified Program.  The output from running a
c   covered work is covered by this License only if the output, given its
c   content, constitutes a covered work.  This License acknowledges your
c   rights of fair use or other equivalent, as provided by copyright law.
c   
c     You may make, run and propagate covered works that you do not
c   convey, without conditions so long as your license otherwise remains
c   in force.  You may convey covered works to others for the sole purpose
c   of having them make modifications exclusively for you, or provide you
c   with facilities for running those works, provided that you comply with
c   the terms of this License in conveying all material for which you do
c   not control copyright.  Those thus making or running the covered works
c   for you must do so exclusively on your behalf, under your direction
c   and control, on terms that prohibit them from making any copies of
c   your copyrighted material outside their relationship with you.
c   
c     Conveying under any other circumstances is permitted solely under
c   the conditions stated below.  Sublicensing is not allowed; section 10
c   makes it unnecessary.
c   
c     3. Protecting Users' Legal Rights From Anti-Circumvention Law.
c   
c     No covered work shall be deemed part of an effective technological
c   measure under any applicable law fulfilling obligations under article
c   11 of the WIPO copyright treaty adopted on 20 December 1996, or
c   similar laws prohibiting or restricting circumvention of such
c   measures.
c   
c     When you convey a covered work, you waive any legal power to forbid
c   circumvention of technological measures to the extent such circumvention
c   is effected by exercising rights under this License with respect to
c   the covered work, and you disclaim any intention to limit operation or
c   modification of the work as a means of enforcing, against the work's
c   users, your or third parties' legal rights to forbid circumvention of
c   technological measures.
c   
c     4. Conveying Verbatim Copies.
c   
c     You may convey verbatim copies of the Program's source code as you
c   receive it, in any medium, provided that you conspicuously and
c   appropriately publish on each copy an appropriate copyright notice;
c   keep intact all notices stating that this License and any
c   non-permissive terms added in accord with section 7 apply to the code;
c   keep intact all notices of the absence of any warranty; and give all
c   recipients a copy of this License along with the Program.
c   
c     You may charge any price or no price for each copy that you convey,
c   and you may offer support or warranty protection for a fee.
c   
c     5. Conveying Modified Source Versions.
c   
c     You may convey a work based on the Program, or the modifications to
c   produce it from the Program, in the form of source code under the
c   terms of section 4, provided that you also meet all of these conditions:
c   
c       a) The work must carry prominent notices stating that you modified
c       it, and giving a relevant date.
c   
c       b) The work must carry prominent notices stating that it is
c       released under this License and any conditions added under section
c       7.  This requirement modifies the requirement in section 4 to
c       "keep intact all notices".
c   
c       c) You must license the entire work, as a whole, under this
c       License to anyone who comes into possession of a copy.  This
c       License will therefore apply, along with any applicable section 7
c       additional terms, to the whole of the work, and all its parts,
c       regardless of how they are packaged.  This License gives no
c       permission to license the work in any other way, but it does not
c       invalidate such permission if you have separately received it.
c   
c       d) If the work has interactive user interfaces, each must display
c       Appropriate Legal Notices; however, if the Program has interactive
c       interfaces that do not display Appropriate Legal Notices, your
c       work need not make them do so.
c   
c     A compilation of a covered work with other separate and independent
c   works, which are not by their nature extensions of the covered work,
c   and which are not combined with it such as to form a larger program,
c   in or on a volume of a storage or distribution medium, is called an
c   "aggregate" if the compilation and its resulting copyright are not
c   used to limit the access or legal rights of the compilation's users
c   beyond what the individual works permit.  Inclusion of a covered work
c   in an aggregate does not cause this License to apply to the other
c   parts of the aggregate.
c   
c     6. Conveying Non-Source Forms.
c   
c     You may convey a covered work in object code form under the terms
c   of sections 4 and 5, provided that you also convey the
c   machine-readable Corresponding Source under the terms of this License,
c   in one of these ways:
c   
c       a) Convey the object code in, or embodied in, a physical product
c       (including a physical distribution medium), accompanied by the
c       Corresponding Source fixed on a durable physical medium
c       customarily used for software interchange.
c   
c       b) Convey the object code in, or embodied in, a physical product
c       (including a physical distribution medium), accompanied by a
c       written offer, valid for at least three years and valid for as
c       long as you offer spare parts or customer support for that product
c       model, to give anyone who possesses the object code either (1) a
c       copy of the Corresponding Source for all the software in the
c       product that is covered by this License, on a durable physical
c       medium customarily used for software interchange, for a price no
c       more than your reasonable cost of physically performing this
c       conveying of source, or (2) access to copy the
c       Corresponding Source from a network server at no charge.
c   
c       c) Convey individual copies of the object code with a copy of the
c       written offer to provide the Corresponding Source.  This
c       alternative is allowed only occasionally and noncommercially, and
c       only if you received the object code with such an offer, in accord
c       with subsection 6b.
c   
c       d) Convey the object code by offering access from a designated
c       place (gratis or for a charge), and offer equivalent access to the
c       Corresponding Source in the same way through the same place at no
c       further charge.  You need not require recipients to copy the
c       Corresponding Source along with the object code.  If the place to
c       copy the object code is a network server, the Corresponding Source
c       may be on a different server (operated by you or a third party)
c       that supports equivalent copying facilities, provided you maintain
c       clear directions next to the object code saying where to find the
c       Corresponding Source.  Regardless of what server hosts the
c       Corresponding Source, you remain obligated to ensure that it is
c       available for as long as needed to satisfy these requirements.
c   
c       e) Convey the object code using peer-to-peer transmission, provided
c       you inform other peers where the object code and Corresponding
c       Source of the work are being offered to the general public at no
c       charge under subsection 6d.
c   
c     A separable portion of the object code, whose source code is excluded
c   from the Corresponding Source as a System Library, need not be
c   included in conveying the object code work.
c   
c     A "User Product" is either (1) a "consumer product", which means any
c   tangible personal property which is normally used for personal, family,
c   or household purposes, or (2) anything designed or sold for incorporation
c   into a dwelling.  In determining whether a product is a consumer product,
c   doubtful cases shall be resolved in favor of coverage.  For a particular
c   product received by a particular user, "normally used" refers to a
c   typical or common use of that class of product, regardless of the status
c   of the particular user or of the way in which the particular user
c   actually uses, or expects or is expected to use, the product.  A product
c   is a consumer product regardless of whether the product has substantial
c   commercial, industrial or non-consumer uses, unless such uses represent
c   the only significant mode of use of the product.
c   
c     "Installation Information" for a User Product means any methods,
c   procedures, authorization keys, or other information required to install
c   and execute modified versions of a covered work in that User Product from
c   a modified version of its Corresponding Source.  The information must
c   suffice to ensure that the continued functioning of the modified object
c   code is in no case prevented or interfered with solely because
c   modification has been made.
c   
c     If you convey an object code work under this section in, or with, or
c   specifically for use in, a User Product, and the conveying occurs as
c   part of a transaction in which the right of possession and use of the
c   User Product is transferred to the recipient in perpetuity or for a
c   fixed term (regardless of how the transaction is characterized), the
c   Corresponding Source conveyed under this section must be accompanied
c   by the Installation Information.  But this requirement does not apply
c   if neither you nor any third party retains the ability to install
c   modified object code on the User Product (for example, the work has
c   been installed in ROM).
c   
c     The requirement to provide Installation Information does not include a
c   requirement to continue to provide support service, warranty, or updates
c   for a work that has been modified or installed by the recipient, or for
c   the User Product in which it has been modified or installed.  Access to a
c   network may be denied when the modification itself materially and
c   adversely affects the operation of the network or violates the rules and
c   protocols for communication across the network.
c   
c     Corresponding Source conveyed, and Installation Information provided,
c   in accord with this section must be in a format that is publicly
c   documented (and with an implementation available to the public in
c   source code form), and must require no special password or key for
c   unpacking, reading or copying.
c   
c     7. Additional Terms.
c   
c     "Additional permissions" are terms that supplement the terms of this
c   License by making exceptions from one or more of its conditions.
c   Additional permissions that are applicable to the entire Program shall
c   be treated as though they were included in this License, to the extent
c   that they are valid under applicable law.  If additional permissions
c   apply only to part of the Program, that part may be used separately
c   under those permissions, but the entire Program remains governed by
c   this License without regard to the additional permissions.
c   
c     When you convey a copy of a covered work, you may at your option
c   remove any additional permissions from that copy, or from any part of
c   it.  (Additional permissions may be written to require their own
c   removal in certain cases when you modify the work.)  You may place
c   additional permissions on material, added by you to a covered work,
c   for which you have or can give appropriate copyright permission.
c   
c     Notwithstanding any other provision of this License, for material you
c   add to a covered work, you may (if authorized by the copyright holders of
c   that material) supplement the terms of this License with terms:
c   
c       a) Disclaiming warranty or limiting liability differently from the
c       terms of sections 15 and 16 of this License; or
c   
c       b) Requiring preservation of specified reasonable legal notices or
c       author attributions in that material or in the Appropriate Legal
c       Notices displayed by works containing it; or
c   
c       c) Prohibiting misrepresentation of the origin of that material, or
c       requiring that modified versions of such material be marked in
c       reasonable ways as different from the original version; or
c   
c       d) Limiting the use for publicity purposes of names of licensors or
c       authors of the material; or
c   
c       e) Declining to grant rights under trademark law for use of some
c       trade names, trademarks, or service marks; or
c   
c       f) Requiring indemnification of licensors and authors of that
c       material by anyone who conveys the material (or modified versions of
c       it) with contractual assumptions of liability to the recipient, for
c       any liability that these contractual assumptions directly impose on
c       those licensors and authors.
c   
c     All other non-permissive additional terms are considered "further
c   restrictions" within the meaning of section 10.  If the Program as you
c   received it, or any part of it, contains a notice stating that it is
c   governed by this License along with a term that is a further
c   restriction, you may remove that term.  If a license document contains
c   a further restriction but permits relicensing or conveying under this
c   License, you may add to a covered work material governed by the terms
c   of that license document, provided that the further restriction does
c   not survive such relicensing or conveying.
c   
c     If you add terms to a covered work in accord with this section, you
c   must place, in the relevant source files, a statement of the
c   additional terms that apply to those files, or a notice indicating
c   where to find the applicable terms.
c   
c     Additional terms, permissive or non-permissive, may be stated in the
c   form of a separately written license, or stated as exceptions;
c   the above requirements apply either way.
c   
c     8. Termination.
c   
c     You may not propagate or modify a covered work except as expressly
c   provided under this License.  Any attempt otherwise to propagate or
c   modify it is void, and will automatically terminate your rights under
c   this License (including any patent licenses granted under the third
c   paragraph of section 11).
c   
c     However, if you cease all violation of this License, then your
c   license from a particular copyright holder is reinstated (a)
c   provisionally, unless and until the copyright holder explicitly and
c   finally terminates your license, and (b) permanently, if the copyright
c   holder fails to notify you of the violation by some reasonable means
c   prior to 60 days after the cessation.
c   
c     Moreover, your license from a particular copyright holder is
c   reinstated permanently if the copyright holder notifies you of the
c   violation by some reasonable means, this is the first time you have
c   received notice of violation of this License (for any work) from that
c   copyright holder, and you cure the violation prior to 30 days after
c   your receipt of the notice.
c   
c     Termination of your rights under this section does not terminate the
c   licenses of parties who have received copies or rights from you under
c   this License.  If your rights have been terminated and not permanently
c   reinstated, you do not qualify to receive new licenses for the same
c   material under section 10.
c   
c     9. Acceptance Not Required for Having Copies.
c   
c     You are not required to accept this License in order to receive or
c   run a copy of the Program.  Ancillary propagation of a covered work
c   occurring solely as a consequence of using peer-to-peer transmission
c   to receive a copy likewise does not require acceptance.  However,
c   nothing other than this License grants you permission to propagate or
c   modify any covered work.  These actions infringe copyright if you do
c   not accept this License.  Therefore, by modifying or propagating a
c   covered work, you indicate your acceptance of this License to do so.
c   
c     10. Automatic Licensing of Downstream Recipients.
c   
c     Each time you convey a covered work, the recipient automatically
c   receives a license from the original licensors, to run, modify and
c   propagate that work, subject to this License.  You are not responsible
c   for enforcing compliance by third parties with this License.
c   
c     An "entity transaction" is a transaction transferring control of an
c   organization, or substantially all assets of one, or subdividing an
c   organization, or merging organizations.  If propagation of a covered
c   work results from an entity transaction, each party to that
c   transaction who receives a copy of the work also receives whatever
c   licenses to the work the party's predecessor in interest had or could
c   give under the previous paragraph, plus a right to possession of the
c   Corresponding Source of the work from the predecessor in interest, if
c   the predecessor has it or can get it with reasonable efforts.
c   
c     You may not impose any further restrictions on the exercise of the
c   rights granted or affirmed under this License.  For example, you may
c   not impose a license fee, royalty, or other charge for exercise of
c   rights granted under this License, and you may not initiate litigation
c   (including a cross-claim or counterclaim in a lawsuit) alleging that
c   any patent claim is infringed by making, using, selling, offering for
c   sale, or importing the Program or any portion of it.
c   
c     11. Patents.
c   
c     A "contributor" is a copyright holder who authorizes use under this
c   License of the Program or a work on which the Program is based.  The
c   work thus licensed is called the contributor's "contributor version".
c   
c     A contributor's "essential patent claims" are all patent claims
c   owned or controlled by the contributor, whether already acquired or
c   hereafter acquired, that would be infringed by some manner, permitted
c   by this License, of making, using, or selling its contributor version,
c   but do not include claims that would be infringed only as a
c   consequence of further modification of the contributor version.  For
c   purposes of this definition, "control" includes the right to grant
c   patent sublicenses in a manner consistent with the requirements of
c   this License.
c   
c     Each contributor grants you a non-exclusive, worldwide, royalty-free
c   patent license under the contributor's essential patent claims, to
c   make, use, sell, offer for sale, import and otherwise run, modify and
c   propagate the contents of its contributor version.
c   
c     In the following three paragraphs, a "patent license" is any express
c   agreement or commitment, however denominated, not to enforce a patent
c   (such as an express permission to practice a patent or covenant not to
c   sue for patent infringement).  To "grant" such a patent license to a
c   party means to make such an agreement or commitment not to enforce a
c   patent against the party.
c   
c     If you convey a covered work, knowingly relying on a patent license,
c   and the Corresponding Source of the work is not available for anyone
c   to copy, free of charge and under the terms of this License, through a
c   publicly available network server or other readily accessible means,
c   then you must either (1) cause the Corresponding Source to be so
c   available, or (2) arrange to deprive yourself of the benefit of the
c   patent license for this particular work, or (3) arrange, in a manner
c   consistent with the requirements of this License, to extend the patent
c   license to downstream recipients.  "Knowingly relying" means you have
c   actual knowledge that, but for the patent license, your conveying the
c   covered work in a country, or your recipient's use of the covered work
c   in a country, would infringe one or more identifiable patents in that
c   country that you have reason to believe are valid.
c   
c     If, pursuant to or in connection with a single transaction or
c   arrangement, you convey, or propagate by procuring conveyance of, a
c   covered work, and grant a patent license to some of the parties
c   receiving the covered work authorizing them to use, propagate, modify
c   or convey a specific copy of the covered work, then the patent license
c   you grant is automatically extended to all recipients of the covered
c   work and works based on it.
c   
c     A patent license is "discriminatory" if it does not include within
c   the scope of its coverage, prohibits the exercise of, or is
c   conditioned on the non-exercise of one or more of the rights that are
c   specifically granted under this License.  You may not convey a covered
c   work if you are a party to an arrangement with a third party that is
c   in the business of distributing software, under which you make payment
c   to the third party based on the extent of your activity of conveying
c   the work, and under which the third party grants, to any of the
c   parties who would receive the covered work from you, a discriminatory
c   patent license (a) in connection with copies of the covered work
c   conveyed by you (or copies made from those copies), or (b) primarily
c   for and in connection with specific products or compilations that
c   contain the covered work, unless you entered into that arrangement,
c   or that patent license was granted, prior to 28 March 2007.
c   
c     Nothing in this License shall be construed as excluding or limiting
c   any implied license or other defenses to infringement that may
c   otherwise be available to you under applicable patent law.
c   
c     12. No Surrender of Others' Freedom.
c   
c     If conditions are imposed on you (whether by court order, agreement or
c   otherwise) that contradict the conditions of this License, they do not
c   excuse you from the conditions of this License.  If you cannot convey a
c   covered work so as to satisfy simultaneously your obligations under this
c   License and any other pertinent obligations, then as a consequence you may
c   not convey it at all.  For example, if you agree to terms that obligate you
c   to collect a royalty for further conveying from those to whom you convey
c   the Program, the only way you could satisfy both those terms and this
c   License would be to refrain entirely from conveying the Program.
c   
c     13. Use with the GNU Affero General Public License.
c   
c     Notwithstanding any other provision of this License, you have
c   permission to link or combine any covered work with a work licensed
c   under version 3 of the GNU Affero General Public License into a single
c   combined work, and to convey the resulting work.  The terms of this
c   License will continue to apply to the part which is the covered work,
c   but the special requirements of the GNU Affero General Public License,
c   section 13, concerning interaction through a network will apply to the
c   combination as such.
c   
c     14. Revised Versions of this License.
c   
c     The Free Software Foundation may publish revised and/or new versions of
c   the GNU General Public License from time to time.  Such new versions will
c   be similar in spirit to the present version, but may differ in detail to
c   address new problems or concerns.
c   
c     Each version is given a distinguishing version number.  If the
c   Program specifies that a certain numbered version of the GNU General
c   Public License "or any later version" applies to it, you have the
c   option of following the terms and conditions either of that numbered
c   version or of any later version published by the Free Software
c   Foundation.  If the Program does not specify a version number of the
c   GNU General Public License, you may choose any version ever published
c   by the Free Software Foundation.
c   
c     If the Program specifies that a proxy can decide which future
c   versions of the GNU General Public License can be used, that proxy's
c   public statement of acceptance of a version permanently authorizes you
c   to choose that version for the Program.
c   
c     Later license versions may give you additional or different
c   permissions.  However, no additional obligations are imposed on any
c   author or copyright holder as a result of your choosing to follow a
c   later version.
c   
c     15. Disclaimer of Warranty.
c   
c     THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
c   APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
c   HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
c   OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
c   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
c   PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
c   IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
c   ALL NECESSARY SERVICING, REPAIR OR CORRECTION.
c   
c     16. Limitation of Liability.
c   
c     IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
c   WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
c   THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
c   GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
c   USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
c   DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
c   PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
c   EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
c   SUCH DAMAGES.
c   
c     17. Interpretation of Sections 15 and 16.
c   
c     If the disclaimer of warranty and limitation of liability provided
c   above cannot be given local legal effect according to their terms,
c   reviewing courts shall apply local law that most closely approximates
c   an absolute waiver of all civil liability in connection with the
c   Program, unless a warranty or assumption of liability accompanies a
c   copy of the Program in return for a fee.
c   
c                        END OF TERMS AND CONDITIONS
c   
c               How to Apply These Terms to Your New Programs
c   
c     If you develop a new program, and you want it to be of the greatest
c   possible use to the public, the best way to achieve this is to make it
c   free software which everyone can redistribute and change under these terms.
c   
c     To do so, attach the following notices to the program.  It is safest
c   to attach them to the start of each source file to most effectively
c   state the exclusion of warranty; and each file should have at least
c   the "copyright" line and a pointer to where the full notice is found.
c   
c       <one line to give the program's name and a brief idea of what it does.>
c       Copyright (C) <year>  <name of author>
c   
c       This program is free software: you can redistribute it and/or modify
c       it under the terms of the GNU General Public License as published by
c       the Free Software Foundation, either version 3 of the License, or
c       (at your option) any later version.
c   
c       This program is distributed in the hope that it will be useful,
c       but WITHOUT ANY WARRANTY; without even the implied warranty of
c       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
c       GNU General Public License for more details.
c   
c       You should have received a copy of the GNU General Public License
c       along with this program.  If not, see <http://www.gnu.org/licenses/>.
c   
c   Also add information on how to contact you by electronic and paper mail.
c   
c     If the program does terminal interaction, make it output a short
c   notice like this when it starts in an interactive mode:
c   
c       <program>  Copyright (C) <year>  <name of author>
c       This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
c       This is free software, and you are welcome to redistribute it
c       under certain conditions; type `show c' for details.
c   
c   The hypothetical commands `show w' and `show c' should show the appropriate
c   parts of the General Public License.  Of course, your program's commands
c   might be different; for a GUI interface, you would use an "about box".
c   
c     You should also get your employer (if you work as a programmer) or school,
c   if any, to sign a "copyright disclaimer" for the program, if necessary.
c   For more information on this, and how to apply and follow the GNU GPL, see
c   <http://www.gnu.org/licenses/>.
c   
c     The GNU General Public License does not permit incorporating your program
c   into proprietary programs.  If your program is a subroutine library, you
c   may consider it more useful to permit linking proprietary applications with
c   the library.  If this is what you want to do, use the GNU Lesser General
c   Public License instead of this License.  But first, please read
c   <http://www.gnu.org/philosophy/why-not-lgpl.html>.
c   
c***********************************************************************
c***********************************************************************
