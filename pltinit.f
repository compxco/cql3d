c
c
      subroutine pltinit
      implicit integer (i-n), real*8 (a-h,o-z)
c................................................................
c     Initiates PGPLOT
c................................................................

      include 'param.h'
      include 'comm.h'
CMPIINSERT_INCLUDE

c      integer pgbeg
      integer pgopen

      INTEGER PG_I, PG_L, PG_C1, PG_C2, PG_NC !YuP
      CHARACTER*256 PG_VAL ! YuP[2020-06-29] changed 16 to 256

CMPIINSERT_IF_RANK_NE_0_RETURN
 ! make plots on mpirank.eq.0 only

      if (noplots.eq."enabled1") return

c      write(*,*) 'PLTINIT-1'
c      Remember, pgbeg should be integer
c      ier=pgbeg(0,'plot.ps/VPS',1,1)
c      Remember, pgopen should be integer
      write(t_,1000) mnemonic(1:length_char(mnemonic))
 1000 format(a,".ps/PS") !YuP: was /VPS (vertical black&white)
                           ! Use /VCPS for vertical Color pages
      ier=PGOPEN(t_)
      CALL PGSCI(1)
      CALL PGSLW(lnwidth)
      write(*,*) 'PLTINIT-1 ier=1 is OK: ier=',ier
c      ier=pgbeg(0,'?',1,1)
c      if (ier.ne.1) write(*,*) 
c     +              'Problem1 with initiating PGPLOT library'
c      ier=pgbeg(0,'?',1,1)
c      write(*,*) 'ier=',ier

      !YuP[2018-02-07] Added:
      !inquire PGPLOT general information: use PGQINF()
      !---------- First arg is the input:
      !'VERSION'     - version of PGPLOT software in use.
      !'STATE'       - status of PGPLOT ('OPEN' if a graphics device
      !                is open for output, 'CLOSED' otherwise).
      !'USER'        - the username associated with the calling program.
      !'NOW'         - current date and time (e.g., '17-FEB-1986 10:04').
      !'DEVICE'    * - current PGPLOT device or file.
      !'FILE'      * - current PGPLOT device or file.
      !'TYPE'      * - device-type of the current PGPLOT device.
      !'DEV/TYPE'  * - current PGPLOT device and type, in a form which
      !                is acceptable as an argument for PGBEG.
      !'HARDCOPY'  * - is the current device a hardcopy device? ('YES' or
      !                'NO').
      !'TERMINAL'  * - is the current device the user's interactive
      !                terminal? ('YES' or 'NO').
      !'CURSOR'    * - does the current device have a graphics cursor?
      !                ('YES' or 'NO').
      ! Two other arg. are outputs:
      ! CALL PGQINF('TYPE', PG_VAL, PG_L)
      ! WRITE (*,*) 'PGPLOT device type: ', PG_VAL(1:PG_L)
      ! CALL PGQINF('DEVICE', PG_VAL, PG_L)
      ! WRITE (*,*) 'PGPLOT device: ', PG_VAL(1:PG_L)
      ! CALL PGQINF('USER', PG_VAL, PG_L)
      ! WRITE (*,*) 'PGPLOT user: ', PG_VAL(1:PG_L)
      ! CALL PGQINF('NOW', PG_VAL, PG_L)
      ! WRITE (*,*) 'PGPLOT time now: ', PG_VAL(1:PG_L)
      
      !Inquire color index range:
      CALL PGQCIR(PG_C1, PG_C2)
      PG_NC = MAX(0, PG_C2-PG_C1+1)
      WRITE (*,*) 'Number of color indices used for image: ', PG_NC
      ! On Yuri's PC: printed --   PG_NC=240
      IF (PG_NC .LT.8) THEN 
         WRITE (*,*) 'Not enough colors available on this device'
         STOP
      ELSE
         WRITE (*,*)
      END IF

      return
      end
