C     PROGRAM TO SOLVE VAN DER POL USING ORIGINAL HAIRER DOP853
      PROGRAM VDPHAIRER
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (N=2)
      DIMENSION Y(N),RTOL(1),ATOL(1),WORK(3000),IWORK(1000)
      DIMENSION RPAR(1),IPAR(1)
      EXTERNAL FCN, SOLOUT
      
      ! Declare timing variables for system_clock
      INTEGER(8) COUNT_START, COUNT_END, COUNT_RATE, COUNT_MAX
      DOUBLE PRECISION ELAPSED_TIME
      
      ! Variables for command-line arguments
      CHARACTER(100) ARG
      INTEGER NARG, ISTAT
      LOGICAL HAS_CSV
      
C     GET COMMAND LINE ARGUMENTS
      NARG = IARGC()
      IF (NARG .LT. 7) THEN
        PRINT *, 'Usage: hw_vanderpol <mu> <t0> <tf> <y0_0> <y0_1>',
     &           ' <atol> <rtol> [--csv]'
        STOP
      ENDIF
      
C     PARSE MU PARAMETER
      CALL GETARG(1, ARG)
      READ(ARG, *, IOSTAT=ISTAT) RPAR(1)
      IF (ISTAT .NE. 0) STOP 'Error reading mu'
      
C     PARSE START TIME
      CALL GETARG(2, ARG)
      READ(ARG, *, IOSTAT=ISTAT) X
      IF (ISTAT .NE. 0) STOP 'Error reading t0'
      
C     PARSE END TIME
      CALL GETARG(3, ARG)
      READ(ARG, *, IOSTAT=ISTAT) XEND
      IF (ISTAT .NE. 0) STOP 'Error reading tf'
      
C     PARSE INITIAL Y VALUES
      CALL GETARG(4, ARG)
      READ(ARG, *, IOSTAT=ISTAT) Y(1)
      IF (ISTAT .NE. 0) STOP 'Error reading y0_0'
      
      CALL GETARG(5, ARG)
      READ(ARG, *, IOSTAT=ISTAT) Y(2)
      IF (ISTAT .NE. 0) STOP 'Error reading y0_1'
      
C     PARSE TOLERANCES
      CALL GETARG(6, ARG)
      READ(ARG, *, IOSTAT=ISTAT) ATOL(1)
      IF (ISTAT .NE. 0) STOP 'Error reading atol'
      
      CALL GETARG(7, ARG)
      READ(ARG, *, IOSTAT=ISTAT) RTOL(1)
      IF (ISTAT .NE. 0) STOP 'Error reading rtol'
      
C     CHECK FOR CSV FLAG
      HAS_CSV = .FALSE.
      IF (NARG .GE. 8) THEN
        CALL GETARG(8, ARG)
        IF (ARG .EQ. '--csv') HAS_CSV = .TRUE.
      ENDIF
      
C     SET METHOD PARAMETERS
      ITOL = 0
      IOUT = 0
      IDID = 0
      
C     INITIALIZE WORK ARRAYS
      DO 10 I=1,3000
         WORK(I) = 0.0D0
10    CONTINUE
      DO 20 I=1,1000
         IWORK(I) = 0
20    CONTINUE
      
C     START TIMING USING SYSTEM_CLOCK
      CALL SYSTEM_CLOCK(COUNT_START, COUNT_RATE, COUNT_MAX)
      
C     CALL THE INTEGRATOR
      CALL DOP853(N,FCN,X,Y,XEND,
     &            RTOL,ATOL,ITOL,
     &            SOLOUT,IOUT,
     &            WORK,3000,IWORK,1000,RPAR,IPAR,IDID)
     
C     END TIMING
      CALL SYSTEM_CLOCK(COUNT_END)
      ELAPSED_TIME = DBLE(COUNT_END - COUNT_START) / DBLE(COUNT_RATE)
      
C     PRINT RESULT IN CONSISTENT FORMAT
      WRITE(*,'(A,E23.15,A,E23.15,A,E15.8)') 
     &     '[',Y(1), ', ', Y(2), '] ', ELAPSED_TIME
      
      END
      
C     ------------------------------------------------
C     RIGHT-HAND SIDE OF VAN DER POL EQUATION
C     ------------------------------------------------
      SUBROUTINE FCN(N,X,Y,F,RPAR,IPAR)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION Y(N),F(N),RPAR(*),IPAR(*)
      
C     VAN DER POL WITH MU FROM RPAR(1)
      DOUBLE PRECISION MU
      MU = RPAR(1)
      
      F(1) = Y(2)
      F(2) = MU*(1.0D0-Y(1)**2)*Y(2) - Y(1)
      
      RETURN
      END
      
C     ------------------------------------------------
C     DUMMY SOLOUT ROUTINE (NOT USED WITH IOUT=0)
C     ------------------------------------------------
      SUBROUTINE SOLOUT(NR,XOLD,X,Y,N,CON,ICOMP,ND,
     &                   RPAR,IPAR,IRTRN,XOUT)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION Y(N),CON(8*ND),ICOMP(ND),RPAR(*),IPAR(*)
      
C     DUMMY ROUTINE - NO OUTPUT DURING INTEGRATION
      IRTRN = 0
      
      RETURN
      END
