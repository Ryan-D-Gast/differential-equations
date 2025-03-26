C     PROGRAM TO SOLVE CR3BP USING ORIGINAL HAIRER DOP853
      PROGRAM CR3BPHAIRER
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (N=6)
      DIMENSION Y(N),RTOL(1),ATOL(1),WORK(8000),IWORK(1000)
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
      IF (NARG .LT. 11) THEN
        PRINT *, 'Usage: hw_cr3bp <mu> <t0> <tf> <y0_0> <y0_1> <y0_2>',
     &           ' <y0_3> <y0_4> <y0_5> <atol> <rtol> [--csv]'
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
      DO I=1,6
        CALL GETARG(I+3, ARG)
        READ(ARG, *, IOSTAT=ISTAT) Y(I)
        IF (ISTAT .NE. 0) STOP 'Error reading initial values'
      ENDDO
      
C     PARSE TOLERANCES
      CALL GETARG(10, ARG)
      READ(ARG, *, IOSTAT=ISTAT) ATOL(1)
      IF (ISTAT .NE. 0) STOP 'Error reading atol'
      
      CALL GETARG(11, ARG)
      READ(ARG, *, IOSTAT=ISTAT) RTOL(1)
      IF (ISTAT .NE. 0) STOP 'Error reading rtol'
      
C     CHECK FOR CSV FLAG
      HAS_CSV = .FALSE.
      IF (NARG .GE. 12) THEN
        CALL GETARG(12, ARG)
        IF (ARG .EQ. '--csv') HAS_CSV = .TRUE.
      ENDIF
      
C     SET METHOD PARAMETERS
      ITOL = 0
      IOUT = 0
      IDID = 0
      
C     INITIALIZE WORK ARRAYS
      DO 10 I=1,8000
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
     &            WORK,8000,IWORK,1000,RPAR,IPAR,IDID)
     
C     END TIMING
      CALL SYSTEM_CLOCK(COUNT_END)
      ELAPSED_TIME = DBLE(COUNT_END - COUNT_START) / DBLE(COUNT_RATE)
      
C     PRINT RESULT IN CONSISTENT FORMAT
      WRITE(*,'(A,E23.15,A,E23.15,A,E23.15,A,E23.15,A,E23.15,A,E23.15,A,
     & E15.8)') '[',Y(1),', ',Y(2),', ',Y(3),', ',Y(4),', ',Y(5),', ',
     & Y(6),'] ',ELAPSED_TIME
      
      END
      
C     ------------------------------------------------
C     RIGHT-HAND SIDE OF CR3BP EQUATION
C     ------------------------------------------------
      SUBROUTINE FCN(N,X,Y,F,RPAR,IPAR)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION Y(N),F(N),RPAR(*),IPAR(*)
      
C     CR3BP WITH MU FROM RPAR(1)
      DOUBLE PRECISION MU
      MU = RPAR(1)
      
C     COMPUTE DISTANCES TO PRIMARY MASSES
      R1 = DSQRT((Y(1) + MU)**2 + Y(2)**2 + Y(3)**2)
      R2 = DSQRT((Y(1) - (1.0D0-MU))**2 + Y(2)**2 + Y(3)**2)
      
C     EQUATIONS OF MOTION IN THE ROTATING FRAME
      F(1) = Y(4)
      F(2) = Y(5)
      F(3) = Y(6)
      F(4) = 2.0D0*Y(5) + Y(1) - (1.0D0-MU)*(Y(1)+MU)/R1**3 
     &       - MU*(Y(1)-(1.0D0-MU))/R2**3
      F(5) = -2.0D0*Y(4) + Y(2) - (1.0D0-MU)*Y(2)/R1**3 
     &       - MU*Y(2)/R2**3
      F(6) = -(1.0D0-MU)*Y(3)/R1**3 - MU*Y(3)/R2**3
      
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
