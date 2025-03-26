program dop853_benchmark

  use dop853_module, wp => dop853_wp
  use iso_fortran_env, only: output_unit

  implicit none

  integer,parameter :: n = 6  !! dimension of the system
  real(wp) :: tol            !! integration tolerance
  real(wp) :: x0             !! initial x value
  real(wp) :: xf             !! endpoint of integration
  real(wp),dimension(n) :: y0 !! initial y value
  real(wp) :: mu             !! mu parameter

  ! Command line arguments
  integer :: num_args, i
  character(len=100) :: arg
  logical :: has_csv = .false.

  type(dop853_class)    :: prop
  real(wp),dimension(n) :: y
  real(wp),dimension(1) :: rtol,atol
  real(wp)              :: x
  integer               :: idid
  logical               :: status_ok
  
  ! Declare timing variables for system_clock
  integer(8) :: count_start, count_end, count_rate, count_max
  real(wp)   :: elapsed_time

  ! Check command line arguments
  num_args = command_argument_count()
  if (num_args < 11) then
    write(output_unit,*) 'Usage: jw_cr3bp <mu> <t0> <tf> <y0_0> <y0_1> <y0_2> '//&
                         '<y0_3> <y0_4> <y0_5> <atol> <rtol> [--csv]'
    stop
  end if

  ! Parse command line arguments
  call get_command_argument(1, arg)
  read(arg,*) mu

  call get_command_argument(2, arg)
  read(arg,*) x0

  call get_command_argument(3, arg)
  read(arg,*) xf

  ! Parse initial state vector
  do i = 1, 6
    call get_command_argument(i+3, arg)
    read(arg,*) y0(i)
  end do

  call get_command_argument(10, arg)
  read(arg,*) atol(1)

  call get_command_argument(11, arg)
  read(arg,*) rtol(1)

  ! Check for CSV flag
  do i = 12, num_args
    call get_command_argument(i, arg)
    if (trim(arg) == '--csv') then
      has_csv = .true.
    end if
  end do

  ! Start timing using system_clock
  call system_clock(count_start, count_rate, count_max)

  x = x0   ! initial conditions
  y = y0   !

  ! Initialize the integrator:
  call prop%initialize(fcn=cr3bp_func,n=n,status_ok=status_ok)
  if (.not. status_ok) error stop 'initialization error'

  ! Now, perform the integration:
  call prop%integrate(x,y,xf,rtol,atol,iout=0,idid=idid)

  ! End timing
  call system_clock(count_end)
  
  ! Calculate elapsed time in seconds with higher precision
  elapsed_time = real(count_end - count_start, wp) / real(count_rate, wp)

  ! Print solution in consistent format with higher precision for timing
  write (output_unit,'(A,E23.15,A,E23.15,A,E23.15,A,E23.15,A,E23.15,A,E23.15,A,E15.8)') &
        '[',y(1),', ',y(2),', ',y(3),', ',y(4),', ',y(5),', ',y(6),'] ',elapsed_time

contains

  subroutine cr3bp_func(me,x,y,f)
  !! Right-hand side of CR3BP equation

  implicit none

  class(dop853_class),intent(inout) :: me
  real(wp),intent(in)               :: x
  real(wp),dimension(:),intent(in)  :: y
  real(wp),dimension(:),intent(out) :: f
  
  ! Local variables
  real(wp) :: r1, r2
  
  ! Compute distances to primary masses
  r1 = sqrt((y(1) + mu)**2 + y(2)**2 + y(3)**2)
  r2 = sqrt((y(1) - (1.0_wp-mu))**2 + y(2)**2 + y(3)**2)
  
  ! Equations of motion in the rotating frame
  f(1) = y(4)
  f(2) = y(5)
  f(3) = y(6)
  f(4) = 2.0_wp*y(5) + y(1) - (1.0_wp-mu)*(y(1)+mu)/r1**3 - mu*(y(1)-(1.0_wp-mu))/r2**3
  f(5) = -2.0_wp*y(4) + y(2) - (1.0_wp-mu)*y(2)/r1**3 - mu*y(2)/r2**3
  f(6) = -(1.0_wp-mu)*y(3)/r1**3 - mu*y(3)/r2**3

  end subroutine cr3bp_func

end program dop853_benchmark
