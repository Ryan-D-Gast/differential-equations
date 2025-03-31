program dop853_benchmark

  use dop853_module, wp => dop853_wp
  use iso_fortran_env, only: output_unit

  implicit none

  integer,parameter :: n = 2  !! dimension of the system
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
  if (num_args < 7) then
    write(output_unit,*) 'Usage: jw_vanderpol <mu> <t0> <tf> <y0_0> <y0_1> <atol> <rtol> [--csv]'
    stop
  end if

  ! Parse command line arguments
  call get_command_argument(1, arg)
  read(arg,*) mu

  call get_command_argument(2, arg)
  read(arg,*) x0

  call get_command_argument(3, arg)
  read(arg,*) xf

  call get_command_argument(4, arg)
  read(arg,*) y0(1)

  call get_command_argument(5, arg)
  read(arg,*) y0(2)

  call get_command_argument(6, arg)
  read(arg,*) tol
  atol = tol

  call get_command_argument(7, arg)
  read(arg,*) tol
  rtol = tol

  ! Check for CSV flag
  do i = 8, num_args
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
  call prop%initialize(fcn=fvpol,n=n,status_ok=status_ok)
  if (.not. status_ok) error stop 'initialization error'

  ! Now, perform the integration:
  call prop%integrate(x,y,xf,rtol,atol,iout=0,idid=idid)

  ! End timing
  call system_clock(count_end)
  
  ! Calculate elapsed time in seconds with higher precision
  elapsed_time = real(count_end - count_start, wp) / real(count_rate, wp)

  ! Print solution in consistent format with higher precision for timing
  write (output_unit,'(A,E23.15,A,E23.15,A,E15.8)') '[',y(1),', ',y(2),'] ',elapsed_time

contains

  subroutine fvpol(me,x,y,f)
  !! Right-hand side of van der Pol's equation

  implicit none

  class(dop853_class),intent(inout) :: me
  real(wp),intent(in)               :: x
  real(wp),dimension(:),intent(in)  :: y
  real(wp),dimension(:),intent(out) :: f

  f(1) = y(2)
  f(2) = mu*(1.0_wp-y(1)**2)*y(2) - y(1)

  end subroutine fvpol

end program dop853_benchmark