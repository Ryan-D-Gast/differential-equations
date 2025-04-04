!*****************************************************************************************
!> author: Jacob Williams
!
!  Modern Fortran Edition of the DOP853 ODE Solver.
!
!### See also
!  * [DOP853.f](http://www.unige.ch/~hairer/prog/nonstiff/dop853.f) -- The original code
!
!### History
!  * Jacob Williams : December 2015 : Created module from the DOP853 Fortran 77 code.
!  * Development continues at [GitHub](https://github.com/jacobwilliams/dop853).
!
!@note The default real kind (`wp`) can be
!      changed using optional preprocessor flags.
!      This library was built with real kind:
#ifdef REAL32
!      `real(kind=real32)` [4 bytes]
#elif REAL64
!      `real(kind=real64)` [8 bytes]
#elif REAL128
!      `real(kind=real128)` [16 bytes]
#else
!      `real(kind=real64)` [8 bytes]
#endif


    module dop853_module

    use, intrinsic :: iso_fortran_env

    implicit none

    private

#ifdef REAL32
    integer,parameter,public :: dop853_wp = real32   !! real kind used by this module [4 bytes]
#elif REAL64
    integer,parameter,public :: dop853_wp = real64   !! real kind used by this module [8 bytes]
#elif REAL128
    integer,parameter,public :: dop853_wp = real128  !! real kind used by this module [16 bytes]
#else
    integer,parameter,public :: dop853_wp = real64   !! real kind used by this module [8 bytes]
#endif

    integer,parameter :: wp = dop853_wp  !! local copy of `dop853_wp` with a shorter name
    real(wp),parameter :: uround = epsilon(1.0_wp)  !! machine \( \epsilon \)

    !integration constants (formerly in dp86co):
    real(wp),parameter :: c2 = 0.526001519587677318785587544488e-01_wp
    real(wp),parameter :: c3 = 0.789002279381515978178381316732e-01_wp
    real(wp),parameter :: c4 = 0.118350341907227396726757197510_wp
    real(wp),parameter :: c5 = 0.281649658092772603273242802490_wp
    real(wp),parameter :: c6 = 0.333333333333333333333333333333_wp
    real(wp),parameter :: c7 = 0.25_wp
    real(wp),parameter :: c8 = 0.307692307692307692307692307692_wp
    real(wp),parameter :: c9 = 0.651282051282051282051282051282_wp
    real(wp),parameter :: c10 = 0.6_wp
    real(wp),parameter :: c11 = 0.857142857142857142857142857142_wp
    real(wp),parameter :: c14 = 0.1_wp
    real(wp),parameter :: c15 = 0.2_wp
    real(wp),parameter :: c16 = 0.777777777777777777777777777778_wp
    real(wp),parameter :: b1 = 5.42937341165687622380535766363e-2_wp
    real(wp),parameter :: b6 = 4.45031289275240888144113950566_wp
    real(wp),parameter :: b7 = 1.89151789931450038304281599044_wp
    real(wp),parameter :: b8 = -5.8012039600105847814672114227_wp
    real(wp),parameter :: b9 = 3.1116436695781989440891606237e-1_wp
    real(wp),parameter :: b10 = -1.52160949662516078556178806805e-1_wp
    real(wp),parameter :: b11 = 2.01365400804030348374776537501e-1_wp
    real(wp),parameter :: b12 = 4.47106157277725905176885569043e-2_wp
    real(wp),parameter :: bhh1 = 0.244094488188976377952755905512_wp
    real(wp),parameter :: bhh2 = 0.733846688281611857341361741547_wp
    real(wp),parameter :: bhh3 = 0.220588235294117647058823529412e-1_wp
    real(wp),parameter :: er1 = 0.1312004499419488073250102996e-01_wp
    real(wp),parameter :: er6 = -0.1225156446376204440720569753e+01_wp
    real(wp),parameter :: er7 = -0.4957589496572501915214079952_wp
    real(wp),parameter :: er8 = 0.1664377182454986536961530415e+01_wp
    real(wp),parameter :: er9 = -0.3503288487499736816886487290_wp
    real(wp),parameter :: er10 = 0.3341791187130174790297318841_wp
    real(wp),parameter :: er11 = 0.8192320648511571246570742613e-01_wp
    real(wp),parameter :: er12 = -0.2235530786388629525884427845e-01_wp
    real(wp),parameter :: a21 = 5.26001519587677318785587544488e-2_wp
    real(wp),parameter :: a31 = 1.97250569845378994544595329183e-2_wp
    real(wp),parameter :: a32 = 5.91751709536136983633785987549e-2_wp
    real(wp),parameter :: a41 = 2.95875854768068491816892993775e-2_wp
    real(wp),parameter :: a43 = 8.87627564304205475450678981324e-2_wp
    real(wp),parameter :: a51 = 2.41365134159266685502369798665e-1_wp
    real(wp),parameter :: a53 = -8.84549479328286085344864962717e-1_wp
    real(wp),parameter :: a54 = 9.24834003261792003115737966543e-1_wp
    real(wp),parameter :: a61 = 3.7037037037037037037037037037e-2_wp
    real(wp),parameter :: a64 = 1.70828608729473871279604482173e-1_wp
    real(wp),parameter :: a65 = 1.25467687566822425016691814123e-1_wp
    real(wp),parameter :: a71 = 3.7109375e-2_wp
    real(wp),parameter :: a74 = 1.70252211019544039314978060272e-1_wp
    real(wp),parameter :: a75 = 6.02165389804559606850219397283e-2_wp
    real(wp),parameter :: a76 = -1.7578125e-2_wp
    real(wp),parameter :: a81 = 3.70920001185047927108779319836e-2_wp
    real(wp),parameter :: a84 = 1.70383925712239993810214054705e-1_wp
    real(wp),parameter :: a85 = 1.07262030446373284651809199168e-1_wp
    real(wp),parameter :: a86 = -1.53194377486244017527936158236e-2_wp
    real(wp),parameter :: a87 = 8.27378916381402288758473766002e-3_wp
    real(wp),parameter :: a91 = 6.24110958716075717114429577812e-1_wp
    real(wp),parameter :: a94 = -3.36089262944694129406857109825_wp
    real(wp),parameter :: a95 = -8.68219346841726006818189891453e-1_wp
    real(wp),parameter :: a96 = 2.75920996994467083049415600797e+1_wp
    real(wp),parameter :: a97 = 2.01540675504778934086186788979e+1_wp
    real(wp),parameter :: a98 = -4.34898841810699588477366255144e+1_wp
    real(wp),parameter :: a101 = 4.77662536438264365890433908527e-1_wp
    real(wp),parameter :: a104 = -2.48811461997166764192642586468_wp
    real(wp),parameter :: a105 = -5.90290826836842996371446475743e-1_wp
    real(wp),parameter :: a106 = 2.12300514481811942347288949897e+1_wp
    real(wp),parameter :: a107 = 1.52792336328824235832596922938e+1_wp
    real(wp),parameter :: a108 = -3.32882109689848629194453265587e+1_wp
    real(wp),parameter :: a109 = -2.03312017085086261358222928593e-2_wp
    real(wp),parameter :: a111 = -9.3714243008598732571704021658e-1_wp
    real(wp),parameter :: a114 = 5.18637242884406370830023853209_wp
    real(wp),parameter :: a115 = 1.09143734899672957818500254654_wp
    real(wp),parameter :: a116 = -8.14978701074692612513997267357_wp
    real(wp),parameter :: a117 = -1.85200656599969598641566180701e+1_wp
    real(wp),parameter :: a118 = 2.27394870993505042818970056734e+1_wp
    real(wp),parameter :: a119 = 2.49360555267965238987089396762_wp
    real(wp),parameter :: a1110 = -3.0467644718982195003823669022_wp
    real(wp),parameter :: a121 = 2.27331014751653820792359768449_wp
    real(wp),parameter :: a124 = -1.05344954667372501984066689879e+1_wp
    real(wp),parameter :: a125 = -2.00087205822486249909675718444_wp
    real(wp),parameter :: a126 = -1.79589318631187989172765950534e+1_wp
    real(wp),parameter :: a127 = 2.79488845294199600508499808837e+1_wp
    real(wp),parameter :: a128 = -2.85899827713502369474065508674_wp
    real(wp),parameter :: a129 = -8.87285693353062954433549289258_wp
    real(wp),parameter :: a1210 = 1.23605671757943030647266201528e+1_wp
    real(wp),parameter :: a1211 = 6.43392746015763530355970484046e-1_wp
    real(wp),parameter :: a141 = 5.61675022830479523392909219681e-2_wp
    real(wp),parameter :: a147 = 2.53500210216624811088794765333e-1_wp
    real(wp),parameter :: a148 = -2.46239037470802489917441475441e-1_wp
    real(wp),parameter :: a149 = -1.24191423263816360469010140626e-1_wp
    real(wp),parameter :: a1410 = 1.5329179827876569731206322685e-1_wp
    real(wp),parameter :: a1411 = 8.20105229563468988491666602057e-3_wp
    real(wp),parameter :: a1412 = 7.56789766054569976138603589584e-3_wp
    real(wp),parameter :: a1413 = -8.298e-3_wp
    real(wp),parameter :: a151 = 3.18346481635021405060768473261e-2_wp
    real(wp),parameter :: a156 = 2.83009096723667755288322961402e-2_wp
    real(wp),parameter :: a157 = 5.35419883074385676223797384372e-2_wp
    real(wp),parameter :: a158 = -5.49237485713909884646569340306e-2_wp
    real(wp),parameter :: a1511 = -1.08347328697249322858509316994e-4_wp
    real(wp),parameter :: a1512 = 3.82571090835658412954920192323e-4_wp
    real(wp),parameter :: a1513 = -3.40465008687404560802977114492e-4_wp
    real(wp),parameter :: a1514 = 1.41312443674632500278074618366e-1_wp
    real(wp),parameter :: a161 = -4.28896301583791923408573538692e-1_wp
    real(wp),parameter :: a166 = -4.69762141536116384314449447206_wp
    real(wp),parameter :: a167 = 7.68342119606259904184240953878_wp
    real(wp),parameter :: a168 = 4.06898981839711007970213554331_wp
    real(wp),parameter :: a169 = 3.56727187455281109270669543021e-1_wp
    real(wp),parameter :: a1613 = -1.39902416515901462129418009734e-3_wp
    real(wp),parameter :: a1614 = 2.9475147891527723389556272149_wp
    real(wp),parameter :: a1615 = -9.15095847217987001081870187138_wp
    real(wp),parameter :: d41 = -0.84289382761090128651353491142e+01_wp
    real(wp),parameter :: d46 = 0.56671495351937776962531783590_wp
    real(wp),parameter :: d47 = -0.30689499459498916912797304727e+01_wp
    real(wp),parameter :: d48 = 0.23846676565120698287728149680e+01_wp
    real(wp),parameter :: d49 = 0.21170345824450282767155149946e+01_wp
    real(wp),parameter :: d410 = -0.87139158377797299206789907490_wp
    real(wp),parameter :: d411 = 0.22404374302607882758541771650e+01_wp
    real(wp),parameter :: d412 = 0.63157877876946881815570249290_wp
    real(wp),parameter :: d413 = -0.88990336451333310820698117400e-01_wp
    real(wp),parameter :: d414 = 0.18148505520854727256656404962e+02_wp
    real(wp),parameter :: d415 = -0.91946323924783554000451984436e+01_wp
    real(wp),parameter :: d416 = -0.44360363875948939664310572000e+01_wp
    real(wp),parameter :: d51 = 0.10427508642579134603413151009e+02_wp
    real(wp),parameter :: d56 = 0.24228349177525818288430175319e+03_wp
    real(wp),parameter :: d57 = 0.16520045171727028198505394887e+03_wp
    real(wp),parameter :: d58 = -0.37454675472269020279518312152e+03_wp
    real(wp),parameter :: d59 = -0.22113666853125306036270938578e+02_wp
    real(wp),parameter :: d510 = 0.77334326684722638389603898808e+01_wp
    real(wp),parameter :: d511 = -0.30674084731089398182061213626e+02_wp
    real(wp),parameter :: d512 = -0.93321305264302278729567221706e+01_wp
    real(wp),parameter :: d513 = 0.15697238121770843886131091075e+02_wp
    real(wp),parameter :: d514 = -0.31139403219565177677282850411e+02_wp
    real(wp),parameter :: d515 = -0.93529243588444783865713862664e+01_wp
    real(wp),parameter :: d516 = 0.35816841486394083752465898540e+02_wp
    real(wp),parameter :: d61 = 0.19985053242002433820987653617e+02_wp
    real(wp),parameter :: d66 = -0.38703730874935176555105901742e+03_wp
    real(wp),parameter :: d67 = -0.18917813819516756882830838328e+03_wp
    real(wp),parameter :: d68 = 0.52780815920542364900561016686e+03_wp
    real(wp),parameter :: d69 = -0.11573902539959630126141871134e+02_wp
    real(wp),parameter :: d610 = 0.68812326946963000169666922661e+01_wp
    real(wp),parameter :: d611 = -0.10006050966910838403183860980e+01_wp
    real(wp),parameter :: d612 = 0.77771377980534432092869265740_wp
    real(wp),parameter :: d613 = -0.27782057523535084065932004339e+01_wp
    real(wp),parameter :: d614 = -0.60196695231264120758267380846e+02_wp
    real(wp),parameter :: d615 = 0.84320405506677161018159903784e+02_wp
    real(wp),parameter :: d616 = 0.11992291136182789328035130030e+02_wp
    real(wp),parameter :: d71 = -0.25693933462703749003312586129e+02_wp
    real(wp),parameter :: d76 = -0.15418974869023643374053993627e+03_wp
    real(wp),parameter :: d77 = -0.23152937917604549567536039109e+03_wp
    real(wp),parameter :: d78 = 0.35763911791061412378285349910e+03_wp
    real(wp),parameter :: d79 = 0.93405324183624310003907691704e+02_wp
    real(wp),parameter :: d710 = -0.37458323136451633156875139351e+02_wp
    real(wp),parameter :: d711 = 0.10409964950896230045147246184e+03_wp
    real(wp),parameter :: d712 = 0.29840293426660503123344363579e+02_wp
    real(wp),parameter :: d713 = -0.43533456590011143754432175058e+02_wp
    real(wp),parameter :: d714 = 0.96324553959188282948394950600e+02_wp
    real(wp),parameter :: d715 = -0.39177261675615439165231486172e+02_wp
    real(wp),parameter :: d716 = -0.14972683625798562581422125276e+03_wp

    type,public :: dop853_class

        private

        !internal variables:
        integer :: n      = 0   !! the dimension of the system
        integer :: nfcn   = 0   !! number of function evaluations
        integer :: nstep  = 0   !! number of computed steps
        integer :: naccpt = 0   !! number of accepted steps
        integer :: nrejct = 0   !! number of rejected steps (due to error test),
                                !! (step rejections in the first step are not counted)
        integer :: nrdens = 0   !! number of components, for which dense output
                                !! is required. for `0 < nrdens < n` the components
                                !! (for which dense output is required) have to be
                                !! specified in `icomp(1),...,icomp(nrdens)`.
        real(wp) :: h = 0.0_wp  !! predicted step size of the last accepted step

        !input paramters:
        !  these parameters allow
        !  to adapt the code to the problem and to the needs of
        !  the user. set them on class initialization.

        integer :: iprint = output_unit   !! switch for printing error messages
                                          !! if `iprint==0` no messages are being printed
                                          !! if `iprint/=0` messages are printed with
                                          !! `write (iprint,*)` ...

        integer :: nmax = 100000       !! the maximal number of allowed steps.
        integer :: nstiff = 1000       !! test for stiffness is activated after step number
                                       !! `j*nstiff` (`j` integer), provided `nstiff>0`.
                                       !! for negative `nstiff` the stiffness test is
                                       !! never activated.
        real(wp) :: hinitial = 0.0_wp  !! initial step size, for `hinitial=0` an initial guess
                                       !! is computed with help of the function [[hinit]].
        real(wp) :: hmax = 0.0_wp      !! maximal step size, defaults to `xend-x` if `hmax=0`.
        real(wp) :: safe = 0.9_wp      !! safety factor in step size prediction
        real(wp) :: fac1 = 0.333_wp    !! parameter for step size selection.
                                       !! the new step size is chosen subject to the restriction
                                       !! `fac1 <= hnew/hold <= fac2`
        real(wp) :: fac2 = 6.0_wp      !! parameter for step size selection.
                                       !! the new step size is chosen subject to the restriction
                                       !! `fac1 <= hnew/hold <= fac2`
        real(wp) :: beta = 0.0_wp      !! is the `beta` for stabilized step size control
                                       !! (see section iv.2). positive values of beta ( <= 0.04 )
                                       !! make the step size control more stable.

        integer,dimension(:),allocatable  :: icomp      !! `dimension(nrdens)`
                                                        !! the components for which dense output is required
        real(wp),dimension(:),allocatable :: cont       !! `dimension(8*nrdens)`

        integer :: iout = 0 !! copy of `iout` input to [[dop853]]

        !formerly in the condo8 common block:
        real(wp) :: xold = 0.0_wp
        real(wp) :: hout = 0.0_wp

        !user-defined procedures:
        procedure(deriv_func),pointer  :: fcn    => null() !! subroutine computing the value of `f(x,y)`
        procedure(solout_func),pointer :: solout => null() !! subroutine providing the
                                                           !! numerical solution during integration.
                                                           !! if `iout>=1`, it is called during integration.

        contains

        private

        procedure,public :: initialize => set_parameters    !! initialization routine.
        procedure,public :: integrate  => dop853            !! main integration routine.
        procedure,public :: destroy    => destroy_dop853    !! destructor.
        procedure,public :: info       => get_dop853_info   !! to get info after a run.

        procedure :: dp86co
        procedure :: hinit
        procedure,public :: contd8  !! can be called in user's [[solout_func]] for dense output.

    end type dop853_class

    abstract interface

        subroutine deriv_func(me,x,y,f)
            !! subroutine computing the value of \( dy/dx = f(x,y) \)
            import :: wp,dop853_class
            implicit none
            class(dop853_class),intent(inout)   :: me
            real(wp),intent(in)                 :: x    !! independent variable \(x\)
            real(wp),dimension(:),intent(in)    :: y    !! state vector \( y(x) \) [size n]
            real(wp),dimension(:),intent(out)   :: f    !! derivative vector \( f(x,y) = dy/dx \) [size n]
        end subroutine deriv_func

        subroutine solout_func(me,nr,xold,x,y,irtrn,xout)
            !! `solout` furnishes the solution `y` at the `nr`-th
            !! grid-point `x` (thereby the initial value is
            !! the first grid-point).
            import :: wp,dop853_class
            implicit none
            class(dop853_class),intent(inout) :: me
            integer,intent(in)                :: nr    !! grid point (0,1,...)
            real(wp),intent(in)               :: xold  !! the preceding grid point
            real(wp),intent(in)               :: x     !! current grid point
            real(wp),dimension(:),intent(in)  :: y     !! state vector \( y(x) \) [size n]
            integer,intent(inout)             :: irtrn !! serves to interrupt the integration. if
                                                       !! `irtrn` is set `<0`, [[dop853]] will return to
                                                       !! the calling program. if the numerical solution
                                                       !! is altered in `solout`, set `irtrn = 2`.
            real(wp),intent(out)              :: xout  !! `xout` can be used for efficient intermediate output
                                                       !! if one puts `iout=3`. when `nr=1` define the first
                                                       !! output point `xout` in `solout`. the subroutine
                                                       !! `solout` will be called only when `xout` is in the
                                                       !! interval `[xold,x]`; during this call
                                                       !! a new value for `xout` can be defined, etc.
        end subroutine solout_func

    end interface

    contains
!*****************************************************************************************

!*****************************************************************************************
!>
!  Get info from a [[dop853_class]].

    subroutine get_dop853_info(me,n,nfcn,nstep,naccpt,nrejct,h,iout)

    implicit none

    class(dop853_class),intent(in) :: me
    integer,intent(out),optional   :: n      !! dimension of the system
    integer,intent(out),optional   :: nfcn   !! number of function evaluations
    integer,intent(out),optional   :: nstep  !! number of computed steps
    integer,intent(out),optional   :: naccpt !! number of accepted steps
    integer,intent(out),optional   :: nrejct !! number of rejected steps (due to error test),
                                             !! (step rejections in the first step are not counted)
    real(wp),intent(out),optional  :: h      !! predicted step size of the last accepted step
    integer,intent(out),optional   :: iout   !! `iout` flag passed into [[dop853]], used to
                                             !! specify how `solout` is called during integration.

    if (present(n     )) n      = me%n
    if (present(nfcn  )) nfcn   = me%nfcn
    if (present(nstep )) nstep  = me%nstep
    if (present(naccpt)) naccpt = me%naccpt
    if (present(nrejct)) nrejct = me%nrejct
    if (present(h))      h      = me%h
    if (present(iout))   iout   = me%iout

    end subroutine get_dop853_info
!*****************************************************************************************

!*****************************************************************************************
!>
!  Destructor for [[dop853_class]].

    subroutine destroy_dop853(me)

    implicit none

    class(dop853_class),intent(out) :: me

    end subroutine destroy_dop853
!*****************************************************************************************

!*****************************************************************************************
!>
!  Set the optional inputs for [[dop853]].
!
!@note In the original code, these were part of the `work` and `iwork` arrays.

    subroutine set_parameters(me,n,fcn,solout,iprint,nstiff,nmax,hinitial,&
                              hmax,safe,fac1,fac2,beta,icomp,status_ok)

    implicit none

    class(dop853_class),intent(inout)        :: me
    integer,intent(in)                       :: n         !! the dimension of the system (size of \(y\) and \(y'\) vectors)
    procedure(deriv_func)                    :: fcn       !! subroutine computing the value of \( y' = f(x,y) \)
    procedure(solout_func),optional          :: solout    !! subroutine providing the
                                                          !! numerical solution during integration.
                                                          !! if `iout>=1`, it is called during integration.
                                                          !! supply a dummy subroutine if `iout=0`.
    integer,intent(in),optional              :: iprint    !! switch for printing error messages
                                                          !! if `iprint==0` no messages are being printed
                                                          !! if `iprint/=0` messages are printed with
                                                          !! `write (iprint,*)` ...
    integer,intent(in),optional              :: nstiff    !! test for stiffness is activated after step number
                                                          !! `j*nstiff` (`j` integer), provided `nstiff>0`.
                                                          !! for negative `nstiff` the stiffness test is
                                                          !! never activated.
    integer,intent(in),optional              :: nmax      !! the maximal number of allowed steps.
    real(wp),intent(in),optional             :: hinitial  !! initial step size, for `hinitial=0` an initial guess
                                                          !! is computed with help of the function [[hinit]].
    real(wp),intent(in),optional             :: hmax      !! maximal step size, defaults to `xend-x` if `hmax=0`.
    real(wp),intent(in),optional             :: safe      !! safety factor in step size prediction
    real(wp),intent(in),optional             :: fac1      !! parameter for step size selection.
                                                          !! the new step size is chosen subject to the restriction
                                                          !! `fac1 <= hnew/hold <= fac2`
    real(wp),intent(in),optional             :: fac2      !! parameter for step size selection.
                                                          !! the new step size is chosen subject to the restriction
                                                          !! `fac1 <= hnew/hold <= fac2`
    real(wp),intent(in),optional             :: beta      !! is the `beta` for stabilized step size control
                                                          !! (see section iv.2). positive values of `beta` ( <= 0.04 )
                                                          !! make the step size control more stable.
    integer,dimension(:),intent(in),optional :: icomp     !! the components for which dense output is required (size from 0 to `n`).
    logical,intent(out)                      :: status_ok !! will be false for invalid inputs.

    call me%destroy()

    status_ok = .true.

    !required inputs:
    me%n = n
    me%fcn => fcn

    !optional inputs:

    if (present(solout)) me%solout => solout

    if (present(iprint))   me%iprint   = iprint
    if (present(nstiff))   me%nstiff   = nstiff
    if (present(hinitial)) me%hinitial = hinitial
    if (present(hmax))     me%hmax     = hmax
    if (present(fac1))     me%fac1     = fac1
    if (present(fac2))     me%fac2     = fac2

    if (present(nmax)) then
        if ( nmax<=0 ) then
            if ( me%iprint/=0 ) &
                write (me%iprint,*) ' wrong input nmax=', nmax
            status_ok = .false.
        else
            me%nmax = nmax
        end if
    end if

    if (present(safe)) then
        if ( safe>=1.0_wp .or. safe<=1.0e-4_wp ) then
            if ( me%iprint/=0 ) &
                write (me%iprint,*) ' curious input for safety factor safe:', &
                                    safe
            status_ok = .false.
        else
            me%safe = safe
        end if
    end if

    if (present(beta)) then
        if ( beta<=0.0_wp ) then
           me%beta = 0.0_wp
        else
           if ( beta>0.2_wp ) then
              if ( me%iprint/=0 ) write (me%iprint,*) &
                                    ' curious input for beta: ', beta
              status_ok = .false.
           else
              me%beta = beta
           end if
        end if
    end if

    if (present(icomp)) then
        me%nrdens = size(icomp)
        !check validity of icomp array:
        if (size(icomp)<=me%n .and. all(icomp>0 .and. icomp<=me%n)) then
            allocate(me%icomp(me%nrdens));  me%icomp = icomp
            allocate(me%cont(8*me%nrdens)); me%cont = 0.0_wp
        else
            if ( me%iprint/=0 ) write (me%iprint,*) &
                                  ' invalid icomp array: ',icomp
            status_ok = .false.
        end if
    end if

    end subroutine set_parameters
!*****************************************************************************************

!*****************************************************************************************
!>
!  Numerical solution of a system of first order
!  ordinary differential equations \( y'=f(x,y) \).
!  This is an explicit Runge-Kutta method of order 8(5,3)
!  due to Dormand & Prince (with stepsize control and
!  dense output).
!
!### Authors
!  * E. Hairer and G. Wanner
!    Universite de Geneve, Dept. De Mathematiques
!    ch-1211 geneve 24, switzerland
!    e-mail:  ernst.hairer@unige.ch
!             gerhard.wanner@unige.ch
!  * Version of October 11, 2009
!    (new option `iout=3` for sparse dense output)
!  * Jacob Williams, Dec 2015: significant refactoring into modern Fortran.
!
!### Reference
!  * E. Hairer, S.P. Norsett and G. Wanner, [Solving Ordinary
!    Differential Equations I. Nonstiff Problems. 2nd Edition](http://www.unige.ch/~hairer/books.html).
!    Springer Series in Computational Mathematics, Springer-Verlag (1993)

      subroutine dop853(me,x,y,xend,rtol,atol,iout,idid)

      implicit none

      class(dop853_class),intent(inout)       :: me
      real(wp),intent(inout)                  :: x      !! *input:* initial value of independent variable.
                                                        !! *output:* `x` for which the solution has been computed
                                                        !! (after successful return `x=xend`).
      real(wp),dimension(:),intent(inout)     :: y      !! *input:* initial values for `y`. [size n]
                                                        !!
                                                        !! *output:* numerical solution at `x`.
      real(wp),intent(in)                     :: xend   !! final x-value (`xend`-`x` may be positive or negative)
      real(wp),dimension(:),intent(in)        :: rtol   !! relative error tolerance. `rtol` and `atol`
                                                        !! can be both scalars or else both vectors of length `n`.
      real(wp),dimension(:),intent(in)        :: atol   !! absolute error tolerance. `rtol` and `atol`
                                                        !! can be both scalars or else both vectors of length `n`.
                                                        !! `atol` should be strictly positive (possibly very small)
      integer,intent(in)                      :: iout   !! switch for calling the subroutine `solout`:
                                                        !!
                                                        !! * `iout=0`: subroutine is never called
                                                        !! * `iout=1`: subroutine is called after every successful step
                                                        !! * `iout=2`: dense output is performed after every successful step
                                                        !! * `iout=3`: dense output is performed in steps defined by the user
                                                        !!          (see `xout` above)
      integer,intent(out)                     :: idid   !! reports on successfulness upon return:
                                                        !!
                                                        !! * `idid=1`  computation successful,
                                                        !! * `idid=2`  comput. successful (interrupted by [[solout]]),
                                                        !! * `idid=-1` input is not consistent,
                                                        !! * `idid=-2` larger `nmax` is needed,
                                                        !! * `idid=-3` step size becomes too small.
                                                        !! * `idid=-4` problem is probably stiff (interrupted).

      real(wp) :: beta,fac1,fac2,hmax,safe
      integer  :: i,ieco,iprint,istore,nrdens,nstiff,nmax
      logical  :: arret
      integer  :: itol    !! switch for `rtol` and `atol`:
                          !!
                          !! * `itol=0`: both `rtol` and `atol` are scalars.
                          !!    the code keeps, roughly, the local error of
                          !!    `y(i)` below `rtol*abs(y(i))+atol`.
                          !!
                          !! * `itol=1`: both `rtol` and `atol` are vectors.
                          !!    the code keeps the local error of `y(i)` below
                          !!    `rtol(i)*abs(y(i))+atol(i)`.

      iprint = me%iprint
      me%iout = iout
      arret = .false.

      !check procedures:
      if (.not. associated(me%fcn)) then
          if ( iprint/=0 ) &
                write (iprint,*) &
                'Error in dop853: procedure FCN is not associated.'
          idid = -1
          return
      end if

      if (iout/=0 .and. .not. associated(me%solout)) then
          if ( iprint/=0 ) &
                write (iprint,*) &
                'Error in dop853: procedure SOLOUT must be associated if IOUT/=0.'
          idid = -1
          return
      end if

      !scalar or vector tolerances:
      if (size(rtol)==1 .and. size(atol)==1) then
          itol = 0
      elseif (size(rtol)==me%n .and. size(atol)==me%n) then
          itol = 1
      else
          if ( iprint/=0 ) &
                write (iprint,*) &
                'Error in dop853: improper dimensions for rtol and/or atol.'
          idid = -1
          return
      end if

      ! setting the parameters
      me%nfcn   = 0
      me%nstep  = 0
      me%naccpt = 0
      me%nrejct = 0

      nmax = me%nmax
      nrdens = me%nrdens  !number of dense output components

      ! nstiff parameter for stiffness detection
      if ( me%nstiff<=0 ) then
          nstiff = nmax + 10  !no stiffness check
      else
          nstiff = me%nstiff
      end if

      if ( nrdens<0 .or. me%nrdens>me%n ) then
         if ( iprint/=0 ) write (iprint,*) ' curious input nrdens=' , nrdens
         arret = .true.
      else
         if ( nrdens>0 .and. iout<2 .and. iprint/=0 ) &
                write (iprint,*) ' warning: set iout=2 or iout=3 for dense output '
      end if

      if (size(y)/=me%n) then
          if ( iprint/=0 ) &
            write (iprint,*) ' error: y must have n elements: size(y)= ',size(y)
          arret = .true.
      end if

      safe = me%safe
      fac1 = me%fac1
      fac2 = me%fac2
      beta = me%beta

      if ( me%hmax==0.0_wp ) then
          hmax = xend - x
      else
          hmax = me%hmax
      end if

      me%h = me%hinitial     ! initial step size

      ! when a fail has occurred, we return with idid=-1
      if ( arret ) then

         idid = -1

      else

        ! call to core integrator
        call me%dp86co(x,y,xend,hmax,me%h,rtol,atol,itol,iprint, &
                       iout,idid,nmax,nstiff,safe,beta,fac1,fac2, &
                       me%nfcn,me%nstep,me%naccpt,me%nrejct)

      end if

      end subroutine dop853
!*****************************************************************************************

!*****************************************************************************************
!>
!  Core integrator for [[dop853]].
!  parameters same as in [[dop853]] with workspace added.

    subroutine dp86co(me,x,y,xend,hmax,h,rtol,atol,itol,iprint, &
                        iout,idid,nmax,nstiff,safe, &
                        beta,fac1,fac2, &
                        nfcn,nstep,naccpt,nrejct)

    implicit none

    class(dop853_class),intent(inout)   :: me
    real(wp),intent(inout)              :: x
    real(wp),dimension(:),intent(inout) :: y
    real(wp),intent(in)                 :: xend
    real(wp),intent(inout)              :: hmax
    real(wp),intent(inout)              :: h
    real(wp),dimension(:),intent(in)    :: rtol
    real(wp),dimension(:),intent(in)    :: atol
    integer,intent(in)                  :: itol
    integer,intent(in)                  :: iprint
    integer,intent(in)                  :: iout
    integer,intent(out)                 :: idid
    integer,intent(in)                  :: nmax
    integer,intent(in)                  :: nstiff
    real(wp),intent(in)                 :: safe
    real(wp),intent(in)                 :: beta
    real(wp),intent(in)                 :: fac1
    real(wp),intent(in)                 :: fac2
    integer,intent(inout)               :: nfcn
    integer,intent(inout)               :: nstep
    integer,intent(inout)               :: naccpt
    integer,intent(inout)               :: nrejct

    real(wp),dimension(me%n) :: y1,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10
    real(wp) :: atoli,bspl,deno,err,err2,erri,expo1,fac,fac11,&
                facc1,facc2,facold,hlamb,hnew,posneg,rtoli,&
                sk,stden,stnum,xout,xph,ydiff
    integer :: i,iasti,iord,irtrn,j,nonsti,nrd
    logical :: reject,last,event,abort

    ! initializations
    nrd = me%nrdens
    facold = 1.0e-4_wp
    expo1 = 1.0_wp/8.0_wp - beta*0.2_wp
    facc1 = 1.0_wp/fac1
    facc2 = 1.0_wp/fac2
    posneg = sign(1.0_wp,xend-x)
    irtrn = 0     ! these were not initialized
    nonsti = 0    ! in the original code
    xout = 0.0_wp !

    ! initial preparations
    atoli = atol(1)
    rtoli = rtol(1)
    last = .false.
    hlamb = 0.0_wp
    iasti = 0
    call me%fcn(x,y,k1)
    hmax = abs(hmax)
    iord = 8
    if ( h==0.0_wp ) then
        h = me%hinit(x,y,posneg,k1,iord,hmax,atol,rtol,itol)
    end if
    nfcn = nfcn + 2
    reject = .false.
    me%xold = x
    if ( iout/=0 ) then
        irtrn = 1
        me%hout = 1.0_wp
        call me%solout(naccpt+1,me%xold,x,y,irtrn,xout)
        abort = ( irtrn<0 )
    else
        abort = .false.
    end if

    if (.not. abort) then
        do
            ! basic integration step
            if ( nstep>nmax ) then
                if ( iprint/=0 ) &
                        write (iprint,'(A,E18.4)') ' exit of dop853 at x=', x
                if ( iprint/=0 ) &
                        write (iprint,*) ' more than nmax =' , nmax , 'steps are needed'
                idid = -2
                return
            elseif ( 0.1_wp*abs(h)<=abs(x)*uround ) then
                if ( iprint/=0 ) &
                        write (iprint,'(A,E18.4)') ' exit of dop853 at x=', x
                if ( iprint/=0 ) &
                        write (iprint,*) ' step size too small, h=' , h
                idid = -3
                return
            else
                if ( (x+1.01_wp*h-xend)*posneg>0.0_wp ) then
                    h = xend - x
                    last = .true.
                end if
                nstep = nstep + 1
                ! the twelve stages
                if ( irtrn>=2 ) call me%fcn(x,y,k1)
                y1 = y + h*a21*k1
                call me%fcn(x+c2*h,y1,k2)
                y1 = y + h*(a31*k1+a32*k2)
                call me%fcn(x+c3*h,y1,k3)
                y1 = y + h*(a41*k1+a43*k3)
                call me%fcn(x+c4*h,y1,k4)
                y1 = y + h*(a51*k1+a53*k3+a54*k4)
                call me%fcn(x+c5*h,y1,k5)
                y1 = y + h*(a61*k1+a64*k4+a65*k5)
                call me%fcn(x+c6*h,y1,k6)
                y1 = y + h*(a71*k1+a74*k4+a75*k5+a76*k6)
                call me%fcn(x+c7*h,y1,k7)
                y1 = y + h*(a81*k1+a84*k4+a85*k5+a86*k6+a87*k7)
                call me%fcn(x+c8*h,y1,k8)
                y1 = y + h*(a91*k1+a94*k4+a95*k5+a96*k6+a97*k7+a98*k8)
                call me%fcn(x+c9*h,y1,k9)
                y1 = y + h*(a101*k1+a104*k4+a105*k5+a106*k6+a107*k7+&
                            a108*k8+a109*k9)
                call me%fcn(x+c10*h,y1,k10)
                y1 = y + h*(a111*k1+a114*k4+a115*k5+a116*k6+a117*k7+&
                            a118*k8+a119*k9+a1110*k10)
                call me%fcn(x+c11*h,y1,k2)
                xph = x + h
                y1 = y + h*(a121*k1+a124*k4+a125*k5+a126*k6+a127*k7+&
                            a128*k8+a129*k9+a1210*k10+a1211*k2)
                call me%fcn(xph,y1,k3)
                nfcn = nfcn + 11
                k4 = b1*k1+b6*k6+b7*k7+b8*k8+b9*k9+b10*k10+b11*k2+b12*k3
                k5 = y + h*k4
                ! error estimation
                err = 0.0_wp
                err2 = 0.0_wp
                if ( itol==0 ) then
                    do i = 1 , me%n
                        sk = atoli + rtoli*max(abs(y(i)),abs(k5(i)))
                        erri = k4(i) - bhh1*k1(i) - bhh2*k9(i) - bhh3*k3(i)
                        err2 = err2 + (erri/sk)**2
                        erri = er1*k1(i) + er6*k6(i) + er7*k7(i) + er8*k8(i) &
                                + er9*k9(i) + er10*k10(i) + er11*k2(i) &
                                + er12*k3(i)
                        err = err + (erri/sk)**2
                    end do
                else
                    do i = 1 , me%n
                        sk = atol(i) + rtol(i)*max(abs(y(i)),abs(k5(i)))
                        erri = k4(i) - bhh1*k1(i) - bhh2*k9(i) - bhh3*k3(i)
                        err2 = err2 + (erri/sk)**2
                        erri = er1*k1(i) + er6*k6(i) + er7*k7(i) + er8*k8(i) &
                                + er9*k9(i) + er10*k10(i) + er11*k2(i) &
                                + er12*k3(i)
                        err = err + (erri/sk)**2
                    end do
                end if
                deno = err + 0.01_wp*err2
                if ( deno<=0.0_wp ) deno = 1.0_wp
                err = abs(h)*err*sqrt(1.0_wp/(me%n*deno))
                ! computation of hnew
                fac11 = err**expo1
                ! lund-stabilization
                fac = fac11/facold**beta
                ! we require  fac1 <= hnew/h <= fac2
                fac = max(facc2,min(facc1,fac/safe))
                hnew = h/fac
                if ( err<=1.0_wp ) then
                    ! step is accepted
                    facold = max(err,1.0e-4_wp)
                    naccpt = naccpt + 1
                    call me%fcn(xph,k5,k4)
                    nfcn = nfcn + 1
                    ! stiffness detection
                    if ( mod(naccpt,nstiff)==0 .or. iasti>0 ) then
                        stnum = 0.0_wp
                        stden = 0.0_wp
                        do i = 1 , me%n
                            stnum = stnum + (k4(i)-k3(i))**2
                            stden = stden + (k5(i)-y1(i))**2
                        end do
                        if ( stden>0.0_wp ) hlamb = abs(h)*sqrt(stnum/stden)
                        if ( hlamb>6.1_wp ) then
                            nonsti = 0
                            iasti = iasti + 1
                            if ( iasti==15 ) then
                                if ( iprint/=0 ) &
                                        write (iprint,*) &
                                        ' the problem seems to become stiff at x = ', x
                                if ( iprint==0 ) then
                                    idid = -4      ! fail exit
                                    return
                                end if
                            end if
                        else
                            nonsti = nonsti + 1
                            if ( nonsti==6 ) iasti = 0
                        end if
                    end if
                    ! final preparation for dense output
                    event = (iout==3) .and. (xout<=xph)
                    if ( iout==2 .or. event ) then
                        ! save the first function evaluations
                        do j = 1 , nrd
                            i = me%icomp(j)
                            me%cont(j) = y(i)
                            ydiff = k5(i) - y(i)
                            me%cont(j+nrd) = ydiff
                            bspl = h*k1(i) - ydiff
                            me%cont(j+nrd*2) = bspl
                            me%cont(j+nrd*3) = ydiff - h*k4(i) - bspl
                            me%cont(j+nrd*4) = d41*k1(i)+d46*k6(i)+d47*k7(i)+&
                                               d48*k8(i)+d49*k9(i)+d410*k10(i)+&
                                               d411*k2(i)+d412*k3(i)
                            me%cont(j+nrd*5) = d51*k1(i)+d56*k6(i)+d57*k7(i)+&
                                               d58*k8(i)+d59*k9(i)+d510*k10(i)+&
                                               d511*k2(i)+d512*k3(i)
                            me%cont(j+nrd*6) = d61*k1(i)+d66*k6(i)+d67*k7(i)+&
                                               d68*k8(i)+d69*k9(i)+d610*k10(i)+&
                                               d611*k2(i)+d612*k3(i)
                            me%cont(j+nrd*7) = d71*k1(i)+d76*k6(i)+d77*k7(i)+&
                                               d78*k8(i)+d79*k9(i)+d710*k10(i)+&
                                               d711*k2(i)+d712*k3(i)
                        end do
                        ! the next three function evaluations
                        y1 = y + h*(a141*k1+a147*k7+a148*k8+a149*k9+&
                                    a1410*k10+a1411*k2+a1412*k3+a1413*k4)
                        call me%fcn(x+c14*h,y1,k10)
                        y1 = y + h*(a151*k1+a156*k6+a157*k7+a158*k8+&
                                    a1511*k2+a1512*k3+a1513*k4+a1514*k10)
                        call me%fcn(x+c15*h,y1,k2)
                        y1 = y + h*(a161*k1+a166*k6+a167*k7+a168*k8+a169*k9+&
                                    a1613*k4+a1614*k10+a1615*k2)
                        call me%fcn(x+c16*h,y1,k3)
                        nfcn = nfcn + 3
                        ! final preparation
                        do j = 1 , nrd
                            i = me%icomp(j)
                            me%cont(j+nrd*4) = h*(me%cont(j+nrd*4)+d413*k4(i)+&
                                               d414*k10(i)+d415*k2(i)+d416*k3(i))
                            me%cont(j+nrd*5) = h*(me%cont(j+nrd*5)+d513*k4(i)+&
                                               d514*k10(i)+d515*k2(i)+d516*k3(i))
                            me%cont(j+nrd*6) = h*(me%cont(j+nrd*6)+d613*k4(i)+&
                                               d614*k10(i)+d615*k2(i)+d616*k3(i))
                            me%cont(j+nrd*7) = h*(me%cont(j+nrd*7)+d713*k4(i)+&
                                               d714*k10(i)+d715*k2(i)+d716*k3(i))
                        end do
                        me%hout = h
                    end if
                    k1 = k4
                    y = k5
                    me%xold = x
                    x = xph
                    if ( iout==1 .or. iout==2 .or. event ) then
                        call me%solout(naccpt+1,me%xold,x,y,irtrn,xout)
                        if ( irtrn<0 ) exit !abort
                    end if
                    ! normal exit
                    if ( last ) then
                        h = hnew
                        idid = 1
                        return
                    end if
                    if ( abs(hnew)>hmax ) hnew = posneg*hmax
                    if ( reject ) hnew = posneg*min(abs(hnew),abs(h))
                    reject = .false.
                else
                    ! step is rejected
                    hnew = h/min(facc1,fac11/safe)
                    reject = .true.
                    if ( naccpt>=1 ) nrejct = nrejct + 1
                    last = .false.
                end if
                h = hnew
            end if
        end do
    end if

    if ( iprint/=0 ) write (iprint,'(A,E18.4)') ' exit of dop853 at x=', x
    idid = 2

    end subroutine dp86co
!*****************************************************************************************

!*****************************************************************************************
!>
!  computation of an initial step size guess

    function hinit(me,x,y,posneg,f0,iord,hmax,atol,rtol,itol)

    implicit none

    class(dop853_class),intent(inout) :: me
    real(wp),intent(in)               :: x
    real(wp),dimension(:),intent(in)  :: y       !! dimension(n)
    real(wp),intent(in)               :: posneg
    real(wp),dimension(:),intent(in)  :: f0      !! dimension(n)
    integer,intent(in)                :: iord
    real(wp),intent(in)               :: hmax
    real(wp),dimension(:),intent(in)  :: atol
    real(wp),dimension(:),intent(in)  :: rtol
    integer,intent(in)                :: itol

    real(wp) :: atoli,der12,der2,dnf,dny,h,h1,hinit,rtoli,sk
    integer :: i
    real(wp),dimension(me%n)  :: f1,y1

    ! compute a first guess for explicit euler as
    !   h = 0.01 * norm (y0) / norm (f0)
    ! the increment for explicit euler is small
    ! compared to the solution
    dnf = 0.0_wp
    dny = 0.0_wp
    atoli = atol(1)
    rtoli = rtol(1)
    if ( itol==0 ) then
        do i = 1 , me%n
            sk = atoli + rtoli*abs(y(i))
            dnf = dnf + (f0(i)/sk)**2
            dny = dny + (y(i)/sk)**2
        end do
    else
        do i = 1 , me%n
            sk = atol(i) + rtol(i)*abs(y(i))
            dnf = dnf + (f0(i)/sk)**2
            dny = dny + (y(i)/sk)**2
        end do
    end if
    if ( dnf<=1.0e-10_wp .or. dny<=1.0e-10_wp ) then
        h = 1.0e-6_wp
    else
        h = sqrt(dny/dnf)*0.01_wp
    end if
    h = min(h,hmax)
    h = sign(h,posneg)
    ! perform an explicit euler step
    do i = 1 , me%n
        y1(i) = y(i) + h*f0(i)
    end do
    call me%fcn(x+h,y1,f1)
    ! estimate the second derivative of the solution
    der2 = 0.0_wp
    if ( itol==0 ) then
        do i = 1 , me%n
            sk = atoli + rtoli*abs(y(i))
            der2 = der2 + ((f1(i)-f0(i))/sk)**2
        end do
    else
        do i = 1 , me%n
            sk = atol(i) + rtol(i)*abs(y(i))
            der2 = der2 + ((f1(i)-f0(i))/sk)**2
        end do
    end if
    der2 = sqrt(der2)/h
    ! step size is computed such that
    !  h**iord * max ( norm (f0), norm (der2)) = 0.01
    der12 = max(abs(der2),sqrt(dnf))
    if ( der12<=1.0e-15_wp ) then
        h1 = max(1.0e-6_wp,abs(h)*1.0e-3_wp)
    else
        h1 = (0.01_wp/der12)**(1.0_wp/iord)
    end if

    h = min(100.0_wp*abs(h),h1,hmax)
    hinit = sign(h,posneg)

    end function hinit
!*****************************************************************************************

!*****************************************************************************************
!>
!  this function can be used for continuous output in connection
!  with the output-subroutine for [[dop853]]. it provides an
!  approximation to the `ii`-th component of the solution at `x`.

    function contd8(me,ii,x) result(y)

    implicit none

    class(dop853_class),intent(in) :: me
    integer,intent(in)             :: ii
    real(wp),intent(in)            :: x
    real(wp)                       :: y

    real(wp) :: conpar, s, s1
    integer :: i,j,nd,ierr

    ! compute place of ii-th component
    do j = 1, me%nrdens
        if ( me%icomp(j)==ii ) i = j
    end do
    if ( i==0 ) then
        !always report this message, since it is an invalid use of the code.
        if (me%iprint==0) then
            ierr = error_unit
        else
            ierr = me%iprint
        end if
        write (ierr,*) &
            ' Error in contd8: no dense output available for component:', ii
        y = 0.0_wp
    else
        nd = me%nrdens
        s = (x-me%xold)/me%hout
        s1 = 1.0_wp - s
        conpar = me%cont(i+nd*4) + &
                 s*(me%cont(i+nd*5)+ &
                 s1*(me%cont(i+nd*6)+s*me%cont(i+nd*7)))
        y = me%cont(i) + &
            s*(me%cont(i+nd)+ &
            s1*(me%cont(i+nd*2)+&
            s*(me%cont(i+nd*3)+s1*conpar)))
    end if

    end function contd8
!*****************************************************************************************

!*****************************************************************************************
    end module dop853_module
!*****************************************************************************************