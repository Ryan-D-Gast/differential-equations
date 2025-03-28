//! Adaptive step size Runge-Kutta methods

use crate::adaptive_runge_kutta_method;

adaptive_runge_kutta_method!(
    /// Runge-Kutta-Fehlberg 4(5) adaptive method
    /// This method uses six function evaluations to calculate a fifth-order accurate
    /// solution, with an embedded fourth-order method for error estimation.
    /// The RKF45 method is one of the most widely used adaptive step size methods due to
    /// its excellent balance of efficiency and accuracy.
    /// 
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0      |
    /// 1/4    | 1/4
    /// 3/8    | 3/32         9/32
    /// 12/13  | 1932/2197    -7200/2197  7296/2197
    /// 1      | 439/216      -8          3680/513    -845/4104
    /// 1/2    | -8/27        2           -3544/2565  1859/4104   -11/40
    /// -----------------------------------------------------------------------
    ///        | 16/135       0           6656/12825  28561/56430 -9/50       2/55    (5th order)
    ///        | 25/216       0           1408/2565   2197/4104   -1/5        0       (4th order)
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method#CITEREFFehlberg1969)
    name: RKF,
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0, 0.0],
        [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0, 0.0],
        [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0, 0.0],
        [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0, 0.0]
    ],
    b: [
        [16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0], // 5th order
        [25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0]           // 4th order
    ],
    c: [0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0],
    order: 5,
    stages: 6
);

adaptive_runge_kutta_method!(
    /// Cash-Karp 4(5) adaptive method
    /// This method uses six function evaluations to calculate a fifth-order accurate
    /// solution, with an embedded fourth-order method for error estimation.
    /// The Cash-Karp method is a variant of the Runge-Kutta-Fehlberg method that uses
    /// different coefficients to achieve a more efficient and accurate solution.
    /// 
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0      |
    /// 1/5    | 1/5
    /// 3/10   | 3/40         9/40
    /// 3/5    | 3/10         -9/10       6/5
    /// 1      | -11/54       5/2         -70/27      35/27
    /// 7/8    | 1631/55296   175/512     575/13824   44275/110592 253/4096
    /// ------------------------------------------------------------------------------------
    ///        | 37/378       0           250/621     125/594     0           512/1771  (5th order)
    ///        | 2825/27648   0           18575/48384 13525/55296 277/14336   1/4       (4th order)
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Cash%E2%80%93Karp_method)
    name: CashKarp,
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/10.0, -9.0/10.0, 6.0/5.0, 0.0, 0.0, 0.0],
        [-11.0/54.0, 5.0/2.0, -70.0/27.0, 35.0/27.0, 0.0, 0.0],
        [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0, 0.0]
    ],
    b: [
        [37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0], // 5th order
        [2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1.0/4.0] // 4th order
    ],
    c: [0.0, 1.0/5.0, 3.0/10.0, 3.0/5.0, 1.0, 7.0/8.0],
    order: 5,
    stages: 6
);

adaptive_runge_kutta_method!(
    /// Dormand-Prince 5(4) adaptive method
    /// This method uses seven function evaluations to calculate a fifth-order accurate 
    /// solution, with an embedded fourth-order method for error estimation.
    /// The DOPRI5 method is one of the most widely used adaptive step size methods due to
    /// its excellent balance of efficiency and accuracy.
    /// The Butcher Tableau is as follows:
    /// ```text
    /// 0      |
    /// 1/5    | 1/5
    /// 3/10   | 3/40         9/40
    /// 4/5    | 44/45        -56/15      32/9
    /// 8/9    | 19372/6561   -25360/2187 64448/6561   -212/729
    /// 1      | 9017/3168    -355/33     46732/5247   49/176        -5103/18656
    /// 1      | 35/384       0           500/1113     125/192       -2187/6784    11/84
    /// ----------------------------------------------------------------------------------------------
    ///        | 35/384       0           500/1113     125/192       -2187/6784    11/84       0       (5th order)
    ///        | 5179/57600   0           7571/16695   393/640       -92097/339200 187/2100    1/40    (4th order)
    /// ```
    /// 
    /// Reference: [Wikipedia](https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method)
    name: DOPRI5,
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
        [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
        [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
        [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
    ],
    b: [
        [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0], // 5th order
        [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0] // 4th order
    ],
    c: [0.0, 0.2, 0.3, 0.8, 8.0/9.0, 1.0, 1.0],
    order: 5,
    stages: 7
);

//adaptive_dense_runge_kutta_method!(
//    /// Verner 6(5) method with 5th order dense output
//    /// 
//    /// This is an efficient 9-stage method with embedded 5th order error estimation
//    /// and continuous 5th order interpolation requiring one additional stage.
//    /// 
//    /// The method has excellent stability properties and high-quality dense output
//    /// that makes it suitable for problems requiring accurate solutions at 
//    /// intermediate points between steps.
//    name: Verner65,
//    a: [
//        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
//        [0.6e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
//        [0.1923996296296296296296296296296296296296e-1, 0.7669337037037037037037037037037037037037e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
//        [0.35975e-1, 0.0, 0.107925, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
//        [1.318683415233148260919747276431735612861, 0.0, -5.042058063628562225427761634715637693344, 4.220674648395413964508014358283902080483, 0.0, 0.0, 0.0, 0.0, 0.0],
//        [-41.87259166432751461803757780644346812905, 0.0, 159.4325621631374917700365669070346830453, -122.1192135650100309202516203389242140663, 5.531743066200053768252631238332999150076, 0.0, 0.0, 0.0, 0.0],
//        [-54.43015693531650433250642051294142461271, 0.0, 207.0672513650184644273657173866509835987, -158.6108137845899991828742424365058599469, 6.991816585950242321992597280791793907096, -0.1859723106220323397765171799549294623692e-1, 0.0, 0.0, 0.0],
//        [-54.66374178728197680241215648050386959351, 0.0, 207.9528062553893734515824816699834244238, -159.2889574744995071508959805871426654216, 7.018743740796944434698170760964252490817, -0.1833878590504572306472782005141738268361e-1, -0.5119484997882099077875432497245168395840e-3, 0.0, 0.0],
//        [0.3438957868357036009278820124728322386520e-1, 0.0, 0.0, 0.2582624555633503404659558098586120858767, 0.4209371189673537150642551514069801967032, 4.405396469669310170148836816197095664891, -176.4831190242986576151740942499002125029, 172.3641334014150730294022582711902413315, 0.0]
//    ],
//    b: [
//        [0.3438957868357036009278820124728322386520e-1, 0.0, 0.0, 0.2582624555633503404659558098586120858767, 0.4209371189673537150642551514069801967032, 4.405396469669310170148836816197095664891, -176.4831190242986576151740942499002125029, 172.3641334014150730294022582711902413315, 0.0],
//        [0.4909967648382489730906854927971225836479e-1, 0.0, 0.0, 0.2251112229516524153401395320539875329485, 0.4694682253029562039431948525047387412553, 0.8065792249988867707634161808995217981443, 0.0, -0.6071194891777959797672951465256217122488, 0.5686113944047569241147603178766138153594e-1]
//    ],
//    c: [
//        0.0, 
//        0.6e-1, 
//        0.9593333333333333333333333333333333333333e-1, 
//        0.1439, 
//        0.4973, 
//        0.9725, 
//        0.9995, 
//        1.0, 
//        1.0
//    ],
//    order: 6,
//    stages: 9,
//    dense_stages: 6,
//    extra_stages: 1,
//    a_dense: [
//        [0.1652415901357280684383619367363197852645e-1, 0.0, 0.0, 0.3053128187514178931377105638345583032476, 0.2071200938201978848991082158995582390341, -1.293879140655123187129665774327355723229, 57.11988411588149149650780257779402737914, -55.87979207510932290773937033203265749155, 0.2483002829776601348057855515823731483430e-1]
//    ],
//    c_dense: [0.5],
//    b_dense: [
//        [0.0, 1.0, -5.308169607103576297743491917539437544903, 10.18168044895868030520877351032733768603, -7.520036991611714828300683961994073691563, 0.9340485368631160925057442706475838478288],
//        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
//        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
//        [0.0, 0.0, 6.272050253212501244827865529084399503479, -16.02618147467745958442607061022576892601, 12.84435632451961742214954703737612797249, -1.148794504476759027536609501260874665600],
//        [0.0, 0.0, 6.876491702846304590450466371720363234704, -24.63576726084633318864583120149461053641, 33.21078648379717088772133447477731248517, -17.49461528263643828092150992351036511970],
//        [0.0, 0.0, -35.54445171059960218765875699270358093032, 165.7016170190242105108446269288474846144, -385.4635395491142731464726480659809841649, 442.4324137015701845319394642134164121973],
//        [0.0, 0.0, 1918.654856698011449175045220651610014945, -9268.121508966042500195164712930044037430, 20858.33702877255011893787944928058522511, -22645.82767158481047968149020787687967272],
//        [0.0, 0.0, -1883.069802132718312960582779305006646556, 9101.025187200633795903395040749471730528, -20473.18855195953591834830509979123557878, 22209.76555125653413900516974418122400018],
//        [0.0, 0.0, 0.1190247963512364356614756628348873476063, -0.1250269670503937512118264468821359362429, 1.779956919394999075328101026471971070697, -4.660932123043762639666625363637083723091],
//        [0.0, 0.0, -8.0, 32.0, -40.0, 16.0]
//    ]
//);