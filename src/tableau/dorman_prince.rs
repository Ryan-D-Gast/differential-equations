//! Dormand-Prince Runge-Kutta methods with dense output interpolation.

use crate::tableau::ButcherTableau;

impl ButcherTableau<7> {
    /// Dormand-Prince 5(4) Tableau with dense output interpolation.
    ///
    /// # Overview
    /// This provides a 7-stage Runge-Kutta method with:
    /// - Primary order: 5
    /// - Embedded order: 4 (for error estimation)
    /// - Number of stages: 7 primary + 0 additional stages for interpolation
    /// - Built-in dense output of order 4
    ///
    /// # Efficiency
    /// The DOPRI5 method is popular due to its efficient balance between accuracy and 
    /// computational cost. It is particularly good for non-stiff problems.
    ///
    /// # Interpolation
    /// - The method provides a 4th-order interpolant using the existing 7 stages
    /// - The interpolant has continuous first derivatives
    /// - The interpolant uses the values of the stages to construct a polynomial
    ///   that allows evaluation at any point within the integration step
    ///
    /// # Notes
    /// - The method was developed by Dormand and Prince in 1980
    /// - It is one of the most widely used Runge-Kutta methods and is implemented
    ///   in many software packages for solving ODEs
    /// - The DOPRI5 method is a member of the Dormand-Prince family of embedded
    ///   Runge-Kutta methods
    ///
    /// # References
    /// - Dormand, J. R. & Prince, P. J. (1980), "A family of embedded Runge-Kutta formulae",
    ///   Journal of Computational and Applied Mathematics, 6(1), pp. 19-26
    /// - Hairer, E., Nørsett, S. P. & Wanner, G. (1993), "Solving Ordinary Differential Equations I: 
    ///   Nonstiff Problems", Springer Series in Computational Mathematics, Vol. 8, Springer-Verlag
    pub const fn dopri5() -> Self {
        let mut c = [0.0; 7];
        let mut a = [[0.0; 7]; 7];
        let mut b = [0.0; 7];
        let mut bh = [0.0; 7];
        let mut bi4 = [[0.0; 7]; 7];

        c[0] = 0.0;
        c[1] = 0.2;
        c[2] = 0.3;
        c[3] = 0.8;
        c[4] = 8.0 / 9.0;
        c[5] = 1.0;
        c[6] = 1.0;

        a[1][0] = 0.2;

        a[2][0] = 3.0 / 40.0;
        a[2][1] = 9.0 / 40.0;

        a[3][0] = 44.0 / 45.0;
        a[3][1] = -56.0 / 15.0;
        a[3][2] = 32.0 / 9.0;

        a[4][0] = 19372.0 / 6561.0;
        a[4][1] = -25360.0 / 2187.0;
        a[4][2] = 64448.0 / 6561.0;
        a[4][3] = -212.0 / 729.0;

        a[5][0] = 9017.0 / 3168.0;
        a[5][1] = -355.0 / 33.0;
        a[5][2] = 46732.0 / 5247.0;
        a[5][3] = 49.0 / 176.0;
        a[5][4] = -5103.0 / 18656.0;

        a[6][0] = 35.0 / 384.0;
        a[6][1] = 0.0;
        a[6][2] = 500.0 / 1113.0;
        a[6][3] = 125.0 / 192.0;
        a[6][4] = -2187.0 / 6784.0;
        a[6][5] = 11.0 / 84.0;

        b[0] = 35.0 / 384.0;
        b[1] = 0.0;
        b[2] = 500.0 / 1113.0;
        b[3] = 125.0 / 192.0;
        b[4] = -2187.0 / 6784.0;
        b[5] = 11.0 / 84.0;
        b[6] = 0.0;

        bh[0] = 71.0 / 57600.0;
        bh[1] = 0.0;
        bh[2] = -71.0 / 16695.0;
        bh[3] = 71.0 / 1920.0;
        bh[4] = -17253.0 / 339200.0;
        bh[5] = 22.0 / 525.0;
        bh[6] = 1.0 / 40.0;

        bi4[0][0] = -12715105075.0 / 11282082432.0;
        bi4[0][1] = 0.0;
        bi4[0][2] = 87487479700.0 / 32700410799.0;
        bi4[0][3] = -10690763975.0 / 1880347072.0;
        bi4[0][4] = 701980252875.0 / 199316789632.0;
        bi4[0][5] = -1453857185.0 / 822651844.0;
        bi4[0][6] = 69997945.0 / 29380423.0;

        ButcherTableau {
            c,
            a,
            b,
            bh: Some(bh),
            bi: Some(bi4),
        }
    }
}

impl ButcherTableau<12, 16> {
    /// Dormand-Prince 8(5,3) Tableau with dense output interpolation.
    ///
    /// # Overview
    /// This provides a 12-stage Runge-Kutta method with:
    /// - Primary order: 8
    /// - Embedded order: 5 (for error estimation)
    /// - Number of stages: 12 primary + 4 additional stages for interpolation
    /// - Built-in dense output of order 7
    ///
    /// # Efficiency
    /// The DOP853 method is a high-order Runge-Kutta method that provides excellent 
    /// accuracy and efficiency for non-stiff problems. It's particularly suitable when 
    /// high precision is required.
    ///
    /// # Interpolation
    /// - The method provides a 7th-order interpolant using the existing stages plus
    ///   3 additional stages for dense output
    /// - The interpolant has continuous derivatives
    /// - The interpolation is performed through a sophisticated continuous extension
    ///   that maintains high accuracy throughout the integration step
    ///
    /// # Notes
    /// - The method was developed by Dormand, Prince and others as an extension
    ///   of their earlier work
    /// - It is one of the most accurate explicit Runge-Kutta implementations
    ///   available for solving ODEs
    /// - The DOP853 method is widely used in scientific computing where high
    ///   precision is required
    ///
    /// # References
    /// - Hairer, E., Nørsett, S. P. & Wanner, G. (1993), "Solving Ordinary Differential Equations I: 
    ///   Nonstiff Problems", Springer Series in Computational Mathematics, Vol. 8, Springer-Verlag
    /// - Dormand, J. R. & Prince, P. J. (1980), "A family of embedded Runge-Kutta formulae",
    ///   Journal of Computational and Applied Mathematics, 6(1), pp. 19-26
    pub const fn dop853() -> Self {
        let mut c = [0.0; 16];
        let mut a = [[0.0; 16]; 16];
        let mut b = [0.0; 12];
        let mut bh = [0.0; 12];
        let mut bi7 = [[0.0; 16]; 16];

        c[0] = 0.0;
        c[1] = 5.260_015_195_876_773E-2;
        c[2] = 7.890_022_793_815_16E-2;
        c[3] = 1.183_503_419_072_274E-1;
        c[4] = 2.816_496_580_927_726E-1;
        c[5] = 3.333_333_333_333_333E-1;
        c[6] = 0.25E+00;
        c[7] = 3.076_923_076_923_077E-1;
        c[8] = 6.512_820_512_820_513E-1;
        c[9] = 0.6E+00;
        c[10] = 8.571_428_571_428_571E-1;
        c[11] = 1.0;

        a[1][0] = 5.260_015_195_876_773E-2;

        a[2][0] = 1.972_505_698_453_79E-2;
        a[2][1] = 5.917_517_095_361_37E-2;

        a[3][0] = 2.958_758_547_680_685E-2;
        a[3][1] = 0.0;
        a[3][2] = 8.876_275_643_042_054E-2;

        a[4][0] = 2.413_651_341_592_667E-1;
        a[4][1] = 0.0;
        a[4][2] = -8.845_494_793_282_861E-1;
        a[4][3] = 9.248_340_032_617_92E-1;

        a[5][0] = 3.703_703_703_703_703_5E-2;
        a[5][1] = 0.0;
        a[5][2] = 0.0;
        a[5][3] = 1.708_286_087_294_738_6E-1;
        a[5][4] = 1.254_676_875_668_224_2E-1;

        a[6][0] = 3.7109375E-2;
        a[6][1] = 0.0;
        a[6][2] = 0.0;
        a[6][3] = 1.702_522_110_195_440_5E-1;
        a[6][4] = 6.021_653_898_045_596E-2;
        a[6][5] = -1.7578125E-2;

        a[7][0] = 3.709_200_011_850_479E-2;
        a[7][1] = 0.0;
        a[7][2] = 0.0;
        a[7][3] = 1.703_839_257_122_399_8E-1;
        a[7][4] = 1.072_620_304_463_732_8E-1;
        a[7][5] = -1.531_943_774_862_440_2E-2;
        a[7][6] = 8.273_789_163_814_023E-3;

        a[8][0] = 6.241_109_587_160_757E-1;
        a[8][1] = 0.0;
        a[8][2] = 0.0;
        a[8][3] = -3.360_892_629_446_941_4;
        a[8][4] = -8.682_193_468_417_26E-1;
        a[8][5] = 2.759_209_969_944_671E1;
        a[8][6] = 2.015_406_755_047_789_4E1;
        a[8][7] = -4.348_988_418_106_996E1;

        a[9][0] = 4.776_625_364_382_643_4E-1;
        a[9][1] = 0.0;
        a[9][2] = 0.0;
        a[9][3] = -2.488_114_619_971_667_7;
        a[9][4] = -5.902_908_268_368_43E-1;
        a[9][5] = 2.123_005_144_818_119_3E1;
        a[9][6] = 1.527_923_363_288_242_3E1;
        a[9][7] = -3.328_821_096_898_486E1;
        a[9][8] = -2.033_120_170_850_862_7E-2;

        a[10][0] = -9.371_424_300_859_873E-1;
        a[10][1] = 0.0;
        a[10][2] = 0.0;
        a[10][3] = 5.186_372_428_844_064;
        a[10][4] = 1.091_437_348_996_729_5;
        a[10][5] = -8.149_787_010_746_927;
        a[10][6] = -1.852_006_565_999_696E1;
        a[10][7] = 2.273_948_709_935_050_5E1;
        a[10][8] = 2.493_605_552_679_652_3;
        a[10][9] = -3.046_764_471_898_219_6;

        a[11][0] = 2.273_310_147_516_538;
        a[11][1] = 0.0;
        a[11][2] = 0.0;
        a[11][3] = -1.053_449_546_673_725E1;
        a[11][4] = -2.000_872_058_224_862_5;
        a[11][5] = -1.795_893_186_311_88E1;
        a[11][6] = 2.794_888_452_941_996E1;
        a[11][7] = -2.858_998_277_135_023_5;
        a[11][8] = -8.872_856_933_530_63;
        a[11][9] = 1.236_056_717_579_430_3E1;
        a[11][10] = 6.433_927_460_157_636E-1;

        b[0] = 5.429_373_411_656_876_5E-2;
        b[1] = 0.0;
        b[2] = 0.0;
        b[3] = 0.0;
        b[4] = 0.0;
        b[5] = 4.450_312_892_752_409;
        b[6] = 1.891_517_899_314_500_3;
        b[7] = -5.801_203_960_010_585;
        b[8] = 3.111_643_669_578_199E-1;
        b[9] = -1.521_609_496_625_161E-1;
        b[10] = 2.013_654_008_040_303_4E-1;
        b[11] = 4.471_061_572_777_259E-2;

        bh[0] = 1.312_004_499_419_488E-2;
        bh[5] = -1.225_156_446_376_204_4;
        bh[6] = -4.957_589_496_572_502E-1;
        bh[7] = 1.664_377_182_454_986_4;
        bh[8] = -3.503_288_487_499_736_6E-1;
        bh[9] = 3.341_791_187_130_175E-1;
        bh[10] = 8.192_320_648_511_571E-2;
        bh[11] = -2.235_530_786_388_629_4E-2;

        c[12] = 0.1E+00;

        a[12][0] = 5.616_750_228_304_795_4E-2;
        a[12][6] = 2.535_002_102_166_248_3E-1;
        a[12][7] = -2.462_390_374_708_025E-1;
        a[12][8] = -1.241_914_232_638_163_7E-1;
        a[12][9] = 1.532_917_982_787_656_8E-1;
        a[12][10] = 8.201_052_295_634_69E-3;
        a[12][11] = 7.567_897_660_545_699E-3;
        a[12][12] = -8.298E-3;

        c[13] = 0.2E+00;
        
        a[13][0] = 3.183_464_816_350_214E-2;
        a[13][5] = 2.830_090_967_236_677_6E-2;
        a[13][6] = 5.354_198_830_743_856_6E-2;
        a[13][7] = -5.492_374_857_139_099E-2;
        a[13][10] = -1.083_473_286_972_493_2E-4;
        a[13][11] = 3.825_710_908_356_584E-4;
        a[13][12] = -3.404_650_086_874_045_6E-4;

        c[14] = 7.777_777_777_777_778E-1;
        
        a[14][0] = -4.288_963_015_837_919_4E-1;
        a[14][5] = -4.697_621_415_361_164;
        a[14][6] = 7.683_421_196_062_599;
        a[14][7] = 4.068_989_818_397_11;
        a[14][8] = 3.567_271_874_552_811E-1;
        a[14][12] = -1.399_024_165_159_014_5E-3;
        a[14][13] = 2.947_514_789_152_772_4;
        a[14][14] = -9.150_958_472_179_87;

        bi7[0][0] = -8.428_938_276_109_013;
        bi7[0][5] = 5.667_149_535_193_777E-1;
        bi7[0][6] = -3.068_949_945_949_891_7;
        bi7[0][7] = 2.384_667_656_512_07;
        bi7[0][8] = 2.117_034_582_445_028;
        bi7[0][9] = -8.713_915_837_779_73E-1;
        bi7[0][10] = 2.240_437_430_260_788_3;
        bi7[0][11] = 6.315_787_787_694_688E-1;
        bi7[0][12] = -8.899_033_645_133_331E-2;
        bi7[0][13] = 1.814_850_552_085_472_7E1;
        bi7[0][14] = -9.194_632_392_478_356;
        bi7[0][15] = -4.436_036_387_594_894;
        
        bi7[1][0] = 1.042_750_864_257_913_4E1;
        bi7[1][5] = 2.422_834_917_752_581_7E2;
        bi7[1][6] = 1.652_004_517_172_702_8E2;
        bi7[1][7] = -3.745_467_547_226_902E2;
        bi7[1][8] = -2.211_366_685_312_530_6E1;
        bi7[1][9] = 7.733_432_668_472_264;
        bi7[1][10] = -3.067_408_473_108_939_8E1;
        bi7[1][11] = -9.332_130_526_430_229;
        bi7[1][12] = 1.569_723_812_177_084_5E1;
        bi7[1][13] = -3.113_940_321_956_517_8E1;
        bi7[1][14] = -9.352_924_358_844_48;
        bi7[1][15] = 3.581_684_148_639_408E1;
        
        bi7[2][0] = 1.998_505_324_200_243_3E1;
        bi7[2][5] = -3.870_373_087_493_518E2;
        bi7[2][6] = -1.891_781_381_951_675_8E2;
        bi7[2][7] = 5.278_081_592_054_236E2;
        bi7[2][8] = -1.157_390_253_995_963E1;
        bi7[2][9] = 6.881_232_694_696_3;
        bi7[2][10] = -1.000_605_096_691_083_8;
        bi7[2][11] = 7.777_137_798_053_443E-1;
        bi7[2][12] = -2.778_205_752_353_508;
        bi7[2][13] = -6.019_669_523_126_412E1;
        bi7[2][14] = 8.432_040_550_667_716E1;
        bi7[2][15] = 1.199_229_113_618_279E1;
        
        bi7[3][0] = -2.569_393_346_270_375E1;
        bi7[3][5] = -1.541_897_486_902_364_3E2;
        bi7[3][6] = -2.315_293_791_760_455E2;
        bi7[3][7] = 3.576_391_179_106_141E2;
        bi7[3][8] = 9.340_532_418_362_432E1;
        bi7[3][9] = -3.745_832_313_645_163E1;
        bi7[3][10] = 1.040_996_495_089_623E2;
        bi7[3][11] = 2.984_029_342_666_05E1;
        bi7[3][12] = -4.353_345_659_001_114E1;
        bi7[3][13] = 9.632_455_395_918_828E1;
        bi7[3][14] = -3.917_726_167_561_544E1;
        bi7[3][15] = -1.497_268_362_579_856_4E2;

        ButcherTableau {
            c,
            a,
            b,
            bh: Some(bh),
            bi: Some(bi7),
        }
    }
}
