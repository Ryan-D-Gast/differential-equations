//! Adaptive Runge-Kutta method with high order dense output via embedded stages.

use crate::adaptive_dense_runge_kutta_method;

adaptive_dense_runge_kutta_method!(
    /// Verner's 6(5) method is 6th order method with a embedded 5th order for
    /// error estimation and 5th order interpolation via dense output.
    ///
    /// This is an efficient 9-stage method with embedded 5th order error estimation
    /// and continuous 5th order interpolation requiring one additional stage.
    ///
    /// The method has excellent stability properties and high-quality dense output
    /// that makes it suitable for problems requiring accurate solutions at
    /// intermediate points between steps.
    ///
    /// Source: [Verner's Website](https://www.sfu.ca/~jverner/RKV65.IIIXb.Efficient.00000144617.081204.RATOnWeb)
    name: RKV65,
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.6e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.923_996_296_296_296_2e-2, 7.669_337_037_037_037e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.35975e-1, 0.0, 0.107925, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.318_683_415_233_148_4, 0.0, -5.042_058_063_628_562, 4.220_674_648_395_414, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-41.872_591_664_327_516, 0.0, 159.432_562_163_137_5, -122.119_213_565_010_03, 5.531_743_066_200_054, 0.0, 0.0, 0.0, 0.0],
        [-54.430_156_935_316_504, 0.0, 207.067_251_365_018_48, -158.610_813_784_59, 6.991_816_585_950_242, -1.859_723_106_220_323_4e-2, 0.0, 0.0, 0.0],
        [-54.663_741_787_281_98, 0.0, 207.952_806_255_389_36, -159.288_957_474_499_5, 7.018_743_740_796_944, -1.833_878_590_504_572_2e-2, -5.119_484_997_882_099e-4, 0.0, 0.0],
        [3.438_957_868_357_036e-2, 0.0, 0.0, 0.258_262_455_563_350_3, 0.420_937_118_967_353_7, 4.405_396_469_669_31, -176.483_119_024_298_65, 172.364_133_401_415_07, 0.0]
    ],
    b: [
        [3.438_957_868_357_036e-2, 0.0, 0.0, 0.258_262_455_563_350_3, 0.420_937_118_967_353_7, 4.405_396_469_669_31, -176.483_119_024_298_65, 172.364_133_401_415_07, 0.0],
        [4.909_967_648_382_49e-2, 0.0, 0.0, 0.225_111_222_951_652_42, 0.469_468_225_302_956_2, 0.806_579_224_998_886_8, 0.0, -0.607_119_489_177_796, 5.686_113_944_047_569_6e-2]
    ],
    c: [
        0.0,
        0.6e-1,
        9.593_333_333_333_333e-2,
        0.1439,
        0.4973,
        0.9725,
        0.9995,
        1.0,
        1.0
    ],
    order: 6,
    stages: 9,
    dense_stages: 10,
    extra_stages: 1,
    a_dense: [
        [1.652_415_901_357_280_6e-2, 0.0, 0.0, 0.305_312_818_751_417_9, 0.207_120_093_820_197_9, -1.293_879_140_655_123, 57.119_884_115_881_49, -55.879_792_075_109_32, 2.483_002_829_776_601_4e-2, 0.0]
    ],
    c_dense: [0.5],
    b_dense: [
        [1.0, -5.308_169_607_103_577, 10.181_680_448_958_68, -7.520_036_991_611_715, 0.934_048_536_863_116_1, 0.746_867_191_577_065],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 6.272_050_253_212_501, -16.026_181_474_677_46, 12.844_356_324_519_618, -1.148_794_504_476_759_1, -1.683_168_143_014_549_8],
        [0.0, 6.876_491_702_846_304, -24.635_767_260_846_333, 33.210_786_483_797_17, -17.494_615_282_636_44, 2.464_041_475_806_649_6],
        [0.0, -35.544_451_710_599_6, 165.701_617_019_024_2, -385.463_539_549_114_3, 442.432_413_701_570_17, -182.720_642_991_211_2],
        [0.0, 1_918.654_856_698_011_4, -9_268.121_508_966_042, 20_858.337_028_772_55, -22_645.827_671_584_81, 8_960.474_176_055_992],
        [0.0, -1_883.069_802_132_718_2, 9_101.025_187_200_634, -20_473.188_551_959_534, 22_209.765_551_256_532, -8_782.168_250_963_5],
        [0.0, 0.119_024_796_351_236_43, -0.125_026_967_050_393_76, 1.779_956_919_394_999_1, -4.660_932_123_043_763, 2.886_977_374_347_921],
        [0.0, -8.0, 32.0, -40.0, 16.0, 0.0]
    ]
);

adaptive_dense_runge_kutta_method!(
    /// Verner's 9(8) method is 9th order method with a embedded 8th order for
    /// error estimation and 9th order interpolation via dense output.
    ///
    /// This is an efficient 16-stage method with embedded 8th order error estimation
    /// and continuous the order interpolation requiring 10 additional stages.
    ///
    /// The method has excellent stability properties and high-quality dense output
    /// that makes it suitable for problems requiring accurate solutions at
    /// intermediate points between steps.
    ///
    /// Source: [Verner's Website](https://www.sfu.ca/~jverner/RKV98.IIa.Efficient.00000034399.240407.BetterEfficientonWebRKV98.IIa.Efficient.00000034399.240407.BetterEfficientonWeb)
    name: RKV98,
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3571e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-3.833_735_636_677_017e-2, 0.137_397_637_279_444_32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.714_760_534_225_28e-2, 0.0, 0.111_442_816_026_758_42, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.674_764_429_871_505, 0.0, -9.982_382_134_885_293, 7.921_017_705_013_789, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [5.242_104_050_577_351e-2, 0.0, 0.0, 0.179_691_118_917_595_32, 6.237_879_371_938_568e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.159_249_222_364_763_22, 0.0, 0.0, -0.429_842_987_724_108_7, 6.665_266_542_726_088e-2, 0.757_805_152_571_522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [7.283_333_333_333_333e-2, 0.0, 0.0, 0.0, 0.0, 0.335_934_459_066_510_37, 0.246_732_207_600_156_3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.729755859375e-1, 0.0, 0.0, 0.0, 0.0, 0.334_800_972_969_933_33, 0.118_415_823_905_066_65, -0.345673828125e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.911_213_663_452_096_4e-2, 0.0, 0.0, 0.0, 0.0, 3.983_857_361_308_652e-2, 0.106_967_528_893_935_49, -2.174_259_165_458_647_7e-2, -0.105_595_647_486_956_49, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-2.707_988_818_641_280_5e-2, 0.0, 0.0, 0.0, 0.0, 0.333e-1, -0.164_552_607_003_605_72, 3.428_266_306_497_39e-2, 0.158_526_406_443_922_1, 0.218_523_425_681_122_5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [5.584_657_769_108_862_5e-2, 0.0, 0.0, 0.0, 0.0, 9.166_533_166_672_539e-2, 0.239_239_965_552_362_7, 1.023_834_712_248_415e-2, -2.679_331_322_859_542_6e-3, 4.235_624_181_474_284_5e-2, 0.225_397_047_016_660_4, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.480_251_051_272_519_6, 0.0, 0.0, 0.0, 0.0, -6.359_610_162_555_930_5, -0.276_231_389_804_084_1, -6.500_796_633_979_847, 0.573_476_587_704_095_7, 1.347_125_994_868_138_9, 5.936_840_409_706_221, 6.590_346_245_333_925, 0.0, 0.0, 0.0, 0.0],
        [0.330_753_306_767_140_1, 0.0, 0.0, 0.0, 0.0, 5.956_207_776_829_962, -0.486_831_640_048_152_77, 4.462_055_288_206_771, 0.741_025_823_144_207_2, -0.711_819_203_457_591_3, -5.454_619_594_516_665, -4.140_803_729_244_71, 0.203_831_972_319_038_66, 0.0, 0.0, 0.0],
        [-0.584_711_112_299_894_5, 0.0, 0.0, 0.0, 0.0, -12.412_684_171_162_67, 1.360_245_445_660_928, -22.426_105_311_118_683, -0.882_885_705_586_545_8, 1.770_155_128_538_230_4, 12.158_096_519_185_339, 22.230_375_204_077_607, -0.663_448_376_020_124_9, 0.450_962_378_725_813_74, 0.0, 0.0],
        [1.940_575_549_810_648_7, 0.0, 0.0, 0.0, 0.0, 21.977_984_081_145_564, 0.823_074_732_698_472_9, 68.164_416_836_263_54, -3.117_097_463_620_267, -4.568_841_021_822_44, -18.741_909_871_262_65, -66.577_118_396_378_32, 1.098_915_553_165_441_8, 0.0, 0.0, 0.0]
    ],
    b: [
        [1.500_669_014_979_724_7e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.055_180_992_746_381_3, 0.238_494_726_378_218_3, 0.128_815_177_428_299_15, 0.227_662_311_104_621_57, 1.229_532_587_437_517_4, 4.624_976_662_810_384e-2, 0.138_619_631_936_629_38, 3.080_010_168_319_435_5e-2, 0.0],
        [1.897_210_532_481_101_4e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.408_110_314_549_493_8, 0.126_032_388_382_092_1, 0.118_837_506_345_114_97, 0.249_104_199_783_868_75, -3.269_966_219_928_978_3, 0.302_379_810_022_888_3, 0.0, 0.0, 4.652_989_552_070_924e-2]
    ],
    c: [
        0.0,
        0.3571e-1,
        9.906_028_091_267_415e-2,
        0.148_590_421_369_011_2,
        0.6134,
        0.232_735_947_360_562_7,
        0.553_864_052_639_437_3,
        0.6555,
        0.491625,
        0.6858e-1,
        0.253,
        0.662_064_179_541_204_6,
        0.8309,
        0.8998,
        1.0,
        1.0
    ],
    order: 9,
    stages: 16,
    dense_stages: 26,
    extra_stages: 10,
    a_dense: [
        [1.500_669_014_979_724_7e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.055_180_992_746_381_3, 0.238_494_726_378_218_3, 0.128_815_177_428_299_15, 0.227_662_311_104_621_57, 1.229_532_587_437_517_4, 4.624_976_662_810_384e-2, 0.138_619_631_936_629_38, 3.080_010_168_319_435_5e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.571_801_061_417_788e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.485_340_345_265_736_33, 0.210_778_756_890_454_67, 0.126_980_241_305_335_42, 0.231_968_701_451_391_92, -0.362_021_471_406_909_66, 5.366_106_712_036_344e-2, -2.806_066_613_385_549_8e-2, -2.378_121_372_710_33e-2, 0.0, 0.026_918_042_619_289_89, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.569_705_832_522_204_4e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.461_607_524_220_211_2, 0.211_394_651_669_811_33, 0.127_033_091_716_710_93, 0.231_854_055_029_870_83, -0.338_526_640_668_837_3, 5.298_251_972_194_236e-2, -2.750_461_365_887_187_8e-2, -2.361_906_185_395_527e-2, 0.0, 0.026_684_580_895_040_36, 0.011_396_834_602_855_415, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.438_964_884_291_216_3e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.206_901_219_123_788_4, 0.250_562_855_463_937_64, 0.130_333_291_570_212_73, 0.224_671_775_926_352_2, 1.308_419_325_781_946_5, 2.589_750_180_376_236e-3, 8.070_743_254_562_857e-3, -1.267_568_255_392_829_4e-2, 0.0, 0.011_291_580_723_733_216, 0.034_220_566_807_097_5, -0.114_972_636_873_414_2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.452_348_029_801_042e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.521_424_310_246_581_9, 0.186_669_884_420_460_4, 0.129_931_635_445_127_3, 0.226_214_108_576_571_93, 0.610_458_263_946_671, 1.418_715_607_022_412_5e-2, 1.480_061_054_412_245_8e-2, -3.711_471_609_871_774_7e-3, 0.0, 0.001_393_256_979_572_559, 1.114_731_020_624_073_3, -1.021_208_555_757_145_8, -0.279_565_079_291_234_1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [8.711_816_186_418_633e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.532_993_247_326_560_3e-2, -1.943_250_606_288_015_1e-3, 1.072_095_950_570_478_4e-3, 2.601_233_036_074_381e-4, -1.817_718_521_410_219_3e-2, -9.877_668_338_996_713e-4, -3.332_383_192_417_756_6e-3, -8.605_833_352_714_281e-4, 0.0, 0.001_392_810_143_886_650_6, -0.161_741_998_789_763, 0.148_515_656_130_604_4, 0.022_890_510_952_530_62, -0.001_429_777_169_141_779_3, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.710_092_628_714_179_4e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.692_310_498_600_388_5, -6.725_243_132_164_495e-2, 8.808_440_659_269_46e-2, 5.982_566_312_199_631e-2, -0.825_129_031_481_552_4, -4.886_453_387_508_305_6e-2, -0.168_260_837_156_651_2, -4.443_170_503_743_608_4e-2, 0.0, 0.070_378_446_394_327_8, -8.818_686_397_504_859, 8.028_821_919_603_92, 1.306_591_406_499_158, -0.162_488_330_722_401_46, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.466_442_426_857_961_2e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.189_839_956_139_912_45, 3.711_530_651_907_537_4e-2, 0.131_624_798_119_951_04, 0.146_662_292_882_268_56, -0.237_733_116_246_783_84, -2.493_071_232_192_949_8e-2, -9.427_777_049_221_307e-2, -2.726_667_627_690_941_2e-2, 0.0, 0.040_976_594_040_633_62, -5.538_725_321_977_495, 4.973_442_175_247_713, 0.862_629_444_918_985_9, -0.225_021_394_821_789_32, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.191_252_689_920_92e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.672_313_785_813_087_6, 0.180_128_426_682_568_9, 0.136_568_944_744_934_72, 0.211_103_839_379_890_66, 0.777_585_472_624_472_9, 2.362_117_948_505_939_8e-2, 6.391_325_607_581_23e-2, 1.197_654_289_246_164_4e-2, 0.0, -0.021_886_221_450_870_318, 3.500_218_060_218_659, -3.195_765_244_251_373_5, -0.599_440_104_860_014_9, 0.011_377_107_372_277_293, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.184_012_014_074_604_2e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.664_126_328_976_879_1, 0.178_891_406_187_319_4, 0.136_756_264_228_708_5, 0.210_775_811_973_028_6, 0.767_918_474_480_633_6, 2.313_496_869_580_297_3e-2, 6.230_605_705_999_707e-2, 1.157_099_029_193_866_4e-2, 0.0, -0.021_294_416_010_421_844, 3.163_166_251_075_345_3, -2.819_544_872_276_817, -0.397_876_904_630_975_86, 0.130_482_177_761_573_7, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    c_dense: [1.0, 0.737_501_813_998_881, 0.749, 0.65, 0.487, 0.97e-2, 0.138, 0.249, 0.439, 0.794],
    b_dense: [
        [1.0, -60.671_564_990_962_76, 669.417_333_989_096_5, -3_377.878_946_225_199_3, 9_286.468_967_391_047, -14_780.477_136_811_021, 13_604.993_863_282_247, -6_724.705_443_356_114, 1_381.867_933_411_056_6],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.597_216_760_566_844_7, 47.936_048_364_131_67, -565.717_041_700_233_3, 2_799.622_707_202_858_7, -7_032.161_794_919_679, 9_404.154_583_498_766, -6_374.661_804_344_304, 1_720.369_337_666_28],
        [0.0, 0.134_984_470_796_006_98, -10.834_629_145_944_527, 127.864_823_186_993_43, -632.777_936_777_218_8, 1_589.427_325_412_255_4, -2_125.551_246_305_557, 1_440.817_483_665_755, -388.842_309_780_701_1],
        [0.0, 7.290_747_606_753_266e-2, -5.851_972_901_033_216, 69.062_029_738_762_99, -341.774_442_716_209_7, 858.477_527_120_137_3, -1_148.047_443_579_864_2, 778.210_749_640_921_4, -210.020_539_601_353_93],
        [0.0, 0.128_853_484_734_578_56, -10.342_520_980_591_798, 122.057_211_066_254_17, -604.037_203_213_700_4, 1_517.235_637_581_122_8, -2_029.008_844_152_904, 1_375.375_645_376_321, -371.181_116_850_131_56],
        [0.0, 0.695_897_172_076_238_7, -55.856_705_135_748_31, 659.192_634_079_572_8, -3_262.215_084_140_846_6, 8_194.112_807_588_29, -10_958.039_044_672_965, 7_427.971_576_643_476, -2_004.632_548_946_417_3],
        [0.0, 2.617_668_058_132_623e-2, -2.101_090_775_094_604_6, 24.796_012_566_600_32, -122.710_603_910_578_77, 308.227_540_243_681_4, -412.194_645_100_066_96, 279.408_578_064_554_5, -75.405_718_003_049_09],
        [0.0, 7.845_665_161_262_075e-2, -6.297_381_611_696_196, 74.318_518_472_037_82, -367.787_774_705_65, 923.818_459_990_864, -1_235.428_287_659_739, 837.442_371_605_615_4, -226.005_743_111_108_05],
        [0.0, 1.743_239_982_411_997e-2, -1.399_224_563_421_080_7, 16.512_941_882_067_313, -81.719_311_330_651_92, 205.264_594_249_812_8, -274.501_644_179_880_54, 186.072_563_019_500_14, -50.216_551_375_567_66],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -3.120_237_034_739_356_2e-2, 2.505_323_513_858_665_6, -29.632_099_441_822_69, 147.173_574_899_191_83, -371.668_262_629_709_6, 500.803_253_762_596_74, -342.949_046_514_438_8, 93.798_458_780_671_25],
        [0.0, 1.005_189_184_600_557_6, -80.145_247_628_498_57, 904.286_287_161_854, -4_159.377_900_797_261, 9_382.948_440_638_738, -10_865.074_362_321_895, 6_171.218_731_586_135, -1_354.861_137_823_673_3],
        [0.0, -1.018_790_304_892_149, 81.229_685_434_330_86, -916.522_099_840_849_8, 4_215.658_051_870_923, -9_509.908_233_268_861, 11_012.088_661_364_269, -6_254.720_911_673_139, 1_373.193_636_418_218_8],
        [0.0, 1.486_756_383_162_592e-2, -1.185_413_256_701_197_5, 13.375_128_087_739_489, -61.520_574_820_481_606, 138.781_422_449_831_18, -160.703_267_695_199_86, 91.277_333_477_520_88, -20.039_495_806_540_547],
        [0.0, -2.611_963_106_691_749e-3, 0.208_255_752_438_779_78, -2.349_769_034_664_475_5, 10.808_056_622_683_038, -24.381_395_596_370_474, 28.232_668_855_390_138, -16.035_783_011_964_465, 3.520_578_375_594_15],
        [0.0, 61.751_365_685_088_68, -750.126_891_055_288_4, 3_898.685_586_244_842, -10_841.822_728_647_458, 17_321.575_664_625_165, -15_937.704_924_918_136, 7_853.047_625_862_507, -1_605.405_697_796_720_2],
        [0.0, -1.934_298_042_224_096_7, 147.588_643_166_560_36, -1_181.453_462_187_682, 4_124.816_268_317_402, -7_708.555_106_592_447, 8_052.349_419_898_65, -4_443.023_055_054_952, 1_010.211_590_494_692_5],
        [0.0, 0.912_642_830_786_966_6, -71.317_106_090_194_78, 695.456_144_607_544_6, -2_620.882_193_436_484, 4_957.008_141_154_351, -5_038.298_085_544_127, 2_633.333_935_059_078_6, -556.213_478_580_953_7],
        [0.0, -0.465_443_997_074_799_17, 37.090_998_104_441_54, -417.138_222_833_985_5, 1_920.758_799_061_188_3, -4_409.960_156_149_582, 5_383.383_503_749_768, -3_353.518_501_772_742, 839.849_023_837_986_2],
        [0.0, -0.117_645_170_825_521_85, 9.481_894_819_354_299, -114.915_675_829_831_94, 591.319_329_131_246_7, -1_559.765_475_086_581_5, 2_198.545_841_718_649, -1_564.562_048_273_728_4, 440.013_778_691_717]
    ]
);

/// Macro to create a Runge-Kutta solver with dense output capabilities
///
/// # Arguments
///
/// * `name`: Name of the solver struct to create
/// * `a`: Matrix of coefficients for intermediate stages
/// * `b`: 2D array where first row is higher order weights, second row is lower order weights
/// * `c`: Time offsets for each stage
/// * `order`: Order of accuracy of the method
/// * `stages`: Number of stages in the method
/// * `dense_stages`: Number of terms in the interpolation polynomial
/// * `extra_stages`: Number of additional stages for interpolation
/// * `a_dense`: Coefficients for additional stages needed for interpolation
/// * `c_dense`: Time offsets for additional interpolation stages
/// * `b_dense`: Coefficients for interpolation polynomial
///
/// # Note
///
/// This macro generates a full solver with the ability to interpolate the solution
/// at any point within a step. The interpolation capability requires additional
/// function evaluations but provides high-order continuous output.
///
#[macro_export]
macro_rules! adaptive_dense_runge_kutta_method {
    (
        $(#[$attr:meta])*
        name: $name:ident,
        a: $a:expr,
        b: $b:expr,
        c: $c:expr,
        order: $order:expr,
        stages: $stages:expr,
        // Interpolation info
        dense_stages: $dense_stages:expr,
        extra_stages: $extra_stages:expr,
        a_dense: $a_dense:expr,
        c_dense: $c_dense:expr,
        b_dense: $b_dense:expr
        $(,)? // Optional trailing comma
    ) => {
        $(#[$attr])*
        #[doc = "\n\n"]
        #[doc = "This adaptive solver with dense output was automatically generated using the `adaptive_dense_runge_kutta_method` macro."]
        pub struct $name<T: $crate::traits::Real, const R: usize, const C: usize, D: $crate::traits::CallBackData> {
            // Initial Step Size
            pub h0: T,

            // Current Step Size
            h: T,

            // Current State
            t: T,
            y: nalgebra::SMatrix<T, R, C>,
            dydt: nalgebra::SMatrix<T, R, C>,

            // Previous State
            h_prev: T,
            t_prev: T,
            y_prev: nalgebra::SMatrix<T, R, C>,

            // Stage values (fixed size array of matrices)
            k: [nalgebra::SMatrix<T, R, C>; $stages + $extra_stages], // Main stages + extra stages for interpolation

            // Constants from Butcher tableau
            a: [[T; $stages]; $stages],
            b_higher: [T; $stages],
            b_lower: [T; $stages],
            c: [T; $stages],

            // Interpolation coefficients
            a_dense: [[T; $stages + $extra_stages]; $extra_stages],  // Type inferred from a_dense
            c_dense: [T; $extra_stages],
            b_dense: [[T; $order]; $dense_stages],
            cont: [T; $dense_stages], // Interpolation polynomial coefficients

            // Settings
            pub rtol: T,
            pub atol: T,
            pub h_max: T,
            pub h_min: T,
            pub max_steps: usize,
            pub max_rejects: usize,
            pub safety_factor: T,
            pub min_scale: T,
            pub max_scale: T,

            // Iteration tracking
            reject: bool,
            n_stiff: usize,
            steps: usize, // Number of steps taken

            // Status
            status: $crate::ode::Status<T, R, C, D>,
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, D: $crate::traits::CallBackData> Default for $name<T, R, C, D> {
            fn default() -> Self {
                // Initialize k vectors with zeros
                let k: [nalgebra::SMatrix<T, R, C>; $stages + $extra_stages] = [nalgebra::SMatrix::<T, R, C>::zeros(); $stages + $extra_stages];

                // Initialize interpolation coefficient storage
                let cont: [T; $dense_stages] = [T::from_f64(0.0).unwrap(); $dense_stages];

                // Convert Butcher tableau values to type T
                let a_t: [[T; $stages]; $stages] = $a.map(|row| row.map(|x| T::from_f64(x).unwrap()));

                // Handle the 2D array for b, where first row is higher order and second row is lower order
                let b_higher: [T; $stages] = $b[0].map(|x| T::from_f64(x).unwrap());
                let b_lower: [T; $stages] = $b[1].map(|x| T::from_f64(x).unwrap());

                let c_t: [T; $stages] = $c.map(|x| T::from_f64(x).unwrap());

                // Convert interpolation coefficients
                let a_dense_t: [[T; $stages + $extra_stages]; $extra_stages] = $a_dense.map(|row| row.map(|x| T::from_f64(x).unwrap()));
                let c_dense_t: [T; $extra_stages] = $c_dense.map(|x| T::from_f64(x).unwrap());
                let b_dense_t: [[T; $order]; $stages + $extra_stages] =
                    $b_dense.map(|row| row.map(|x| T::from_f64(x).unwrap()));

                $name {
                    h0: T::from_f64(0.0).unwrap(),
                    h: T::from_f64(0.0).unwrap(),
                    t: T::from_f64(0.0).unwrap(),
                    y: nalgebra::SMatrix::<T, R, C>::zeros(),
                    dydt: nalgebra::SMatrix::<T, R, C>::zeros(),
                    h_prev: T::from_f64(0.0).unwrap(),
                    t_prev: T::from_f64(0.0).unwrap(),
                    y_prev: nalgebra::SMatrix::<T, R, C>::zeros(),
                    k,
                    a: a_t,
                    b_higher,
                    b_lower,
                    c: c_t,
                    a_dense: a_dense_t,
                    c_dense: c_dense_t,
                    b_dense: b_dense_t,
                    cont,
                    rtol: T::from_f64(1.0e-6).unwrap(),
                    atol: T::from_f64(1.0e-6).unwrap(),
                    h_max: T::infinity(),
                    h_min: T::from_f64(0.0).unwrap(),
                    max_steps: 10000,
                    max_rejects: 100,
                    safety_factor: T::from_f64(0.9).unwrap(),
                    min_scale: T::from_f64(0.2).unwrap(),
                    max_scale: T::from_f64(10.0).unwrap(),
                    reject: false,
                    n_stiff: 0,
                    steps: 0,
                    status: $crate::ode::Status::Uninitialized,
                }
            }
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, D: $crate::traits::CallBackData> $crate::ode::NumericalMethod<T, R, C, D> for $name<T, R, C, D> {
            fn init<F>(&mut self, ode: &F, t0: T, tf: T, y: &nalgebra::SMatrix<T, R, C>) -> Result<usize, $crate::ode::Error<T, R, C>>
            where
                F: $crate::ode::ODE<T, R, C, D>,
            {
                let mut evals = 0;

                // If h0 is zero, calculate initial step size using the utility function
                if self.h0 == T::zero() {
                    // Call standalone hinit function with order parameter
                    self.h0 = $crate::ode::methods::h_init(ode, t0, tf, y, $order, self.rtol, self.atol, self.h_min, self.h_max);
                    evals += 2; // hinit uses 2 evaluation
                }

                // Check bounds
                match $crate::utils::validate_step_size_parameters::<T, R, C, D>(self.h0, self.h_min, self.h_max, t0, tf) {
                    Ok(h0) => self.h = h0,
                    Err(status) => return Err(status),
                }

                // Initialize Statistics
                self.reject = false;
                self.n_stiff = 0;

                // Initialize State
                self.t = t0;
                self.y = y.clone();
                ode.diff(t0, y, &mut self.dydt);
                evals += 1; // Initial derivative evaluation

                // Initialize previous state
                self.t_prev = t0;
                self.y_prev = y.clone();

                // Initialize Status
                self.status = $crate::ode::Status::Initialized;

                Ok(evals)
            }

            fn step<F>(&mut self, ode: &F) -> Result<usize, $crate::ode::Error<T, R, C>>
            where
                F: $crate::ode::ODE<T, R, C, D>,
            {
                // Make sure step size isn't too small
                if self.h.abs() < T::default_epsilon() {
                    self.status = $crate::ode::Status::Error($crate::ode::Error::StepSize {
                        t: self.t, y: self.y
                    });
                    return Err($crate::ode::Error::StepSize {
                        t: self.t, 
                        y: self.y
                    });
                }

                // Check if max steps has been reached
                if self.steps >= self.max_steps {
                    self.status = $crate::ode::Status::Error($crate::ode::Error::MaxSteps {
                        t: self.t, 
                        y: self.y
                    });
                    return Err($crate::ode::Error::MaxSteps {
                        t: self.t, 
                        y: self.y
                    });
                }
                self.steps += 1;

                // Save k[0] as the current derivative
                self.k[0] = self.dydt;

                // Compute stages
                for i in 1..$stages {
                    let mut y_stage = self.y;

                    for j in 0..i {
                        y_stage += self.k[j] * (self.a[i][j] * self.h);
                    }

                    ode.diff(self.t + self.c[i] * self.h, &y_stage, &mut self.k[i]);
                }
                let mut evals = $stages - 1; // We already have k[0]

                // Compute higher order solution
                let mut y_high = self.y;
                for i in 0..$stages {
                    y_high += self.k[i] * (self.b_higher[i] * self.h);
                }

                // Compute lower order solution for error estimation
                let mut y_low = self.y;
                for i in 0..$stages {
                    y_low += self.k[i] * (self.b_lower[i] * self.h);
                }

                // Compute error estimate
                let err = y_high - y_low;

                // Calculate error norm using WRMS (weighted root mean square) norm
                let mut err_norm: T = T::zero();

                // Iterate through matrix elements
                for r in 0..R {
                    for c in 0..C {
                        let tol = self.atol + self.rtol * self.y[(r, c)].abs().max(y_high[(r, c)].abs());
                        err_norm = err_norm.max((err[(r, c)] / tol).abs());
                    }
                }

                // Determine if step is accepted
                if err_norm <= T::one() {
                    // Log previous state
                    self.t_prev = self.t;
                    self.y_prev = self.y;
                    self.h_prev = self.h;

                    if self.reject {
                        // Not rejected this time
                        self.n_stiff = 0;
                        self.reject = false;
                        self.status = $crate::ode::Status::Solving;
                    }

                    // Compute extra stages for dense / continious output via interpolation
                    for i in 0..$extra_stages {
                        let mut y_stage = self.y;
                        // Sum over the main stages
                        for j in 0..($stages + $extra_stages) {
                            y_stage += self.k[j] * (self.a_dense[i][j] * self.h);
                        }

                        ode.diff(self.t + self.c_dense[i] * self.h, &y_stage, &mut self.k[$stages + i]);
                    }
                    evals += $extra_stages; // Extra stages for interpolation

                    // Update state with the higher-order solution
                    self.t += self.h;
                    self.y = y_high;
                    ode.diff(self.t, &self.y, &mut self.dydt);
                    evals += 1; // Final derivative evaluation
                } else {
                    // Step rejected
                    self.reject = true;
                    self.status = $crate::ode::Status::RejectedStep;
                    self.n_stiff += 1;

                    // Check for stiffness
                    if self.n_stiff >= self.max_rejects {
                        self.status = $crate::ode::Status::Error($crate::ode::Error::Stiffness {
                            t: self.t, 
                            y: self.y
                        });
                        return Err($crate::ode::Error::Stiffness {
                            t: self.t, 
                            y: self.y
                        });
                    }
                }

                // Calculate new step size
                let order = T::from_usize($order).unwrap();
                let err_order = T::one() / order;

                // Standard step size controller formula
                let scale = self.safety_factor * err_norm.powf(-err_order);

                // Apply constraints to step size changes
                let scale = scale.max(self.min_scale).min(self.max_scale);

                // Update step size
                self.h *= scale;

                // Ensure step size is within bounds
                self.h = $crate::utils::constrain_step_size(self.h, self.h_min, self.h_max);
                Ok(evals)
            }

            fn t(&self) -> T {
                self.t
            }

            fn y(&self) -> &nalgebra::SMatrix<T, R, C> {
                &self.y
            }

            fn t_prev(&self) -> T {
                self.t_prev
            }

            fn y_prev(&self) -> &nalgebra::SMatrix<T, R, C> {
                &self.y_prev
            }

            fn h(&self) -> T {
                self.h
            }

            fn set_h(&mut self, h: T) {
                self.h = h;
            }

            fn status(&self) -> &$crate::ode::Status<T, R, C, D> {
                &self.status
            }

            fn set_status(&mut self, status: $crate::ode::Status<T, R, C, D>) {
                self.status = status;
            }
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, D: $crate::traits::CallBackData> $crate::interpolate::Interpolation<T, R, C> for $name<T, R, C, D> {
            fn interpolate(&mut self, t_interp: T) -> Result<nalgebra::SMatrix<T, R, C>, $crate::interpolate::InterpolationError<T, R, C>> {
                // Check if t is within bounds
                if t_interp < self.t_prev || t_interp > self.t {
                    return Err($crate::interpolate::InterpolationError::OutOfBounds {
                        t_interp, 
                        t_prev: self.t_prev, 
                        t_curr: self.t
                    });
                }

                // Calculate the normalized distance within the step [0, 1]
                let s = (t_interp - self.t_prev) / self.h_prev;

                // Compute the interpolation coefficients using Horner's method
                for i in 0..$dense_stages {
                    // Start with the highest-order term
                    self.cont[i] = self.b_dense[i][$order-1];

                    // Apply Horner's method for polynomial evaluation
                    for j in (0..$order-1).rev() {
                        self.cont[i] = self.cont[i] * s + self.b_dense[i][j];
                    }

                    // Multiply by s as all interpolation terms start at s^1
                    self.cont[i] *= s;
                }

                // Compute the interpolated value
                let mut y_interp = self.y_prev;
                for i in 0..($stages + $extra_stages) {
                    y_interp += self.k[i] * self.cont[i] * self.h_prev;
                }

                Ok(y_interp)
            }
        }

        impl<T: $crate::traits::Real, const R: usize, const C: usize, D: $crate::traits::CallBackData> $name<T, R, C, D> {
            /// Create a new solver with the specified initial step size
            pub fn new() -> Self {
                Self {
                    ..Default::default()
                }
            }

            /// Get the number of terms in the dense output interpolation polynomial
            pub fn dense_stages(&self) -> usize {
                $dense_stages
            }

            /// Set the relative tolerance for error control
            pub fn rtol(mut self, rtol: T) -> Self {
                self.rtol = rtol;
                self
            }

            /// Set the absolute tolerance for error control
            pub fn atol(mut self, atol: T) -> Self {
                self.atol = atol;
                self
            }

            /// Set the initial step size
            pub fn h0(mut self, h0: T) -> Self {
                self.h0 = h0;
                self
            }

            /// Set the minimum allowed step size
            pub fn h_min(mut self, h_min: T) -> Self {
                self.h_min = h_min;
                self
            }

            /// Set the maximum allowed step size
            pub fn h_max(mut self, h_max: T) -> Self {
                self.h_max = h_max;
                self
            }

            /// Set the maximum number of steps allowed
            pub fn max_steps(mut self, max_steps: usize) -> Self {
                self.max_steps = max_steps;
                self
            }

            /// Set the maximum number of consecutive rejected steps before declaring stiffness
            pub fn max_rejects(mut self, max_rejects: usize) -> Self {
                self.max_rejects = max_rejects;
                self
            }

            /// Set the safety factor for step size control (default: 0.9)
            pub fn safety_factor(mut self, safety_factor: T) -> Self {
                self.safety_factor = safety_factor;
                self
            }

            /// Set the minimum scale factor for step size changes (default: 0.2)
            pub fn min_scale(mut self, min_scale: T) -> Self {
                self.min_scale = min_scale;
                self
            }

            /// Set the maximum scale factor for step size changes (default: 10.0)
            pub fn max_scale(mut self, max_scale: T) -> Self {
                self.max_scale = max_scale;
                self
            }

            /// Get the order of the method
            pub fn order(&self) -> usize {
                $order
            }

            /// Get the number of stages in the method
            pub fn stages(&self) -> usize {
                $stages
            }
        }
    };
}
