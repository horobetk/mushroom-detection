package com.pw.mushroom.model

import androidx.compose.ui.graphics.Color

/**
 * Edibility class for a detected species. Each entry carries the Polish UI label
 * and the overlay/banner colour. Confidence thresholds are asymmetric and no
 * longer per-class: see [MushroomRegistry.UNSAFE_THRESHOLD] and
 * [MushroomRegistry.SAFE_THRESHOLD].
 *
 * Mapping aligns with thesis appendix "Wykaz 147 gatunków":
 * Jadalny / Niejadalny / Trujący / Śmiertelnie trujący.
 * [CAUTION] is retained for legacy persisted finds only.
 */
enum class Toxicity(val labelPl: String, val color: Color) {
    EDIBLE("Jadalny", Color(0xFF4CAF50)),             // Green
    CAUTION("Warunkowo jadalny", Color(0xFFFFEB3B)),  // Yellow (legacy)
    INEDIBLE("Niejadalny", Color(0xFF9E9E9E)),        // Gray
    POISONOUS("Trujący", Color(0xFFFF9800)),          // Orange
    DEADLY("Śmiertelnie trujący", Color(0xFFF44336)); // Red

    /** True for species that must trigger an immediate low-confidence warning. */
    val isDangerous: Boolean
        get() = this == POISONOUS || this == DEADLY

    companion object {
        /** Resolve a toxicity from its Polish label (e.g. for persisted finds). */
        fun fromLabelPl(label: String): Toxicity? =
            entries.firstOrNull { it.labelPl == label }
    }
}

/**
 * One catalogue entry mapped to a model class index.
 *
 * @property classId  Class index emitted by the TFLite model (0..146).
 * @property name     Scientific species name.
 * @property toxicity Edibility class driving colour, label, and threshold.
 */
data class MushroomSpecies(
    val classId: Int,
    val name: String,
    val toxicity: Toxicity
)

/**
 * Static catalogue of the 147 species the YOLO11m model was trained on.
 *
 * Indexed by class id so labelling never depends on a bundled asset file and
 * cannot drift out of sync at runtime. Order matches class_registry / thesis
 * appendix (Lp. 1..147 -> classId 0..146).
 */
object MushroomRegistry {

    val speciesMap: Map<Int, MushroomSpecies> = mapOf(
        // 0..104 — original train-6 set, toxicity synced to thesis appendix
        0 to MushroomSpecies(0, "Agaricus campestris", Toxicity.EDIBLE),
        1 to MushroomSpecies(1, "Amanita muscaria", Toxicity.POISONOUS),
        2 to MushroomSpecies(2, "Amanita pantherina", Toxicity.DEADLY),
        3 to MushroomSpecies(3, "Amanita phalloides", Toxicity.DEADLY),
        4 to MushroomSpecies(4, "Armillaria mellea", Toxicity.EDIBLE),
        5 to MushroomSpecies(5, "Auricularia auricula-judae", Toxicity.EDIBLE),
        6 to MushroomSpecies(6, "Boletus edulis", Toxicity.EDIBLE),
        7 to MushroomSpecies(7, "Boletus reticulatus", Toxicity.EDIBLE),
        8 to MushroomSpecies(8, "Calvatia gigantea", Toxicity.EDIBLE),
        9 to MushroomSpecies(9, "Cantharellus cibarius", Toxicity.EDIBLE),
        10 to MushroomSpecies(10, "Clavulina coralloides", Toxicity.EDIBLE),
        11 to MushroomSpecies(11, "Clitocybe nebularis", Toxicity.INEDIBLE),
        12 to MushroomSpecies(12, "Coprinus comatus", Toxicity.EDIBLE),
        13 to MushroomSpecies(13, "Craterellus cornucopioides", Toxicity.EDIBLE),
        14 to MushroomSpecies(14, "Daedaleopsis confragosa", Toxicity.INEDIBLE),
        15 to MushroomSpecies(15, "Exidia glandulosa", Toxicity.INEDIBLE),
        16 to MushroomSpecies(16, "Fistulina hepatica", Toxicity.EDIBLE),
        17 to MushroomSpecies(17, "Flammulina velutipes", Toxicity.EDIBLE),
        18 to MushroomSpecies(18, "Fomes fomentarius", Toxicity.INEDIBLE),
        19 to MushroomSpecies(19, "Ganoderma lucidum", Toxicity.INEDIBLE),
        20 to MushroomSpecies(20, "Gyromitra esculenta", Toxicity.DEADLY),
        21 to MushroomSpecies(21, "Hydnum repandum", Toxicity.EDIBLE),
        22 to MushroomSpecies(22, "Hygrophorus marzuolus", Toxicity.EDIBLE),
        23 to MushroomSpecies(23, "Imleria badia", Toxicity.EDIBLE),
        24 to MushroomSpecies(24, "Lactarius deliciosus", Toxicity.EDIBLE),
        25 to MushroomSpecies(25, "Leccinum scabrum", Toxicity.EDIBLE),
        26 to MushroomSpecies(26, "Lepista nuda", Toxicity.EDIBLE),
        27 to MushroomSpecies(27, "Macrolepiota procera", Toxicity.EDIBLE),
        28 to MushroomSpecies(28, "Morchella esculenta", Toxicity.EDIBLE),
        29 to MushroomSpecies(29, "Panellus stipticus", Toxicity.INEDIBLE),
        30 to MushroomSpecies(30, "Piptoporus betulinus", Toxicity.INEDIBLE),
        31 to MushroomSpecies(31, "Pleurotus ostreatus", Toxicity.EDIBLE),
        32 to MushroomSpecies(32, "Russula cyanoxantha", Toxicity.EDIBLE),
        33 to MushroomSpecies(33, "Russula emetica", Toxicity.POISONOUS),
        34 to MushroomSpecies(34, "Russula virescens", Toxicity.EDIBLE),
        35 to MushroomSpecies(35, "Sarcoscypha coccinea", Toxicity.EDIBLE),
        36 to MushroomSpecies(36, "Schizophyllum commune", Toxicity.INEDIBLE),
        37 to MushroomSpecies(37, "Suillus luteus", Toxicity.EDIBLE),
        38 to MushroomSpecies(38, "Trametes versicolor", Toxicity.INEDIBLE),
        39 to MushroomSpecies(39, "Tremella mesenterica", Toxicity.INEDIBLE),
        40 to MushroomSpecies(40, "Tricholoma equestre", Toxicity.EDIBLE),
        41 to MushroomSpecies(41, "Xerocomus chrysenteron", Toxicity.EDIBLE),
        42 to MushroomSpecies(42, "Cantharellus tubaeformis", Toxicity.EDIBLE),
        43 to MushroomSpecies(43, "Collybia dryophila", Toxicity.EDIBLE),
        44 to MushroomSpecies(44, "Gymnopilus junonius", Toxicity.POISONOUS),
        45 to MushroomSpecies(45, "Hygrocybe conica", Toxicity.INEDIBLE),
        46 to MushroomSpecies(46, "Laccaria laccata", Toxicity.EDIBLE),
        47 to MushroomSpecies(47, "Mycena galericulata", Toxicity.INEDIBLE),
        48 to MushroomSpecies(48, "Mycena pura", Toxicity.POISONOUS),
        49 to MushroomSpecies(49, "Hypholoma fasciculare", Toxicity.POISONOUS),
        50 to MushroomSpecies(50, "Pholiota squarrosa", Toxicity.INEDIBLE),
        51 to MushroomSpecies(51, "Stropharia aeruginosa", Toxicity.INEDIBLE),
        52 to MushroomSpecies(52, "Psilocybe semilanceata", Toxicity.POISONOUS),
        53 to MushroomSpecies(53, "Inocybe geophylla", Toxicity.POISONOUS),
        54 to MushroomSpecies(54, "Cortinarius armillatus", Toxicity.INEDIBLE),
        55 to MushroomSpecies(55, "Cortinarius violaceus", Toxicity.EDIBLE),
        56 to MushroomSpecies(56, "Paxillus involutus", Toxicity.POISONOUS),
        57 to MushroomSpecies(57, "Chroogomphus rutilus", Toxicity.EDIBLE),
        58 to MushroomSpecies(58, "Gomphidius glutinosus", Toxicity.EDIBLE),
        59 to MushroomSpecies(59, "Boletus pinophilus", Toxicity.EDIBLE),
        60 to MushroomSpecies(60, "Leccinum aurantiacum", Toxicity.EDIBLE),
        61 to MushroomSpecies(61, "Tylopilus felleus", Toxicity.INEDIBLE),
        62 to MushroomSpecies(62, "Scleroderma citrinum", Toxicity.POISONOUS),
        63 to MushroomSpecies(63, "Lycoperdon perlatum", Toxicity.EDIBLE),
        64 to MushroomSpecies(64, "Bovista plumbea", Toxicity.EDIBLE),
        65 to MushroomSpecies(65, "Cyathus striatus", Toxicity.INEDIBLE),
        66 to MushroomSpecies(66, "Crucibulum laeve", Toxicity.INEDIBLE),
        67 to MushroomSpecies(67, "Helvella crispa", Toxicity.INEDIBLE),
        68 to MushroomSpecies(68, "Verpa bohemica", Toxicity.EDIBLE),
        69 to MushroomSpecies(69, "Amanita rubescens", Toxicity.EDIBLE),
        70 to MushroomSpecies(70, "Amanita citrina", Toxicity.INEDIBLE),
        71 to MushroomSpecies(71, "Amanita fulva", Toxicity.EDIBLE),
        72 to MushroomSpecies(72, "Russula nigricans", Toxicity.INEDIBLE),
        73 to MushroomSpecies(73, "Russula xerampelina", Toxicity.EDIBLE),
        74 to MushroomSpecies(74, "Russula paludosa", Toxicity.EDIBLE),
        75 to MushroomSpecies(75, "Lactarius torminosus", Toxicity.INEDIBLE),
        76 to MushroomSpecies(76, "Lactarius turpis", Toxicity.INEDIBLE),
        77 to MushroomSpecies(77, "Lactarius piperatus", Toxicity.INEDIBLE),
        78 to MushroomSpecies(78, "Lactarius rufus", Toxicity.INEDIBLE),
        79 to MushroomSpecies(79, "Suillus granulatus", Toxicity.EDIBLE),
        80 to MushroomSpecies(80, "Suillus grevillei", Toxicity.EDIBLE),
        81 to MushroomSpecies(81, "Xerocomellus pruinatus", Toxicity.EDIBLE),
        82 to MushroomSpecies(82, "Pseudoboletus parasiticus", Toxicity.EDIBLE),
        83 to MushroomSpecies(83, "Gyroporus castaneus", Toxicity.EDIBLE),
        84 to MushroomSpecies(84, "Clitopilus prunulus", Toxicity.EDIBLE),
        85 to MushroomSpecies(85, "Entoloma sinuatum", Toxicity.DEADLY),
        86 to MushroomSpecies(86, "Lepiota procera", Toxicity.EDIBLE),
        87 to MushroomSpecies(87, "Chlorophyllum rhacodes", Toxicity.EDIBLE),
        88 to MushroomSpecies(88, "Agaricus arvensis", Toxicity.EDIBLE),
        89 to MushroomSpecies(89, "Marasmius oreades", Toxicity.EDIBLE),
        90 to MushroomSpecies(90, "Volvariella gloiocephala", Toxicity.EDIBLE),
        91 to MushroomSpecies(91, "Pluteus cervinus", Toxicity.EDIBLE),
        92 to MushroomSpecies(92, "Tubaria furfuracea", Toxicity.INEDIBLE),
        93 to MushroomSpecies(93, "Galerina marginata", Toxicity.DEADLY),
        94 to MushroomSpecies(94, "Phallus impudicus", Toxicity.INEDIBLE),
        95 to MushroomSpecies(95, "Hygrophoropsis aurantiaca", Toxicity.INEDIBLE),
        96 to MushroomSpecies(96, "Amanita virosa", Toxicity.DEADLY),
        97 to MushroomSpecies(97, "Agaricus xanthodermus", Toxicity.POISONOUS),
        98 to MushroomSpecies(98, "Rubroboletus satanas", Toxicity.POISONOUS),
        99 to MushroomSpecies(99, "Omphalotus olearius", Toxicity.POISONOUS),
        100 to MushroomSpecies(100, "Caloboletus calopus", Toxicity.INEDIBLE),
        101 to MushroomSpecies(101, "Lepiota cristata", Toxicity.POISONOUS),
        102 to MushroomSpecies(102, "Tricholoma sulphureum", Toxicity.POISONOUS),
        103 to MushroomSpecies(103, "Cortinarius rubellus", Toxicity.DEADLY),
        104 to MushroomSpecies(104, "Lactarius helvus", Toxicity.POISONOUS),

        // 105..146 — fine_147classes extension set
        105 to MushroomSpecies(105, "Amanita excelsa", Toxicity.EDIBLE),
        106 to MushroomSpecies(106, "Amanita gemmata", Toxicity.POISONOUS),
        107 to MushroomSpecies(107, "Amanita porphyria", Toxicity.INEDIBLE),
        108 to MushroomSpecies(108, "Calvatia utriformis", Toxicity.EDIBLE),
        109 to MushroomSpecies(109, "Cerioporus squamosus", Toxicity.EDIBLE),
        110 to MushroomSpecies(110, "Clitocybe odora", Toxicity.EDIBLE),
        111 to MushroomSpecies(111, "Coprinellus micaceus", Toxicity.EDIBLE),
        112 to MushroomSpecies(112, "Coprinopsis atramentaria", Toxicity.INEDIBLE),
        113 to MushroomSpecies(113, "Crepidotus variabilis", Toxicity.INEDIBLE),
        114 to MushroomSpecies(114, "Geastrum triplex", Toxicity.INEDIBLE),
        115 to MushroomSpecies(115, "Grifola frondosa", Toxicity.EDIBLE),
        116 to MushroomSpecies(116, "Gymnopus dryophilus", Toxicity.EDIBLE),
        117 to MushroomSpecies(117, "Gymnopus fusipes", Toxicity.INEDIBLE),
        118 to MushroomSpecies(118, "Gymnopus peronatus", Toxicity.INEDIBLE),
        119 to MushroomSpecies(119, "Hebeloma crustuliniforme", Toxicity.POISONOUS),
        120 to MushroomSpecies(120, "Hypholoma capnoides", Toxicity.EDIBLE),
        121 to MushroomSpecies(121, "Hypholoma lateritium", Toxicity.INEDIBLE),
        122 to MushroomSpecies(122, "Kuehneromyces mutabilis", Toxicity.EDIBLE),
        123 to MushroomSpecies(123, "Lactarius blennius", Toxicity.INEDIBLE),
        124 to MushroomSpecies(124, "Lactarius quietus", Toxicity.INEDIBLE),
        125 to MushroomSpecies(125, "Lactarius vellereus", Toxicity.INEDIBLE),
        126 to MushroomSpecies(126, "Laetiporus sulphureus", Toxicity.EDIBLE),
        127 to MushroomSpecies(127, "Lentinula edodes", Toxicity.EDIBLE),
        128 to MushroomSpecies(128, "Lycoperdon pyriforme", Toxicity.EDIBLE),
        129 to MushroomSpecies(129, "Mycena epipterygia", Toxicity.INEDIBLE),
        130 to MushroomSpecies(130, "Mycena haematopus", Toxicity.INEDIBLE),
        131 to MushroomSpecies(131, "Neoboletus luridiformis", Toxicity.EDIBLE),
        132 to MushroomSpecies(132, "Panaeolus foenisecii", Toxicity.POISONOUS),
        133 to MushroomSpecies(133, "Pleurotus pulmonarius", Toxicity.EDIBLE),
        134 to MushroomSpecies(134, "Pluteus salicinus", Toxicity.INEDIBLE),
        135 to MushroomSpecies(135, "Russula foetens", Toxicity.INEDIBLE),
        136 to MushroomSpecies(136, "Russula ochroleuca", Toxicity.INEDIBLE),
        137 to MushroomSpecies(137, "Russula vesca", Toxicity.EDIBLE),
        138 to MushroomSpecies(138, "Scleroderma verrucosum", Toxicity.POISONOUS),
        139 to MushroomSpecies(139, "Stereum hirsutum", Toxicity.INEDIBLE),
        140 to MushroomSpecies(140, "Stropharia rugosoannulata", Toxicity.EDIBLE),
        141 to MushroomSpecies(141, "Suillus bovinus", Toxicity.EDIBLE),
        142 to MushroomSpecies(142, "Suillus variegatus", Toxicity.EDIBLE),
        143 to MushroomSpecies(143, "Trametes gibbosa", Toxicity.INEDIBLE),
        144 to MushroomSpecies(144, "Tricholoma scalpturatum", Toxicity.EDIBLE),
        145 to MushroomSpecies(145, "Tricholoma terreum", Toxicity.EDIBLE),
        146 to MushroomSpecies(146, "Xylaria polymorpha", Toxicity.INEDIBLE)
    )

    /** Number of catalogued species (used to sanity-check the model output). */
    val speciesCount: Int = speciesMap.size

    // Asymmetric thresholds: toxic warns early, edible needs a high bar + vote buffer.
    const val UNSAFE_THRESHOLD = 0.18f
    const val SAFE_THRESHOLD = 0.60f

    // Mid band: detection exists but app asserts neither SAFE nor UNSAFE.
    const val NEUTRAL_MIN_CONFIDENCE = 0.20f
    const val NEUTRAL_MAX_CONFIDENCE = 0.55f

    // SAFE badge: species must win 4 of the last 5 frames.
    const val VOTE_REQUIRED = 4
    const val VOTE_WINDOW = 5

    // Global decode floor (lowest bar that can still surface a warning).
    val minSafetyThreshold: Float = UNSAFE_THRESHOLD

    fun fromId(classId: Int): MushroomSpecies? = speciesMap[classId]

    // Decode / pre-tracker gate: toxic from 0.18, everything else from 0.20.
    fun thresholdFor(classId: Int): Float =
        if (speciesMap[classId]?.toxicity?.isDangerous == true) UNSAFE_THRESHOLD else NEUTRAL_MIN_CONFIDENCE

    fun displayThresholdFor(classId: Int): Float = thresholdFor(classId)

    // Badge policy:
    // - toxic/deadly @ >= 0.18 -> immediate warning colour
    // - edible @ >= 0.60 AND voteConfirmed -> green "Jadalny"
    // - otherwise -> neutral gray ("Obiekt nierozpoznany / Zbliż kamerę")
    fun displayStatus(
        toxicity: Toxicity,
        confidence: Float,
        voteConfirmed: Boolean = true
    ): DisplayStatus =
        when {
            toxicity.isDangerous && confidence >= UNSAFE_THRESHOLD ->
                DisplayStatus(toxicity.labelPl, toxicity.color)

            toxicity == Toxicity.EDIBLE &&
                confidence >= SAFE_THRESHOLD &&
                voteConfirmed ->
                DisplayStatus(toxicity.labelPl, toxicity.color)

            else ->
                DisplayStatus(NEUTRAL_LABEL, NEUTRAL_COLOR)
        }

    const val NEUTRAL_LABEL = "Obiekt nierozpoznany / Zbliż kamerę"
    val NEUTRAL_COLOR = Color(0xFF757575)
}

/**
 * Resolved badge for a detection: the Polish label and colour to render after
 * the safety policy has been applied.
 */
data class DisplayStatus(val label: String, val color: Color)
