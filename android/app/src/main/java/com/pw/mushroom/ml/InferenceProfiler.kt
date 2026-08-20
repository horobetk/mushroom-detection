package com.pw.mushroom.ml

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.util.Log
import com.pw.mushroom.BuildConfig
import java.io.File
import java.util.Locale
import kotlin.math.sqrt

/**
 * Debug-only wall-clock logger for [MushroomDetector.detect] on a physical
 * device. Inactive in release builds.
 *
 * Protocol for the S23 Ultra H1 measurement:
 *  1. Point the camera at a printed (or on-screen) mushroom photo. A live
 *     fruiting body is not required: YOLO11m always runs the same 640×640
 *     graph, so latency does not depend on whether the subject is real.
 *  2. Keep the phone slightly moving so the scene never locks, OR leave
 *     [bypassSceneLock] on (default in debug) to force 4 FPS continuously.
 *  3. Session A — cold: phone at room temperature, collect ~150 frames
 *     (~40 s at 4 FPS).
 *  4. Session B — hot: keep inferring for 5–10 min, then collect another
 *     ~150 frames without restarting the app.
 *
 * Each sample is printed to Logcat (`InferMs`) and appended to
 * `Android/data/com.pw.mushroom/files/inference_profile.csv`. Pull with:
 * ```
 * adb pull /sdcard/Android/data/com.pw.mushroom/files/inference_profile.csv
 * ```
 */
object InferenceProfiler {

    const val TAG = "InferMs"

    /** Flip to false after the two sessions are recorded. */
    val enabled: Boolean get() = BuildConfig.DEBUG

    /**
     * When true, [MushroomViewModel] never drops to the 1.5 s heartbeat.
     * Required for the thermal-load session: a static photo would otherwise
     * lock the scene within a few seconds and the SoC would cool down.
     */
    val bypassSceneLock: Boolean get() = enabled

    private const val WARMUP = 8
    private val samples = ArrayList<Sample>(256)
    private var seq = 0
    private var csvFile: File? = null
    private var headerWritten = false

    fun record(
        context: Context,
        detectMs: Double,
        nDetections: Int,
        sceneLocked: Boolean
    ) {
        if (!enabled) return
        seq += 1
        if (seq <= WARMUP) {
            Log.i(TAG, "warmup #$seq  ${fmt(detectMs)} ms  (discarded)")
            return
        }

        val battC = batteryCelsius(context)
        val sample = Sample(seq - WARMUP, detectMs, nDetections, sceneLocked, battC)
        samples += sample
        appendCsv(context, sample)

        Log.i(
            TAG,
            "n=${sample.index}  detect=${fmt(detectMs)} ms  " +
                "det=${nDetections}  locked=$sceneLocked  batt=${fmt(battC.toDouble())} C"
        )

        if (samples.size % 50 == 0) {
            Log.w(TAG, summarise())
        }
    }

    fun summarise(): String {
        val xs = samples.map { it.detectMs }
        if (xs.isEmpty()) return "InferMs: no samples yet"
        val sorted = xs.sorted()
        val mean = xs.average()
        val p50 = percentile(sorted, 0.50)
        val p95 = percentile(sorted, 0.95)
        val sd = stdev(xs, mean)
        val batt = samples.map { it.battC }.average()
        return "InferMs SUMMARY n=${xs.size}  " +
            "mean=${fmt(mean)}  sd=${fmt(sd)}  " +
            "p50=${fmt(p50)}  p95=${fmt(p95)}  " +
            "min=${fmt(sorted.first())}  max=${fmt(sorted.last())}  " +
            "batt=${fmt(batt)} C"
    }

    private fun appendCsv(context: Context, sample: Sample) {
        try {
        val dir = context.getExternalFilesDir(null) ?: context.filesDir
        val file = csvFile ?: File(dir, "inference_profile.csv").also {
            csvFile = it
            Log.w(TAG, "Writing samples to ${it.absolutePath}")
        }
            if (!headerWritten) {
                file.writeText("index,detect_ms,n_detections,scene_locked,battery_c\n")
                headerWritten = true
            }
            file.appendText(
                "${sample.index},${fmt(sample.detectMs)},${sample.nDetections}," +
                    "${sample.sceneLocked},${fmt(sample.battC.toDouble())}\n"
            )
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to write CSV", t)
        }
    }

    private fun batteryCelsius(context: Context): Float {
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            ?: Int.MIN_VALUE
        return if (tenths == Int.MIN_VALUE) Float.NaN else tenths / 10f
    }

    private fun percentile(sorted: List<Double>, p: Double): Double {
        if (sorted.isEmpty()) return Double.NaN
        val idx = ((sorted.size - 1) * p).toInt().coerceIn(0, sorted.lastIndex)
        return sorted[idx]
    }

    private fun stdev(xs: List<Double>, mean: Double): Double {
        if (xs.size < 2) return 0.0
        val varSum = xs.sumOf { (it - mean) * (it - mean) }
        return sqrt(varSum / (xs.size - 1))
    }

    private fun fmt(x: Double): String = String.format(Locale.US, "%.2f", x)

    private data class Sample(
        val index: Int,
        val detectMs: Double,
        val nDetections: Int,
        val sceneLocked: Boolean,
        val battC: Float
    )
}
