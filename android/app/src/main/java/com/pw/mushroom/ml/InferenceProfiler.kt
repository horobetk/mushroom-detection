package com.pw.mushroom.ml

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.util.Log
import com.pw.mushroom.BuildConfig
import com.pw.mushroom.model.MushroomRegistry
import com.pw.mushroom.model.Toxicity
import java.io.File
import java.util.Locale
import kotlin.math.sqrt

/** Debug-only latency logger for the live detection pipeline. */
object InferenceProfiler {

    const val TAG = "InferMs"

    val enabled: Boolean get() = BuildConfig.DEBUG

    /** Keeps 4 FPS while profiling so a static scene never drops to the heartbeat. */
    val bypassSceneLock: Boolean get() = enabled

    /** Per-frame timestamps in ns; all but [sensorNs] use elapsedRealtimeNanos. */
    data class FrameTiming(
        val sensorNs: Long,
        val arrivedNs: Long,
        val acceptedNs: Long,
        val startNs: Long,
        val detectStartNs: Long,
        val detectEndNs: Long,
        val publishedNs: Long
    )

    private const val WARMUP = 8
    private const val ARRIVALS_KEPT = 64
    private const val MAX_SENSOR_LAG_MS = 2_000.0

    private val lock = Any()
    private val samples = ArrayList<Sample>(256)
    private var seq = 0
    private var frameCsv: File? = null
    private var responseCsv: File? = null

    // Accepted-frame time (ms, the tracker clock) -> analyzer arrival (ns).
    private val arrivals = object : LinkedHashMap<Long, Long>() {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<Long, Long>?): Boolean =
            size > ARRIVALS_KEPT
    }
    private val displayedTracks = HashSet<Int>()
    private val greenTracks = HashSet<Int>()

    fun noteFrame(acceptedMs: Long, arrivedNs: Long) {
        if (!enabled) return
        synchronized(lock) { arrivals[acceptedMs] = arrivedNs }
    }

    fun record(
        context: Context,
        timing: FrameTiming,
        nDetections: Int,
        sceneLocked: Boolean
    ) {
        if (!enabled) return
        val sample: Sample
        synchronized(lock) {
            seq += 1
            val detectMs = ms(timing.detectEndNs - timing.detectStartNs)
            if (seq <= WARMUP) {
                Log.i(TAG, "warmup #$seq  ${fmt(detectMs)} ms  (discarded)")
                return
            }
            val sensorLag = ms(timing.arrivedNs - timing.sensorNs)
            sample = Sample(
                index = seq - WARMUP,
                detectMs = detectMs,
                nDetections = nDetections,
                sceneLocked = sceneLocked,
                battC = batteryCelsius(context),
                sensorToAppMs = if (timing.sensorNs > 0 && sensorLag in 0.0..MAX_SENSOR_LAG_MS) sensorLag else Double.NaN,
                convertMs = ms(timing.acceptedNs - timing.arrivedNs),
                queueMs = ms(timing.startNs - timing.acceptedNs),
                letterboxMs = ms(timing.detectStartNs - timing.startNs),
                postMs = ms(timing.publishedNs - timing.detectEndNs),
                e2eMs = ms(timing.publishedNs - timing.arrivedNs)
            )
            samples += sample
        }
        appendFrameCsv(context, sample)

        Log.i(
            TAG,
            "n=${sample.index}  detect=${fmt(sample.detectMs)} ms  e2e=${fmt(sample.e2eMs)} ms  " +
                "det=${sample.nDetections}  locked=${sample.sceneLocked}  batt=${fmt(sample.battC.toDouble())} C"
        )

        if (sample.index % 50 == 0) {
            Log.w(TAG, summarise())
        }
    }

    /** Log first-display and first-green latencies for tracks published at [publishedNs]. */
    fun recordResponses(context: Context, tracked: List<TrackedDetection>, publishedNs: Long) {
        if (!enabled) return
        val rows = ArrayList<String>()
        synchronized(lock) {
            if (seq <= WARMUP) return
            for (t in tracked) {
                val arrived = arrivals[t.firstSeenMs] ?: continue
                val latency = ms(publishedNs - arrived)
                val tox = t.species.toxicity
                val green = tox == Toxicity.EDIBLE &&
                    t.confidence >= MushroomRegistry.SAFE_THRESHOLD && t.voteConfirmed
                val status = when {
                    tox.isDangerous && t.confidence >= MushroomRegistry.UNSAFE_THRESHOLD -> "warning"
                    green -> "green"
                    else -> "neutral"
                }
                if (displayedTracks.add(t.id)) {
                    rows += row(t, tox, "display", status, latency)
                }
                if (green && greenTracks.add(t.id)) {
                    rows += row(t, tox, "green", status, latency)
                }
            }
        }
        if (rows.isEmpty()) return
        rows.forEach { Log.i(TAG, "response $it") }
        appendResponseCsv(context, rows)
    }

    fun summarise(): String = synchronized(lock) {
        if (samples.isEmpty()) return "InferMs: no samples yet"
        val batt = samples.map { it.battC }.average()
        "InferMs SUMMARY n=${samples.size}  " +
            "detect[${stats(samples.map { it.detectMs })}]  " +
            "e2e[${stats(samples.map { it.e2eMs })}]  " +
            "batt=${fmt(batt)} C"
    }

    private fun row(t: TrackedDetection, tox: Toxicity, event: String, status: String, latencyMs: Double) =
        "${t.id},${t.species.name},${tox.name},$event,$status,${fmt(latencyMs)}"

    private fun appendFrameCsv(context: Context, s: Sample) {
        try {
            val file = frameCsv ?: File(dir(context), "inference_profile.csv").also {
                frameCsv = it
                it.writeText(
                    "index,detect_ms,n_detections,scene_locked,battery_c," +
                        "sensor_to_app_ms,convert_ms,queue_ms,letterbox_ms,post_ms,e2e_ms\n"
                )
                Log.w(TAG, "Writing samples to ${it.absolutePath}")
            }
            file.appendText(
                "${s.index},${fmt(s.detectMs)},${s.nDetections},${s.sceneLocked},${fmt(s.battC.toDouble())}," +
                    "${fmtOrEmpty(s.sensorToAppMs)},${fmt(s.convertMs)},${fmt(s.queueMs)}," +
                    "${fmt(s.letterboxMs)},${fmt(s.postMs)},${fmt(s.e2eMs)}\n"
            )
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to write CSV", t)
        }
    }

    private fun appendResponseCsv(context: Context, rows: List<String>) {
        try {
            val file = responseCsv ?: File(dir(context), "response_profile.csv").also {
                responseCsv = it
                it.writeText("track_id,species,toxicity,event,status,latency_ms\n")
            }
            file.appendText(rows.joinToString(separator = "\n", postfix = "\n"))
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to write response CSV", t)
        }
    }

    private fun dir(context: Context): File = context.getExternalFilesDir(null) ?: context.filesDir

    private fun batteryCelsius(context: Context): Float {
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            ?: Int.MIN_VALUE
        return if (tenths == Int.MIN_VALUE) Float.NaN else tenths / 10f
    }

    private fun stats(xs: List<Double>): String {
        val sorted = xs.sorted()
        val mean = xs.average()
        return "mean=${fmt(mean)} sd=${fmt(stdev(xs, mean))} " +
            "p50=${fmt(percentile(sorted, 0.50))} p95=${fmt(percentile(sorted, 0.95))} " +
            "max=${fmt(sorted.last())}"
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

    private fun ms(ns: Long): Double = ns / 1_000_000.0

    private fun fmt(x: Double): String = String.format(Locale.US, "%.2f", x)

    private fun fmtOrEmpty(x: Double): String = if (x.isNaN()) "" else fmt(x)

    private data class Sample(
        val index: Int,
        val detectMs: Double,
        val nDetections: Int,
        val sceneLocked: Boolean,
        val battC: Float,
        val sensorToAppMs: Double,
        val convertMs: Double,
        val queueMs: Double,
        val letterboxMs: Double,
        val postMs: Double,
        val e2eMs: Double
    )
}
