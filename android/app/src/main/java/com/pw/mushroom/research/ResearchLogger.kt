package com.pw.mushroom.research

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.os.BatteryManager
import android.os.Environment
import android.util.Log
import com.pw.mushroom.ml.TrackedDetection
import com.pw.mushroom.model.MushroomRegistry
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Writes Research Mode capture bundles to Documents/MushroomResearch. */
object ResearchLogger {

    private const val TAG = "ResearchLogger"
    private const val ROOT_DIR = "MushroomResearch"
    private const val CSV_NAME = "session_log.csv"
    private const val JPEG_QUALITY = 92
    private const val FULLRES_JPEG_QUALITY = 95
    private const val ANNOTATED_JPEG_QUALITY = 88
    private const val TOP_N_CSV = 3 // how many top-N predictions go into CSV columns

    private var sessionDir: File? = null
    private var sessionId: String? = null
    private var csvHeaderWritten = false

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val labelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt()
        textSize = 36f
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    /** The directory is created on first capture, so empty sessions leave no folder. */
    fun startSession(): String? {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val id = "session_$ts"

        val docsDir = Environment.getExternalStoragePublicDirectory(
            Environment.DIRECTORY_DOCUMENTS
        )
        sessionDir = File(docsDir, "$ROOT_DIR/$id")
        sessionId = id
        csvHeaderWritten = false
        Log.i(TAG, "Research session started (dir pending first capture): $id")
        return id
    }

    fun endSession() {
        Log.i(TAG, "Research session ended: $sessionId")
        sessionDir = null
        sessionId = null
        csvHeaderWritten = false
    }

    val isSessionActive: Boolean get() = sessionId != null

    /** Create the session directory and ground-truth stub on first use. */
    private fun ensureSessionDir(): File? {
        val dir = sessionDir ?: return null
        if (!dir.exists() && !dir.mkdirs()) {
            Log.e(TAG, "Failed to create session directory: ${dir.absolutePath}")
            return null
        }
        val gt = File(dir, "ground_truth.txt")
        if (!gt.exists()) {
            gt.writeText(
                "# Ground truth for ${sessionId ?: dir.name}\n" +
                    "# Write the ACTUAL species you photographed (one per line, or\n" +
                    "# 'captureId=Species' for per-capture truth). Used to verify detections.\n" +
                    "expected_species=\n"
            )
        }
        return dir
    }

    /** Write one capture bundle; bitmaps are not recycled here. Null on failure. */
    fun capture(
        captureId: Int,
        fullFrame: Bitmap,
        fullResFrame: Bitmap?,
        screenBitmap: Bitmap?,
        detections: List<TrackedDetection>,
        inferenceMs: Double,
        fps: Double,
        gpuDelegate: Boolean,
        context: Context
    ): String? {
        val dir = ensureSessionDir() ?: return null
        // e.g. capture_001__Amanita-porphyria__71pct_annotated.jpg
        val prefix = "capture_%03d__%s".format(captureId, filenameLabel(detections))
        val now = System.currentTimeMillis()
        val isoTimestamp = SimpleDateFormat(
            "yyyy-MM-dd'T'HH:mm:ss.SSSXXX", Locale.US
        ).format(Date(now))

        try {
            val fullFrameFile = File(dir, "${prefix}_fullframe.jpg")
            FileOutputStream(fullFrameFile).use { out ->
                fullFrame.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
            }

            // Clean full-resolution still for manual species verification.
            if (fullResFrame != null) {
                val fullResFile = File(dir, "${prefix}_fullres.jpg")
                FileOutputStream(fullResFile).use { out ->
                    fullResFrame.compress(Bitmap.CompressFormat.JPEG, FULLRES_JPEG_QUALITY, out)
                }
            }

            val annotated = createAnnotatedFrame(fullFrame, detections)
            val annotatedFile = File(dir, "${prefix}_annotated.jpg")
            FileOutputStream(annotatedFile).use { out ->
                annotated.compress(Bitmap.CompressFormat.JPEG, ANNOTATED_JPEG_QUALITY, out)
            }
            annotated.recycle()

            if (screenBitmap != null) {
                val screenFile = File(dir, "${prefix}_screen.png")
                FileOutputStream(screenFile).use { out ->
                    screenBitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                }
            }

            val battery = readBattery(context)
            val json = buildMetaJson(
                captureId, isoTimestamp, now, detections,
                inferenceMs, fps, battery, gpuDelegate
            )
            File(dir, "${prefix}_meta.json").writeText(json.toString(2))

            appendCsv(
                dir, captureId, isoTimestamp, detections,
                inferenceMs, fps, battery, gpuDelegate
            )

            Log.i(TAG, "Capture #$captureId saved: ${dir.absolutePath}/$prefix*")
            return dir.absolutePath

        } catch (t: Throwable) {
            Log.e(TAG, "Capture #$captureId failed", t)
            return null
        }
    }

    /** "Species-name__NNpct" for the top detection, "__xN" if several, or "none". */
    private fun filenameLabel(detections: List<TrackedDetection>): String {
        val primary = detections.maxByOrNull { it.confidence }
            ?: return "none"
        val species = primary.species.name
            .replace(Regex("[^A-Za-z0-9]+"), "-")
            .trim('-')
        val pct = (primary.confidence * 100).toInt()
        val extra = if (detections.size > 1) "__x${detections.size}" else ""
        return "${species}__${pct}pct$extra"
    }

    /** Draw boxes and labels on a copy of the frame, independent of PixelCopy. */
    private fun createAnnotatedFrame(
        source: Bitmap,
        detections: List<TrackedDetection>
    ): Bitmap {
        val bmp = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(bmp)
        val w = bmp.width.toFloat()
        val h = bmp.height.toFloat()

        for (det in detections) {
            val status = MushroomRegistry.displayStatus(
                det.species.toxicity, det.confidence, det.voteConfirmed
            )
            val argb = colorToArgb(status.color)
            boxPaint.color = argb
            boxPaint.strokeWidth = if (det.frozen) 8f else 4f
            labelBgPaint.color = (argb and 0x00FFFFFF) or 0xE6000000.toInt() // 90% opacity

            val rect = RectF(
                det.box.left * w, det.box.top * h,
                det.box.right * w, det.box.bottom * h
            )
            canvas.drawRoundRect(rect, 12f, 12f, boxPaint)

            val conf = "%.0f%%".format(det.confidence * 100)
            val label = "${det.species.name} — ${status.label} ($conf)"
            val textWidth = labelTextPaint.measureText(label)
            val textHeight = labelTextPaint.textSize
            val padH = 8f
            val padV = 4f
            val labelY = (rect.top - textHeight - padV * 2 - 4f).coerceAtLeast(0f)

            canvas.drawRoundRect(
                RectF(rect.left, labelY, rect.left + textWidth + padH * 2, labelY + textHeight + padV * 2),
                8f, 8f, labelBgPaint
            )
            canvas.drawText(label, rect.left + padH, labelY + padV + textHeight - 4f, labelTextPaint)
        }
        return bmp
    }

    private fun colorToArgb(composeColor: androidx.compose.ui.graphics.Color): Int {
        return android.graphics.Color.argb(
            (composeColor.alpha * 255).toInt(),
            (composeColor.red * 255).toInt(),
            (composeColor.green * 255).toInt(),
            (composeColor.blue * 255).toInt()
        )
    }

    private fun buildMetaJson(
        captureId: Int,
        isoTimestamp: String,
        epochMs: Long,
        detections: List<TrackedDetection>,
        inferenceMs: Double,
        fps: Double,
        battery: BatteryInfo,
        gpuDelegate: Boolean
    ): JSONObject = JSONObject().apply {
        put("captureId", captureId)
        put("timestamp", isoTimestamp)
        put("epochMs", epochMs)
        put("inferenceLatencyMs", num("%.1f", inferenceMs).toDouble())
        put("fps", num("%.1f", fps).toDouble())
        put("batteryPercent", battery.percent)
        // org.json rejects NaN/Infinity, so fall back to JSON null when unavailable.
        if (battery.celsius.isNaN()) {
            put("batteryCelsius", JSONObject.NULL)
        } else {
            put("batteryCelsius", num("%.1f", battery.celsius.toDouble()).toDouble())
        }
        put("gpuDelegate", gpuDelegate)
        put("groundTruth", JSONObject.NULL)

        val detsArray = JSONArray()
        for (det in detections) {
            val detJson = JSONObject().apply {
                put("trackId", det.id)
                put("species", det.species.name)
                put("classId", det.species.classId)
                put("toxicity", det.species.toxicity.name)
                put("confidence", num("%.4f", det.confidence).toDouble())
                put("voteConfirmed", det.voteConfirmed)
                put("frozen", det.frozen)
                put("boundingBox", JSONObject().apply {
                    put("left", num("%.4f", det.box.left).toDouble())
                    put("top", num("%.4f", det.box.top).toDouble())
                    put("right", num("%.4f", det.box.right).toDouble())
                    put("bottom", num("%.4f", det.box.bottom).toDouble())
                })

                val topNArray = JSONArray()
                for (cs in det.topScores) {
                    val sp = MushroomRegistry.fromId(cs.classId)
                    topNArray.put(JSONObject().apply {
                        put("classId", cs.classId)
                        put("species", sp?.name ?: "unknown_${cs.classId}")
                        put("score", num("%.4f", cs.score).toDouble())
                    })
                }
                put("topN", topNArray)
            }
            detsArray.put(detJson)
        }
        put("detections", detsArray)
    }

    private fun appendCsv(
        dir: File,
        captureId: Int,
        isoTimestamp: String,
        detections: List<TrackedDetection>,
        inferenceMs: Double,
        fps: Double,
        battery: BatteryInfo,
        gpuDelegate: Boolean
    ) {
        val file = File(dir, CSV_NAME)
        // Rank 1 is the species column; ranks 2..TOP_N_CSV add species,classId,score each.
        val extraRanks = 2..TOP_N_CSV
        val telemetry = listOf(
            num("%.1f", inferenceMs),
            num("%.1f", fps),
            battery.percent.toString(),
            num("%.1f", battery.celsius.toDouble()),
            gpuDelegate.toString()
        )

        if (!csvHeaderWritten) {
            val topNHeaders = extraRanks.flatMap { n ->
                listOf("top${n}_species", "top${n}_classId", "top${n}_score")
            }
            val header = listOf(
                "session", "captureId", "timestamp", "trackId", "species", "classId", "toxicity",
                "confidence", "voteConfirmed", "frozen",
                "box_left", "box_top", "box_right", "box_bottom"
            ) + topNHeaders + listOf("latencyMs", "fps", "battery%", "batteryC", "gpu")
            file.writeText(header.joinToString(",") + "\n")
            csvHeaderWritten = true
        }

        if (detections.isEmpty()) {
            val emptyDetectionCols = List(11) { "" }              // trackId..box_bottom
            val emptyTopN = List(extraRanks.count() * 3) { "" }   // species,classId,score * ranks
            val row = listOf(sessionId ?: "", captureId.toString(), isoTimestamp) +
                emptyDetectionCols + emptyTopN + telemetry
            file.appendText(row.joinToString(",") + "\n")
            return
        }

        for (det in detections) {
            val topN = det.topScores
            val topNCols = extraRanks.flatMap { rank ->
                // rank 1 == topN[0] (primary species), so rank N maps to index N-1.
                val entry = topN.getOrNull(rank - 1)
                if (entry != null) {
                    val sp = MushroomRegistry.fromId(entry.classId)
                    listOf(sp?.name ?: "?", entry.classId.toString(), num("%.4f", entry.score))
                } else {
                    listOf("", "", "")
                }
            }
            val row = listOf(
                sessionId ?: "", captureId.toString(), isoTimestamp, det.id.toString(),
                det.species.name, det.species.classId.toString(), det.species.toxicity.name,
                num("%.4f", det.confidence), det.voteConfirmed.toString(), det.frozen.toString(),
                num("%.4f", det.box.left), num("%.4f", det.box.top),
                num("%.4f", det.box.right), num("%.4f", det.box.bottom)
            ) + topNCols + telemetry
            file.appendText(row.joinToString(",") + "\n")
        }
    }

    // Locale.US keeps a dot as the decimal separator in JSON and CSV.
    private fun num(pattern: String, value: Number): String = String.format(Locale.US, pattern, value)

    private data class BatteryInfo(val percent: Int, val celsius: Float)

    private fun readBattery(context: Context): BatteryInfo {
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = intent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = intent?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val percent = if (scale > 0) (level * 100) / scale else -1
        val tenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            ?: Int.MIN_VALUE
        val celsius = if (tenths == Int.MIN_VALUE) Float.NaN else tenths / 10f
        return BatteryInfo(percent, celsius)
    }
}
