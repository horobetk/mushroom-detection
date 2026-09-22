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

/**
 * File-based research data logger.
 *
 * Each capture produces:
 *  - `capture_NNN_fullframe.jpg`  — the raw camera frame at full resolution
 *  - `capture_NNN_annotated.jpg`  — the same frame with bounding boxes + labels drawn on it
 *  - `capture_NNN_screen.png`     — PixelCopy window screenshot (if available)
 *  - `capture_NNN_meta.json`      — detailed detection JSON with top-N scores
 *  - an append row in `session_log.csv`
 *
 * Files are written to
 * `Documents/MushroomResearch/session_YYYYMMDD_HHmmss/`.
 * This directory is accessible via USB MTP without root.
 */
object ResearchLogger {

    private const val TAG = "ResearchLogger"
    private const val ROOT_DIR = "MushroomResearch"
    private const val CSV_NAME = "session_log.csv"
    private const val JPEG_QUALITY = 92
    private const val FULLRES_JPEG_QUALITY = 95
    private const val ANNOTATED_JPEG_QUALITY = 88
    private const val TOP_N_CSV = 3 // how many top-N predictions go into CSV columns

    // Resolved session path and id. The directory itself is created lazily on the
    // first successful capture, so toggling Research Mode never leaves empty folders.
    private var sessionDir: File? = null
    private var sessionId: String? = null
    private var csvHeaderWritten = false

    // Reusable Paint objects for annotated frame rendering.
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

    /**
     * Start a new capture session. The on-disk directory is *not* created yet —
     * it is materialised lazily on the first capture so cancelled sessions leave
     * no empty folders behind.
     *
     * @return the session ID (folder name), or null if the path could not be resolved.
     */
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

    /** End the current session. */
    fun endSession() {
        Log.i(TAG, "Research session ended: $sessionId")
        sessionDir = null
        sessionId = null
        csvHeaderWritten = false
    }

    /** True once a session has been started (directory may not exist until first capture). */
    val isSessionActive: Boolean get() = sessionId != null

    /**
     * Ensure the session directory exists on disk, creating it (and a ground-truth
     * stub) on first use. Returns null if creation failed.
     */
    private fun ensureSessionDir(): File? {
        val dir = sessionDir ?: return null
        if (!dir.exists() && !dir.mkdirs()) {
            Log.e(TAG, "Failed to create session directory: ${dir.absolutePath}")
            return null
        }
        // Ground-truth stub: fill in the real species here for later verification.
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

    /**
     * Capture a complete research data bundle.
     *
     * @param captureId     Sequential capture number within this session.
     * @param fullFrame     The analysis frame (matches the detection boxes; not recycled here).
     * @param fullResFrame  Optional full-resolution sensor photo via ImageCapture, saved
     *                      clean (no boxes) for visual species verification. Not recycled here.
     * @param screenBitmap  PixelCopy window screenshot, or null if capture failed.
     * @param detections    Current tracked detections with top-N scores.
     * @param inferenceMs   Last inference latency in milliseconds.
     * @param fps           Current effective FPS.
     * @param gpuDelegate   Whether GPU delegate is active.
     * @param context       Application context for battery telemetry.
     *
     * @return the session directory path, or null on failure.
     */
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
        // Encode the primary prediction into the filename so the folder is
        // scannable at a glance without opening JSON, e.g.
        //   capture_001__Amanita-porphyria__71pct__annotated.jpg
        //   capture_002__none__annotated.jpg      (nothing detected)
        val prefix = "capture_%03d__%s".format(captureId, filenameLabel(detections))
        val now = System.currentTimeMillis()
        val isoTimestamp = SimpleDateFormat(
            "yyyy-MM-dd'T'HH:mm:ss.SSSXXX", Locale.US
        ).format(Date(now))

        try {
            // 1. Full-frame JPEG
            val fullFrameFile = File(dir, "${prefix}_fullframe.jpg")
            FileOutputStream(fullFrameFile).use { out ->
                fullFrame.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
            }

            // 1b. Full-resolution sensor photo (clean, for visual species verification).
            if (fullResFrame != null) {
                val fullResFile = File(dir, "${prefix}_fullres.jpg")
                FileOutputStream(fullResFile).use { out ->
                    fullResFrame.compress(Bitmap.CompressFormat.JPEG, FULLRES_JPEG_QUALITY, out)
                }
            }

            // 2. Annotated frame (full-frame with bounding boxes + labels drawn on top)
            val annotated = createAnnotatedFrame(fullFrame, detections)
            val annotatedFile = File(dir, "${prefix}_annotated.jpg")
            FileOutputStream(annotatedFile).use { out ->
                annotated.compress(Bitmap.CompressFormat.JPEG, ANNOTATED_JPEG_QUALITY, out)
            }
            annotated.recycle()

            // 3. PixelCopy screen capture (if available)
            if (screenBitmap != null) {
                val screenFile = File(dir, "${prefix}_screen.png")
                FileOutputStream(screenFile).use { out ->
                    screenBitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                }
            }

            // 4. Detailed metadata JSON
            val battery = readBattery(context)
            val json = buildMetaJson(
                captureId, isoTimestamp, now, detections,
                inferenceMs, fps, battery, gpuDelegate
            )
            File(dir, "${prefix}_meta.json").writeText(json.toString(2))

            // 5. Append CSV row (one row per detection, or one row if no detections)
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

    /**
     * Build the human-scannable filename fragment from the primary (highest
     * confidence) detection: "Species-name__NNpct" or "none__x0". A trailing
     * "__x{n}" is appended when more than one object was detected.
     */
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

    // ------------------------------------------------------------------
    // Annotated frame rendering
    // ------------------------------------------------------------------

    /**
     * Draw bounding boxes, species names, confidence, and status labels
     * directly on a copy of the full-frame bitmap. This produces a
     * self-contained annotated image that does not depend on PixelCopy.
     */
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

            // Label: "Species — Status (XX%)"
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

    /** Convert a Compose Color (ARGB Long) to an Android int ARGB. */
    private fun colorToArgb(composeColor: androidx.compose.ui.graphics.Color): Int {
        return android.graphics.Color.argb(
            (composeColor.alpha * 255).toInt(),
            (composeColor.red * 255).toInt(),
            (composeColor.green * 255).toInt(),
            (composeColor.blue * 255).toInt()
        )
    }

    // ------------------------------------------------------------------
    // JSON builder
    // ------------------------------------------------------------------

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
        put("inferenceLatencyMs", "%.1f".format(inferenceMs).toDouble())
        put("fps", "%.1f".format(fps).toDouble())
        put("batteryPercent", battery.percent)
        // org.json rejects NaN/Infinity, so fall back to JSON null when unavailable.
        if (battery.celsius.isNaN()) {
            put("batteryCelsius", JSONObject.NULL)
        } else {
            put("batteryCelsius", "%.1f".format(battery.celsius.toDouble()).toDouble())
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
                put("confidence", "%.4f".format(det.confidence).toDouble())
                put("voteConfirmed", det.voteConfirmed)
                put("frozen", det.frozen)
                put("boundingBox", JSONObject().apply {
                    put("left", "%.4f".format(det.box.left).toDouble())
                    put("top", "%.4f".format(det.box.top).toDouble())
                    put("right", "%.4f".format(det.box.right).toDouble())
                    put("bottom", "%.4f".format(det.box.bottom).toDouble())
                })

                val topNArray = JSONArray()
                for (cs in det.topScores) {
                    val sp = MushroomRegistry.fromId(cs.classId)
                    topNArray.put(JSONObject().apply {
                        put("classId", cs.classId)
                        put("species", sp?.name ?: "unknown_${cs.classId}")
                        put("score", "%.4f".format(cs.score).toDouble())
                    })
                }
                put("topN", topNArray)
            }
            detsArray.put(detJson)
        }
        put("detections", detsArray)
    }

    // ------------------------------------------------------------------
    // CSV logger
    // ------------------------------------------------------------------

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
        // Ranks 2..TOP_N_CSV are emitted as extra columns (rank 1 is the primary
        // species column). Each rank contributes species,classId,score.
        val extraRanks = 2..TOP_N_CSV
        val telemetry = listOf(
            "%.1f".format(inferenceMs),
            "%.1f".format(fps),
            battery.percent.toString(),
            "%.1f".format(battery.celsius.toDouble()),
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
            // Log a row even when nothing was detected: empty detection columns.
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
                    listOf(sp?.name ?: "?", entry.classId.toString(), "%.4f".format(entry.score))
                } else {
                    listOf("", "", "")
                }
            }
            val row = listOf(
                sessionId ?: "", captureId.toString(), isoTimestamp, det.id.toString(),
                det.species.name, det.species.classId.toString(), det.species.toxicity.name,
                "%.4f".format(det.confidence), det.voteConfirmed.toString(), det.frozen.toString(),
                "%.4f".format(det.box.left), "%.4f".format(det.box.top),
                "%.4f".format(det.box.right), "%.4f".format(det.box.bottom)
            ) + topNCols + telemetry
            file.appendText(row.joinToString(",") + "\n")
        }
    }

    // ------------------------------------------------------------------
    // Battery telemetry
    // ------------------------------------------------------------------

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
