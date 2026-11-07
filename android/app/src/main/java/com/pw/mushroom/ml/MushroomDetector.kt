package com.pw.mushroom.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.pw.mushroom.model.MushroomRegistry
import com.pw.mushroom.model.MushroomSpecies
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/** Class score kept for Research Mode top-N diagnostics. */
data class ClassScore(val classId: Int, val score: Float)

/** Single species prediction produced by [MushroomDetector]. */
data class DetectionResult(
    val species: MushroomSpecies,
    val confidence: Float,
    val boundingBox: NormalizedBox,
    /** Empty unless Research Mode is active. */
    val topScores: List<ClassScore> = emptyList()
)

/** TensorFlow Lite wrapper and decoder for the YOLO11m model. */
class MushroomDetector(context: Context) {

    private val interpreter: Interpreter

    // Null when running on CPU.
    private var gpuDelegate: GpuDelegate? = null

    val isGpuEnabled: Boolean get() = gpuDelegate != null

    // Read from the model at load time, so a re-exported model needs no code changes.
    val inputResolution: Int
    private val numChannels: Int // 4 box coords + numClasses
    private val numAnchors: Int
    private val numClasses: Int

    // Reusable buffers to avoid per-frame allocations.
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val pixels: IntArray

    init {
        val options = Interpreter.Options()
        attachAcceleration(options)

        interpreter = try {
            Interpreter(loadModelFile(context, MODEL_ASSET), options)
        } catch (t: Throwable) {
            // Some GPUs reject an op only at interpreter creation; retry on CPU.
            if (gpuDelegate != null) {
                Log.w(TAG, "Interpreter init failed with GPU delegate, retrying on CPU.", t)
                gpuDelegate?.close()
                gpuDelegate = null
                val cpuOptions = Interpreter.Options().apply { numThreads = CPU_THREADS }
                Interpreter(loadModelFile(context, MODEL_ASSET), cpuOptions)
            } else {
                throw t
            }
        }

        // Input tensor is NHWC: [1, res, res, 3].
        val inShape = interpreter.getInputTensor(0).shape()
        inputResolution = inShape[1]

        // Output tensor is transposed YOLO11: [1, 4 + numClasses, numAnchors].
        val outShape = interpreter.getOutputTensor(0).shape()
        numChannels = outShape[1]
        numAnchors = outShape[2]
        numClasses = numChannels - BOX_COORDS

        if (numClasses != MushroomRegistry.speciesCount) {
            Log.w(
                TAG,
                "Model reports $numClasses classes but the registry defines " +
                    "${MushroomRegistry.speciesCount}. Labels may be misaligned."
            )
        }

        pixels = IntArray(inputResolution * inputResolution)
        inputBuffer = ByteBuffer
            .allocateDirect(inputResolution * inputResolution * CHANNELS * FLOAT_BYTES)
            .order(ByteOrder.nativeOrder())
        outputBuffer = ByteBuffer
            .allocateDirect(numChannels * numAnchors * FLOAT_BYTES)
            .order(ByteOrder.nativeOrder())

        Log.i(
            TAG,
            "Detector ready. input=$inputResolution classes=$numClasses " +
                "out=[$numChannels, $numAnchors] gpu=${gpuDelegate != null}"
        )
    }

    /** Use the GPU delegate when the device is on the compatibility list, else CPU threads. */
    private fun attachAcceleration(options: Interpreter.Options) {
        val compatList = CompatibilityList()
        if (compatList.isDelegateSupportedOnThisDevice) {
            try {
                val delegateOptions = compatList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
                Log.i(TAG, "GPU delegate attached (device is compatible).")
                return
            } catch (t: Throwable) {
                Log.w(TAG, "GPU delegate creation failed despite compatibility.", t)
                gpuDelegate?.close()
                gpuDelegate = null
            }
        } else {
            Log.i(TAG, "GPU delegate unsupported on this device.")
        }
        options.numThreads = CPU_THREADS
    }

    /** Run inference on a square (letterboxed) bitmap and return post-NMS detections. */
    fun detect(squareBitmap: Bitmap): List<DetectionResult> {
        val resized = if (squareBitmap.width != inputResolution ||
            squareBitmap.height != inputResolution
        ) {
            Bitmap.createScaledBitmap(squareBitmap, inputResolution, inputResolution, true)
        } else {
            squareBitmap
        }

        fillInputBuffer(resized)
        if (resized !== squareBitmap) resized.recycle()

        inputBuffer.rewind()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)

        val floats = readOutputFloats()
        val candidates = decode(floats)
        val kept = nonMaxSuppression(candidates)

        return kept.mapNotNull { raw ->
            val species = MushroomRegistry.fromId(raw.classIndex) ?: return@mapNotNull null
            DetectionResult(
                species = species,
                confidence = raw.score,
                boundingBox = normalizeBox(raw.x1, raw.y1, raw.x2, raw.y2),
                topScores = raw.topScores
            )
        }
    }

    /** Write RGB values scaled to [0, 1] into the Float32 input buffer. */
    private fun fillInputBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        bitmap.getPixels(pixels, 0, inputResolution, 0, 0, inputResolution, inputResolution)

        for (pixel in pixels) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f) // R
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)  // G
            inputBuffer.putFloat((pixel and 0xFF) / 255f)          // B
        }
    }

    /** Layout of the result: index = channel * numAnchors + anchor. */
    private fun readOutputFloats(): FloatArray {
        outputBuffer.rewind()
        val result = FloatArray(numChannels * numAnchors)
        outputBuffer.asFloatBuffer().get(result)
        return result
    }

    /** Channels 0..3 are (cx, cy, w, h), the rest are class scores. */
    private fun decode(data: FloatArray): List<RawDetection> {
        val detections = ArrayList<RawDetection>()
        val globalMin = MushroomRegistry.minSafetyThreshold
        val wantTopN = collectTopN

        for (anchor in 0 until numAnchors) {
            var bestClass = -1
            var bestScore = 0f

            for (c in 0 until numClasses) {
                val score = data[(BOX_COORDS + c) * numAnchors + anchor]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }

            // Cheap global gate first, then the per-class safety threshold.
            if (bestClass < 0 || bestScore < globalMin) continue
            if (bestScore < MushroomRegistry.thresholdFor(bestClass)) continue

            val topScores: List<ClassScore> = if (wantTopN) {
                (0 until numClasses)
                    .map { c -> ClassScore(c, data[(BOX_COORDS + c) * numAnchors + anchor]) }
                    .sortedByDescending { it.score }
                    .take(TOP_N)
            } else {
                emptyList()
            }

            val cx = data[0 * numAnchors + anchor]
            val cy = data[1 * numAnchors + anchor]
            val w = data[2 * numAnchors + anchor]
            val h = data[3 * numAnchors + anchor]
            val half = 0.5f

            detections.add(
                RawDetection(
                    x1 = cx - w * half,
                    y1 = cy - h * half,
                    x2 = cx + w * half,
                    y2 = cy + h * half,
                    score = bestScore,
                    classIndex = bestClass,
                    topScores = topScores
                )
            )
        }
        return detections
    }

    /** Ultralytics exports may emit either pixel or already-normalised coordinates. */
    private fun normalizeBox(x1: Float, y1: Float, x2: Float, y2: Float): NormalizedBox {
        fun norm(v: Float): Float = if (v > 1.5f) v / inputResolution else v.coerceIn(0f, 1f)
        return NormalizedBox(
            left = norm(min(x1, x2)),
            top = norm(min(y1, y2)),
            right = norm(max(x1, x2)),
            bottom = norm(max(y1, y2))
        )
    }

    /** Greedy class-agnostic NMS over the top [MAX_NMS_INPUT] candidates. */
    private fun nonMaxSuppression(input: List<RawDetection>): List<RawDetection> {
        if (input.isEmpty()) return emptyList()

        val sorted = input.sortedByDescending { it.score }
            .take(MAX_NMS_INPUT)
            .toMutableList()
        val kept = ArrayList<RawDetection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            if (kept.size >= MAX_DETECTIONS) break

            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                if (iou(best, iterator.next()) > IOU_THRESHOLD) iterator.remove()
            }
        }
        return kept
    }

    private fun iou(a: RawDetection, b: RawDetection): Float {
        val interLeft = max(a.x1, b.x1)
        val interTop = max(a.y1, b.y1)
        val interRight = min(a.x2, b.x2)
        val interBottom = min(a.y2, b.y2)

        val interW = max(0f, interRight - interLeft)
        val interH = max(0f, interBottom - interTop)
        val intersection = interW * interH

        val areaA = max(0f, a.x2 - a.x1) * max(0f, a.y2 - a.y1)
        val areaB = max(0f, b.x2 - b.x1) * max(0f, b.y2 - b.y1)
        val union = areaA + areaB - intersection

        return if (union <= 0f) 0f else intersection / union
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        gpuDelegate = null
    }

    private data class RawDetection(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val score: Float,
        val classIndex: Int,
        val topScores: List<ClassScore> = emptyList()
    )

    /** Memory-maps the model; it must stay uncompressed in the APK. */
    private fun loadModelFile(context: Context, assetName: String): ByteBuffer {
        context.assets.openFd(assetName).use { fd ->
            fd.createInputStream().use { input ->
                return input.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.startOffset,
                    fd.declaredLength
                )
            }
        }
    }

    companion object {
        private const val TAG = "MushroomDetector"

        const val MODEL_ASSET = "best_float16.tflite"

        private const val CHANNELS = 3
        private const val BOX_COORDS = 4
        private const val FLOAT_BYTES = 4
        private const val CPU_THREADS = 4

        private const val IOU_THRESHOLD = 0.45f

        /** Bounds the quadratic NMS cost per frame. */
        private const val MAX_NMS_INPUT = 50

        private const val MAX_DETECTIONS = 20

        private const val TOP_N = 5

        /** Set by [MushroomViewModel] while Research Mode is on. */
        @Volatile
        @JvmStatic
        var collectTopN: Boolean = false
    }
}
