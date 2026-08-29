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

/**
 * A single species prediction produced by [MushroomDetector].
 *
 * @property species     Catalogue entry describing the detected species.
 * @property confidence  Detection score in the range 0.0 .. 1.0.
 * @property boundingBox Normalised box in model input space.
 */
data class DetectionResult(
    val species: MushroomSpecies,
    val confidence: Float,
    val boundingBox: NormalizedBox
)

/**
 * Low-level TensorFlow Lite wrapper for a Float32 YOLO11m detection model.
 *
 * Pipeline for a single frame:
 *  1. Normalise a square RGB bitmap into a Float32 input buffer (values [0,1]).
 *  2. Run the interpreter, which writes a Float32 output tensor of shape
 *     [1, 4 + numClasses, numAnchors] (YOLO11 transposed layout).
 *  3. Decode anchors, threshold each by its per-class safety threshold, cap the
 *     survivors, and run Non-Maximum Suppression.
 *
 * The input resolution (e.g. 640 or 1920) and the anchor/class counts are read
 * from the loaded model at runtime, so a re-exported model of a different size
 * runs without code changes and cannot cause index-out-of-bounds crashes.
 *
 * The class is free of Android UI and CameraX dependencies so it can be reused
 * and tested in isolation.
 */
class MushroomDetector(context: Context) {

    private val interpreter: Interpreter

    // Held as a field so it can be released in close(); null when running on CPU.
    private var gpuDelegate: GpuDelegate? = null

    // Geometry read from the model at load time (supports variable sizes/classes).
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
            // Some devices fail at interpreter creation when a delegate is attached
            // but an op in the graph is unsupported; retry on plain CPU.
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

    /**
     * Attach hardware acceleration. Uses the TFLite GPU compatibility list to
     * decide whether the device can run the delegate; if not, falls back to
     * multi-threaded CPU execution.
     */
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
        // CPU fallback.
        options.numThreads = CPU_THREADS
    }

    /**
     * Run inference on a square bitmap.
     *
     * @param squareBitmap Center-cropped square frame; resized to the model input
     *                     resolution if it does not already match.
     * @return All detections after NMS that clear their per-class threshold.
     */
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
                boundingBox = normalizeBox(raw.x1, raw.y1, raw.x2, raw.y2)
            )
        }
    }

    // -------------------------------------------------------------------------
    // Input preparation
    // -------------------------------------------------------------------------

    /**
     * Write normalised RGB float values [0.0, 1.0] into the input buffer.
     * Each channel occupies 4 bytes (Float32).
     */
    private fun fillInputBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        bitmap.getPixels(pixels, 0, inputResolution, 0, 0, inputResolution, inputResolution)

        for (pixel in pixels) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255f) // R
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255f)  // G
            inputBuffer.putFloat((pixel and 0xFF) / 255f)          // B
        }
    }

    // -------------------------------------------------------------------------
    // Output reading
    // -------------------------------------------------------------------------

    /**
     * Copy the output ByteBuffer into a FloatArray for indexed access.
     * Layout: index = channel * numAnchors + anchor.
     */
    private fun readOutputFloats(): FloatArray {
        outputBuffer.rewind()
        val result = FloatArray(numChannels * numAnchors)
        outputBuffer.asFloatBuffer().get(result)
        return result
    }

    /**
     * Decode the Float32 [1, numChannels, numAnchors] tensor into box candidates.
     *
     * Channels 0..3 are (cx, cy, w, h); channels 4..(3+numClasses) are per-class
     * scores. Fast-fail order per anchor:
     *   1. Find the best class + score.
     *   2. Reject early if below the global minimum threshold (cheapest gate).
     *   3. Reject if below that class's specific safety threshold.
     * Only survivors allocate a [RawDetection], keeping GC pressure low.
     */
    private fun decode(data: FloatArray): List<RawDetection> {
        val detections = ArrayList<RawDetection>()
        val globalMin = MushroomRegistry.minSafetyThreshold

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
                    classIndex = bestClass
                )
            )
        }
        return detections
    }

    /**
     * Convert raw box coordinates to normalised [0, 1] space. Ultralytics exports
     * may emit pixel coords (0..inputResolution) or already-normalised values.
     */
    private fun normalizeBox(x1: Float, y1: Float, x2: Float, y2: Float): NormalizedBox {
        fun norm(v: Float): Float = if (v > 1.5f) v / inputResolution else v.coerceIn(0f, 1f)
        return NormalizedBox(
            left = norm(min(x1, x2)),
            top = norm(min(y1, y2)),
            right = norm(max(x1, x2)),
            bottom = norm(max(y1, y2))
        )
    }

    // -------------------------------------------------------------------------
    // Non-Maximum Suppression
    // -------------------------------------------------------------------------

    /**
     * Greedy NMS. To bound work, only the [MAX_NMS_INPUT] highest-scoring
     * candidates are considered; the highest-scoring box is kept and overlapping
     * boxes above [IOU_THRESHOLD] are suppressed, up to [MAX_DETECTIONS].
     */
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

    /** Intersection-over-Union of two axis-aligned boxes. */
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

    /** Release native resources held by the interpreter and the GPU delegate. */
    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        gpuDelegate = null
    }

    /** Internal decoded box before it is mapped to a catalogue species. */
    private data class RawDetection(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val score: Float,
        val classIndex: Int
    )

    /** Memory-map the .tflite model directly from the app assets. */
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

        /** FP16 model bundled under app/src/main/assets. */
        const val MODEL_ASSET = "best_float16.tflite"

        private const val CHANNELS = 3
        private const val BOX_COORDS = 4
        private const val FLOAT_BYTES = 4
        private const val CPU_THREADS = 4

        /** IoU above which overlapping boxes are suppressed. */
        private const val IOU_THRESHOLD = 0.45f

        /** Cap on candidates fed into NMS to keep per-frame work bounded. */
        private const val MAX_NMS_INPUT = 50

        /** Upper bound on detections kept per frame. */
        private const val MAX_DETECTIONS = 20
    }
}
