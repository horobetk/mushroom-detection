package com.pw.mushroom.camera

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import androidx.camera.core.ImageProxy
import com.pw.mushroom.ml.NormalizedBox
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Utilities for turning a CameraX [ImageProxy] into a rotation-corrected
 * [Bitmap], preparing the square model input, and crop-mapping detections back
 * into the frame.
 *
 * Coordinate convention: the preview uses FILL_CENTER, so the visible region is
 * the largest centred crop of the frame matching the view's aspect ratio. The
 * model is fed exactly that visible region, letterboxed (aspect-preserving) into
 * the square input. Detection boxes are therefore normalised to the *visible*
 * region and map 1:1 onto the full screen.
 */
object FrameCropper {

    /**
     * Result of letterboxing the visible region into the square model input.
     *
     * @property bitmap   Square [size]x[size] input bitmap (aspect-preserved, padded).
     * @property cropRect Visible region of the source frame (view-aspect centre crop).
     * @property padX     Horizontal padding inside the square (pixels).
     * @property padY     Vertical padding inside the square (pixels).
     * @property contentW Width of the scaled content inside the square (pixels).
     * @property contentH Height of the scaled content inside the square (pixels).
     * @property size     Square side length (model input resolution).
     */
    data class Letterboxed(
        val bitmap: Bitmap,
        val cropRect: Rect,
        val padX: Float,
        val padY: Float,
        val contentW: Float,
        val contentH: Float,
        val size: Int
    )

    /**
     * Convert an ImageProxy (RGBA_8888) into a rotation-corrected, full-resolution
     * upright [Bitmap]. The caller owns the returned bitmap and must recycle it.
     */
    fun toUprightBitmap(imageProxy: ImageProxy): Bitmap {
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val raw = imageProxy.toBitmap()

        if (rotationDegrees == 0) return raw

        val matrix = Matrix().apply {
            postRotate(rotationDegrees.toFloat(), raw.width / 2f, raw.height / 2f)
        }
        return Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, matrix, true).also {
            if (it !== raw) raw.recycle()
        }
    }

    /**
     * Letterbox the visible region of [upright] (a view-aspect centre crop) into a
     * [targetSize] square, preserving aspect ratio with neutral padding.
     *
     * @param viewAspect Preview aspect ratio (width / height). Falls back to 1.0.
     */
    fun letterbox(upright: Bitmap, viewAspect: Float, targetSize: Int): Letterboxed {
        val crop = viewCropRect(upright.width, upright.height, viewAspect)
        val cropW = crop.width()
        val cropH = crop.height()

        val scale = min(targetSize.toFloat() / cropW, targetSize.toFloat() / cropH)
        val contentW = cropW * scale
        val contentH = cropH * scale
        val padX = (targetSize - contentW) / 2f
        val padY = (targetSize - contentH) / 2f

        val output = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        canvas.drawColor(PAD_COLOR)
        val dst = Rect(
            padX.roundToInt(),
            padY.roundToInt(),
            (padX + contentW).roundToInt(),
            (padY + contentH).roundToInt()
        )
        canvas.drawBitmap(upright, crop, dst, Paint(Paint.FILTER_BITMAP_FLAG))

        return Letterboxed(output, crop, padX, padY, contentW, contentH, targetSize)
    }

    /**
     * Convert a box normalised to the letterboxed square into a box normalised to
     * the *visible* region (padding removed), ready to map onto the full screen.
     */
    fun toVisibleNormalized(box: NormalizedBox, lb: Letterboxed): NormalizedBox {
        fun ux(nx: Float): Float = ((nx * lb.size - lb.padX) / lb.contentW).coerceIn(0f, 1f)
        fun uy(ny: Float): Float = ((ny * lb.size - lb.padY) / lb.contentH).coerceIn(0f, 1f)
        return NormalizedBox(
            left = ux(box.left),
            top = uy(box.top),
            right = ux(box.right),
            bottom = uy(box.bottom)
        )
    }

    /**
     * Crop the region described by a visible-normalised [box] out of the
     * full-resolution [upright] frame, at native resolution.
     *
     * @param viewAspect Preview aspect ratio (width / height), to reconstruct the
     *                   same visible crop used during inference.
     */
    fun cropNormalizedBox(upright: Bitmap, box: NormalizedBox, viewAspect: Float): Bitmap? {
        val crop = viewCropRect(upright.width, upright.height, viewAspect)
        val cropW = crop.width()
        val cropH = crop.height()

        val left = (crop.left + box.left * cropW).roundToInt().coerceIn(0, upright.width - 1)
        val top = (crop.top + box.top * cropH).roundToInt().coerceIn(0, upright.height - 1)
        val right = (crop.left + box.right * cropW).roundToInt().coerceIn(left + 1, upright.width)
        val bottom = (crop.top + box.bottom * cropH).roundToInt().coerceIn(top + 1, upright.height)

        val width = right - left
        val height = bottom - top
        if (width <= 0 || height <= 0) return null

        return Bitmap.createBitmap(upright, left, top, width, height)
    }

    /**
     * Largest centred crop of a [w]x[h] frame matching [viewAspect] (width/height).
     */
    private fun viewCropRect(w: Int, h: Int, viewAspect: Float): Rect {
        val aspect = if (viewAspect > 0f) viewAspect else 1f
        val frameAspect = w.toFloat() / h.toFloat()

        return if (frameAspect > aspect) {
            // Frame is wider than the view: crop the sides.
            val cropW = (h * aspect).roundToInt().coerceIn(1, w)
            val left = (w - cropW) / 2
            Rect(left, 0, left + cropW, h)
        } else {
            // Frame is taller than the view: crop top/bottom.
            val cropH = (w / aspect).roundToInt().coerceIn(1, h)
            val top = (h - cropH) / 2
            Rect(0, top, w, top + cropH)
        }
    }

    /** Neutral grey padding used when letterboxing (matches common YOLO padding). */
    private const val PAD_COLOR = Color.GRAY
}
