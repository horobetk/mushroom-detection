package com.pw.mushroom.ml

/**
 * Axis-aligned bounding box in normalised [0, 1] coordinates relative to the
 * model input square (800x800). Origin is top-left.
 */
data class NormalizedBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
) {
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f

    /** True when the normalised point lies inside this box (inclusive edges). */
    fun contains(normalizedX: Float, normalizedY: Float): Boolean =
        normalizedX in left..right && normalizedY in top..bottom

    /** Squared Euclidean distance from the box centre to a normalised point. */
    fun distanceSquaredToCenter(targetX: Float = 0.5f, targetY: Float = 0.5f): Float {
        val dx = centerX - targetX
        val dy = centerY - targetY
        return dx * dx + dy * dy
    }
}
