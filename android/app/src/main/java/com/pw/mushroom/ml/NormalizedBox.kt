package com.pw.mushroom.ml

/** Axis-aligned box in normalised [0, 1] coordinates, origin top-left. */
data class NormalizedBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)
