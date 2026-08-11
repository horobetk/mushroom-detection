package com.pw.mushroom.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.sp
import com.pw.mushroom.ml.NormalizedBox
import com.pw.mushroom.ml.TrackedDetection
import com.pw.mushroom.model.MushroomRegistry

/**
 * AR overlay drawn over the camera preview.
 *
 * Every tracked detection is rendered as a rounded box tinted by its toxicity
 * colour, with a bilingual label (Latin species name + Polish status). Frozen
 * (locked) detections use a thicker stroke. Tapping inside a box saves that
 * find via [onTapDetection].
 */
@Composable
fun DetectionOverlay(
    detections: List<TrackedDetection>,
    onTapDetection: (TrackedDetection) -> Unit,
    modifier: Modifier = Modifier
) {
    val textMeasurer = rememberTextMeasurer()

    BoxWithConstraints(modifier = modifier.fillMaxSize()) {
        val screenW = constraints.maxWidth.toFloat()
        val screenH = constraints.maxHeight.toFloat()

        // Boxes are normalised to the visible (FILL_CENTER) region, so they map
        // directly onto the full screen with no letterbox offset.
        val hitTargets: List<Pair<TrackedDetection, Rect>> =
            remember(detections, screenW, screenH) {
                detections.map { it to mapBoxToScreen(it.box, screenW, screenH) }
            }

        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(hitTargets) {
                    detectTapGestures { tap ->
                        hitTargets.firstOrNull { it.second.contains(tap) }
                            ?.let { onTapDetection(it.first) }
                    }
                }
        ) {
            hitTargets.forEach { (detection, rect) ->
                // Toxic: immediate warning. Edible: green only if voteConfirmed.
                val status = MushroomRegistry.displayStatus(
                    detection.species.toxicity,
                    detection.confidence,
                    detection.voteConfirmed
                )
                val boxColor = status.color
                val strokeWidth = if (detection.frozen) 6f else 3f

                drawRoundRect(
                    color = boxColor,
                    topLeft = rect.topLeft,
                    size = rect.size,
                    cornerRadius = CornerRadius(12f, 12f),
                    style = Stroke(width = strokeWidth)
                )

                val label = "${detection.species.name} - ${status.label}"
                val textStyle = TextStyle(
                    color = Color.White,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.SemiBold
                )
                val textLayout = textMeasurer.measure(label, textStyle)
                val padH = 6f
                val padV = 3f
                val labelW = textLayout.size.width + padH * 2
                val labelH = textLayout.size.height + padV * 2
                val labelTop = (rect.top - labelH - 4f).coerceAtLeast(0f)

                drawRoundRect(
                    color = boxColor.copy(alpha = 0.92f),
                    topLeft = Offset(rect.left, labelTop),
                    size = Size(labelW, labelH),
                    cornerRadius = CornerRadius(6f, 6f)
                )
                drawText(
                    textLayoutResult = textLayout,
                    topLeft = Offset(rect.left + padH, labelTop + padV)
                )
            }
        }
    }
}

/** Map a visible-normalised box into full-screen pixel coordinates. */
fun mapBoxToScreen(
    box: NormalizedBox,
    screenW: Float,
    screenH: Float
): Rect = Rect(
    left = box.left * screenW,
    top = box.top * screenH,
    right = box.right * screenW,
    bottom = box.bottom * screenH
)
