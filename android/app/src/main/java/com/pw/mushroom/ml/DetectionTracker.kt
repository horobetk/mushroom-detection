package com.pw.mushroom.ml

import com.pw.mushroom.model.MushroomRegistry
import com.pw.mushroom.model.MushroomSpecies
import kotlin.math.max
import kotlin.math.min

/**
 * A detection promoted to a stable, cross-frame track.
 *
 * @property id      Stable identifier for this track across frames.
 * @property species Latest catalogue species matched to the track.
 * @property confidence Latest detection score.
 * @property box     Smoothed bounding box (frozen once [frozen] is true).
 * @property frozen  True once the box has been present in the same area long
 *                   enough to be locked (see [DetectionTracker.FREEZE_MS]).
 * @property voteConfirmed True once [species] won the temporal voting buffer
 *                   (top pick in at least [MushroomRegistry.VOTE_REQUIRED] of
 *                   the last [MushroomRegistry.VOTE_WINDOW] frames). Required
 *                   before the SAFE (edible) badge is allowed to show.
 */
data class TrackedDetection(
    val id: Int,
    val species: MushroomSpecies,
    val confidence: Float,
    val box: NormalizedBox,
    val frozen: Boolean,
    val voteConfirmed: Boolean
)

/**
 * Lightweight IoU-based multi-object tracker with a spatial cooldown.
 *
 * Detections that stay in roughly the same place across frames are matched to a
 * persistent track. Once a track has lived longer than [FREEZE_MS] its box is
 * frozen (no longer smoothed/moved) and flagged so the UI can render it as a
 * locked result. When every visible track is frozen the scene is considered
 * "locked" ([isSceneLocked]); the caller can then drop the inference rate to a
 * heartbeat to conserve battery instead of running the model every interval.
 *
 * Not thread-safe: call [update] from a single inference thread.
 */
class DetectionTracker {

    private val tracks = ArrayList<Track>()
    private var nextId = 0

    /**
     * Fold a fresh set of detections into the tracked state.
     *
     * @param detections Post-NMS detections for this frame.
     * @param nowMs      Current monotonic time in milliseconds.
     * @return Current tracked detections to render.
     */
    fun update(detections: List<DetectionResult>, nowMs: Long): List<TrackedDetection> {
        val unmatched = detections.toMutableList()

        // Greedily match each existing track to its best-overlapping detection.
        for (track in tracks) {
            var bestIdx = -1
            var bestIou = MATCH_IOU
            for (i in unmatched.indices) {
                val iou = iou(track.box, unmatched[i].boundingBox)
                if (iou > bestIou) {
                    bestIou = iou
                    bestIdx = i
                }
            }
            if (bestIdx >= 0) {
                track.absorb(unmatched.removeAt(bestIdx), nowMs)
            }
        }

        for (detection in unmatched) {
            tracks.add(Track(nextId++, detection, nowMs))
        }

        tracks.removeAll { nowMs - it.lastSeenMs > STALE_MS }

        // Only publish tracks confirmed over a streak of consecutive frames.
        // This rejects momentary texture glitches (keyboards, desks) that appear
        // for one or two frames before disappearing.
        return tracks
            .filter { it.confirmCount >= CONFIRM_STREAK }
            .map { it.toTracked() }
    }

    /** True when there is at least one confirmed track and all of them are frozen. */
    fun isSceneLocked(): Boolean {
        val confirmed = tracks.filter { it.confirmCount >= CONFIRM_STREAK }
        return confirmed.isNotEmpty() && confirmed.all { it.frozen }
    }

    /** Forget all tracked state (e.g. when scanning is restarted). */
    fun reset() {
        tracks.clear()
    }

    private class Track(
        val id: Int,
        seed: DetectionResult,
        nowMs: Long
    ) {
        var species: MushroomSpecies = seed.species
        var confidence: Float = seed.confidence
        var box: NormalizedBox = seed.boundingBox
        val firstSeenMs: Long = nowMs
        var lastSeenMs: Long = nowMs
        var frozen: Boolean = false

        // Number of consecutive frames this track kept the same species class.
        var confirmCount: Int = 1

        // Rolling top-class ids for SAFE vote: need VOTE_REQUIRED hits in VOTE_WINDOW.
        private val recentClassIds = ArrayDeque<Int>(MushroomRegistry.VOTE_WINDOW)

        init {
            recentClassIds.addLast(seed.species.classId)
        }

        fun absorb(detection: DetectionResult, nowMs: Long) {
            lastSeenMs = nowMs
            confidence = detection.confidence

            confirmCount = if (detection.species.classId == species.classId) {
                confirmCount + 1
            } else {
                1
            }
            species = detection.species

            recentClassIds.addLast(detection.species.classId)
            if (recentClassIds.size > MushroomRegistry.VOTE_WINDOW) {
                recentClassIds.removeFirst()
            }

            if (nowMs - firstSeenMs >= FREEZE_MS) {
                frozen = true
                return
            }
            box = smooth(box, detection.boundingBox)
        }

        // True when current species won >= VOTE_REQUIRED of the recent window.
        fun voteConfirmed(): Boolean =
            recentClassIds.count { it == species.classId } >= MushroomRegistry.VOTE_REQUIRED

        fun toTracked(): TrackedDetection =
            TrackedDetection(id, species, confidence, box, frozen, voteConfirmed())

        private fun smooth(old: NormalizedBox, new: NormalizedBox): NormalizedBox =
            NormalizedBox(
                left = old.left * SMOOTHING + new.left * (1f - SMOOTHING),
                top = old.top * SMOOTHING + new.top * (1f - SMOOTHING),
                right = old.right * SMOOTHING + new.right * (1f - SMOOTHING),
                bottom = old.bottom * SMOOTHING + new.bottom * (1f - SMOOTHING)
            )
    }

    companion object {
        /** Minimum IoU for a detection to be considered the same object. */
        private const val MATCH_IOU = 0.35f

        /** A track older than this (continuously seen) is frozen/locked. */
        const val FREEZE_MS = 5_000L

        /** Consecutive same-class frames required before a track is displayed. */
        private const val CONFIRM_STREAK = 2

        /** A track not seen for this long is discarded. */
        private const val STALE_MS = 1_200L

        /** Weight of the previous box position when smoothing (0..1). */
        private const val SMOOTHING = 0.6f

        private fun iou(a: NormalizedBox, b: NormalizedBox): Float {
            val interLeft = max(a.left, b.left)
            val interTop = max(a.top, b.top)
            val interRight = min(a.right, b.right)
            val interBottom = min(a.bottom, b.bottom)

            val interW = max(0f, interRight - interLeft)
            val interH = max(0f, interBottom - interTop)
            val intersection = interW * interH

            val areaA = max(0f, a.right - a.left) * max(0f, a.bottom - a.top)
            val areaB = max(0f, b.right - b.left) * max(0f, b.bottom - b.top)
            val union = areaA + areaB - intersection

            return if (union <= 0f) 0f else intersection / union
        }
    }
}
