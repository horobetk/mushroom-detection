package com.pw.mushroom.ml

import com.pw.mushroom.model.MushroomRegistry
import com.pw.mushroom.model.MushroomSpecies
import kotlin.math.max
import kotlin.math.min

/** Detection promoted to a stable cross-frame track. */
data class TrackedDetection(
    val id: Int,
    val species: MushroomSpecies,
    val confidence: Float,
    val box: NormalizedBox,
    val frozen: Boolean,
    val voteConfirmed: Boolean,
    /** Empty unless Research Mode is on. */
    val topScores: List<ClassScore> = emptyList(),
    /** Time (ms) of the frame in which this track was first detected. */
    val firstSeenMs: Long = 0L
)

/** IoU-based tracker with box freezing and the SAFE vote buffer. Not thread-safe. */
class DetectionTracker {

    private val tracks = ArrayList<Track>()
    private var nextId = 0

    fun update(detections: List<DetectionResult>, nowMs: Long): List<TrackedDetection> {
        val unmatched = detections.toMutableList()

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

        // The streak filters one-frame false positives on background textures.
        return tracks
            .filter { it.confirmCount >= CONFIRM_STREAK }
            .map { it.toTracked() }
    }

    /** True when there is at least one confirmed track and all of them are frozen. */
    fun isSceneLocked(): Boolean {
        val confirmed = tracks.filter { it.confirmCount >= CONFIRM_STREAK }
        return confirmed.isNotEmpty() && confirmed.all { it.frozen }
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
        var topScores: List<ClassScore> = seed.topScores

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

            topScores = detection.topScores

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
            TrackedDetection(id, species, confidence, box, frozen, voteConfirmed(), topScores, firstSeenMs)

        private fun smooth(old: NormalizedBox, new: NormalizedBox): NormalizedBox =
            NormalizedBox(
                left = old.left * SMOOTHING + new.left * (1f - SMOOTHING),
                top = old.top * SMOOTHING + new.top * (1f - SMOOTHING),
                right = old.right * SMOOTHING + new.right * (1f - SMOOTHING),
                bottom = old.bottom * SMOOTHING + new.bottom * (1f - SMOOTHING)
            )
    }

    companion object {
        private const val MATCH_IOU = 0.35f

        const val FREEZE_MS = 5_000L

        /** Consecutive same-class frames required before a track is displayed. */
        private const val CONFIRM_STREAK = 2

        private const val STALE_MS = 1_200L

        /** Weight of the previous box when smoothing. */
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
