package com.pw.mushroom.ui

import com.pw.mushroom.ml.TrackedDetection

/**
 * Immutable UI state exposed by [MushroomViewModel] and rendered by the screen.
 */
sealed interface MushroomUiState {

    /** Detector is still loading; camera preview may run but no boxes yet. */
    data object Initializing : MushroomUiState

    /**
     * Live continuous scanning.
     *
     * @property detections Tracked detections to draw on the AR overlay.
     * @property locked     True when the scene is frozen (all boxes locked) and
     *                      inference has dropped to a battery-saving heartbeat.
     */
    data class Scanning(
        val detections: List<TrackedDetection> = emptyList(),
        val locked: Boolean = false
    ) : MushroomUiState

    /** Detector failed to initialise. */
    data class Error(
        val message: String
    ) : MushroomUiState
}

/** One-shot event asking the UI to show a transient message (e.g. a snackbar). */
data class SaveEvent(val message: String, val success: Boolean)
