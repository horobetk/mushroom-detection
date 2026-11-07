package com.pw.mushroom.ui

import com.pw.mushroom.ml.TrackedDetection

/** Camera screen state exposed by [MushroomViewModel]. */
sealed interface MushroomUiState {

    data object Initializing : MushroomUiState

    /** [locked] means all tracks are frozen and inference runs at the heartbeat rate. */
    data class Scanning(
        val detections: List<TrackedDetection> = emptyList(),
        val locked: Boolean = false
    ) : MushroomUiState

    data class Error(
        val message: String
    ) : MushroomUiState
}

/** One-shot snackbar message. */
data class SaveEvent(val message: String, val success: Boolean)
