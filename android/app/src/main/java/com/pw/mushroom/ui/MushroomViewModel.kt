package com.pw.mushroom.ui

import android.app.Application
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import androidx.annotation.StringRes
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.pw.mushroom.R
import com.pw.mushroom.camera.FrameCropper
import com.pw.mushroom.data.FindsRepository
import com.pw.mushroom.ml.DetectionTracker
import com.pw.mushroom.ml.InferenceProfiler
import com.pw.mushroom.ml.MushroomDetector
import com.pw.mushroom.ml.TrackedDetection
import com.pw.mushroom.model.MushroomRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex

/**
 * Owns the detection pipeline for the camera screen:
 *  - throttles inference to [INFERENCE_INTERVAL_MS] (4 FPS) to limit heat/battery,
 *  - folds detections through a [DetectionTracker] for the 5-second freeze,
 *  - drops to a slow heartbeat once the scene is locked,
 *  - crops and persists a find via [FindsRepository] on demand.
 */
class MushroomViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow<MushroomUiState>(MushroomUiState.Initializing)
    val uiState: StateFlow<MushroomUiState> = _uiState.asStateFlow()

    private val _saveEvents = MutableSharedFlow<SaveEvent>(extraBufferCapacity = 4)
    val saveEvents: SharedFlow<SaveEvent> = _saveEvents.asSharedFlow()

    private val repository = FindsRepository(application)
    private val tracker = DetectionTracker()

    private var detector: MushroomDetector? = null
    private val inferenceMutex = Mutex()
    private var lastInferenceAt = 0L

    // Guards [latestFrame], shared between the analyzer thread and save requests.
    private val frameLock = Any()
    private var latestFrame: Bitmap? = null

    // Snapshot of the most recent tracked detections, for capture-button saves.
    @Volatile
    private var currentDetections: List<TrackedDetection> = emptyList()

    // Preview aspect ratio (width / height); used to align the analysed region
    // with the FILL_CENTER preview. Updated by the UI when the view is measured.
    @Volatile
    private var viewAspectRatio: Float = 1f

    /** Report the preview's aspect ratio (width / height) from the UI layer. */
    fun setViewAspectRatio(ratio: Float) {
        if (ratio > 0f) viewAspectRatio = ratio
    }

    init {
        initialiseDetector()
    }

    private fun initialiseDetector() {
        _uiState.value = MushroomUiState.Initializing
        viewModelScope.launch(Dispatchers.Default) {
            try {
                detector = MushroomDetector(getApplication())
                _uiState.value = MushroomUiState.Scanning()
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to initialise detector", t)
                _uiState.value = MushroomUiState.Error(t.message ?: "Unknown error")
            }
        }
    }

    /**
     * Receive one upright, full-resolution frame from CameraX. Ownership of
     * [upright] transfers to the ViewModel: it is either recycled here or kept
     * as the latest frame for cropping.
     */
    fun onFrame(upright: Bitmap) {
        val activeDetector = detector
        if (activeDetector == null) {
            upright.recycle()
            return
        }

        val now = SystemClock.elapsedRealtime()
        val sceneLocked = tracker.isSceneLocked()
        val interval = if (sceneLocked && !InferenceProfiler.bypassSceneLock) {
            LOCK_HEARTBEAT_MS
        } else {
            INFERENCE_INTERVAL_MS
        }
        if (now - lastInferenceAt < interval) {
            upright.recycle()
            return
        }
        // Skip if a previous inference is still running (keep only the latest frame).
        if (!inferenceMutex.tryLock()) {
            upright.recycle()
            return
        }
        lastInferenceAt = now

        val aspect = viewAspectRatio
        viewModelScope.launch(Dispatchers.Default) {
            try {
                // Letterbox the visible region into the square input (aspect-preserving),
                // run inference, then strip the padding so boxes are normalised to the
                // visible region (they map 1:1 onto the full-screen preview).
                val lb = FrameCropper.letterbox(upright, aspect, activeDetector.inputResolution)
                val t0 = SystemClock.elapsedRealtimeNanos()
                val rawDetections = activeDetector.detect(lb.bitmap)
                val detectMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000.0
                InferenceProfiler.record(
                    context = getApplication(),
                    detectMs = detectMs,
                    nDetections = rawDetections.size,
                    sceneLocked = sceneLocked
                )
                val detections = rawDetections
                    .map { it.copy(boundingBox = FrameCropper.toVisibleNormalized(it.boundingBox, lb)) }
                    // Asymmetric floor: toxic @ 0.18, other classes @ 0.20.
                    .filter { it.confidence >= MushroomRegistry.displayThresholdFor(it.species.classId) }
                lb.bitmap.recycle()

                // Tracker applies IoU matching + SAFE temporal vote (4/5 frames).
                val tracked = tracker.update(detections, now)
                currentDetections = tracked

                synchronized(frameLock) {
                    latestFrame?.let { if (it !== upright) it.recycle() }
                    latestFrame = upright
                }

                _uiState.value = MushroomUiState.Scanning(tracked, tracker.isSceneLocked())
            } catch (t: Throwable) {
                Log.e(TAG, "Inference failed for frame", t)
                upright.recycle()
            } finally {
                inferenceMutex.unlock()
            }
        }
    }

    /** Crop and persist a specific tapped detection. */
    fun saveDetection(detection: TrackedDetection) {
        val aspect = viewAspectRatio
        val crop: Bitmap? = synchronized(frameLock) {
            latestFrame?.let { FrameCropper.cropNormalizedBox(it, detection.box, aspect) }
        }
        if (crop == null) {
            emitSave(false, R.string.save_failed)
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            try {
                repository.save(
                    speciesName = detection.species.name,
                    toxicityLabel = detection.species.toxicity.labelPl,
                    confidence = detection.confidence,
                    crop = crop
                )
                emitSave(true, R.string.save_success)
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to save find", t)
                emitSave(false, R.string.save_failed)
            } finally {
                crop.recycle()
            }
        }
    }

    /** Save the best current detection (frozen first, then highest confidence). */
    fun saveCurrentBest() {
        val best = currentDetections
            .sortedWith(
                compareByDescending<TrackedDetection> { it.frozen }
                    .thenByDescending { it.confidence }
            )
            .firstOrNull()

        if (best == null) {
            emitSave(false, R.string.save_no_target)
            return
        }
        saveDetection(best)
    }

    fun retry() {
        if (detector == null) {
            initialiseDetector()
        } else {
            _uiState.value = MushroomUiState.Scanning()
        }
    }

    private fun emitSave(success: Boolean, @StringRes messageRes: Int) {
        val message = getApplication<Application>().getString(messageRes)
        _saveEvents.tryEmit(SaveEvent(message, success))
    }

    override fun onCleared() {
        super.onCleared()
        detector?.close()
        detector = null
        synchronized(frameLock) {
            latestFrame?.recycle()
            latestFrame = null
        }
    }

    companion object {
        private const val TAG = "MushroomViewModel"

        /** Inference cadence: 4 FPS (one frame every 250 ms). */
        private const val INFERENCE_INTERVAL_MS = 250L

        /** Slow heartbeat used while the scene is locked, to conserve battery. */
        private const val LOCK_HEARTBEAT_MS = 1_500L
    }
}
