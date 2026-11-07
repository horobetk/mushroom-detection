package com.pw.mushroom.ui

import android.app.Application
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.PixelCopy
import android.view.Window
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
import com.pw.mushroom.research.ResearchLogger
import com.pw.mushroom.research.ResearchPrefs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

/** Live detection pipeline, saving and Research Mode for the camera screen. */
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

    // Written by the inference coroutine; the tracker itself is not thread-safe.
    @Volatile
    private var lastSceneLocked = false

    // Guards [latestFrame], shared between the analyzer thread and save requests.
    private val frameLock = Any()
    private var latestFrame: Bitmap? = null

    @Volatile
    private var currentDetections: List<TrackedDetection> = emptyList()

    // Preview width / height, used to match the FILL_CENTER visible region.
    @Volatile
    private var viewAspectRatio: Float = 1f

    val researchPrefs = ResearchPrefs(application)

    private val _researchEnabled = MutableStateFlow(researchPrefs.isEnabled)
    val researchEnabled: StateFlow<Boolean> = _researchEnabled.asStateFlow()

    private val _captureCount = MutableStateFlow(researchPrefs.captureCounter)
    val captureCount: StateFlow<Int> = _captureCount.asStateFlow()

    @Volatile
    private var lastDetectMs: Double = 0.0

    /** EMA of the incoming camera frame rate. */
    @Volatile
    private var smoothedFps: Double = 4.0
    private var prevFrameTimestamp: Long = 0L

    fun setViewAspectRatio(ratio: Float) {
        if (ratio > 0f) viewAspectRatio = ratio
    }

    init {
        initialiseDetector()
        syncResearchFlag()
    }

    private fun syncResearchFlag() {
        val enabled = _researchEnabled.value
        MushroomDetector.collectTopN = enabled
        if (enabled && !ResearchLogger.isSessionActive) {
            val sessionId = ResearchLogger.startSession()
            if (sessionId != null) {
                researchPrefs.startNewSession(sessionId)
            }
        }
    }

    fun toggleResearchMode() {
        val newState = !_researchEnabled.value
        researchPrefs.isEnabled = newState
        _researchEnabled.value = newState
        MushroomDetector.collectTopN = newState

        if (newState) {
            if (!ResearchLogger.isSessionActive) {
                val sessionId = ResearchLogger.startSession()
                if (sessionId != null) {
                    researchPrefs.startNewSession(sessionId)
                    _captureCount.value = 0
                }
            }
        } else {
            ResearchLogger.endSession()
            researchPrefs.endSession()
            _captureCount.value = 0
        }
    }

    private fun initialiseDetector() {
        _uiState.value = MushroomUiState.Initializing
        viewModelScope.launch(Dispatchers.Default) {
            try {
                detector = MushroomDetector(getApplication())
                _uiState.value = MushroomUiState.Scanning()
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to initialise detector", t)
                _uiState.value = MushroomUiState.Error(t.message ?: "Nieznany błąd")
            }
        }
    }

    /** Takes ownership of [upright]: it is recycled or kept as the latest frame. */
    fun onFrame(
        upright: Bitmap,
        arrivedNs: Long = SystemClock.elapsedRealtimeNanos(),
        sensorTimestampNs: Long = 0L
    ) {
        val activeDetector = detector
        if (activeDetector == null) {
            upright.recycle()
            return
        }

        val now = SystemClock.elapsedRealtime()

        if (prevFrameTimestamp > 0L) {
            val dtMs = now - prevFrameTimestamp
            if (dtMs > 0) {
                val instantFps = 1000.0 / dtMs
                smoothedFps = smoothedFps * 0.8 + instantFps * 0.2
            }
        }
        prevFrameTimestamp = now

        val sceneLocked = lastSceneLocked
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
        val acceptedNs = SystemClock.elapsedRealtimeNanos()
        InferenceProfiler.noteFrame(now, arrivedNs)

        val aspect = viewAspectRatio
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val tStart = SystemClock.elapsedRealtimeNanos()
                val lb = FrameCropper.letterbox(upright, aspect, activeDetector.inputResolution)
                val t0 = SystemClock.elapsedRealtimeNanos()
                val rawDetections = activeDetector.detect(lb.bitmap)
                val t1 = SystemClock.elapsedRealtimeNanos()
                val detectMs = (t1 - t0) / 1_000_000.0
                lastDetectMs = detectMs
                val detections = rawDetections
                    .map { it.copy(boundingBox = FrameCropper.toVisibleNormalized(it.boundingBox, lb)) }
                    // Asymmetric floor: toxic @ 0.18, other classes @ 0.20.
                    .filter { it.confidence >= MushroomRegistry.displayThresholdFor(it.species.classId) }
                lb.bitmap.recycle()

                val tracked = tracker.update(detections, now)
                val locked = tracker.isSceneLocked()
                lastSceneLocked = locked
                currentDetections = tracked

                synchronized(frameLock) {
                    latestFrame?.let { if (it !== upright) it.recycle() }
                    latestFrame = upright
                }

                _uiState.value = MushroomUiState.Scanning(tracked, locked)
                val publishedNs = SystemClock.elapsedRealtimeNanos()

                InferenceProfiler.record(
                    context = getApplication(),
                    timing = InferenceProfiler.FrameTiming(
                        sensorNs = sensorTimestampNs,
                        arrivedNs = arrivedNs,
                        acceptedNs = acceptedNs,
                        startNs = tStart,
                        detectStartNs = t0,
                        detectEndNs = t1,
                        publishedNs = publishedNs
                    ),
                    nDetections = rawDetections.size,
                    sceneLocked = sceneLocked
                )
                InferenceProfiler.recordResponses(getApplication(), tracked, publishedNs)
            } catch (t: Throwable) {
                Log.e(TAG, "Inference failed for frame", t)
                upright.recycle()
            } finally {
                inferenceMutex.unlock()
            }
        }
    }

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
                    voteConfirmed = detection.voteConfirmed,
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

    /** Save a Research Mode bundle; takes ownership of [fullResFrame]. */
    fun researchCapture(window: Window?, fullResFrame: Bitmap? = null) {
        if (!_researchEnabled.value || !ResearchLogger.isSessionActive) {
            fullResFrame?.recycle()
            return
        }

        val detections = currentDetections
        val frame: Bitmap? = synchronized(frameLock) {
            latestFrame?.let { it.copy(it.config ?: Bitmap.Config.ARGB_8888, false) }
        }
        if (frame == null) {
            fullResFrame?.recycle()
            _saveEvents.tryEmit(SaveEvent("Tryb badawczy: brak klatki", false))
            return
        }

        val captureId = _captureCount.value + 1
        _captureCount.value = captureId
        researchPrefs.captureCounter = captureId

        val inferenceMs = lastDetectMs
        val fps = smoothedFps
        val gpuDelegate = detector?.isGpuEnabled ?: false

        viewModelScope.launch(Dispatchers.IO) {
            val screenBitmap = if (window != null) {
                try {
                    captureScreen(window)
                } catch (t: Throwable) {
                    Log.w(TAG, "PixelCopy failed, proceeding without screenshot", t)
                    null
                }
            } else null

            val path = ResearchLogger.capture(
                captureId = captureId,
                fullFrame = frame,
                fullResFrame = fullResFrame,
                screenBitmap = screenBitmap,
                detections = detections,
                inferenceMs = inferenceMs,
                fps = fps,
                gpuDelegate = gpuDelegate,
                context = getApplication()
            )

            frame.recycle()
            fullResFrame?.recycle()
            screenBitmap?.recycle()

            if (path != null) {
                _saveEvents.tryEmit(SaveEvent("Zapisano pakiet #$captureId", true))
            } else {
                _saveEvents.tryEmit(SaveEvent("Nie udało się zapisać pakietu #$captureId", false))
            }
        }
    }

    private suspend fun captureScreen(window: Window): Bitmap? =
        suspendCancellableCoroutine { cont ->
            val view = window.decorView
            val bitmap = Bitmap.createBitmap(view.width, view.height, Bitmap.Config.ARGB_8888)
            val handler = Handler(Looper.getMainLooper())

            try {
                PixelCopy.request(window, bitmap, { result ->
                    if (result == PixelCopy.SUCCESS) {
                        cont.resume(bitmap)
                    } else {
                        Log.w(TAG, "PixelCopy returned result=$result")
                        bitmap.recycle()
                        cont.resume(null)
                    }
                }, handler)
            } catch (t: Throwable) {
                bitmap.recycle()
                cont.resume(null)
            }
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
        MushroomDetector.collectTopN = false
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
