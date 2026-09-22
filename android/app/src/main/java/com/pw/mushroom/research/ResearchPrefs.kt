package com.pw.mushroom.research

import android.content.Context
import android.content.SharedPreferences

/**
 * Lightweight wrapper around [SharedPreferences] for Research Mode state.
 *
 * Research Mode is a hidden, debug-oriented feature for in-situ field data
 * collection. When enabled, a "Capture" FAB appears on the camera screen
 * and each press saves a full diagnostic bundle (full-frame image, annotated
 * screenshot via PixelCopy, detection metadata JSON, and a CSV log row) to
 * public external storage for later analysis on a laptop.
 *
 * The toggle persists across app restarts so a field session is never lost
 * if the OS reclaims the process.
 */
class ResearchPrefs(context: Context) {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /** Whether Research Mode is currently enabled (persisted). */
    var isEnabled: Boolean
        get() = prefs.getBoolean(KEY_ENABLED, false)
        set(value) = prefs.edit().putBoolean(KEY_ENABLED, value).apply()

    /** Running capture counter for the current session (reset on new session). */
    var captureCounter: Int
        get() = prefs.getInt(KEY_COUNTER, 0)
        set(value) = prefs.edit().putInt(KEY_COUNTER, value).apply()

    /** ISO-formatted session start time (used as folder name). */
    var sessionId: String?
        get() = prefs.getString(KEY_SESSION_ID, null)
        set(value) = prefs.edit().putString(KEY_SESSION_ID, value).apply()

    /** Reset session state when starting a new session. */
    fun startNewSession(sessionId: String) {
        prefs.edit()
            .putString(KEY_SESSION_ID, sessionId)
            .putInt(KEY_COUNTER, 0)
            .apply()
    }

    /** Clear session state when ending a session. */
    fun endSession() {
        prefs.edit()
            .remove(KEY_SESSION_ID)
            .putInt(KEY_COUNTER, 0)
            .apply()
    }

    companion object {
        private const val PREFS_NAME = "research_mode_prefs"
        private const val KEY_ENABLED = "research_enabled"
        private const val KEY_COUNTER = "capture_counter"
        private const val KEY_SESSION_ID = "session_id"
    }
}
