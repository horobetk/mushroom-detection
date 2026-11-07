package com.pw.mushroom.research

import android.content.Context
import android.content.SharedPreferences

/** Persisted Research Mode toggle and session counters. */
class ResearchPrefs(context: Context) {

    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    var isEnabled: Boolean
        get() = prefs.getBoolean(KEY_ENABLED, false)
        set(value) = prefs.edit().putBoolean(KEY_ENABLED, value).apply()

    var captureCounter: Int
        get() = prefs.getInt(KEY_COUNTER, 0)
        set(value) = prefs.edit().putInt(KEY_COUNTER, value).apply()

    var sessionId: String?
        get() = prefs.getString(KEY_SESSION_ID, null)
        set(value) = prefs.edit().putString(KEY_SESSION_ID, value).apply()

    fun startNewSession(sessionId: String) {
        prefs.edit()
            .putString(KEY_SESSION_ID, sessionId)
            .putInt(KEY_COUNTER, 0)
            .apply()
    }

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
