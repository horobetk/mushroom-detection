package com.pw.mushroom.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.pw.mushroom.data.FindsRepository
import com.pw.mushroom.data.MushroomFind
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/**
 * Backs the history screen: streams saved finds from Room and handles deletion
 * (row + backing image file).
 */
class HistoryViewModel(application: Application) : AndroidViewModel(application) {

    private val repository = FindsRepository(application)

    /** All saved finds, newest first. */
    val finds: StateFlow<List<MushroomFind>> = repository.getAllFinds()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = emptyList()
        )

    /** Delete a find and its stored image. */
    fun deleteFind(find: MushroomFind) {
        viewModelScope.launch {
            repository.delete(find)
        }
    }
}
