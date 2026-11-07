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

/** State holder for the history screen. */
class HistoryViewModel(application: Application) : AndroidViewModel(application) {

    private val repository = FindsRepository(application)

    val finds: StateFlow<List<MushroomFind>> = repository.getAllFinds()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = emptyList()
        )

    fun deleteFind(find: MushroomFind) {
        viewModelScope.launch {
            repository.delete(find)
        }
    }
}
