package com.pw.mushroom.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

/** Data-access object for [MushroomFind] rows. */
@Dao
interface MushroomFindDao {

    /** Insert a new find and return its generated row id. */
    @Insert
    suspend fun insert(find: MushroomFind): Long

    /** Observe all finds, newest first. */
    @Query("SELECT * FROM mushroom_finds ORDER BY timestamp DESC")
    fun observeAll(): Flow<List<MushroomFind>>

    /** Delete a find by id (does not remove the backing image file). */
    @Query("DELETE FROM mushroom_finds WHERE id = :id")
    suspend fun deleteById(id: Long)
}
