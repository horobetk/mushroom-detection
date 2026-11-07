package com.pw.mushroom.data

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/** Saved mushroom find row. */
@Entity(tableName = "mushroom_finds")
data class MushroomFind(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val speciesName: String,
    val toxicityLabel: String,
    val confidence: Float,
    /** Whether the 4/5 vote was reached at save time; gates the green badge in history. */
    @ColumnInfo(defaultValue = "1")
    val voteConfirmed: Boolean = false,
    val timestamp: Long,
    val imagePath: String
)

