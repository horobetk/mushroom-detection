package com.pw.mushroom.data

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * A single logged mushroom find persisted in the local Room database.
 *
 * @property id             Auto-generated row id.
 * @property speciesName    Scientific name from the registry at save time.
 * @property toxicityLabel  Polish edibility label (e.g. "Trujący").
 * @property confidence     Detection score in the range 0.0 .. 1.0.
 * @property voteConfirmed  Whether the tracker's 4/5 temporal consensus was
 *                          reached at save time. When false, the history screen
 *                          must not display the green "Jadalny" badge even if
 *                          the confidence exceeds [SAFE_THRESHOLD].
 * @property timestamp      Epoch milliseconds when the find was saved.
 * @property imagePath      Absolute path to the cropped JPEG in internal storage.
 */
@Entity(tableName = "mushroom_finds")
data class MushroomFind(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val speciesName: String,
    val toxicityLabel: String,
    val confidence: Float,
    @ColumnInfo(defaultValue = "1")
    val voteConfirmed: Boolean = true,
    val timestamp: Long,
    val imagePath: String
)

