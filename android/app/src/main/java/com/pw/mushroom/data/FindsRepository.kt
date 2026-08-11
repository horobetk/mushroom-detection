package com.pw.mushroom.data

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

/**
 * Persists mushroom finds: writes the cropped JPEG into app-private internal
 * storage (never the public gallery) and stores the row in Room.
 */
class FindsRepository(context: Context) {

    private val appContext = context.applicationContext
    private val dao = MushroomDatabase.get(appContext).findDao()

    /** All saved finds, newest first, as a reactive stream. */
    val finds: Flow<List<MushroomFind>> = dao.observeAll()

    /** All saved finds, newest first (alias for readability at call sites). */
    fun getAllFinds(): Flow<List<MushroomFind>> = finds

    /**
     * Save one find. The [crop] bitmap is compressed to JPEG under
     * filesDir/finds and the metadata row is inserted.
     *
     * @return the persisted [MushroomFind] with its generated id.
     */
    suspend fun save(
        speciesName: String,
        toxicityLabel: String,
        confidence: Float,
        crop: Bitmap
    ): MushroomFind = withContext(Dispatchers.IO) {
        val dir = File(appContext.filesDir, FINDS_DIR).apply { mkdirs() }
        val file = File(dir, "find_${UUID.randomUUID()}.jpg")

        FileOutputStream(file).use { out ->
            crop.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
        }

        val find = MushroomFind(
            speciesName = speciesName,
            toxicityLabel = toxicityLabel,
            confidence = confidence,
            timestamp = System.currentTimeMillis(),
            imagePath = file.absolutePath
        )
        val id = dao.insert(find)
        Log.i(TAG, "Saved find id=$id species=$speciesName path=${file.absolutePath}")
        find.copy(id = id)
    }

    /** Delete a find row and its backing image file (best-effort). */
    suspend fun delete(find: MushroomFind) = withContext(Dispatchers.IO) {
        dao.deleteById(find.id)
        runCatching { File(find.imagePath).delete() }
    }

    companion object {
        private const val TAG = "FindsRepository"
        private const val FINDS_DIR = "finds"
        private const val JPEG_QUALITY = 92
    }
}
