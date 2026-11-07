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

/** Stores finds in Room with their crops in app-private storage. */
class FindsRepository(context: Context) {

    private val appContext = context.applicationContext
    private val dao = MushroomDatabase.get(appContext).findDao()

    /** All saved finds, newest first, as a reactive stream. */
    val finds: Flow<List<MushroomFind>> = dao.observeAll()

    fun getAllFinds(): Flow<List<MushroomFind>> = finds

    suspend fun save(
        speciesName: String,
        toxicityLabel: String,
        confidence: Float,
        voteConfirmed: Boolean,
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
            voteConfirmed = voteConfirmed,
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
