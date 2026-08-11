package com.pw.mushroom.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

/** App-wide Room database holding logged mushroom finds. */
@Database(entities = [MushroomFind::class], version = 1, exportSchema = false)
abstract class MushroomDatabase : RoomDatabase() {

    abstract fun findDao(): MushroomFindDao

    companion object {
        @Volatile
        private var instance: MushroomDatabase? = null

        /** Lazily create (or return) the singleton database instance. */
        fun get(context: Context): MushroomDatabase =
            instance ?: synchronized(this) {
                instance ?: Room.databaseBuilder(
                    context.applicationContext,
                    MushroomDatabase::class.java,
                    "mushroom_finds.db"
                ).build().also { instance = it }
            }
    }
}
