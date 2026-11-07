plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "2.0.20" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.20" apply false
    // KSP version must match the Kotlin version.
    id("com.google.devtools.ksp") version "2.0.20-1.0.24" apply false
}
