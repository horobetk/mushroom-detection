plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.pw.mushroom"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.pw.mushroom"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        vectorDrawables { useSupportLibrary = true }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    // Do not compress the TFLite model; it must be memory-mapped at runtime.
    androidResources {
        noCompress += "tflite"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // --- Jetpack Compose (BOM keeps all Compose artifacts on one aligned version) ---
    val composeBom = platform("androidx.compose:compose-bom:2024.09.02")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.6")

    // --- Navigation (camera <-> history) ---
    implementation("androidx.navigation:navigation-compose:2.8.0")

    // --- Coil (async loading of saved crop thumbnails from internal storage) ---
    implementation("io.coil-kt:coil-compose:2.7.0")

    // --- CameraX (preview + asynchronous frame analysis) ---
    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // --- Kotlin Coroutines (off-main-thread inference) ---
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // --- Room (local SQLite persistence for saved mushroom finds) ---
    val roomVersion = "2.6.1"
    implementation("androidx.room:room-runtime:$roomVersion")
    implementation("androidx.room:room-ktx:$roomVersion")
    ksp("androidx.room:room-compiler:$roomVersion")

    // --- TensorFlow Lite (low-level Interpreter API for a custom YOLO decoder) ---
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    // GPU delegate for hardware-accelerated inference (OpenCL / OpenGL backend).
    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
    // GPU delegate API types (CompatibilityList, GpuDelegateFactory.Options).
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.14.0")

    debugImplementation("androidx.compose.ui:ui-tooling")
}
