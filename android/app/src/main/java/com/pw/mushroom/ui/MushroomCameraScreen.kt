package com.pw.mushroom.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.History
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.FloatingActionButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.pw.mushroom.R
import com.pw.mushroom.camera.FrameCropper
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

/**
 * Top-level screen: CameraX preview, continuous AR detection overlay, and a
 * capture button. All ML state lives in [MushroomViewModel]. Save results are
 * surfaced as Polish snackbars.
 *
 * Research Mode (hidden): activated by 5 rapid taps on the status hint. Adds
 * a "Capture" FAB and a red "REC" indicator. Each capture saves a full data
 * bundle (full-frame, PixelCopy screenshot, JSON metadata) to Documents/.
 */
@Composable
fun MushroomCameraScreen(
    onOpenHistory: () -> Unit,
    viewModel: MushroomViewModel = viewModel()
) {
    val context = LocalContext.current
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted -> hasCameraPermission = granted }

    Box(modifier = Modifier.fillMaxSize()) {
        if (hasCameraPermission) {
            CameraContent(viewModel = viewModel, onOpenHistory = onOpenHistory)
        } else {
            PermissionRequest(onRequest = {
                permissionLauncher.launch(Manifest.permission.CAMERA)
            })
        }
    }
}

@Composable
private fun CameraContent(viewModel: MushroomViewModel, onOpenHistory: () -> Unit) {
    val uiState by viewModel.uiState.collectAsState()
    val researchEnabled by viewModel.researchEnabled.collectAsState()
    val captureCount by viewModel.captureCount.collectAsState()
    val snackbarHostState = remember { SnackbarHostState() }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // Flash overlay state for capture feedback.
    var showFlash by remember { mutableStateOf(false) }

    // Full-resolution still capture use-case (Research Mode only). Bound alongside
    // the preview/analysis so we can grab a clean high-res photo on demand.
    val imageCapture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
    }
    val captureExecutor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { captureExecutor.shutdown() } }

    LaunchedEffect(Unit) {
        viewModel.saveEvents.collect { event ->
            snackbarHostState.showSnackbar(event.message)
        }
    }

    // Get the Activity window for PixelCopy.
    val activity = context as? android.app.Activity
    val window = activity?.window

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) },
        floatingActionButton = {
            if (uiState is MushroomUiState.Scanning) {
                Column(
                    horizontalAlignment = Alignment.End,
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Research capture button (only visible in Research Mode).
                    if (researchEnabled) {
                        BadgedBox(
                            badge = {
                                if (captureCount > 0) {
                                    Badge(
                                        containerColor = Color(0xFFFF5722),
                                        contentColor = Color.White
                                    ) {
                                        Text("#$captureCount")
                                    }
                                }
                            }
                        ) {
                            FloatingActionButton(
                                onClick = {
                                    // Haptic feedback (best-effort; never crash the capture).
                                    try {
                                        val vibrator = context.getSystemService(Vibrator::class.java)
                                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                                            vibrator?.vibrate(
                                                VibrationEffect.createOneShot(
                                                    50, VibrationEffect.DEFAULT_AMPLITUDE
                                                )
                                            )
                                        }
                                    } catch (_: Exception) {
                                        // Ignore: haptics are non-essential.
                                    }
                                    // Visual flash.
                                    showFlash = true
                                    scope.launch {
                                        delay(120)
                                        showFlash = false
                                    }
                                    // Grab a full-resolution still, then save the bundle.
                                    // On any capture error we still save the analysis-frame bundle.
                                    imageCapture.takePicture(
                                        captureExecutor,
                                        object : ImageCapture.OnImageCapturedCallback() {
                                            override fun onCaptureSuccess(image: ImageProxy) {
                                                val bmp = image.toUprightBitmap()
                                                image.close()
                                                viewModel.researchCapture(window, bmp)
                                            }

                                            override fun onError(exc: ImageCaptureException) {
                                                viewModel.researchCapture(window, null)
                                            }
                                        }
                                    )
                                },
                                containerColor = Color(0xFFFF9800),
                                contentColor = Color.White,
                                elevation = FloatingActionButtonDefaults.elevation(6.dp)
                            ) {
                                Icon(
                                    Icons.Filled.CameraAlt,
                                    contentDescription = "Research Capture"
                                )
                            }
                        }
                    }

                    // Standard save button.
                    ExtendedFloatingActionButton(
                        onClick = { viewModel.saveCurrentBest() },
                        icon = { Icon(Icons.Filled.Add, contentDescription = null) },
                        text = { Text(stringResource(R.string.action_save)) }
                    )
                }
            }
        },
        containerColor = Color.Black
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .onSizeChanged { size ->
                    if (size.height > 0) {
                        viewModel.setViewAspectRatio(size.width.toFloat() / size.height.toFloat())
                    }
                }
        ) {
            CameraPreview(
                onFrame = viewModel::onFrame,
                imageCapture = imageCapture,
                modifier = Modifier.fillMaxSize()
            )

            when (val state = uiState) {
                is MushroomUiState.Scanning -> {
                    DetectionOverlay(
                        detections = state.detections,
                        onTapDetection = viewModel::saveDetection,
                        modifier = Modifier.fillMaxSize()
                    )
                    StatusHint(
                        locked = state.locked,
                        onResearchToggle = { viewModel.toggleResearchMode() },
                        modifier = Modifier
                            .align(Alignment.TopCenter)
                            .padding(innerPadding)
                            .padding(top = 16.dp)
                    )
                }

                is MushroomUiState.Initializing -> {
                    LoadingIndicator(modifier = Modifier.align(Alignment.Center))
                }

                is MushroomUiState.Error -> {
                    ErrorPanel(
                        message = state.message,
                        onRetry = viewModel::retry,
                        modifier = Modifier.align(Alignment.Center)
                    )
                }
            }

            // Research Mode indicator ("REC" dot).
            if (researchEnabled) {
                ResearchIndicator(
                    captureCount = captureCount,
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(innerPadding)
                        .padding(top = 16.dp, end = 16.dp)
                )
            }

            // Capture flash overlay.
            AnimatedVisibility(
                visible = showFlash,
                enter = fadeIn(),
                exit = fadeOut()
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.White.copy(alpha = 0.4f))
                )
            }

            FilledIconButton(
                onClick = onOpenHistory,
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = Color.Black.copy(alpha = 0.5f),
                    contentColor = Color.White
                ),
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(innerPadding)
                    .padding(start = 19.dp, bottom = 19.dp)
                    .size(48.dp)
            ) {
                Icon(
                    imageVector = Icons.Filled.History,
                    contentDescription = stringResource(R.string.history_open),
                    modifier = Modifier.size(29.dp)
                )
            }
        }
    }
}

/**
 * CameraX preview + asynchronous ImageAnalysis. Frames are delivered in
 * RGBA_8888 and converted to a rotation-corrected, full-resolution upright
 * bitmap on a background executor before being forwarded to [onFrame], which
 * takes ownership of the bitmap. [onFrame] also receives the frame arrival time
 * ([SystemClock.elapsedRealtimeNanos]) and the CameraX sensor timestamp.
 */
@Composable
private fun CameraPreview(
    onFrame: (Bitmap, Long, Long) -> Unit,
    imageCapture: ImageCapture,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }

    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    DisposableEffect(Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val cameraProvider = cameraProviderFuture.get()

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        analysis.setAnalyzer(analysisExecutor) { imageProxy ->
            try {
                val arrivedNs = SystemClock.elapsedRealtimeNanos()
                onFrame(
                    FrameCropper.toUprightBitmap(imageProxy),
                    arrivedNs,
                    imageProxy.imageInfo.timestamp
                )
            } finally {
                imageProxy.close()
            }
        }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis,
                imageCapture
            )
        } catch (t: Throwable) {
            // Binding can fail if the lifecycle is not ready; it rebinds on recomposition.
        }

        onDispose {
            cameraProvider.unbindAll()
            analysisExecutor.shutdown()
        }
    }

    AndroidView(factory = { previewView }, modifier = modifier)
}

/**
 * Status hint with a hidden activation gesture: 5 rapid taps within 3 seconds
 * toggles Research Mode. Shows a Toast confirmation.
 */
@Composable
private fun StatusHint(
    locked: Boolean,
    onResearchToggle: () -> Unit,
    modifier: Modifier = Modifier
) {
    val text = if (locked) {
        stringResource(R.string.status_locked)
    } else {
        stringResource(R.string.scanning_hint)
    }

    val context = LocalContext.current
    var tapCount by remember { mutableIntStateOf(0) }
    var firstTapTime by remember { mutableStateOf(0L) }

    Surface(
        modifier = modifier.clickable(
            indication = null,
            interactionSource = remember { MutableInteractionSource() }
        ) {
            val now = System.currentTimeMillis()
            if (now - firstTapTime > 3000) {
                tapCount = 1
                firstTapTime = now
            } else {
                tapCount++
            }
            if (tapCount >= 5) {
                tapCount = 0
                onResearchToggle()
                Toast.makeText(
                    context,
                    "Research Mode toggled",
                    Toast.LENGTH_SHORT
                ).show()
            }
        },
        color = Color.Black.copy(alpha = 0.5f),
        shape = RoundedCornerShape(12.dp)
    ) {
        Text(
            text = text,
            color = Color.White,
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
        )
    }
}

/** Red pulsing "REC" indicator shown in Research Mode. */
@Composable
private fun ResearchIndicator(captureCount: Int, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier,
        color = Color(0xCCF44336),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .background(Color.Red, CircleShape)
            )
            Spacer(Modifier.width(6.dp))
            Text(
                text = if (captureCount > 0) "REC · #$captureCount" else "REC",
                color = Color.White,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
private fun LoadingIndicator(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        CircularProgressIndicator(color = Color.White)
        Text(
            text = stringResource(R.string.loading_model),
            color = Color.White,
            fontSize = 14.sp,
            modifier = Modifier.padding(top = 16.dp)
        )
    }
}

@Composable
private fun PermissionRequest(onRequest: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = stringResource(R.string.permission_title),
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface
        )
        Text(
            text = stringResource(R.string.permission_rationale),
            fontSize = 15.sp,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
            modifier = Modifier.padding(top = 12.dp, bottom = 24.dp)
        )
        Button(onClick = onRequest) {
            Text(stringResource(R.string.permission_grant))
        }
    }
}

/**
 * Decode a JPEG [ImageProxy] from [ImageCapture] into an upright [Bitmap],
 * applying the sensor rotation so the saved photo is oriented correctly.
 */
private fun ImageProxy.toUprightBitmap(): Bitmap {
    val bmp = toBitmap()
    val degrees = imageInfo.rotationDegrees
    if (degrees == 0) return bmp
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
    if (rotated !== bmp) bmp.recycle()
    return rotated
}

@Composable
private fun ErrorPanel(
    message: String,
    onRetry: () -> Unit,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier.padding(24.dp),
        color = MaterialTheme.colorScheme.surface,
        shape = RoundedCornerShape(20.dp),
        tonalElevation = 4.dp
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = stringResource(R.string.error_generic),
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = message,
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
                modifier = Modifier.padding(vertical = 12.dp)
            )
            Button(onClick = onRetry) {
                Text(stringResource(R.string.error_retry))
            }
        }
    }
}

