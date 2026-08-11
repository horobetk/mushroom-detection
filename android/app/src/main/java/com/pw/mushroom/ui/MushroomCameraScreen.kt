package com.pw.mushroom.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.History
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilledIconButton
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
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
import java.util.concurrent.Executors

/**
 * Top-level screen: CameraX preview, continuous AR detection overlay, and a
 * capture button. All ML state lives in [MushroomViewModel]. Save results are
 * surfaced as Polish snackbars.
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
    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(Unit) {
        viewModel.saveEvents.collect { event ->
            snackbarHostState.showSnackbar(event.message)
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) },
        floatingActionButton = {
            if (uiState is MushroomUiState.Scanning) {
                ExtendedFloatingActionButton(
                    onClick = { viewModel.saveCurrentBest() },
                    icon = { Icon(Icons.Filled.Add, contentDescription = null) },
                    text = { Text(stringResource(R.string.action_save)) }
                )
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
 * takes ownership of the bitmap.
 */
@Composable
private fun CameraPreview(
    onFrame: (Bitmap) -> Unit,
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
                onFrame(FrameCropper.toUprightBitmap(imageProxy))
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
                analysis
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

@Composable
private fun StatusHint(locked: Boolean, modifier: Modifier = Modifier) {
    val text = if (locked) {
        stringResource(R.string.status_locked)
    } else {
        stringResource(R.string.scanning_hint)
    }
    Surface(
        modifier = modifier,
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
