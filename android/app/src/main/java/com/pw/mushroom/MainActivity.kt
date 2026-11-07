package com.pw.mushroom

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.pw.mushroom.ui.HistoryScreen
import com.pw.mushroom.ui.MushroomCameraScreen
import com.pw.mushroom.ui.theme.MushroomTheme

/** Single-activity host for the Compose UI. */
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MushroomTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    MushroomApp()
                }
            }
        }
    }
}

@Composable
private fun MushroomApp() {
    val navController = rememberNavController()
    NavHost(navController = navController, startDestination = ROUTE_CAMERA) {
        composable(ROUTE_CAMERA) {
            MushroomCameraScreen(
                onOpenHistory = { navController.navigate(ROUTE_HISTORY) }
            )
        }
        composable(ROUTE_HISTORY) {
            HistoryScreen(onBack = { navController.popBackStack() })
        }
    }
}

private const val ROUTE_CAMERA = "camera"
private const val ROUTE_HISTORY = "history"
