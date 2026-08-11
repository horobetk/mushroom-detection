package com.pw.mushroom.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Minimalist, nature-inspired palette (deep greens + neutral surfaces).
private val ForestGreen = Color(0xFF2E7D53)
private val ForestGreenDark = Color(0xFF1B5E3F)
private val SandNeutral = Color(0xFFF4F1EA)
private val CharcoalSurface = Color(0xFF14181C)

// Semantic status colours used by the result banner.
val StatusEdible = Color(0xFF2E7D32)
val StatusPoisonous = Color(0xFFE65100)
val StatusDeadly = Color(0xFFC62828)
val StatusInedible = Color(0xFF757575)
val StatusCaution = Color(0xFFF9A825)

private val LightColors = lightColorScheme(
    primary = ForestGreen,
    onPrimary = Color.White,
    secondary = ForestGreenDark,
    background = SandNeutral,
    surface = Color.White,
    onSurface = Color(0xFF14181C)
)

private val DarkColors = darkColorScheme(
    primary = ForestGreen,
    onPrimary = Color.White,
    secondary = ForestGreenDark,
    background = CharcoalSurface,
    surface = Color(0xFF1C2126),
    onSurface = Color(0xFFECEFF1)
)

@Composable
fun MushroomTheme(
    useDarkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = if (useDarkTheme) DarkColors else LightColors,
        typography = Typography(),
        content = content
    )
}
