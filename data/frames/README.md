# Mushroom Dataset - High Resolution

##  Final Statistics

| Metric | Value |
|--------|-------|
| **Total images** | **5,720** |
| Original frames | 715 |
| Augmented | 5,005 |
| Rejected (blur) | ~50 |
| Processing time | ~6 minutes |

## ✅ Quality Validation

Sample images show:
- ✅ Clear mushroom boundaries
- ✅ Visible texture details
- ✅ Multiple small objects distinguishable
- ✅ Sufficient resolution for YOLO detection

##  Source

- 7 video recordings (~127s total)
- Original: 4K (3840×2160 / 2160×3840)
- Extraction: every 5th frame
- Blur filtering: Laplacian threshold = 30

##  Augmentation

Each original frame → 7 variants:
- Brightness: ×2 (0.7, 1.3)
- Contrast: ×2 (0.8, 1.2)
- Horizontal flip: ×1
- Rotation: ×2 (-5°, +5°)

##  Structure
```
frames/
├── original/      715 high-quality frames (1920×1920)
├── augmented/     5,005 augmented variants
└── statistics.json
```

