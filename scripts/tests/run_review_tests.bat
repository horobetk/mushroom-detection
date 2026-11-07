@echo off
rem Per-category metrics (PyTorch, GPU) and H2 CFPR on the TFLite FP16 artefact (CPU).
rem The TFLite file must be the same model as android/app/src/main/assets/best_float16.tflite
rem (40 465 695 bytes); its size is printed at start-up.
setlocal
cd /d "%~dp0"

set TFLITE=E:\Kiril_Horobets\mushroom_data\runs\fine_147classes_v2\weights\best_saved_model_fp16.tflite
if not "%~1"=="" set TFLITE=%~1

python test_category_map.py > category_map.log 2>&1
findstr /b /l /c:"[OK]" category_map.log

python test_tflite_cfpr.py --tflite "%TFLITE%" > tflite_cfpr.log 2>&1
findstr /b /l /c:"[OK]" tflite_cfpr.log

endlocal
