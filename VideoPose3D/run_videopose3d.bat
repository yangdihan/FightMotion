@echo off
setlocal

:: Define variables for easy tuning
set VENV_PATH=..\venv_cuda\Scripts\activate
set CHECKPOINT=checkpoint
set MODEL_CFG=COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml
set PRETRAINED_MODEL=pretrained_h36m_detectron_coco.bin

set INPUT_DIR=D:\Documents\devs\fight_motion\VideoPose3D\dataset\input
set OUTPUT_DIR=D:\Documents\devs\fight_motion\VideoPose3D\dataset\output

set VIDEO_FILE=fighter_0
set RENDER_FILE=fighter_0.mp4

set CUSTOM_DATASET=fighters

:: Activate virtual environment
call %VENV_PATH%

:: Run inference
@REM cd inference
@REM python infer_video_d2.py ^
@REM --cfg %MODEL_CFG% ^
@REM --image-ext mp4 "%INPUT_DIR%" ^
@REM --output-dir "%OUTPUT_DIR%" 
@REM cd ..

:: Prepare custom dataset
@REM cd data
@REM python prepare_data_2d_custom.py ^
@REM -i "%OUTPUT_DIR%" ^
@REM -o %CUSTOM_DATASET%
@REM cd ..

:: Run VideoPose3D
python run.py ^
-c %CHECKPOINT% --evaluate %PRETRAINED_MODEL% ^
-d custom -k %CUSTOM_DATASET% ^
-arc 3,3,3,3,3 ^
--render --viz-subject %VIDEO_FILE% --viz-action custom --viz-camera 0 ^
--viz-video "%INPUT_DIR%/%VIDEO_FILE%".mp4 --viz-output "%OUTPUT_DIR%/%RENDER_FILE%"

endlocal
