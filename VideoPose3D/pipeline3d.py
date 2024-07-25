import subprocess

def infer_video_d2():
    command = [
        'python', 'infer_video_d2.py',
        '--cfg', 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
        '--output-dir', 'D:/Documents/devs/VideoPose3D/dataset/output',
        '--image-ext', 'mp4',
        '--im_or_folder', 'D:/Documents/devs/VideoPose3D/dataset/input',
    ]

    result = subprocess.run(command, cwd='inference', capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    return

def prepare_data_2d():
    command = [
        'python', 'prepare_data_2d_custom.py',
        '-i', "D:\Documents\devs\VideoPose3D\dataset\output",
        '-o', "detectron_pt_coco",
    ]

    result = subprocess.run(command, cwd='data', capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    return


def run_videopose3d():
    command = [
        'python', 'run.py',
        '-d', 'custom',
        '-k', 'myvideos',
        '-arc', '3,3,3,3,3',
        '-c', 'checkpoint',
        '--evaluate', 'pretrained_h36m_detectron_coco.bin',
        '--render',
        '--viz-subject', 'aldo_holloway_angle1_fighter_op_0.mp4',
        '--viz-action', 'custom',
        '--viz-camera', '0',
        '--viz-video', 'D:/Documents/devs/VideoPose3D/dataset/input/aldo_holloway_angle1_fighter_op_0.mp4',
        '--viz-output', 'output.mp4',
        '--viz-size', '6'
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    return

if __name__ == "__main__":
    infer_video_d2()
    prepare_data_2d()
    run_videopose3d()