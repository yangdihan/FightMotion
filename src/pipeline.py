from extract_fighters.main import run_extract_fighters
import torch

if __name__ == "__main__":

    input_video_path = (
        "D:/Documents/devs/fight_motion/data/raw/aldo_holloway_single_angle.mp4"
    )
    output_folder = "D:/Documents/devs/fight_motion/data/interim/"
    run_extract_fighters(
        input_video_path,
        output_folder,
    )