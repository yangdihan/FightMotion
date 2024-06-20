from clip import run_extract_fighters

if __name__ == "__main__":
    input_video_path = (
        # "D:/Documents/devs/fight_motion/data/raw/aldo_holloway_single_angle.mp4"
        # "D:/Documents/devs/fight_motion/data/interim/output_video_bbox.mp4"
        "D:/Documents/devs/fight_motion/data/interim/output_video_contour.mp4"
    )
    output_folder = "D:/Documents/devs/fight_motion/data/interim/"
    run_extract_fighters(
        input_video_path,
        output_folder,
    )
