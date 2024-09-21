import os
from clip import run_extract_fighters

if __name__ == "__main__":

    DIR_IN = "D:/Documents/devs/fight_motion/data/raw"

    for file_name in os.listdir(DIR_IN):
        if file_name.endswith(".mp4"):
            # file_path = os.path.join(DIR_IN, file_name)
            # data = np.load(file_path, allow_pickle=True)

            clip_name = file_name[:-4]
            run_extract_fighters(clip_name)
