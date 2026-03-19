from pathlib import Path
import shutil

src_base = Path(r"C:/Users/Logan/Downloads/data_odometry_calib/dataset/sequences")
dst_base = Path(r"C:/Users/Logan/Documents/Datasets/odom_gray/sequences")

files_to_copy = ["calib.txt", "times.txt"]

for seq_path in src_base.iterdir():
    if seq_path.is_dir():
        seq_name = seq_path.name  # e.g., "11"

        for filename in files_to_copy:
            src_file = seq_path / filename
            dst_file = dst_base / seq_name / filename

            if src_file.exists():
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_file, dst_file)
                print(f"Copied {filename} for sequence {seq_name}")
            else:
                print(f"Missing {filename} in sequence {seq_name}")