import sys
import time
import zipfile
from pathlib import Path

pairs = [
    (r"04_experiments\data\welding defect final0.v1i.yolov8.zip", r"04_experiments\data\final0"),
    (r"04_experiments\data\Weld Defect.v1i.yolov8.v2i.yolov8.zip", r"04_experiments\data\ikkj"),
]

for zpath, dest in pairs:
    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        print(f"{zpath}: {len(names)} entries -> {dest}")
        for i, name in enumerate(names):
            # retry each file a few times in case Dropbox briefly locks it
            for attempt in range(5):
                try:
                    zf.extract(name, dest_p)
                    break
                except (PermissionError, OSError):
                    if attempt == 4:
                        print(f"FAILED: {name}")
                    else:
                        time.sleep(0.5)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(names)}")
    print(f"DONE {dest}")
print("ALL DONE")
