import zipfile
from pathlib import Path

BASE = Path(r"C:\Sync_cloud\Dropbox\군산대-랩-공동작업장\Papers\paper_wedling2")
pairs = [
    (BASE / r"04_experiments\data\welding defect final0.v1i.yolov8.zip", BASE / r"04_experiments\data\final0"),
    (BASE / r"04_experiments\data\Weld Defect.v1i.yolov8.v2i.yolov8.zip", BASE / r"04_experiments\data\ikkj"),
]

for zpath, dest in pairs:
    fixed = 0
    still_missing = []
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = dest / info.filename
            # extended-length path bypasses the 260-char MAX_PATH limit
            ext_target = Path("\\\\?\\" + str(target))
            if ext_target.exists():
                continue
            ext_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as src, open(ext_target, "wb") as out:
                    out.write(src.read())
                fixed += 1
            except OSError as e:
                still_missing.append((info.filename, str(e)))
    print(f"{dest.name}: re-extracted {fixed}, still missing {len(still_missing)}")
    for name, err in still_missing[:5]:
        print("  MISS:", name, err)
print("DONE")
