from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
FACE_DIR = BASE_DIR / "faces"
EMBEDDING_DIR = BASE_DIR / "face_embeddings"
CAPTURE_DIR = BASE_DIR / "captures"
LOG_DIR = BASE_DIR / "logs"
BACKUP_DIR = BASE_DIR / "backups"

ATTENDANCE_DATA_FILE = DATA_DIR / "attendance_data.xlsx"
ATTENDANCE_ARCHIVE_FILE = DATA_DIR / "attendance_archive.xlsx"

DATE_FORMAT = "%d-%m-%Y"
TIME_FORMAT = "%H:%M:%S"

CHECKIN_START = "08:30"
CHECKIN_END = "12:00"

CHECKOUT_START = "17:30"
CHECKOUT_END = "22:00"

DUPLICATE_WINDOW_MINUTES = 15
