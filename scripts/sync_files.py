import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.SqliteManager import HFIRBG_DB


def main():
    db = HFIRBG_DB()
    db.sync_files()
    db.sync_db()
    db.close()


if __name__ == "__main__":
    main()
