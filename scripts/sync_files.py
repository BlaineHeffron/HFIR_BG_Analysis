from src.database.SqliteManager import HFIRBG_DB


def main():
    db = HFIRBG_DB()
    db.sync_files()


if __name__ == "__main__":
    main()
