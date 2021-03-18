BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "detector_settings" (
	"id"	INTEGER PRIMARY KEY AUTOINCREMENT,
	"bias"	REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS "detector_configuration" (
   "id"	INTEGER PRIMARY KEY AUTOINCREMENT,
   "detector" INTEGER NOT NULL,
   "detector_settings"	INTEGER NOT NULL,
   "acquisition_settings" INTEGER NOT NULL,
   "shield"	INTEGER,
   FOREIGN KEY("detector") REFERENCES "detector"("id"),
   FOREIGN KEY("acquisition_settings") REFERENCES "acquisition_settings"("id"),
   FOREIGN KEY("detector_settings") REFERENCES "detector_settings"("id"),
   FOREIGN KEY("shield") REFERENCES "shield_configuration"("id")
);
CREATE TABLE IF NOT EXISTS "detector_coordinates" (
    "id"    INTEGER PRIMARY KEY AUTOINCREMENT,
    "Rx"	REAL,
    "Rz"	REAL,
    "Lx"	REAL,
    "Lz"	REAL,
    "angle"	REAL,
    "track" INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS "runs" (
	"id"	INTEGER PRIMARY KEY AUTOINCREMENT,
	"file_list"	INTEGER,
	"description"	TEXT,
	"name"	TEXT,
	"detector_configuration" INTEGER,
	"detector_coordinates" INTEGER,
	FOREIGN KEY("file_list") REFERENCES "file_list"("id"),
    FOREIGN KEY("detector_coordinates") REFERENCES "detector_coordinates"("id"),
    FOREIGN KEY("detector_configuration") REFERENCES "detector_configuration"("id")
);
CREATE TABLE IF NOT EXISTS "file_list" (
	"id"	INTEGER PRIMARY KEY AUTOINCREMENT,
	"file_id"	INTEGER NOT NULL UNIQUE,
	"list_id"	INTEGER NOT NULL,
	FOREIGN KEY("file_id") REFERENCES "datafile"("id")
);
CREATE TABLE IF NOT EXISTS "calibrations" (
	"id"	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	"file_id"	INTEGER NOT NULL,
	"det"	INTEGER NOT NULL,
	"A0"	REAL,
	"A1"	REAL,
	FOREIGN KEY("file_id") REFERENCES "datafile"("id"),
	FOREIGN KEY("det") REFERENCES "detector"("id")
);
CREATE TABLE IF NOT EXISTS "shield_configuration" (
	"id"	INTEGER PRIMARY KEY AUTOINCREMENT,
	"name"	TEXT NOT NULL,
	"description"	TEXT
);
CREATE TABLE IF NOT EXISTS "detector" (
	"id"	INTEGER PRIMARY KEY AUTOINCREMENT,
	"type"	TEXT NOT NULL,
	"description"	TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS "acquisition_settings" (
	"id"	INTEGER PRIMARY KEY AUTOINCREMENT,
	"coarse_gain"	REAL NOT NULL,
	"PUR_guard"	REAL,
	"offset"	INTEGER,
	"fine_gain"	REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS "directory" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "path" TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS "datafile" (
	"id"	INTEGER PRIMARY KEY AUTOINCREMENT,
	"name"	TEXT,
	"directory_id"	INTEGER NOT NULL,
	"creation_time"	INTEGER,
	"run_number"	INTEGER,
	FOREIGN KEY("directory_id") REFERENCES "directory"("id")
);
COMMIT;
