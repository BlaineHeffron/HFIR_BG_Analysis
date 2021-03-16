import sqlite3
import os

from src.utilities.util import get_data_dir, retrieve_file_extension, creation_date, file_number


class SQLiteBase:

    def __init__(self, path):
        self.path = os.path.expanduser(path)
        self.__db_connection = sqlite3.connect(self.path)
        self.cur = self.__db_connection.cursor()

    def close(self):
        self.__db_connection.close()

    def execute(self, new_data):
        self.cur.execute(new_data)

    def insert_or_ignore(self, table, columns, values):
        instxt = columns[0]
        qtxt = "?"
        if len(columns) > 1:
            for col in columns[1:]:
                instxt += "," + col
                qtxt += ",?"
        self.cur.execute("INSERT OR IGNORE INTO {0} ({1}) VALUES ({2})".format(table, instxt, qtxt), values)

    def fetchone(self, sql):
        self.execute(sql)
        result = self.cur.fetchone()
        return result

    def fetchall(self, sql):
        self.execute(sql)
        result = self.cur.fetchall()
        return result

    def create_table(self, name, collist):
        txt = "CREATE TABLE IF NOT EXISTS {0}(".format(name)
        i = 0
        for col in collist:
            if i == 0:
                txt += col
            else:
                txt += ',' + col
        txt += ')'
        self.cur.execute(txt)

    def commit(self):
        self.__db_connection.commit()

    def __del__(self):
        self.__db_connection.close()

    def __enter__(self):
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        self.cur.close()
        if isinstance(exc_value, Exception):
            self.__db_connection.rollback()
        else:
            self.__db_connection.commit()
        self.__db_connection.close()


class HFIRBG_DB(SQLiteBase):
    def __init__(self, path=None):
        if path is None:
            if "HFIRBG_CALDB" not in os.environ:
                raise RuntimeError("Please set environment variable HFIRBG_CALDB to the path of your calibration sqlite database")
            path = os.environ["HFIRBG_CALDB"]
        super().__init__(path)

    def sync_files(self, base_path=None):
        if base_path is None:
            base_path = get_data_dir()
        fs = retrieve_file_extension(base_path, ".txt")
        dir_key = {}
        for f in fs:
            base_dir = os.path.dirname(f)
            if base_dir not in dir_key.keys():
                dir_key[base_dir] = []
            name = os.path.basename(f)
            dir_key[base_dir].append(name)
        dir_ids = {}
        for dir in dir_key.keys():
            self.insert_or_ignore("directory",["path"],[dir])
            dir_ids[dir] = self.cur.lastrowid
        self.commit()
        self.execute('BEGIN TRANSACTION')
        for dir in dir_key.keys():
            dirid = dir_ids[dir]
            for fname in dir_key[dir]:
                create_time = creation_date(os.path.join(dir,fname))
                run_number = file_number(fname)
                if run_number == -1:
                    self.insert_or_ignore("datafile", ["name", "directory_id", "creation_time"],
                                          [fname, dirid, create_time])
                else:
                    self.insert_or_ignore("datafile", ["name", "directory_id", "creation_time", "run_number"],
                                          [fname, dirid, create_time, run_number])
        self.execute("END TRANSACTION")
        self.commit()



