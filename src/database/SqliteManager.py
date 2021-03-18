import sqlite3
import os

from src.utilities.util import get_data_dir, retrieve_file_extension, creation_date, file_number, \
    read_csv_list_of_tuples


def column_placeholder_string(columns):
    instxt = "(" + columns[0]
    qtxt = "(?"
    if len(columns) > 1:
        for col in columns[1:]:
            instxt += "," + col
            qtxt += ",?"
    return instxt + ")", qtxt + ")"

def value_format(value):
    if isinstance(value, str):
        return "'{}'".format(value)
    else:
        return value

def generate_equals_where(columns, values):
    txt = ""
    for i, col in enumerate(columns):
        if i == 0:
            txt = "{0} = {1}".format(col, value_format(values[i]))
        else:
            txt += "AND {0} = {1}".format(col, value_format(values[i]))
    return txt

def get_db_dir():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(base_path,"db")


class SQLiteBase:

    def __init__(self, path):
        self.path = os.path.expanduser(path)
        self.__db_connection = sqlite3.connect(self.path)
        self.cur = self.__db_connection.cursor()

    def close(self):
        self.__db_connection.close()

    def execute(self, new_data):
        self.cur.execute(new_data)

    def insert_or_ignore(self, table, columns, values, retrieve_key=False):
        """
        :param table: table name
        :param columns: list of column names to insert
        :param values:  list of column values to insert
        :param retrieve_key: if True, will return the key of the inserted row. If ignored,
        will select the row to retrieve the key
        :return: null if retrieve_key is false, otherwise integer representing the key of the inserted row
        """
        instxt, qtxt = column_placeholder_string(columns)
        self.cur.execute("INSERT OR IGNORE INTO {0} {1} VALUES {2}".format(table, instxt, qtxt), values)
        if retrieve_key:
            if self.cur.lastrowid == 0:
                return self.fetchone("SELECT id FROM {0} WHERE {1}".format(table, generate_equals_where(columns, values)))
            else:
                return self.cur.lastrowid


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

    def execute_many(self, sql, data):
        """
        :param sql: sql statement (string)
        :param data: list of tuples of data
        :return: null
        """
        self.cur.executemany(sql, data)

    def batch_insert(self, table, columns, data):
        """
        :param table: table name
        :param columns: list of column names in same order as data tuples
        :param data: list of tuples containing rows
        :return:
        """
        instxt, qtxt = column_placeholder_string(columns)
        sql = "INSERT OR IGNORE INTO {0} {1} VALUES {2}".format(table, instxt, qtxt)
        self.execute_many(sql, data)
        self.commit()

    def commit(self):
        self.__db_connection.commit()

    def retrieve_tables(self):
        ts = self.fetchall("SELECT tbl_name FROM SQLite_master WHERE type = 'table'")
        names = list(map(lambda x: x[0], ts))
        return names

    def retrieve_columns(self, table):
        self.execute("SELECT * FROM {} LIMIT 1".format(table))
        names = list(map(lambda x: x[0], self.cur.description))
        return names

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

    def retrieve_datafiles(self):
        return self.fetchall("SELECT * FROM datafile")


    def sync_files(self, base_path=None):
        cur_files = self.retrieve_datafiles()
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
            dir_ids[dir] = self.insert_or_ignore("directory",["path"],[dir], retrieve_key=True)
        self.commit()
        self.execute('BEGIN TRANSACTION')
        for dir in dir_key.keys():
            dirid = dir_ids[dir]
            for fname in dir_key[dir]:
                create_time = creation_date(os.path.join(dir,fname))
                run_number = file_number(fname)
                skip = False
                for curf in cur_files:
                    if fname == curf[1] and dirid == curf[2]:
                        skip = True
                        print("skipping)")
                        break
                if not skip:
                    if run_number == -1:
                        self.insert_or_ignore("datafile", ["name", "directory_id", "creation_time"],
                                              [fname, dirid, create_time])
                    else:
                        self.insert_or_ignore("datafile", ["name", "directory_id", "creation_time", "run_number"],
                                              [fname, dirid, create_time, run_number])
        self.execute("END TRANSACTION")
        self.commit()

    def sync_db(self):
        dbdir = get_db_dir()
        fs = retrieve_file_extension(dbdir, ".csv")
        tables = self.retrieve_tables()
        for f in fs:
            fname = os.path.basename(f)[0:-4]
            if fname in tables:
                data = read_csv_list_of_tuples(f)
                columns = self.retrieve_columns(fname)
                self.batch_insert(fname, columns, data)
            else:
                raise RuntimeWarning("Warning: the file {} is not being included because it doesnt match a table in "
                                     "your database!".format(fname))





