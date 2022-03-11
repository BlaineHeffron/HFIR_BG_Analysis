import sqlite3
import os
import re

from src.utilities.util import get_data_dir, retrieve_file_extension, file_number, \
    read_csv_list_of_tuples, get_json, start_date

FOREIGNKEYREGEX = re.compile(r'FOREIGN KEY\(\"(.*)\"\) REFERENCES \"(.*)\"\(\"(.*)\"\)')
PARENTHREGEX = re.compile(r'\"(.*)\"')


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
            txt += " AND {0} = {1}".format(col, value_format(values[i]))
    return txt


def get_db_dir():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(base_path, "db")


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
                unique_columns = self.retrieve_unique_columns(table)
                if unique_columns:
                    unique_values = [values[i] for i in range(len(columns)) if columns[i] in unique_columns]
                    theid = self.fetchone(
                        "SELECT id FROM {0} WHERE {1}".format(table, generate_equals_where(unique_columns, unique_values)))
                else:
                    theid = self.fetchone(
                        "SELECT id FROM {0} WHERE {1}".format(table, generate_equals_where(columns, values)))
                if theid:
                    return theid[0]
                else:
                    raise ValueError("insert to table {0} ignored and unable to "
                                     "retrieve id with WHERE clause {1}".format(table, generate_equals_where(columns, values)))
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

    def retrieve_unique_columns(self, table):
        """
        :param table: name of table
        :return: list of columns that are marked unique (or their combination is unique)
        """
        ts = self.fetchone("SELECT sql FROM SQLite_master WHERE type = 'table' AND name = '{}'".format(table))
        if ts:
            ts = ts[0]
        else:
            raise ValueError("No table named {0} found".format(table))
        cols = []
        for row in ts.split(","):
            if row.startswith("UNIQUE("):
                matches = re.finditer(PARENTHREGEX, row)
                for match in matches:
                    cols.append(match[1])
            elif "UNIQUE" in row:
                matches = re.finditer(PARENTHREGEX, row)
                for match in matches:
                    cols.append(match[1])
        return cols

    def retrieve_foreign_keys(self):
        """
        returns dictionary where key is table name, value is dict of column, 2-tuple foreign table, foreign column name
        """
        ts = self.fetchall("SELECT name, sql FROM SQLite_master WHERE type = 'table'")
        foreign_key_dict = {}
        for row in ts:
            matches = re.finditer(FOREIGNKEYREGEX, row[1])
            for match in matches:
                if row[0] not in foreign_key_dict:
                    foreign_key_dict[row[0]] = {}
                foreign_key_dict[row[0]][match[1]] = (match[2], match[3])
        return foreign_key_dict

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
                raise RuntimeError(
                    "Please set environment variable HFIRBG_CALDB to the path of your calibration sqlite database")
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
            dir_ids[dir] = self.insert_or_ignore("directory", ["path"], [dir], retrieve_key=True)
        self.commit()
        self.execute('BEGIN TRANSACTION')
        for dir in dir_key.keys():
            dirid = dir_ids[dir]
            for fname in dir_key[dir]:
                create_time = start_date(os.path.join(dir, fname))
                run_number = file_number(fname)
                skip = False
                for curf in cur_files:
                    if fname == curf[1] and dirid == curf[2]:
                        skip = True
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

    def retrieve_run_flist(self, run_id):
        """
        :param run_id: run id (int)
        :return: list of file ids in the run
        """
        fids = self.fetchall("SELECT file_id FROM run_file_list WHERE run_id = {0}".format(run_id))
        return list(map(lambda x: x[0], fids))

    def retrieve_run_range(self, run_range):
        """
        :param run_range: string in format a-b
        :return: list of file ids covering the run range given
        """
        myrange = run_range.split("-")
        fids = self.fetchall(
            "SELECT id FROM datafile WHERE run_number >= {0} AND run_number <= {1}".format(myrange[0], myrange[1]))
        return list(map(lambda x: x[0], fids))

    def retrieve_file_ids(self, names):
        """
        :param names: list of file names or run numbers
        """
        fids = []
        for name in names:
            if isinstance(name, str):
                fid = self.fetchone("SELECT id FROM datafile WHERE name = '{}'".format(name))
            else:
                fid = self.fetchone("SELECT id FROM datafile WHERE run_number = {}".format(name))
            if fid:
                fids.append(fid[0])
        return fids

    def insert_file_list(self, run_id, fids):
        data = [(run_id, fid) for fid in fids]
        self.batch_insert("run_file_list", ["run_id", "file_id"], data)

    def sync_db(self):
        dbdir = get_db_dir()
        fs = retrieve_file_extension(dbdir, ".csv")
        tables = self.retrieve_tables()
        for f in fs:
            fname = os.path.basename(f)[0:-4]
            if fname in tables:
                data = read_csv_list_of_tuples(f)
                columns = self.retrieve_columns(fname)
                if len(data[0]) < len(columns):
                    columns = columns[0:len(data[0])]
                try:
                    self.batch_insert(fname, columns, data)
                except Exception as e:
                    raise RuntimeError(e)
            elif fname.startswith("position_scan"):
                print("position scan insert not yet implemented (implement me!)")
            else:
                raise RuntimeWarning("Warning: the file {} is not being included because it doesnt match a table in "
                                     "your database!".format(fname))

        columns = self.retrieve_columns("runs")[1:]
        runsfile = get_json(os.path.join(dbdir, "runs.json"))
        coord_cols = self.retrieve_columns("detector_coordinates")[1:-1]
        for run in runsfile["runs"]:
            if isinstance(run["detector_coordinates"], list):
                coord_id = self.insert_or_ignore("detector_coordinates", coord_cols, run["detector_coordinates"], retrieve_key=True)
                run["detector_coordinates"] = coord_id
            run_id = self.insert_or_ignore("runs", columns, [run[key] for key in columns], retrieve_key=True)
            flist = run["file_list"]
            if isinstance(flist, str):
                if "-" in flist:
                    fids = self.retrieve_run_range(flist)
                    if fids is not None:
                        self.insert_file_list(run_id, fids)
                else:
                    raise RuntimeError("file_list must be a string in the format a-b (run number range). Otherwise it "
                                       "is a list of file names and or run numbers")
            else:
                if not isinstance(flist, list):
                    raise RuntimeError("file_list must be a string in the format a-b (run number range). Otherwise it "
                                       "is a list of file names and or run numbers")

                fids = self.retrieve_file_ids(flist)
                if fids is not None:
                    self.insert_file_list(run_id, fids)

    def insert_calibration(self, A0, A1, fname, replace=True):
        rp = "REPLACE"
        if not replace:
            rp = "IGNORE"
        myid = self.fetchone("SELECT id FROM datafile WHERE name = '{}'".format(fname))
        if myid:
            myid = myid[0]
        else:
            print("no id found for filename {}, make sure it exists in the database.".format(fname))
            return None
        vals = "{0},{1},1,{2}".format(A0,A1,myid)
        self.cur.execute("INSERT OR {0} INTO calibrations (A0,A1,det,file_id) VALUES ({1})".format(rp, vals))
        self.commit()
        return self.cur.lastrowid

    def retrieve_calibration(self, fname):
        return self.fetchone("SELECT A0, A1 FROM calibrations where file_id = (select id from datafile where name = '{}')".format(fname))

    def get_file_path_from_name(self, fname):
        data = self.fetchone("SELECT f.directory_id, d.path from datafile f join directory d on f.directory_id = d.id where f.name = '{}'".format(fname))
        if data:
            return os.path.join(data[1],fname)
        else:
            print("file name {} not in database!".format(fname))
            return None


