import sqlite3
import os
import re

from src.utilities.util import is_number

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


def generate_in_clause(values):
    txt = " IN ("
    isnum = is_number(values[0])
    for i, val in enumerate(values):
        if i == 0:
            if isnum:
                txt += str(val)
            else:
                txt += "'{}'".format(val)
        else:
            if isnum:
                txt += ",{}".format(val)
            else:
                txt += ",'{}'".format(val)
    return txt + ") "


def get_db_dir():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return os.path.join(base_path, "db")


class SQLiteBase:

    def __init__(self, path):
        self.path = os.path.expanduser(path)
        self.__db_connection = sqlite3.connect(self.path)
        self.cur = self.__db_connection.cursor()
        self.orig_row_factory = self.__db_connection.row_factory

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
            if self.cur.rowcount and self.cur.lastrowid:
                return self.cur.lastrowid
            else:
                unique_columns = self.retrieve_unique_columns(table)
                if unique_columns:
                    unique_values = []
                    my_columns = []
                    for col in unique_columns:
                        for i, val in enumerate(values):
                            if columns[i] == col:
                                unique_values.append(val)
                                my_columns.append(col)
                                break
                    theid = self.fetchone(
                        "SELECT id FROM {0} WHERE {1}".format(table,
                                                              generate_equals_where(my_columns, unique_values)))
                else:
                    theid = self.fetchone(
                        "SELECT id FROM {0} WHERE {1}".format(table, generate_equals_where(columns, values)))
                if theid:
                    return theid[0]
                else:
                    raise ValueError("insert to table {0} ignored and unable to "
                                     "retrieve id with WHERE clause {1}".format(table,
                                                                                generate_equals_where(columns, values)))

    def fetchone(self, sql, return_dict=False):
        if return_dict:
            self.set_row_mode_dict()
        self.execute(sql)
        result = self.cur.fetchone()
        if return_dict:
            result = dict(result)
            self.reset_row_mode()
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
        self.commit()

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
        for row in ts.split('\n'):
            if row.startswith("UNIQUE("):
                matches = re.finditer(PARENTHREGEX, row)
                for match in matches:
                    if '","' in match[1]:
                        cols += match[1].split('","')
                    else:
                        cols.append(match[1])
            elif "UNIQUE" in row:
                matches = re.finditer(PARENTHREGEX, row)
                for match in matches:
                    if '","' in match[1]:
                        cols += match[1].split('","')
                    else:
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

    def dictionary_select(self, table, dict):
        """
        Use a dictionary to specify the where criterion
        """
        where = "WHERE "
        for key in dict.keys():
            if is_number(dict[key]):
                where += "{0} = {1} AND ".format(key, dict[key])
            else:
                where += "{0} = '{1}' AND ".format(key, dict[key])
        where = where[0:-5]
        result = self.fetchall("SELECT * FROM {0} {1}".format(table, where))
        if not result:
            raise ValueError(f"No records found in table '{table}' with the specified criteria: '{where}'")
        
        return result

    def set_row_mode_dict(self):
        self.__db_connection.row_factory = sqlite3.Row
        self.cur = self.__db_connection.cursor()

    def reset_row_mode(self):
        self.__db_connection.row_factory = self.orig_row_factory
        self.cur = self.__db_connection.cursor()

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


