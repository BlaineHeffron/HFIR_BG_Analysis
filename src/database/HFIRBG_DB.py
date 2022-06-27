import os

from src.database.SqliteManager import SQLiteBase, get_db_dir, generate_in_clause
from src.utilities.util import get_data_dir, retrieve_file_extension, start_date, file_number, get_calibration, \
    read_csv_list_of_tuples, get_json, is_number, timestring_to_dt


def is_rxon_name(name):
    name = name.lower()
    if "outage" in name or "precycle" in name or "postcycle" in name or "shutdown" in name or "_cal" in name or "startup" in name:
        return False
    return True

def is_rxoff_name(name):
    name = name.lower()
    if "outage" in name or "shutdown" in name or "_cal" in name or "startup" in name:
        return False
    elif "precycle" in name or "postcycle" in name:
        return True
    return False

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
                start_time, live_time = start_date(os.path.join(dir, fname))
                run_number = file_number(fname)
                skip = False
                if fname.endswith(".txt"):
                    fname = fname[0:-4]
                for curf in cur_files:
                    if fname == curf[1] and dirid == curf[2]:
                        skip = True
                        break
                if not skip:
                    if run_number == -1:
                        self.insert_or_ignore("datafile", ["name", "directory_id", "start_time", "live_time"],
                                              [fname, dirid, start_time, live_time])
                    else:
                        self.insert_or_ignore("datafile",
                                              ["name", "directory_id", "start_time", "live_time", "run_number"],
                                              [fname, dirid, start_time, live_time, run_number])
        self.execute("END TRANSACTION")
        self.commit()

    def set_calibration_groups(self):
        """set calibration groups based on files"""
        files = self.retrieve_datafiles()
        fids = list(map(lambda x: x[0], files))
        files = self.get_file_paths_from_ids(fids)
        if not files:
            return
        for f in files:
            A0, A1 = get_calibration(f)
            if A0 is None or A1 is None:
                print("no calibration found for file {}".format(f))
            else:
                self.insert_calibration(A0, A1, os.path.basename(f), False)

    def retrieve_run_from_file_id(self, file_id):
        """
        :param file_id: file id (int)
        :return: run id
        """
        run_id = self.fetchone("SELECT run_id FROM run_file_list WHERE file_id = {0}".format(file_id))
        if run_id:
            return run_id[0]
        else:
            return None

    def retrieve_run_name_from_file_name(self, file_name):
        """
        :param file_name: file name (string)
        :return: run name
        """
        if file_name.endswith(".txt"):
            file_name = file_name[0:-4]
        run_id = self.fetchone(
            "SELECT r.name, rfl.run_id FROM run_file_list rfl JOIN runs r on r.id = rfl.run_id  WHERE rfl.file_id = (SELECT id from datafile where name = '{0}')".format(
                file_name))
        if run_id:
            return run_id[0]
        else:
            return None

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
                if name.endswith(".txt"):
                    name = name[0:-4]
                fid = self.fetchone("SELECT id FROM datafile WHERE name = '{}'".format(name))
            else:
                fid = self.fetchone("SELECT id FROM datafile WHERE run_number = {}".format(name))
            if fid:
                fids.append(fid[0])
        return fids

    def insert_file_list(self, run_id, fids):
        data = [(run_id, fid) for fid in fids]
        self.batch_insert("run_file_list", ["run_id", "file_id"], data)

    def insert_position_scan(self, f):
        fname = os.path.basename(f)[0:-4]
        data = read_csv_list_of_tuples(f, '|')
        config_id = int(data[1][-1])
        run_columns = ["description", "name", "detector_configuration", "detector_coordinates"]
        for row in data[1:]:
            R = row[0].split(',')
            L = row[1].split(',')
            angle = float(row[2])
            Rx = float(R[0].strip())
            Rz = float(R[1].strip())
            Lx = float(L[0].strip())
            Lz = float(L[1].strip())
            data = [Rx, Rz, Lx, Lz, angle]
            coord_id = self.insert_or_ignore("detector_coordinates", ["Rx", "Rz", "Lx", "Lz", "angle"], data, True)
            file_id = self.retrieve_file_ids([row[6]])
            if not file_id:
                print("Could not find file {0} in position scan {1}".format(row[6], fname))
                continue
            description = "{0} run at orientation {1}, {2}, {3} - file {4}".format(fname, row[0], row[1], angle, row[6])
            name = "{0}_{1}".format(fname, row[6])
            run_id = self.insert_or_ignore("runs", run_columns, [description, name, config_id, coord_id],
                                           retrieve_key=True)
            if not run_id:
                raise RuntimeError("Couldnt insert run {0} in position scan".format(name))
            self.insert_file_list(run_id, file_id)

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
                    self.commit()
                except Exception as e:
                    raise RuntimeError(e)
            elif fname.startswith("position_scan"):
                self.insert_position_scan(f)
            else:
                print("Warning: the file {} is not being included because it doesnt match a table in "
                      "your database!".format(fname))
        self.insert_track_scans()
        run_names = set()
        columns = self.retrieve_columns("runs")[1:]
        runsfile = get_json(os.path.join(dbdir, "runs.json"))
        coord_cols = self.retrieve_columns("detector_coordinates")[1:-1]
        for run in runsfile["runs"]:
            if run["name"] in run_names:
                print("WARNING: run name {} already added".format(run["name"]))
            run_names.add(run["name"])
            if isinstance(run["detector_coordinates"], list):
                coord_id = self.insert_or_ignore("detector_coordinates", coord_cols, run["detector_coordinates"],
                                                 retrieve_key=True)
                run["detector_coordinates"] = coord_id
            run_id = self.insert_or_ignore("runs", columns, [run[key] for key in columns], retrieve_key=True)
            flist = run["file_list"]
            if isinstance(flist, str):
                if "-" in flist and is_number(flist.split("-")[0]):
                    myrange = flist.split("-")
                    n = int(myrange[1]) - int(myrange[0]) + 1
                    fids = self.retrieve_run_range(flist)
                    if not fids or len(fids) != n:
                        print("warning, couldnt find all files in range {0} - {1}".format(myrange[0], myrange[1]))
                        if fids:
                            thenums = set()
                            expected = set([i for i in range(int(myrange[0]), int(myrange[1]) + 1)])
                            for fid in fids:
                                thenums.add(self.retrieve_run_number_from_file_id(fid))
                            print("run numbers not found: {}".format(expected - thenums))
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
                if not fids or len(fids) != len(flist):
                    print("warning, couldnt find all files in list {}".format(flist))
                if fids is not None:
                    self.insert_file_list(run_id, fids)
            self.commit()

    def insert_calgroup_with_name(self, name, A0, A1, replace=False):
        cal = self.fetchone("SELECT id, A0, A1 from calibration_group WHERE name = '{0}'".format(name))
        if cal:
            if A0 == cal[1] and A1 == cal[2]:
                return cal[0]
            else:
                if not replace:
                    print(
                        "Tried to insert a calibration group {4} but A0 and A1 do not match. database A0 = {0}, A1 = {1}, attempted A0 = {2}, A1 = {3}".format(
                            cal[1], cal[2], A0, A1, name))
                else:
                    print(
                        "updating calibration group {0} with new values A0 = {1}, A1 = {2} (previous values A0 = {3}, A1 = {4}".format(
                            name, A0, A1, cal[1], cal[2]))
                    self.update_calibration(cal[0], A0, A1)
                return cal[0]
        else:
            cal_id = self.insert_or_ignore("calibration_group", ["name", "A0", "A1"], [name, A0, A1], True)
            return cal_id

    def update_calibration(self, id, A0, A1, verbose=False):
        if verbose:
            print("updating calibration id {}".format(id))
        data = self.fetchone("SELECT A0, A1, name from calibration_group where id = {}".format(id))
        cur_A0 = data[0]
        cur_A1 = data[1]
        if verbose:
            print("updating calibration group {0} with A0 = {1}, A1 = {2}".format(data[2], A0, A1))
        if abs(A1 - cur_A1) / cur_A1 > 0.01:
            print("warning: A1 is changing by more than 1% for calibration group {}".format(data[2]))
            print("current A0, A1 = {0}, {1}".format(cur_A0, cur_A1))
            print("new A0, A1 = {0}, {1}".format(A0, A1))
        self.execute("UPDATE calibration_group SET A0 = {0}, A1 = {1} WHERE id = {2}".format(A0, A1, id))

    def insert_calgroup_with_run_id(self, run_id, A0, A1, replace=False):
        rname = self.fetchone("SELECT name FROM runs WHERE id = '{}'".format(run_id))
        if rname:
            rname = rname[0]
        else:
            raise RuntimeError("Unable to retrieve run name from run id {}, exiting".format(run_id))
        cal = self.fetchone("SELECT id, A0, A1 from calibration_group WHERE name = '{0}'".format(rname))
        if cal:
            if A0 == cal[1] and A1 == cal[2]:
                return cal[0]
            else:
                if not replace:
                    print(
                        "Tried to insert a calibration group based on run {4} but A0 and A1 do not match. database A0 = {0}, A1 = {1}, attempted A0 = {2}, A1 = {3}".format(
                            cal[1], cal[2], A0, A1, rname))
                else:
                    print(
                        "updating calibration group for run {0} with new values A0 = {1}, A1 = {2} (previous values A0 = {3}, A1 = {4}".format(
                            rname, A0, A1, cal[1], cal[2]))
                    self.update_calibration(cal[0], A0, A1)
                return cal[0]
        else:
            cal_id = self.insert_or_ignore("calibration_group", ["name", "A0", "A1"], [rname, A0, A1], True)
            return cal_id

    def insert_calibration(self, A0, A1, fname, replace=True):
        if fname.endswith(".txt"):
            fname = fname[0:-4]
        myid = self.fetchone("SELECT id FROM datafile WHERE name = '{}'".format(fname))
        if myid:
            myid = myid[0]
        else:
            print("no id found for filename {}, make sure it exists in the database.".format(fname))
            return None
        groupid = self.fetchone(
            "SELECT group_id FROM file_calibration_group WHERE file_id = {} AND det = 1".format(myid))
        if groupid:
            groupid = groupid[0]
            if replace:
                self.update_calibration(groupid, A0, A1)
            self.commit()
            return groupid
        else:
            # first check if file belongs to a known run
            run_id = self.retrieve_run_from_file_id(myid)
            if run_id:
                # print("no calibration group found for file {}, creating one based on its run name".format(fname))
                groupid = self.insert_calgroup_with_run_id(run_id, A0, A1, replace)
            else:
                print("no run found for filename {}. Creating a new calibration group with the filename.".format(fname))
                groupid = self.insert_calgroup_with_name(fname, A0, A1, replace)
            self.insert_or_ignore("file_calibration_group", ["group_id", "det", "file_id"], [groupid, 1, myid])
            self.commit()
            return groupid

    def retrieve_calibration(self, fname):
        if fname.endswith(".txt"):
            fname = fname[0:-4]
        return self.fetchone(
            "SELECT A0, A1 FROM calibration_group where id = (select group_id from file_calibration_group where file_id = (select id from datafile where name = '{}'))".format(
                fname))

    def retrieve_file_time(self, fname):
        """returns the start_time, live_time"""
        if fname.endswith(".txt"):
            fname = fname[0:-4]
        return self.fetchone("SELECT start_time, live_time FROM datafile where name = '{}'".format(fname))

    def get_file_path_from_name(self, fname):
        if fname.endswith(".txt"):
            fname = fname[0:-4]
        data = self.fetchone(
            "SELECT f.directory_id, d.path from datafile f join directory d on f.directory_id = d.id where f.name = '{}'".format(
                fname))
        if data:
            return os.path.join(data[1], fname)
        else:
            print("file name {} not in database!".format(fname))
            return None

    def get_file_paths_from_ids_min_time(self, fids, min_time):
        data = self.fetchall(
            "SELECT f.directory_id, d.path, f.name, f.live_time from datafile f join directory d on f.directory_id = d.id where f.id {0} AND live_time > {1}".format(
                generate_in_clause(fids), min_time))
        fs = []
        if data:
            for row in data:
                fs.append(os.path.join(row[1], row[2] + ".txt"))
        return fs

    def get_file_paths_from_ids(self, fids):
        data = self.fetchall(
            "SELECT f.directory_id, d.path, f.name from datafile f join directory d on f.directory_id = d.id where f.id {}".format(
                generate_in_clause(fids)))
        fs = []
        if data:
            for row in data:
                fs.append(os.path.join(row[1], row[2] + ".txt"))
        return fs

    def retrieve_file_ids_from_detector_config(self, detector_config_id, coord=None):
        files = []
        if coord:
            rows = self.fetchall("SELECT id FROM runs WHERE detector_configuration = {0} and detector_coordinates = {1}".format(detector_config_id, coord))
        else:
            rows = self.fetchall("SELECT id FROM runs WHERE detector_configuration = {}".format(detector_config_id))
        if rows:
            for row in rows:
                fids = self.retrieve_run_flist(row[0])
                files += fids
        return files

    def retrieve_run_name_from_id(self, run_id):
        row = self.fetchone("SELECT name from runs where id = {}".format(run_id))
        if row:
            return row[0]
        else:
            return None

    def retrieve_file_paths_from_detector_config(self, detector_config_id, coord=None, exclude_cal=False, min_time=None,
                                                 rxon_only=False, rxoff_only=False):
        ids = self.retrieve_file_ids_from_detector_config(detector_config_id, coord)
        if exclude_cal or rxon_only or rxoff_only:
            my_ids = []
            for id in ids:
                run_id = self.retrieve_run_from_file_id(id)
                run_name = self.retrieve_run_name_from_id(run_id)
                if exclude_cal and "_cal" in run_name:
                    continue
                if rxon_only and not is_rxon_name(run_name):
                    continue
                if rxoff_only and not is_rxoff_name(run_name):
                    continue
                my_ids.append(id)
            if not my_ids:
                return []
            if min_time:
                return self.get_file_paths_from_ids_min_time(my_ids, min_time)
            else:
                return self.get_file_paths_from_ids(my_ids)
        else:
            if min_time:
                return self.get_file_paths_from_ids_min_time(ids, min_time)
            else:
                return self.get_file_paths_from_ids(ids)

    def retrieve_run_number_from_file_id(self, id):
        name = self.fetchone("SELECT run_number from datafile where id = {}".format(id))
        if name:
            return name[0]
        else:
            return None

    def get_rd_files(self, run_grouping=False, gain_setting=None, min_time=None, exclude_cal=True, rxon_only=False, rxoff_only=False):
        """retrieves files taken while in russian doll shield, returns dictionary with key 1 = shield, key 2 = acquisition id
        acquisition id 5 = low gain, 7 = highest gain, 17 = second highest gain (covers 70 keV)
        if run_grouping is true, returns keys with run name, value file list
        if gain_setting is set, must be set to one of 'low', 'medium', 'high', or '90', will retrieve
        files for that setting"""
        qry = "SELECT id, acquisition_settings, shield FROM detector_configuration WHERE shield > 1"
        if rxon_only and rxoff_only:
            raise ValueError("rxon only and rxoff only cannot both be true")
        if gain_setting is not None:
            map = {"high": 7, "90": 17, "medium": 16, "low": 5}
            if isinstance(gain_setting, str):
                gain_setting = [gain_setting]
            for i, g in enumerate(gain_setting):
                if g not in map.keys():
                    raise ValueError("gain setting must be one of 'high', 'medium', 'low', or '90'")
                if i == 0:
                    qry += " AND acquisition_settings IN ({}".format(map[g])
                else:
                    qry += ",{}".format(map[g])
            qry += ")"
        rows = self.fetchall(qry)
        fdict = {}
        if run_grouping:
            for row in rows:
                runs = self.fetchall("SELECT id, name from runs WHERE detector_configuration = {}".format(row[0]))
                for run in runs:
                    if run[1] in fdict.keys():
                        print("ERROR: run {} already in file dictionary!".format(run[1]))
                    if exclude_cal and "_cal" in run[1]:
                        continue
                    if rxon_only and not is_rxon_name(run[1]):
                        continue
                    if rxoff_only and not is_rxoff_name(run[1]):
                        continue
                    fids = self.retrieve_run_flist(run[0])
                    if min_time:
                        fdict[run[1]] = self.get_file_paths_from_ids_min_time(fids, min_time)
                    else:
                        fdict[run[1]] = self.get_file_paths_from_ids(fids)
        else:
            for row in rows:
                if row[2] not in fdict:
                    fdict[row[2]] = {}
                if row[1] not in fdict[row[2]]:
                    fdict[row[2]][row[1]] = []
                fdict[row[2]][row[1]] += self.retrieve_file_paths_from_detector_config(row[0], exclude_cal=exclude_cal, min_time=min_time, rxon_only=rxon_only, rxoff_only=rxoff_only)
        return fdict

    def get_files_from_config(self, config):
        config_dict = {}
        if "acquisition_settings" in config.keys():
            acq_id = self.dictionary_select("acquisition_settings", config["acquisition_settings"])
            if acq_id:
                config_dict["acquisition_settings"] = acq_id[0][0]
        if "shield" in config.keys():
            config_dict["shield"] = config["shield"]
        coord_id = None
        if "detector_coordinates" in config.keys():
            coord_id = self.dictionary_select("detector_coordinates", config["detector_coordinates"])
            if coord_id:
                coord_id = coord_id[0][0]
        detector_config_ids = None
        if config_dict:
            detector_config_ids = self.dictionary_select("detector_configuration", config_dict)
        file_ids = []
        if detector_config_ids:
            for row in detector_config_ids:
                file_ids += self.retrieve_file_ids_from_detector_config(row[0], coord_id)
        elif coord_id:
            file_ids += self.retrieve_file_ids_from_detector_coord(coord_id)
        where = "WHERE "
        if file_ids:
            where += "id " + generate_in_clause(file_ids) + " AND "
        if "min_time" in config.keys():
            where += "live_time >= {} AND ".format(config["min_time"])
        if "max_date" in config.keys():
            where += "start_time <= {} AND ".format(timestring_to_dt(config["max_date"]))
        if "min_date" in config.keys():
            where += "start_time >= {} AND ".format(timestring_to_dt(config["min_date"]))
        if where == "WHERE ":
            files = self.fetchall("SELECT id FROM datafile")
        else:
            where = where[0:-5]
            files = self.fetchall("SELECT id FROM datafile {}".format(where))
        fids = list(map(lambda x: x[0], files))
        if fids:
            return self.get_file_paths_from_ids(fids)
        else:
            return None

    def insert_track_scans(self):
        files = self.fetchall("SELECT id, name FROM datafile WHERE name LIKE '%TRACK_POS%'")
        if not files:
            return
        run_columns = ["description", "name", "detector_configuration", "detector_coordinates"]
        config_id = 5  # optimal gain for 11.4 MeV
        for row in files:
            name = row[1]
            data = name[9:].split("_")
            Rx = 41.5
            Rz = float(data[0]) + 24.75
            Lz = float(data[0]) + 24.75
            Lx = 58.
            angle = float(data[1])
            track = 1
            coord_data = [Rx, Rz, Lx, Lz, angle, track]
            coord_id = self.insert_or_ignore("detector_coordinates", ["Rx", "Rz", "Lx", "Lz", "angle", "track"],
                                             coord_data, True)
            file_id = row[0]
            description = "track run at position {0}, angle {1} - file {2}".format(data[0], data[1], name)
            run_id = self.insert_or_ignore("runs", run_columns, [description, name, config_id, coord_id],
                                           retrieve_key=True)
            if not run_id:
                raise RuntimeError("Couldnt insert run {0} in position scan".format(name))
            self.insert_file_list(run_id, [file_id])

    def retrieve_file_ids_from_calibration_group_id(self, id):
        qry = "SELECT d.id from datafile d join file_calibration_group f on f.file_id = d.id where f.group_id = {}".format(
            id)
        result = self.fetchall(qry)
        file_ids = []
        if result:
            for row in result:
                file_ids.append(row[0])
        return file_ids

    def retrieve_file_names_from_calibration_group_id(self, id):
        qry = "SELECT d.name from datafile d join file_calibration_group f on f.file_id = d.id where f.group_id = {}".format(
            id)
        result = self.fetchall(qry)
        file_ids = []
        if result:
            for row in result:
                file_ids.append(row[0])
        return file_ids

    def retrieve_file_paths_from_calibration_group_id(self, id):
        file_ids = self.retrieve_file_ids_from_calibration_group_id(id)
        return self.get_file_paths_from_ids(file_ids)

    def retrieve_file_paths_in_date_range(self, start=None, stop=None):
        """
        expects start and stop to be  timestamps
        """
        where = ""
        if start or stop:
            where = " WHERE "
            if start and stop:
                where += "start_time > {0} AND start_time < {1}".format(start, stop)
            elif start:
                where += "start_time > {0}".format(start)
            else:
                where += "start_time < {0}".format(stop)
        data = self.fetchall("SELECT id FROM datafile {}".format(where))
        if data:
            file_ids = [d[0] for d in data]
            return self.get_file_paths_from_ids(file_ids)
        return None

    def retrieve_file_metadata(self, fname):
        fid = self.retrieve_file_ids([fname])
        if fid:
            fid = fid[0]
        qry = """
        SELECT 
        a.coarse_gain, a.PUR_guard, a.offset, a.fine_gain, a.LLD, a.LTC_mode, a.memory_group, 
        c.A0, c.A1, 
        d.start_time, d.live_time, d.name as filename,
        det.type as detector_type, det.description as detector_description, 
        coo.Rx, coo.Rz, coo.Lx, coo.Lz, coo.angle, coo.track, coo.id as coordinate_id,
        ds.bias, 
        s.name as shield_name, s.description as shield_description,
        r.description as run_description, r.name as run_name
        from datafile d 
        join run_file_list rfl on d.id = rfl.file_id 
        join runs r on rfl.run_id = r.id
        join detector_configuration dc on dc.id = r.detector_configuration 
        join detector_coordinates coo on coo.id = r.detector_coordinates
        join shield_configuration s on dc.shield = s.id
        join detector_settings ds on dc.detector_settings = ds.id
        join acquisition_settings a on dc.acquisition_settings = a.id
        join detector det on dc.detector = det.id
        join file_calibration_group fcg on d.id = fcg.file_id and det.id = fcg.det
        join calibration_group c on fcg.group_id = c.id
        where d.id = {}
        """
        qry = qry.format(fid)
        data_dict = self.fetchone(qry, True)
        return data_dict

    def retrieve_compatible_calibration_groups(self, filename, dt=604800):
        """
        returns ids of calibration groups within +/- dt in seconds with same detector configuration
        """
        if isinstance(filename, str):
            if filename.endswith(".txt"):
                filename = filename[0:-4]
            data = self.fetchone("SELECT id, start_time FROM datafile WHERE name = '{}'".format(filename))
        else:
            data = self.fetchone("SELECT id, start_time FROM datafile WHERE run_number = {}".format(filename))
        if data:
            file_id = data[0]
            start_time = data[1]
        else:
            return None
        cal_ids = set()
        start_max = start_time + dt
        start_min = start_time - dt
        qry = """
        SELECT fcg.group_id, d.id, d.start_time, r.detector_configuration from datafile d 
            join run_file_list rfl on d.id = rfl.file_id 
            join runs r on rfl.run_id = r.id
            join detector_configuration dc on dc.id = r.detector_configuration 
            join detector det on dc.detector = det.id
            join file_calibration_group fcg on d.id = fcg.file_id and det.id = fcg.det
            WHERE r.detector_configuration = 
            (SELECT r.detector_configuration from datafile d
            join run_file_list rfl on d.id = rfl.file_id 
            join runs r on rfl.run_id = r.id 
            WHERE d.id = {0})
            AND d.start_time < {1} AND d.start_time > {2}
        """
        qry = qry.format(file_id, start_max, start_min)
        rows = self.fetchall(qry)
        if rows:
            for row in rows:
                cal_ids.add(row[0])
        return cal_ids

    def query_position_files(self):
        """
        returns files based on angle (detector angle) and orientation (cart angle)
        """
        qry = """
        select d.start_time, d.live_time, d.name as filename,
        coo.Rx, coo.Rz, coo.Lx, coo.Lz, coo.angle, coo.track, 
        c.A0, c.A1, 
        dir.path,
        s.name as shield_name, s.description as shield_description,
        r.description as run_description, r.name as run_name
        from datafile d 
        join run_file_list rfl on d.id = rfl.file_id 
        join runs r on rfl.run_id = r.id
        join detector_configuration dc on dc.id = r.detector_configuration 
        join detector_coordinates coo on coo.id = r.detector_coordinates
        join directory dir on dir.id = d.directory_id
        join shield_configuration s on dc.shield = s.id
        join detector det on dc.detector = det.id
        join file_calibration_group fcg on d.id = fcg.file_id and det.id = fcg.det
        join calibration_group c on fcg.group_id = c.id
        WHERE shield_name = 'collimator30'
        """
        self.set_row_mode_dict()
        data = self.fetchall(qry)
        self.reset_row_mode()
        return data

    def update_datafile_time(self, fname, st, lt):
        """
        update datafile live time and start time based on file name
        """
        self.execute("UPDATE datafile SET start_time = {0}, live_time = {1} WHERE name = '{2}'".format(st, lt, fname))

    def update_file_position_with_offset(self, fname, z_offset):
        """
        update datafile z coordaintes with offset
        """
        data = self.retrieve_file_metadata(fname)
        if data:
            Rz = data["Rz"] + z_offset
            Lz = data["Lz"] + z_offset
            coo_id = data["coordinate_id"]
            self.execute("UPDATE detector_coordinates SET Rz = {0}, Lz = {1} WHERE id = {2}".format(Rz, Lz, coo_id))
        else:
            print("couldnt find datafile with filename {}".format(fname))

    def retrieve_file_ids_from_detector_coord(self, coord_id):
        files = []
        rows = self.fetchall("SELECT id FROM runs WHERE detector_coordinates = {0}".format(coord_id))
        if rows:
            for row in rows:
                fids = self.retrieve_run_flist(row[0])
                files += fids
        return files

    def retrieve_shield_info(self, shield_id):
        row = self.fetchone("SELECT name, description from shield_configuration where id = {}".format(shield_id))
        if row:
            return row[0], row[1]
        else:
            return None
