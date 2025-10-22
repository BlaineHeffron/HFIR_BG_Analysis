import sys
from os.path import dirname, realpath, join, basename
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import retrieve_file_extension, retrieve_data
from ROOT import TFile, TVectorF
import os


def check_and_fix_livetime(root_path, db):
    """
    Check if ROOT file has LiveTime, if not add it from database
    
    :param root_path: path to ROOT file
    :param db: database connection
    :return: True if fixed, False if already had LiveTime, None if couldn't fix
    """
    # Try to open file and check for LiveTime
    try:
        myFile = TFile.Open(root_path, "READ")
        livetime_obj = myFile.Get("LiveTime")
        has_livetime = livetime_obj and not livetime_obj.IsZombie()
        myFile.Close()
        
        if has_livetime:
            print(f"{basename(root_path)}: Already has LiveTime")
            return False
            
        # Get the corresponding .txt file name
        if root_path.endswith(".root"):
            txt_name = basename(root_path)[:-5] + ".txt"
        else:
            print(f"Warning: {root_path} doesn't end with .root")
            return None
            
        # Get file path from database
        txt_path = db.get_file_path_from_name(txt_name)
        if not txt_path:
            print(f"Error: Could not find {txt_name} in database")
            return None
            
        # Retrieve spectrum to get live time
        spec = retrieve_data(txt_path + ".txt", db)
        
        # Open ROOT file in UPDATE mode and add LiveTime
        myFile = TFile.Open(root_path, "UPDATE")
        lt = TVectorF(1)
        lt[0] = spec.live
        myFile.WriteObject(lt, "LiveTime")
        myFile.Close()
        
        print(f"{basename(root_path)}: Added LiveTime = {spec.live} seconds")
        return True
        
    except Exception as e:
        print(f"Error processing {root_path}: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_root_livetimes.py <directory>")
        print("Scans directory for ROOT files and adds missing LiveTime objects")
        sys.exit(1)
        
    directory = sys.argv[1]
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
        
    db = HFIRBG_DB()
    
    # Get all ROOT files in directory
    root_files = retrieve_file_extension(directory, ".root")
    
    if not root_files:
        print(f"No ROOT files found in {directory}")
        return
        
    print(f"Found {len(root_files)} ROOT files")
    print("Checking and fixing LiveTime objects...\n")
    
    fixed_count = 0
    already_ok_count = 0
    error_count = 0
    
    for root_file in root_files:
        result = check_and_fix_livetime(root_file, db)
        if result is True:
            fixed_count += 1
        elif result is False:
            already_ok_count += 1
        else:
            error_count += 1
            
    print(f"\nSummary:")
    print(f"  Fixed: {fixed_count}")
    print(f"  Already OK: {already_ok_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(root_files)}")


if __name__ == "__main__":
    main()