import glob
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np

output_directory = "/p/vast1/ghosh8/2026_01_02"
MAX_EVENTS_PER_RUN_NEW = int(1e6)  # Chunk size to use

# Energies to be simulated
energies = []
energies.extend(range(40, 1000, 20))
energies.append(1000)
energies.extend([1021, 1022, 1023])
energies.extend(range(2000, 12001, 1000))
energies = sorted(set(energies))

# Measured efficiency ratios (front_events_needed / iso_events_needed)
# from comparing detection efficiency of front vs isotropic runs:
#   1000 keV: iso_eff=0.000649, front_eff=0.012467 -> ratio = 0.0521
#  10000 keV: iso_eff=0.004152, front_eff=0.030797 -> ratio = 0.1348
# Front advantage decreases at higher energy, so ratio increases with energy.
MEASURED_RATIO_1000 = 0.052
MEASURED_RATIO_10000 = 0.135

def get_efficiency_ratio(energy_kev):
    """Ratio of front events needed vs isotropic events for equivalent statistics.

    Interpolates log-linearly between measured values at 1000 and 10000 keV.
    Below 1000 keV the front advantage is even larger, so we extrapolate down.
    """
    log_e = np.log10(energy_kev)
    log_e1 = np.log10(1000)
    log_e2 = np.log10(10000)
    t = (log_e - log_e1) / (log_e2 - log_e1)
    ratio = MEASURED_RATIO_1000 + t * (MEASURED_RATIO_10000 - MEASURED_RATIO_1000)
    # Clamp to reasonable bounds
    return np.clip(ratio, 0.02, 0.20)

def get_total_events(energy_kev):
    """Total front events needed, based on isotropic baseline scaled by efficiency ratio."""
    if energy_kev <= 1100:
        base_events = int(1e8)
    else:
        slope = (4e9 - 1e8) / (12000 - 1100)
        base_events = int(1e8 + slope * (energy_kev - 1100))
    efficiency = get_efficiency_ratio(energy_kev)
    return int(base_events * efficiency)

# Step 1: Sum events already simulated per energy
events_done = defaultdict(int)
for xml_file in glob.glob(os.path.join(output_directory, "*_keV_dir_front_*.h5.xml")):
    m = re.match(r".*/([0-9]+)_keV_dir_front_([0-9]+)\.h5\.xml$", xml_file)
    if m:
        energy = int(m.group(1))
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            events = None
            pg = root.find('.//PrimaryGenerator')
            if pg is not None and 'throws' in pg.attrib:
                events = int(pg.attrib['throws'])
            if events is None:
                for uicmd in root.findall('.//uicmd'):
                    cmd = uicmd.attrib.get('cmd', '')
                    if cmd.startswith('/run/beamOn'):
                        events = int(cmd.split()[1])
                        break
            if events is not None:
                events_done[energy] += events
            else:
                print(f"WARNING: Could not find event count in {xml_file}")
        except Exception as e:
            print(f"ERROR parsing {xml_file}: {e}")

# Step 2: Calculate remaining events and needed runs
total_remaining = 0
total_runs = 0
for energy in energies:
    total_events = get_total_events(energy)
    done = events_done[energy]
    remaining = total_events - done
    if remaining <= 0:
        continue  # Energy is complete
    num_new_runs = int(np.ceil(remaining / MAX_EVENTS_PER_RUN_NEW))
    total_remaining += remaining
    total_runs += num_new_runs
    print(f"Energy {energy} keV: Total Events {total_events}, Simulated {done}, Remaining {remaining}, New Runs Needed {num_new_runs}")

print(f"\nTotal remaining events: {total_remaining:,}")
print(f"Total new runs needed: {total_runs:,}")
