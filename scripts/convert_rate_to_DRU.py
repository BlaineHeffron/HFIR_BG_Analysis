import numpy as np

# Original data from the provided table
# Structure: {config: {range_key: {'on': (rate, err), 'off': (rate, err)}}}
# Ranges: '30-60' for 30-60 keV, '50-11380' for 50-11380 keV
data = {
    'no water': {
        '30-60': {'on': (1.382, 0.001), 'off': (0.2923, 0.0008)},
        '50-11380': {'on': (18.957, 0.009), 'off': (3.725, 0.002)},
    },
    '7 layers': {
        '30-60': {'on': (0.859, 0.001), 'off': (0.2791, 0.0003)},
        '50-11380': {'on': (11.770, 0.004), 'off': (3.516, 0.002)},
    },
    '6 layers+': {
        '30-60': {'on': (1.052, 0.002), 'off': (0.2785, 0.0004)},
        '50-11380': {'on': (None, None), 'off': (None, None)},  # Missing data
    },
    '7 layers+': {
        '30-60': {'on': (0.850, 0.001), 'off': (0.2814, 0.0006)},
        '50-11380': {'on': (None, None), 'off': (None, None)},  # Missing data
    },
}

# Energy range widths (in keV)
range_widths = {
    '30-60': 30,     # 60 - 30 = 30 keV
    '50-11380': 11330,  # 11380 - 50 = 11330 keV
}

# Detector mass and uncertainty
mass = 1.0  # kg
mass_unc = 0.05  # kg (5% uncertainty)

def compute_rate_density(rate, err, delta_e, mass, mass_unc):
    """
    Compute rate density in hz/kg/keV with propagated uncertainty.
    - rate_density = rate / delta_e / mass
    - Relative error: sqrt( (err/rate)^2 + (mass_unc/mass)^2 )
    """
    if rate is None or err is None:
        return '---', '---'
    
    rate_density = rate / delta_e / mass
    rel_err_rate = err / rate
    rel_err_mass = mass_unc / mass
    rel_err_total = np.sqrt(rel_err_rate**2 + rel_err_mass**2)
    err_density = rate_density * rel_err_total
    
    # Format to scientific notation with appropriate precision
    rate_str = f"{rate_density:.3e}"
    err_str = f"{err_density:.3e}"
    return rate_str, err_str

def main():
    # Build the table rows
    rows = []
    for config in data:
        row = f"& {config:<16}"
        
        for range_key in ['30-60', '50-11380']:
            for state in ['on', 'off']:
                rate, err = data[config][range_key][state]
                delta_e = range_widths[range_key]
                density, d_density = compute_rate_density(rate, err, delta_e, mass, mass_unc)
                if density == '---':
                    row += f" & {density:<10}"
                else:
                    row += f" & {density} $\\pm$ {d_density}"
        
        rows.append(row + " \\\\")

    # Print the LaTeX table
    print("\\begin{tabular}{llllll}")
    print("\\toprule")
    print("& \\makecell{shield                                                                                    \\\\ config} & \\makecell{30 - 60 keV \\\\ rate densities rxon [hz/kg/keV]} & \\makecell{30 - 60 keV \\\\ rate densities rxoff [hz/kg/keV]} & \\makecell{50 - 11380 keV \\\\ rate densities rxon [hz/kg/keV]} & \\makecell{50 - 11380 keV \\\\ rate densities rxoff [hz/kg/keV]} \\\\ \\hline")
    print("\\midrule")
    for row in rows:
        print(row)
    print("\\bottomrule")
    print("\\end{tabular}")

if __name__ == "__main__":
    main()