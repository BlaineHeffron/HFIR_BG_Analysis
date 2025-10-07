import re

# The LaTeX table as a string
latex_table = r"""
\begin{tabular}{lllll}
        \toprule
         & isotope             & energy [kev] & yield [\%]~\cite{nndc} & reaction          \\
        \midrule
         & $^{59}\textrm{ni}$  & 11386.5      & 48.206     & neutron capture   \\
         & $^{59}\textrm{ni}$  & 10054.14     & 18.43      & neutron capture   \\
         & $^{59}\textrm{ni}$  & 9102.1       & 19.798     & neutron capture   \\
         & $^{53}\textrm{cr}$  & 9718.79      & 20         & neutron capture   \\
         & $^{53}\textrm{cr}$  & 8884.81      & 56.203     & neutron capture   \\
         & $^{53}\textrm{cr}$  & 7100.11      & 9.633      & neutron capture   \\
         & $^{53}\textrm{cr}$  & 6645.64      & 12.291     & neutron capture   \\
         & $^{63}\textrm{ni}$  & 9656.89      & 80.702     & neutron capture   \\
         & $^{63}\textrm{ni}$  & 8311.45      & 9.825      & neutron capture   \\
         & $^{54}\textrm{fe}$  & 9297.8       & 100        & neutron capture   \\
         & $^{54}\textrm{fe}$  & 8886.4       & 18.636     & neutron capture   \\
         & $^{58}\textrm{ni}$  & 8998.63      & 100        & neutron capture   \\
         & $^{58}\textrm{ni}$  & 8533.71      & 47.839     & neutron capture   \\
         & $^{58}\textrm{ni}$  & 8120.75      & 8.501      & neutron capture   \\
         & $^{50}\textrm{cr}$  & 8512.2       & 37.5       & neutron capture   \\
         & $^{50}\textrm{cr}$  & 8484.2       & 32.5       & neutron capture   \\
         & $^{50}\textrm{cr}$  & 7362.6       & 15.525     & neutron capture   \\
         & $^{50}\textrm{cr}$  & 6135.9       & 11.263     & neutron capture   \\
         & $^{52}\textrm{cr}$  & 7938.58      & 100        & neutron capture   \\
         & $^{52}\textrm{cr}$  & 7374.58      & 18.942     & neutron capture   \\
         & $^{52}\textrm{cr}$  & 5618.23      & 32.287     & neutron capture   \\
         & $^{52}\textrm{cr}$  & 5269         & 11.331     & neutron capture   \\
         & $^{60}\textrm{ni}$  & 7819.56      & 100        & neutron capture   \\
         & $^{60}\textrm{ni}$  & 7536.62      & 57.0388    & neutron capture   \\
         & $^{27}\textrm{al}$  & 7724.034     & 96.0573    & neutron capture   \\
         & $^{27}\textrm{al}$  & 7693.398     & 12.366     & neutron capture   \\
         & $^{56}\textrm{fe}$  & 7645.58      & 86.207     & neutron capture   \\
         & $^{56}\textrm{fe}$  & 7631.18      & 100        & neutron capture   \\
         & $^{56}\textrm{fe}$  & 7278.82      & 20.69      & neutron capture   \\
         & $^{56}\textrm{fe}$  & 6018.42      & 34.138     & neutron capture   \\
         & $^{56}\textrm{fe}$  & 5920.35      & 33.103     & neutron capture   \\
         & $^{63}\textrm{cu}$  & 7916.26      & 100        & neutron capture   \\
         & $^{63}\textrm{cu}$  & 7638         & 48.943     & neutron capture   \\
         & $^{63}\textrm{cu}$  & 7307.31      & 27.0695    & neutron capture   \\
         & $^{63}\textrm{cu}$  & 7253.05      & 12.538     & neutron capture   \\
         & $^{9}\textrm{be}$   & 6809.61      & 100        & neutron capture   \\
         & $^{9}\textrm{be}$   & 3367.45      & 49.165     & neutron capture   \\
         & $^{9}\textrm{be}$   & 3443.406     & 16.84      & neutron capture   \\
         & $^{9}\textrm{be}$   & 2590.014     & 32.929     & neutron capture   \\
         & $^{9}\textrm{be}$   & 853.63       & 35.812     & neutron capture   \\
         & $^{48}\textrm{ti}$  & 6760.12      & 54.15      & neutron capture   \\
         & $^{48}\textrm{ti}$  & 6418.53      & 35.67      & neutron capture   \\
         & $^{24}\textrm{na}$ & 1368.63 & 99.99 & radioactive decay \\
         & $^{24}\textrm{na}$ & 2754.01 & 99.87 & radioactive decay \\
         & $^{208}\textrm{tl}$ & 2614.51 & 99.75 & radioactive decay \\
         & $^{154}\textrm{eu}$ & 1274.43      & 34.83      & radioactive decay \\
         & $^{154}\textrm{eu}$ & 723.30       & 20.06      & radioactive decay \\
         & $^{152}\textrm{eu}$ & 778.9        & 12.93      & radioactive decay \\
         & $^{152}\textrm{eu}$ & 841.63       & 14.2       & radioactive decay \\
         & $^{152}\textrm{eu}$ & 963.38       & 11.6       & radioactive decay \\
         & $^{152}\textrm{eu}$ & 964.01       & 14.51      & radioactive decay \\
         & $^{152}\textrm{eu}$ & 1085.84      & 10.11      & radioactive decay \\
         & $^{152}\textrm{eu}$ & 1112.08      & 13.67      & radioactive decay \\
         & $^{152}\textrm{eu}$ & 1408.01      & 20.87      & radioactive decay \\
         & $^{41}\textrm{ar}$  & 1293.64      & 99.16      & radioactive decay \\
         & $^{40}\textrm{k}$   & 1460.82      & 10.66      & radioactive decay \\
         & $^{60}\textrm{co}$  & 1332.5       & 99.98      & radioactive decay \\
         & $^{60}\textrm{co}$  & 1173.2       & 99.85      & radioactive decay \\
        \bottomrule
    \end{tabular}
"""

# Extract the lines of the table
lines = latex_table.strip().split('\n')

# Find the start and end of the data rows
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '\\midrule' in line:
        start_idx = i + 1
    elif '\\bottomrule' in line:
        end_idx = i
        break

# Collect data rows and parse them
data_rows = []
for i in range(start_idx, end_idx):
    row_line = lines[i].strip()
    if row_line.startswith('&'):
        # Split by '&' and strip whitespace
        parts = [p.strip() for p in row_line.split('&')]
        # parts[0] is empty, parts[1] isotope, parts[2] energy, parts[3] yield, parts[4] reaction with possible '\\'
        energy_str = parts[2]
        # Clean the reaction part by removing trailing '\\'
        parts[4] = parts[4].rstrip('\\').strip()
        try:
            energy = float(energy_str)
        except ValueError:
            raise ValueError(f"Invalid energy value: {energy_str}")
        # Store the energy and the original line for reconstruction
        data_rows.append((energy, lines[i]))

# Sort the data rows by energy in ascending order
sorted_rows = sorted(data_rows, key=lambda x: x[0])

# Reconstruct the LaTeX table
sorted_latex = []
for i in range(0, start_idx):
    sorted_latex.append(lines[i])
for _, row_line in sorted_rows:
    sorted_latex.append(row_line)
for i in range(end_idx, len(lines)):
    sorted_latex.append(lines[i])

# Print the sorted LaTeX table
print('\n'.join(sorted_latex))