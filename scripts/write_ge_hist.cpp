#include "TFile.h"
#include "TH1D.h"
#include "TVectorT.h"
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    char name[512];
    double live = 0, A0 = 0, A1 = 0;
    int nbins = 0;
    if (std::fscanf(stdin, "%511s %lf %d %lf %lf", name, &live, &nbins, &A0, &A1) != 5)
        return 3;
    std::vector<double> counts(static_cast<size_t>(nbins));
    for (int i = 0; i < nbins; ++i) {
        if (std::fscanf(stdin, "%lf", &counts[static_cast<size_t>(i)]) != 1) return 4;
    }
    const double xmin = A0 + 0.5 * A1;
    const double xmax = A0 + (nbins + 0.5) * A1;
    TH1D hist("GeDataHist", name, nbins, xmin, xmax);
    hist.Sumw2(false);
    for (int i = 0; i < nbins; ++i)
        hist.SetBinContent(i + 1, counts[static_cast<size_t>(i)]);
    TVectorT<float> live_vec(1);
    live_vec[0] = static_cast<float>(live);
    TFile out(argv[1], "RECREATE");
    hist.Write();
    live_vec.Write("LiveTime");
    out.Close();
    std::printf("wrote %s nbins=%d live=%g xmin=%g xmax=%g\n", argv[1], nbins, live, xmin, xmax);
    return 0;
}
