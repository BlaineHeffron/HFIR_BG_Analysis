import h5py
from argparse import ArgumentParser
from ROOT import TH2F, TCanvas
import os

def getHistNum(z, nz, zmin, zmax):
    inc = (zmax-zmin)/nz
    for i in range(int(nz)):
        if (i*inc + zmin) <= z < ((i+1)*inc + zmin):
            return i
    print("z {0} not found in range".format(z))
    return int(nz) - 1

def main():
    arg = ArgumentParser()
    arg.add_argument("f",help="path to h5 file containing ioni branch", type=str)
    arg.add_argument("outpath",help="path to output graphs", type=str)
    args = arg.parse_args()
    zrange = 80. #0 to 79 mm
    nz = 20.
    zinc = zrange/nz
    hists = [TH2F("layer_{}".format(i),"z = {0} to {1} mm xy cross section".format(zinc*i - zrange/2.,zinc*(i+1) - zrange/2.),100,-30,30,100,30,30) for i in range(int(nz))]
    with h5py.File(args.f, 'r') as h5obj:
        for val in h5obj["ioni"]:
            hn = getHistNum(val[2][2], nz, -zrange/2., zrange/2)
            hists[hn].Fill(val[2][0], val[2][1])
    outdir =os.path.join(args.outpath,"zhists")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    canvas = TCanvas("canvas")
    canvas.cd()
    canvas.Print(os.path.join(outdir, "xsectionhists.pdf["))
    for h in hists:
        h.Draw("colz")
        canvas.Print(os.path.join(outdir, "xsectionhists.pdf"))
    canvas.Print(os.path.join(outdir, "xsectionhists.pdf]"))

if __name__=="__main__":
    main()