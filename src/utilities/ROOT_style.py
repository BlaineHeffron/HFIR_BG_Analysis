from ROOT import TColor,TStyle,gStyle,gROOT,TGaxis
from array import array

def ROOT_style():
    global rootStyle
    rootStyle = gStyle
    icol = 0
    rootStyle.SetFrameBorderMode(icol)
    rootStyle.SetCanvasBorderMode(icol)
    rootStyle.SetPadBorderMode(icol)
    rootStyle.SetPadColor(icol)
    rootStyle.SetCanvasColor(icol)
    rootStyle.SetStatColor(icol)
    
    gROOT.GetColor(10)
    
    NRGBs = 5
    NCont = 99
    stops = array('d',[ 0.00, 0.34, 0.61, 0.84, 1.00 ])
    red   = array('d',[ 0.00, 0.00, 0.87, 1.00, 0.51 ])
    green = array('d',[ 0.00, 0.81, 1.00, 0.20, 0.00 ])
    blue = array('d',[ 0.51, 1.00, 0.12, 0.00, 0.00 ])
    TColor.CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont)
    rootStyle.SetNumberContours(NCont)
    rootStyle.SetPaperSize(TStyle.kUSLetter)
    
    rootStyle.SetPadTopMargin(.03)
    rootStyle.SetPadLeftMargin(.151)
    rootStyle.SetPadRightMargin(.04)
    rootStyle.SetPadBottomMargin(.12)
    
    font=32
    tsize=0.05 
    
    rootStyle.SetTextFont(font)
    rootStyle.SetTextSize(tsize)
    
    rootStyle.SetLabelFont(font,"xyz")
    rootStyle.SetLabelSize(tsize,"xyz")
    rootStyle.SetLabelOffset(0.003,"xyz")
    
    rootStyle.SetTitleFont(font,"xyz")
    rootStyle.SetTitleSize(tsize,"xyz")
    rootStyle.SetTitleOffset(1.25,"z")
    rootStyle.SetTitleOffset(1.2,"y")
    rootStyle.SetTitleOffset(1.1,"x")
    rootStyle.SetTitleBorderSize(0)
    
    rootStyle.SetLegendBorderSize(0)
    
    rootStyle.SetHistLineWidth(2)
    rootStyle.SetLineStyleString(2,"[12 12]")
    
    
    rootStyle.SetOptTitle(0)
    rootStyle.SetOptStat("")
    rootStyle.SetOptFit(0)
    
    rootStyle.SetStatBorderSize(1)
    rootStyle.SetStatFont(132)
    rootStyle.SetStatX(0.95)
    rootStyle.SetStatY(0.95)
    rootStyle.SetLegendBorderSize(1)
    
    rootStyle.SetPadTickX(0)
    rootStyle.SetPadTickY(0)
    rootStyle.SetNdivisions(506,"XYZ")
    
    rootStyle.SetLineStyleString(2,"[30 10]")
    rootStyle.SetLineStyleString(3,"[4 8]")
    rootStyle.SetLineStyleString(4,"[15 12 4 12]")
    rootStyle.SetLineStyleString(5,"[15 15]")
    rootStyle.SetLineStyleString(6,"[15 12 4 12 4 12]")
    rootStyle.SetOptDate(0)
    rootStyle.SetDateY(.98)
    rootStyle.SetStripDecimals(False)
    
    TGaxis.SetMaxDigits(3)
