##############################
# D. Jason Koskinen
# Dec. 28, 2015
#
#
# Without knowing the value of pi
# calculate the the area of a circle 
# and make a plot of the method.
# Basically using x^2+y^2=r^2 and 
# random number generators.
#
##############################

from ROOT import *
from array import array

import math

import numpy as np

# Create a new function which squares values
def sqr(a):
    return a*a
# end def

saveImages = True

# Random numbers from the Mersenne-Twister:
r = TRandom3()
r.SetSeed(1)

# Set parameters:
Nnumbers = 100       # Number of random numbers produced.

# Create variables to save the positions
# of the points that are inside the circle
# and points outside the circle.

xpos_in = []
ypos_in = []
xpos_out = []
ypos_out = []

radius = 5.2

eq = TF1("eq1", "sqrt(%f-x*x)"% (sqr(radius)),0,radius*1.01)

# Create Nnumber different random numbers
# and see how often they fall inside/outside
# the circle equation, which has been
# restricted to the first quadrant, i.e.
# positive numbers in both x and y.

for i in range(0, Nnumbers):
    x = float(r.Uniform(0,radius))
    y = float(r.Uniform(0,radius))
    if (x*x+y*y) < (radius*radius):
        xpos_in.append(x)
        ypos_in.append(y)
    else:
        xpos_out.append(x)
        ypos_out.append(y)
    # end if/else
# end for i


# Create graph obejcts in ROOT for display purposes

gr_in  = TGraph( len(xpos_in), array('f', xpos_in), array('f', ypos_in))
gr_out = TGraph( len(ypos_out), array('f', xpos_out), array('f', ypos_out))

gr_in.SetMarkerColor(2)
gr_in.SetMarkerStyle(20)

gr_out.SetMarkerColor(4)
gr_out.SetMarkerStyle(20)

# Draw the graphs w/ the colored
# points to denote what falls inside
# and what falls outside the cirlce
# equation.

tCan_0 = TCanvas("can0","canvas0", 500, 500)
gr_in.Draw("AP")
gr_in.SetTitle("Area of Circle Monte Carlo")
gr_in.GetXaxis().SetTitle("X")
gr_in.GetYaxis().SetTitle("Y")
gr_out.Draw("Psame")
eq.Draw("same")
eq.SetLineColor(1)

tCan_0.Update()

if saveImages:
    tCan_0.SaveAs("plots/Lecture2_CircleAreaVisualization.pdf")
# end if

print " area=%0.3f   %0.3f" % (len(xpos_in)*1.0/Nnumbers*radius*radius*4, math.pi*radius*radius)

# Create a function which calculates
# the area of the circle, so that I can do
# it repeatedly with only a simple
# function call.

def circleArea(Npoints, radius):

    # Random numbers from the Mersenne-Twister:
    r = TRandom3()
    r.SetSeed(0)

    inCount = 0.0
    outCount = 0.0

    for i in range(0, Npoints):
        x = float(r.Uniform(0,radius))
        y = float(r.Uniform(0,radius))
        if (x*x+y*y) < (radius*radius):
            inCount += 1
        else:
            outCount += 1
        # end if/else
    # end for i
    return inCount/Npoints*radius*radius*4
# end def

# loop over my area calculation
# and make a histogram of all the values

hArea_0 = TH1F("areaHist0", "Circle Area Histogram", 300, 70, 100)
hArea_1 = TH1F("areaHist1", "", 30, 70, 100)
hArea_2 = TH1F("areaHist2", "", 10, 70, 100)


for i in range(0, 1000):
    hArea_0.Fill(circleArea(100, 5.2))
# end for i

hArea_1 = hArea_0.Rebin(10, "areaHist1")
hArea_2 = hArea_0.Rebin(30, "areaHist2")
hArea_1.SetLineColor(2)
hArea_2.SetLineColor(1)

hArea_0.SetLineWidth(3)
hArea_1.SetLineWidth(3)
hArea_2.SetLineWidth(3)

tLeg = TLegend( 0.65, 0.65, 0.89, 0.86)
tLeg.AddEntry( hArea_0, "bin width 0.1 m^{2}", "l")
tLeg.AddEntry( hArea_1, "bin width 1 m^{2}", "l")
tLeg.AddEntry( hArea_2, "bin width 3 m^{2}", "l")

tCan_2 = TCanvas()
hArea_2.SetStats(0)
hArea_2.Draw()
hArea_2.GetXaxis().SetTitle("Area of Circle from MC [m^2]")
hArea_2.GetYaxis().SetTitle("Frequency")
hArea_0.Draw("same")
hArea_1.Draw("same")
tLeg.Draw()

tCan_2.Update()

if saveImages:
    tCan_2.SaveAs("plots/Lecture2_CircleAreaHistoBinWidths.pdf")
# end if
    

# Code up the random number generator with blum blum schub
# x_{n+1}=x_{n}^2 * mod(M), where M=pq, i.e. the product
# of two prime numbers

# create the blum blum schub psuedo-random number generator

def bbs( x, p, q):
    return x*x % (p*q)
# end bbs

for i in range(0,20):
    print bbs(i, 7, 49)
# end for

sampling1   =  [10, 100, 1000, 10000, 100000]
piEstimate1 = []

for i in sampling1:
    piEstimate1.append(circleArea( i, 5.2)/5.2**2)
# end for

grPiEstimate1 = TGraph( len(sampling1), array('f', sampling1), array( 'f', piEstimate1))

tcPi1 = TCanvas()

grPiEstimate1.Draw("AP*")
grPiEstimate1.SetTitle("#pi estimate from circle area sampling")
grPiEstimate1.GetXaxis().SetTitle("samples")
grPiEstimate1.GetYaxis().SetTitle("#pi estimate")
grPiEstimate1.GetYaxis().SetRangeUser(2.5, 3.5)

tcPi1.Update()

if saveImages:
    tcPi1.SaveAs("plots/Lecture2_PiEstimate1.pdf")
# end if

sampling2   =  np.arange(10, 10000, 50)
piEstimate2 = []

for i in sampling2:
    piEstimate2.append(circleArea( i, 5.2)/5.2**2)
# end for

grPiEstimate2 = TGraph( len(sampling2), array('f', sampling2), array( 'f', piEstimate2))

tcPi2 = TCanvas()

grPiEstimate2.Draw("AP*")
grPiEstimate2.SetTitle("#pi estimate from circle area sampling")
grPiEstimate2.GetXaxis().SetTitle("samples")
grPiEstimate2.GetYaxis().SetTitle("#pi estimate")
grPiEstimate2.GetYaxis().SetRangeUser(2.5, 3.5)

tcPi2.Update()

if saveImages:
    tcPi2.SaveAs("plots/Lecture2_PiEstimate2.pdf")
# end if


raw_input('Press Enter to exit')

    

