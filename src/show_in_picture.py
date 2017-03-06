import numpy as np
import pylab as pl
import src.TyphoonSeqLstm as ty
import src.SimpleLstm_bak as sl_
import src.SimpleLstm as sl
import src.OtherLstm as ol
import src.simple_network as slnk

def print_picture(lines):
    for i in range(lines.__len__()):
        line = lines[i]
        x = [float(var[0]) for var in line]
        y = [float(var[1]) for var in line]
        if i %2 ==0:
            color= "r"
        else:
            color = "g"
        pl.plot(x,y,color)
    pl.show()



slm = ty.SeqLstmModel()
slb = sl_.SimpleLstmSeq()
# predict,y= slb.demo()
ols = ol.SimpleLstmSeq()

ols.lr = 1.0
predict,y= slnk.SNW().demo()
print_picture([predict,y])