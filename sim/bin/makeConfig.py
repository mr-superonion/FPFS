#!/usr/bin/env python
import os
import numpy as np
from configparser import ConfigParser

def addInfoSparse(parser,fwhm,variance):
    #PSF
    parser['psf']=      {'fwhm'      :'%s' %fwhm
                        }
    #noise
    parser[noise]=      {'variance':'%s'  %variance,
                         'corFname':'%s'  %corFname}
    return parser



if __name__=='__main__':
    conDir  =   'noiPsfConfig'
    if not os.path.exists(conDir):
        os.mkdir(conDir)
    for ifwhm in range(8):
        fwhm    =   0.45+ifwhm*0.05
        for ivar in range(8):
            variance    =   0.003+0.0012*ivar   
            configName  =   os.path.join(conDir,'config_fwhm%d_var%d.ini' %(ifwhm,ivar))
            parser      =   ConfigParser()
            parser      =   addInfo(parser,fwhm,variance)
            #file
            with open(configName, 'w') as configfile:
                parser.write(configfile)
