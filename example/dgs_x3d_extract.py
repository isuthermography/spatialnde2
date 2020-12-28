
import sys
import os
import os.path

import dg_file as dgf
from lxml import etree

from io import StringIO,BytesIO

import dataguzzler as dg
import dg_metadata as dgm
import dg_eval
import dg_image as dgi

unitsperintensity=float(1.0)
offset=float(0.0)
colormap="hot"
frametime=1.19

if (len(sys.argv) < 8) :
    print ("Usage: %s <file.dgs> <x3d_channel>  <outdir> <texture_frametime> <unitsperintensity> <offset> <channelb:unitsperintensity=xxx> <channelb:offset=yyy>  <colormap> " % (sys.argv[0]))
    sys.exit(1);
    pass

dgs_file=sys.argv[1]
x3d_channel_name = sys.argv[2]

outdir=sys.argv[3]
frametime=float(sys.argv[4]);
argpos=5
 

    


(md,wfmdict) = dgf.loadsnapshot(dgs_file)

wfm=wfmdict[x3d_channel_name]
TexChanPrefix=""
if "TexChanPrefix" in wfm.MetaData:
    TexChanPrefix=wfm.MetaData["TexChanPrefix"].Value
    pass

if "X3DGeomRef" in wfm.MetaData:
    wfm=wfmdict[wfm.MetaData["X3DGeomRef"].Value]
    pass

if "X3DGeom" in wfm.MetaData:
    Geom=wfm.MetaData["X3DGeom"].Value
    pass
    

GeomBuf=BytesIO(Geom.encode('utf-8'))
p = etree.XMLParser(huge_tree=True) # support very large files
GeomXML=etree.parse(GeomBuf,parser=p)

Tex = GeomXML.xpath('//ImageTexture')
assert(len(Tex)==1)

url=Tex[0].attrib["url"]

if '\"' in url:
    url=url[1:-1]  # url properly quoted per spec
    pass

assert(url.startswith('#'))
texchan = TexChanPrefix+url[1:]

dgs_basename=os.path.splitext(os.path.split(dgs_file)[1])[0]

texoutputfile = dgs_basename+"_"+texchan+".png"

# Modify XML in memory url to point at outputfile we are going to create
Tex[0].attrib["url"]='"%s"' % (texoutputfile)


(argpos,unitsperintensity,offset)=dgi.proccontrastparams(sys.argv,argpos,unitsperintensity,offset,texchan,wfmdict) 

    
try :
    colormap=str(sys.argv[argpos]);
    argpos+=1
    pass
except: 
    pass
    



texwfm=wfmdict[texchan]; # get waveform structure
    
(ndim,DimLen,IniVal,Step,bases)=dg_eval.geom(texwfm)
    
width=DimLen[0];
height=DimLen[1];

imagedat=None

t0=0.0
dt=1.0
if len(IniVal) > 2:
    t0=IniVal[2]
    dt=Step[2]
    pass

frameno = (frametime-t0)/dt

# if (len(wfm.data.shape) > 2) :
#    exec "imagedat=dg.colormap_%s(wfm.data[:,:,frameno].transpose(),offset,unitsperintensity)" % colormap
#    pass
# else :
#     exec "imagedat=dg.colormap_%s(wfm.data.transpose(),offset,unitsperintensity)" % colormap
#     pass


# # convert to PIL image object
# img=scipy.misc.toimage(imagedat[::-1,:], cmin=0.0, cmax=1.0);
img=dgi.toimage(texwfm,wfmdict,frameno,unitsperintensity,offset,colormap)


# save image
x3doutputfile=dgs_basename+"_"+x3d_channel_name+".x3d"
print("Writing %s" % (os.path.join(outdir,x3doutputfile)))
GeomXML.write(os.path.join(outdir,x3doutputfile))
print("Writing %s" % (os.path.join(outdir,texoutputfile)))
img.save(os.path.join(outdir,texoutputfile))


