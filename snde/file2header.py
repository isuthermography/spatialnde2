#! /usr/bin/env python

import os.path
import sys


def file2header(infilename,outfilename):

    infh = open(infilename, "r")
    buf = infh.read()
    infh.close()

    # Make needed output directories
    curdir = os.path.split(outfilename)[0]
    
    dirstack = []
    while not os.path.exists(curdir):
        dirstack.append(curdir)
        curdir = os.path.split(curdir)[0]
        pass
    
    for dir_to_make in dirstack[::-1]:
        os.mkdir(dir_to_make)
        pass
    
    outfh = open(outfilename, "w")
    
    preproc_symbol = "SNDE_"+os.path.split(outfilename)[1].replace(".","_").upper()
    
    outfh.write("#ifndef %s\n#define %s\nstatic const char *%s_%s=" % (preproc_symbol,preproc_symbol,os.path.split(os.path.splitext(infilename)[0])[1], os.path.splitext(infilename)[1][1:]))
    
    pos = 0
    while pos < len(buf):
        chunksz = 40
        
        chunk = buf[pos:(pos + chunksz)]
        
        outfh.write("  \"")
        for chr in chunk:
            outfh.write("\\x%2.2x" % (ord(chr)))
            pass
        
        outfh.write("\"\n")
        pos += chunksz
        pass
    outfh.write("  ;\n")
    outfh.write("#endif // %s\n\n" % (preproc_symbol))
    outfh.close()
    pass

if __name__=="__main__":
    file2header(sys.argv[1],sys.argv[2])
    pass

