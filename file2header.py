#! /usr/bin/env python

import sys
import os.path

infilename=sys.argv[1]
outfilename=sys.argv[2]

infh = file(infilename,"r")
buf=infh.read();
infh.close()
outfh=file(outfilename,"wa")
outfh.write(b"const char *%s_%s=" % (os.path.splitext(infilename)[0],os.path.splitext(infilename)[1][1:]))

pos=0;
while pos < len(buf):
    chunksz=40

    chunk=buf[pos:(pos+chunksz)]

    outfh.write("  \"")
    for chr in chunk:
        outfh.write("\\x%2.2x" % (ord(chr)))
        pass
    
    
    outfh.write("\"\n")
    pos+=chunksz
    pass
outfh.write("  ;\n")
outfh.close()
            
