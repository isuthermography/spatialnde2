#! /usr/bin/env python

import os.path
import sys

infilename = sys.argv[1]
outfilename = sys.argv[2]

infh = open(infilename, "r")
buf = infh.read()
infh.close()
outfh = open(outfilename, "w")

preproc_symbol = "SNDE_"+os.path.split(outfilename)[1].replace(".","_").upper()

outfh.write("#ifndef %s\n#define %s\nstatic const char *%s_%s=" % (preproc_symbol,preproc_symbol,os.path.splitext(infilename)[0], os.path.splitext(infilename)[1][1:]))

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
