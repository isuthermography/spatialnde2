@echo on

set runall=False
if "%1" == ""     set runall=True
if "%1" == "all"  set runall=True

set runhtml=False
set runpdf=False
if "%runall%" == "True"  set runhtml=True
if "%runall%" == "True"  set runpdf=True

if "%1" == "spatialnde2_reference.pdf" set runpdf=True
if "%1" == "ReferenceHTML" set runhtml=True


if not "%1" == "clean" goto next_a
    del core.*
    del *.swp
    del *~
    del .bak*
:next_a

if not "%1" == "distclean" goto next_b
    del *.pdf
    rmdir /S /Q .\spatialnde2_referenceHTML
:next_b

if not "%1" == "commit" goto next_c
    git commit -a
:next_c

if not "%runpdf%" == "True" goto next_d
    doxygen ReferenceGenerationPDFDoxfile
    cd latexout
    make
    cd ..
    mv latexout\refman.pdf .\spatialnde2_reference.pdf
    rmdir /S /Q .\latexout
:next_d

if not "%runhtml%" == "True" goto next_e
    rmdir /S /Q .\spatialnde2_referenceHTML
    doxygen ReferenceGenerationHTMLDoxfile
    ren htmlout spatialnde2_referenceHTML
:next_e

