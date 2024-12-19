cd UsersGuide/
rm -rf build/
make latexpdf
make html
cd ..
cd ProgramRef/
make distclean
make spatialnde2_reference.pdf
make ReferenceHTML
cd ..
zip -r /tmp/snde2_docs.zip UsersGuide/build/latex/spatialnde2.pdf UsersGuide/build/html/ ProgramRef/spatialnde2_reference.pdf ProgramRef/spatialnde2_referenceHTML/
