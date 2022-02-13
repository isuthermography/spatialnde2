all:

clean:
	rm -f core.* *.swp *~ .bak*

distclean: clean
	rm -f *.pdf
	rm -rf ./spatialnde2_referenceHTML

commit: clean
	hg addrem
	hg commit


# Rules for developer reference generation:
spatialnde2_reference.pdf:
	doxygen ReferenceGenerationPDFDoxfile
	$(MAKE) -C latexout/
	mv latexout/refman.pdf ./spatialnde2_reference.pdf
	rm -rf latexout/

ReferenceHTML:
	rm -rf ./spatialnde2_referenceHTML
	doxygen ./ReferenceGenerationHTMLDoxfile
	mv ./htmlout ./spatialnde2_referenceHTML
	rm -rf ./htmlout
