.PHONY: all commit

all: ridge.ipynb testing.ipynb solutions-ridge.ipynb solutions-testing.ipynb

ridge.ipynb: solutions-ridge.ipynb build.py
	./build.py ridge --include-readme

testing.ipynb: solutions-testing.ipynb build.py
	./build.py testing

solutions-%.ipynb: solutions-src-%.ipynb
	@# name python3 is the auto one from currently active conda env...we hope
	jupyter nbconvert --to=notebook --execute $< --output=$@ \
		--ExecutePreprocessor.kernel_name=python3 \

commit: ridge.ipynb testing.ipynb solutions-ridge.ipynb solutions-testing.ipynb
	./commit.py

