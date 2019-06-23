.PHONY: commit run-solutions

practical.ipynb: solutions.ipynb build.py
	./build.py

commit: practical.ipynb
	./commit.py

run-solutions: solutions.ipynb
	@# name python3 is the auto one from currently active conda env...we hope
	jupyter nbconvert --to=notebook --execute $< --output=$< \
		--ExecutePreprocessor.kernel_name=python3 \
