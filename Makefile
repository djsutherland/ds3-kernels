.PHONY: commit

practical.ipynb: solutions.ipynb build.py
	./build.py

commit: practical.ipynb
	./commit.py
