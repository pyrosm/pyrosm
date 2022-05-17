# Cython building utility commands for Unix
# Clean all C-files, pyd-files, pyrobuf-directory, build-directory, and egg-info
.PHONY: clean
clean:
	rm -f pyrosm/*.so
	rm -f pyrosm/*.c
	rm -f .coverage
	rm -f coverage.xml
	rm -rf pyrosm.egg-info
	rm -rf pyrobuf
	rm -rf build
	rm -rf .pytest_cache
	rm -rf dist
	rm -f *.so
	rm -f *.pyd
	rm -f *.c
	rm -f *.so