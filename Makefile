# Cython building utility commands for Unix
# Clean all C-files, pyd-files, pyrobuf-directory, build-directory, and egg-info
.PHONY: clean
clean:
	rm -f .coverage
	rm -rf pyrosm.egg-info
	rm -rf pyrobuf
	rm -rf build
	rm -rf .pytest_cache
	rm -rf dist
	cd pyrosm
	rm -f *.pyd
	rm -f *.c