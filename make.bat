@echo off

REM Cython building utility commands for Windows

REM Clean all C-files, pyd-files, pyrobuf-directory, build-directory, and egg-info
if "%1" == "clean" (
    IF EXIST *.pyd (
        del /S *.pyd
    )

    IF EXIST *.c (
        del /S *.c
    )

    IF EXIST .coverage (
        del /S .coverage
    )

    IF EXIST pyrosm.egg-info (
        RMDIR /S /Q pyrosm.egg-info
    )

    IF EXIST pyrobuf (
        RMDIR /S /Q pyrobuf
    )

    IF EXIST build (
        RMDIR /S /Q build
    )

    IF EXIST .pytest_cache (
        RMDIR /S /Q .pytest_cache
    )
)
