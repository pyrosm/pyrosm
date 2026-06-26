# Read buildings or the road network from one PBF with osmextract and print "seconds,features".
#
# osmextract_benchmark_scaling.ipynb runs this as a separate process wrapped with /usr/bin/time
# so its peak memory (peak resident set size) can be measured the same way as the Python readers
# in benchmarks_scaling.ipynb. force_vectortranslate re-does the PBF -> features work each run
# (osmextract's analogue of QuackOSM's ignore_cache), so we measure parsing, not a cache hit.
#
# Usage: Rscript bench_worker.R <task: buildings|network> <pbf>
suppressPackageStartupMessages(library(osmextract))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2 || !nzchar(args[1]) || !nzchar(args[2])) {
    stop("Usage: Rscript bench_worker.R <task: buildings|network> <pbf>", call. = FALSE)
}

task <- args[1]
pbf <- args[2]
if (!file.exists(pbf)) {
    stop(sprintf("PBF does not exist: %s", pbf), call. = FALSE)
}

spec <- switch(task,
    buildings = list(layer = "multipolygons", where = "building IS NOT NULL"),
    network   = list(layer = "lines",         where = "highway IS NOT NULL"),
    stop(sprintf("task must be 'buildings' or 'network', got: %s", task), call. = FALSE))
query <- sprintf('SELECT * FROM "%s" WHERE %s', spec$layer, spec$where)

started <- Sys.time()
x <- oe_read(pbf, layer = spec$layer, query = query,
             force_vectortranslate = TRUE, quiet = TRUE)
seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))

cat(sprintf("%f,%d\n", seconds, nrow(x)))
