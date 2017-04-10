snakemake --timestamp -j 100 --cluster "bsub -M 12000 -R 'rusage[mem=12000]'"
