module load python/3.10
[ -f "env/bin/activate" ] || echo "No environment!"
if [ -d "$SLURM_TMPDIR" ]; then
    virtualenv --no-download "$SLURM_TMPDIR/env"
    source "$SLURM_TMPDIR/env/bin/activate"
    pip install --no-index -r conf/cc-base-requirements.txt
    pip install --no-index --no-deps .
    realpath env/lib/python3.10/site-packages \
        > "$SLURM_TMPDIR/env/lib/python3.10/site-packages/base.pth"
else
    source env/bin/activate
fi