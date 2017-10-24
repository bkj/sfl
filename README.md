### sfl
Sparse Fused Lasso (NYC Taxi dataset)

Effort to reproduce and expand upon the NYC Taxi experiment in [Trend Filtering on Graphs](https://arxiv.org/abs/1410.7690)

#### Dependencies

Install `snap.py` per https://snap.stanford.edu/snappy/

Then 
```
pip install -r requirements.txt
```

If you want to be able to pull your own street networks, you'll also have to install [osmnx](https://github.com/gboeing/osmnx), which was sortof painful on my machine.

#### Usage

See `./run.sh`

#### License

MIT
