### sfl

Sparse Fused Lasso (NYC Taxi dataset)

Effort to reproduce and expand upon the NYC Taxi experiment in [Trend Filtering on Graphs](https://arxiv.org/abs/1410.7690)

Uses a modified and wrapped version of [SnapVX](https://github.com/snap-stanford/snapvx), which itself wraps [CVXPY](http://www.cvxpy.org/en/latest/install/) for the optimization.  SnapVX includes an ADMM implementation that allows us to distribute the problem across multiple processers.  Right now, this uses a python `multiprocessing` backend, but I have an implementation that uses `dask` and could scale across multiple machines -- open an issue if interested.

#### Dependencies

1) Install `snap.py` per https://snap.stanford.edu/snappy/

2) Install `CVXPY` per http://www.cvxpy.org/en/latest/install/

3) `pip install -r requirements.txt`

4) (Optional) If you want to be able to pull your own street networks, you'll also have to install [osmnx](https://github.com/gboeing/osmnx), which was sortof painful on my machine.

Tested on `python2.7` only.

#### Usage

See `./run.sh`

#### Notes

CVXPY is a general-purpose convex optimization library.  On the positive side, that means that you can make minor adjustments to this code to solve a variety of other optimization problems.  On the negative side, it also means that the implementation may not be optimized for your particular problem.

#### License

MIT
