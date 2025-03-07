import numpy as np
from snnpy import *
from time import time


if __name__ == "__main__":
    radius = 3.5
    rng = np.random.RandomState(0)
    X = np.loadtxt("data.txt")

    # build SNN model
    st = time()
    snn_model = build_snn_model(X)  
    print("SNN index time:", time()-st)
    # will be faster if return_dist is False, then no distance information come out

    for i in [0, 49, 99]:
        # query neighbors of X[0]
        ind, dist = snn_model.query_radius(X[i], radius, return_distance=True)
        # If remove the returning of the associated distance, use: ind, dist = snn_model.query_radius(X[0], radius, return_distance=False)
        sort_ind = np.argsort(dist)

        # print total number and top five indices
        print("number of neighbors:", len(ind))
        print("indices of closest five:", ", ".join([str(i) for i in ind[sort_ind][:5]]))
