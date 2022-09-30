# Scan

## General

* Each output value _y(n)_ is calculated as function involving input from 1 to _n_ : _y(n)_ = _f(x1, x2, ..., xn)_

* Looks sequential (just like Reduce)

## Practice

* Similar to reduce

* Require N-1 threads

* Step size keeps doubling

* Number of threads reduced by step size

* Each thread n does _x[n+step] += x[n];_
