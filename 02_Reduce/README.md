# Reduce

## Motivation

* Consider a calculation of _Sum(i = 1 .. n)(x)_

* Can be expressed as:

   _( (x1 + x2) + x3 ) + ..._

* Since _+_ is associative, the subspace can be divided:

  _Sum(i = 1 .. n)(x) = Sum(i = 1 .. n/2)(x) + Sum(i = n/2 + 1 .. n)(x)_


## Paractice
* Adding up N data elements
* Use 1 block of N/2 threads
* Each thread does _x[i] += x[j];_
* At each step number of threads is halved and Distance (j-i) is doubled
* x[0] is the result
