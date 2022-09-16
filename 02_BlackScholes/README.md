# Black Scholes Formula

## Notation
* _K_ is the strike price of the option, also known as the exercise price.
* _S_ is the price of the underlying asset at time _t_.
* _r_ is the annualized risk-free interest rate, continuously compounded.
* _σ_ is the standard deviation of the stock's returns.

## Formula

* The price of a European call option

   _C_ = _N(d1) S - N(d2) K exp(-rt)_

* The price of a European put option.

   _P_ = _K exp(-rt) - S + C_

*  Black–Scholes parameters

   _d1_ = _( ln(S/K) + (r + 0.5 σ σ) t ) / ( σ sqrt(t) )_

   _d2_ = _d1 - σ sqrt(t)_

* Standard normal cumulative distribution function

   _N(x)_ = _0.5 ( 1 + erf ( x / sqrt(2) ) )_

## References
* [Black–Scholes model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)