# Monte Carlo Pi
* Evaluate the value of `pi` numerically
* Area of circle `S_c = pi * r * r`
* Area of square `S_s = 2 * r * 2 * r = 4 * r * r`
* `pi = 4 * S_c / S_r`
* Generate random points within square
* Count the number of points in the circle: `r > sqrt(x * x + y * y)`
* `pi = 4 * points_in_circle / total_number_points`
