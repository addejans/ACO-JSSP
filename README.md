# ACO-JSSP
## Job Shop Scheduling Problem via Ant Colony Optimization

#### Introduction to Ant Colony Optimization

Combinatorial optimization problems naturally arise in the industrial world all the time. Often, having sufficient solutions to these problems can save companies millions of dollars. We
know of many examples in which optimization problems arise; for example: bus scheduling,
telecommunication network design, travelling, and vehicle routing problems. Perhaps the
most famous combinatorial optimization problem is the travelling salesman problem (TSP).
The TSP can be simply thought of as the problem of figuring out a tour of cities a salesman
must travel (visiting each city exactly once) so that the total distance travelled is minimized.

With the evergrowing arise of combinatorial optimization problems and their intrinsic link
to industrial problem, many researchers, mathematicians and computer scientists, have developed a plethora of algorithms to solve these problems. We now distinguish the two types
of algorithms; complete and approximate. Complete algorithms are able to solve these problems in such a way that in the end, the optimal solution is given. However, since many of
these problems are N P-hard the optimal solution may take a long time to obtain. This is
the reason why there are approximate algorithms. Approximate algorithms are designed, as
expected, to give approximately the optimal solution. The great thing about using an approximate algorithm is that they can obtain good results in a relatively short amount of time.

Ant colony optimization (ACO) algorithms are some of the most recent class of algorithms
designed to approximate combinatorial optimization problems. The algorithm behaves similar to real ants and their biological abilities to find the nearest food source and bring it back
to their nest. The main source of communication between ants is the depositing of chemically produced pheromone onto their paths. It is with this key idea that Marco Dorigo and
colleagues were inspired to create this new class of algorithms.
