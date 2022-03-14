
import time
import random
import math
import numpy as np


class SimulatedAnnealing():
    """ 
        Simulated Annealing implmentation. Used to solve QAP, TSP, N-Queens, etc.
        Uses a cooling schedule to move from solution space exploration to exploitation.
    """

    def __init__(
        self,
        num_items,
        t_stop = 0,
        temp_iter = 50,
        temp = 300,
        alpha = 0.7,
        #k = 1.38064852 * 10**-23, # Boltzmann constant,
        data = None,
        objective_function = None,
        print_logs = False,
        timer = None,
        threshold = None,
    ):
        """
        num_items (int): Number of items in the problem. For example, a QAP problem with 5 facilities.
            num_items = 5.
        t_stop (int): Stopping temperature.
        temp_iter (int): Number of iterations at each temperature.
        temp (float): Starting temperature.
        alpha (float): Temperature decrement.
        k (float): Constant (in physical annealing, the Boltzmann constant) affecting probability of accepting a worse solution.
        data (array): Data for a specific problem.
        timer (float): If specified, stop search after this amount of time.
        threshold (float): If specified, stop search after this cost value is reached.
        objective function (str): The objective function to calculate the cost of a solution.
        print_logs (bool): Whether to show output for each iteration.
        """
        self.num_items = num_items
        self.t_stop = t_stop
        self.temp_iter = temp_iter
        self.temp = temp
        self.alpha = alpha
        self.timer = timer
        self.costs = []
        self.threshold = threshold
        #self.k = k
        self.print_logs = print_logs
        self.data = data
        if objective_function:
            if objective_function == "tsp":
                self.obj_function = self.tsp
            elif objective_function == "nqueens":
                self.obj_function = self.nqueens
        else:
            self.obj_function = self.default_obj_function

    def get_initial_soln(self):
        """Generate a random starting solution."""
        initial_soln = list(range(1, self.num_items + 1))
        random.shuffle(initial_soln)
        return initial_soln

    def qap(self, candidate):
        """Default objective function. Calculates distance and Flow for QAP. 

        candidate (list): a candidate soln.
        """
        cost = 0
        for i, facility_i in enumerate(candidate):
            for j in range(i, len(candidate)):
                cost += self.data["flow"][facility_i - 1][candidate[j] - 1] * self.data["distance"][i][j]
        return cost

    def tsp(self, candidate):
        cost = 0
        for i in range(len(candidate) - 1):
            cost += self.data[candidate[i] - 1][candidate[i + 1] - 1]
        cost += self.data[candidate[0] - 1][candidate[-1] - 1]
        return cost

    def swap(self, arr, i , j):
        """Swap values at two indices."""
        arr[i], arr[j] = arr[j], arr[i]

    def solve(self, initial_soln = None):
        """Apply the SA algorithm.

        initial_soln (list[int]): Starting solution. Will be randomly generated if not specified. 
        """
        t1 = time.time()
        t0 = time.time()

        curr_soln = initial_soln if initial_soln else self.get_initial_soln()
        best_soln_so_far = float('inf')

        t = self.temp

        num_solns_considered = 0
        while t > self.t_stop:
            for _ in np.arange(self.temp_iter):
                cost_curr = self.obj_function(curr_soln)
                num_solns_considered += 1
                if num_solns_considered % 100 == 0:
                    self.costs.append(cost_curr)

                [i, j] = np.random.permutation(np.arange(self.num_items))[0:2]
                self.swap(curr_soln, i, j)
                cost_new = self.obj_function(curr_soln)
                
                #move = (min(curr_soln[i], curr_soln[j]), max(curr_soln[i], curr_soln[j]))
            
                #candidate_move = {
                #    "move": move,
                #    "cost": cost_ew,
                #}

                cost_delta = cost_new - cost_curr
                if cost_delta > 0:
                    #P = math.exp(-cost_delta/ (self.k * t))
                    P = math.exp(-cost_delta/ t)
                    if random.uniform(0, 1) >= P:
                        # unswap - current solution is kept
                        self.swap(curr_soln, i, j)
                        cost_new = cost_curr

                if cost_curr < best_soln_so_far:
                    best_soln_so_far = cost_curr

            t = t - self.alpha

            if self.print_logs:
                print(f"Best soln so far: {best_soln_so_far}")
                print(f"Next Soln: {curr_soln}")
                print(f"Next Soln Cost: {cost_curr}")
                print(f"\n")

            # Check solution after preset amount of time

            if self.timer is not None and time.time() - t0 > self.timer:
                print(f"Timer Length: {self.timer}")
                print(f"Actual time elapsed: {t1-t0}")
                return(curr_soln, self.obj_function(curr_soln))

            # Check time to reach preset solution cost threshold
            if self.threshold and cost_new < self.threshold:
                print(f"Threshold: {self.threshold}")
                print(f"Time to reach threshold: {time.time()-t0}")

                return(curr_soln, self.obj_function(curr_soln))
        return curr_soln, self.obj_function(curr_soln), self.costs

