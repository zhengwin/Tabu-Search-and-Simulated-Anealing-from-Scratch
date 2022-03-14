import random
import math
import time


class TabuSearch():
    """ 
        Tabu Search implmentation. Used to solve QAP, TSP, N-Queens, etc.
        Uses a tabu list for recency based memory and a frequency list for long term memory.
        Continuous diversification is chosen as the diversification strategy if the frequency list
        is used.
    """

    def __init__(
        self,
        num_items,
        data,
        max_iterations = 1000,
        use_aspiration = True,
        use_freq_list = True,
        bias_factor = 2,
        objective_function = None,
        timer = None,
        threshold = None,
        print_logs = False,
    ):
        """
        num_items (int): Number of items in the problem. For example, a QAP problem with 5 facilities.
            num_items = 5.
        data (Dict[str, List[matrix]]): Data containing the problem
        max_iterations (int): Max number of iterations before termination.
        use_aspiration (bool): Whether to use aspiration criteria. The criteria used here is if the
            the best solution so far is the top candidate and is also in the tabu list, we will choose
            the candidate regardless if it is in the tabu list.
        use_freq_list (bool): Whether or not to use a frequency list. If a frequency list is chosen,
            the cost of each candidate will be applied a penalty based on the frequency of the move 
            in the frequency list. The penalty is calculated as: bias_factor * frequency of move
        bias_factor (int): This is used if freq_list is set to True. Its use is mentioned in
            use_freq_list.
        objective function (func): The objective function to calculate the cost of a solution.
            Options are "QAP", "TSP", or "NQUEENS"
        print_logs (bool): Whether to show output for each iteration.
        """
        self.tabu_list = {}
        self.num_items = num_items
        self.data = data
        self.max_iterations = max_iterations
        self.use_aspiration = use_aspiration
        self.use_freq_list = use_freq_list
        self.bias_factor = bias_factor
        self.timer = timer
        self.print_logs = print_logs
        self.costs = []
        self.threshold = threshold
        if objective_function is not None:
            if objective_function == "QAP":
                self.obj_function = self.qap
            elif objective_function == "TSP":
                self.obj_function = self.tsp
            else:
                raise ValueError("Obj function must be QSP, TSP or NQUEENS")
        else:
            self.obj_function = self.qap # default value

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

    def get_next_move(self, candidate_moves):
        """Gets the best candidate solution that is not in the tabu list.

        candidate_moves (list): List of candidate solns.
        """
        move = None
        for c in candidate_moves:
            move = c["move"]
            if move not in self.tabu_list:
                return move
            else:
                if self.print_logs:
                    print(f"Move {move} is in tabu list")

        return move

    def decrease_tabu_tenure(self):
        """Decreases each tabu tenure by 1. Remove from a move from the tabu list if it is 0."""
        moves_to_remove = []

        for move in self.tabu_list.keys():
            self.tabu_list[move] -= 1
            if self.tabu_list[move] == 0:
                moves_to_remove.append(move)

        for move in moves_to_remove:
            self.tabu_list.pop(move)

    def swap(self, arr, i , j):
        """Swap values at two indices."""
        arr[i], arr[j] = arr[j], arr[i]
    
    def remove_bias(self, candidate_solns, freq_list):
        """This is needed if a bias was added to the costs. After the candidates
        are ranked, we update the true value of costs.
        """
        for soln in candidate_solns:
            if soln["move"] in freq_list:
                soln["cost"] -= self.bias_factor * freq_list[soln["move"]]

    def solve(self, tenure = None, initial_soln = None):
        """Apply the tabu search algorithm.

        tenure (int): The tabu tenure.
        initial_soln (list[int]): Starting solution. Will be randomly generated if not specified. 
        """
        t0 = time.time()
        t1 = time.time()

        self.tabu_list = {}
        freq_list = {}
        curr_soln = initial_soln if initial_soln else self.get_initial_soln()
        if tenure is None:
            tenure = math.floor(math.sqrt(len(curr_soln)))

        n = 0
        best_soln_so_far = float('inf')
        num_solns_considered = 0
        while n < self.max_iterations:
            curr_cost = self.obj_function(curr_soln)

            if self.threshold and curr_cost < self.threshold:
                print(f"Threshold: {self.threshold}")
                print(f"Time to reach threshold: {time.time()-t0}")
                return curr_soln, curr_cost

            if self.print_logs:
                print(f"----- Iteration: {n + 1} ------")
                print(f"Current Step: {curr_soln}")
                print(f"Cost: {curr_cost}")
                print(f"Tabu List: {self.tabu_list}")

            candidate_moves = [] # candidate solutions in best to worst order.

            for i in range(self.num_items):
                for j in range(i + 1, self.num_items):
                    num_solns_considered += 1
                    if num_solns_considered % 100 == 0:
                        self.costs.append(curr_cost)

                    self.swap(curr_soln, i, j)

                    move = (min(curr_soln[i], curr_soln[j]), max(curr_soln[i], curr_soln[j]))
                    if self.use_freq_list:
                        # apply continuous diversification
                        cost = self.bias_factor * freq_list.get(move, 0) + self.obj_function(curr_soln)
                    else:
                        cost = self.obj_function(curr_soln)

                    candidate_moves.append({
                        # store a move as a tuple (i, j). Always Store in ascending order because
                        # (1, 7) and (7, 1) is the same move.
                        "move": move,
                        "cost": cost,
                    })

                    # unswap before doing next permutation
                    self.swap(curr_soln, i, j)

            # Get best candidate and move curr_soln to it.

            candidate_moves = sorted(candidate_moves, key=lambda candidate: candidate["cost"])
            if self.use_freq_list:
                # If freq list is used, then the costs have been biased.
                # we want to remove the bias after we have a sorted list.
                self.remove_bias(candidate_moves, freq_list)

            if self.use_aspiration and candidate_moves[0]["cost"] <= best_soln_so_far:
                best_soln_so_far = candidate_moves[0]["cost"]
                if candidate_moves[0]["move"] in self.tabu_list:
                    self.tabu_list.pop(candidate_moves[0]["move"])

            next_move = self.get_next_move(candidate_moves)
            move_a, move_b = next_move
            i, j = curr_soln.index(move_a), curr_soln.index(move_b) # get indices of the values
            self.swap(curr_soln, i, j) # go to the move that was chosen.

            # Increment count of move in freq list.
            if next_move in freq_list:
                freq_list[next_move] += 1
            else:
                freq_list[next_move] = 1

            # Decrease tabu tenure and add next move to the list
            self.decrease_tabu_tenure()
            self.tabu_list[next_move] = tenure

            n += 1

            if self.print_logs:
                print(f"Best soln so far: {best_soln_so_far}")
                print(f"Best Candidate Soln: {next_move}")
                print(f"Top Five Candidates: {candidate_moves[:5]}")
                print(f"\n\n\n")

            # Check solution after preset amount of time

            if self.timer is not None and time.time() - t0 > self.timer:
                print(f"Timer Length: {self.timer}")
                print(f"Actual time elapsed: {t1-t0}")
                return(curr_soln, self.obj_function(curr_soln))
        return curr_soln, self.obj_function(curr_soln), self.costs
