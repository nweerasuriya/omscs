class SemanticNetsAgent:
    def __init__(self):
        # If you want to do any initial processing, add it here.
        pass

    def _is_valid_state(self, sheep, wolves):
        """
        Check if wolves do not outnumber sheep on either side of the river.
        """
        if sheep < 0 or wolves < 0:
            return False
        if sheep > 0 and wolves > sheep:
            return False
        return True

    def solve(self, initial_sheep, initial_wolves):
        # Add your code here! Your solve method should receive
        # the initial number of sheep and wolves as integers,
        # and return a list of 2-tuples that represent the moves
        # required to get all sheep and wolves from the left
        # side of the river to the right.
        #
        # If it is impossible to move the animals over according
        # to the rules of the problem, return an empty list of
        # moves.

        # Define the initial state and goal state (last one is where the boat is)
        initial_state = (initial_sheep, initial_wolves, "left")
        goal_state = (0, 0, "right")
        print(f"Initial state: {initial_state}, Goal state: {goal_state}")

        move_sequence = []
        possible_move_set = [(1, 1), (2, 0), (0, 2), (1, 0), (0, 1)]

        state_list = [initial_state]
        visited_states = {
            initial_state: {
                "Node": None,
                "Move": None,
            }
        }
        # Use BFS not a greedy approach to find the solution
        while state_list:
            state = state_list.pop(0)
            # If goal state has been reached, find the path to the goal state and return it
            if state == goal_state:
                while visited_states[state]["Node"] is not None:
                    move_sequence.append(visited_states[state]["Move"])
                    state = visited_states[state]["Node"]
                move_sequence.append(visited_states[state]["Move"])
                move_sequence = [move for move in move_sequence if move is not None]
                return move_sequence[::-1]  # Reverse the list to get the correct order

            sheep, wolves, boat_side = state
            for move in possible_move_set:
                if boat_side == "left":
                    new_sheep = sheep - move[0]
                    new_wolves = wolves - move[1]
                    new_boat_side = "right"
                else:
                    new_sheep = sheep + move[0]
                    new_wolves = wolves + move[1]
                    new_boat_side = "left"

                # Check for non negative and not exceeding initial values
                if not (
                    0 <= new_sheep <= initial_sheep
                    and 0 <= new_wolves <= initial_wolves
                ):
                    continue

                new_state = (new_sheep, new_wolves, new_boat_side)
                # Check if more wolves than sheep on other side of the river
                other_sheep = initial_sheep - new_sheep
                other_wolves = initial_wolves - new_wolves

                if (
                    self._is_valid_state(new_sheep, new_wolves)
                    and self._is_valid_state(other_sheep, other_wolves)
                    and new_state not in visited_states
                ):
                    visited_states[new_state] = {
                        "Node": state,
                        "Move": move,
                    }
                    state_list.append(new_state)

        return []  # Return an empty list if no solution is found
