# import gym
# from gym import spaces
import jax as jx
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial, reduce
import operator
from math import ceil

def prod(seq):
    return reduce(operator.mul, seq, 1)

class PanFlute:
    def __init__(self, num_pipes=5, spontaneous_reward=True):
        self._num_actions = num_pipes
        self.num_pipes = num_pipes

        if(spontaneous_reward):
            self.spontaneous_activation_probability = 1/self._num_actions**2
        else:
            self.spontaneous_activation_probability = 0.0

    @partial(jit, static_argnums=(0,))
    def step(self, key, env_state, action):
        pipes = env_state
        reward = prod(p[-1] for p in pipes)

        #occasionally enter rewarding state spontaneously
        key, subkey = jx.random.split(key)
        spontaneous_activation = jx.random.bernoulli(subkey, p=self.spontaneous_activation_probability)

        new_pipes = []
        for i,p in enumerate(pipes):
            p = jnp.roll(p,1)
            p = jnp.where(i==action, p.at[0].set(True), p.at[0].set(False))
            p = p.at[-1].set(jnp.where(spontaneous_activation, True, p[-1]))
            new_pipes+=[p]

        env_state = new_pipes
        return (env_state, self.get_observation(env_state), reward, False, {})

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        pipes = [jnp.zeros(i+1, dtype=bool) for i in range(self._num_actions)]
        env_state = pipes

        return env_state, self.get_observation(env_state)

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        pipes = env_state
        return jnp.concatenate(pipes)

    def num_actions(self):
        return self._num_actions

class ButtonGrid:
    def __init__(self, grid_size=10, num_buttons=3, spontaneous_reset=True, activate_on_reset=True):
        #0: no-op, 1: up, 2: left, 3: down, 4: right
        self.move_map = jnp.asarray([[0, 0], [-1,0], [0,-1], [1,0], [0,1]])

        self._num_actions = 5
        self.num_buttons = num_buttons
        self.grid_size = grid_size
        self.channels ={
            'player':0,
            'button_on':1,
            'button_off':2
        }

        self.activate_on_reset = activate_on_reset

        if(spontaneous_reset):
            self.spontaneous_reset_probability = 1/(10*self.grid_size*self.grid_size)
        else:
            self.spontaneous_reset_probability = 0.0

    @partial(jit, static_argnums=(0,))
    def step(self, key, env_state, action):
        pos, button_locations, button_states = env_state
        terminal = False

        # Success occurs only the step after all buttons are active, hence check this before changing anything
        success = jnp.all(button_states)

        # Move player
        new_pos = jnp.clip(pos+self.move_map[action], 0, self.grid_size-1)
        moved = jnp.logical_not(jnp.array_equal(pos,new_pos))
        pos = new_pos

        # Toggle button on or off if an agent steps on it
        button_states = jnp.where(jnp.logical_and(moved,jnp.logical_and(pos[0]==button_locations[0],pos[1]==button_locations[1])),jnp.logical_not(button_states), button_states)

        # Turn all buttons on at once with low probability
        key, subkey = jx.random.split(key)
        spontaneous_reset = jx.random.bernoulli(subkey, p=self.spontaneous_reset_probability)

        # Give reward if all buttons are on
        reward = jnp.where(success, 1.0, 0.0)

        reset = success

        if(self.activate_on_reset):
            button_states = jnp.where(spontaneous_reset, jnp.ones(button_states.shape[0], dtype=bool), button_states)
        else:
            reset = jnp.logical_or(reset,spontaneous_reset)

        # Reset on success
        key, subkey = jx.random.split(key)
        reset_state = self.reset(subkey)[0]
        env_state = (pos, button_locations, button_states)
        env_state = jx.tree_map(lambda x,y: jnp.where(reset,x,y), reset_state, env_state)
        return env_state, self.get_observation(env_state), reward, terminal, {}

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        key, subkey = jx.random.split(key)
        pos = jx.random.choice(subkey, self.grid_size, (2,))
        key, subkey = jx.random.split(key)
        button_locations = jnp.unravel_index(jx.random.choice(subkey, self.grid_size*self.grid_size, shape=(self.num_buttons,), replace=False), (self.grid_size,self.grid_size))
        button_states = jnp.zeros((self.num_buttons,), dtype=bool)
        env_state = (pos, button_locations, button_states)
        return env_state, self.get_observation(env_state)

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        pos, button_locations, button_states = env_state
        obs = jnp.zeros((self.grid_size, self.grid_size, len(self.channels)), dtype=bool)
        obs = obs.at[pos[0],pos[1],self.channels['player']].set(True)
        obs = obs.at[button_locations[0],button_locations[1], jnp.full(self.num_buttons,self.channels['button_on'])].set(button_states)
        obs = obs.at[button_locations[0],button_locations[1], jnp.full(self.num_buttons,self.channels['button_off'])].set(jnp.logical_not(button_states))
        # Flatten obs so we can input to a feed forward network, could skip this if you want to use a conv net
        return jnp.ravel(obs)

    def num_actions(self):
        return self._num_actions

class ProcMaze:
    def __init__(self, grid_size=10, spontaneous_termination=True, teleport_on_termination=True):
        self.move_map = jnp.asarray([[0, 0], [-1,0], [0,-1], [1,0], [0,1]])

        self._num_actions = 5
        self.grid_size = grid_size

        #1/10th as often as the optimal time to solve the worst case maze for a given grid size
        if(spontaneous_termination):
            self.spontaneous_goal_probability=0.1/((self.grid_size+1)*ceil(self.grid_size/2)-self.grid_size%2)
        else:
            self.spontaneous_goal_probability=0.0

        self.teleport_on_termination = teleport_on_termination

        self.channels ={
            'player':0,
            'goal':1,
            'wall':2,
            'empty':3
        }

    @partial(jit, static_argnums=(0,))
    def step(self, key, env_state, action):
        goal, wall_grid, pos = env_state
        terminal = False

        # Reset the step after if goal is reached, so agent sees the state where pos==goal
        terminal = jnp.array_equal(pos, goal)

        # punish agent for each step until termination
        reward = -1

        # Move if the new position is on the grid and open
        new_pos = jnp.clip(pos+self.move_map[action], 0, self.grid_size-1)
        pos = jnp.where(jnp.logical_not(wall_grid[new_pos[0], new_pos[1]]), new_pos, pos)

        # With small probability, teleport to goal
        key, subkey = jx.random.split(key)
        spontanteous_goal = jx.random.bernoulli(subkey, p=self.spontaneous_goal_probability)
        if(self.teleport_on_termination):
            pos = jnp.where(spontanteous_goal, goal, pos)
        else:
            terminal = jnp.logical_or(terminal, spontanteous_goal)

        env_state = goal, wall_grid, pos

        return env_state, self.get_observation(env_state), reward, terminal, {}

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        def push(stack, top, x):
            stack= stack.at[top].set(x)
            top+=1
            return stack, top

        def pop(stack, top):
            top-=1
            return stack[top], top

        #takes and flattened index, returns neighbours as (x,y) pair
        def neighbours(cell):
            coord_tuple = jnp.unravel_index(cell, (self.grid_size, self.grid_size))
            coord_array = jnp.stack(list(coord_tuple))
            return coord_array+self.move_map

        #takes (x,y) pair
        def can_expand(cell, visited):
            # A neighbour can be expanded as long as it is on the grid, it has not been visited, and it's only visited neighbour is the current cell
            flat_cell = jnp.ravel_multi_index((cell[0],cell[1]),(self.grid_size,self.grid_size),mode='clip')
            not_visited = jnp.logical_not(visited[flat_cell])
            ns = neighbours(flat_cell)
            ns_on_grid = jnp.all(jnp.logical_and(ns>=0,ns<=self.grid_size-1), axis=1)
            flat_ns = jnp.ravel_multi_index((ns[:,0],ns[:,1]),(self.grid_size,self.grid_size),mode='clip')
            # Only count neighbours which are actually on the grid
            only_one_visited_neighbor = jnp.equal(jnp.sum(jnp.logical_and(visited[flat_ns],ns_on_grid)),1)
            on_grid = jnp.all(jnp.logical_and(cell>=0,cell<=self.grid_size-1))
            return jnp.logical_and(jnp.logical_and(not_visited,only_one_visited_neighbor),on_grid)
        can_expand = vmap(can_expand, in_axes=(0,None))

        wall_grid = jnp.ones((self.grid_size, self.grid_size), dtype=bool)

        #Visited node map
        visited = jnp.zeros(self.grid_size*self.grid_size, dtype=bool)

        #big enough to hold every location in the grid, indices should be flattened to store here
        stack = jnp.zeros(self.grid_size*self.grid_size, dtype=int)
        top = 0

        #Location of current cell being searched
        key, subkey = jx.random.split(key)
        curr = jx.random.choice(subkey, self.grid_size, (2,))
        flat_curr = jnp.ravel_multi_index((curr[0],curr[1]),(self.grid_size,self.grid_size),mode='clip')
        wall_grid = wall_grid.at[curr[0], curr[1]].set(False)

        visited = visited.at[flat_curr].set(True)
        stack, top = push(stack,top, flat_curr)

        def cond_fun(carry):
            visited, stack, top, wall_grid, key = carry
            #continue until stack is empty
            return top!=0

        def body_fun(carry):
            visited, stack, top, wall_grid, key = carry
            curr, top = pop(stack,top)
            ns = neighbours(curr)
            flat_ns = jnp.ravel_multi_index((ns[:,0],ns[:,1]),(self.grid_size,self.grid_size),mode='clip')

            expandable = can_expand(ns,visited)

            has_expandable_neighbour = jnp.any(expandable)

            # This will all be used only conditioned on has_expandable neighbor
            _stack, _top = push(stack, top, curr)
            key, subkey = jx.random.split(key)
            selected = jx.random.choice(subkey, flat_ns,p=expandable/jnp.sum(expandable))
            _stack, _top = push(_stack, _top, selected)
            _wall_grid = wall_grid.at[jnp.unravel_index(selected, (self.grid_size, self.grid_size))].set(False)
            _visited = visited.at[selected].set(True)

            stack = jnp.where(has_expandable_neighbour, _stack, stack)
            top = jnp.where(has_expandable_neighbour, _top, top)
            wall_grid = jnp.where(has_expandable_neighbour, _wall_grid, wall_grid)
            visited = jnp.where(has_expandable_neighbour, _visited, visited)
            return (visited, stack, top, wall_grid, key)



        key, subkey = jx.random.split(key)
        carry = (visited, stack, top, wall_grid, subkey)
        visited, stack, top, wall_grid, key = jx.lax.while_loop(cond_fun, body_fun, carry)

        flat_open = jnp.logical_not(jnp.ravel(wall_grid))

        key, subkey = jx.random.split(key)
        pos = jx.random.choice(subkey, self.grid_size*self.grid_size, p=flat_open/jnp.sum(flat_open))
        pos = jnp.stack(list(jnp.unravel_index(pos, (self.grid_size, self.grid_size))))
        key, subkey = jx.random.split(key)
        goal = jx.random.choice(subkey, self.grid_size*self.grid_size, p=flat_open/jnp.sum(flat_open))
        goal = jnp.stack(list(jnp.unravel_index(goal, (self.grid_size, self.grid_size))))

        wall_grid = wall_grid.at[goal[0], goal[1]].set(False)
        wall_grid = wall_grid.at[pos[0], pos[1]].set(False)

        env_state = goal, wall_grid, pos
        return env_state, self.get_observation(env_state)

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        goal, wall_grid, pos = env_state
        obs = jnp.zeros((self.grid_size, self.grid_size, len(self.channels)), dtype=bool)
        obs = obs.at[pos[0],pos[1],self.channels['player']].set(True)
        obs = obs.at[goal[0],goal[1],self.channels['goal']].set(True)
        obs = obs.at[:,:,self.channels['wall']].set(wall_grid)
        obs = obs.at[:,:,self.channels['empty']].set(jnp.logical_not(wall_grid))
        # Flatten obs so we can input to a feed forward network, could skip this if you want to use a conv net
        return jnp.ravel(obs)

    def num_actions(self):
        return self._num_actions

class OpenGrid:
    def __init__(self, grid_size=10, spontaneous_termination=True, teleport_on_termination=True):
        #0: no-op, 1: up, 2: left, 3: down, 4: right
        self.move_map = jnp.asarray([[0, 0], [-1,0], [0,-1], [1,0], [0,1]])

        self._num_actions = 5
        self.grid_size = grid_size
        self.channels ={
            'player':0
        }

        self.goal = jnp.asarray([grid_size-1,grid_size-1])

        #1/10th as often as the optimal time to solve the worst case layout for gridsize
        if(spontaneous_termination):
            self.spontaneous_goal_probability=0.1/self.grid_size
        else:
            self.spontaneous_goal_probability=0.0

        self.teleport_on_termination = teleport_on_termination

    @partial(jit, static_argnums=(0,))
    def step(self, key, env_state, action):
        # print(env_state)
        pos = env_state
        terminal = False

        # Reset the step after if goal is reached, so agent sees the state where pos==goal
        terminal = jnp.array_equal(pos, self.goal)

        # punish agent for each step until termination
        reward = -1

        # Move if the new position is on the grid
        pos = jnp.clip(pos+self.move_map[action], 0, self.grid_size-1)

        # With small probability, teleport to goal
        key, subkey = jx.random.split(key)
        spontanteous_goal = jx.random.bernoulli(subkey, p=self.spontaneous_goal_probability)
        if(self.teleport_on_termination):
            pos = jnp.where(spontanteous_goal, self.goal, pos)
        else:
            terminal = jnp.logical_or(terminal, spontanteous_goal)

        env_state = pos

        return env_state, self.get_observation(env_state), reward, terminal, {}

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        key, subkey = jx.random.split(key)
        pos = jx.random.choice(subkey, self.grid_size, (2,))
        env_state = pos
        return env_state, self.get_observation(env_state)

    @partial(jit, static_argnums=(0,))
    def get_observation(self, env_state):
        pos = env_state
        obs = jnp.zeros((self.grid_size, self.grid_size, len(self.channels)), dtype=bool)
        obs = obs.at[pos[0],pos[1],self.channels['player']].set(True)
        # Flatten obs so we can input to a feed forward network, could skip this if you want to use a conv net
        return jnp.ravel(obs)

    def num_actions(self):
        return self._num_actions
