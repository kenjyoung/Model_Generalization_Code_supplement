import jax as jx
import jax.numpy as jnp
import environments

from jax import grad, jit, vmap

from jax.lax import stop_gradient as SG

from optimizers import adamw

import haiku as hk

import argparse

import json

from functools import partial

from tqdm import tqdm

import pickle as pkl

from types import SimpleNamespace

from tree_utils import tree_stack, tree_unstack

activation_dict = {"relu": jx.nn.relu, "silu": jx.nn.silu, "elu": jx.nn.elu}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int, default=0)
parser.add_argument("--output", "-o", type=str, default="perfect_model")
parser.add_argument("--config", "-c", type=str)
args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)

with open(args.config, 'r') as f:
    config=json.load(f)
    config.update({"agent_type":"perfect_model", "seed":args.seed})

def set_default(d, k, v):
    if k not in d:
        d[k] = v

set_default(config, "double_DQN", False)
set_default(config, "episodic_env", False)
set_default(config, "updates_per_step", 1)
config = SimpleNamespace(**config)

Environment = getattr(environments, config.environment)

env_config = config.env_config

min_denom = 0.000001

########################################################################
# Networks
########################################################################

class Q_function(hk.Module):
    def __init__(self, config, num_actions, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config.num_hidden_units
        self.num_hidden_layers = config.num_hidden_layers
        self.activation_function = activation_dict[config.activation]
        self.num_actions = num_actions

    def __call__(self, phi):
        x = jnp.ravel(phi)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        Q = hk.Linear(self.num_actions)(x)
        return Q

########################################################################
# Losses
########################################################################

def get_single_sample_Q_loss(Q_function):
    def single_sample_Q_loss(Q_params, Q_target_params, curr_obs, action, reward, next_obs, terminal, weight):
        Q_curr = Q_function(Q_params,curr_obs)[action]
        if(config.double_DQN):
            Q_next = jnp.where(terminal,0.0,Q_function(SG(Q_target_params),next_obs)[jnp.argmax(Q_function(SG(Q_params),next_obs))])
        else:
            Q_next = jnp.where(terminal,0.0,jnp.max(Q_function(SG(Q_target_params),next_obs)))
        return weight*(Q_curr-(reward+config.gamma*Q_next))**2
    return single_sample_Q_loss

def get_single_model_rollout_func(env, Q_function, rollout_length, num_actions):
    def single_model_rollout_func(env_state, Q_params, key, t=0):
        def loop_function(carry, data):
            env_state, obs, terminal, weight, key = carry

            weight = jnp.where(terminal, 0.0, weight)

            #reset if terminated
            # key, subkey = jx.random.split(key)
            # reset_state, reset_obs = env.reset(subkey)
            # env_state = jx.tree_map(lambda x,y: jnp.where(terminal, x,y), reset_state, env_state)
            # last_obs = jnp.where(terminal, reset_obs, obs)
            last_obs = obs

            Q_curr = Q_function(Q_params, obs.astype(float))
            if(config.exploration_strat=="epsilon_greedy"):
                key, subkey = jx.random.split(key)
                randomize_action = jx.random.bernoulli(subkey, config.epsilon)
                key, subkey = jx.random.split(key)
                action = jnp.where(randomize_action, jx.random.choice(subkey, Q_curr.shape[0]), jnp.argmax(Q_curr))
            elif(config.exploration_strat=="softmax"):
                key, subkey = jx.random.split(key)
                action = jx.random.categorical(subkey, Q_curr/config.softmax_temp)
            else:
                raise ValueError("Unknown Exploration Strategy.")

            key, subkey = jx.random.split(key)
            env_state, obs, reward, terminal, _ = env.step(subkey, env_state, action)

            return (env_state, obs, terminal, weight, key), (last_obs, action, reward, obs, terminal, weight)

        key, subkey = jx.random.split(key)
        obs = env.get_observation(env_state)

        key, subkey = jx.random.split(key)
        _, sample_transitions = jx.lax.scan(loop_function, (env_state, obs, False, 1.0, subkey), None, length=rollout_length)
        return sample_transitions
    return single_model_rollout_func

def get_agent_environment_interaction_loop_function(env, Q_function, Q_opt_update, get_Q_params, state_buffer, num_iterations, num_actions):
    batch_Q_loss = lambda *x: jnp.mean(vmap(get_single_sample_Q_loss(Q_function), in_axes=(None,None,0,0,0,0,0,0))(*x))
    Q_loss_grad = grad(batch_Q_loss)

    #get batch of rollouts and combine the batch and step dimensions
    batch_model_rollout= lambda *x: [jnp.reshape(y,(config.batch_size*config.rollout_length,-1)) for y in vmap(get_single_model_rollout_func(env, Q_function, config.rollout_length, num_actions),in_axes=(0,None,0,None))(*x)]

    def agent_environment_interaction_loop_function(env_state, Q_opt_state, Q_target_params, buffer_state, opt_t, t, key, train):
        obs = env.get_observation(env_state)
        total_reward = 0.0
        total_Q = jnp.zeros(num_actions)

        def loop_function(carry, data):
            env_state, Q_opt_state, Q_target_params, buffer_state, opt_t, t, total_reward, total_Q, obs, key = carry
            buffer_state = state_buffer.add(buffer_state, env_state)
            key, subkey = jx.random.split(key)
            Q_curr = Q_function(get_Q_params(Q_opt_state), obs.astype(float))
            total_Q+=Q_curr
            if(config.exploration_strat=="epsilon_greedy"):
                key, subkey = jx.random.split(key)
                randomize_action = jx.random.bernoulli(subkey, config.epsilon)
                key, subkey = jx.random.split(key)
                action = jnp.where(randomize_action, jx.random.choice(subkey, Q_curr.shape[0]), jnp.argmax(Q_curr))
            elif(config.exploration_strat=="softmax"):
                key, subkey = jx.random.split(key)
                action = jx.random.categorical(subkey, Q_curr/config.softmax_temp)
            else:
                raise ValueError("Unknown Exploration Strategy.")
            key, subkey = jx.random.split(key)
            env_state, obs, reward, terminal, _ = env.step(subkey, env_state, action)
            #reset if terminated
            key, subkey = jx.random.split(key)
            env_state = jx.tree_map(lambda x,y: jnp.where(terminal, x,y), env.reset(subkey)[0], env_state)
            if(train):
                if(config.use_target):
                    Q_target_params = jx.tree_map(lambda x,y: jnp.where(t%config.target_update_frequency==0,x,y),get_Q_params(Q_opt_state),Q_target_params)
                else:
                    Q_target_params = get_Q_params(Q_opt_state)

                def update_loop_function(carry, data):
                    Q_opt_state, Q_target_params, buffer_state, opt_t, key = carry
                    buffer_state, sample_states = state_buffer.sample(buffer_state)

                    key, subkey = jx.random.split(key)
                    subkeys = jx.random.split(subkey, num=config.batch_size)
                    model_transitions = batch_model_rollout(sample_states, get_Q_params(Q_opt_state), subkeys, opt_t)

                    Q_grad = Q_loss_grad(get_Q_params(Q_opt_state), Q_target_params, *model_transitions)

                    Q_grad = Q_loss_grad(get_Q_params(Q_opt_state), Q_target_params, *model_transitions)
                    Q_opt_state = Q_opt_update(opt_t, Q_grad, Q_opt_state)
                    opt_t+=1
                    return (Q_opt_state, Q_target_params, buffer_state, opt_t, key), None
                key, subkey = jx.random.split(key)
                carry = (Q_opt_state, Q_target_params, buffer_state, opt_t, subkey)
                (Q_opt_state, Q_target_params, buffer_state, opt_t, _), _ = jx.lax.scan(update_loop_function,carry, None, length=config.updates_per_step)

            total_reward+=reward
            t+=1
            return (env_state, Q_opt_state, Q_target_params, buffer_state, opt_t, t, total_reward, total_Q, obs, key), None

        key, subkey = jx.random.split(key)
        carry = (env_state, Q_opt_state, Q_target_params, buffer_state, opt_t, t, total_reward, total_Q, obs, subkey)

        (env_state, Q_opt_state, Q_target_params, buffer_state, opt_t, t, total_reward, total_Q, obs, _), _ = jx.lax.scan(loop_function, carry, None, length=num_iterations)

        return env_state, Q_opt_state, Q_target_params, buffer_state, opt_t, t, total_reward/num_iterations, total_Q/num_iterations
    return jit(agent_environment_interaction_loop_function, static_argnames=('train',))

def get_agent_eval_function_episodic(env, Q_function, get_agent_params, num_iterations):
    def agent_eval_function_episodic(agent_opt_state, key):
        key, subkey = jx.random.split(key)
        env_state, phi = env.reset(subkey)
        total_reward = 0.0

        def loop_function(carry, data):
            env_state, agent_opt_state, total_reward, phi, terminated, key = carry
            Q_curr = Q_function(get_agent_params(agent_opt_state),phi.astype(float))
            action = jnp.argmax(Q_curr)
            key, subkey = jx.random.split(key)
            env_state, phi, reward, terminal, _ = env.step(subkey, env_state, action)
            total_reward+=reward*jnp.logical_not(terminated)
            terminated = jnp.logical_or(terminated,terminal)
            return (env_state, agent_opt_state, total_reward, phi, terminated, key), None

        key, subkey = jx.random.split(key)
        carry = (env_state, agent_opt_state, total_reward, phi, False, subkey)

        (env_state, agent_opt_state, total_reward, phi, _, _), _ = jx.lax.scan(loop_function, carry, None, length=num_iterations)

        return total_reward
    return jit(agent_eval_function_episodic)

def get_agent_eval_function_continuing(env, Q_function, get_Q_params, num_iterations):
    def agent_eval_function_continuing(Q_opt_state, key):
        key, subkey = jx.random.split(key)
        env_state, obs = env.reset(subkey)
        total_reward = 0.0

        def loop_function(carry, data):
            env_state, Q_opt_state, total_reward, obs, key = carry
            key, subkey = jx.random.split(key)
            Q_curr = Q_function(get_Q_params(Q_opt_state),obs.astype(float))
            key, subkey = jx.random.split(key)
            action = jnp.argmax(Q_curr)
            key, subkey = jx.random.split(key)
            env_state, obs, reward, terminal, _ = env.step(subkey, env_state, action)
            total_reward+=reward
            return (env_state, Q_opt_state, total_reward, obs, key), None

        key, subkey = jx.random.split(key)
        carry = (env_state, Q_opt_state, total_reward, obs, subkey)

        (env_state, Q_opt_state, total_reward, obs, _), _ = jx.lax.scan(loop_function, carry, None, length=num_iterations)

        return total_reward/num_iterations
    return jit(agent_eval_function_continuing)

########################################################################
# Define Replay Buffer
########################################################################

# Note: Replay buffer here operates on env_states explicitly
class state_buffer:
    def __init__(self, buffer_size, batch_size, dummy_env_state):
        self.buffer_size = buffer_size
        self.dummy_env_state = dummy_env_state
        self.batch_size = batch_size

    @partial(jit, static_argnums=(0,))
    def initialize(self, key):
        location = 0
        full = False
        buffers = jx.tree_map(lambda x: jnp.zeros([self.buffer_size]+list(x.shape), dtype=x.dtype), self.dummy_env_state)
        return (location, full, buffers, key)

    @partial(jit, static_argnums=(0,))
    def add(self, buffer_state, env_state):
        location, full, buffers, key = buffer_state
        buffers = jx.tree_map(lambda x, y: x.at[location].set(y), buffers, env_state)
        full = jnp.where(location == self.buffer_size-1, True, full)
        location = (location + 1) % self.buffer_size
        state = (location, full, buffers, key)
        return state

    @partial(jit, static_argnums=(0,))
    def sample(self, state):
        location, full, buffers, key = state
        key, subkey = jx.random.split(key)
        indices = jx.random.randint(subkey, minval=0, maxval=jnp.where(full, self.buffer_size, location),shape=(self.batch_size,))

        sample = jx.tree_map(lambda x: x.take(indices, axis=0), buffers)

        state = (location, full, buffers, key)
        return state, sample

env = Environment(**env_config)
key, subkey = jx.random.split(key)
env_state, obs = env.reset(subkey)
num_actions = env.num_actions()

Q_net = hk.without_apply_rng(hk.transform(lambda phi: Q_function(config,num_actions)(phi)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
Q_params = [Q_net.init(subkey, obs.astype(float)) for subkey in subkeys]
Q_func = Q_net.apply

Q_target_params = tree_stack(Q_params)

Q_opt_init, Q_opt_update, get_Q_params = adamw(config.alpha_adam, eps=config.eps_adam, b1=config.b1_adam, b2=config.b2_adam, wd=config.wd_adam)
Q_opt_states = tree_stack([Q_opt_init(p) for p in Q_params])
Q_opt_update = jit(Q_opt_update)

buffer = state_buffer(config.buffer_size, config.batch_size, env_state)

key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
buffer_states = tree_stack([buffer.initialize(subkey) for subkey in subkeys])

interaction_loop = get_agent_environment_interaction_loop_function(env, Q_func, Q_opt_update, get_Q_params, buffer, config.eval_frequency, num_actions)

if(config.episodic_env):
    eval_agent = jit(lambda *x: jnp.mean(vmap(get_agent_eval_function_episodic(env, Q_func, get_Q_params, config.eval_steps), in_axes=(None,0))(*x)))
else:
    eval_agent = jit(lambda *x: jnp.mean(vmap(get_agent_eval_function_continuing(env, Q_func, get_Q_params, config.eval_steps), in_axes=(None,0))(*x)))

multiseed_interaction_loop = jit(vmap(interaction_loop, in_axes=(0,0,0,0,None,None,0,None), out_axes=(0,0,0,0,None,None,0,0)), static_argnames='train')
multiseed_eval_agent = jit(vmap(eval_agent, in_axes=(0,0)))

key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
env_states = tree_stack([(lambda subkey: env.reset(subkey)[0])(s) for s in subkeys])

opt_t = 0
t = 0

metrics = {"reward_rates":[], "eval_times": []}

time_since_last_save = 0

for i in tqdm(range(config.num_steps//config.eval_frequency)):
    time = config.eval_frequency*i
    if(config.save_params and (time_since_last_save>=config.save_frequency)):
        with open(args.output+".params", 'wb') as f:
            pkl.dump({
                'Q' :[get_Q_params(opt_state) for opt_state in tree_unstack(Q_opt_states)]
            }, f)
        time_since_last_save = 0

    # Train step
    key, subkey = jx.random.split(key)
    subkeys = jx.random.split(subkey, num=config.num_seeds)
    env_states, Q_opt_states, Q_target_params, buffer_states, opt_t, t, _, _ = multiseed_interaction_loop(env_states, Q_opt_states, Q_target_params, buffer_states, opt_t, t, subkeys, time>=config.training_start_time)

    # Evaluation step
    key, subkey = jx.random.split(key)
    subkeys = jx.random.split(subkey, num=config.num_seeds*config.eval_batch_size).reshape(config.num_seeds,config.eval_batch_size,2)
    reward_rate = multiseed_eval_agent(Q_opt_states, subkeys)

    # Logging
    metrics["reward_rates"]+=[reward_rate]
    metrics["eval_times"]+=[time]
    log_dict = {"reward_rate":reward_rate, "time": time}
    write_string ="| ".join([k+": "+str(v) for k,v in log_dict.items()])
    tqdm.write(write_string)
    time_since_last_save+=config.eval_frequency

with open(args.output+".out", 'wb') as f:
    pkl.dump({
        'config': config,
        'metrics':metrics
    }, f)

# save params once more at the end
if(config.save_params):
    with open(args.output+".params", 'wb') as f:
        pkl.dump({
            'Q' :[get_Q_params(opt_state) for opt_state in tree_unstack(Q_opt_states)]
        }, f)
