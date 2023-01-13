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
parser.add_argument("--output", "-o", type=str, default="ER")
parser.add_argument("--config", "-c", type=str)
args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)

with open(args.config, 'r') as f:
    config=json.load(f)
    config.update({"agent_type":"ER", "seed":args.seed})

def set_default(d, k, v):
    if k not in d:
        d[k] = v

set_default(config, "double_DQN", False)
set_default(config, "episodic_env", False)
set_default(config, "updates_per_step", 1)
set_default(config, "save_final_params", True)
config = SimpleNamespace(**config)

Environment = getattr(environments, config.environment)

env_config = config.env_config

def get_single_sample_loss(Q_function):
    def single_sample_loss(Q_params, Q_target_params, curr_phi, action, reward, next_phi, terminal):
        Q_curr = Q_function(Q_params,curr_phi)[action]
        if(config.double_DQN):
            Q_next = jnp.where(terminal,0.0,Q_function(SG(Q_target_params),next_phi)[jnp.argmax(Q_function(SG(Q_params),next_phi))])
        else:
            Q_next = jnp.where(terminal,0.0,jnp.max(Q_function(SG(Q_target_params),next_phi)))
        return (Q_curr-(reward+config.gamma*Q_next))**2
    return single_sample_loss

def get_agent_environment_interaction_loop_function(env, Q_function, agent_opt_update, get_agent_params, replay_buffer, num_iterations):
    batch_loss = lambda *x: jnp.mean(vmap(get_single_sample_loss(Q_function), in_axes=(None,None,0,0,0,0,0))(*x))
    loss_grad = grad(batch_loss)

    def agent_environment_interaction_loop_function(env_state, agent_opt_state, Q_target_params, buffer_state, opt_t, t, key, train):
        phi = env.get_observation(env_state)
        total_reward = 0.0
        total_Q = jnp.zeros(num_actions)

        def loop_function(carry, data):
            env_state, agent_opt_state, Q_target_params, buffer_state, opt_t, t, total_reward, total_Q, phi, key = carry
            Q_curr = Q_function(get_agent_params(agent_opt_state),phi.astype(float))
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
            last_phi = phi
            key, subkey = jx.random.split(key)
            env_state, phi, reward, terminal, _ = env.step(subkey, env_state, action)
            #reset if terminated
            key, subkey = jx.random.split(key)
            env_state = jx.tree_map(lambda x,y: jnp.where(terminal, x,y), env.reset(subkey)[0], env_state)
            buffer_state = replay_buffer.add(buffer_state, last_phi, action, reward, phi, terminal)

            if(train):
                if(config.use_target):
                    Q_target_params = jx.tree_map(lambda x,y: jnp.where(t%config.target_update_frequency==0,x,y),get_agent_params(agent_opt_state),Q_target_params)
                else:
                    Q_target_params = get_agent_params(agent_opt_state)

                def update_loop_function(carry, data):
                    agent_opt_state, Q_target_params, buffer_state, opt_t = carry
                    buffer_state, sample_transitions = replay_buffer.sample(buffer_state)
                    grad = loss_grad(get_agent_params(agent_opt_state), Q_target_params, *sample_transitions)
                    agent_opt_state = agent_opt_update(opt_t, grad, agent_opt_state)
                    opt_t+=1
                    return (agent_opt_state, Q_target_params, buffer_state, opt_t), None
                carry = (agent_opt_state, Q_target_params, buffer_state, opt_t)
                (agent_opt_state, Q_target_params, buffer_state, opt_t), _ = jx.lax.scan(update_loop_function,carry, None, length=config.updates_per_step)

            total_reward+=reward
            t+=1
            return (env_state, agent_opt_state, Q_target_params, buffer_state, opt_t, t, total_reward, total_Q, phi, key), None

        key, subkey = jx.random.split(key)
        carry = (env_state, agent_opt_state, Q_target_params, buffer_state, opt_t, t, total_reward, total_Q, phi, subkey)

        (env_state, agent_opt_state, Q_target_params, buffer_state, opt_t, t, total_reward, total_Q, phi, _), _ = jx.lax.scan(loop_function, carry, None, length=num_iterations)

        metrics = {"reward_rate":total_reward/num_iterations}

        return env_state, agent_opt_state, Q_target_params, buffer_state, opt_t, t, metrics
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


def get_agent_eval_function_continuing(env, Q_function, get_agent_params, num_iterations):
    def agent_eval_function_continuing(agent_opt_state, key):
        key, subkey = jx.random.split(key)
        env_state, phi = env.reset(subkey)
        total_reward = 0.0

        def loop_function(carry, data):
            env_state, agent_opt_state, total_reward, phi, key = carry
            Q_curr = Q_function(get_agent_params(agent_opt_state),phi.astype(float))
            action = jnp.argmax(Q_curr)
            key, subkey = jx.random.split(key)
            env_state, phi, reward, terminal, _ = env.step(subkey, env_state, action)
            total_reward+=reward
            return (env_state, agent_opt_state, total_reward, phi, key), None

        key, subkey = jx.random.split(key)
        carry = (env_state, agent_opt_state, total_reward, phi, subkey)

        (env_state, agent_opt_state, total_reward, phi, _), _ = jx.lax.scan(loop_function, carry, None, length=num_iterations)

        return total_reward/num_iterations
    return jit(agent_eval_function_continuing)

class Q_function(hk.Module):
    def __init__(self, config, num_actions, name=None):
        super().__init__(name=name)
        self.num_hidden_units = config.num_hidden_units
        self.num_hidden_layers = config.num_hidden_layers
        self.activation_function = activation_dict[config.activation]
        self.num_actions = num_actions

    def __call__(self, phi):
        x = phi
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        Q = hk.Linear(self.num_actions)(x)
        return Q

########################################################################
# Replay Buffer
########################################################################

class replay_buffer:
    def __init__(self, buffer_size, batch_size, item_shapes, item_types):
        self.buffer_size = buffer_size
        self.types = item_types
        self.shapes = item_shapes
        self.batch_size = batch_size

    @partial(jit, static_argnums=(0,))
    def initialize(self, key):
        location = 0
        full = False
        buffers = [jnp.zeros([self.buffer_size]+list(s),dtype=t) for s,t in zip(self.shapes,self.types)]
        state = (location, full, buffers, key)
        return state

    @partial(jit, static_argnums=(0,))
    def add(self, state, *args):
        location, full, buffers, key = state
        # Append when the buffer is not full but overwrite when the buffer is full
        for i,(a,t) in enumerate(zip(args,self.types)):
            buffers[i]=buffers[i].at[location].set(jnp.asarray(a,dtype=t))
        full = jnp.where(location == self.buffer_size-1, True, full)
        # Increment the buffer location
        location = (location + 1) % self.buffer_size
        state = (location, full, buffers, key)
        return state

    @partial(jit, static_argnums=(0,))
    def sample(self, state):
        location, full, buffers, key = state
        key, subkey = jx.random.split(key)
        indices = jx.random.randint(subkey, minval=0, maxval=jnp.where(full, self.buffer_size, location),shape=(self.batch_size,))
        #indices = jx.random.choice(subkey,location,shape=(self.batch_size,))

        sample = []
        for b in buffers:
            sample += [b.take(indices,axis=0)]

        state = (location, full, buffers, key)
        return state, sample

env = Environment(**env_config)
key, subkey = jx.random.split(key)
env_state, phi = env.reset(subkey)
num_actions = env.num_actions()

Q_net = hk.without_apply_rng(hk.transform(lambda phi: Q_function(config,num_actions)(phi)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
Q_params = [Q_net.init(subkey, phi.astype(float)) for subkey in subkeys]
Q_func = Q_net.apply

Q_target_params = tree_stack(Q_params)

opt_init, opt_update, get_params = adamw(config.alpha_adam, eps=config.eps_adam, b1=config.b1_adam, b2=config.b2_adam, wd=config.wd_adam)
opt_states = tree_stack([opt_init(p) for p in Q_params])
opt_update = jit(opt_update)

buffer = replay_buffer(config.buffer_size, config.batch_size, (phi.shape, (), (), phi.shape, ()), (float,int,float,float,bool))

key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
buffer_states = tree_stack([buffer.initialize(subkey) for subkey in subkeys])

interaction_loop = get_agent_environment_interaction_loop_function(env, Q_func, opt_update, get_params, buffer, config.eval_frequency)

if(config.episodic_env):
    eval_agent = jit(lambda *x: jnp.mean(vmap(get_agent_eval_function_episodic(env, Q_func, get_params, config.eval_steps), in_axes=(None,0))(*x)))
else:
    eval_agent = jit(lambda *x: jnp.mean(vmap(get_agent_eval_function_continuing(env, Q_func, get_params, config.eval_steps), in_axes=(None,0))(*x)))

multiseed_interaction_loop = jit(vmap(interaction_loop, in_axes=(0,0,0,0,None,None,0,None), out_axes=(0,0,0,0,None,None,0)), static_argnames='train')

multiseed_eval_agent = jit(vmap(eval_agent, in_axes=(0,0)))

key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
env_states = tree_stack([(lambda subkey: env.reset(subkey)[0])(s) for s in subkeys])

opt_t = 0
t = 0

metrics = {"reward_rates":[], "eval_times":[]}

time_since_last_save = 0

for i in tqdm(range(config.num_steps//config.eval_frequency)):
    time = config.eval_frequency*i
    if(config.save_params and (time_since_last_save>=config.save_frequency)):
        with open(args.output+".params", 'wb') as f:
            pkl.dump({
                'Q' :[get_params(opt_state) for opt_state in tree_unstack(opt_states)]
            }, f)
        time_since_last_save = 0

    # Train step
    key, subkey = jx.random.split(key)
    subkeys = jx.random.split(subkey, num=config.num_seeds)
    env_states, opt_states, Q_target_params, buffer_states, opt_t, t, train_metrics= multiseed_interaction_loop(env_states, opt_states, Q_target_params, buffer_states, opt_t, t, subkeys, time>=config.training_start_time)

    # Evaluation step
    key, subkey = jx.random.split(key)
    subkeys = jx.random.split(subkey, num=config.num_seeds*config.eval_batch_size).reshape(config.num_seeds,config.eval_batch_size,2)
    reward_rate = multiseed_eval_agent(opt_states, subkeys)

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
            'Q' :[get_params(opt_state) for opt_state in tree_unstack(opt_states)]
        }, f)
