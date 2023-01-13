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
parser.add_argument("--output", "-o", type=str, default="simple_model")
parser.add_argument("--config", "-c", type=str)
args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)

with open(args.config, 'r') as f:
    config=json.load(f)
    config.update({"agent_type":"simple_model", "seed":args.seed})

def set_default(d, k, v):
    if k not in d:
        d[k] = v

set_default(config, "double_DQN", False)
set_default(config, "episodic_env", False)
set_default(config, "updates_per_step", 1)
set_default(config, "save_params", True)
config = SimpleNamespace(**config)

Environment = getattr(environments, config.environment)

env_config = config.env_config

min_denom = 0.000001

########################################################################
# Probability Helper Functions
########################################################################

def log_gaussian_probability(x, params):
    mu = params['mu']
    sigma = params['sigma']
    return -(jnp.log(sigma) + 0.5 * jnp.log(2 * jnp.pi) + 0.5 * ((x - mu) / sigma)**2)


def log_binary_probability(x, params):
    logit = params['logit']
    return jnp.where(x, jx.nn.log_sigmoid(logit), jx.nn.log_sigmoid(-logit))


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

    def __call__(self, obs):
        x = jnp.ravel(obs)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        Q = hk.Linear(self.num_actions)(x)
        return Q

class reward_function(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]

    def __call__(self, obs, action, key=None):
        x = jnp.concatenate([jnp.ravel(obs),action])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        mu = hk.Linear(1)(x)[0]
        sigma = jnp.ones(mu.shape)
        return {'mu':mu, 'sigma':sigma}

class termination_function(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]

    def __call__(self, obs, action, key=None):
        x = jnp.concatenate([jnp.ravel(obs),action])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        logit = hk.Linear(1)(x)[0]
        return {'logit':logit}

class next_obs_function(hk.Module):
    def __init__(self, config, obs_width, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.obs_width = obs_width
        self.binary_obs = config.binary_obs
        self.activation_function = activation_dict[config.activation]

    def __call__(self, obs, action, key):
        x = jnp.concatenate([jnp.ravel(obs),action])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        if(self.binary_obs):
            logit = hk.Linear(self.obs_width)(x)
            x = jx.random.bernoulli(key, logit)
            return x.astype(float), {'logit':logit}
        else:
            mu = hk.Linear(self.obs_width)(x)
            sigma = jnp.ones(mu.shape)
            x = mu+sigma*jx.random.normal(key,mu.shape)
            return x, {'mu':mu, 'sigma':sigma}

########################################################################
# Losses
########################################################################

def get_single_sample_model_loss(model_functions, binary_obs):
    def single_sample_model_loss(model_params, curr_obs, action, reward, next_obs, terminal, key):
        reward_network = model_functions['reward']
        termination_network = model_functions['termination']
        next_obs_network = model_functions['next_obs']

        reward_params = model_params['reward']
        termination_params = model_params['termination']
        next_obs_params = model_params['next_obs']

        key, subkey = jx.random.split(key)
        _, o_hat_dist = next_obs_network(next_obs_params, curr_obs, jnp.eye(num_actions)[action], subkey)

        if(binary_obs):
            o_hat_log_probs = jnp.sum(log_binary_probability(next_obs, o_hat_dist))
        else:
            o_hat_log_probs = jnp.sum(log_gaussian_probability(next_obs, o_hat_dist))

        # no need to reconstruct state on terminal steps, just need to get reward and terminal right
        obs_prediction_loss = jnp.where(terminal, 0.0, -o_hat_log_probs)

        r_dist = reward_network(reward_params, curr_obs, jnp.eye(num_actions)[action])
        reward_loss = -log_gaussian_probability(reward, r_dist)

        gamma_dist = termination_network(termination_params, curr_obs, jnp.eye(num_actions)[action])
        termination_loss = -log_binary_probability(jnp.logical_not(terminal), gamma_dist)

        loss = (config.reward_weight*reward_loss+
                config.termination_weight*termination_loss+
                config.obs_prediction_weight*obs_prediction_loss)
        return loss
    return single_sample_model_loss

def get_single_sample_Q_loss(Q_function):
    def single_sample_Q_loss(Q_params, Q_target_params, curr_obs, action, reward, next_obs, continuation_prob, weight):
        Q_curr = Q_function(Q_params,curr_obs)[action]
        if(config.double_DQN):
            Q_next = Q_function(SG(Q_target_params),next_obs)[jnp.argmax(Q_function(SG(Q_params),next_obs))]
        else:
            Q_next = jnp.max(Q_function(SG(Q_target_params),next_obs))
        return weight*(Q_curr-(reward+config.gamma*continuation_prob*Q_next))**2
    return single_sample_Q_loss

def get_single_model_rollout_func(model_functions, Q_function, rollout_length, num_actions):
    def single_model_rollout_func(initial_obs, model_params, Q_params, key):
        next_obs_network = model_functions['next_obs']
        reward_network = model_functions['reward']
        termination_network = model_functions['termination']

        next_obs_params = model_params['next_obs']
        reward_params = model_params['reward']
        termination_params = model_params['termination']

        def loop_function(carry, data):
            obs, continuation_prob, weight, key = carry

            if(config.episodic_env):
                weight = weight*continuation_prob
            else:
                weight = 1.0

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

            r_dist = reward_network(reward_params, obs, jnp.eye(num_actions)[action])
            reward = r_dist["mu"]

            gamma_dist = termination_network(termination_params, obs, jnp.eye(num_actions)[action])

            key, subkey = jx.random.split(key)
            obs, _ = next_obs_network(next_obs_params, obs, jnp.eye(num_actions)[action], subkey)

            continuation_prob = jnp.exp(log_binary_probability(True, gamma_dist))
            return (obs, continuation_prob, weight, key), (last_obs, action, reward, obs, continuation_prob, weight)

        key, subkey = jx.random.split(key)

        key, subkey = jx.random.split(key)
        _, sample_transitions = jx.lax.scan(loop_function, (initial_obs.astype(float),1.0,1.0,subkey), None, length=rollout_length)
        return sample_transitions
    return single_model_rollout_func

def get_agent_environment_interaction_loop_function(env, Q_function, model_functions, Q_opt_update, model_opt_update, get_Q_params, get_model_params, replay_buffer, num_iterations, num_actions):
    batch_Q_loss = lambda *x: jnp.mean(vmap(get_single_sample_Q_loss(Q_function), in_axes=(None,None,0,0,0,0,0,0))(*x))
    batch_model_loss = lambda *x: jnp.mean(vmap(get_single_sample_model_loss(model_functions, config.binary_obs), in_axes=(None,0,0,0,0,0,0))(*x))
    Q_loss_grad = grad(batch_Q_loss)
    model_loss_grad = grad(batch_model_loss)

    batch_model_rollout= lambda *x: [jnp.reshape(y,(config.batch_size*config.rollout_length,-1)) for y in vmap(get_single_model_rollout_func(model_functions, Q_function, config.rollout_length, num_actions),in_axes=(0,None,None,0))(*x)]

    def agent_environment_interaction_loop_function(env_state, Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, t, key, train):
        obs = env.get_observation(env_state)
        total_reward = 0.0
        total_Q = jnp.zeros(num_actions)

        def loop_function(carry, data):
            env_state, Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, t, total_reward, total_Q, obs, key = carry
            key, subkey = jx.random.split(key)
            Q_curr = Q_function(get_Q_params(Q_opt_state),obs.astype(float))
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
            last_obs = obs
            key, subkey = jx.random.split(key)
            env_state, obs, reward, terminal, _ = env.step(subkey, env_state, action)
            #reset if terminated
            key, subkey = jx.random.split(key)
            env_state = jx.tree_map(lambda x,y: jnp.where(terminal, x,y), env.reset(subkey)[0], env_state)
            buffer_state = replay_buffer.add(buffer_state, last_obs, action, reward, obs, terminal)
            if(train):
                if(config.use_target):
                    Q_target_params = jx.tree_map(lambda x,y: jnp.where(t%config.target_update_frequency==0,x,y),get_Q_params(Q_opt_state),Q_target_params)
                else:
                    Q_target_params = get_Q_params(Q_opt_state)

                def update_loop_function(carry, data):
                    Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, key = carry
                    buffer_state, sample_transitions = replay_buffer.sample(buffer_state)
                    key, subkey = jx.random.split(key)
                    subkeys = jx.random.split(subkey, num=config.batch_size)
                    model_grad = model_loss_grad(get_model_params(model_opt_state), *sample_transitions, subkeys)
                    model_opt_state = model_opt_update(opt_t, model_grad, model_opt_state)

                    sample_obs = sample_transitions[0]

                    key, subkey = jx.random.split(key)
                    subkeys = jx.random.split(subkey, num=config.batch_size)
                    model_transitions = batch_model_rollout(sample_obs, get_model_params(model_opt_state), get_Q_params(Q_opt_state), subkeys)

                    Q_grad = Q_loss_grad(get_Q_params(Q_opt_state), Q_target_params, *model_transitions)
                    Q_opt_state = Q_opt_update(opt_t, Q_grad, Q_opt_state)

                    opt_t+=1
                    return (Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, key), None
                key, subkey = jx.random.split(key)
                carry = (Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, subkey)
                (Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, _), _ = jx.lax.scan(update_loop_function,carry, None, length=config.updates_per_step)
            total_reward+=reward
            t+=1
            return (env_state, Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, t, total_reward, total_Q, obs, key), None

        key, subkey = jx.random.split(key)
        carry = (env_state, Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, t, total_reward, total_Q, obs, subkey)

        (env_state, Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, t, total_reward, total_Q, obs, _), _ = jx.lax.scan(loop_function, carry, None, length=num_iterations)

        return env_state, Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, t, total_reward/num_iterations, total_Q/num_iterations
    return jit(agent_environment_interaction_loop_function, static_argnames=('train',))

def get_agent_eval_function_episodic(env, Q_function, model_functions, get_Q_params, get_model_params, num_iterations):
    def agent_eval_function_episodic(Q_opt_state, model_opt_state, key):
        key, subkey = jx.random.split(key)
        env_state, phi = env.reset(subkey)
        total_reward = 0.0
        nonterminal_steps = 0

        next_obs_network = model_functions['next_obs']

        def loop_function(carry, data):
            env_state, Q_opt_state, total_reward, nonterminal_steps, phi, terminated, key = carry
            Q_curr = Q_function(get_Q_params(Q_opt_state),phi.astype(float))
            action = jnp.argmax(Q_curr)
            key, subkey = jx.random.split(key)

            next_obs_params = get_model_params(model_opt_state)['next_obs']
            _, o_hat_dist = next_obs_network(next_obs_params, obs, jnp.eye(num_actions)[action], subkey)

            env_state, phi, reward, terminal, _ = env.step(subkey, env_state, action)

            total_reward+=reward*jnp.logical_not(terminated)
            nonterminal_steps += jnp.logical_not(terminated)

            terminated = jnp.logical_or(terminated,terminal)
            return (env_state, Q_opt_state, total_reward, nonterminal_steps, phi, terminated, key), None

        key, subkey = jx.random.split(key)
        carry = (env_state, Q_opt_state, total_reward, nonterminal_steps, phi, False, subkey)

        (env_state, Q_opt_state, total_reward, nonterminal_steps, phi, _, _), _ = jx.lax.scan(loop_function, carry, None, length=num_iterations)

        return total_reward
    return jit(agent_eval_function_episodic)

def get_agent_eval_function_continuing(env, Q_function, model_functions, get_Q_params, get_model_params, num_iterations):
    def agent_eval_function_continuing(Q_opt_state, model_opt_state, key):
        key, subkey = jx.random.split(key)
        env_state, obs = env.reset(subkey)
        total_reward = 0.0

        next_obs_network = model_functions['next_obs']

        def loop_function(carry, data):
            env_state, Q_opt_state, model_opt_state, total_reward, obs, key = carry
            key, subkey = jx.random.split(key)
            Q_curr = Q_function(get_Q_params(Q_opt_state),obs)
            key, subkey = jx.random.split(key)
            action = jnp.argmax(Q_curr)
            key, subkey = jx.random.split(key)

            next_obs_params = get_model_params(model_opt_state)['next_obs']
            _, o_hat_dist = next_obs_network(next_obs_params, obs, jnp.eye(num_actions)[action], subkey)

            env_state, obs, reward, terminal, _ = env.step(subkey, env_state, action)

            total_reward += reward
            return (env_state, Q_opt_state, model_opt_state, total_reward, obs, key), None

        key, subkey = jx.random.split(key)
        carry = (env_state, Q_opt_state, model_opt_state, total_reward, obs, subkey)

        (env_state, Q_opt_state, model_opt_state, total_reward, obs, _), _ = jx.lax.scan(loop_function, carry, None, length=num_iterations)

        return total_reward/num_iterations
    return jit(agent_eval_function_continuing)

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

    @partial(jit, static_argnums=(0,2))
    def sample(self, state):
        location, full, buffers, key = state
        key, subkey = jx.random.split(key)
        indices = jx.random.randint(subkey, minval=0, maxval=jnp.where(full, self.buffer_size, location),shape=(self.batch_size,))

        sample = []
        for b in buffers:
            sample += [b.take(indices,axis=0)]

        state = (location, full, buffers, key)
        return state, sample

env = Environment(**env_config)
key, subkey = jx.random.split(key)
env_state, obs = env.reset(subkey)
num_actions = env.num_actions()

dummy_a = jnp.zeros((num_actions))

Q_net = hk.without_apply_rng(hk.transform(lambda obs: Q_function(config,num_actions)(obs)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
Q_params = [Q_net.init(subkey, obs.astype(float)) for subkey in subkeys]
Q_func = Q_net.apply

Q_target_params = tree_stack(Q_params)

reward_net = hk.without_apply_rng(hk.transform(lambda obs, a: reward_function(config)(obs, a)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
reward_params = [reward_net.init(subkey, obs.astype(float), dummy_a) for subkey in subkeys]
reward_func = reward_net.apply

termination_net = hk.without_apply_rng(hk.transform(lambda obs, a: termination_function(config)(obs, a)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
termination_params = [termination_net.init(subkey, obs.astype(float), dummy_a) for subkey in subkeys]
termination_func = termination_net.apply

next_obs_net = hk.without_apply_rng(hk.transform(lambda obs, a, key: next_obs_function(config, obs.shape[0])(obs, a, key)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
next_obs_params = [next_obs_net.init(subkey, obs.astype(float), dummy_a, subkey) for subkey in subkeys]
next_obs_func = next_obs_net.apply

model_funcs = {"reward":reward_func, "termination":termination_func, "next_obs":next_obs_func}

model_params = [{"reward":rp, "termination":tp, "next_obs":no} for rp,tp,no in zip(reward_params, termination_params, next_obs_params)]


Q_opt_init, Q_opt_update, get_Q_params = adamw(config.Q_alpha, eps=config.eps_adam, b1=config.b1_adam, b2=config.b2_adam, wd=config.wd_adam)
Q_opt_states = tree_stack([Q_opt_init(p) for p in Q_params])
Q_opt_update = jit(Q_opt_update)

model_opt_init, model_opt_update, get_model_params = adamw(config.model_alpha, eps=config.eps_adam, b1=config.b1_adam, b2=config.b2_adam, wd=config.wd_adam)
model_opt_states = tree_stack([model_opt_init(p) for p in model_params])
model_opt_update = jit(model_opt_update)

buffer = replay_buffer(config.buffer_size, config.batch_size, (obs.shape, (), (), obs.shape, ()), (float,int,float,float,bool))

key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
buffer_states = tree_stack([buffer.initialize(subkey) for subkey in subkeys])

interaction_loop = get_agent_environment_interaction_loop_function(env, Q_func, model_funcs, Q_opt_update, model_opt_update, get_Q_params, get_model_params, buffer, config.eval_frequency, num_actions)

if(config.episodic_env):
    eval_agent = jit(lambda *x: jnp.mean(vmap(get_agent_eval_function_episodic(env, Q_func, model_funcs, get_Q_params, get_model_params, config.eval_steps), in_axes=(None,None,0))(*x)))
else:
    eval_agent = jit(lambda *x: jnp.mean(vmap(get_agent_eval_function_continuing(env, Q_func, model_funcs, get_Q_params, get_model_params, config.eval_steps), in_axes=(None,None,0))(*x)))

multiseed_interaction_loop = jit(vmap(interaction_loop, in_axes=(0,0,0,0,0,None,None,0,None), out_axes=(0,0,0,0,0,None,None,0,0)), static_argnames='train')

multiseed_eval_agent = jit(vmap(eval_agent, in_axes=(0,0,0)))

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
                'model': [get_model_params(model_opt_state) for model_opt_state in tree_unstack(model_opt_states)],
                'Q' : [get_Q_params(Q_opt_state) for Q_opt_state in tree_unstack(Q_opt_states)]
            }, f)
        time_since_last_save = 0

    # Train step
    key, subkey = jx.random.split(key)
    subkeys = jx.random.split(subkey, num=config.num_seeds)
    env_states, Q_opt_states, Q_target_params, model_opt_states, buffer_states, opt_t, t, _, _ = multiseed_interaction_loop(env_states, Q_opt_states, Q_target_params, model_opt_states, buffer_states, opt_t, t, subkeys, time>=config.training_start_time)

    # Evaluation step
    key, subkey = jx.random.split(key)
    subkeys = jx.random.split(subkey, num=config.num_seeds*config.eval_batch_size).reshape(config.num_seeds,config.eval_batch_size,2)
    reward_rate = multiseed_eval_agent(Q_opt_states, model_opt_states, subkeys)

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
            'model': [get_model_params(model_opt_state) for model_opt_state in tree_unstack(model_opt_states)],
            'Q' : [get_Q_params(Q_opt_state) for Q_opt_state in tree_unstack(Q_opt_states)]
        }, f)
