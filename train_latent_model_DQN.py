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
std_activation_dict = {"softplus": jx.nn.softplus, "sigmoid": jx.nn.sigmoid, "sigmoid2": lambda x: 2*jx.nn.sigmoid(x/2)}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int, default=0)
parser.add_argument("--output", "-o", type=str, default="latent_model")
parser.add_argument("--config", "-c", type=str)
args = parser.parse_args()
key = jx.random.PRNGKey(args.seed)

with open(args.config, 'r') as f:
    config=json.load(f)
    config.update({"agent_type":"latent_model", "seed":args.seed})

def set_default(d, k, v):
    if k not in d:
        d[k] = v

set_default(config, "double_DQN", False)
set_default(config, "episodic_env", False)
set_default(config, "seperate_rollout_batch_size", False)
set_default(config, "updates_per_step", 1)
set_default(config, "min_std", 0.0)
set_default(config, "std_act", "softplus")
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


def gaussian_cross_entropy(params_1, params_2):
    mu_1 = params_1['mu']
    sigma_1 = params_1['sigma']
    mu_2 = params_2['mu']
    sigma_2 = params_2['sigma']
    return 0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma_2) + (sigma_1**2 + (mu_1 - mu_2)**2) / (2 * sigma_2**2)


def gaussian_entropy(params):
    sigma = params['sigma']
    return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma)


def gaussian_KL(params_1, params_2):
    return gaussian_cross_entropy(params_1, params_2) - gaussian_entropy(params_1)


def log_binary_probability(x, params):
    logit = params['logit']
    return jnp.where(x, jx.nn.log_sigmoid(logit), jx.nn.log_sigmoid(-logit))


def binary_entropy(params):
    logit = params['logit']
    return jx.nn.sigmoid(logit) * jx.nn.log_sigmoid(logit) + jx.nn.sigmoid(-logit) * jx.nn.log_sigmoid(-logit)


def categorical_cross_entropy(params_1, params_2):
    probs_1 = params_1['probs']
    log_probs_2 = params_2['log_probs']
    return -jnp.sum(probs_1 * log_probs_2, axis=(-1))


def categorical_entropy(params):
    probs = params['probs']
    log_probs = params['log_probs']
    return -jnp.sum(probs * log_probs, axis=(-1))


def categorical_KL(params_1, params_2):
    return categorical_cross_entropy(params_1, params_2) - categorical_entropy(params_1)

if(config.latent_type=='gaussian' or config.latent_type=='tanh_gaussian'):
    latent_KL = gaussian_KL
    latent_entropy = gaussian_entropy
    latent_cross_entropy = gaussian_cross_entropy
    base_dist ={'mu':jnp.zeros(config.num_features), 'sigma':jnp.ones(config.num_features)}
elif(config.latent_type=='categorical'):
    latent_KL = categorical_KL
    latent_entropy = categorical_entropy
    latent_cross_entropy = categorical_cross_entropy
    base_logits = jnp.zeros((config.num_features, config.feature_width))
    base_probs = jx.nn.softmax(base_logits)
    base_log_probs = jx.nn.log_softmax(base_logits)
    base_dist = {'probs':base_probs, 'log_probs':base_log_probs}
else:
    raise ValueError('Unrecognized latent type.')

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

class reward_function(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]

    def __call__(self, phi, action, key=None):
        x = jnp.concatenate([jnp.ravel(phi),action])
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

    def __call__(self, phi, action, key=None):
        x = jnp.concatenate([jnp.ravel(phi),action])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        logit = hk.Linear(1)(x)[0]
        return {'logit':logit}

class next_phi_function(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_features = config.num_features
        self.num_hidden_units = config.num_hidden_units
        self.feature_width = config.feature_width
        self.activation_function = activation_dict[config.activation]
        self.std_activation_function = std_activation_dict[config.std_act]
        self.latent_type = config.latent_type

    def __call__(self, phi, action, key):
        x = jnp.concatenate([jnp.ravel(phi),action])
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))

        if(self.latent_type=='gaussian'):
            mu = hk.Linear(self.num_features)(jnp.ravel(x))
            sigma = self.std_activation_function(hk.Linear(self.num_features)(jnp.ravel(x)))
            sigma = jnp.clip(sigma,min_denom, None)

            x = mu+sigma*jx.random.normal(key,mu.shape)
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='tanh_gaussian'):
            mu = hk.Linear(self.num_features)(jnp.ravel(x))
            sigma = self.std_activation_function(hk.Linear(self.num_features)(jnp.ravel(x)))
            sigma = jnp.clip(sigma,min_denom, None)

            x = jx.nn.tanh(mu+sigma*jx.random.normal(key,mu.shape))
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='categorical'):
            logits = jnp.reshape(hk.Linear(self.num_features*self.feature_width)(jnp.ravel(x)), [self.num_features,self.feature_width])

            probs = jx.nn.softmax(logits)
            log_probs = jx.nn.log_softmax(logits)

            x = jx.nn.one_hot(jx.random.categorical(key, logits),self.feature_width, axis=1)
            x = probs+jx.lax.stop_gradient(x-probs)
            return x, {'probs':probs, 'log_probs':log_probs}

class phi_function(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.num_features = config.num_features
        self.feature_width = config.feature_width
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.std_activation_function = std_activation_dict[config.std_act]
        self.latent_type = config.latent_type

    def __call__(self, obs, key):
        x = obs
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))

        if(self.latent_type=='gaussian'):
            mu = hk.Linear(self.num_features)(jnp.ravel(x))
            sigma = self.std_activation_function(hk.Linear(self.num_features)(jnp.ravel(x)))
            sigma = jnp.clip(sigma,min_denom, None)+config.min_std

            x = mu+sigma*jx.random.normal(key,mu.shape)
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='tanh_gaussian'):
            mu = hk.Linear(self.num_features)(jnp.ravel(x))
            sigma = self.std_activation_function(hk.Linear(self.num_features)(jnp.ravel(x)))
            sigma = jnp.clip(sigma,min_denom, None)+config.min_std

            x = jx.nn.tanh(mu+sigma*jx.random.normal(key,mu.shape))
            return x, {'mu':mu, 'sigma':sigma}
        elif(self.latent_type=='categorical'):
            logits = jnp.reshape(hk.Linear(self.num_features*self.feature_width)(jnp.ravel(x)), [self.num_features,self.feature_width])

            probs = jx.nn.softmax(logits)
            log_probs = jx.nn.log_softmax(logits)

            x = jx.nn.one_hot(jx.random.categorical(key, logits),self.feature_width, axis=1)
            x = probs+jx.lax.stop_gradient(x-probs)
            return x, {'probs':probs, 'log_probs':log_probs}

class obs_function(hk.Module):
    def __init__(self, config, obs_width, name=None):
        super().__init__(name=name)
        self.num_features = config.num_features
        self.feature_width = config.feature_width
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_units = config.num_hidden_units
        self.activation_function = activation_dict[config.activation]
        self.binary_obs = config.binary_obs
        self.obs_width = obs_width

    def __call__(self, phi):
        x = jnp.ravel(phi)
        for i in range(self.num_hidden_layers):
            x = self.activation_function(hk.Linear(self.num_hidden_units)(x))
        if(self.binary_obs):
            logit = hk.Linear(self.obs_width)(x)
            return {'logit':logit}
        else:
            mu = hk.Linear(self.obs_width)(x)
            if(config.learn_obs_variance):
                sigma = jx.nn.softplus(hk.Linear(self.obs_width)(x))
                sigma = jnp.clip(sigma,min_denom, None)
            else:
                sigma = jnp.ones(mu.shape)
            return {'mu':mu, 'sigma':sigma}

########################################################################
# Losses
########################################################################

def get_single_sample_model_loss(model_functions, binary_obs):
    def single_sample_model_loss(model_params, curr_obs, action, reward, next_obs, terminal, key):
        phi_network = model_functions['phi']
        next_phi_network = model_functions['next_phi']
        reward_network = model_functions['reward']
        termination_network = model_functions['termination']
        obs_network = model_functions['obs']

        phi_params = model_params['phi']
        next_phi_params = model_params['next_phi']
        reward_params = model_params['reward']
        termination_params = model_params['termination']
        obs_params = model_params['obs']

        key, subkey = jx.random.split(key)
        phi, phi_dist = phi_network(phi_params, curr_obs, subkey)

        o_hat_dist = obs_network(obs_params, phi)

        if(binary_obs):
            o_hat_log_probs = jnp.sum(log_binary_probability(curr_obs, o_hat_dist))
        else:
            o_hat_log_probs = jnp.sum(log_gaussian_probability(curr_obs, o_hat_dist))

        # no need to reconstruct state on terminal steps, just need to get reward and terminal right
        obs_prediction_loss = jnp.where(terminal, 0.0, -o_hat_log_probs)

        r_dist = reward_network(reward_params, phi, jnp.eye(num_actions)[action])
        reward_loss = -log_gaussian_probability(reward, r_dist)

        gamma_dist = termination_network(termination_params, phi, jnp.eye(num_actions)[action])
        termination_loss = -log_binary_probability(jnp.logical_not(terminal), gamma_dist)

        key, subkey = jx.random.split(key)
        _, next_phi_dist = phi_network(phi_params, next_obs, subkey)

        key, subkey = jx.random.split(key)
        _, phi_hat_dist = next_phi_network(next_phi_params, phi, jnp.eye(num_actions)[action], subkey)

        # KL loss applied to make current phi closer to prediction
        KL_posterior_loss = jnp.sum(latent_KL(next_phi_dist,SG(phi_hat_dist)))

        # Optional entropy regularization on posterior
        posterior_entropy_loss = jnp.sum(latent_KL(phi_dist,base_dist))

        # KL loss applied to make prediction closer to current phi
        KL_prior_loss = jnp.sum(latent_KL(SG(next_phi_dist),phi_hat_dist))

        loss = (config.KL_posterior_weight*KL_posterior_loss+
                config.KL_prior_weight*KL_prior_loss+
                config.posterior_entropy_weight*posterior_entropy_loss+
                config.reward_weight*reward_loss+
                config.termination_weight*termination_loss+
                config.obs_prediction_weight*obs_prediction_loss)
        return loss
    return single_sample_model_loss

def get_single_sample_Q_loss(Q_function):
    def single_sample_Q_loss(Q_params, Q_target_params, curr_phi, action, reward, next_phi, continuation_prob, weight):
        Q_curr = Q_function(Q_params,curr_phi)[action]
        if(config.double_DQN):
            Q_next = Q_function(SG(Q_target_params),next_phi)[jnp.argmax(Q_function(SG(Q_params),next_phi))]
        else:
            Q_next = jnp.max(Q_function(SG(Q_target_params),next_phi))
        Q_next = jnp.max(Q_function(SG(Q_target_params),next_phi))
        return weight*(Q_curr-(reward+config.gamma*continuation_prob*Q_next))**2
    return single_sample_Q_loss

def get_single_model_rollout_func(model_functions, Q_function, rollout_length, num_actions):
    def single_model_rollout_func(initial_obs, model_params, Q_params, key):
        phi_network = model_functions['phi']
        next_phi_network = model_functions['next_phi']
        reward_network = model_functions['reward']
        termination_network = model_functions['termination']

        phi_params = model_params['phi']
        next_phi_params = model_params['next_phi']
        reward_params = model_params['reward']
        termination_params = model_params['termination']

        def loop_function(carry, data):
            phi, continuation_prob, weight, key = carry

            if(config.episodic_env):
                weight = weight*continuation_prob
            else:
                weight = 1.0

            last_phi = phi

            Q_curr = Q_function(Q_params, phi)
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

            r_dist = reward_network(reward_params, phi, jnp.eye(num_actions)[action])
            reward = r_dist["mu"]

            gamma_dist = termination_network(termination_params, phi, jnp.eye(num_actions)[action])

            key, subkey = jx.random.split(key)
            phi, _ = next_phi_network(next_phi_params, phi, jnp.eye(num_actions)[action], subkey)

            continuation_prob = jnp.exp(log_binary_probability(True, gamma_dist))

            return (phi, continuation_prob, weight, key), (last_phi, action, reward, phi, continuation_prob, weight)

        key, subkey = jx.random.split(key)
        phi, phi_dist = phi_network(phi_params, initial_obs, subkey)

        key, subkey = jx.random.split(key)
        _, sample_transitions = jx.lax.scan(loop_function, (phi,1.0,1.0,subkey), None, length=rollout_length)
        return sample_transitions
    return single_model_rollout_func

def get_agent_environment_interaction_loop_function(env, Q_function, model_functions, Q_opt_update, model_opt_update, get_Q_params, get_model_params, replay_buffer, num_iterations, num_actions):
    batch_Q_loss = lambda *x: jnp.mean(vmap(get_single_sample_Q_loss(Q_function), in_axes=(None,None,0,0,0,0,0,0))(*x))
    batch_model_loss = lambda *x: jnp.mean(vmap(get_single_sample_model_loss(model_functions, config.binary_obs), in_axes=(None,0,0,0,0,0,0))(*x))
    Q_loss_grad = grad(batch_Q_loss)
    model_loss_grad = grad(batch_model_loss)

    #get batch of rollouts and combine the batch and step dimensions
    if(config.seperate_rollout_batch_size):
        batch_model_rollout= lambda *x: [jnp.reshape(y,(config.rollout_batch_size*config.rollout_length,-1)) for y in vmap(get_single_model_rollout_func(model_functions, Q_function, config.rollout_length, num_actions),in_axes=(0,None,None,0))(*x)]
    else:
        batch_model_rollout= lambda *x: [jnp.reshape(y,(config.batch_size*config.rollout_length,-1)) for y in vmap(get_single_model_rollout_func(model_functions, Q_function, config.rollout_length, num_actions),in_axes=(0,None,None,0))(*x)]

    def agent_environment_interaction_loop_function(env_state, Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, t, key, train):
        obs = env.get_observation(env_state)
        total_reward = 0.0
        total_Q = jnp.zeros(num_actions)

        def loop_function(carry, data):
            env_state, Q_opt_state, Q_target_params, model_opt_state, buffer_state, opt_t, t, total_reward, total_Q, obs, key = carry
            key, subkey = jx.random.split(key)
            phi, _ = model_functions["phi"](get_model_params(model_opt_state)["phi"],obs, subkey)
            Q_curr = Q_function(get_Q_params(Q_opt_state),phi)
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

                    #Note: model_rollouts normally use same sampled obs as model_training step
                    #if we wish to use a seperate rollout batchsize, we need to sample again
                    if(config.seperate_rollout_batch_size):
                        buffer_state, sample_transitions = replay_buffer.sample(buffer_state, config.rollout_batch_size)
                    sample_obs = sample_transitions[0]

                    key, subkey = jx.random.split(key)
                    if(config.seperate_rollout_batch_size):
                        subkeys = jx.random.split(subkey, num=config.rollout_batch_size)
                    else:
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
        env_state, obs = env.reset(subkey)
        total_reward = 0.0

        def loop_function(carry, data):
            env_state, Q_opt_state, model_opt_state, total_reward, obs, terminated, key = carry
            key, subkey = jx.random.split(key)
            phi, _ = model_functions["phi"](get_model_params(model_opt_state)["phi"],obs, subkey)
            Q_curr = Q_function(get_Q_params(Q_opt_state),phi)
            key, subkey = jx.random.split(key)
            action = jnp.argmax(Q_curr)
            key, subkey = jx.random.split(key)
            env_state, obs, reward, terminal, _ = env.step(subkey,env_state, action)
            total_reward+=reward*jnp.logical_not(terminated)
            terminated = jnp.logical_or(terminated,terminal)
            return (env_state, Q_opt_state, model_opt_state, total_reward, obs, terminated, key), None

        key, subkey = jx.random.split(key)
        carry = (env_state, Q_opt_state, model_opt_state, total_reward, obs, False, subkey)

        (env_state, Q_opt_state, model_opt_state, total_reward, obs, _, _), _ = jx.lax.scan(loop_function, carry, None, length=num_iterations)

        return total_reward
    return jit(agent_eval_function_episodic)

def get_agent_eval_function_continuing(env, Q_function, model_functions, get_Q_params, get_model_params, num_iterations):
    def agent_eval_function_continuing(Q_opt_state, model_opt_state, key):
        key, subkey = jx.random.split(key)
        env_state, obs = env.reset(subkey)
        total_reward = 0.0

        def loop_function(carry, data):
            env_state, Q_opt_state, model_opt_state, total_reward, obs, key = carry
            key, subkey = jx.random.split(key)
            phi, _ = model_functions["phi"](get_model_params(model_opt_state)["phi"],obs, subkey)
            Q_curr = Q_function(get_Q_params(Q_opt_state),phi)
            key, subkey = jx.random.split(key)
            action = jnp.argmax(Q_curr)
            key, subkey = jx.random.split(key)
            env_state, obs, reward, terminal, _ = env.step(subkey, env_state, action)
            total_reward+=reward
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
    def sample(self, state, batch_size=None):
        if(batch_size is None):
            batch_size=self.batch_size
        location, full, buffers, key = state
        key, subkey = jx.random.split(key)
        indices = jx.random.randint(subkey, minval=0, maxval=jnp.where(full, self.buffer_size, location),shape=(batch_size,))
        #indices = jx.random.choice(subkey,location,shape=(self.batch_size,))

        sample = []
        for b in buffers:
            sample += [b.take(indices,axis=0)]

        state = (location, full, buffers, key)
        return state, sample

env = Environment(**env_config)
key, subkey = jx.random.split(key)
env_state, obs = env.reset(subkey)
num_actions = env.num_actions()

dummy_phi = jnp.zeros((config.num_features*(config.feature_width if config.latent_type=='categorical' else 1)))
dummy_a = jnp.zeros((num_actions))

Q_net = hk.without_apply_rng(hk.transform(lambda phi: Q_function(config,num_actions)(phi)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
Q_params = [Q_net.init(subkey, dummy_phi) for subkey in subkeys]
Q_func = Q_net.apply

Q_target_params = tree_stack(Q_params)

reward_net = hk.without_apply_rng(hk.transform(lambda phi, a: reward_function(config)(phi, a)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
reward_params = [reward_net.init(subkey, dummy_phi, dummy_a) for subkey in subkeys]
reward_func = reward_net.apply

termination_net = hk.without_apply_rng(hk.transform(lambda phi, a: termination_function(config)(phi, a)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
termination_params = [termination_net.init(subkey, dummy_phi, dummy_a) for subkey in subkeys]
termination_func = termination_net.apply

phi_net = hk.without_apply_rng(hk.transform(lambda obs, key: phi_function(config)(obs, key)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
phi_params = [phi_net.init(subkey, obs.astype(float), subkey) for subkey in subkeys]
phi_func = phi_net.apply

next_phi_net = hk.without_apply_rng(hk.transform(lambda phi, a, key: next_phi_function(config)(phi, a, key)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
next_phi_params = [next_phi_net.init(subkey, dummy_phi, dummy_a, subkey) for subkey in subkeys]
next_phi_func = next_phi_net.apply

obs_net = hk.without_apply_rng(hk.transform(lambda phi: obs_function(config, obs.shape[0])(phi)))
key, subkey = jx.random.split(key)
subkeys = jx.random.split(subkey, num=config.num_seeds)
obs_params = [obs_net.init(subkey, dummy_phi) for subkey in subkeys]
obs_func = obs_net.apply

model_funcs = {"reward":reward_func, "termination":termination_func, "phi":phi_func, "next_phi":next_phi_func, "obs":obs_func}

model_params = [{"reward":rp, "termination":tp, "phi":pp, "next_phi":npp, "obs":op} for rp,tp,pp,npp,op in zip(reward_params, termination_params, phi_params, next_phi_params, obs_params)]


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
