import jax.numpy as jnp
import itertools
from jax import grad
import jax
import optax
import time
import pickle

from gmfm_forcing_jax import *


class sparse_actor:
	def __init__(self, a,w,p,beta = 0.1, gamma = 0.1):
		self.a = a
		self.w = w
		self.p = p
		self.beta = beta
		self.gamma = gamma
	# def eval_policy(self, env = 'gmfm',ts = jnp.linspace(0, 21 * jnp.pi,501), y0= jnp.ones(4)):
	# 	policy = [self.a, self.w, self.p,self.beta]
	# 	if env == 'gmfm':
	# 		y_all, pi_all = get_ode_res(gmfm_forcing_dsdt,y0 = y0, t = ts,args  = policy)
	# 		J = jnp.mean(jnp.linalg.norm(y_all[:2],axis = 1)**2) + self.gamma * jnp.mean(pi_all **2)
	# 		return J

def eval_policy(params,p ,y0 = jnp.ones(4),gamma = 0.1, t = jnp.linspace(0, 21 * jnp.pi,501), l1_penalty = 0.01):
	policy = [params['a'],params['w'],p,params['beta']]
	# policy = [params['a'],params['w'],p,beta]
	# print(len(policy))
	y_all, pi_all = get_ode_res(gmfm_forcing_dsdt,t = t,y0 = y0, args = policy)
	policy_cost = jnp.mean(jnp.linalg.norm(y_all[:,:2], axis = 1)**2) + gamma * jnp.mean(pi_all **2)
	l1_cost =  l1_penalty * jnp.linalg.norm(params['w'], ord = 1)
	J = policy_cost + l1_cost
	# J = get_ode_res(gmfm_forcing_dsdt,t = t,y0 = y0, args = policy)
	# print("res",J)
	jax.debug.print("policy cost is {x} here, l1 cost = {s}", x = policy_cost, s = l1_cost)
	return J



def init_lorenz_actor():
	pass

def init_gmfm_actor(beta  = 0.1):
	D1 = 8
	D2 = 8
	D = D1 * D2
	s_bounds = [-0.4,0.4]
	p0 = jnp.empty((D,2))
	s1 = jnp.linspace(s_bounds[0], s_bounds[1], D1)
	s2 = jnp.linspace(s_bounds[0], s_bounds[1],D2)
	i = 0
	p0 = jnp.empty((D,2))
	for (xi,yi) in itertools.product(s1,s2):
		p0 = p0.at[i,:].set(jnp.array((xi,yi)))
		i += 1

	a_bounds = jnp.array([-1,1])
	key = jax.random.PRNGKey(3)
	a0 = jax.random.uniform(key, minval = a_bounds[0], maxval = a_bounds[1], shape = (D,))
	w0 = jax.random.uniform(key,shape = (D,)) #[0,1]
	actor = sparse_actor(a0,w0,p0,beta)
	return actor


def main(env = 'gmfm'):
	#first init sparse actor
	if env == 'gmfm':
		actor = init_gmfm_actor(beta = 0.1)
	elif env == 'lorenz':
		actor = init_lorenz_actor(beta = 0.1)

	#next train sparse actor
	max_iters = 500
	# l1_penalty = 0.0
	y0 = jnp.array([0.3162,0.0051,0,0])
	a_bounds = jnp.array([-1,1])
	beta_bounds = jnp.array([0,1])
	optimizer = optax.adabelief(1e-3)
	params = {'a': actor.a, 'w':actor.w, 'beta': actor.beta}
	# params = {'a': actor.a, 'w':actor.w}
	# print(f"hey: {params['beta']}")
	opt_params = optimizer.init(params)
	for i in range(max_iters):
		grads = jax.grad(eval_policy)(params, actor.p,y0)
		# grads = jax.grad(test_loss)(params)
		# print("yo", grads)
		updates,opt_params = optimizer.update(grads, opt_params)
		params = optax.apply_updates(params,updates)
		params['a'] = jnp.where(params['a'] < a_bounds[0], a_bounds[0], params['a'])
		params['a'] = jnp.where(params['a'] > a_bounds[1], a_bounds[1], params['a'])
		params['w'] = jnp.where(params['w'] < 0, 0, params['w'])
		params['w'] = jnp.where(params['w'] > 1, 1, params['w'])
		params['beta'] = jnp.where(params['beta'] < 0.01, 0.01, params['beta'])
		print(f"w[last] at iter{i}, {params['w'][-1]}")
		print(f"beta at iter{i}, {params['beta']}")


	#save actor data to a file
	f = open("gmfm_sparse_actor.pkl","wb")
	actor_info={}
	actor_info['a'] = params['a']
	actor_info['w'] = params['w']
	actor_info['p'] = actor.p
	actor_info['beta'] = params['beta']
	actor_info['gamma'] = actor.gamma
	actor_info['y0'] = y0
	pickle.dump(actor_info,f)
	f.close()




if __name__ == '__main__':
	t1 = time.time()
	main()
	print("this took %.1f seconds" % (time.time() - t1))