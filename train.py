import jax.numpy as jnp
import itertools
from jax import grad
import jax
import optax
import time
import pickle
import os

from env_jax import *


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

def eval_policy(params,p ,y0 = jnp.ones(4),xref = jnp.zeros(2), gamma = 0.1, t = jnp.linspace(0, 21 * jnp.pi,501), l1_penalty = 0.01):
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

def eval_policy_lorenz(params,p ,y0 = jnp.ones(3),xref = jnp.ones(3), gamma = 0.1, t = jnp.linspace(0, 21 * jnp.pi,501), l1_penalty = 0.01):
	policy = [params['a'],params['w'],p,params['beta']]
	# policy = [params['a'],params['w'],p,beta]
	# print(len(policy))
	y_all, pi_all = get_ode_res_lorenz(lorenz_forcing_dsdt,t = t,y0 = y0, args = policy)
	# print(y_all.shape, "y shape")
	# print("y all [4]", y_all[4,:])
	# print("testing jnp pi-all mean", pi_all)
	policy_cost = jnp.mean(jnp.linalg.norm(y_all - xref, axis = 1)**2) + gamma * jnp.mean(pi_all **2)
	l1_cost =  l1_penalty * jnp.linalg.norm(params['w'], ord = 1)
	J = policy_cost + l1_cost
	# jax.debug.print("y_all max: {x}", x = jnp.max(jnp.abs(y_all), axis = 0))
	# J = get_ode_res(gmfm_forcing_dsdt,t = t,y0 = y0, args = policy)
	# print("res",J)
	jax.debug.print("policy cost is {x} here, l1 cost = {s}", x = policy_cost, s = l1_cost)
	return J

def eval_policy_pendulum(params,p ,y0 = jnp.ones(2),xref = jnp.zeros(2), gamma_thdot = 0.1,
	gamma_u = 0.001, l1_penalty = 0.0001, n_steps = 120, max_speed = 8.0, g = 9.8):
	policy = [params['a'],params['w'],p,params['beta']]
	# policy = [params['a'],params['w'],p,beta]
	# print(len(policy))
	# y_all, pi_all = get_pendulum_res(pendulum_dsdt, t = t, y0 = y0, args = policy)
	y_all, pi_all = get_pendulum_res_2(y0 = y0, args = policy, n_steps = n_steps, 
		max_speed = max_speed,g = g)
	# print(y_all.shape, "y shape")
	# print("y all [4]", y_all[4,:])
	# print("testing jnp pi-all mean", pi_all)
	# y_all_th = y_all[:,0]
	# y_all_th = y_all_th.at[y_all_th > jnp.pi].set(y_all_th[y_all_th > jnp.pi] - 2 * jnp.pi) #normalization for cost
	y_all = y_all.at[:,0].set((y_all[:,0] + jnp.pi) % (2 * jnp.pi) - jnp.pi) #normalize for angle
	policy_cost = jnp.mean((y_all[:,0] - xref[0])**2) + gamma_thdot * jnp.mean((y_all[:,1] - xref[1])**2) + gamma_u * jnp.mean(pi_all **2)
	l1_cost =  l1_penalty * jnp.linalg.norm(params['w'], ord = 1)
	J = policy_cost + l1_cost
	# jax.debug.print("y_all max: {x}", x = jnp.max(jnp.abs(y_all), axis = 0))
	# jax.debug.print("y_all:{x}", x = y_all)
	# J = get_ode_res(gmfm_forcing_dsdt,t = t,y0 = y0, args = policy)
	# print("res",J)
	jax.debug.print("policy cost is {x} here, l1 cost = {s}", x = policy_cost, s = l1_cost)
	return J



def init_lorenz_actor(beta = 1.0):
	D1 = 5
	D2 = 5
	D3 = 5
	D = D1 * D2 * D3
	s_bounds = [[-20,20],[-20,20], [0,40]]
	p0 = jnp.empty((D,3))
	s0 = jnp.linspace(s_bounds[0][0], s_bounds[0][1], D1)
	s1 = jnp.linspace(s_bounds[1][0], s_bounds[1][1], D2)
	s2 = jnp.linspace(s_bounds[2][0], s_bounds[2][1], D3)
	i = 0
	p0 = jnp.empty((D,3))
	for (xi,yi,zi) in itertools.product(s0,s1,s2):
		p0 = p0.at[i,:].set(jnp.array((xi,yi,zi)))
		i += 1

	a_bounds = jnp.array([-1.,1.])
	key = jax.random.PRNGKey(3)
	a0 = jax.random.uniform(key, minval = a_bounds[0], maxval = a_bounds[1], shape = (D,))
	w0 = jax.random.uniform(key,shape = (D,)) #[0,1]
	actor = sparse_actor(a0,w0,p0,beta)
	return actor

def init_pendulum_actor(beta  = jnp.array([0.5,0.5]), max_speed = 2.):
	D1 = 100
	D2 = 100
	D = D1 * D2
	# s_bounds = [[0,2 * jnp.pi],[-8,8]]
	# s_bounds = [[0,2 * jnp.pi],[-max_speed,max_speed]]
	s_bounds = [[-jnp.pi,jnp.pi],[-max_speed,max_speed]]
	p0 = jnp.empty((D,2))
	s1 = jnp.linspace(s_bounds[0][0], s_bounds[0][1], D1)
	s2 = jnp.linspace(s_bounds[1][0], s_bounds[1][1],D2)
	i = 0
	p0 = jnp.empty((D,2))
	for (xi,yi) in itertools.product(s1,s2):
		p0 = p0.at[i,:].set(jnp.array((xi,yi)))
		i += 1
	print("p", p0)

	a_bounds = jnp.array([-2,2])
	# key = jax.random.PRNGKey(3)
	key = jax.random.PRNGKey(0)
	a0 = jax.random.uniform(key, minval = a_bounds[0], maxval = a_bounds[1], shape = (D,))
	w0 = jax.random.uniform(key,shape = (D,)) #[0,1]
	actor = sparse_actor(a0,w0,p0,beta)
	return actor


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


def main(env = 'gmfm', save_freq = 50, max_steps = 200, max_speed = 8.0,g = 9.8, init_actor = None, init_actor_iter_idx = None):
	#first init sparse actor
	if env == 'gmfm':
		actor = init_gmfm_actor(beta = 0.1)
		y0 = jnp.array([0.3162,0.0051,0,0])
		a_bounds = jnp.array([-1,1])
		xref = jnp.array([0,0]) #not actually used by gmfm_lorenz
		eval_policy_fn = eval_policy
	elif env == 'lorenz':
		actor = init_lorenz_actor(beta = 1.0)
		y0 = jnp.array([-3.8929, -1.3406, 25.8386])
		a_bounds = jnp.array([-1.,1.])
		xref = jnp.array([-8.4853,-8.4853,27.00])
		eval_policy_fn = eval_policy_lorenz
	elif env == 'pendulum':
		if init_actor == None:
			actor = init_pendulum_actor(beta =jnp.array([0.5,0.5]), max_speed = max_speed)
		else:
			actor = init_actor
		# y0 = jnp.array([jnp.pi,1])
		# y0 = jnp.array([0.1,0.5])
		y0 = jnp.array([jnp.pi,0.])
		a_bounds = jnp.array([-2,2])
		xref = jnp.array([0,0])
		eval_policy_fn = eval_policy_pendulum

	#next train sparse actor
	# max_iters =1000
	max_iters =1000
	# l1_penalty = 0.0
	beta_bounds = jnp.array([0,1])
	# optimizer = optax.adabelief(1e-3)
	# optimizer = optax.adabelief(1e-3)
	optimizer = optax.adabelief(1e-3)
	params = {'a': actor.a, 'w':actor.w, 'beta': actor.beta}
	# params = {'a': actor.a, 'w':actor.w}
	# print(f"hey: {params['beta']}")
	opt_params = optimizer.init(params)
	init_res = eval_policy_fn(params, actor.p,y0, xref, n_steps = max_steps,
		max_speed = max_speed, g = g)
	# print("init res here:", init_res)
	for i in range(max_iters):
		key = jax.random.PRNGKey(i)
		# rand_y0 = jax.random.uniform(key = key, shape = (2,), 
		# 	minval = jnp.array([-jnp.pi,-.5]), maxval = jnp.array([jnp.pi, .5]))
		# print("rand_y0", rand_y0)
		# grads = jax.grad(eval_policy_fn)(params, actor.p,rand_y0, xref, n_steps = 120)
		grads = jax.grad(eval_policy_fn)(params, actor.p,y0, xref, n_steps = max_steps,
			max_speed = max_speed, g = g)
		# grads = jax.grad(test_loss)(params)
		# print("yo", grads)
		updates,opt_params = optimizer.update(grads, opt_params)
		params = optax.apply_updates(params,updates)
		params['a'] = jnp.where(params['a'] < a_bounds[0], a_bounds[0], params['a'])
		params['a'] = jnp.where(params['a'] > a_bounds[1], a_bounds[1], params['a'])
		params['w'] = jnp.where(params['w'] < 0, 0, params['w'])
		params['w'] = jnp.where(params['w'] > 1, 1, params['w'])
		params['beta'] = jnp.where(params['beta'] < 0.01, 0.01, params['beta'])
		# print(f"w[last] at iter{i}, {params['w'][-1]}")
		# print(f"beta at iter{i}, {params['beta']}")
		print(f"iter{i}")

		if i % save_freq == 0 or i == max_iters - 1:

			out_path = "policies/%s/" %(env)
			if not os.path.exists(out_path):
				os.makedirs(out_path)
			#save actor data to a file
			if env == "gmfm":
				f = open(out_path + "gmfm_sparse_actor_iter=%d.pkl" %i,"wb")
			elif env == "lorenz":
				f = open(out_path + "lorenz_sparse_actor=%d.pkl" % i ,"wb")
			elif env == "pendulum":
				if init_actor_iter_idx != None:
					f = open(out_path + "pendulum_sparse_actor=%d.pkl" % (i + init_actor_iter_idx), "wb")
				else:
					f = open(out_path + "pendulum_sparse_actor=%d.pkl" % i, "wb")
			actor_info={}
			actor_info['a'] = params['a']
			actor_info['w'] = params['w']
			actor_info['p'] = actor.p
			actor_info['beta'] = params['beta']
			actor_info['gamma'] = actor.gamma
			actor_info['y0'] = y0
			actor_info['max_steps'] = max_steps
			actor_info['max_speed'] = max_speed
			actor_info['g'] = g
			pickle.dump(actor_info,f)
			f.close()




if __name__ == '__main__':
	t1 = time.time()
	env = "pendulum"
	main(env, max_steps = 200, g  = 9.8, max_speed = 8.0)
	print("this took %.1f seconds" % (time.time() - t1))