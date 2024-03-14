import pickle
import jax.numpy as jnp
import matplotlib.pyplot as plt


from env_jax import *

from train import * 

from pend_for_visualization import PendulumEnv

from os import path
from typing import Optional

import numpy as np
import time

import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


def main(env = "gmfm", in_path = None, in_path_2 = None, out_path = None, iter_idx = 100, iter_idx_2 = None,
	visualize = False, max_steps = 120, dt = 0.05, max_speed = 8.0, stabilize = False, combine = False):

	if env == "gmfm":
		file = open(in_path + 'gmfm_sparse_actor=%d.pkl'%iter_idx , 'rb')
	elif env == "pendulum":
		file = open(in_path + 'pendulum_sparse_actor=%d_max_speed=%.1f.pkl' % (iter_idx,max_speed), 'rb')
		if combine == True:
			file2 = open(in_path_2 + 'pendulum_sparse_actor=%d_max_speed=%.1f.pkl' % (iter_idx_2,max_speed), 'rb')
			actor2 = pickle.load(file2)
			file2.close()
	actor = pickle.load(file)
	file.close()
	w = actor['w']
	a = actor['a']
	p = actor['p']
	beta = actor['beta']
	gamma = actor['gamma']
	if combine == True:
		w2 = actor2['w']
		a2 = actor2['a']
		p2 = actor2['p']
		beta2 = actor2['beta']
	y0 = actor['y0']
	g = actor['g']
	max_speed = actor['max_speed']
	max_steps = actor['max_steps']
	if stabilize == True:
		max_steps = 100
		y0 = jnp.array([-0.3,0.4])
	if combine == True:
		max_steps = 250
		y0 = jnp.array([jnp.pi,0.])
	if env == "gmfm":
		ts = jnp.linspace(0,21 * jnp.pi,501)
	elif env == "pendulum":
		T = dt * max_steps
		ts = jnp.linspace(0,T,max_steps)
	# print("w", w)
	# print("p non-zero weight coords", p[w > 0,:])
	print(actor['beta'], "beta")
	w_amax = jnp.argmax(w)
	# print("argmax w", jnp.argmax(w))
	# print("a", a[jnp.where(w > 0.)])
	# print("p at w_amax", actor['p'][w_amax,:])
	policy = [a,w,p,beta]
	if combine == True:
		policy = [a,w,p,beta,a2,w2,p2,beta2]
	if env == "gmfm":
		y_all, pi_all = get_ode_res(gmfm_forcing_dsdt,y0 = y0, t = ts,args  = policy)
		y_baseline,pi_baseline = get_ode_res(gmfm_dsdt,y0 = y0, t = ts,args  = policy)
	elif env =="pendulum":
		print("THIS IS y0: ", y0)
		# y_all, pi_all = get_pendulum_res(pendulum_dsdt, y0 = y0, t = ts, args = policy)
		init_actor = init_pendulum_actor(beta =jnp.array([0.5,0.5]), max_speed = max_speed)
		init_policy = [init_actor.a,init_actor.w,init_actor.p,init_actor.beta]
		xref = jnp.array([0,0])
		if combine == False:
			y_all, pi_all = get_pendulum_res_2(y0 = y0, args = policy, n_steps = max_steps)
		else: #combine two policies
			with jax.disable_jit():
				y_all, pi_all = get_pendulum_res_2(y0 = y0, args = policy, n_steps = max_steps, combine = True)
		y_baseline, pi_baseline= get_pendulum_res_2(y0 = y0, args = init_policy, n_steps = max_steps)
	# print(y_all[-50:,:],"hey")
	print("y_all", y_all)
	print("pi_all", pi_all)
	# print(jnp.mean(jnp.linalg.norm(y_all[:,:2], axis = 1)**2) + gamma * jnp.mean(pi_all **2))
	plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

	for idx in range(y_all.shape[1]):
		if env == "gmfm":
			plt.plot(ts[:300], y_all[:300,idx], label = "sparse clustering")
			plt.plot(ts[:300], y_baseline[:300,idx], label = "baseline")
		elif env == "pendulum": 
			plt.plot(ts, y_all[:,idx], label = "sparse clustering")
			plt.plot(ts, y_baseline[:,idx], label = "baseline")
			if idx ==0:
				plt.ylabel(r"$\rho$" )
			elif idx == 1:
				plt.ylabel(r"$\dot{\rho}$")

		plt.xlabel("time (s)")
		plt.legend()
		# plt.title("s0 plot")
		plt.grid()
		plt.savefig(out_path + "s%d_vs_baseline_traj_%s_iter_idx=%d.pdf" % (idx +1, env, iter_idx), 
			bbox_inches = 'tight')
		plt.close()

	plt.scatter(p[:,0],p[:,1], c = "#440154")
	plt.savefig(out_path + "anchor_location_%s_iter_idx=%d.pdf" % (env,iter_idx), bbox_inches = 'tight')
	plt.scatter(p[:,0],p[:,1], c = w, cmap = 'viridis')
	if env == 'pendulum':
		plt.xlabel(r"$\rho$")
		plt.ylabel(r"$\dot{\rho}$")
	else:
		plt.xlabel(r"$s_1$")
		plt.ylabel(r"$s_2$")
	plt.colorbar()
	plt.savefig(out_path + "w_heatmap_%s_after_iter_idx=%d.pdf" % (env,iter_idx), bbox_inches = 'tight')
	plt.close()


	if combine == True:
		plt.scatter(p2[:,0],p2[:,1], c = "#440154")
		plt.savefig(out_path + "anchor_location_%s_iter_idx=%d_stabilizer.pdf" % (env,iter_idx_2), bbox_inches = 'tight')
		plt.scatter(p2[:,0],p2[:,1], c = w2, cmap = 'viridis')
		if env == 'pendulum':
			plt.xlabel(r"$\rho$")
			plt.ylabel(r"$\dot{\rho}$")
		else:
			plt.xlabel(r"$s_1$")
			plt.ylabel(r"$s_2$")
		plt.colorbar()
		plt.savefig(out_path + "w_heatmap_%s_after_iter_idx=%d_stabilizer.pdf" % (env,iter_idx_2), bbox_inches = 'tight')
		plt.close()

	if env == "pendulum" and visualize == True:
		pend = PendulumEnv(g = g, max_speed = max_speed, max_episode_steps = max_steps)
		init_state = np.asarray(y0)
		cmd = np.asarray(pi_all)
		# pend.visualize(init_state = init_state, cmd = cmd[:])
		pend.visualize_v2(states = y_all, cmd = cmd[:])

if __name__ == '__main__':
	# env = "gmfm"
	env = 'pendulum'
	# stabilize = True
	# stabilize = False
	combine = True
	max_speed = 8.0
	in_path = "policies/%s/stabilize=%d/" % (env, False)
	in_path_2 = "policies/%s/stabilize=%d/" % (env, True)
	if combine == True:
		out_path = 'plots/%s/combine/' % (env)
	else:
		out_path = 'plots/%s/stabilize=%d/' % (env,stabilize)
	if not os.path.exists(out_path):
		os.makedirs(out_path)
	iter_idx = 200
	iter_idx_2 = 999
	# main(env, in_path = in_path, out_path = out_path, iter_idx = iter_idx, iter_idx_2 = iter_idx_2,
	# 	visualize = False, max_speed = max_speed, stabilize = stabilize)
	main(env, in_path = in_path, in_path_2 = in_path_2, out_path = out_path, iter_idx = iter_idx, iter_idx_2 = iter_idx_2,
	visualize = False, max_speed = max_speed, combine = combine)


