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


def main(env = "gmfm", in_path = None, out_path = None, iter_idx = 100, 
	visualize = False, max_steps = 120, dt = 0.05):

	if env == "gmfm":
		file = open(in_path + 'gmfm_sparse_actor=%d.pkl'%iter_idx , 'rb')
	elif env == "pendulum":
		file = open(in_path + 'pendulum_sparse_actor=%d.pkl' % iter_idx, 'rb')
	actor = pickle.load(file)
	file.close()
	w = actor['w']
	a = actor['a']
	p = actor['p']
	beta = actor['beta']
	gamma = actor['gamma']
	y0 = actor['y0']
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
	if env == "gmfm":
		y_all, pi_all = get_ode_res(gmfm_forcing_dsdt,y0 = y0, t = ts,args  = policy)
		y_baseline,pi_baseline = get_ode_res(gmfm_dsdt,y0 = y0, t = ts,args  = policy)
	elif env =="pendulum":
		# y_all, pi_all = get_pendulum_res(pendulum_dsdt, y0 = y0, t = ts, args = policy)
		init_actor = init_pendulum_actor(beta =jnp.array([0.5,0.5]))
		init_policy = [init_actor.a,init_actor.w,init_actor.p,init_actor.beta]
		xref = jnp.array([0,0])
		y_all, pi_all = get_pendulum_res_2(y0 = y0, args = policy, n_steps = max_steps)
		y_baseline, pi_baseline= get_pendulum_res_2(y0 = y0, args = init_policy, n_steps = max_steps)
	# print(y_all[-50:,:],"hey")
	print("y_all", y_all)
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

	if env == "pendulum" and visualize == True:
		pend = PendulumEnv(g = 2.0, max_speed = 2.0,max_episode_steps = 120)
		init_state = np.asarray(y0)
		cmd = np.asarray(pi_all)
		pend.visualize(init_state = init_state, cmd = cmd[:])

if __name__ == '__main__':
	# env = "gmfm"
	env = 'pendulum'
	in_path = "policies/%s/" % env
	out_path = 'plots/%s/' % env
	iter_idx = 700
	main(env, in_path = in_path, out_path = out_path, iter_idx = iter_idx, visualize = False)


