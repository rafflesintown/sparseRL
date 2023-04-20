import pickle
import jax.numpy as jnp
import matplotlib.pyplot as plt


from gmfm_forcing_jax import *




def main():

	file = open('gmfm_sparse_actor.pkl', 'rb')
	actor = pickle.load(file)
	file.close()
	w = actor['w']
	a = actor['a']
	p = actor['p']
	beta = actor['beta']
	gamma = actor['gamma']
	y0 = actor['y0']
	ts = jnp.linspace(0,21 * jnp.pi,501)
	print("w", w)
	print(actor['beta'], "beta")
	w_amax = jnp.argmax(w)
	# print("argmax w", jnp.argmax(w))
	# print("a", a[jnp.where(w > 0.)])
	# print("p at w_amax", actor['p'][w_amax,:])
	policy = [a,w,p,beta]
	y_all, pi_all = get_ode_res(gmfm_forcing_dsdt,y0 = y0, t = ts,args  = policy)
	# print(y_all[-50:,:],"hey")
	print(jnp.mean(jnp.linalg.norm(y_all[:,:2], axis = 1)**2) + gamma * jnp.mean(pi_all **2))
	plt.plot(ts, y_all[:,0])
	plt.title("s0 plot")
	plt.grid()
	plt.savefig("s0_traj.pdf", bbox_inches = 'tight')
	plt.close()
	plt.plot(ts, pi_all)
	plt.title("pi_s plot")
	plt.grid()
	plt.savefig("pi_traj.pdf", bbox_inches = 'tight')
	plt.close()
	plt.scatter(p[:,0],p[:,1], c = w)
	plt.savefig("w_heatmap.pdf", bbox_inches = 'tight')
	plt.close()


if __name__ == '__main__':
	main()


