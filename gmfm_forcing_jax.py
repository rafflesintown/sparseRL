import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import jit
import jax
import time


# def gmfm_dsdt(t,s):
# 	dsdt = np.zeros(len(s))
# 	if t < 20. * np.pi:
# 		b = 0
# 	else:
# 		b = 0.2 * np.cos(np.pi * t)

# 	sigma = 0.1 - np.linalg.norm(s)**2
# 	dsdt[0] = sigma * s[0] - s[1]
# 	dsdt[1] = sigma * s[1] + s[0]
# 	dsdt[2] = -0.1 * s[2] - np.pi * s[3]
# 	dsdt[3] = -0.1*s[3] + np.pi * s[2] + b
# 	return dsdt


def gmfm_forcing_dsdt(s,t,a, w,p,beta):
	alpha_s = jnp.e**(-jnp.linalg.norm(s[:2] - p[:,:2], axis = 1)**2/beta) * w
	pi_s = jnp.dot(alpha_s/jnp.sum(alpha_s), a)
	# jax.debug.print("{x} here, s = {s}", x = jnp.sum(alpha_s), s = s)
	# pi_s  = 1 #just for testing

	sigma = 0.1 - jnp.linalg.norm(s)**2
	dsdt = jnp.empty(len(s))
	dsdt = dsdt.at[0].set(sigma * s[0] - s[1])
	dsdt = dsdt.at[1].set(sigma * s[1] + s[0])
	dsdt = dsdt.at[2].set(-0.1 * s[2] - jnp.pi * s[3])
	dsdt = dsdt.at[3].set(-0.1*s[3] + jnp.pi * s[2] + pi_s)
	# print("wow")
	# jax.debug.print("{x}", x = dsdt[0])

	return dsdt


def get_ode_res(model, t = jnp.linspace(0, 21* jnp.pi,501), y0 = jnp.zeros(4), 
	args = (jnp.zeros(20), jnp.ones(20), jnp.ones((20,2)), 0.1)):
	a, w,p,beta = args
	ode_res = odeint(model, y0 , t, *args)
	# return ode_res[-1,0]

	y_hist = ode_res
	#see https://stackoverflow.com/questions/33677183/subtracting-numpy-arrays-of-different-shape-efficiently to understand code below
	alpha_all = jnp.e**(-jnp.linalg.norm(p[:,:2] - y_hist[:,:2][:,None], axis = 2)**2/beta) * w 
	pi_all = jnp.dot(alpha_all/jnp.sum(alpha_all, axis=1)[:,None], a)
	# print(y_hist[-10:])
	# print(pi_all.shape)
	return (y_hist, pi_all)










def main():
	t1 = time.time()
	# print(gmfm_forcing_dsdt(0.5,np.array([0.,0,1,1]), 0,0,0,0))
	args = (jnp.ones(20), jnp.ones(20), jnp.ones((20,2)), 0.1)
	sol = get_ode_res(gmfm_forcing_dsdt,t = jnp.linspace(0,jnp.pi,10), y0 = jnp.ones(4)*0.1, 
		args = args)
	# print(gmfm_forcing_dsdt(jnp.zeros(4), 0,*args))
	# params = 
	# grads = jax.grad(get_ode_res)(params)
	# print(sol.y.shape)
	print("this took %.1f seconds" %(time.time() - t1) )

if __name__ == '__main__':
	main()



