import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import jit
from jax import lax
import jax
import time





def update(s,t):
	s= s.at[0].set(s[0] + 0.1)
	s = s.at[1].set(s[1]-0.1)
	return s, s










def main():
	t1 = time.time()
	# print(gmfm_forcing_dsdt(0.5,np.array([0.,0,1,1]), 0,0,0,0))
	# args = (jnp.ones(20), jnp.ones(20), jnp.ones((20,2)), 0.1)
	# sol = get_ode_res(gmfm_forcing_dsdt,t = jnp.linspace(0,jnp.pi,10), y0 = jnp.ones(4)*0.1, 
	# 	args = args)
	# print(gmfm_forcing_dsdt(jnp.zeros(4), 0,*args))
	# params = 
	# grads = jax.grad(get_ode_res)(params)
	# print(sol.y.shape)

	t = jnp.linspace(0,5,200)
	s_init = jnp.array([0.5,1])
	final, result = lax.scan(update, s_init, t)
	print("fnal", final)
	print("result", result)
	print("this took %.3f seconds" %(time.time() - t1) )

	t1 = time.time()
	s = s_init
	for i in range(200):
		s= s.at[0].set(s[0] + 0.1)
		s = s.at[1].set(s[1]-0.1)
	print("this took %.3f seconds" %(time.time() - t1) )

if __name__ == '__main__':
	main()



