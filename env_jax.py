import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import jit
from jax import lax
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

#Adds no control
def gmfm_dsdt(s,t,a, w,p,beta):
	# alpha_s = jnp.e**(-jnp.linalg.norm(s[:2] - p[:,:2], axis = 1)**2/beta) * w
	# pi_s = jnp.dot(alpha_s/jnp.sum(alpha_s), a)
	# # jax.debug.print("{x} here, s = {s}", x = jnp.sum(alpha_s), s = s)
	# # pi_s  = 1 #just for testing

	sigma = 0.1 - jnp.linalg.norm(s)**2
	dsdt = jnp.empty(len(s))
	dsdt = dsdt.at[0].set(sigma * s[0] - s[1])
	dsdt = dsdt.at[1].set(sigma * s[1] + s[0])
	dsdt = dsdt.at[2].set(-0.1 * s[2] - jnp.pi * s[3])
	dsdt = dsdt.at[3].set(-0.1*s[3] + jnp.pi * s[2])
	# print("wow")
	# jax.debug.print("{x}", x = dsdt[0])

	return dsdt


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

def lorenz_forcing_dsdt(s,t,a,w,p,beta):
	sigma = 10.
	rho = 28.
	beta_lorenz = 8./3.

	alpha_s = jnp.e**(-jnp.linalg.norm(s - p, axis = 1)**2/beta) * w
	pi_s = jnp.dot(alpha_s/jnp.sum(alpha_s),a)
	dsdt = jnp.empty(len(s))
	dsdt = dsdt.at[0].set(sigma * (s[1] - s[0]))
	dsdt = dsdt.at[1].set(s[0]*(rho - s[2]) - s[1])
	dsdt = dsdt.at[2].set(s[0] * s[1] - beta_lorenz * s[2])

	return dsdt


def pendulum_dsdt(s,t,a,w,p,beta):
	g = 10.0
	m = 1.0
	l = 1.0
	max_speed = 8
	s_norm = s.at[0].set(s[0] % (2. * jnp.pi))
	# alpha_s = jnp.e**(-jnp.linalg.norm(s_norm - p, axis = 1)**2/beta) * w
	alpha_s = jnp.e**(-(s_norm[0] - p[:,0])**2/beta[0] - (s_norm[1] - p[:,1])**2/beta[1]) * w
	pi_s = jnp.dot(alpha_s/jnp.sum(alpha_s),a)	
	dsdt = jnp.empty(len(s))
	dsdt = dsdt.at[1].set((3. * g/(2. * l) * jnp.sin(s[0]) + 3./(m * l**2) * pi_s))
	dsdt = dsdt.at[0].set(s[1])
	return dsdt

def get_pendulum_res(model, t = jnp.linspace(0, 5,100), y0 = jnp.zeros(2), 
	args = (jnp.zeros(20), jnp.ones(20), jnp.ones((20,2)), 0.5)):
	a, w,p,beta = args
	ode_res = odeint(model, y0 , t, *args)
	# return ode_res[-1,0]
	print(ode_res.shape, "ode shape")

	y_hist = ode_res.at[:,0].set(ode_res[:,0] % (2. * jnp.pi))
	#see https://stackoverflow.com/questions/33677183/subtracting-numpy-arrays-of-different-shape-efficiently to understand code below
	# alpha_all = jnp.e**(-jnp.linalg.norm(p - y_hist[:,None], axis = 2)**2/beta) * w 
	# print("y_hist[:,0][:,None] shape", y_hist[:,0][:,None].shape)
	alpha_all = jnp.e**(-(p[:,0] - y_hist[:,0][:,None])**2/beta[0] - (p[:,1] - y_hist[:,1][:,None])**2/beta[1]) * w 
	# print("alpha_all shape", alpha_all.shape)
	# print("alpha_all[4] all ", alpha_all[4,:])
	pi_all = jnp.dot(alpha_all/jnp.sum(alpha_all, axis=1)[:,None], a)
	# print(y_hist[-10:])
	# print(pi_all.shape)
	return (y_hist, pi_all)

#for use in lax.scan
#i is unused
def pendulum_1step(carry,i, max_speed = 8., g = 2.0):
	s, a,w,p,beta, max_speed,g = carry
	# g = 10.0
	m = 1.0
	l = 1.0
	dt = 0.05
	# s_norm = s.at[0].set(s[0] % (2. * jnp.pi))
	# alpha_s = jnp.e**(-jnp.linalg.norm(s_norm - p, axis = 1)**2/beta) * w
	# intermediate =  jnp.linalg.norm(s - p, axis = 1)**2
	# jax.debug.print("intermediate shape {x}", x = intermediate.shape)
	# print("intermediate shape", intermediate.shape)
	# print("beta shape", beta.shape)
	alpha_s = jnp.e**(-jnp.sum((s - p)**2/beta, axis = 1)) * w
	pi_s = jnp.dot(alpha_s/jnp.sum(alpha_s),a)	
	s = s.at[1].set(s[1] + (3. * g/(2. * l) * jnp.sin(s[0]) + 3./(m * l**2) * pi_s) * dt)
	s = s.at[1].set(jnp.clip(s[1], -max_speed, max_speed))
	# print("state is: ", s)
	# jax.debug.print("s1:{s1}, pi_s:{pi}", s1 = s[1] + (3. * g/(2. * l) * jnp.sin(s[0]) + 3./(m * l**2) * pi_s) * dt, pi = pi_s)
	s = s.at[0].set(s[0] + s[1]*dt) 
	s = s.at[0].set((s[0] + jnp.pi) % (2.*jnp.pi) -jnp.pi) #angle normalization
	s_and_pi = jnp.concatenate((s,pi_s), axis = None)
	# jax.debug.print("s and pi = {x}", x = s_and_pi)
	# jax.debug.print("max alpha_s = {x}", x = jnp.max(alpha_s))
	# print("s and pi: ",s_and_pi)
	# print("max alpha_s: ",jnp.max(alpha_s))
	return (s,a,w,p,beta, max_speed,g), s_and_pi #carry, res to be accumulated


#for use in lax.scan
#i is unused
#combines two policies
def pendulum_1step_combine(carry,i, max_speed = 8., g = 2.0):
	s, a,w,p,beta,a2,w2,p2,beta2, max_speed,g = carry
	# g = 10.0
	m = 1.0
	l = 1.0
	dt = 0.05
	if jnp.abs(s[0]) <= 0.3 and jnp.abs(s[1]) <= 0.4:
		a = a2 
		w = w2
		p = p2 
		beta = beta2
	# s_norm = s.at[0].set(s[0] % (2. * jnp.pi))
	# alpha_s = jnp.e**(-jnp.linalg.norm(s_norm - p, axis = 1)**2/beta) * w
	# intermediate =  jnp.linalg.norm(s - p, axis = 1)**2
	# jax.debug.print("intermediate shape {x}", x = intermediate.shape)
	# print("intermediate shape", intermediate.shape)
	# print("beta shape", beta.shape)
	alpha_s = jnp.e**(-jnp.sum((s - p)**2/beta, axis = 1)) * w
	pi_s = jnp.dot(alpha_s/jnp.sum(alpha_s),a)	
	s = s.at[1].set(s[1] + (3. * g/(2. * l) * jnp.sin(s[0]) + 3./(m * l**2) * pi_s) * dt)
	s = s.at[1].set(jnp.clip(s[1], -max_speed, max_speed))
	# print("state is: ", s)
	# jax.debug.print("s1:{s1}, pi_s:{pi}", s1 = s[1] + (3. * g/(2. * l) * jnp.sin(s[0]) + 3./(m * l**2) * pi_s) * dt, pi = pi_s)
	s = s.at[0].set(s[0] + s[1]*dt) 
	s = s.at[0].set((s[0] + jnp.pi) % (2.*jnp.pi) -jnp.pi) #angle normalization
	s_and_pi = jnp.concatenate((s,pi_s), axis = None)
	# jax.debug.print("s and pi = {x}", x = s_and_pi)
	# jax.debug.print("max alpha_s = {x}", x = jnp.max(alpha_s))
	# print("s and pi: ",s_and_pi)
	# print("max alpha_s: ",jnp.max(alpha_s))
	return (s,a,w,p,beta, a2,w2,p2,beta2,max_speed,g), s_and_pi #carry, res to be accumulated


def get_pendulum_res_2(args = (jnp.zeros(20),jnp.ones(20),jnp.ones((20,2)),0.1),
	n_steps = 120,y0 = jnp.array([jnp.pi,1]), max_speed = 8.0, g = 9.8, combine = False):

	if combine == False:
		a,w,p,beta = args
	else:
		a,w,p,beta, a2,w2,p2,beta2 = args
	s_init = y0
	if combine == False:
		final, s_pi_hist = lax.scan(pendulum_1step, (s_init,a,w,p,beta,max_speed, g), 
			jnp.arange(n_steps))
	else:
		final, s_pi_hist = lax.scan(pendulum_1step_combine, (s_init,a,w,p,beta,a2,w2,p2,beta2,max_speed, g), 
			jnp.arange(n_steps))
	y_hist = s_pi_hist[:,:2]
	pi_all = s_pi_hist[:,-1]
	return (y_hist,pi_all)


# def get_pendulum_res(args = (jnp.zeros(20),jnp.ones(20),jnp.ones((20,2)),0.1),n_steps = 100,y0 = jnp.array([jnp.pi,1])):

# 	a, w,p,beta = args
# 	g = 10.0
# 	m = 1.0
# 	l = 1.0
# 	dt = 0.05
# 	max_speed = 8
# 	s = y0
# 	y_hist = jnp.empty((n_steps,len(s)))
# 	pi_all = jnp.empty(n_steps)
# 	for i in jnp.arange(n_steps):
# 		alpha_s = jnp.e**(-jnp.linalg.norm(s - p, axis = 1)**2/beta) * w
# 		# jax.debug.print("s-p: {diff}", diff = s-p)
# 		# jax.debug.print("alpha_s: {alpha_s}", alpha_s = alpha_s)
# 		# jax.debug.print("w:{w}", w = w)
# 		pi_s = jnp.dot(alpha_s/jnp.sum(alpha_s),a)
# 		s = s.at[1].set(s[1] + (3. * g/(2. * l) * jnp.sin(s[0]) + 3./(m * l**2) * pi_s) * dt)
# 		s = s.at[1].set(jnp.clip(s[1], -max_speed, max_speed))
# 		# jax.debug.print("s1:{s1}, pi_s:{pi}", s1 = s[1] + (3. * g/(2. * l) * jnp.sin(s[0]) + 3./(m * l**2) * pi_s) * dt, pi = pi_s)
# 		s = s.at[0].set(s[0] + s[1]*dt) 
# 		s = s.at[0].set(s[0] % (2.*jnp.pi)) #angle normalization
# 		# jax.debug.print("s:{s}, i = {i}", s = s, i = i)
# 		y_hist = y_hist.at[i,:].set(s)
# 		pi_all = pi_all.at[i].set(pi_s)
# 	return (y_hist,pi_all)





# for gmfm
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


#for lorenz
def get_ode_res_lorenz(model, t = jnp.linspace(0, 21* jnp.pi,501), y0 = jnp.zeros(3), 
	args = (jnp.zeros(20), jnp.ones(20), jnp.ones((20,3)), 0.1)):
	a, w,p,beta = args
	ode_res = odeint(model, y0 , t, *args)
	# return ode_res[-1,0]

	y_hist = ode_res
	#see https://stackoverflow.com/questions/33677183/subtracting-numpy-arrays-of-different-shape-efficiently to understand code below
	alpha_all = jnp.e**(-jnp.linalg.norm(p - y_hist[:,None], axis = 2)**2/beta) * w 
	# print("alpha_all shape", alpha_all.shape)
	# print("alpha_all[4] all ", alpha_all[4,:])
	pi_all = jnp.dot(alpha_all/jnp.sum(alpha_all, axis=1)[:,None], a)
	# print(y_hist[-10:])
	# print(pi_all.shape)
	return (y_hist, pi_all)







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

	args = (jnp.ones(20), jnp.ones(20), jnp.ones((20,3)), 0.1)
	sol = get_ode_res(gmfm_forcing_dsdt,t = jnp.linspace(0,21 * jnp.pi,50), y0 = jnp.ones(3)*0.1, 
		args = args)
	print(sol)
	print("this took %.1f seconds" %(time.time() - t1) )

if __name__ == '__main__':
	main()



