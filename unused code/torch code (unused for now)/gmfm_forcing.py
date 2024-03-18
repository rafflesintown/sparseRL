import numpy as np
import torch
import xitorch
import xitorch.integrate.solve_ivp as torch_solve_ivp
import time


def gmfm_dsdt(t,s):
	dsdt = np.zeros(len(s))
	if t < 20. * np.pi:
		b = 0
	else:
		b = 0.2 * np.cos(np.pi * t)

	sigma = 0.1 - np.linalg.norm(s)**2
	dsdt[0] = sigma * s[0] - s[1]
	dsdt[1] = sigma * s[1] + s[0]
	dsdt[2] = -0.1 * s[2] - np.pi * s[3]
	dsdt[3] = -0.1*s[3] + np.pi * s[2] + b
	return dsdt

# def gmfm_forcing_dsdt(t,s,a, w,p,beta):
# 	if t > 20. * np.pi + 2.* np.pi/4.:
# 		pi_s = 0
# 	else:
# 		alpha_s = np.e**(-np.linalg.norm(s[:2] - p[:,:2], axis = 1)**2/beta) * w
# 		pi_s = np.dot(alpha_s/np.sum(alpha_s), a)
# 		# pi_s  = 1 #just for testing

# 	sigma = 0.1 - np.linalg.norm(s)**2
# 	dsdt = np.empty(len(s))
# 	dsdt[0] = sigma * s[0] - s[1]
# 	dsdt[1] = sigma * s[1] + s[0]
# 	dsdt[2] = -0.1 * s[2] - np.pi * s[3]
# 	dsdt[3] = -0.1*s[3] + np.pi * s[2] + pi_s
# 	return dsdt

# def ode_int(model, t_span = (0, 21* np.pi), y0 = np.zeros(4), 
# 	args = (np.zeros(20), np.ones(20), np.ones((20,2)), 1.0)):
# 	a, w,p,beta = args
# 	n_eval = 501
# 	ode_res = solve_ivp(model, t_span = t_span, y0 = y0, args = args, t_eval = np.linspace(t_span[0],t_span[1],n_eval))
# 	y_hist = ode_res.y.reshape(n_eval,-1)
# 	#see https://stackoverflow.com/questions/33677183/subtracting-numpy-arrays-of-different-shape-efficiently to understand code below
# 	alpha_all = np.e**(-np.linalg.norm(p[:,:2] - y_hist[:,:2][:,None], axis = 2)**2/beta) * w 
# 	pi_all = np.dot(alpha_all/np.sum(alpha_all, axis=1)[:,None], a)
# 	print(y_hist.shape)
# 	print(pi_all.shape)
# 	return (y_hist, pi_all)



def gmfm_forcing_dsdt(t,s,a, w,p,beta):
	if t > 20. * torch.pi + 2.* torch.pi/4.:
		pi_s = 0
	else:
		alpha_s = torch.e**(-torch.linalg.norm(s[:2] - p [:,:2], axis = 1)**2/beta) * w
		pi_s = torch.dot(alpha_s/torch.sum(alpha_s), a)
		# pi_s  = 1 #just for testing

	sigma = 0.1 - torch.linalg.norm(s)**2
	dsdt = torch.empty(len(s))
	dsdt[0] = sigma * s[0] - s[1]
	dsdt[1] = sigma * s[1] + s[0]
	dsdt[2] = -0.1 * s[2] - torch.pi * s[3]
	dsdt[3] = -0.1*s[3] + torch.pi * s[2] + pi_s
	return dsdt

def ode_int(model, ts = torch.linspace(0, 21.* torch.pi,501), y0 = torch.zeros(4), 
	params = (torch.zeros(20), torch.ones(20), torch.ones((20,2)), 0.1)):
	a, w,p,beta = params
	n_eval = len(ts)
	ode_res = torch_solve_ivp(model, ts = ts, y0 = y0, params = params)
	y_hist = ode_res.reshape(n_eval,-1)
	#see https://stackoverflow.com/questions/33677183/subtracting-numpy-arrays-of-different-shape-efficiently to understand code below
	alpha_all = torch.e**(-torch.linalg.norm(p[:,:2] - y_hist[:,:2][:,None], axis = 2)**2/beta) * w 
	pi_all = torch.mm(alpha_all/torch.sum(alpha_all, axis=1)[:,None], a[:,None])
	print(y_hist.shape)
	print(pi_all.shape)
	return (y_hist, pi_all)







def main():
	t1 = time.time()
	# print(gmfm_forcing_dsdt(0.5,np.array([0.,0,1,1]), 0,0,0,0))
	sol = ode_int(gmfm_forcing_dsdt,ts = torch.linspace(0,torch.pi,501), y0 = torch.zeros(4), 
		params = (torch.zeros(20), torch.ones(20), torch.ones((20,2)), 1.0))
	# print(sol.y.shape)
	print("this took %.1f seconds" %(time.time() - t1) )

if __name__ == '__main__':
	main()



