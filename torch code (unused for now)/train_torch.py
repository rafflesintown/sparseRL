import numpy as np
import torch
import itertools

from gmfm_forcing import *


class sparse_actor:
	def __init__(self, a,w,p,beta = 0.1, gamma = 0.1):
		self.a = torch.tensor(a, requires_grad = True, dtype = torch.float32)
		self.w = torch.tensor(w, requires_grad = True, dtype = torch.float32)
		self.p = torch.tensor(p, requires_grad = False, dtype = torch.float32)
		self.beta = torch.tensor(beta, requires_grad = True, dtype = torch.float32)
		self.gamma = gamma
	def eval_policy(self, env = 'gmfm',ts = torch.linspace(0, 21 * np.pi,501), y0= torch.zeros(4)):
		policy = [self.a, self.w, self.p,self.beta]
		if env == 'gmfm':
			y_all, pi_all = ode_int(gmfm_forcing_dsdt,ts = ts, y0 = y0, 
		params = policy)
			J = torch.mean(torch.linalg.norm(y_all[:2])**2) + self.gamma * torch.mean(pi_all **2)
			return J


def init_gmfm_actor(beta  = 0.1):
	D1 = 5
	D2 = 5
	D = D1 * D2
	s_bounds = [-1,1]
	p0 = torch.empty((D,2))
	s1 = torch.linspace(s_bounds[0], s_bounds[1], D1)
	s2 = torch.linspace(s_bounds[0], s_bounds[1],D2)
	i = 0
	p0 = torch.empty((D,2))
	for (xi,yi) in itertools.product(s1,s2):
		p0[i,:] = torch.tensor((xi,yi))
		i += 1

	torch.manual_seed(0)
	a_bounds = torch.tensor([-1,1])
	a0 = torch.FloatTensor(D,).uniform_(a_bounds[0], a_bounds[1])
	w0 = torch.FloatTensor(D,).uniform_(0,1)
	actor = sparse_actor(a0,w0,p0,beta)
	return actor

def main(env = 'gmfm'):
	#first init sparse actor
	if env == 'gmfm':
		actor = init_gmfm_actor(beta = 0.1)

	#next train sparse actor
	max_iters = 10
	l1_penalty = 1.0
	y0 = torch.zeros(4)
	a_bounds = torch.tensor(np.array([-1,1]), dtype = torch.float32)
	beta_bounds = torch.tensor(np.array([0,1]), dtype = torch.float32)
	optimizer = torch.optim.Adam([actor.a, actor.w,actor.beta], lr = 1e-3)
	for i in range(max_iters):
		optimizer.zero_grad()
		policy_cost = actor.eval_policy(y0 = y0)
		loss = policy_cost + l1_penalty * torch.linalg.norm(actor.w, ord = 1)
		loss.backward(retain_graph = True)
		optimizer.step()
		actor.a = torch.where(actor.a < a_bounds[0], a_bounds[0], actor.a)
		actor.a = torch.where(actor.a > a_bounds[1], a_bounds[1], actor.a)
		actor.w = torch.where(actor.w < 0, 0, actor.w)
		actor.w = torch.where(actor.w > 1, 1, actor.w)
		actor.beta = torch.where(actor.beta < beta_bounds[0], beta_bounds[0], actor.beta)
		actor.beta = torch.where(actor.beta > beta_bounds[1], beta_bounds[1], actor.beta)
		print(f"evaluation cost at iter{i}: {policy_cost}" )
		print(f"l1 cost at iter{i}: {torch.linalg.norm(actor.w, ord = 1)}")
		print(f"beta at iter{i}, {actor.beta}")
		print(actor.a.requires_grad)





if __name__ == '__main__':
	main()