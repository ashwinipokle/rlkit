import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rlkit.torch.networks import FlattenMlp

class FeatureMap():
	def __init__(input_dim, feature_dim):
		# Maps s_t to \phi(s_t)
		self.feature_map = FlattenMlp(
			input_size=input_dim,
        	output_size=feature_dim,
        	hidden_sizes=[64, 64])

	def forward(s_t):
		return self.feature_map.forward(s_t)

class ForwardDynamicsModule():
	def __init__(feature_dim, action_dim):
		# Takes in \phi(s_t) and \phi(s_t+1) and outputs a_t (assumes continuous space)
		self.forward_module = FlattenMlp(
			input_size = action_dim + feature_dim,
        	output_size = action_dim,
        	hidden_sizes=[64, 64]
        	)
	def forward(a_t, phi_s_t):
		concat_feats = torch.concat((a_t, phi_s_t), 0)
		return self.forward_module(concat_feats)


# Assumes MLP since we are feeding in global state, for images use CNN
class InverseDynamicsModule():
	def __init__(action_dim, feature_dim=32):
		# Takes in \phi(s_t) and \phi(s_t+1) and outputs a_t (assumes continuous space)
		self.inverse_module = FlattenMlp(
			input_size=2*feature_dim,
        	output_size=action_dim,
        	hidden_sizes=[64, 64]
        	)

	def forward(phi_st, phi_st1):
		concat_feats = torch.concat((phi_st, phi_st1), 0)
		return self.inverse_module(concat_feats)

class IntrinsicCuriosityModule():
	def __init__(input_dim, action_dim, feature_dim, 
					optimizer_class=optim.Adam,
					eta=1):
		self.feature_map = FeatureMap(input_dim, feature_dim)
		self.forward_module = ForwardDynamicsModule(feature_dim, action_dim)
		self.inverse_module = InverseDynamicsModule(action_dim, feature_dim)

	def forward(s_t, s_t1, a_t):
		phi_st = self.feature_map.forward(s_t)
		phi_st1 = self.feature_map.forward(s_t1)
		phi_hat_t = self.forward_module.forward(a_t, phi_st)
		intrinsic_reward = eta/2.0 * torch.norm(phi_hat_t - phi_st1, p=2)**2
		a_t_1 = self.inverse_module.forward(phi_st, phi_st1)
		return a_t_1, intrinsic_reward