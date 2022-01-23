import yaml
import numpy as np


class UtilsSolutionFile:
	def load(self, filename: str) -> None:
		with open(filename) as f:
			self.file = yaml.safe_load(f)
		self.states = np.array(self.file['result'][0]['states'])
		if 'actions' in self.file['result'][0]:
			self.actions = np.array(self.file['result'][0]['actions'])

	def T(self) -> int:
		return self.states.shape[0] - 1

	def save_rescaled(self, filename:str, T: int) -> None:
		T_orig = self.T()
		state_dim = self.states.shape[1]
		states_interp = np.empty((T+1, state_dim))
		for k in range(state_dim):
			states_interp[:,k] = np.interp(np.linspace(0,1,T+1), np.linspace(0, 1, T_orig+1), self.states[:,k])
		
		if 'actions' in self.file['result'][0]:
			action_dim = self.actions.shape[1]
			actions_interp = np.empty((T, action_dim))
			for k in range(self.actions.shape[1]):
				actions_interp[:,k] = np.interp(np.linspace(0,1,T), np.linspace(0, 1, T_orig), self.actions[:,k])

		with open(filename, 'w') as f:
			self.file['result'][0]['states'] = states_interp.tolist()
			if 'actions' in self.file['result'][0]:
				self.file['result'][0]['actions'] = actions_interp.tolist()
			yaml.dump(self.file, f)
