import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, make_barplot, features_blobs, setup_cam, returnCAM
import matplotlib as mpl
import random

mpl.use('Agg')
import matplotlib.pyplot as plt
import model.metric
import model.loss



class Trainer(BaseTrainer):
	"""
	Trainer class
	"""
	def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
		super().__init__(model, criterion, metric_ftns, optimizer, config)
		self.data_loader = data_loader

		if len_epoch is None:
			# epoch-based training
			self.len_epoch = len(self.data_loader)
		else:
			# iteration-based training
			self.data_loader = inf_loop(data_loader)
			self.len_epoch = len_epoch

		self.criterion = criterion
		self.valid_data_loader = valid_data_loader
		self.do_validation = self.valid_data_loader is not None
		self.lr_scheduler = lr_scheduler
		self.log_step = int(np.sqrt(data_loader.batch_size))

		self.train_metrics = MetricTracker('time','loss', 'balanced_accuracy', writer=self.writer)
		self.valid_metrics = MetricTracker('time','loss', 'balanced_accuracy', writer=self.writer)


	def _train_epoch(self, epoch, phase="train"):
		"""
		Training logic for an epoch

		:param epoch: Integer, current training epoch.
		:return: A log that contains average loss and metric in this epoch.
		"""
		import torch.nn.functional as F
		import model.loss
		import time

		start = time.time()

		print("Finding LR")
		for param_group in self.optimizer.param_groups:
			print(param_group['lr'])

		if phase == "train":
			self.model.train()
			self.train_metrics.reset()
			torch.set_grad_enabled(True)
			metrics = self.train_metrics
		elif phase == "val" or phase == "test":
			self.model.eval()
			self.valid_metrics.reset()
			torch.set_grad_enabled(False)
			metrics = self.valid_metrics

		outputs = []
		outputs_continuous = []
		targets = []
		targets_continuous = []

		data_loader = self.data_loader if phase == "train" else self.valid_data_loader


		for batch_idx, (data, target) in enumerate(data_loader):
			# print(data.size())
			data, target = data.to(self.device), target.to(self.device)

			if phase == "train":
				self.optimizer.zero_grad()

			out = self.model(data)

			loss = 0
			loss_categorical = self.criterion(out['categorical'], target)
			loss += loss_categorical
			
			if phase == "train":
				loss.backward()
				self.optimizer.step()

			output = out['categorical'].cpu().detach().numpy()
			target = target.cpu().detach().numpy()
			outputs.append(output)
			targets.append(target)

			if batch_idx % self.log_step == 0:

				self.logger.debug('{} Epoch: {} {} Loss: {:.6f} '.format(
					phase,
					epoch,
					self._progress(batch_idx, data_loader),
					loss.item()))

			if batch_idx == self.len_epoch:
				break

		if phase == "train":
			self.writer.set_step(epoch)
		else:
			self.writer.set_step(epoch, "valid")

		metrics.update('loss', loss.item())


		metrics.update('time', time.time()-start)

		output = np.concatenate(outputs, axis=0)
		target = np.concatenate(targets, axis=0)
		
		ap = model.metric.balanced_accuracy(output, target)

		metrics.update("balanced_accuracy", np.mean(ap))
	
		log = metrics.result()
		
		if phase == "train":

			if self.do_validation:
				val_log = self._train_epoch(epoch, phase="val")
				log.update(**{'val_' + k: v for k, v in val_log.items()})

			return log

		elif phase == "val" or phase == 'test':
			self.writer.save_results(output, f"{phase}_output")

			if self.lr_scheduler is not None and phase != 'test':
				# self.lr_scheduler.step(np.mean(ap))
				self.lr_scheduler.step()

			return metrics.result()


	def _progress(self, batch_idx, data_loader):
		base = '[{}/{} ({:.0f}%)]'
		total = len(data_loader)
		current = batch_idx
		return base.format(current, total, 100.0 * current / total)
