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
	def __init__(self, model, criterion, criterion_continuous, metric_ftns, metric_ftns_continuous, optimizer, config, data_loader, categorical=True, continuous=True,
				 valid_data_loader=None, lr_scheduler=None, len_epoch=None, embed=False, lossembed="mse", audio=False):
		super().__init__(model, criterion, metric_ftns, optimizer, config)
		self.data_loader = data_loader
		self.categorical = categorical
		self.continuous = continuous
		self.audio = audio

		if len_epoch is None:
			# epoch-based training
			self.len_epoch = len(self.data_loader)
		else:
			# iteration-based training
			self.data_loader = inf_loop(data_loader)
			self.len_epoch = len_epoch

		self.valid_data_loader = valid_data_loader
		self.do_validation = self.valid_data_loader is not None
		self.lr_scheduler = lr_scheduler
		self.log_step = int(np.sqrt(data_loader.batch_size))

		self.metric_ftns_continuous = metric_ftns_continuous

		self.criterion_continuous = criterion_continuous
		self.criterion_categorical = criterion

		self.lossembed = lossembed

		# setup_cam(self.model)

		self.categorical_class_metrics = [_class + "_" + m.__name__ for _class in valid_data_loader.dataset.categorical_emotions for m in self.metric_ftns]

		self.continuous_class_metrics = [_class + "_" + m.__name__ for _class in valid_data_loader.dataset.continuous_emotions for m in self.metric_ftns_continuous]

		self.train_metrics = MetricTracker('mre', 'loss', 'loss_categorical', 'loss_continuous', 'loss_embed',
			'map', 'mse', 'r2', 'roc_auc_micro', 'roc_auc_macro', 'f1', writer=self.writer)
		self.valid_metrics = MetricTracker('mre', 'loss', 'loss_categorical', 'loss_continuous', 'loss_embed',
			'map', 'mse', 'r2', 'roc_auc_micro', 'roc_auc_macro', 'f1', writer=self.writer)

		self.embed = embed

	def _train_epoch(self, epoch, phase="train"):
		"""
		Training logic for an epoch

		:param epoch: Integer, current training epoch.
		:return: A log that contains average loss and metric in this epoch.
		"""
		import torch.nn.functional as F
		import model.loss
		print("Finding LR")
		for param_group in self.optimizer.param_groups:
			print(param_group['lr'])

		# self.criterion_categorical = model.loss.lsep_for_categ

		if phase == "train":
			self.model.train()
			self.train_metrics.reset()
			torch.set_grad_enabled(True)
			metrics = self.train_metrics
		elif phase == "val" or phase == 'test':
			self.model.eval()
			self.valid_metrics.reset()
			torch.set_grad_enabled(False)
			metrics = self.valid_metrics

		outputs = []
		outputs_continuous = []
		targets = []
		targets_continuous = []

		data_loader = self.data_loader if phase == "train" else self.valid_data_loader
		# print(self.data_loader)

		paths = []
		for batch_idx, (data, target, target_continuous) in enumerate(data_loader):
			data, target, target_continuous = data.to(self.device), target.to(self.device), target_continuous.to(self.device)

			if phase == "train":
				self.optimizer.zero_grad()

			out = self.model(data)

			loss = 0

			if self.categorical:
				loss_categorical = self.criterion_categorical(out['categorical'], target)
				loss += loss_categorical
			if self.continuous:
				loss_continuous = self.criterion_continuous(torch.sigmoid(out['continuous']), target_continuous)
				loss += loss_continuous

			if phase == "train":
				loss.backward()
				self.optimizer.step()

			if self.categorical:
				output = out['categorical'].cpu().detach().numpy()
				target = target.cpu().detach().numpy()
				outputs.append(output)
				targets.append(target)
				paths.append(lengths)

			if self.continuous:
				output_continuous = torch.sigmoid(out['continuous']).cpu().detach().numpy()
				target_continuous = target_continuous.cpu().detach().numpy()
				outputs_continuous.append(output_continuous)
				targets_continuous.append(target_continuous)

			if batch_idx % self.log_step == 0:
				if not self.continuous:
					loss_continuous = torch.tensor([np.nan]).float()

				if not self.categorical:
					loss_categorical = torch.tensor([np.nan]).float()

				self.logger.debug('{} Epoch: {} {} Loss: {:.6f} Loss categorical: {:.6f} Loss continuous: {:.6f}'.format(
					phase,
					epoch,
					self._progress(batch_idx),
					loss.item(),loss_categorical.item(), loss_continuous.item()))

			if batch_idx == self.len_epoch:
				break

		if phase == "train":
			self.writer.set_step(epoch)
		else:
			self.writer.set_step(epoch, phase)

		metrics.update('loss', loss.item())


		if self.categorical:
			metrics.update('loss_categorical', loss_categorical.item())
			output = np.concatenate(outputs, axis=0)
			target = np.concatenate(targets, axis=0)
			target[target>=0.5] = 1 # threshold to get binary labels
			target[target<0.5] = 0

			ap = model.metric.average_precision(output, target)
			roc_auc = model.metric.roc_auc(output, target)

			metrics.update("map", np.mean(ap))
			metrics.update("roc_auc_micro", model.metric.roc_auc(output, target, average='micro'))
			metrics.update("roc_auc_macro", model.metric.roc_auc(output, target, average='macro'))

			self.writer.add_figure('%s ap per class' % phase, make_barplot(ap, self.valid_data_loader.dataset.categorical_emotions, 'average_precision'))
			self.writer.add_figure('%s roc auc per class' % phase, make_barplot(roc_auc, self.valid_data_loader.dataset.categorical_emotions, 'roc auc'))

		if self.continuous:
			metrics.update('loss_continuous', loss_continuous.item())
			output_continuous = np.vstack(outputs_continuous)
			target_continuous = np.vstack(targets_continuous)

			mse = model.metric.mean_squared_error(output_continuous, target_continuous)
			r2 = model.metric.r2(output_continuous, target_continuous)

			metrics.update("r2", np.mean(r2))
			metrics.update("mse", np.mean(mse))

			self.writer.add_figure('%s r2 per class' % phase, make_barplot(r2, self.valid_data_loader.dataset.continuous_emotions, 'r2'))
			self.writer.add_figure('%s mse auc per class' % phase, make_barplot(mse, self.valid_data_loader.dataset.continuous_emotions, 'mse'))


		if self.categorical and self.continuous:
			metrics.update("mre", model.metric.ERS(np.mean(r2), np.mean(ap), np.mean(roc_auc)))

		log = metrics.result()


		if phase == "train":
			if self.lr_scheduler is not None:
				self.lr_scheduler.step()

			if self.categorical:
				self.writer.save_results(output, "output_train")
				self.writer.save_results(target, "target_train")
				paths = np.concatenate(paths, axis=0)
				self.writer.save_results(paths, "paths_train")


			if self.do_validation:
				val_log = self._train_epoch(epoch, phase="val")
				log.update(**{'val_' + k: v for k, v in val_log.items()})

			return log

		elif phase == "val" or phase == 'test':
			if self.categorical:
				self.writer.save_results(output, "output")
			if self.continuous:
				self.writer.save_results(output_continuous, "output_continuous")

			return metrics.result()


	def _progress(self, batch_idx):
		base = '[{}/{} ({:.0f}%)]'
		if hasattr(self.data_loader, 'n_samples'):
			current = batch_idx * self.data_loader.batch_size
			total = self.data_loader.n_samples
		else:
			current = batch_idx
			total = self.len_epoch
		return base.format(current, total, 100.0 * current / total)
