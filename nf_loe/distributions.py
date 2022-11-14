import torch
from torch import distributions, nn
import torch.nn.functional as F
import numpy as np
import math

class SSLGaussMixture(torch.distributions.Distribution):

    def __init__(self, means, inv_cov_stds=None, device=None):
        self.n_components, self.d = means.shape
        self.means = means

        if inv_cov_stds is None:
            self.inv_cov_stds = math.log(math.exp(1.0) - 1.0) * torch.ones((len(means)), device=device)
        else:
            self.inv_cov_stds = inv_cov_stds

        self.weights = torch.ones((len(means)), device=device)
        self.device = device

    @property
    def gaussians(self):
        gaussians = [distributions.MultivariateNormal(mean, F.softplus(inv_std)**2 * torch.eye(self.d).to(self.device))
                          for mean, inv_std in zip(self.means, self.inv_cov_stds)]
        return gaussians


    def parameters(self):
       return [self.means, self.inv_cov_std, self.weights]
        
    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1), p=F.softmax(self.weights))
            all_samples = [g.sample(sample_shape) for g in self.gaussians]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)[0]
                samples[mask] = all_samples[i][mask]
        return samples
        
    def log_prob(self, x, y=None, label_weight=1.):
        all_log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights)), dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] += mixture_log_probs[mask]
            for i in range(self.n_components):
                #Pavel: add class weights here? 
                mask = (y == i)
                log_probs[mask] += all_log_probs[:, i][mask] * label_weight
            return log_probs
        else:
            return mixture_log_probs

    def class_logits(self, x):
        log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        log_probs_weighted = log_probs + torch.log(F.softmax(self.weights))
        return log_probs_weighted

    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)