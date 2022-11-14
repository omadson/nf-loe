from typing import List
import torch

class Nelson:
    def __init__(self, model, data) -> None:
        self.model = model
        self.data = data

    def predict(self) -> List:
        ...
        self.model.eval()
        zs, prior_logprob, log_det = self.model(self.data)
        # zs[0] é o xs[-1] ?
        xs, log_det = self.model.backward(zs[-1])
        self.x_avg = xs.mean(axis=1)
        self.x_std = xs.std(axis=1)

        r1 = self._rule1(xs[-1])
        

    def _rule1(self, x):
        """
        One point is more than 3 standard deviations from the mean.
        """
        z_score = (x - self.x_avg) / self.x_std
        return torch.abs(z_score) > 3


    def _rule2(self):
        """
        Nine (or more) points in a row are on the same side of the mean.
        """
        ...


    def _rule3(self):
        """
        Six (or more) points in a row are continually increasing (or decreasing).
        """
        ...


    def _rule4(self):
        """
        Fourteen (or more) points in a row alternate in direction, increasing then decreasing.
        """
        ...
        

    def _rule5(self):
        """
        Two (or three) out of three points in a row are more than 2 standard deviations from the mean in the same direction.
        """
        ...

    def _rule6(self):
        """
        Four (or five) out of five points in a row are more than 1 standard deviation from the mean in the same direction.
        """
        ...

    def _rule7(self):
        """
        Fifteen points in a row are all within 1 standard deviation of the mean on either side of the mean.
        """
        ...

    def _rule8(self):
        """
        Eight points in a row exist, but none within 1 standard deviation of the mean, and the points are in both directions from the mean.
        """
        ...