import torch


class Normalizer:
    def __init__(self):
        self.location_mean = torch.tensor([31.5863, 32.0618])
        self.location_std = torch.tensor([16.1025, 16.1353])

    def normalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return (location - self.location_mean.to(location.device)) / (
            self.location_std.to(location.device) + 1e-6
        )

    def unnormalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return location * self.location_std.to(location.device) + self.location_mean.to(
            location.device
        )

    def unnormalize_mse(self, mse):
        return mse * (self.location_std.to(mse.device) ** 2)
