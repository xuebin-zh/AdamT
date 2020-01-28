import torch
from torch.optim.optimizer import Optimizer


class AMSGrad_Trend(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gammas=(0.9, 0.999),
                 phis=(1.0, 1.0), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= gammas[0] < 1.0:
            raise ValueError("Invalid gamma parameter at index 0: {}".format(gammas[0]))
        if not 0.0 <= gammas[1] < 1.0:
            raise ValueError("Invalid gamma parameter at index 1: {}".format(gammas[1]))
        if not 0.0 <= phis[0] <= 1.0:
            raise ValueError("Invalid phi parameter at index 0: {}".format(phis[0]))
        if not 0.0 <= phis[1] <= 1.0:
            raise ValueError("Invalid phi parameter at index 1: {}".format(phis[1]))
        defaults = dict(lr=lr, betas=betas, gammas=gammas, phis=phis, eps=eps, weight_decay=weight_decay)
        super(AMSGrad_Trend, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AMSGrad_Trend, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AMSGrad_Trend does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    # First-order
                    state["m"] = torch.zeros_like(p.data)
                    # First-order level information
                    state["L_m"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    # Second-order
                    state["v"] = torch.zeros_like(p.data)
                    # Second-order level information
                    state["L_v"] = torch.zeros_like(p.data)
                    # Holt's linear trend information for first-order
                    state["B_m"] = torch.zeros_like(p.data)
                    # Holt's linear trend information for second-order
                    state["B_v"] = torch.zeros_like(p.data)
                    # 2nd order level information used in max step
                    state["L_vmax"] = torch.zeros_like(p.data)
                    # 2nd order trend information used in max step
                    state["B_vmax"] = torch.zeros_like(p.data)
                    # 2nd order used in max step
                    state["v_max"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                L_m, L_v, B_m, B_v = state["L_m"], state["L_v"], state["B_m"], state["B_v"]
                L_vmax, B_vmax = state["L_vmax"], state["B_vmax"]
                v_max = state["v_max"]
                beta1, beta2 = group["betas"]
                gamma1, gamma2 = group["gammas"]
                phi1, phi2 = group["phis"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)

                # Partially update first-order trend information
                B_m.mul_(gamma1 * phi1).add_((gamma1 - 1), L_m)
                # Partially update second-order trend information
                B_v.mul_(gamma2 * phi2).add_((gamma2 - 1), L_v)
                # Fully update first-order level information
                torch.add(beta1 * m, (1 - beta1) * grad, out=L_m)
                # Fully update second-order level information
                torch.addcmul(beta2 * v, (1 - beta2), grad, grad, out=L_v)
                # Fully update first-order trend information
                B_m.add_((1 - gamma1), L_m)
                # Fully update second-order trend information
                B_v.add_((1 - gamma2), L_v)
                # Update first-order and second-order
                torch.add(L_m, phi1 * B_m, out=m)
                torch.add(L_v, phi2 * B_v, out=v)
                # Max step
                L_vmax[v > v_max] = L_v[v > v_max]
                B_vmax[v > v_max] = B_v[v > v_max]
                torch.add(L_vmax, phi2 * B_vmax, out=v_max)
                # Bias correction
                m_hat = torch.div(L_m, (1 - beta1 ** state["step"])) + \
                        torch.div(B_m, ((1 - gamma1) * torch.div(1 - (gamma1 * phi1) ** state["step"], 1 - gamma1 * phi1)))

                v_hat = torch.div(L_vmax, (1 - beta2 ** state["step"])) + \
                        torch.div(B_vmax, ((1 - gamma2) * torch.div(1 - (gamma2 * phi2) ** state["step"], 1 - gamma2 * phi2)))
                # Update the parameters
                p.data.add_(-group["lr"], torch.div(m_hat, (torch.sqrt(v_hat.abs()) + group["eps"])))

        return loss




