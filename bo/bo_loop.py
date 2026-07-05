import torch
from torch import Tensor
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import qUpperConfidenceBound, qLogExpectedImprovement
from utils.utils import concat_data, unstandardize, standardize
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from unittest.mock import patch
import scipy.optimize
import cyipopt
N_TOL = -1e-6


class BaseBayesianOptimization:
    def __init__(self, obj, bounds, threshold = None, unstd_threshold= None, num_acq_samps = [1], constraints=True, main_task=0):   
        self.obj = obj
        self.bounds = bounds
        self.threshold = threshold
        self.unstd_threshold = unstd_threshold
        self.num_acq_samps = num_acq_samps
        self.run = 0
        self.best_y = []
        self.best_x = []
        self.dim = bounds.size(-1)
        self.gp = None
        self.constraints = constraints
        self.main_task = main_task  
        if self.constraints and (threshold is None or unstd_threshold is None):
            raise ValueError("Threshold and unstandardized threshold must be provided for constrained optimization.")

    def step(self):
        raise NotImplementedError("Subclasses must implement the `step` method.")
    

    def update_gp(self, gp, sqrtbeta = 2.):
        with torch.no_grad():
            if self.dim < gp.train_inputs[0].size(-1):
                self.train_inputs = gp.train_inputs[0][..., :-1]
                self.train_tasks = gp.train_inputs[0][..., -1:].to(dtype=torch.int32)
            else:
                self.train_inputs = gp.train_inputs[0]
            self.train_targets = gp.train_targets.unsqueeze(-1)
            self.unstd_train_targets = unstandardize(
                self.train_targets, self.unstd_threshold
            )
            self.sqrtbeta = sqrtbeta.detach()
        if self.gp is None:
            self.observed_max = self._get_max_observed()
            self.best_y.append(self.observed_max[0])
            self.best_x.append(self._get_best_input()[0])
        self.gp = gp

    def _line_search(self, initial_condition, step_size=0.1):
        k = 1000
        direction = torch.randn(initial_condition.size())
        direction /= (
            torch.linalg.norm(direction, dim=-1, ord=2)
            .unsqueeze(-1)
            .repeat(1, 1, self.dim)
        )
        steps = torch.linspace(0, step_size, k).view(1, k, 1) - step_size / 2
        line_search = initial_condition + steps * direction
        inds = (
            (self.inequality_consts(line_search) >= 1e-5).view(
                initial_condition.size(0), -1
            )
            & torch.all(line_search <= self.bounds[1, :].view(1, 1, self.dim), dim=-1)
            & torch.all(line_search >= self.bounds[0, :].view(1, 1, self.dim), dim=-1)
        )
        for id in range(inds.size(0)):
            possible_steps = steps[:, inds[id, :].squeeze(), :].squeeze()
            if possible_steps.numel() <= 1:
                return initial_condition
            max_step_ind = possible_steps.abs().argmax()
            initial_condition[id] = (
                initial_condition[id] + possible_steps[max_step_ind] * direction[id]
            )
        return initial_condition

    def inequality_consts(self, input: Tensor):
        raise NotImplementedError("Subclasses must implement `inequality_consts`.")
    
    def _get_initial_cond(self):
        raise NotImplementedError("Subclasses must implement `_get_initial_cond`.")

    def _get_max_observed(self):
        raise NotImplementedError("Subclasses must implement `_get_max_observed`.")

    def _get_best_input(self):
        raise NotImplementedError("Subclasses must implement `_get_best_input`.")

    def get_next_point(self, task, posterior_transform):
        raise NotImplementedError("Subclasses must implement `get_next_point`.")


class MultiTaskBayesianOptimization(BaseBayesianOptimization):
    def __init__(self, obj, tasks, bounds, threshold = None, unstd_threshold= None, num_acq_samps = [1], constraints=True):
        super().__init__(obj, bounds, threshold, unstd_threshold, num_acq_samps, constraints)
        self.tasks = tasks
        if len(self.num_acq_samps) != len(self.tasks):
            raise ValueError("Number of tasks and number of samples must match")

    def step(self):
        self.run += 1
        print("Run : ", self.run)
        print(f"Best value: {self.observed_max[0]: .3f}")
        print(f"Worst value: {self._get_min_observed()[0]: .3f}")
        W = torch.eye(len(self.tasks))
        for i in self.tasks:
            posterior_transform = ScalarizedPosteriorTransform(W[:, i].squeeze())
            new_point = self.get_next_point(i, posterior_transform)
            if i == self.main_task:
                print(f"New Point: {new_point}")
                new_point_task0 = new_point
                new_result = self.obj.f(new_point, self.main_task)
                print(f"New Observation: {new_result}")
            if i != self.main_task:
                new_point = torch.vstack((new_point, new_point_task0))
                new_result = self.obj.f(new_point, i)
            self.train_inputs, self.train_tasks, self.unstd_train_targets = concat_data(
                (new_point, i * torch.ones(new_point.shape[0], 1), new_result),
                (self.train_inputs, self.train_tasks, self.unstd_train_targets),
            )
        self.train_targets = standardize(
            self.unstd_train_targets, T=self.unstd_threshold
        )
        self.observed_max = self._get_max_observed()
        self.best_y.append(self.observed_max[0])
        self.best_x.append(self._get_best_input()[0])
        return self.train_inputs, self.train_tasks, self.train_targets
    
    def inequality_consts(self, input: Tensor):
        self.gp.eval()
        inputx = input.view(int(input.numel() / self.dim), self.dim)
        output = self.gp(torch.hstack((inputx, self.main_task*torch.ones(inputx.size(0), 1))))
        val = (
            output.mean
            - output.covariance_matrix.diag().sqrt() * self.sqrtbeta
            - self.threshold
        )
        return val.view(inputx.shape[0], 1)

    def _get_max_observed(self):
        return [
            torch.max(self.unstd_train_targets[self.train_tasks == i]).item()
            for i in self.tasks
        ]

    def _get_min_observed(self):
        return [
            torch.min(self.unstd_train_targets[self.train_tasks == i]).item()
            for i in self.tasks
        ]

    def _get_best_input(self):
        return [
            self.train_inputs[self.train_tasks.squeeze() == i, ...][
                torch.argmax(self.train_targets[self.train_tasks == i])
            ]
            for i in self.tasks
        ]
    
    def _get_initial_cond(self):
        train_x0 = self.train_inputs[self.train_tasks.squeeze() == 0]
        train_x = self.train_inputs[self.train_tasks.squeeze() != 0]
        probabilities_task0 = torch.softmax(self.train_targets[self.train_tasks.squeeze() == 0].view(-1), dim=0)
        probabilities_task_other = torch.softmax(self.train_targets[self.train_tasks.squeeze() != 0].view(-1), dim=0)
        sampled_indices0 = torch.multinomial(probabilities_task0, num_samples=min(5, probabilities_task0.numel()), replacement=False)
        sampled_indices = torch.multinomial(probabilities_task_other, num_samples=min(10, probabilities_task_other.numel()), replacement=False)
        sampled_train_inp = torch.vstack((train_x0[sampled_indices0], train_x[sampled_indices]))
        eqfull = self.inequality_consts(sampled_train_inp).squeeze()
        pot_cond = sampled_train_inp[eqfull >= 0, ...]
        unique_cond = []
        for i, inp in enumerate(pot_cond):
            if all(torch.linalg.norm(inp - uinp) >= 0.05 for uinp in unique_cond):
                unique_cond.append(inp)
        pot_cond = torch.stack(unique_cond) if unique_cond else pot_cond[:1]
        return pot_cond.view(pot_cond.size(0), 1, self.dim)


    
    def get_next_point(self, task, posterior_transform):
        with patch.object(scipy.optimize,'minimize', new=custom_ipopt_minimize):
            if task == self.main_task and self.constraints:
                init_cond = self._get_initial_cond()
                if init_cond.numel() == 0:
                    print(
                        "No feasible initial condition found. Randomly sampling a new one."
                    )
                    x_new = self.train_inputs[
                        self.train_targets[self.train_tasks == 0].argmax(), :
                    ].view(1, self.dim)
                    offset = torch.randn(1, self.dim) * 0.005
                    ind = (x_new + offset <= self.bounds[1, :].view(1, self.dim)) & (
                        x_new + offset >= self.bounds[0, :].view(1, self.dim)
                    )
                    x_new[ind] = x_new[ind] + offset[ind]
                    x_new[~ind] = x_new[~ind] - offset[~ind]
                    return x_new
                else:
                    init_cond = self._line_search(init_cond)
                
                acq_base = qUpperConfidenceBound(
                    self.gp,
                    self.sqrtbeta,
                    posterior_transform=posterior_transform,
                )
                candidate, tt = optimize_acqf(
                acq_function=acq_base,
                bounds=(
                    self.bounds
                ),
                q=self.num_acq_samps[task],
                num_restarts=init_cond.size(0) if task == self.main_task else 8,
                raw_samples=512 if task != self.main_task else None,
                nonlinear_inequality_constraints=(
                    [self.inequality_consts] if task == self.main_task else None
                ),
                batch_initial_conditions=init_cond if task == self.main_task else None,
                options={"maxiter": 50},
                )
            else:
                acq = qLogExpectedImprovement(
                    self.gp,
                    best_f=self.observed_max[task],
                    posterior_transform=posterior_transform,
                )
                with patch.object(scipy.optimize,'minimize', new=custom_ipopt_minimize):
                    candidate, tt = optimize_acqf(
                        acq_function=acq,
                        bounds=(
                            self.bounds
                        ),
                        q=self.num_acq_samps[task],
                        num_restarts= 8,
                        raw_samples=512,
                        options={"maxiter": 50},
                    )
        return candidate
    
def custom_ipopt_minimize(*args, **kwargs):
    kwargs.pop('method', None)
    
    options = kwargs.get('options', {}).copy()
    if 'maxiter' in options:
        options['max_iter'] = options.pop('maxiter')
        
    # Optional: Suppress IPOPT's heavy console output
    options.setdefault('print_level', 0)
    options.setdefault('acceptable_tol', 1e-2)
    options.setdefault('constr_viol_tol', 1e-1)
    options.setdefault('hessian_approximation', 'limited-memory')
    kwargs['options'] = options
    
    res = cyipopt.minimize_ipopt(*args, **kwargs)

    if isinstance(res.message, bytes):
        res.message = res.message.decode('utf-8', errors='ignore')
    else:
        res.message = str(res.message)
            
    if not res.success:
        # Check if the failure was specifically due to hitting the iteration limit
        hit_max_iter = (res.status == -1) or ('Maximum_Iterations_Exceeded' in str(res.message))
        local_infeasibility = (res.status == 2) or ('local infeasibility' in str(res.message).lower())
        acceptable_sol = (res.status == 1) 
        
        if hit_max_iter or acceptable_sol:
            res.success = True  # Trick BoTorch into keeping the candidate
            res.status = 0      # Clear the error status code

        elif local_infeasibility:
            res.success = False

    return res

class SingleTaskBayesianOptimization(BaseBayesianOptimization):
    def __init__(self, obj, bounds, threshold, unstd_threshold, num_acq_samps, constraints=True):
        super().__init__(obj, bounds, threshold, unstd_threshold, num_acq_samps, constraints)

    def step(self):
        self.run += 1
        print(f"Run: {self.run}")
        print(f"Best value: {self.observed_max[0]: .3f}")
        new_point = self.get_next_point()
        new_result = self.obj.f(new_point,0)
        print(f"New observation: {new_result}")
        self.train_inputs, self.unstd_train_targets = concat_data(
            (new_point, new_result),
            (self.train_inputs, self.unstd_train_targets),
        )
        self.train_targets = standardize(
            self.unstd_train_targets, T=self.unstd_threshold
        )
        self.observed_max = self._get_max_observed()
        self.best_y.append(self.observed_max[0])
        self.best_x.append(self._get_best_input()[0])
        return self.train_inputs, None, self.train_targets
    
    def inequality_consts(self, input: Tensor):
        self.gp.eval()
        inputx = input.view(int(input.numel() / self.dim), self.dim)
        output = self.gp(inputx)
        val = (
            output.mean
            - output.variance.sqrt() * self.sqrtbeta
            - self.threshold
        )
        return val.view(inputx.shape[0], 1)/self.sqrtbeta

    def _get_max_observed(self):
        return [torch.max(self.unstd_train_targets).item()]

    def _get_best_input(self):
        return [
            self.train_inputs[torch.argmax(self.train_targets)]
        ]
    
    def _get_initial_cond(self):
        train_x = self.train_inputs
        mask = self.inequality_consts(train_x).squeeze() >= 0
        train_y = self.train_targets.squeeze()
        probabilities = torch.softmax(train_y[mask], dim=0)
        sampled_indices = torch.multinomial(probabilities, num_samples=min(10, probabilities.numel()), replacement=False)
        pot_cond = train_x[sampled_indices]
        return pot_cond.view(pot_cond.size(0), 1, self.dim)


    def get_next_point(self):
        acq = qUpperConfidenceBound(
                self.gp,
                beta=self.sqrtbeta
            )
        with patch.object(scipy.optimize,'minimize', new=custom_ipopt_minimize):
            if self.constraints:
                init_cond = self._get_initial_cond()
                init_cond = self._line_search(init_cond)
                mask2 = self.inequality_consts(init_cond)
                init_cond = init_cond[mask2.squeeze() >= 0, ...].view(-1,1,self.dim)
                
                candidate, tt = optimize_acqf(
                    acq_function=acq,
                    bounds=self.bounds,
                    q=1,
                    num_restarts=init_cond.size(0), 
                    nonlinear_inequality_constraints=[self.inequality_consts],
                    batch_initial_conditions=init_cond,
                    options={"maxiter": 50},
                )
            else:
                candidate, tt = optimize_acqf(
                    acq_function=acq,
                    bounds=self.bounds,
                    q=1,
                    num_restarts=8,
                    raw_samples=512,
                    options={"maxiter": 50},
                )
        return candidate
