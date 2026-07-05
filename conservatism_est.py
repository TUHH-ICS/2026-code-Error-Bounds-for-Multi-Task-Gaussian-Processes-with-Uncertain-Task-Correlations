import argparse
import os
import torch
from concurrent.futures import ThreadPoolExecutor

torch.set_default_dtype(torch.float64)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.kernels import ScaleKernel, RBFKernel

from utils.functions import MTRKHSFunction
from utils.get_robust_gp import get_nu_optimized, beta_bayes
from utils.utils import build_mtgp

# torch.set_num_threads(1)

tsk_cov = 0.95
n_1 = 1
n_2 = 1
corr_min =  0.3# 0.999   
corr_max =  0.6#  0.9999 
rho_values = [0.05, 0.1, 0.2, 0.5]
seeds = list(range(1000))
failure_mode = "sample"  # "mean" or "sample" or "stochastic"
num_trials = 1000
num_grid = 100
corr_batch_size = 1000
num_posterior_samples = 1000
parallel = False
num_workers = None
output_dir = "data"
output_name = "conservatism_rho_sweep"


def corr_interval(rho):
    return corr_min + rho * (corr_max - corr_min), corr_max


def generate_training_data(obj, n_1, n_2, bounds):
    data = []

    # Task 1 and task 2 are encoded as zero-based task indices 0 and 1.
    for task, n_task in enumerate((n_1, n_2)):
        x_task = draw_sobol_samples(bounds=torch.tensor([[0.],[1.]]), n=n_task, q=1).reshape(
            n_task, obj.dim
        )
        t_task = torch.full((n_task, 1), task, dtype=torch.int64)
        y_task = obj.f(x_task, t_task)
        data.append((x_task, t_task, y_task))

    train_inputs = torch.vstack([x for x, _, _ in data])
    train_tasks = torch.vstack([t for _, t, _ in data])
    train_targets = torch.vstack([y for _, _, y in data])
    return train_inputs, train_tasks, train_targets


def corr_matrix(corr):
    return torch.tensor([[1.0, corr], [corr, 1.0]])


def corr_chol(corr):
    return torch.linalg.cholesky(corr_matrix(corr)).contiguous()


def make_gp(x, t, y, corr=None):
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-8))
    likelihood.noise = torch.tensor(1e-8)
    kernel = ScaleKernel(RBFKernel())
    kernel.base_kernel.lengthscale = 0.08
    gp = build_mtgp((x, t), y, varf=.1, likelihood=likelihood, kernel=kernel)
    if corr is not None:
        gp.task_covar_module._set_covar_factor(corr_chol(corr))
    gp.eval()
    return gp


def make_sample_gps(x, t, y, corrs):
    gp = make_gp(x, t, y)
    gp.task_covar_module.add_prior()
    gp.pyro_load_from_samples(
        {
            "task_covar_module.covar_factor_prior": torch.stack(
                [corr_chol(float(corr)) for corr in corrs]
            ).contiguous()
        }
    )
    gp.eval()
    return gp


def boundary_nu(x, t, y, rho):
    corr_l, corr_u = corr_interval(rho)
    gp = make_gp(x, t, y)
    gp.task_covar_module.add_prior()
    gp.pyro_load_from_samples(
        {
            "task_covar_module.covar_factor_prior": torch.stack(
                [
                    corr_chol(corr_u),
                    corr_chol(corr_l),
                    corr_chol(corr_u),
                ]
            ).contiguous()
        }
    )
    nu = get_nu_optimized(gp).detach()
    return nu[0, 1:].max()


def posterior_mean_std(gp, x, task, batch_size=None):
    xt = torch.hstack((x, task * torch.ones(x.size(0), 1)))
    if batch_size is not None:
        xt = xt.expand(batch_size, -1, -1)
    with torch.no_grad():
        posterior = gp.posterior(xt)
    return posterior.mean.squeeze(-1), posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)


def posterior_sample(gp, x, task, batch_size=None, sample_shape=torch.Size()):
    xt = torch.hstack((x, task * torch.ones(x.size(0), 1)))
    if batch_size is not None:
        xt = xt.expand(batch_size, -1, -1)
    with torch.no_grad():
        return gp.posterior(xt).rsample(sample_shape).squeeze(-1)


def max_scaled_distances(values, mu_prime, radius):
    return torch.amax(torch.abs(values - mu_prime) / radius, dim=-1)


def num_failure_checks(failure_mode, num_posterior_samples, num_corrs=None):
    if num_corrs is None:
        num_corrs = num_corrs_for_mode(failure_mode)
    if failure_mode == "mean":
        return num_corrs
    if failure_mode == "sample" or failure_mode == "stochastic":
        return num_corrs * num_posterior_samples
    raise ValueError(f"Unknown failure_mode: {failure_mode}")


def num_corrs_for_mode(failure_mode):
    if failure_mode == "stochastic":
        return 1
    if failure_mode in {"mean", "sample"}:
        return num_trials
    raise ValueError(f"Unknown failure_mode: {failure_mode}")


def count_failures(
    x,
    t,
    y,
    rho,
    nu,
    failure_mode=failure_mode,
    num_posterior_samples=num_posterior_samples,
):
    _, corr_u = corr_interval(rho)
    corrs = torch.rand(num_corrs_for_mode(failure_mode)) * (corr_u - corr_min) + corr_min
    gp_prime = make_gp(x, t, y, corr_u)
    grid = torch.linspace(0, 1, num_grid).view(-1, 1)
    mu_prime, std_prime = posterior_mean_std(gp_prime, grid, task=0)
    tau = 0.15
    with torch.no_grad():
        failures = 0
        sqrtbeta = beta_bayes(tau=tau, delta=rho if failure_mode=="stochastic" else 0.05, method="MC", gp=gp_prime, batch_size=10000, n_mc=20000,n_x=1024).sqrt()
        print(f"nu: {nu:.3f}, sqrtbeta: {sqrtbeta:.3f}")
        T = []
        for corr_batch in corrs.split(corr_batch_size):
            gp = make_sample_gps(x, t, y, corr_batch) if failure_mode != "stochastic" else gp_prime
            mu, _ = posterior_mean_std(gp, grid, task=0, batch_size=corr_batch.numel())
            mean_distances = max_scaled_distances(mu, mu_prime, nu * std_prime)
            if failure_mode == "mean":
                batch_distances = mean_distances.unsqueeze(0)
                failed = mean_distances > 1
            elif failure_mode == "sample" or failure_mode == "stochastic":
                samples = posterior_sample(
                    gp,
                    grid,
                    task=0,
                    batch_size=corr_batch.numel(),
                    sample_shape=torch.Size([num_posterior_samples]),
                )
                r = (sqrtbeta + nu) * std_prime if failure_mode != "stochastic" else sqrtbeta * std_prime
                sample_distances = max_scaled_distances(samples, mu_prime, r)
                batch_distances = torch.vstack((mean_distances.unsqueeze(0), sample_distances))
                failed = sample_distances > 1
            else:
                raise ValueError(f"Unknown failure_mode: {failure_mode}")
            T.append(batch_distances.mean())
            failures += int(failed.sum().item())
    return failures, torch.hstack(T).mean()


def estimate_conservatism(
    seed,
    rho,
    failure_mode=failure_mode,
    num_posterior_samples=num_posterior_samples,
):
    torch.manual_seed(seed)
    obj = MTRKHSFunction(B=15,n=15)
    obj.index_kernel = corr_matrix(tsk_cov)
    x, t, y = generate_training_data(obj, n_1, n_2, torch.tensor([[0.], [1.]]))
    nu = boundary_nu(x, t, y, rho)
    failures, T = count_failures(
        x,
        t,
        y,
        rho,
        nu,
        failure_mode=failure_mode,
        num_posterior_samples=num_posterior_samples,
    )
    num_checks = num_failure_checks(
        failure_mode, num_posterior_samples, num_corrs=num_corrs_for_mode(failure_mode)
    )
    return {
        "seed": seed,
        "rho": rho,
        "failures": failures,
        "T": T,
        "T_mean_row": 0,
        "rate": failures / num_checks,
        "nu": nu,
        "failure_mode": failure_mode,
        "num_posterior_samples": (
            num_posterior_samples if failure_mode in {"sample", "stochastic"} else 1
        ),
        "num_failure_checks": num_checks,
        "num_corrs": num_corrs_for_mode(failure_mode),
    }


def run_seed(
    seed,
    failure_mode=failure_mode,
    num_posterior_samples=num_posterior_samples,
):
    return [
        estimate_conservatism(
            seed,
            rho,
            failure_mode=failure_mode,
            num_posterior_samples=num_posterior_samples,
        )
        for rho in rho_values
    ]


def run_experiments(
    parallel=parallel,
    failure_mode=failure_mode,
    num_posterior_samples=num_posterior_samples,
):
    results = []
    if parallel:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            seed_results = pool.map(
                lambda seed: run_seed(seed, failure_mode, num_posterior_samples), seeds
            )
            for seed_result in seed_results:
                for result in seed_result:
                    results.append(result)
                    print_result(result)
    else:
        for seed in seeds:
            for result in run_seed(
                seed,
                failure_mode=failure_mode,
                num_posterior_samples=num_posterior_samples,
            ):
                results.append(result)
                print_result(result)
    return results


def print_summary(results):
    print("\nrho    pooled rate  95% CI          seed median +- 0.95     target")
    for rho in rho_values:
        rho_results = [r for r in results if r["rho"] == rho]
        if not rho_results:
            continue
        total_failures = sum(r["failures"] for r in rho_results)
        distances = torch.stack([r["T"] for r in rho_results])
        dist_quants = torch.quantile(distances, torch.tensor([0.05,0.5, 0.95]))
        total_trials = sum(r.get("num_failure_checks", num_trials) for r in rho_results)
        pooled = total_failures / total_trials
        se = (pooled * (1 - pooled) / total_trials) ** 0.5
        lo, hi = max(0.0, pooled - 1.96 * se), min(1.0, pooled + 1.96 * se)
        print(
            f"{rho:<5.2f}  {pooled:>10.3f}  [{lo:.3f}, {hi:.3f}]  "
            f"{dist_quants[0]:.3f} {dist_quants[1]:.3f} {dist_quants[2]:.3f}    {rho:.2f}"
        )


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--num-seeds", type=int, default=len(seeds))
    parser.add_argument("--out-dir", default=output_dir)
    parser.add_argument("--out-name", default=output_name)
    parser.add_argument("--failure-mode", choices=("mean", "sample", "stochastic"), default=failure_mode)
    parser.add_argument(
        "--num-posterior-samples",
        type=positive_int,
        default=num_posterior_samples,
    )
    return parser.parse_args()


def print_result(result):
    sample_info = (
        f" samples={result.get('num_posterior_samples', 1)}"
        if result.get("failure_mode", "mean") in {"sample", "stochastic"}
        else ""
    )
    print(
        f"seed={result['seed']:02d} rho={result['rho']:.2f} "
        f"rate={result['rate']:.3f} nu={result['nu']:.3f} "
        f"mode={result.get('failure_mode', 'mean')}{sample_info} "
        f"Dist={result['T']:.3f}"
    )


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(results, path)


def seed_result_path(out_dir, out_name, seed):
    return os.path.join(out_dir, f"{out_name}_seed_{seed:04d}.pt")


def aggregate_results(out_dir, out_name, num_seeds):
    results = []
    missing = []
    for seed in range(num_seeds):
        path = seed_result_path(out_dir, out_name, seed)
        if not os.path.exists(path):
            missing.append(seed)
            continue
        results.extend(torch.load(path, map_location="cpu", weights_only=False))

    if missing:
        raise FileNotFoundError(f"Missing result files for seeds: {missing[:10]}")

    by_rho = {}
    for rho in rho_values:
        rho_results = [result for result in results if result["rho"] == rho]
        if not rho_results:
            continue
        failures = torch.tensor([result["failures"] for result in rho_results])
        T = torch.stack([result["T"].detach().cpu() for result in rho_results])
        rates = torch.tensor([result["rate"] for result in rho_results])
        failure_checks = torch.tensor(
            [result.get("num_failure_checks", num_trials) for result in rho_results]
        )
        nus = torch.stack(
            [
                result["nu"].detach().cpu()
                if torch.is_tensor(result["nu"])
                else torch.tensor(result["nu"])
                for result in rho_results
            ]
        )
        by_rho[rho] = {
            "rho": rho,
            "seeds": torch.tensor([result["seed"] for result in rho_results]),
            "failures": failures,
            "failure_checks": failure_checks,
            "T": T,
            "T_mean_row": 0,
            "dist": T,
            "rates": rates,
            "nus": nus,
            "pooled_rate": failures.sum().item() / failure_checks.sum().item(),
            "mean_rate": rates.mean(),
            "std_rate": rates.std(unbiased=False),
        }

    return {
        "results": results,
        "by_rho": by_rho,
        "num_trials": num_trials,
        "num_posterior_samples": num_posterior_samples,
    }


def main():
    args = parse_args()
    slurm_seed = os.environ.get("SLURM_ARRAY_TASK_ID")
    seed = args.seed if args.seed is not None else int(slurm_seed) if slurm_seed else None

    if args.aggregate:
        aggregated = aggregate_results(args.out_dir, args.out_name, args.num_seeds)
        print_summary(aggregated["results"])
        path = os.path.join("data/summary/", f"{failure_mode}4_all.pt")
        # path = os.path.join("data/summary/Extra_all.pt")
        save_results(aggregated, path)
        print(f"Saved aggregated results to {path}")
        return

    if seed is not None:
        results = run_seed(
            seed,
            failure_mode=args.failure_mode,
            num_posterior_samples=args.num_posterior_samples,
        )
        for result in results:
            print_result(result)
        path = seed_result_path(args.out_dir, args.out_name, seed)
        save_results(results, path)
        return

    results = run_experiments(
        failure_mode=args.failure_mode,
        num_posterior_samples=args.num_posterior_samples,
    )
    print_summary(results)
    save_results(results, os.path.join(args.out_dir, f"{args.out_name}.pt"))


# def plot_task1_bound(obj, x, t, y, true_corr, rho=0.2):
#     def np(tensor):
#         return tensor.detach().numpy()

#     _, corr_u = corr_interval(rho)
#     nu = boundary_nu(x, t, y, rho)
#     gp = make_gp(x, t, y, corr_min)
#     gp_prime = make_gp(x, t, y, corr_u)
#     grid = torch.linspace(0, 1, num_grid).view(-1, 1)

#     f_grid = obj.f(grid, 0).squeeze().detach()
#     mu, _ = posterior_mean_std(gp, grid, 0)
#     mu_prime, std_prime = posterior_mean_std(gp_prime, grid, 0)
#     radius = nu * std_prime
#     task1 = t.squeeze() == 0

#     grid = grid.squeeze().detach()
#     plt.figure()
#     plt.plot(np(grid), np(f_grid), "k", label="f task 1")
#     plt.plot(np(grid), np(mu), label="mu Sigma")
#     plt.plot(np(grid), np(mu_prime), label="mu Sigma'")
#     plt.fill_between(
#         np(grid),
#         np(mu_prime - radius),
#         np(mu_prime + radius),
#         alpha=0.2,
#         label="mu Sigma' +/- nu sigma Sigma'",
#     )
#     plt.scatter(np(x[task1].squeeze()), np(y[task1].squeeze()), c="k", s=25, marker="x")
#     plt.scatter(np(x[~task1].squeeze()), np(y[~task1].squeeze()), c="k", s=25, marker="o")
#     plt.xlabel("x")
#     plt.ylabel("task 1")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"plots/conservatism_task1_bound_rho_{int(100 * rho)}.png", dpi=200)
#     plt.show()


# torch.manual_seed(seeds[0])
# f = MTRKHSFunction(B=8)
# f.index_kernel = corr_matrix(tsk_cov)
# train_inputs, train_tasks, train_targets = generate_training_data(f, n_1, n_2, torch.tensor([[0.], [1.]]))

if __name__ == "__main__":
    main()
