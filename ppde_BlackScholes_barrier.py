import torch
import numpy as np
import argparse
import tqdm
import os
import matplotlib.pyplot as plt

from lib.bsde import PPDE_BlackScholes as PPDE
from lib.options import BarrierOption


def sample_x0(batch_size, dim, device, s0):
    return s0 * torch.ones(batch_size, dim, device=device)


def write(msg, logfile, pbar):
    pbar.write(msg)
    with open(logfile, "a") as f:
        f.write(msg)
        f.write("\n")


def train(T,
          n_steps,
          d,
          mu,
          sigma,
          depth,
          rnn_hidden,
          ffn_hidden,
          max_updates,
          batch_size,
          lag,
          base_dir,
          device,
          method,
          s0,
          K,
          B,
          option_type,
          barrier_direction,
          knock,
          seed):

    if n_steps % lag != 0:
        raise ValueError("n_steps must be divisible by lag")

    torch.manual_seed(seed)
    np.random.seed(seed)

    logfile = os.path.join(base_dir, "log.txt")
    ts = torch.linspace(0, T, n_steps + 1, device=device)

    barrier = BarrierOption(
        K=K,
        B=B,
        option_type=option_type,
        barrier_direction=barrier_direction,
        knock=knock,
        asset_idx=0
    )

    ppde = PPDE(d, mu, sigma, depth, rnn_hidden, ffn_hidden).to(device)
    optimizer = torch.optim.RMSprop(ppde.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.2)

    pbar = tqdm.tqdm(total=max_updates)
    losses = []

    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size, d, device, s0)

        if method == "bsde":
            loss, _, _ = ppde.fbsdeint(ts=ts, x0=x0, option=barrier, lag=lag)
        else:
            loss, _, _ = ppde.conditional_expectation(ts=ts, x0=x0, option=barrier, lag=lag)

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().cpu().item())

        if (idx + 1) % 10 == 0:
            with torch.no_grad():
                x0 = sample_x0(5000, d, device, s0)
                loss_eval, Y, payoff = ppde.fbsdeint(ts=ts, x0=x0, option=barrier, lag=lag)
                payoff = torch.exp(-mu * ts[-1]) * payoff.mean()

            pbar.update(10)
            write(
                "loss={:.4f}, Monte Carlo price={:.4f}, predicted={:.4f}".format(
                    loss_eval.item(), payoff.item(), Y[0, 0, 0].item()
                ),
                logfile,
                pbar
            )

    result = {"state": ppde.state_dict(), "loss": losses}
    torch.save(result, os.path.join(base_dir, "result.pth.tar"))

    x0 = sample_x0(1, d, device, s0)
    with torch.no_grad():
        x, _ = ppde.sdeint(ts=ts, x0=x0)

    fig, ax = plt.subplots()
    ax.plot(ts.cpu().numpy(), x[0, :, 0].cpu().numpy())
    ax.set_ylabel(r"$X(t)$")
    fig.savefig(os.path.join(base_dir, "path_eval.pdf"))

    pred, mc_pred = [], []
    for idx, _ in enumerate(ts[::lag]):
        pred.append(ppde.eval(ts=ts, x=x[:, :(idx * lag) + 1, :], lag=lag).detach())
        mc_pred.append(
            ppde.eval_mc(
                ts=ts,
                x=x[:, :(idx * lag) + 1, :],
                lag=lag,
                option=barrier,
                mc_samples=10000
            )
        )

    pred = torch.cat(pred, 0).view(-1).cpu().numpy()
    mc_pred = torch.cat(mc_pred, 0).view(-1).cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(ts[::lag].cpu().numpy(), pred, '--', label="LSTM + BSDE + sign")
    ax.plot(ts[::lag].cpu().numpy(), mc_pred, '-', label="MC")
    ax.set_ylabel(r"$v(t, X_t)$")
    ax.legend()

    option_name = f"{barrier_direction}_and_{knock}_{option_type}"
    fig.savefig(os.path.join(base_dir, f"{option_name}_LSTM_sol.pdf"))
    print("THE END")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--d', default=1, type=int)
    parser.add_argument('--max_updates', default=1000, type=int)
    parser.add_argument('--ffn_hidden', default=[20, 20], nargs="+", type=int)
    parser.add_argument('--rnn_hidden', default=20, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--T', default=1.0, type=float)
    parser.add_argument('--n_steps', default=100, type=int, help="number of time steps")
    parser.add_argument('--lag', default=10, type=int, help="evaluation lag")
    parser.add_argument('--mu', default=0.05, type=float, help="risk-free rate")
    parser.add_argument('--sigma', default=0.2, type=float, help="volatility")

    parser.add_argument('--s0', default=100.0, type=float)
    parser.add_argument('--K', default=100.0, type=float)
    parser.add_argument('--B', default=90.0, type=float)
    parser.add_argument('--option_type', default='call', choices=['call', 'put'])
    parser.add_argument('--barrier_direction', default='down', choices=['down', 'up'])
    parser.add_argument('--knock', default='out', choices=['in', 'out'])
    parser.add_argument('--method', default="bsde", type=str, choices=["bsde", "orthogonal"])

    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_cuda:
        device = f"cuda:{args.device}"
    else:
        device = "cpu"

    option_name = f"{args.barrier_direction}_and_{args.knock}_{args.option_type}"
    results_path = os.path.join(args.base_dir, "BS_barrier", option_name, args.method)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    train(
        T=args.T,
        n_steps=args.n_steps,
        d=args.d,
        mu=args.mu,
        sigma=args.sigma,
        depth=args.depth,
        rnn_hidden=args.rnn_hidden,
        ffn_hidden=args.ffn_hidden,
        max_updates=args.max_updates,
        batch_size=args.batch_size,
        lag=args.lag,
        base_dir=results_path,
        device=device,
        method=args.method,
        s0=args.s0,
        K=args.K,
        B=args.B,
        option_type=args.option_type,
        barrier_direction=args.barrier_direction,
        knock=args.knock,
        seed=args.seed
    )