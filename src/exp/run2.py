import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax

from numba import njit, i8, f8
from datafactory import load_util_artefacts, dump_aset_artefacts
from utils import argnonz, argtopk, myrecall, rand_member_arr_nb, seedutils


"""
    Readings, this module is based on:
    ----------------------------------
    
    [1991] Development and Testing of a Model of Consideration Set Composition
        John H. Roberts and James M. Lattin

    [1995] Studying Consideration Effects in Empirical Choice Models Using Scanner Panel Data
        Rick L. Andrews and T. C. Srinivasan

    [1996] Limited Choice Sets, Local Price Response and Implied Measures of Price Competition
        Bart J. Bronnenberg and Wilfried R. Vanhonacker

    [2009] Blockbuster Culture's Next Rise or Fall: The Impact of Recommender Systems on Sales Diversity
        Daniel M. Fleder and Kartik Hosanagar
"""


beta_max = gamma_max = 8.0


@jax.jit
def logp_fn(params, u, u_thr, P_aset):
    """
    Probability that the user will add item j to his consideration set:
        pi[j] = P(j=C | j=A) * P(j=A)

    Probability that the user will eventually select item j:
        p[j] = pi[j] * exp(beta u[j]) / sum(pi[k] * exp(beta u[k]))
    """

    _d = None

    # normalize
    beta, gamma = jnp.clip(params["beta"], 0, beta_max)[:,_d], jnp.clip(params["gamma"], 0, gamma_max)[:,_d]

    # probability to add item to consideration set
    pi = jax.nn.sigmoid(gamma*(u - u_thr[:,_d])) * P_aset

    # log probability of choice
    logp = jax.nn.log_softmax(jnp.maximum(-gamma_max, jnp.log(pi)) + beta * u, axis=1)

    return logp


@jax.jit
def llik_fn(params, y, u, u_thr, P_aset):
    a = jnp.sum(y, axis=1, keepdims=True)
    L = jnp.sum(logp_fn(params, u, u_thr, P_aset) - jnp.log(1/a), where=(y == 1), axis=1).mean()  # --> max
    return L


def learn_PIAL(y, u, u_thr, P_aset, te_y):

    assert np.all(P_aset >= 0) and np.all(P_aset <= 1), "Incorrectly specified probability mass (0 <= p <= 1)"

    # -------------------------------------------------------------------------

    n_epochs = 10000
    n_epochs_print = 1000

    params = {
        "beta": jnp.ones(len(y), dtype="f4") * beta_max, "gamma": jnp.ones(len(y), dtype="f4") * gamma_max,
    }

    # -------------------------------------------------------------------------

    # step decay lr sheduler: lr = warmup + 1e-2 * (0.8)**(count//2500)
    # maps count to lr
    lr = optax.warmup_exponential_decay_schedule(
        init_value=1e-3,
        peak_value=1e-2, warmup_steps=1000, transition_steps=1000, decay_rate=0.8, staircase=True, end_value=1e-3,
    )
    # optimizer
    chain = optax.chain(
        optax.clip(1.),
        optax.adam(lr),
    )
    optim = optax.multi_transform(
        {
            "T": chain,
            "F": optax.set_to_zero()
        },
        param_labels={"beta": "T", "gamma": "T"},
    )
    state = optim.init(params)

    # -------------------------------------------------------------------------

    llik = -np.inf
    for epoch in range(n_epochs):
        llik, grad = jax.value_and_grad(llik_fn)(params, y, u, u_thr, P_aset)
        params_update, state = optim.update(grad, state)
        params = jax.tree.map(lambda p, u: (p-u), params, params_update)  # updated learnable parameters after SGD step

        if ((epoch+1) % n_epochs_print == 0):
            te_llik = llik_fn(params, te_y, u, u_thr, P_aset)
            print(f"#n={epoch+1:>05d} lr={lr(epoch):.4f} | tr_llik={llik:>8.4f} te_llik={te_llik:>8.4f}")

    # -------------------------------------------------------------------------

    learned_params = {
        "beta"  : np.array(params["beta" ], dtype="f8"),
        "gamma" : np.array(params["gamma"], dtype="f8"),
        "u_thr" : u_thr}

    tr_llik = llik
    te_llik = llik_fn(params, te_y, u, u_thr, P_aset)

    return learned_params, tr_llik, te_llik


def learn(y, u, pop_lamb, delta, topk, te_y):

    assert len(y) == len(te_y), "Each user has to be presented both in train and test splits"

    N = len(y)
    P_pop = y.sum(0) / y.sum(0).max()
    P_knn = get_knn_modulated_awareness_distribution(y, u, delta, topk)
    # estimate probability to get known about item j
    P_aset = pop_lamb * P_pop + (1-pop_lamb) * P_knn

    # estimate two-stage choice model parameters
    learned_params, tr_llik, te_llik = learn_PIAL(y, u, np.min(u, where=(y == 1), axis=1, initial=np.inf), P_aset, te_y)

    # estimate consideration set
    A = np.zeros(y.shape, dtype="i8")
    for n in range(N):
        # sample from an awareness distribution
        # guarantee that y in A
        A[n,np.random.random(J) < P_aset[n,:]+y[n,:]] = 1

    return learned_params, A, P_aset, tr_llik, te_llik


@njit((i8[::1], i8[:,::1], f8))
def get_neighbours(q, Q, similarity_thr):  # --> find neighbours and neighbours similarities w.r.t. given `q` query
    N = len(Q)
    similarity = np.empty(N, dtype="f8")
    for n in range(N):
        similarity[n] = myrecall(q, Q[n,:])
    neighbours = argnonz(
        similarity >=
        similarity_thr
    )
    return neighbours, similarity[neighbours]


@njit((i8[:,::1], f8[:,::1], f8, i8))
def get_knn_modulated_awareness_distribution(y, u, delta, topk):  # --> awareness distribution

    N = len(y)
    P = np.empty(y.shape, dtype="f8")
    u_argtopk = np.empty((N, topk), dtype="i8")
    for n in range(N):
        u_argtopk[n,:] = argtopk(u[n,:], k=topk)  # unsorted topk! items

    eps = 1e-4
    for n in range(N):
        neighbours, _ = get_neighbours(u_argtopk[n,:], u_argtopk, similarity_thr=delta)
        P[n,:] = np.divide(
            np.sum(y[neighbours], axis=0), len(neighbours) + eps)
    return P


def check_data(y, u, A, eps=1e-8):

    N, J = y.shape
    assert ((N, J) == u.shape), "Shape mismatch"
    assert ((N, J) == A.shape), "Shape mismatch"
    assert set(y.ravel()) == {0,1}, "Bad y"
    assert set(A.ravel()) == {0,1}, "Bad A"
    assert np.all(0-eps <= u) and np.all(u <= 1+eps), "Value outside [0,1] range for utility"

    # iterate over all users and collect their choices
    recall_at_A = np.zeros(N, dtype="f8")
    recall_at_J = np.zeros(N, dtype="f8")
    UI_mat = np.zeros((N, J), dtype="i8")

    for n in range(N):
        # number of choices
        tn = y[n,:].sum()
        # true choices
        cn = argnonz(y[n,:])
        # make choices un|A
        cn_at_A = argtopk(np.where(A[n,:] == 1, u[n,:], -np.inf), k=tn)
        # make choices un|J
        cn_at_J = argtopk(u[n,:], k=tn)
        # update buffers
        recall_at_A[n] = myrecall(cn, cn_at_A)
        recall_at_J[n] = myrecall(cn, cn_at_J)
        UI_mat[n,cn_at_A] = 1

    Q = np.sum(y     , axis=0)  # observed in population choices
    P = np.sum(UI_mat, axis=0)  # predicted choices
    q = Q/Q.sum()
    p = P/P.sum()

    chisq = np.sum(q*(p/q-1)**2)
    kl_QP = np.sum(q*(np.log(q+eps) - np.log(p+eps)))
    kl_PQ = np.sum(p*(np.log(p+eps) - np.log(q+eps)))

    print(f"Estimated over a population of {N} users KL: QP_loss={kl_QP:.6f} PQ_loss={kl_PQ:.6f} chi2_loss={chisq:.6f}")
    print(f"Recall|A={np.mean(recall_at_A):.4f} sd={np.std(recall_at_A):.4f}  ")
    print(f"Recall|J={np.mean(recall_at_J):.4f} sd={np.std(recall_at_J):.4f}  ")

    return UI_mat


if __name__ == "__main__":

    # -------------------------------------------------------------------------

    # python3 run2.py --help
    parser = argparse.ArgumentParser(description="Synthetic awareness set estimation protocol")
    parser.add_argument("--util_artefacts_file", type=str, required=True, help="file with estimated user preferences")
    parser.add_argument("--aset_artefacts_file", type=str, required=True, help="file to save output")
    parser.add_argument("--topk", type=int, required=True, help="top k items for similarity calculation")
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    args = parser.parse_args()

    assert (args.topk >= 1), "Bad input"

    print(f"Detected script configuration for synthetic awareness set estimation protocol:\n"
          f"topk={args.topk} "
          f"seed={args.seed} ")

    # -------------------------------------------------------------------------

    # reproducibility
    np.random.seed(args.seed)
    seedutils(args.seed)
    # load file
    true_data_map, _ = load_util_artefacts(args.util_artefacts_file)
    N, J = true_data_map["y"].shape

    # -------------------------------------------------------------------------

    # init structure for awareness set and related consideration set parameters
    # such as beta, gamma, u_thr
    A_map: dict[str,dict[str,np.ndarray]] = {}

    # model awareness/consideration set
    # estimation is based on a sampling process from a probabilistic model: P(C|A)*P(A)
    # with respect to oracle preferences and multinomial choice behavioural constraints

    for key in true_data_map:

        if key == "y":
            continue
        print(f"\nSet recommendation key={key} as a ground truth of [un]biased preferences")

        y = true_data_map["y"]  # {0,1} observations
        u = true_data_map[key]  # oracle preferences

        # split data
        N = len(y)
        tr_y = np.zeros(y.shape, dtype="i8")
        te_y = np.zeros(y.shape, dtype="i8")
        tr_frac = 0.5
        for n in range(N):
            pos = argnonz(y[n,:])
            tr_pos = rand_member_arr_nb(pos, size=max(1, int(len(pos) * tr_frac)))
            te_pos = np.setdiff1d(pos, tr_pos)
            tr_y[n,tr_pos] = 1
            te_y[n,te_pos] = 1
        assert np.array_equal(y, tr_y+te_y), "Wrong split"

        # params search
        best_score = best_pop_lamb = best_delta = -np.inf

        for pop_lamb in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for delta in [0.1, 0.3, 0.5, 0.7, 0.9]:
                print(f"pop_lamb={pop_lamb:.2f} delta={delta:.2f}")

                _, _, _, tr_llik, te_llik = learn(
                    y=tr_y, u=u, pop_lamb=pop_lamb, delta=delta, topk=args.topk, te_y=te_y)

                if (te_llik > best_score):
                    best_pop_lamb = pop_lamb
                    best_delta = delta
                    best_score = te_llik

                print(f"current "
                      f"tr_llik={tr_llik:8.4f} "
                      f"te_llik={te_llik:8.4f} "
                      f"******** best te_llik={best_score:8.4f} pop_lamb={best_pop_lamb} delta={best_delta}")
        del tr_llik, te_llik, best_score, pop_lamb, delta

        # retraining with the best parameters
        print(f"Awareness set and consideration set estimation INIT.")
        best_learned_params, best_A, best_P_aset, best_llik, _ = learn(
            y=y, u=u, pop_lamb=best_pop_lamb, delta=best_delta, topk=args.topk, te_y=y)
        print(f"Awareness set and consideration set estimation DONE. Retrained llik={best_llik:8.4f}")

        check_data(y=y, u=u, A=best_A)

        A_map[key] = {
            "A"      : best_A,
            "P_aset" : best_P_aset,
            "beta"   : best_learned_params["beta"],
            "gamma"  : best_learned_params["gamma"],
            "u_thr"  : best_learned_params["u_thr"],
        }

        # model substitution set
        # estimation is based on the complement of awareness set with superior utility properties

        def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

        q = [0.05, 0.25, 0.5, 0.75]
        p = sigmoid(best_learned_params["gamma"][:,np.newaxis] * (u - best_learned_params["u_thr"][:,np.newaxis]))
        print(f"quantiles of interest={q}")
        print(f"A: {np.quantile(np.sum(best_A, axis=1), q=q)}")
        print(f"y: {np.quantile(np.sum(y, axis=1), q=q)}")
        S = np.row_stack((
            np.quantile(np.sum(p >= 0.5, where=(best_A == 0), axis=1), q=q),
            np.quantile(np.sum(p >= 0.6, where=(best_A == 0), axis=1), q=q),
            np.quantile(np.sum(p >= 0.7, where=(best_A == 0), axis=1), q=q),
            np.quantile(np.sum(p >= 0.8, where=(best_A == 0), axis=1), q=q),
            np.quantile(np.sum(p >= 0.9, where=(best_A == 0), axis=1), q=q),
        ))
        print(f"S[probability of consideration > [0.5], [0.6], [0.7], [0.8], [0.9]]:\n{S}")

    # -------------------------------------------------------------------------

    # dump estimated awareness set artefacts in file for later usage
    dump_aset_artefacts(args.aset_artefacts_file, y=true_data_map["y"], A_map=A_map)

    # -------------------------------------------------------------------------

    print(f"DONE!")
