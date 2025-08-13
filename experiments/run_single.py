from AdOEM_LDP.experiments.runner import RunnerConfig, SingleRunner

cfg = RunnerConfig(
    K=50, T=100*100, eps=1.0,
    mech="RRRR",              
    hashing_enabled=True,     
    online="online_em",         
    offline="offline_em",
    eps1_coeff_vec=[0.8, 0.9, 1.0],
    P_int_EM=100, M_int_EM=50,
    P_quad_prog=0,         
    M_EM_final=100,
    alpha_EM=0.65,
    rho_coeff=0.05,
    seed=42,
)
res = SingleRunner(cfg).run()
print("TV online :", res.tv_online)
print("TV offline:", res.tv_offline)