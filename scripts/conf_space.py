config = {
    "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-5, 0.1),
    "kernw": tune.choice([40, 50, 60, 70, 80, 90]),
    "kernlayers": tune.choice([6, 8, 10, 12]),
    "weight": tune.choice([[1.,1.],[1.,5.],[1.,10.],[1.,15.],[1.,20.]]),
    "batch_size": tune.choice([4, 8, 16]),E
    "dropout": tune.choice([0.4,0.5,0.6])
}
