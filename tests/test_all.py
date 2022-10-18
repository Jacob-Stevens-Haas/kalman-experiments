import kal_exp


def test_gen_data():
    _, x_true, _, _, _ = kal_exp.gen_data(1, nt=20)
    # Should be a standard normal distribution, passing 99.8 percent of the time
    assert abs(x_true[-2]) < 3
