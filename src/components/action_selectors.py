import torch as th
from torch.distributions import Normal


from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        try:
            random_actions = Categorical(avail_actions.float()).sample().long()
        except Exception as e:
            print(f"[Error] Action selection error: {e}")
            raise e  # 可根据需要决定是否重新抛出异常
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions


REGISTRY["soft_policies"] = SoftPoliciesSelector





class GaussianActionSelector:

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, mu, sigma, test_mode=False):
        # expects the following input dimensionalities:
        # mu: [batch_size, num_agents, num_actions]
        # sigma: [batch_size, num_agents, num_actions]
        assert mu.dim() == 3, "incorrect input dim: mu"
        assert sigma.dim() == 3, "incorrect input dim: sigma"

        if test_mode and self.test_greedy:
            picked_actions = mu
        else:
            # Generate diagonal covariance matrix
            sigma_diag = th.diag_embed(sigma)  # [batch_size, num_agents, num_actions, num_actions]

            # Reshape for MultivariateNormal
            mu_reshaped = mu.view(-1, mu.shape[-1])  # [batch_size * num_agents, num_actions]
            sigma_diag_reshaped = sigma_diag.view(-1, mu.shape[-1], mu.shape[-1])  # [batch_size * num_agents, num_actions, num_actions]

            # Construct the MultivariateNormal distribution
            dst = th.distributions.MultivariateNormal(mu_reshaped, sigma_diag_reshaped)
            try:
                picked_actions = dst.sample().view(*mu.shape)  # Reshape back to [batch_size, num_agents, num_actions]
            except Exception as e:
                print(f"Sampling error: {e}")
                picked_actions = mu  # Default to mean actions if sampling fails
        return picked_actions


REGISTRY["gaussian"] = GaussianActionSelector
