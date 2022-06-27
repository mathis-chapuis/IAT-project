from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
from controller import AgentInterface
from epsilon_profile import EpsilonProfile

def test_maze(game: SpaceInvaders, agent: AgentInterface, max_steps: int, nepisodes : int = 1, speed: float = 0., same = True, display: bool = False):
    n_steps = max_steps
    sum_rewards = 0.
    for _ in range(nepisodes):
        state = game.reset_using_existing_maze() if (same) else game.reset()

        for step in range(max_steps):
            action = agent.select_greedy_action(state)
            next_state, reward, terminal = game.step(action)

            sum_rewards += reward
            if terminal:
                n_steps = step+1  # number of steps taken
                break
            state = next_state
    return n_steps, sum_rewards


def main():
    n_episodes = 10
    max_steps = 1000
    gamma = 1.
    alpha = 0.2
    eps_profile = EpsilonProfile(1.0, 0)

    # Init agent and learn
    game = SpaceInvaders(display=False)
    agent = QAgent(game, eps_profile, gamma, alpha)
    agent.learn(game, n_episodes, max_steps)
    print('Agent Q {}'.format(agent.Q))

    #controller = KeyboardController()
    # controller = RandomAgent(game.na)
 
    # state = game.reset()
    # while True:
    #     action = controller.select_action(state)
    #     state, reward, is_done = game.step(action)
    #     sleep(0.0001)

if __name__ == '__main__' :
    main()
