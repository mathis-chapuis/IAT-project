from time import sleep
from sys import argv, exit

from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
from controller import AgentInterface
from epsilon_profile import EpsilonProfile

def test_game(game: SpaceInvaders, agent: AgentInterface, nepisodes : int = 1, display: bool = False):
    sum_rewards = 0.
    game.set_display(display)

    for _ in range(nepisodes):
        terminal = False
        state = game.reset()

        while not terminal:
            action = agent.select_greedy_action(state)
            next_state, reward, terminal = game.step(action)

            sum_rewards += reward
            state = next_state
    return sum_rewards


def main():
    if len(argv) > 7:
        n_episodes = int(argv[1])
        max_steps = int(argv[2])
        final_episodes = int(argv[3])
        gamma = float(argv[4])
        alpha = float(argv[5])
        eps_profile = EpsilonProfile(float(argv[6]), float(argv[7]))
    else:
        print('\n\nUsage: python3 run_game.py <n_epispdes> <n_steps> <final_episodes> <gamma> <alpha> <eps_begin> <eps_end>\n')
        exit(1)
        # n_episodes = 100
        # max_steps = 30000
        # gamma = 1.
        # alpha = 0.1
        # eps_profile = EpsilonProfile(1.0, 0.1)
        # final_episodes = 10

    # Init agent and learn
    game = SpaceInvaders(display=False)
    agent = QAgent(game, eps_profile, gamma, alpha)
    agent.learn(game, n_episodes, max_steps)
    sum_rewards = test_game(game, agent, n_episodes=final_episodes, display=False)

    print('Training score: {}'.format(sum_rewards / final_episodes))

    #controller = KeyboardController()
    controller = RandomAgent(game.na)
 
    state = game.reset()
    random_score = 0
    is_done = False
    for _ in range(final_episodes):
        while not is_done:
            action = controller.select_action(state)
            state, reward, is_done = game.step(action)
            random_score += reward

    print('random_score : {}'.format(random_score / final_episodes))

if __name__ == '__main__' :
    main()
