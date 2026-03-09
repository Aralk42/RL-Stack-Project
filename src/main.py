import matplotlib.pyplot as plt

from train import train


def main():

    rewards, agent = train(episodes=2000)

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    plt.show()


if __name__ == "__main__":
    main()