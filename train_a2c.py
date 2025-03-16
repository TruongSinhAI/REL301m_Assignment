import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

from src.agents.a2c import A2CAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_agent(env, agent, num_episodes=500, max_steps=500,
                eval_freq=20, save_dir="models", render_freq=None):
    """Training function for the advanced A2C agent"""

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Track metrics
    all_rewards = []
    all_scores = []
    all_steps = []
    all_losses = {'policy': [], 'value': [], 'entropy': [], 'total': []}
    best_reward = -float('inf')
    best_score = -float('inf')
    recent_rewards = deque(maxlen=100)
    recent_scores = deque(maxlen=100)
    episode_start_time = time.time()

    for episode in range(1, num_episodes + 1):
        # Reset environment
        state_tuple = env.reset()
        if isinstance(state_tuple, tuple):  # Handle new gym API
            state = state_tuple[0]
        else:
            state = state_tuple

        total_reward = 0
        done = False
        step_count = 0

        # Storage for episode data
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        # Run one episode
        while not done and step_count < max_steps:
            # Render if specified
            if render_freq and episode % render_freq == 0:
                env.render()

            # Select action
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Store experience
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)

            # Update state and counters
            state = next_state
            total_reward += reward
            step_count += 1

            # Handle dictionary-style info
            if isinstance(info, dict) and 'score' in info:
                current_score = info['score']
            else:
                current_score = getattr(env, 'score', 0)

        # Bootstrap value for non-terminal states
        if not done and step_count >= max_steps:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(agent.device)
                _, _, _, next_value = agent.network.get_action_and_value(state_tensor)
        else:
            next_value = 0

        # Update with full episode data
        metrics = agent.optimize_model(log_probs, values, rewards, dones,
                                       states=states, actions=actions, next_value=next_value)

        # Track losses
        for key in all_losses.keys():
            if key + '_loss' in metrics:
                all_losses[key].append(metrics[key + '_loss'])
            elif key in metrics:
                all_losses[key].append(metrics[key])

        # Track performance
        all_rewards.append(total_reward)
        all_scores.append(current_score)
        all_steps.append(step_count)
        recent_rewards.append(total_reward)
        recent_scores.append(current_score)

        # Calculate statistics
        avg_reward = np.mean(recent_rewards)
        avg_score = np.mean(recent_scores)

        # Save best models
        if total_reward > best_reward:
            best_reward = total_reward
            model_path = f"{save_dir}/best_reward.pth"
            torch.save(agent.policy_net.state_dict(), model_path)
            print(f"New best reward model saved: {total_reward:.2f}")

        if current_score > best_score:
            best_score = current_score
            model_path = f"{save_dir}/best_score.pth"
            torch.save(agent.policy_net.state_dict(), model_path)
            print(f"New best score model saved: {current_score}")

        # Regular evaluation and checkpointing
        if episode % eval_freq == 0:
            elapsed = time.time() - episode_start_time
            print(f"\nEpisode {episode}/{num_episodes} | "
                  f"Reward: {total_reward:.2f} (avg100: {avg_reward:.2f}) | "
                  f"Score: {current_score} (avg100: {avg_score:.2f}) | "
                  f"Steps: {step_count} | "
                  f"Learning method: {agent.learning_method} | "
                  f"Time: {elapsed:.2f}s")

            if len(all_losses['policy']) > 0:
                print(f"Losses - Policy: {all_losses['policy'][-1]:.4f}, "
                      f"Value: {all_losses['value'][-1]:.4f}, "
                      f"Entropy: {all_losses['entropy'][-1]:.4f}")

            # Save checkpoint
            checkpoint_path = f"{save_dir}/checkpoint.pth"
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'best_reward': best_reward,
                'best_score': best_score
            }, checkpoint_path)

            episode_start_time = time.time()

    # Save final model
    final_model_path = f"{save_dir}/final.pth"
    torch.save(agent.policy_net.state_dict(), final_model_path)
    print(f"Final model saved as {final_model_path}")

    # Plot learning curves
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(all_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(2, 2, 2)
    plt.plot(all_scores)
    plt.title('Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    plt.subplot(2, 2, 3)
    plt.plot(all_losses['policy'], label='Policy Loss')
    plt.plot(all_losses['value'], label='Value Loss')
    plt.title('Training Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(all_losses['entropy'], label='Entropy')
    plt.title('Policy Entropy')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_curves.png")

    print("Training completed.")
    return {
        'rewards': all_rewards,
        'scores': all_scores,
        'steps': all_steps,
        'losses': all_losses,
        'best_reward': best_reward,
        'best_score': best_score
    }


def test_agent(env, agent, num_episodes=10, render=False, model_path=None):
    """Test function for the advanced A2C agent"""

    # Load model if path is provided
    if model_path and os.path.exists(model_path):
        agent.policy_net.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")

    agent.policy_net.eval()

    # Track metrics
    total_rewards = []
    episode_scores = []
    episode_steps = []

    for episode in range(num_episodes):
        # Reset environment
        state_tuple = env.reset()
        if isinstance(state_tuple, tuple):
            state = state_tuple[0]
        else:
            state = state_tuple

        total_reward = 0
        step_count = 0
        done = False

        # Run episode
        while not done:
            if render:
                env.render()

            # Select action
            state_tensor = torch.FloatTensor(state).to(agent.device)
            with torch.no_grad():
                action, _, _, _ = agent.network.get_action_and_value(state_tensor)
                action = action.cpu().item()

            # Take action in environment
            next_state, reward, done, info = env.step(action)

            # Update state and metrics
            state = next_state
            total_reward += reward
            step_count += 1

            # Handle dictionary-style info
            if isinstance(info, dict) and 'score' in info:
                current_score = info['score']
            else:
                current_score = getattr(env, 'score', 0)

        # Record episode results
        total_rewards.append(total_reward)
        episode_scores.append(current_score)
        episode_steps.append(step_count)

        print(f"Test Episode {episode + 1}: Reward = {total_reward:.2f}, "
              f"Score = {current_score}, Steps = {step_count}")

    # Calculate and print summary statistics
    avg_reward = sum(total_rewards) / num_episodes
    avg_score = sum(episode_scores) / num_episodes
    avg_steps = sum(episode_steps) / num_episodes

    print(f"\nTest Results ({num_episodes} episodes):")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Max Reward: {max(total_rewards):.2f}")
    print(f"Max Score: {max(episode_scores)}")

    return {
        'rewards': total_rewards,
        'scores': episode_scores,
        'steps': episode_steps,
        'avg_reward': avg_reward,
        'avg_score': avg_score
    }


if __name__ == "__main__":
    from src.env.env import CarCrossingEnv

    # Create training environment
    env = CarCrossingEnv()
    state_dim = 19
    action_dim = 5

    # Experiment with different learning methods
    learning_methods = {
        "TD": "td",  # 1-step TD learning (standard Bellman equation)
        "N-Step": "nstep",  # n-step TD learning
        "Monte-Carlo": "mc",  # Full Monte Carlo returns
        "TD-Lambda": "tdlambda",  # TD(λ) with eligibility traces
    }

    # Results storage
    results = {}

    # Train with each learning method
    for name, method in learning_methods.items():
        print(f"\n{'=' * 50}")
        print(f"Training A2C with {name} learning method...")
        print(f"{'=' * 50}\n")

        # Create agent with the specific learning method
        agent = A2CAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_method=method,
            lr=3e-4,
            gamma=0.99,
            n_steps=5,  # For n-step TD
            td_lambda=0.95,  # For TD(λ)
            entropy_coef=0.01,
            value_loss_coef=0.5,
            max_grad_norm=0.5
        )

        # Train agent
        training_results = train_agent(
            env=env,
            agent=agent,
            num_episodes=500,
            eval_freq=20,
            save_dir=f"models/A2C_{method}"
        )

        results[name] = {
            'train': training_results,
            'agent': agent
        }

    # Test the best method (determined by highest average reward)
    best_method = max(learning_methods.items(),
                      key=lambda x: np.mean(results[x[0]]['train']['rewards'][-100:]))

    print(f"\n{'=' * 50}")
    print(f"Best learning method: {best_method[0]} with method '{best_method[1]}'")
    print(f"{'=' * 50}\n")

    # Get the best agent
    best_agent = results[best_method[0]]['agent']
    model_path = f"models/A2C_{best_method[1]}/best_score.pth"

    # Test the best agent
    test_results = test_agent(
        env=env,
        agent=best_agent,
        num_episodes=10,
        render=True,
        model_path=model_path
    )

    # Compare all methods in a single plot
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    for name in learning_methods:
        rewards = results[name]['train']['rewards']
        plt.plot(rewards, label=name)
    plt.title('Training Rewards Across Methods')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for name in learning_methods:
        scores = results[name]['train']['scores']
        plt.plot(scores, label=name)
    plt.title('Training Scores Across Methods')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig("models/a2c_method_comparison.png")

    env.close()